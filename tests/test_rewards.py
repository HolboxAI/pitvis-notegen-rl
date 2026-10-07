import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notegen_rl import rewards as R  # noqa: E402
from notegen_rl.prompts import PLACEHOLDER  # noqa: E402

ROW = {
    "step_vocab": json.dumps(["Nasal corridor creation", "Sellotomy", "Tumour excision", "Haemostasis", "Durotomy"]),
    "instrument_vocab": json.dumps(["Suction", "Bipolar forceps", "Kerrison rongeurs"]),
    "gold_steps": json.dumps(["Nasal corridor creation", "Tumour excision"]),
    "gold_steps_any": json.dumps(["Nasal corridor creation", "Tumour excision", "Haemostasis"]),
    "gold_instruments": json.dumps(["Suction"]),
    "gold_instruments_any": json.dumps(["Suction", "Bipolar forceps"]),
    "gold_segments": json.dumps([["Nasal corridor creation", 0, 300], ["Tumour excision", 300, 900],
                                 ["Haemostasis", 900, 920]]),
    "reliability": json.dumps({"Nasal corridor creation": 0.6, "Tumour excision": 0.75, "Suction": 0.8}),
    "reliability_floor": 0.2,
}

PERFECT_STEPS = ["- Nasal corridor creation [00:00:00-00:05:00] (conf=1.0)",
                 "- Tumour excision [00:05:00-00:15:00] (conf=1.0)"]
PERFECT_INSTR = ["- Suction (conf=1.0)"]


def note(steps, instr, findings=PLACEHOLDER, complications=PLACEHOLDER, closure=PLACEHOLDER):
    return ("<facts_used>used the perception output</facts_used>\n<note>\n"
            "<procedure>Endoscopic transsphenoidal pituitary surgery</procedure>\n"
            f"<findings>{findings}</findings>\n<steps>\n" + "\n".join(steps) + "\n</steps>\n"
            "<instruments>\n" + "\n".join(instr) + "\n</instruments>\n"
            f"<complications>{complications}</complications>\n<closure>{closure}</closure>\n</note>")


def test_perfect_note_scores_one_everywhere():
    s = R.score_all(note(PERFECT_STEPS, PERFECT_INSTR), ROW)
    for k in ("format", "grounding", "calibration", "temporal", "safety"):
        assert abs(s[k] - 1.0) < 1e-9, (k, s[k])


def test_hallucinated_instrument_costs_precision():
    """Every allowed name in the note counts as a claim, so an extra instrument lowers precision."""
    s = R.score_all(note(PERFECT_STEPS, PERFECT_INSTR + ["- Kerrison rongeurs (conf=0.9)"]), ROW)
    assert s["instr_precision"] == 0.5
    assert s["grounding"] < 1.0


def test_listing_everything_is_worse_than_the_truth():
    everything = note([f"- {n} (conf=0.5)" for n in json.loads(ROW["step_vocab"])],
                      [f"- {n} (conf=0.5)" for n in json.loads(ROW["instrument_vocab"])])
    assert R.score_all(everything, ROW)["grounding"] < R.score_all(note(PERFECT_STEPS, PERFECT_INSTR), ROW)["grounding"]


def test_short_gold_step_is_not_a_hallucination():
    s = R.score_all(note(PERFECT_STEPS + ["- Haemostasis [00:15:00-00:15:20] (conf=0.5)"], PERFECT_INSTR), ROW)
    assert s["step_precision"] == 1.0


def test_malformed_note_gets_only_partial_format():
    s = R.score_all("<note><procedure>x</procedure></note>", ROW)
    assert s["grounding"] == 0.0 and s["calibration"] == 0.0 and 0.0 < s["format"] < 1.0


def test_wrong_section_order_is_rejected():
    bad = note(PERFECT_STEPS, PERFECT_INSTR).replace("<findings>", "<x>").replace("</findings>", "</x>")
    bad = bad.replace("</closure>", f"</closure>\n<findings>{PLACEHOLDER}</findings>")
    assert R.score_all(bad, ROW)["grounding"] == 0.0


def test_calibration_rewards_honest_low_confidence():
    wrong_conf = note(PERFECT_STEPS + ["- Durotomy [00:10:00-00:11:00] (conf=0.9)"], PERFECT_INSTR)
    wrong_hedged = note(PERFECT_STEPS + ["- Durotomy [00:10:00-00:11:00] (conf=0.1)"], PERFECT_INSTR)
    assert R.score_all(wrong_hedged, ROW)["calibration"] > R.score_all(wrong_conf, ROW)["calibration"]


def test_missing_confidence_is_penalised():
    s = R.score_all(note(["- Nasal corridor creation [00:00:00-00:05:00]",
                          "- Tumour excision [00:05:00-00:15:00] (conf=1.0)"], PERFECT_INSTR), ROW)
    assert s["calibration"] < 1.0 and s["format"] < 1.0


def test_recall_is_reliability_weighted():
    miss_reliable = note([PERFECT_STEPS[0]], PERFECT_INSTR)      # misses Tumour excision (0.75)
    miss_unreliable = note([PERFECT_STEPS[1]], PERFECT_INSTR)    # misses Nasal corridor (0.6)
    assert R.score_all(miss_reliable, ROW)["step_recall"] < R.score_all(miss_unreliable, ROW)["step_recall"]


def test_temporal_iou_partial_and_mmss():
    s = R.score_all(note(["- Nasal corridor creation [00:00-02:30] (conf=1.0)",
                          "- Tumour excision [00:05:00-00:15:00] (conf=1.0)"], PERFECT_INSTR), ROW)
    assert abs(s["temporal"] - (0.5 + 1.0) / 2) < 1e-9


def test_invented_complication_loses_safety():
    s = R.score_all(note(PERFECT_STEPS, PERFECT_INSTR, complications="CSF leak"), ROW)
    assert s["safety"] == 0.5


def test_think_block_is_ignored():
    s = R.score_all("<think>Kerrison rongeurs maybe</think>" + note(PERFECT_STEPS, PERFECT_INSTR), ROW)
    assert s["grounding"] == 1.0


def test_ece():
    assert R.expected_calibration_error([(1.0, 1.0), (0.0, 0.0)]) == 0.0
    assert R.expected_calibration_error([(None, 1.0)]) is None


def test_swift_plugin_orms():
    from notegen_rl import swift_plugin as P
    comps = [note(PERFECT_STEPS, PERFECT_INSTR), "garbage"]
    kwargs = {k: [v, v] for k, v in ROW.items()}
    assert P.orms["note_grounding"]()(comps, **kwargs) == [1.0, 0.0]
    assert set(P.orms) >= {"note_format", "note_grounding", "note_calibration", "note_temporal", "note_safety"}
