import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from notegen_rl.data import Taxonomy, VideoLabels  # noqa: E402
from notegen_rl.facts import (facts_from_labels, facts_from_predictions, gold_sets,  # noqa: E402
                              merge_short, runs, viterbi)
from notegen_rl.prompts import build_messages, render_template_note  # noqa: E402
from notegen_rl.rewards import ParsedNote  # noqa: E402
from notegen_rl.step_model import estimate_transitions  # noqa: E402

TAX = Taxonomy(step_codes=[-1, 1, 2], step_names=["Out of patient", "Nasal corridor creation", "Tumour excision"],
               instr_codes=[1, 2], instr_names=["Suction", "Bipolar forceps"], non_surgical=["Out of patient"])
FCFG = {"smoothing": "viterbi", "median_window_s": 5, "min_segment_s": 5,
        "instrument_threshold": 0.5, "instrument_min_fraction": 0.2}


def one_hot_probs(path, n, p=0.8):
    probs = np.full((len(path), n), (1 - p) / (n - 1))
    probs[np.arange(len(path)), path] = p
    return probs


def test_taxonomy_drops_no_instrument_codes_listed_twice(tmp_path):
    """PitVis lists instrument 0 as both no_visible_instrument and occluded_image_inside_patient."""
    from notegen_rl.data import PitVisLayout, build_taxonomy
    (tmp_path / "map_instruments.csv").write_text(
        "int_instrument,str_instrument\n-1,out_of_patient\n-2,no_secondary_instrument\n"
        "0,no_visible_instrument\n0,occluded_image_inside_patient\n1,bipolar_forceps\n16,suction\n")
    (tmp_path / "map_steps.csv").write_text(  # PitVis lists -1 under three names
        "int_step,str_step\n-1,operation_ended\n-1,operation_not_started\n-1,out_of_patient\n"
        "1,nasal_corridor_creation\n")
    layout = PitVisLayout(tmp_path, {}, {}, tmp_path / "map_steps.csv", tmp_path / "map_instruments.csv")
    tables = {1: np.array([[0, 1, 0, -2], [1, 1, 16, 1], [2, -1, -1, -2]])}
    cfg = {"data": {"instrument_absent_codes": [-1, -2], "non_surgical_steps": ["out of patient"]}}
    tax = build_taxonomy(layout, tables, cfg)
    assert tax.instr_names == ["Bipolar forceps", "Suction"]
    assert tax.step_names == ["Out of patient", "Nasal corridor creation"]
    assert tax.non_surgical == ["Out of patient"]
    assert tax.surgical_steps == ["Nasal corridor creation"]


def test_runs_and_merge_short():
    path = np.array([1] * 10 + [2] * 2 + [1] * 10 + [2] * 20)
    assert runs(path)[0] == (1, 0, 10)
    merged = merge_short(runs(path), 5)
    assert merged == [(1, 0, 22), (2, 22, 42)]


def test_viterbi_removes_flicker():
    truth = np.array([1] * 30 + [2] * 30)
    noisy = truth.copy()
    noisy[[10, 40]] = [2, 1]
    lt, li = estimate_transitions([truth, truth], 3)
    path = viterbi(np.log(one_hot_probs(noisy, 3, 0.6)), lt, li)
    assert (path == truth).all()


def test_facts_from_predictions_drops_non_surgical_and_lists_instruments():
    path = np.array([0] * 10 + [1] * 30 + [2] * 30)
    instr = np.zeros((70, 2))
    instr[10:40, 0] = 0.9           # suction during nasal corridor
    instr[45:48, 1] = 0.9           # bipolar too briefly (<20%) -> not listed
    facts, _ = facts_from_predictions("v", np.arange(70), one_hot_probs(path, 3), instr, TAX,
                                      {**FCFG, "smoothing": "none"})
    assert [s.step for s in facts.segments] == ["Nasal corridor creation", "Tumour excision"]
    assert facts.segments[0].instruments == {"Suction": 0.9}
    assert facts.segments[1].instruments == {}
    assert facts.segments[0].start_s == 10 and facts.segments[0].end_s == 40


def test_labels_gold_sets_and_prompt_roundtrip():
    steps = np.array([0] * 5 + [1] * 20 + [2] * 3)
    instr = np.zeros((28, 2), dtype=np.uint8)
    instr[5:25, 0] = 1
    lab = VideoLabels(np.arange(28), steps, instr)
    gs, gi = gold_sets(lab, TAX, min_step_s=10, min_instr_s=5)
    assert gs == ["Nasal corridor creation"] and gi == ["Suction"]
    facts = facts_from_labels("v", lab, TAX)
    assert [s.step for s in facts.segments] == ["Nasal corridor creation", "Tumour excision"]

    msgs = build_messages(facts, {"Suction": 0.8}, TAX.surgical_steps, TAX.instr_names, "Pituitary surgery")
    assert msgs[0]["role"] == "system" and "Allowed step names" in msgs[1]["content"]
    # the template baseline must satisfy the reward parser's format
    p = ParsedNote(render_template_note(facts, "Pituitary surgery"))
    assert p.ok and len(p.step_items) == 2 and all(i.conf is not None for i in p.step_items)
