import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notegen_rl import final_notes as F  # noqa: E402
from notegen_rl.prompts import PLACEHOLDER  # noqa: E402

STEPS = ["Nasal corridor creation", "Anterior sphenoidotomy", "Sphenoid sinus clearance", "Sellotomy", "Durotomy",
         "Tumour excision", "Haemostasis", "Fat graft placement"]
INSTR = ["Suction", "Kerrisons", "Tissue glue"]
NOTE = ("<facts_used>x</facts_used><note><procedure>p</procedure><findings>" + PLACEHOLDER + "</findings><steps>\n"
        "- Nasal corridor creation [00:00:10-00:01:00] (conf=0.95)\n"
        "- Sphenoid sinus clearance [00:01:00-00:05:00] (conf=0.90)\n"
        "- Sphenoid sinus clearance [00:05:30-00:08:00] (conf=0.55)\n"
        "- Tumour excision [00:10:00-00:30:00] (conf=0.92)\n"
        "</steps><instruments>\n- Suction (conf=0.9)\n- Tissue glue (conf=0.4)\n</instruments>"
        "<complications>" + PLACEHOLDER + "</complications><closure>" + PLACEHOLDER + "</closure></note>")


def test_plan_merges_fragments_and_flags():
    pl = F.plan(NOTE, STEPS, INSTR)
    sinus = [g for g in pl["groups"] if g["step"] == "Sphenoid sinus clearance"][0]
    assert sinus["intervals"] == 2 and sinus["start_ts"] == "00:01:00" and sinus["end_ts"] == "00:08:00"
    assert sinus["verify"]                       # one fragment below 0.6
    assert "Sellotomy" in pl["not_identified"] and "Durotomy" in pl["not_identified"]
    assert [i["name"] for i in pl["instruments"] if i["verify"]] == ["Tissue glue"]


def test_plan_lists_each_step_once_with_best_confidence():
    pl = F.plan(NOTE, STEPS, INSTR)
    sinus = [e for e in pl["steps"] if e["step"] == "Sphenoid sinus clearance"][0]
    assert sinus["occurrences"] == 1 and sinus["max_conf"] == 0.90 and not sinus["verify"]
    assert [e["step"] for e in pl["steps"]] == ["Nasal corridor creation", "Sphenoid sinus clearance", "Tumour excision"]


def test_template_prose_is_faithful_narrative():
    pl = F.plan(NOTE, STEPS, INSTR)
    text = F.template_prose(pl, "Endoscopic transsphenoidal pituitary surgery")
    assert text.startswith("The patient underwent endoscopic transsphenoidal pituitary surgery for " + F.INDICATION_PH)
    assert "00:" not in text and "tissue glue [verify]" in text and F.POSTOP_PH in text
    assert F.verify(text, pl, STEPS, INSTR) == []


def test_verify_reads_paraphrases_and_catches_invented_or_missing_steps():
    pl = F.plan(NOTE, STEPS, INSTR)
    good = F.template_prose(pl, "p")
    assert any("Durotomy" in p for p in F.verify(good.replace("the tumour was excised",
                                                              "the dura was opened and the tumour was excised"),
                                                 pl, STEPS, INSTR))
    assert any("not mentioned: Tumour excision" in p for p in F.verify(good.replace(" Finally, the tumour was excised.", ""),
                                                                      pl, STEPS, INSTR))
    assert any("Kerrisons" in p for p in F.verify(good.replace("suction", "suction, Kerrisons"), pl, STEPS, INSTR))
    assert any("times" in p for p in F.verify(good.replace("was created", "was created at 00:00:10"), pl, STEPS, INSTR))


def test_verify_rejects_what_the_video_cannot_show_unless_given():
    pl = F.plan(NOTE, STEPS, INSTR)
    good = F.template_prose(pl, "p")
    claim = good.replace(F.POSTOP_PH, "The patient tolerated the procedure well and was discharged home.")
    assert any("cannot support" in p for p in F.verify(claim, pl, STEPS, INSTR))
    case = {"indication": "a non-functioning pituitary macroadenoma",
            "postoperative": "The patient tolerated the procedure well and was discharged home."}
    given = F.template_prose(pl, "p", case)
    assert "macroadenoma" in given and F.verify(given, pl, STEPS, INSTR, case) == []
    assert any("complications" in p for p in F.verify(good.replace("Complications: " + PLACEHOLDER, "Complications: None."),
                                                      pl, STEPS, INSTR))


def test_verify_keeps_verify_markers_and_strips_reasoning():
    note = NOTE.replace("Tumour excision [00:10:00-00:30:00] (conf=0.92)", "Tumour excision [00:10:00-00:30:00] (conf=0.42)")
    pl = F.plan(note, STEPS, INSTR)
    good = F.template_prose(pl, "p")
    assert "the tumour was excised [verify]" in good and F.verify(good, pl, STEPS, INSTR) == []
    assert any("missing [verify]" in p for p in F.verify(good.replace("excised [verify]", "excised"), pl, STEPS, INSTR))
    assert F.verify("<unused94>thought\nplan<unused95>" + good, pl, STEPS, INSTR) == []


def test_phrases_allow_an_adverb():
    assert {"Sellotomy", "Durotomy", "Tumour excision"} <= F.mentioned_steps(
        "The sella was then opened, the dura was subsequently opened and the tumour was carefully excised.", STEPS)


def test_plural_step_mentions_count():
    steps = STEPS + ["Septum displacement"]
    found = F.mentioned_steps("repeated anterior sphenoidotomies, repeated septum displacements and repeated "
                              "sphenoid sinus clearings; repeated durotomies", steps)
    assert {"Anterior sphenoidotomy", "Septum displacement", "Sphenoid sinus clearance", "Durotomy"} <= found
    assert "Durotomy" not in F.mentioned_steps("a durotromy was performed", steps)   # misspelt names do not count


def test_set_flags_places_markers_from_the_plan():
    note = NOTE.replace("Tumour excision [00:10:00-00:30:00] (conf=0.92)", "Tumour excision [00:10:00-00:30:00] (conf=0.42)")
    pl = F.plan(note, STEPS, INSTR)
    good = F.template_prose(pl, "p")
    # the writer dropped one marker and invented another; the plan decides
    written = good.replace("excised [verify]", "excised").replace("suction", "suction [verify]")
    fixed = F.set_flags(written, pl)
    assert "the tumour was excised [verify]" in fixed and "suction [verify]" not in fixed
    assert "tissue glue [verify]" in fixed and fixed.count("[verify]") == 2
    assert F.verify(fixed, pl, STEPS, INSTR) == []
    assert F.set_flags(fixed, pl) == fixed                      # idempotent
    clause = F.set_flags(f"The patient underwent p for {F.INDICATION_PH}. The tumour was carefully removed, "
                         f"and the cavity was inspected. {F.POSTOP_PH}", pl)
    assert "The tumour was carefully removed [verify], and" in clause
