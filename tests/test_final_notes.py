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


def test_template_prose_is_faithful():
    pl = F.plan(NOTE, STEPS, INSTR)
    assert F.verify(F.template_prose(pl, "Endoscopic transsphenoidal pituitary surgery"), pl, STEPS, INSTR) == []


def test_verify_catches_invented_content():
    pl = F.plan(NOTE, STEPS, INSTR)
    good = F.template_prose(pl, "p")
    assert any("Kerrisons" in p for p in F.verify(good.replace("Suction", "Suction, Kerrisons"), pl, STEPS, INSTR))
    assert any("time" in p for p in F.verify(good.replace("00:30:00", "00:31:00"), pl, STEPS, INSTR))
    bad = good.replace("COMPLICATIONS\n" + PLACEHOLDER, "COMPLICATIONS\nNone.")
    assert any("complications" in p for p in F.verify(bad, pl, STEPS, INSTR))
