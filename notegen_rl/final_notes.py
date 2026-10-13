"""Final operative notes in plain narrative language from the structured (GRPO) note: plan -> write -> verify.

1. plan()     deterministic content planning: group the structured step lines into surgical phases,
              merge consecutive fragments of the same step, collect low-confidence items and expected
              steps the system did not identify. No language model involved.
2. messages() prompt for a medical instruction-tuned LLM (default: MedGemma-27B-text-it) that turns
              the plan into a dictated-style narrative ("The patient underwent ... The operation began
              with ..."), with one worked example (in-context learning). Indication and postoperative
              course come only from optional case details, otherwise they stay surgeon placeholders.
3. set_flags() deterministic [verify] markers: the writer's markers are removed and one is placed after
              the first mention of every uncertain step (end of its clause) and instrument (after its name).
4. verify()   deterministic faithfulness check of the narrative against the plan: every step it
              mentions (by name or by a standard phrase, e.g. "the sella was opened") and every
              instrument must come from the structured note, every identified step must be mentioned,
              uncertain steps keep their [verify] marker, findings/complications stay with the surgeon,
              and nothing the video cannot show (indication, diagnosis, outcome, discharge, blood loss)
              is stated unless it was given as a case detail. Failing prose is regenerated once, then
              replaced by template_prose() (rule-based, always faithful).

Design follows template-filling for pituitary op notes (Das et al., UCL) for faithfulness, and
LLM summarisation with in-context examples (Van Veen et al., Nat Med 2024) for readability; the
verifier addresses the fabrication risk those studies report.
"""
from __future__ import annotations

import re

from .prompts import PLACEHOLDER, fmt_ts
from .rewards import ParsedNote, match_terms

HEDGE = 0.6
PHASES = [  # (heading, steps) in surgical order
    ("Approach", ["Nasal corridor creation", "Septum displacement", "Anterior sphenoidotomy",
                  "Sphenoid sinus clearance"]),
    ("Sellar and dural opening", ["Sellotomy", "Durotomy"]),
    ("Tumour resection and haemostasis", ["Tumour excision", "Haemostasis", "Debris clearance"]),
    ("Reconstruction and closure", ["Synthetic graft placement", "Fat graft placement", "Gasket seal construct",
                                    "Dural sealant", "Nasal packing"]),
]
CORE_STEPS = ["Nasal corridor creation", "Anterior sphenoidotomy", "Sphenoid sinus clearance", "Sellotomy",
              "Durotomy", "Tumour excision", "Haemostasis"]
_TS = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
INDICATION_PH = "[INDICATION — SURGEON TO COMPLETE]"
POSTOP_PH = "[POSTOPERATIVE COURSE — SURGEON TO COMPLETE]"

# Plain-language clause per step (rule-based narrative) and the phrases that count as mentioning it.
STEP_CLAUSE = {
    "Nasal corridor creation": "the nasal corridor was created",
    "Septum displacement": "the nasal septum was displaced",
    "Anterior sphenoidotomy": "an anterior sphenoidotomy was performed",
    "Sphenoid sinus clearance": "the sphenoid sinus was cleared",
    "Sellotomy": "the sella was opened",
    "Durotomy": "the dura was opened",
    "Tumour excision": "the tumour was excised",
    "Haemostasis": "haemostasis was secured",
    "Debris clearance": "the operative field was cleared of debris",
    "Synthetic graft placement": "a synthetic graft was placed",
    "Fat graft placement": "a fat graft was placed",
    "Gasket seal construct": "a gasket seal was constructed",
    "Dural sealant": "dural sealant was applied",
    "Nasal packing": "the nose was packed",
}
STEP_PHRASES = {
    "Nasal corridor creation": r"nasal corridor",
    "Septum displacement": r"septum was (?:displaced|deviated|reflected)|displace\w* (?:of )?the (?:nasal )?septum|septal displacement",
    "Anterior sphenoidotomy": r"sphenoidotomy|sphenoid(?:al)? (?:sinus )?was opened|opened the sphenoid|anterior wall of the sphenoid",
    "Sphenoid sinus clearance": r"sphenoid sinus was cleared|clear\w* (?:of )?the sphenoid|sphenoid sinus clearance|sinus was cleared",
    "Sellotomy": r"sellotomy|sell(?:a|ar floor) was opened|open\w* (?:of )?the sell(?:a|ar floor)|sellar opening",
    "Durotomy": r"durotomy|dura (?:mater )?was (?:opened|incised)|open\w* (?:of )?the dura|incis\w* (?:of )?the dura|dural opening",
    "Tumour excision": r"tumou?r (?:was )?(?:excised|resected|removed)|(?:excision|resection|removal) of the tumou?r|tumou?r excision|tumou?r resection",
    "Haemostasis": r"ha?emostasis",
    "Debris clearance": r"debris",
    "Synthetic graft placement": r"synthetic graft",
    "Fat graft placement": r"fat graft",
    "Gasket seal construct": r"gasket",
    "Dural sealant": r"(?:dural )?sealant",
    "Nasal packing": r"(?:nasal )?pack(?:ing|ed)",
}
# plural forms ("repeated anterior sphenoidotomies", "septum displacements", "sinus clearings")
for _k, _a, _b in (("Anterior sphenoidotomy", "sphenoidotomy", "sphenoidotom(?:y|ies)"),
                   ("Sellotomy", "sellotomy", "sellotom(?:y|ies)"), ("Durotomy", "durotomy", "durotom(?:y|ies)"),
                   ("Septum displacement", "septal displacement", "septal displacements?"),
                   ("Sphenoid sinus clearance", "sphenoid sinus clearance", "sphenoid sinus clear(?:ance|ings?)")):
    STEP_PHRASES[_k] = STEP_PHRASES[_k].replace(_a, _b)
STEP_PHRASES["Septum displacement"] += r"|septum displacements?"
# "was opened" also matches "was then opened", "was subsequently opened", ...
STEP_PHRASES = {k: v.replace("was ", r"was (?:\w+ )?") for k, v in STEP_PHRASES.items()}
# Statements the video cannot support; allowed only inside text given as a case detail.
UNOBSERVABLE = re.compile(
    r"tolerat\w*|discharg\w*|uneventful|extubat\w*|recover\w*|stable condition|blood loss|transfus\w*|"
    r"no (?:intra-?operative )?complications|without complications?|(?:macro|micro)?adenoma|carcinoma|"
    r"craniopharyngioma|meningioma|rathke|apoplexy|csf leak|cerebrospinal fluid leak|gross total|subtotal|"
    r"histolog\w*|patholog\w*|diagnos\w*|vision (?:improved|was)|visual field", re.I)


def plan(completion: str, step_vocab: list, instr_vocab: list) -> dict:
    p = ParsedNote(completion)
    lines = []
    for it in p.step_items:
        names = match_terms(it.text, step_vocab)
        if len(names) == 1 and it.start is not None and it.end is not None and it.end > it.start:
            lines.append({"step": names[0], "start": it.start, "end": it.end, "conf": it.conf})
    lines.sort(key=lambda d: d["start"])
    groups = []  # consecutive lines of the same step -> one group
    for ln in lines:
        if groups and groups[-1]["step"] == ln["step"]:
            g = groups[-1]
            g["end"] = max(g["end"], ln["end"])
            g["intervals"] += 1
            g["min_conf"] = min(g["min_conf"], ln["conf"] if ln["conf"] is not None else 0.0)
        else:
            groups.append({"step": ln["step"], "start": ln["start"], "end": ln["end"], "intervals": 1,
                           "min_conf": ln["conf"] if ln["conf"] is not None else 0.0})
    phase_of = {s: h for h, steps in PHASES for s in steps}
    instruments = []
    for it in p.instr_items:
        names = match_terms(it.text, instr_vocab)
        if len(names) == 1 and names[0] not in [i["name"] for i in instruments]:
            instruments.append({"name": names[0], "conf": it.conf if it.conf is not None else 0.0})
    present = {g["step"] for g in groups}
    steps = {}  # each identified step once, in order of first appearance, with its best confidence
    for g in groups:
        e = steps.setdefault(g["step"], {"step": g["step"], "phase": phase_of.get(g["step"], "Other"),
                                         "occurrences": 0, "max_conf": 0.0})
        e["occurrences"] += 1
        e["max_conf"] = max(e["max_conf"], max(ln["conf"] or 0.0 for ln in lines
                                               if ln["step"] == g["step"] and g["start"] <= ln["start"] <= g["end"]))
    return {
        "steps": [dict(e, verify=e["max_conf"] < HEDGE) for e in steps.values()],
        "parsed": p.ok,
        "groups": [dict(g, phase=phase_of.get(g["step"], "Other"),
                        verify=g["min_conf"] < HEDGE,
                        start_ts=fmt_ts(g["start"]), end_ts=fmt_ts(g["end"])) for g in groups],
        "instruments": [dict(i, verify=i["conf"] < HEDGE) for i in instruments],
        "not_identified": [s for s in CORE_STEPS if s not in present],
        "closure_steps": [g["step"] for g in groups if phase_of.get(g["step"]) == "Reconstruction and closure"],
        "span": (fmt_ts(groups[0]["start"]), fmt_ts(groups[-1]["end"])) if groups else None,
    }


def plan_text(pl: dict) -> str:
    out = []
    for g in pl["groups"]:
        extra = f", {g['intervals']} intervals" if g["intervals"] > 1 else ""
        out.append(f"- [{g['phase']}] {g['step']}: {g['start_ts']}-{g['end_ts']}{extra}"
                   + (" [verify]" if g["verify"] else ""))
    out.append("Instruments: " + ", ".join(i["name"] + (" [verify]" if i["verify"] else "") for i in pl["instruments"]))
    out.append("Core steps not identified by the video analysis: " + (", ".join(pl["not_identified"]) or "none"))
    return "\n".join(out)


def narrative_plan_text(pl: dict, case: dict | None = None) -> str:
    """Plan for the narrative writer: steps per phase in surgical order, without times."""
    case = case or {}
    out = [f"Indication: {case.get('indication') or 'not provided'}",
           f"Postoperative course: {case.get('postoperative') or 'not provided'}",
           "Events in the order they occurred (uncertain items are marked [verify]):"]
    for heading, _ in PHASES + [("Other", [])]:
        st = [e for e in pl["steps"] if e["phase"] == heading]
        if st:
            out.append(f"- {heading}: " + "; ".join(
                e["step"] + (" (repeatedly)" if e["occurrences"] > 1 else "") + (" [verify]" if e["verify"] else "")
                for e in st))
    out.append("Instruments: " + (", ".join(i["name"] + (" [verify]" if i["verify"] else "")
                                            for i in pl["instruments"]) or "none identified"))
    out.append("Not identified on the video: " + (", ".join(pl["not_identified"]) or "none"))
    return "\n".join(out)


SYSTEM = (
    "You are a neurosurgical documentation assistant. Turn a machine-generated list of surgical events into "
    "the operative note a surgeon would dictate: plain clinical English, past tense, one flowing paragraph "
    "that tells the operation in order, followed by three short lines. Use ONLY the events and instruments "
    "given, in the order given; mention every event once; do not add anatomy, techniques, findings, "
    "complications, blood loss, diagnoses, implants or outcomes that are not listed; do not give times. "
    "Keep every [verify] marker directly after the item it belongs to. Use the indication and postoperative "
    f"course only if they are provided; otherwise write {INDICATION_PH} and {POSTOP_PH}. Findings and "
    f"complications are always {PLACEHOLDER}."
)

EXAMPLE_PLAN = """Indication: not provided
Postoperative course: not provided
Events in the order they occurred (uncertain items are marked [verify]):
- Approach: Nasal corridor creation; Anterior sphenoidotomy; Sphenoid sinus clearance
- Sellar and dural opening: Sellotomy
- Tumour resection and haemostasis: Tumour excision; Haemostasis (repeatedly) [verify]
- Reconstruction and closure: Fat graft placement
Instruments: Freer elevator, Kerrisons, Pituitary rongeurs, Ring curette, Suction, Tissue glue [verify]
Not identified on the video: Durotomy"""

EXAMPLE_NOTE = f"""The patient underwent endoscopic transsphenoidal pituitary surgery for {INDICATION_PH}. The operation began with creation of the nasal corridor, followed by an anterior sphenoidotomy, and the sphenoid sinus was cleared. The sella was then opened. The tumour was excised, and haemostasis was secured repeatedly during the resection [verify]. The defect was reconstructed with a fat graft. Instruments used included the Freer elevator, Kerrisons, pituitary rongeurs, ring curette, suction and tissue glue [verify]. {POSTOP_PH}

Findings: {PLACEHOLDER}
Complications: {PLACEHOLDER}
Not identified on the video (surgeon to confirm): durotomy."""


def messages(pl: dict, procedure: str, case: dict | None = None) -> list:
    task = ("Write the note: one narrative paragraph, then the lines 'Findings:', 'Complications:' and "
            "'Not identified on the video (surgeon to confirm):'.")
    user = f"Procedure: {procedure}\n{narrative_plan_text(pl, case)}\n\n{task}"
    ex_user = f"Procedure: {procedure}\n{EXAMPLE_PLAN}\n\n{task}"
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex_user}, {"role": "assistant", "content": EXAMPLE_NOTE},
            {"role": "user", "content": user}]


def clean(text: str) -> str:
    """Writer output without reasoning blocks or surrounding whitespace."""
    from .rewards import _THINK
    return _THINK.sub("", text or "").strip()


def mentioned_steps(text: str, step_vocab: list) -> set:
    found = set(match_terms(text, step_vocab))
    for step, pat in STEP_PHRASES.items():
        if step in step_vocab and re.search(pat, text, re.I):
            found.add(step)
    return found


_FLAG = re.compile(r"[ \t]*[\[(]verify[\])]", re.I)
_CLAUSE_END = re.compile(r"[,;:\n]|[.!?](?=\s|$)")


def _flag_pos(narrative: str, name: str, is_step: bool) -> int | None:
    """Where the [verify] marker of name goes: end of the clause of its first mention (steps) or
    directly after it (instruments); None when the narrative does not mention it."""
    if is_step:
        pat = re.escape(name) + (f"|{STEP_PHRASES[name]}" if name in STEP_PHRASES else "")
    else:
        pat = r"(?<![a-z0-9])" + re.escape(name) + r"(?:e?s)?(?![a-z0-9])"
    m = re.search(pat, narrative, re.I)
    if not m:
        return None
    if not is_step:
        return m.end()
    end = _CLAUSE_END.search(narrative, m.end())
    return end.start() if end else len(narrative.rstrip())


def set_flags(prose: str, pl: dict) -> str:
    """Narrative with the plan's [verify] markers: the writer's own markers are removed, then one is
    placed after the first mention of every uncertain step and instrument."""
    text = clean(prose)
    narrative, sep, tail = text.partition("\nFindings:")
    narrative = _FLAG.sub("", narrative)
    flagged = [(e["step"], True) for e in pl["steps"] if e["verify"]]
    flagged += [(i["name"], False) for i in pl["instruments"] if i["verify"]]
    marks = {pos for pos in (_flag_pos(narrative, n, is_step) for n, is_step in flagged) if pos is not None}
    for pos in sorted(marks, reverse=True):
        narrative = narrative[:pos].rstrip() + " [verify]" + narrative[pos:]
    return narrative + sep + tail


def _sentences(text: str) -> list:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def verify(prose: str, pl: dict, step_vocab: list, instr_vocab: list, case: dict | None = None) -> list:
    """List of faithfulness problems (empty = faithful)."""
    problems = []
    prose = clean(prose)
    case = case or {}
    narrative, _, tail = prose.partition("\nFindings:")
    tail = "Findings:" + tail
    identified = {e["step"] for e in pl["steps"]}
    for s in sorted(mentioned_steps(narrative, step_vocab) - identified):
        problems.append(f"step not in structured note: {s}")
    for s in sorted(identified - mentioned_steps(narrative, step_vocab)):
        problems.append(f"identified step not mentioned: {s}")
    allowed_instr = {i["name"] for i in pl["instruments"]}
    for s in match_terms(prose, instr_vocab):
        if s not in allowed_instr:
            problems.append(f"instrument not in structured note: {s}")
    if _TS.search(narrative):
        problems.append("times are not part of the narrative")
    for label in ("Findings", "Complications"):
        m = re.search(label + r":\s*(.*)", tail)
        if not m or PLACEHOLDER not in m.group(1):
            problems.append(f"{label.lower()} must be left as {PLACEHOLDER}")
    checked = narrative
    for key in ("indication", "postoperative"):
        if case.get(key):
            checked = re.sub(re.escape(case[key]), " ", checked, flags=re.I)
    for m in UNOBSERVABLE.finditer(checked):
        problems.append(f"statement the video cannot support: '{m.group(0)}'")
    if not case.get("indication") and INDICATION_PH not in narrative:
        problems.append(f"indication must be {INDICATION_PH}")
    if not case.get("postoperative") and POSTOP_PH not in narrative:
        problems.append(f"postoperative course must be {POSTOP_PH}")
    sents = _sentences(narrative)
    for e in pl["steps"]:
        if e["verify"]:
            hits = [s for s in sents if e["step"] in mentioned_steps(s, step_vocab)]
            if hits and not any("[verify]" in s for s in hits):
                problems.append(f"missing [verify] for {e['step']}")
    return problems


def _join(parts: list) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def template_prose(pl: dict, procedure: str, case: dict | None = None) -> str:
    """Rule-based narrative (always faithful): one sentence per surgical phase."""
    case = case or {}
    sents = [f"The patient underwent {procedure[:1].lower() + procedure[1:]} for "
             f"{case.get('indication') or INDICATION_PH}."]
    lead = ["First,", "Next,", "Then,", "Finally,"]
    phases = [h for h, _ in PHASES + [("Other", [])] if any(e["phase"] == h for e in pl["steps"])]
    for k, heading in enumerate(phases):
        clauses = []
        for e in [e for e in pl["steps"] if e["phase"] == heading]:
            c = STEP_CLAUSE.get(e["step"], f"{e['step'].lower()} was performed")
            if e["occurrences"] > 1 and e["step"] == "Haemostasis":
                c += " repeatedly"
            clauses.append(c + (" [verify]" if e["verify"] else ""))
        opener = lead[0] if k == 0 else (lead[-1] if k == len(phases) - 1 else lead[min(k, 2)])
        sents.append(f"{opener} {_join(clauses)}.")
    if pl["instruments"]:
        names = [i["name"][:1].lower() + i["name"][1:] if i["name"] not in ("Kerrisons", "Cottle", "Freer elevator")
                 else i["name"] for i in pl["instruments"]]
        names = [n + (" [verify]" if i["verify"] else "") for n, i in zip(names, pl["instruments"])]
        sents.append(f"Instruments used included {_join(names)}.")
    sents.append(case.get("postoperative") or POSTOP_PH)
    missing = ", ".join(s.lower() for s in pl["not_identified"]) or "none"
    return (" ".join(sents) + f"\n\nFindings: {PLACEHOLDER}\nComplications: {PLACEHOLDER}\n"
            f"Not identified on the video (surgeon to confirm): {missing}.")
