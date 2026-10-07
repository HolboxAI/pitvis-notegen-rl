"""Final, human-readable operative notes from the structured (GRPO) note: plan -> write -> verify.

1. plan()     deterministic content planning: group the structured step lines into surgical phases,
              merge consecutive fragments of the same step, collect low-confidence items and expected
              steps the system did not identify. No language model involved.
2. messages() prompt for a medical instruction-tuned LLM (default: MedGemma-27B-text-it) that turns
              the plan into clinical prose, with one worked example (in-context learning).
3. verify()   deterministic faithfulness check of the prose against the plan: every allowed step or
              instrument name and every HH:MM:SS time mentioned must come from the structured note,
              and findings/complications must stay with the surgeon. Failing prose is regenerated
              once, then replaced by template_prose() (rule-based, always faithful).

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
_TS = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")


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
    return {
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


SYSTEM = (
    "You are a neurosurgical documentation assistant. Rewrite a structured, machine-generated list of "
    "surgical events into a clear operative note in standard clinical English. Use ONLY the events, times "
    "and instruments given. Do not add anatomy, techniques, findings, complications, blood loss, implants "
    "or outcomes that are not listed. Keep every [verify] marker next to the item it belongs to. Leave "
    f"every field the video cannot show as {PLACEHOLDER}. Times are HH:MM:SS from the start of the recording."
)

EXAMPLE_PLAN = """- [Approach] Nasal corridor creation: 00:01:10-00:03:05
- [Approach] Anterior sphenoidotomy: 00:03:05-00:09:40
- [Approach] Sphenoid sinus clearance: 00:09:40-00:21:30, 2 intervals
- [Sellar and dural opening] Sellotomy: 00:21:30-00:27:15
- [Tumour resection and haemostasis] Tumour excision: 00:30:02-00:58:44
- [Tumour resection and haemostasis] Haemostasis: 00:58:44-01:02:10 [verify]
- [Reconstruction and closure] Fat graft placement: 01:02:10-01:04:00
Instruments: Freer elevator, Kerrisons, Pituitary rongeurs, Ring curette, Suction, Tissue glue [verify]
Core steps not identified by the video analysis: Durotomy"""

EXAMPLE_NOTE = f"""OPERATIVE PROCEDURE
Endoscopic transsphenoidal pituitary surgery.

OPERATIVE DETAILS
Approach: The nasal corridor was created (00:01:10-00:03:05), followed by an anterior sphenoidotomy (00:03:05-00:09:40). The sphenoid sinus was then cleared in two intervals between 00:09:40 and 00:21:30.
Sellar and dural opening: A sellotomy was performed (00:21:30-00:27:15). A durotomy was not identified by the video analysis.
Tumour resection and haemostasis: Tumour excision was carried out from 00:30:02 to 00:58:44, followed by haemostasis (00:58:44-01:02:10) [verify].
Reconstruction and closure: A fat graft was placed (01:02:10-01:04:00).

INSTRUMENTS
Freer elevator, Kerrisons, pituitary rongeurs, ring curette, suction and tissue glue [verify].

FINDINGS
{PLACEHOLDER}

COMPLICATIONS
{PLACEHOLDER}"""


def messages(pl: dict, procedure: str) -> list:
    user = (f"Procedure: {procedure}\n\nStructured events:\n{plan_text(pl)}\n\n"
            "Write the note with exactly these headings: OPERATIVE PROCEDURE, OPERATIVE DETAILS, "
            "INSTRUMENTS, FINDINGS, COMPLICATIONS. Under OPERATIVE DETAILS write one short paragraph per phase "
            "that has events, prefixed by the phase name, and mention core steps that were not identified.")
    ex_user = (f"Procedure: {procedure}\n\nStructured events:\n{EXAMPLE_PLAN}\n\n" + user.split("\n\n", 2)[2])
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex_user}, {"role": "assistant", "content": EXAMPLE_NOTE},
            {"role": "user", "content": user}]


def verify(prose: str, pl: dict, step_vocab: list, instr_vocab: list) -> list:
    """List of faithfulness problems (empty = faithful)."""
    problems = []
    allowed_steps = {g["step"] for g in pl["groups"]} | set(pl["not_identified"])
    for s in match_terms(prose, step_vocab):
        if s not in allowed_steps:
            problems.append(f"step not in structured note: {s}")
    allowed_instr = {i["name"] for i in pl["instruments"]}
    for s in match_terms(prose, instr_vocab):
        if s not in allowed_instr:
            problems.append(f"instrument not in structured note: {s}")
    allowed_ts = {g["start_ts"] for g in pl["groups"]} | {g["end_ts"] for g in pl["groups"]}
    for t in set(_TS.findall(prose)):
        if t not in allowed_ts:
            problems.append(f"time not in structured note: {t}")
    for sec in ("FINDINGS", "COMPLICATIONS"):
        m = re.search(sec + r"\s*\n(.*?)(?:\n[A-Z][A-Z ]+\n|\Z)", prose, re.S)
        if not m or PLACEHOLDER not in m.group(1):
            problems.append(f"{sec.lower()} must be left as {PLACEHOLDER}")
    for g in pl["groups"]:
        if g["verify"] and g["step"].lower() in prose.lower():
            seg = prose.lower().split(g["step"].lower(), 1)[1][:250]
            if "[verify]" not in seg:
                problems.append(f"missing [verify] after {g['step']}")
    return problems


def template_prose(pl: dict, procedure: str) -> str:
    """Rule-based fallback realisation (always faithful)."""
    paras = []
    for heading, steps in PHASES:
        gs = [g for g in pl["groups"] if g["phase"] == heading]
        parts = []
        for g in gs:
            when = (f"in {g['intervals']} intervals between {g['start_ts']} and {g['end_ts']}" if g["intervals"] > 1
                    else f"from {g['start_ts']} to {g['end_ts']}")
            parts.append(f"{g['step']} {when}" + (" [verify]" if g["verify"] else ""))
        missing = [s for s in pl["not_identified"] if s in steps]
        if parts or missing:
            text = (f"{heading}: " + "; ".join(parts) + "." if parts else f"{heading}:")
            if missing:
                text += " Not identified by the video analysis: " + ", ".join(missing) + "."
            paras.append(text)
    instr = ", ".join(i["name"] + (" [verify]" if i["verify"] else "") for i in pl["instruments"]) or PLACEHOLDER
    return (f"OPERATIVE PROCEDURE\n{procedure}.\n\nOPERATIVE DETAILS\n" + "\n".join(paras) +
            f"\n\nINSTRUMENTS\n{instr}.\n\nFINDINGS\n{PLACEHOLDER}\n\nCOMPLICATIONS\n{PLACEHOLDER}")
