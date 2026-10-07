"""Render a model completion as a human-readable draft note (Markdown)."""
from __future__ import annotations

from .prompts import SECTIONS
from .rewards import ParsedNote


def to_markdown(completion: str, hedge: float, title: str) -> str:
    p = ParsedNote(completion)
    if not p.ok:
        return f"# Operative note (DRAFT) -- {title}\n\nModel output could not be parsed:\n\n```\n{completion}\n```\n"
    out = [f"# Operative note (DRAFT -- requires surgeon review) -- {title}", ""]
    for tag in SECTIONS:
        out.append(f"## {tag.capitalize()}")
        if tag in ("steps", "instruments"):
            for it in (p.step_items if tag == "steps" else p.instr_items):
                flag = " **[verify]**" if it.conf is None or it.conf < hedge else ""
                out.append(f"- {it.text}{flag}")
        else:
            out.append(p.sections[tag].strip())
        out.append("")
    return "\n".join(out)
