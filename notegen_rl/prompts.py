"""Prompt construction, the output format, and the deterministic template baseline.

Output format (parsed by rewards.py -- change both together):

<facts_used>...</facts_used>
<note>
<procedure>...</procedure>
<findings>[SURGEON TO COMPLETE]</findings>
<steps>
- Step name [HH:MM:SS-HH:MM:SS] (conf=0.85)
</steps>
<instruments>
- Instrument name (conf=0.90)
</instruments>
<complications>[SURGEON TO COMPLETE]</complications>
<closure>...</closure>
</note>
"""
from __future__ import annotations

PLACEHOLDER = "[SURGEON TO COMPLETE]"
SECTIONS = ["procedure", "findings", "steps", "instruments", "complications", "closure"]
CLOSURE_KEYWORDS = ("graft", "seal", "packing", "closure", "reconstruction")

SYSTEM_PROMPT = (
    "You draft structured operative notes for endoscopic pituitary surgery from the output of an "
    "automatic video-analysis system. That system makes mistakes: each detection comes with the "
    "model's confidence and with how reliable that system has historically been for that class. "
    "Only document steps and instruments that the evidence supports, give each one an honest "
    "confidence between 0 and 1, and never invent findings or complications -- those cannot be "
    "observed by the system and are left for the surgeon. Every note is a draft that a surgeon "
    "reviews and signs."
)


def fmt_ts(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render_perception(facts, reliability: dict) -> str:
    lines = []
    for seg in facts.segments:
        instr = ", ".join(f"{n} ({c:.2f})" for n, c in sorted(seg.instruments.items(), key=lambda x: -x[1]))
        rel = reliability.get(seg.step)
        rel_txt = f"; historical reliability {rel:.2f}" if rel is not None else ""
        lines.append(f"- [{fmt_ts(seg.start_s)}-{fmt_ts(seg.end_s)}] {seg.step} "
                     f"(confidence {seg.conf:.2f}{rel_txt}) | instruments: {instr or 'none detected'}")
    return "\n".join(lines) if lines else "(the system detected no surgical steps in this interval)"


def render_instrument_reliability(instr_names: list, reliability: dict) -> str:
    parts = [f"{n} {reliability[n]:.2f}" for n in instr_names if n in reliability]
    return ", ".join(parts) if parts else "unknown"


def build_messages(facts, reliability: dict, step_vocab: list, instr_vocab: list,
                   procedure: str, window: tuple | None = None) -> list:
    scope = (f"This excerpt covers {fmt_ts(window[0])}-{fmt_ts(window[1])} of the procedure. "
             f"Document only what happens inside it.\n" if window else
             "This is the complete procedure.\n")
    user = f"""Procedure: {procedure}
{scope}
Automatic perception output, time-ordered (may contain errors):
{render_perception(facts, reliability)}

Historical reliability of instrument detection: {render_instrument_reliability(instr_vocab, reliability)}

Allowed step names: {"; ".join(step_vocab)}
Allowed instrument names: {"; ".join(instr_vocab)}

Write the note in EXACTLY this format, sections in this order, nothing outside the tags:

<facts_used>one or two sentences on which detections you relied on and which you doubted</facts_used>
<note>
<procedure>{procedure}</procedure>
<findings>{PLACEHOLDER}</findings>
<steps>
- <allowed step name> [HH:MM:SS-HH:MM:SS] (conf=<0.00-1.00>)
</steps>
<instruments>
- <allowed instrument name> (conf=<0.00-1.00>)
</instruments>
<complications>{PLACEHOLDER}</complications>
<closure>closure/reconstruction steps supported by the evidence, else {PLACEHOLDER}</closure>
</note>

Rules: one step or instrument per line; use the allowed names verbatim; omit anything you
believe did not happen; conf is the probability the line is correct."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def render_template_note(facts, procedure: str) -> str:
    """Deterministic note straight from the perception facts -- the no-LLM baseline and the
    optional SFT format target. Confidences are the perception model's own."""
    steps = "\n".join(f"- {s.step} [{fmt_ts(s.start_s)}-{fmt_ts(s.end_s)}] (conf={s.conf:.2f})"
                      for s in facts.segments)
    instr = {}
    for s in facts.segments:
        for n, c in s.instruments.items():
            instr[n] = max(instr.get(n, 0.0), c)
    instruments = "\n".join(f"- {n} (conf={c:.2f})" for n, c in sorted(instr.items(), key=lambda x: -x[1]))
    closure_steps = [s.step for s in facts.segments if any(k in s.step.lower() for k in CLOSURE_KEYWORDS)]
    closure = ("Closure steps recorded: " + ", ".join(dict.fromkeys(closure_steps)) + "."
               if closure_steps else PLACEHOLDER)
    return (f"<facts_used>All {len(facts.segments)} perceived segments used as-is.</facts_used>\n"
            f"<note>\n<procedure>{procedure}</procedure>\n<findings>{PLACEHOLDER}</findings>\n"
            f"<steps>\n{steps}\n</steps>\n<instruments>\n{instruments}\n</instruments>\n"
            f"<complications>{PLACEHOLDER}</complications>\n<closure>{closure}</closure>\n</note>")
