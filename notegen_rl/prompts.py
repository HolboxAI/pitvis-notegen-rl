"""Prompt construction, the output format, and the deterministic template baseline.

Output format (parsed by rewards.py -- change both together):

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
IMAGE_TAG = "<image>"            # one per keyframe, in order (ms-swift multimodal dataset format)
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

KEYFRAME_SYSTEM = (
    " Keyframes from the endoscopic video are shown under the detections they were taken from. "
    "Use them to check each detection: keep what the frames support, and lower the confidence of, "
    "or omit, detections the frames contradict. The frames do not replace the timeline -- times "
    "and order come from the detections. Do not describe anatomy or pathology from the frames."
)

KEYFRAME_USER = (
    "Under each detection, \"keyframe HH:MM:SS\" lines show frames taken from inside that "
    "detected segment at that time."
)


def fmt_ts(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render_perception(facts, reliability: dict, keyframes: list | None = None) -> str:
    by_seg = {}
    for k in sorted(keyframes or [], key=lambda k: (k.segment, k.t)):
        by_seg.setdefault(k.segment, []).append(k)
    lines = []
    for si, seg in enumerate(facts.segments):
        instr = ", ".join(f"{n} ({c:.2f})" for n, c in sorted(seg.instruments.items(), key=lambda x: -x[1]))
        rel = reliability.get(seg.step)
        rel_txt = f"; historical reliability {rel:.2f}" if rel is not None else ""
        lines.append(f"- [{fmt_ts(seg.start_s)}-{fmt_ts(seg.end_s)}] {seg.step} "
                     f"(confidence {seg.conf:.2f}{rel_txt}) | instruments: {instr or 'none detected'}")
        # text follows each image tag on its line: vLLM's Gemma 3 processor merges the newlines it inserts
        # around an image with an adjacent newline token, and then no longer finds the image placeholder
        lines += [f"  keyframe {fmt_ts(k.t)}: {IMAGE_TAG} (frame {j} of this segment)"
                  for j, k in enumerate(by_seg.get(si, []), 1)]
    if not lines:
        return "(the system detected no surgical steps in this interval)"
    lines.append("Instruments detected in this interval, each once, most confident detection first: "
                 + (render_instrument_summary(facts) or "none detected"))
    return "\n".join(lines)


def render_instrument_summary(facts) -> str:
    """Every instrument detected in any segment, once, ordered by its highest detection confidence.
    Names only: the note's confidences are the writer's own, not copies of detection confidences."""
    best = {}
    for seg in facts.segments:
        for n, c in seg.instruments.items():
            best[n] = max(best.get(n, 0.0), c)
    return ", ".join(n for n, _ in sorted(best.items(), key=lambda x: -x[1]))


def render_instrument_reliability(instr_names: list, reliability: dict) -> str:
    parts = [f"{n} {reliability[n]:.2f}" for n in instr_names if n in reliability]
    return ", ".join(parts) if parts else "unknown"


def build_messages(facts, reliability: dict, step_vocab: list, instr_vocab: list,
                   procedure: str, window: tuple | None = None, max_step_lines: int = 25,
                   keyframes: list | None = None) -> list:
    """Chat messages for one prompt. With keyframes, each one is an IMAGE_TAG placed under its
    segment; the matching image list is keyframes.image_paths(keyframes)."""
    scope = (f"This excerpt covers {fmt_ts(window[0])}-{fmt_ts(window[1])} of the procedure. "
             f"Document only what happens inside it.\n" if window else
             "This is the complete procedure.\n")
    user = f"""Procedure: {procedure}
{scope}
Automatic perception output, time-ordered (may contain errors):
{render_perception(facts, reliability, keyframes)}{chr(10) + KEYFRAME_USER if keyframes else ""}

Historical reliability of instrument detection: {render_instrument_reliability(instr_vocab, reliability)}

Allowed step names: {"; ".join(step_vocab)}
Allowed instrument names: {"; ".join(instr_vocab)}

Write the note in EXACTLY this format, sections in this order, nothing outside the tags:

<note>
<procedure>{procedure}</procedure>
<findings>{PLACEHOLDER}</findings>
<steps>
- Step name [HH:MM:SS-HH:MM:SS] (conf=0.00-1.00)
</steps>
<instruments>
- Instrument name (conf=0.00-1.00)
</instruments>
<complications>{PLACEHOLDER}</complications>
<closure>closure/reconstruction steps supported by the evidence, else {PLACEHOLDER}</closure>
</note>

Rules: one step or instrument per line; replace "Step name" and "Instrument name" with the allowed names verbatim; omit anything you
believe did not happen; end every step and instrument line with (conf=0.00-1.00), your own probability
that the line is correct. Each line holds only the name,
the time range for steps, and conf -- do not copy the detection confidence or historical reliability
into the note. Keep the note concise: list each step occurrence once, give one line spanning a step
that is only briefly interrupted or split into fragments, list each instrument once (the summary line of detected instruments has each one once), never repeat a
line, use at most {max_step_lines} step lines, and write nothing after </note>."""
    system = SYSTEM_PROMPT + (KEYFRAME_SYSTEM if keyframes else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
    return (f"<note>\n<procedure>{procedure}</procedure>\n<findings>{PLACEHOLDER}</findings>\n"
            f"<steps>\n{steps}\n</steps>\n<instruments>\n{instruments}\n</instruments>\n"
            f"<complications>{PLACEHOLDER}</complications>\n<closure>{closure}</closure>\n</note>")
