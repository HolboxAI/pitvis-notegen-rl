"""ms-swift reward plugin. Loaded by path via `--external_plugins`, so it puts the project
root on sys.path itself. Each ORM returns one component of rewards.score_all; the
weights live in config.yaml (grpo.reward_weights), not here."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notegen_rl import rewards as R  # noqa: E402

try:
    from swift.plugin import ORM, orms
except ImportError:
    try:
        from swift.rewards import ORM, orms
    except ImportError:  # unit tests without ms-swift installed
        class ORM:
            def __init__(self, *args, **kwargs):
                pass

        orms = {}


def _text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return c.get("content", "")
    if isinstance(c, list) and c:
        return _text(c[-1])
    return str(c)


class _NoteORM(ORM):
    component = ""

    def __call__(self, completions, **kwargs):
        n = len(completions)
        rows = [{k: kwargs[k][i] for k in R.ROW_KEYS if isinstance(kwargs.get(k), (list, tuple))}
                for i in range(n)]
        return [float(R.score_all(_text(c), row)[self.component]) for c, row in zip(completions, rows)]


class NoteFormat(_NoteORM):
    component = "format"


class NoteGrounding(_NoteORM):
    component = "grounding"


class NoteCalibration(_NoteORM):
    component = "calibration"


class NoteTemporal(_NoteORM):
    component = "temporal"


class NoteSafety(_NoteORM):
    component = "safety"


orms["note_format"] = NoteFormat
orms["note_grounding"] = NoteGrounding
orms["note_calibration"] = NoteCalibration
orms["note_temporal"] = NoteTemporal
orms["note_safety"] = NoteSafety
