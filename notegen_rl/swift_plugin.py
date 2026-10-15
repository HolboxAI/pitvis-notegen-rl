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


def _gemma3_vision_classes() -> list:
    """Every Gemma3VisionTemplate class ms-swift exposes: known module paths, the template registry
    (TEMPLATE_MAPPING['gemma3_vision'].template_cls) and any already-imported module."""
    import importlib
    found = []
    for mod in ("swift.template.templates.gemma", "swift.llm.template.template.gemma"):
        try:
            found.append(getattr(importlib.import_module(mod), "Gemma3VisionTemplate", None))
        except Exception:  # noqa: BLE001 - layout differs between ms-swift versions
            pass
    for reg in ("swift.template", "swift.llm.template", "swift.llm"):
        try:
            meta = (getattr(importlib.import_module(reg), "TEMPLATE_MAPPING", None) or {}).get("gemma3_vision")
            found.append(getattr(meta, "template_cls", None))
        except Exception:  # noqa: BLE001
            pass
    for m in list(sys.modules.values()):
        found.append(getattr(m, "Gemma3VisionTemplate", None) if m is not None else None)
    out = []
    for c in found:
        if isinstance(c, type) and c not in out:
            out.append(c)
    return out


def _patch_gemma3_vision_for_vllm() -> list:
    """ms-swift's gemma3_vision template expands every <start_of_image> into the full image-token
    sequence in every mode. vLLM expands the marker itself, so in vLLM mode (GRPO rollouts) the
    prompt must keep the plain marker; training-mode encoding is unchanged."""
    patched = []
    for cls in _gemma3_vision_classes():
        if getattr(cls, "_notegen_vllm_patch", False):
            patched.append(cls)
            continue

        def make(cls, original):
            def _encode(self, inputs):
                if getattr(self, "mode", None) == "vllm":
                    encoded = super(cls, self)._encode(inputs)
                    encoded["token_type_ids"] = [0] * len(encoded["input_ids"])
                    return encoded
                return original(self, inputs)
            return _encode

        cls._encode = make(cls, cls._encode)
        cls._notegen_vllm_patch = True
        patched.append(cls)
    names = [f"{c.__module__}.{c.__name__}" for c in patched]
    print(f"[notegen] gemma3_vision vLLM image-marker patch {'applied to ' + ', '.join(names) if names else 'NOT applied (class not found)'}",
          flush=True)
    return patched


_patch_gemma3_vision_for_vllm()


BOI_TOKEN = "<start_of_image>"      # vLLM's image placeholder for Gemma 3 / MedGemma (get_placeholder_str)


def fix_image_markers(ids: list, n_images: int, boi_id: int, tag_variants: list) -> tuple:
    """Make the prompt carry exactly one <start_of_image> token per image, as vLLM's Gemma 3 processor
    expects. Literal "<image>" tags that reached the tokenizer unconverted (any of `tag_variants`, the
    token sequences "<image>" can take) are replaced by the marker. Returns (ids, status)."""
    if n_images == 0 or ids.count(boi_id) == n_images:
        return ids, "ok"
    out, i, found = [], 0, 0
    variants = sorted({tuple(v) for v in tag_variants if v}, key=len, reverse=True)
    while i < len(ids):
        for v in variants:
            if tuple(ids[i:i + len(v)]) == v:
                out.append(boi_id); i += len(v); found += 1
                break
        else:
            out.append(ids[i]); i += 1
    if out.count(boi_id) == n_images:
        return out, f"replaced {found} literal <image> tags"
    return ids, f"mismatch: {ids.count(boi_id)} markers, {found} literal tags, {n_images} images"


def _patch_vllm_requests():
    """Check every multimodal request ms-swift sends to vLLM and repair its image markers (see above)."""
    import importlib
    cls = None
    for mod in ("swift.infer_engine.vllm_engine", "swift.llm.infer.infer_engine.vllm_engine"):
        try:
            cls = getattr(importlib.import_module(mod), "VllmEngine", None)
            if cls is not None:
                break
        except Exception:  # noqa: BLE001
            continue
    if cls is None or getattr(cls, "_notegen_marker_patch", False):
        print(f"[notegen] vLLM request marker check {'already installed' if cls else 'NOT installed (VllmEngine not found)'}",
              flush=True)
        return
    original = cls._add_request
    state = {"logged": 0}

    def _add_request(self, inputs, *args, **kwargs):
        images = inputs.get("images") or []
        if images and isinstance(inputs.get("input_ids"), list):
            tok = getattr(self.template, "tokenizer", None) or getattr(self, "tokenizer", None)
            boi_id = tok.convert_tokens_to_ids(BOI_TOKEN)
            variants = [tok.encode(t, add_special_tokens=False) for t in ("<image>", " <image>", "\n<image>", ": <image>")]
            variants = [v[1:] if t.startswith(": ") else v for v, t in zip(variants, ("<image>", " <image>", "\n<image>", ": <image>"))]
            ids, status = fix_image_markers(inputs["input_ids"], len(images), boi_id, variants)
            if status.startswith("mismatch"):
                text = tok.decode(inputs["input_ids"])
                k = max(0, text.find("keyframe"))
                raise RuntimeError(f"[notegen] image markers {status}; prompt excerpt: {text[k:k + 400]!r}")
            if status != "ok" and state["logged"] < 3:
                print(f"[notegen] vLLM request: {status}", flush=True); state["logged"] += 1
            # vLLM's Gemma 3 processor merges the "\n\n" it inserts after an image with a following
            # newline token and then cannot find the image; drop such a token (vLLM adds the newlines itself)
            nl = {tok.convert_tokens_to_ids(t) for t in ("\n", "\n\n", "\n\n\n", "\n\n\n\n")}
            cleaned = [t for j, t in enumerate(ids) if not (t in nl and j > 0 and ids[j - 1] == boi_id)]
            if len(cleaned) != len(ids) and state["logged"] < 3:
                print(f"[notegen] vLLM request: removed {len(ids) - len(cleaned)} newline tokens after image markers",
                      flush=True); state["logged"] += 1
            inputs["input_ids"] = cleaned
        return original(self, inputs, *args, **kwargs)

    cls._add_request = _add_request
    cls._notegen_marker_patch = True
    print(f"[notegen] vLLM request marker check installed on {cls.__module__}.{cls.__name__}", flush=True)


_patch_vllm_requests()


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


class NoteConcise(_NoteORM):
    component = "concise"


orms["note_format"] = NoteFormat
orms["note_grounding"] = NoteGrounding
orms["note_calibration"] = NoteCalibration
orms["note_temporal"] = NoteTemporal
orms["note_safety"] = NoteSafety
orms["note_concise"] = NoteConcise
