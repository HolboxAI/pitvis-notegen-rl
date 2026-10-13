import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_gemma3_vision_keeps_image_marker_in_vllm_mode(monkeypatch):
    class Gemma3Template:
        def _encode(self, inputs):
            return {"input_ids": [1, 255999, 2]}           # plain <start_of_image> marker

    class Gemma3VisionTemplate(Gemma3Template):
        def _encode(self, inputs):
            return {"input_ids": [1] + [262144] * 256 + [2]}   # expanded image tokens

    mod = types.ModuleType("swift.template.templates.gemma")
    mod.Gemma3VisionTemplate = Gemma3VisionTemplate
    for name in ("swift", "swift.template", "swift.template.templates"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "swift.template.templates.gemma", mod)

    from notegen_rl import swift_plugin
    swift_plugin._patch_gemma3_vision_for_vllm()
    t = Gemma3VisionTemplate()
    t.mode = "vllm"
    assert t._encode(None)["input_ids"] == [1, 255999, 2]
    t.mode = "train"
    assert len(t._encode(None)["input_ids"]) == 258


def test_fix_image_markers_with_medgemma_tokenizer():
    import pytest
    tokenizers = pytest.importorskip("tokenizers")
    path = Path.home() / "medgemma-1.5-4b-it" / "tokenizer.json"
    if not path.exists():
        pytest.skip("MedGemma tokenizer not available locally")
    from notegen_rl.swift_plugin import fix_image_markers
    tok = tokenizers.Tokenizer.from_file(str(path))
    boi = tok.token_to_id("<start_of_image>")
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    variants = [enc("<image>"), enc(" <image>"), enc("\n<image>"), enc(": <image>")[1:]]
    literal = enc("- [00:15:00-00:17:41] Sellotomy\n  keyframe 00:16:46: <image>\n  keyframe 00:17:02: <image>\n- next")
    fixed, status = fix_image_markers(literal, 2, boi, variants)
    assert fixed.count(boi) == 2 and "replaced 2" in status and "<image>" not in tok.decode(fixed)
    good = enc("keyframe 00:16:46: <start_of_image>\nkeyframe 00:17:02: <start_of_image>\n")
    assert fix_image_markers(good, 2, boi, variants) == (good, "ok")
    assert fix_image_markers(literal, 3, boi, variants)[1].startswith("mismatch")
