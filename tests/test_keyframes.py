import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from notegen_rl.data import Taxonomy  # noqa: E402
from notegen_rl.facts import Segment, VideoFacts  # noqa: E402
from notegen_rl.keyframes import (FrameDir, Keyframe, cap_keyframes, image_paths,  # noqa: E402
                                  select_keyframes, store_keyframes)
from notegen_rl.llm import to_conversation  # noqa: E402
from notegen_rl.prompts import IMAGE_TAG, build_messages  # noqa: E402

TAX = Taxonomy(step_codes=[-1, 1, 2], step_names=["Out of patient", "Nasal corridor creation", "Tumour excision"],
               instr_codes=[1], instr_names=["Suction"], non_surgical=["Out of patient"])
KCFG = {"frames_per_segment": 1, "long_segment_s": 300, "frames_per_long_segment": 2,
        "max_images_per_prompt": 16, "candidates": 8, "out_of_patient_max": 0.5,
        "min_brightness": 25, "min_relative_sharpness": 0.5}


def _write(d: Path, t: int, kind: str):
    rng = np.random.default_rng(t)
    img = rng.integers(60, 200, size=(120, 160)).astype(np.uint8)    # textured = sharp
    if kind == "blur":
        img = cv2.GaussianBlur(img, (31, 31), 10)
    elif kind == "dark":
        img = (img // 20).astype(np.uint8)
    cv2.imwrite(str(d / f"{t}.jpg"), img)


def _probs(T, cls, conf):
    p = np.full((T, 3), (1 - conf) / 2)
    p[:, cls] = conf
    return p


def test_picks_most_confident_frame_that_is_bright_and_sharp(tmp_path):
    T = 60
    times = np.arange(T)
    probs = _probs(T, 1, 0.6)
    probs[10, 1], probs[11, 1], probs[12, 1], probs[13, 1] = 0.99, 0.98, 0.97, 0.96
    for t in times:
        _write(tmp_path, t, "sharp")
    _write(tmp_path, 10, "dark")       # most confident, but dark
    _write(tmp_path, 11, "blur")       # second, but blurred
    facts = VideoFacts("1", "predicted", [Segment("Nasal corridor creation", 0, 60, 0.7)])
    kfs = select_keyframes(facts, times, probs, TAX, FrameDir(tmp_path), KCFG)
    assert [(k.segment, k.t) for k in kfs] == [(0, 12)]


def test_skips_out_of_patient_and_splits_long_segments(tmp_path):
    T = 600
    times = np.arange(T)
    probs = _probs(T, 2, 0.8)
    probs[:50] = [0.9, 0.05, 0.05]     # out of patient at the start of the segment
    probs[:50, 2] = 0.95               # ...even though the step looks confident there
    for t in times:
        _write(tmp_path, t, "sharp")
    facts = VideoFacts("1", "predicted", [Segment("Tumour excision", 0, 600, 0.8)])
    kfs = select_keyframes(facts, times, probs, TAX, FrameDir(tmp_path), KCFG)
    assert len(kfs) == 2 and all(k.t >= 50 for k in kfs)
    assert kfs[0].t < 325 <= kfs[1].t                    # one frame from each half
    assert kfs[0].primary and not kfs[1].primary


def test_cap_keeps_one_frame_per_segment_and_least_confident_segments():
    facts = VideoFacts("1", "predicted", [Segment("A", 0, 10, 0.9), Segment("B", 10, 20, 0.3),
                                          Segment("C", 20, 400, 0.5)])
    kfs = [Keyframe(0, 5, "a"), Keyframe(1, 15, "b"), Keyframe(2, 100, "c1"), Keyframe(2, 300, "c2", primary=False)]
    assert [k.path for k in cap_keyframes(kfs, facts, 3)] == ["a", "b", "c1"]
    assert [k.path for k in cap_keyframes(kfs, facts, 2)] == ["b", "c1"]
    assert cap_keyframes(kfs, facts, 0) == []


def test_prompt_images_line_up_with_tags(tmp_path):
    for t in (5, 15, 30):
        _write(tmp_path, t, "sharp")
    facts = VideoFacts("1", "predicted", [Segment("Nasal corridor creation", 0, 10, 0.9),
                                          Segment("Tumour excision", 10, 40, 0.6)])
    kfs = store_keyframes([Keyframe(1, 30, str(tmp_path / "30.jpg"), primary=False),
                           Keyframe(0, 5, str(tmp_path / "5.jpg")),
                           Keyframe(1, 15, str(tmp_path / "15.jpg"))], tmp_path / "kf")
    msgs = build_messages(facts, {}, TAX.surgical_steps, TAX.instr_names, "proc", keyframes=kfs)
    user = msgs[1]["content"]
    assert user.count(IMAGE_TAG) == 3
    assert user.index("keyframe 00:00:05") < user.index("Tumour excision") < user.index("keyframe 00:00:15") \
        < user.index("keyframe 00:00:30")
    images = image_paths(kfs)
    assert [Path(p).name for p in images] == ["5.jpg", "15.jpg", "30.jpg"]
    text = build_messages(facts, {}, TAX.surgical_steps, TAX.instr_names, "proc")
    row = {"messages": msgs, "images": images, "messages_text": json.dumps(text)}
    conv = to_conversation(row)
    parts = conv[1]["content"]
    assert [p["path"] for p in parts if p["type"] == "image"] == images
    assert conv[0] == msgs[0]
    plain = to_conversation(row, text_only=True)
    assert IMAGE_TAG not in plain[1]["content"] and "Keyframes" not in plain[0]["content"]


def test_no_newline_directly_after_an_image_tag():
    facts = VideoFacts("1", "predicted", [Segment("Tumour excision", 10, 40, 0.6)])
    kfs = [Keyframe(0, 15, "a.jpg"), Keyframe(0, 30, "b.jpg", primary=False)]
    user = build_messages(facts, {}, TAX.surgical_steps, TAX.instr_names, "proc", keyframes=kfs)[1]["content"]
    assert user.count(IMAGE_TAG) == 2 and IMAGE_TAG + "\n" not in user
    assert "(frame 2 of this segment)" in user


def test_perception_lists_each_detected_instrument_once():
    facts = VideoFacts("1", "predicted", [
        Segment("Nasal corridor creation", 0, 10, 0.9, {"Suction": 0.6, "Kerrisons": 0.8}),
        Segment("Tumour excision", 10, 40, 0.6, {"Suction": 0.9})])
    user = build_messages(facts, {}, TAX.surgical_steps, TAX.instr_names, "proc")[1]["content"]
    summary = [ln for ln in user.splitlines() if ln.startswith("Instruments detected in this interval")]
    assert summary == ["Instruments detected in this interval, each once, most confident detection first: "
                       "Suction, Kerrisons"]
