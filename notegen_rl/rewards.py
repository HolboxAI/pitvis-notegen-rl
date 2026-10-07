"""Verifiable rewards for operative-note generation (pure Python, no swift/torch).

Every reward compares the note to PitVis *labels* for the same video/window, while the
prompt only showed the model *predicted* facts. So the policy is rewarded for writing a
correct note from imperfect perception -- including dropping perception errors and
being honest about uncertainty -- not for copying its input.

Components (all in [0, 1]; all but `format` are 0 for a malformed note):
  format       graded tag presence/order + fraction of item lines with a parseable conf
  grounding    mean of step-F1 and instrument-F1. Claims = every allowed name mentioned
               anywhere in <note>, so hallucinated names cost precision. Recall is
               reliability-weighted: missing a class the perception system detects well costs
               more than missing one it can barely see.
  calibration  1 - mean Brier score of per-line conf vs correctness (missing conf = worst case)
  temporal     mean IoU between each step line's [start-end] and that step's labeled time
  safety       findings and complications left as the surgeon placeholder (not invented)
"""
from __future__ import annotations

import json
import re

from .prompts import PLACEHOLDER, SECTIONS

ROW_KEYS = ("gold_steps", "gold_steps_any", "gold_instruments", "gold_instruments_any",
            "gold_segments", "step_vocab", "instrument_vocab", "reliability", "reliability_floor")
COMPONENTS = ("format", "grounding", "calibration", "temporal", "safety")

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_CONF = re.compile(r"conf(?:idence)?\s*[=:]\s*(\d*\.?\d+)", re.I)
_TS = r"(\d{1,2}:\d{2}(?::\d{2})?)"
_RANGE = re.compile(_TS + r"\s*[-–—]\s*" + _TS)
_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(.+)$")


def _norm(s) -> str:
    return " ".join(re.sub(r"[_\-]", " ", str(s).lower()).split())


def _load(v, default):
    if v is None:
        return default
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return default
    return v


def parse_ts(s: str) -> float:
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return float(h * 3600 + m * 60 + sec)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
class Item:
    def __init__(self, text: str):
        self.text = text
        m = _CONF.search(text)
        self.conf = min(1.0, max(0.0, float(m.group(1)))) if m else None
        r = _RANGE.search(text)
        self.start, self.end = (parse_ts(r.group(1)), parse_ts(r.group(2))) if r else (None, None)


class ParsedNote:
    def __init__(self, text: str):
        text = _THINK.sub("", text or "")
        mf = re.search(r"<facts_used>.*?</facts_used>", text, re.S | re.I)
        mn = re.search(r"<note>(.*?)</note>", text, re.S | re.I)
        self.has_facts_used, self.has_note = bool(mf), bool(mn)
        self.body = mn.group(1) if mn else ""
        self.sections, positions = {}, []
        for tag in SECTIONS:
            m = re.search(rf"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>", self.body, re.S | re.I)
            if m:
                self.sections[tag] = m.group(1)
                positions.append(m.start())
        self.order_ok = positions == sorted(positions) and (not (mf and mn) or mf.start() < mn.start())
        self.ok = self.has_facts_used and self.has_note and len(self.sections) == len(SECTIONS) and self.order_ok
        self.step_items = self._items("steps")
        self.instr_items = self._items("instruments")

    def _items(self, tag) -> list:
        out = []
        for line in self.sections.get(tag, "").splitlines():
            m = _ITEM.match(line)
            if m:
                out.append(Item(m.group(1)))
        return out


def match_terms(text: str, vocab) -> list:
    """Allowed names mentioned in text (longest first, each span used once)."""
    t = " " + _norm(text) + " "
    found = []
    for term in sorted(vocab, key=lambda x: -len(_norm(x))):
        nt = _norm(term)
        if not nt:
            continue
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(nt) + r"(?![a-z0-9])")
        if pat.search(t):
            found.append(term)
            t = pat.sub(lambda m: " " * len(m.group(0)), t)
    return found


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def format_score(p: ParsedNote) -> float:
    units = 9
    score = p.has_facts_used + p.has_note + len(p.sections)
    items = p.step_items + p.instr_items
    if items:
        score += sum(i.conf is not None for i in items) / len(items)
    score /= units
    return score if p.order_ok else score * 0.5


def _f1(pred: set, gold_recall: set, gold_any: set, rel: dict, floor: float):
    if not pred and not gold_recall:
        return 1.0, 1.0, 1.0
    prec = len(pred & gold_any) / len(pred) if pred else 0.0
    w = {g: max(floor, float(rel.get(g, floor))) for g in gold_recall}
    rec = sum(w[g] for g in gold_recall if g in pred) / sum(w.values()) if gold_recall else 1.0
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def grounding(p: ParsedNote, row: dict) -> dict:
    rel = _load(row.get("reliability"), {})
    floor = float(row.get("reliability_floor") or 0.2)
    step_vocab, instr_vocab = _load(row.get("step_vocab"), []), _load(row.get("instrument_vocab"), [])
    pred_steps = set(match_terms(p.body, step_vocab))
    pred_instr = set(match_terms(p.body, instr_vocab))
    gs, gsa = set(_load(row.get("gold_steps"), [])), set(_load(row.get("gold_steps_any"), []))
    gi, gia = set(_load(row.get("gold_instruments"), [])), set(_load(row.get("gold_instruments_any"), []))
    sp, sr, sf = _f1(pred_steps, gs, gsa | gs, rel, floor)
    out = {"step_precision": sp, "step_recall": sr, "step_f1": sf}
    parts = [sf]
    if instr_vocab:
        ip, ir, inf = _f1(pred_instr, gi, gia | gi, rel, floor)
        out.update(instr_precision=ip, instr_recall=ir, instr_f1=inf)
        parts.append(inf)
    out["grounding"] = sum(parts) / len(parts)
    return out


def calibration_pairs(p: ParsedNote, row: dict) -> list:
    """[(conf or None, correct)] for every item line naming exactly one allowed term."""
    pairs = []
    for items, vocab_key, gold_keys in ((p.step_items, "step_vocab", ("gold_steps", "gold_steps_any")),
                                        (p.instr_items, "instrument_vocab", ("gold_instruments", "gold_instruments_any"))):
        vocab = _load(row.get(vocab_key), [])
        gold = set().union(*(set(_load(row.get(k), [])) for k in gold_keys))
        for it in items:
            names = match_terms(it.text, vocab)
            if len(names) == 1:
                pairs.append((it.conf, float(names[0] in gold)))
    return pairs


def calibration(pairs: list) -> float:
    if not pairs:
        return 0.0
    brier = [1.0 if c is None else (c - y) ** 2 for c, y in pairs]
    return 1.0 - sum(brier) / len(brier)


def temporal(p: ParsedNote, row: dict) -> float:
    vocab = _load(row.get("step_vocab"), [])
    gold = {}
    for name, s, e in _load(row.get("gold_segments"), []):
        gold.setdefault(name, []).append((float(s), float(e)))
    scores = []
    for it in p.step_items:
        names = match_terms(it.text, vocab)
        if len(names) != 1:
            continue
        if it.start is None or it.end is None or it.end <= it.start:
            scores.append(0.0)
            continue
        ivs = gold.get(names[0], [])
        inter = sum(max(0.0, min(it.end, e) - max(it.start, s)) for s, e in ivs)
        union = (it.end - it.start) + sum(e - s for s, e in ivs) - inter
        scores.append(inter / union if union > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def safety(p: ParsedNote) -> float:
    return sum(0.5 for tag in ("findings", "complications")
               if PLACEHOLDER.lower() in p.sections.get(tag, "").lower())


def score_all(text: str, row: dict) -> dict:
    p = ParsedNote(text)
    out = {"format": format_score(p), "parsed_ok": float(p.ok)}
    if not p.ok:
        out.update(grounding=0.0, calibration=0.0, temporal=0.0, safety=0.0, conf_pairs=[])
        return out
    out.update(grounding(p, row))
    pairs = calibration_pairs(p, row)
    out.update(calibration=calibration(pairs), temporal=temporal(p, row), safety=safety(p),
               conf_pairs=pairs)
    return out


def expected_calibration_error(pairs: list, bins: int = 10) -> float | None:
    pairs = [(c, y) for c, y in pairs if c is not None]
    if not pairs:
        return None
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, y) for c, y in pairs if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if bucket:
            conf = sum(c for c, _ in bucket) / len(bucket)
            acc = sum(y for _, y in bucket) / len(bucket)
            ece += len(bucket) / len(pairs) * abs(conf - acc)
    return ece
