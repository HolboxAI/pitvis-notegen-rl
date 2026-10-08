import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from notegen_rl.llm import find_latest_checkpoint, list_checkpoints  # noqa: E402

_spec = importlib.util.spec_from_file_location("grpo_watchdog", ROOT / "scripts" / "grpo_watchdog.py")
WD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(WD)

LIMITS = {"window": 10, "min_step": 30, "max_kl": 0.5, "max_grad_norm": 10.0, "max_clipped_ratio": 0.1,
          "max_length_ratio": 1.5, "min_reward_ratio": 0.6}


def rows(n, **over):
    out = []
    for i in range(1, n + 1):
        r = {"step": i, "reward": 1.8, "kl": 0.05, "grad_norm": 0.7, "clipped": 0.0, "length": 380.0}
        for k, f in over.items():
            r[k] = f(i)
        out.append(r)
    return out


def test_stable_training_is_not_stopped():
    assert WD.check(rows(80), LIMITS) is None


def test_length_growth_trips_watchdog():
    reason = WD.check(rows(80, length=lambda i: 380.0 if i <= 60 else 700.0), LIMITS)
    assert reason and "length" in reason


def test_reward_drop_and_gradient_blowup_trip():
    assert "reward" in WD.check(rows(80, reward=lambda i: 1.8 if i <= 60 else 0.5), LIMITS)
    assert "gradient" in WD.check(rows(80, grad_norm=lambda i: 0.7 if i <= 60 else 500.0), LIMITS)


def test_no_check_before_min_step():
    assert WD.check(rows(25, grad_norm=lambda i: 1e9), LIMITS) is None


def _ckpt(root, name, full):
    d = root / "v0-run" / name
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    if full:
        (d / "optimizer.pt").write_text("x")
        (d / "trainer_state.json").write_text("{}")


def test_checkpoints_sorted_by_step_and_resumability(tmp_path):
    _ckpt(tmp_path, "checkpoint-100", full=False)
    _ckpt(tmp_path, "checkpoint-25", full=True)
    _ckpt(tmp_path, "checkpoint-50", full=True)
    assert [Path(c).name for c in list_checkpoints(tmp_path)] == ["checkpoint-25", "checkpoint-50", "checkpoint-100"]
    assert Path(find_latest_checkpoint(tmp_path, resumable=True)).name == "checkpoint-50"
    assert Path(find_latest_checkpoint(tmp_path)).name == "checkpoint-100"
