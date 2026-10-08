"""Stage 5: (optional) SFT format warm-start, then GRPO with the note rewards, via ms-swift.

  python scripts/05_train.py --stage sft    # only if sft.enabled
  python scripts/05_train.py --stage grpo
  add --dry-run to print the command without running it.

GRPO starts from (first match): the SFT adapter if sft.enabled, else llm.cold_start_adapter,
else the plain base model.
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notegen_rl.config import PROJECT_DIR, load_config, resolve, upload_results, wpath  # noqa: E402
from notegen_rl.llm import find_latest_checkpoint  # noqa: E402


def tuner_flag(sub: str) -> str:
    """ms-swift renamed --train_type to --tuner_type in newer releases; ask the installed one."""
    try:
        out = subprocess.run(["swift", sub, "--help"], capture_output=True, text=True, timeout=900)
        return "--tuner_type" if "--tuner_type" in (out.stdout + out.stderr) else "--train_type"
    except (OSError, subprocess.TimeoutExpired):
        return "--train_type"


def common(cfg, sub, rank, alpha):
    llm = cfg["llm"]
    cmd = ["swift", sub, "--model", llm["base"]]
    if llm.get("model_type"):
        cmd += ["--model_type", llm["model_type"]]
    cmd += [tuner_flag(sub), "lora", "--lora_rank", str(rank), "--lora_alpha", str(alpha),
            "--target_modules", "all-linear", "--torch_dtype", "bfloat16",
            "--gradient_checkpointing", "true", "--split_dataset_ratio", "0"]
    return cmd


def sft_cmd(cfg, dataset, out_dir):
    s = cfg["sft"]
    cmd = common(cfg, "sft", s["lora_rank"], s["lora_alpha"])
    if cfg["llm"].get("cold_start_adapter"):
        cmd += ["--adapters", str(resolve(cfg["llm"]["cold_start_adapter"]))]
    return cmd + [
        "--dataset", dataset,
        "--num_train_epochs", str(s["epochs"]), "--learning_rate", str(s["lr"]),
        "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "8",
        "--max_length", str(cfg["grpo"]["max_length"]),
        "--save_total_limit", "2", "--logging_steps", "5",
        "--output_dir", out_dir, "--report_to", cfg["grpo"]["report_to"],
    ]


def grpo_cmd(cfg, dataset, out_dir, start_adapter=None):
    g = cfg["grpo"]
    if len(g["reward_funcs"]) != len(g["reward_weights"]):
        raise SystemExit("grpo.reward_funcs and grpo.reward_weights must have the same length")
    cmd = common(cfg, "rlhf", g["lora_rank"], g["lora_alpha"])
    start = start_adapter
    if start is None and cfg["sft"]["enabled"]:
        start = find_latest_checkpoint(wpath(cfg, "sft"))
        if not start:
            raise SystemExit("sft.enabled but no SFT checkpoint found -- run --stage sft first")
    elif start is None and cfg["llm"].get("cold_start_adapter"):
        start = str(resolve(cfg["llm"]["cold_start_adapter"]))
    if start:
        print("GRPO starts from adapter:", start)
        cmd += ["--adapters", start]
    cmd += [
        "--rlhf_type", "grpo",
        "--dataset", dataset,
        "--external_plugins", str(PROJECT_DIR / "notegen_rl" / "swift_plugin.py"),
        "--reward_funcs", *g["reward_funcs"],
        "--reward_weights", *[str(w) for w in g["reward_weights"]],
        "--num_generations", str(g["num_generations"]),
        "--per_device_train_batch_size", str(g["per_device_train_batch_size"]),
        "--gradient_accumulation_steps", str(g["gradient_accumulation_steps"]),
        "--learning_rate", str(g["learning_rate"]),
        "--lr_scheduler_type", g.get("lr_scheduler_type", "cosine"), "--warmup_steps", str(g["warmup_steps"]),
        "--max_steps", str(g["max_steps"]), "--num_train_epochs", "100",
        "--temperature", str(g["temperature"]), "--beta", str(g["beta"]),
        "--max_grad_norm", str(g.get("max_grad_norm", 0.5)),
        "--max_length", str(g["max_length"]), "--max_completion_length", str(g["max_completion_length"]),
        "--save_steps", str(g["save_steps"]), "--save_only_model", "false",
        "--logging_steps", "1", "--log_completions", "true",
        "--output_dir", out_dir, "--report_to", g["report_to"],
    ]
    if g.get("loss_type"):
        cmd += ["--loss_type", g["loss_type"]]
    if g.get("save_total_limit"):  # null keeps every checkpoint
        cmd += ["--save_total_limit", str(g["save_total_limit"])]
    if g["use_vllm"]:
        cmd += ["--use_vllm", "true", "--vllm_mode", g["vllm_mode"],
                "--vllm_gpu_memory_utilization", str(g["vllm_gpu_memory_utilization"]),
                "--vllm_max_model_len", str(g["max_length"] + g["max_completion_length"]),
                "--sleep_level", "1"]
    return cmd + [str(a) for a in g.get("extra_args") or []]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--stage", choices=["sft", "grpo"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dataset", help="file under work_dir/datasets (default: sft_train / grpo_train .jsonl)")
    ap.add_argument("--output-subdir", help="work_dir subdir for checkpoints (default: the stage name)")
    ap.add_argument("--start-adapter", help="GRPO only: LoRA checkpoint to start from (overrides config)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the highest-step full checkpoint in the output dir, if any")
    ap.add_argument("--resume-from", help="continue from this specific checkpoint directory")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides, e.g. grpo.max_steps=40 sft.enabled=true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for kv in args.set:
        key, val = kv.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(val)
        print(f"override: {key} = {node[leaf]!r}")
    if not cfg["llm"].get("base"):
        raise SystemExit("config.yaml: set llm.base")
    if args.stage == "sft" and not cfg["sft"]["enabled"]:
        raise SystemExit("sft.enabled is false (set it in config.yaml or pass --set sft.enabled=true)")

    sub = args.output_subdir or args.stage
    dataset = args.dataset or f"{args.stage}_train.jsonl"
    dataset = dataset if Path(dataset).is_absolute() else str(wpath(cfg, "datasets", dataset))
    out_dir = str(wpath(cfg, sub))
    resume_from = args.resume_from or (find_latest_checkpoint(out_dir, resumable=True) if args.resume else None)
    if resume_from and not (Path(resume_from) / "optimizer.pt").exists():
        raise SystemExit(f"{resume_from} is not a full checkpoint (no optimizer.pt) -- cannot resume from it")
    cmd = (sft_cmd(cfg, dataset, out_dir) if args.stage == "sft"
           else grpo_cmd(cfg, dataset, out_dir, None if resume_from else args.start_adapter))
    if resume_from:
        # restores adapter, optimizer, scheduler and step counter; max_steps still applies to the total
        print("resuming from", resume_from)
        cmd += ["--resume_from_checkpoint", resume_from]
    env = dict(os.environ)
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # avoid nvcc-dependent JIT (see notegen_rl/llm.py)
    nproc = int(cfg["grpo"]["nproc_per_node"])
    if nproc > 1:
        env["NPROC_PER_NODE"] = str(nproc)
    print(("NPROC_PER_NODE=%d " % nproc if nproc > 1 else "") + shlex.join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True, env=env, cwd=PROJECT_DIR)
    upload_results(cfg, sub)


if __name__ == "__main__":
    main()
