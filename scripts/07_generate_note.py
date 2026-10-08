"""Stage 7: draft an operative note for a NEW, unlabeled pituitary video.

  python scripts/07_generate_note.py --video /path/to/case.mp4 [--adapter DIR] [--template-only]

Writes <work_dir>/notes/<video stem>.{md,json}. Lines below eval.hedge_threshold confidence
are marked [verify]; findings/complications are left for the surgeon. Always a draft.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from notegen_rl.config import load_config, wpath  # noqa: E402
from notegen_rl.data import Taxonomy  # noqa: E402
from notegen_rl.endofm import extract, load_backbone, video_duration_s  # noqa: E402
from notegen_rl.facts import facts_from_predictions  # noqa: E402
from notegen_rl.llm import Generator, find_latest_checkpoint  # noqa: E402
from notegen_rl.prompts import build_messages, render_template_note  # noqa: E402
from notegen_rl.render import to_markdown  # noqa: E402
from notegen_rl.step_model import predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--video", required=True)
    ap.add_argument("--adapter", help="LoRA checkpoint; default = newest under work_dir/grpo")
    ap.add_argument("--template-only", action="store_true", help="skip the LLM (perception template note)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    video = Path(args.video)

    tax = Taxonomy.load(wpath(cfg, "taxonomy.json"))
    bundle = torch.load(wpath(cfg, "step_model", "final.pt"), map_location="cpu", weights_only=False)
    reliability = json.loads(wpath(cfg, "facts", "reliability.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    duration = video_duration_s(video)
    if duration <= 0:
        raise SystemExit(f"could not read duration of {video}")
    times = np.arange(int(duration))
    got = {}
    backbone = load_backbone(cfg, device)
    extract(backbone, [(0, video, times)], cfg, device, lambda v, t, f: got.update(times=t, feats=f))
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sp, ip = predict(bundle, got["feats"], device)
    facts, _ = facts_from_predictions(video.stem, got["times"], sp, ip, tax, cfg["facts"],
                                      bundle["log_trans"], bundle["log_init"])
    procedure = cfg["dataset"]["procedure_name"]
    if args.template_only:
        completion, adapter = render_template_note(facts, procedure), None
    else:
        adapter = args.adapter or find_latest_checkpoint(wpath(cfg, "grpo"))
        if not adapter:
            print("WARNING: no GRPO checkpoint found -- using the untrained base model")
        msgs = build_messages(facts, reliability, tax.surgical_steps, tax.instr_names, procedure,
                              max_step_lines=cfg["dataset"].get("max_step_lines", 25))
        completion = Generator(cfg["llm"]["base"], adapter, cfg["llm"]).generate([msgs])[0]

    out_dir = wpath(cfg, "notes", mkdir=True)
    md = to_markdown(completion, cfg["eval"]["hedge_threshold"], video.name)
    (out_dir / f"{video.stem}.md").write_text(md)
    (out_dir / f"{video.stem}.json").write_text(json.dumps({
        "video": str(video), "adapter": adapter, "duration_s": duration,
        "facts": [s.__dict__ for s in facts.segments], "completion": completion}, indent=2))
    print(md)
    print(f"\nsaved: {out_dir / (video.stem + '.md')}")


if __name__ == "__main__":
    main()
