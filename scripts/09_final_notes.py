"""Stage 9: final, human-readable operative notes from structured notes (plan -> write -> verify).

  python scripts/09_final_notes.py --completions runs/pitvis-notegen/smoke/grpo/completions.jsonl \
      --dataset smoke_eval.jsonl --out-subdir final_notes/smoke_grpo
  add --template-only to skip the language model (rule-based prose only)

The writer model (final_notes.model in config.yaml, default MedGemma-27B-text-it) only rephrases the
structured note; notegen_rl/final_notes.verify() rejects any step, instrument or time it was not given.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notegen_rl import final_notes as F  # noqa: E402
from notegen_rl.config import load_config, resolve, upload_results, wpath  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--completions", required=True, help="completions.jsonl from an evaluation run")
    ap.add_argument("--dataset", default="grpo_eval.jsonl", help="matching prompts (for vocabularies)")
    ap.add_argument("--out-subdir", default="final_notes")
    ap.add_argument("--template-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fcfg = cfg["final_notes"]
    procedure = cfg["dataset"]["procedure_name"]

    rows = {(r["video_id"], r["window"]): r for r in map(json.loads, open(wpath(cfg, "datasets", args.dataset)))}
    comps = [json.loads(line) for line in open(resolve(args.completions))]
    comps = [c for c in comps if not c["window"]] if fcfg.get("full_procedures_only", True) else comps

    plans = []
    for c in comps:
        r = rows[(c["video_id"], c["window"])]
        plans.append(F.plan(c["completion"], json.loads(r["step_vocab"]), json.loads(r["instrument_vocab"])))

    prose = [F.template_prose(pl, procedure) for pl in plans]
    source = ["template"] * len(plans)
    if not args.template_only:
        from notegen_rl.llm import Generator
        gen_cfg = dict(cfg["llm"], max_new_tokens=fcfg["max_new_tokens"], quantization=fcfg.get("quantization"))
        gen = Generator(fcfg["model"], None, gen_cfg)
        todo = list(range(len(plans)))
        for attempt in range(1 + int(fcfg.get("retries", 1))):
            if not todo:
                break
            outs = gen.generate([F.messages(plans[i], procedure) for i in todo])
            still = []
            for i, text in zip(todo, outs):
                r = rows[(comps[i]["video_id"], comps[i]["window"])]
                problems = F.verify(text, plans[i], json.loads(r["step_vocab"]), json.loads(r["instrument_vocab"]))
                if problems:
                    print(f"video {comps[i]['video_id']} attempt {attempt + 1}: rejected -- {problems[:3]}")
                    still.append(i)
                else:
                    prose[i], source[i] = text.strip(), fcfg["model"]
            todo = still
        for i in todo:
            print(f"video {comps[i]['video_id']}: using rule-based prose (LLM output failed verification)")

    out = wpath(cfg, args.out_subdir, mkdir=True)
    with open(out / "final_notes.jsonl", "w") as f:
        for c, pl, text, src in zip(comps, plans, prose, source):
            f.write(json.dumps({"video_id": c["video_id"], "window": c["window"], "writer": src,
                                "plan": F.plan_text(pl), "final_note": text}) + "\n")
            (out / f"video{c['video_id']:02d}.md").write_text(
                f"# Operative note — DRAFT for surgeon review (video {c['video_id']:02d})\n\n{text}\n\n---\n"
                f"Written by: {src}. Generated automatically from the surgical video; every item must be "
                f"confirmed by the operating surgeon.\n")
    print(f"wrote {len(comps)} final notes to {out} ({source.count('template')} rule-based)")
    upload_results(cfg, args.out_subdir)


if __name__ == "__main__":
    main()
