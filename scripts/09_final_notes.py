"""Stage 9: final operative notes in plain narrative language from structured notes (plan -> write -> verify).

  python scripts/09_final_notes.py --completions runs/pitvis-notegen/smoke/grpo/completions.jsonl \
      --dataset smoke_eval.jsonl --out-subdir final_notes/smoke_grpo
  add --template-only to skip the language model (rule-based prose only)
  add --case-details cases.json for facts the video cannot show, per video:
      {"25": {"indication": "a non-functioning pituitary macroadenoma", "postoperative": "..."}}

The writer model (final_notes.model in config.yaml, default MedGemma-27B-text-it) only rephrases the
structured note; notegen_rl/final_notes.verify() rejects any step or instrument it was not given, omitted
steps, lost [verify] markers and statements the video cannot support.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notegen_rl import final_notes as F  # noqa: E402
from notegen_rl.config import dataset_path, load_config, resolve, upload_results, wpath  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--completions", required=True, help="completions.jsonl from an evaluation run")
    ap.add_argument("--dataset", default="grpo_eval.jsonl", help="matching prompts (for vocabularies)")
    ap.add_argument("--out-subdir", default="final_notes")
    ap.add_argument("--template-only", action="store_true")
    ap.add_argument("--case-details", help="JSON {video_id: {indication, postoperative}} (optional)")
    ap.add_argument("--writer-model", help="writer model id or path (default: final_notes.model)")
    ap.add_argument("--writer-quantization", help="vLLM quantization for the writer, 'none' for full precision "
                                                  "(default: final_notes.quantization)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fcfg = cfg["final_notes"]
    if args.writer_model:
        fcfg["model"] = args.writer_model
    if args.writer_quantization:
        fcfg["quantization"] = None if args.writer_quantization == "none" else args.writer_quantization
    procedure = cfg["dataset"]["procedure_name"]

    rows = {(r["video_id"], r["window"]): r for r in map(json.loads, open(dataset_path(cfg, args.dataset)))}
    comps = [json.loads(line) for line in open(resolve(args.completions))]
    comps = [c for c in comps if not c["window"]] if fcfg.get("full_procedures_only", True) else comps
    cases = json.loads(resolve(args.case_details).read_text()) if args.case_details else {}
    case_of = [cases.get(str(c["video_id"])) or {} for c in comps]

    plans = []
    for c in comps:
        r = rows[(c["video_id"], c["window"])]
        plans.append(F.plan(c["completion"], json.loads(r["step_vocab"]), json.loads(r["instrument_vocab"])))

    # a structured note that could not be parsed has no events to narrate: no final note is written
    no_note = "No final note: the structured note was not in the required format."
    prose = [F.template_prose(pl, procedure, cs) if pl["parsed"] else no_note for pl, cs in zip(plans, case_of)]
    source = ["template" if pl["parsed"] else "none" for pl in plans]
    attempts = [0] * len(plans)
    rejections = [[] for _ in plans]
    if not args.template_only:
        from notegen_rl.llm import Generator
        # the writer's reply starts with the opening words of the note, so it writes the note itself
        start = fcfg.get("answer_start") or ""
        gen_cfg = dict(cfg["llm"], max_new_tokens=fcfg["max_new_tokens"], quantization=fcfg.get("quantization"),
                       stop=[], response_prefix=(cfg["llm"].get("response_prefix") or "") + start or None)
        gen = Generator(fcfg["model"], None, gen_cfg)
        todo = [i for i, pl in enumerate(plans) if pl["parsed"]]
        for attempt in range(1 + int(fcfg.get("retries", 1))):
            if not todo:
                break
            outs = [start + o for o in gen.generate([F.messages(plans[i], procedure, case_of[i]) for i in todo])]
            still = []
            for i, text in zip(todo, outs):
                r = rows[(comps[i]["video_id"], comps[i]["window"])]
                problems = F.verify(text, plans[i], json.loads(r["step_vocab"]), json.loads(r["instrument_vocab"]),
                                    case_of[i])
                attempts[i] += 1
                if problems:
                    rejections[i].append({"problems": problems, "output": text})
                    print(f"video {comps[i]['video_id']} attempt {attempt + 1}: rejected -- {problems[:3]}")
                    still.append(i)
                else:
                    prose[i], source[i] = F.clean(text), fcfg["model"]
            todo = still
        for i in todo:
            print(f"video {comps[i]['video_id']}: using rule-based prose (LLM output failed verification)")

    out = wpath(cfg, args.out_subdir, mkdir=True)
    with open(out / "final_notes.jsonl", "w") as f:
        for c, pl, text, src, n, rej in zip(comps, plans, prose, source, attempts, rejections):
            f.write(json.dumps({"video_id": c["video_id"], "window": c["window"], "writer": src,
                                "structured_note_parsed": pl["parsed"], "attempts": n, "rejections": rej,
                                "plan": F.plan_text(pl), "final_note": text}) + "\n")
            (out / f"video{c['video_id']:02d}.md").write_text(
                f"# Operative note — DRAFT for surgeon review (video {c['video_id']:02d})\n\n{text}\n\n---\n"
                f"Written by: {src}. Generated automatically from the surgical video; every item must be "
                f"confirmed by the operating surgeon.\n")
    print(f"wrote {len(comps)} final notes to {out} ({source.count('template')} rule-based, "
          f"{source.count('none')} without a parseable structured note)")
    upload_results(cfg, args.out_subdir)


if __name__ == "__main__":
    main()
