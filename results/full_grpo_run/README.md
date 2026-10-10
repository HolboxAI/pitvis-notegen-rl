# Full GRPO run (PitVis-2023)

Qwen3.5-9B + LoRA (rank 32) trained with GRPO for 400 steps on facts produced by frozen Endo-FM
features and the MS-TCN step / instrument model, then evaluated on the held-out videos 21–25.
Run with `infra/job_grpo_v2.sh` (output directory `grpo_v2`, evaluation `eval_v2`).

## Setup

| | |
|---|---|
| GRPO prompts | 17 videos, 258 prompts (full procedures + 15-minute windows) |
| Checkpoint selection | videos 5, 12, 18 (44 prompts); checkpoints every 25 steps, scored every 50 |
| Held-out test | videos 21–25 (85 prompts: 5 full procedures + 80 windows) |
| Rewards (weight) | grounding 1.0 · calibration 0.5 · timing 0.3 · safety 0.3 · format 0.2 · concise 0.5 (max 2.8) |
| Optimisation | 8 completions per prompt, lr 5e-6 cosine, KL β 0.03, grad clip 0.5, `dr_grpo` loss, 900-token budget |
| Training time | 5 h 36 min on one L40S (≈50 s / step) |

## Held-out results (85 prompts)

| Metric | Template | Base model | GRPO |
|---|---|---|---|
| Combined reward (max 2.8) | 2.244 | 2.238 | **2.364** |
| Combined reward without concise (max 2.3) | 1.806 | 1.818 | **1.869** |
| Grounding | 0.737 | 0.735 | **0.760** |
| Step F1 | **0.841** | 0.822 | 0.838 |
| Instrument F1 | 0.634 | 0.649 | **0.681** |
| Calibration (1 − Brier) | 0.888 | 0.878 | **0.898** |
| ECE (lower is better) | 0.171 | 0.205 | **0.157** |
| Timing (IoU) | 0.414 | 0.480 | **0.534** |
| Concise | 0.877 | 0.840 | **0.990** |
| Safety | 1.000 | 1.000 | 1.000 |
| Parsed / truncated | 100% / 0% | 100% / 0% | 100% / 0% |

Full procedures only (5 notes): combined reward 2.299 / 2.292 / **2.438**, timing 0.267 / 0.388 / **0.448**.

## Training

Validation score (videos 5, 12, 18) rose at every scored checkpoint: 2.193 (step 50) → 2.246 → 2.275 →
2.294 → 2.297 → 2.302 → 2.304 → **2.308 (step 400, selected)**. KL to the starting policy stayed below
0.025, mean gradient norm below 0.5, mean completion length 300–390 tokens, and the watchdog did not
intervene.

## Smoke-test pilot → full run (full-procedure notes, videos 21–25)

| Metric | Pilot (40 steps) | Full run (400 steps) |
|---|---|---|
| Grounding | 0.818 | **0.841** |
| Step F1 | 0.897 | **0.924** |
| Instrument F1 | 0.740 | **0.758** |
| Timing | 0.404 | **0.448** |
| Note length (characters) | 1,469 | **1,362** |

## Files

```
report.html                        results report (open in a browser)
notes.docx                         generated operative notes, videos 21–25
notes/video21.md … video25.md      generated notes with per-note scores
metrics/summary.md                 held-out summary for template, base and GRPO
metrics/{grpo,base,template}.json  all metrics (all prompts, full procedures, windows)
training/logging.jsonl             per-step training log
training/training_curve_25step.tsv 25-step means of reward components, length, KL and grad norm
training/best_checkpoint.json      validation scores per checkpoint and the selected step
```

LoRA checkpoints are kept in S3 under `runs/pitvis-notegen/grpo_v2/`.
