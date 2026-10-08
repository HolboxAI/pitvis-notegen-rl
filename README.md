# pitvis-notegen-rl

Automated operative-note generation for endoscopic transsphenoidal pituitary surgery (PitVis-2023).
A frozen endoscopy video foundation model (Endo-FM) and a temporal recogniser turn the surgical
video into time-ordered steps and instruments; a language model trained with reinforcement learning
(GRPO) writes a structured note from them; a final writing layer turns the structured note into a
clinical operative note, checked by a deterministic faithfulness verifier.

**Branch `medgemma`: MedGemma variant.** The language side uses MedGemma throughout: MedGemma-4B
(Gemma 3, medical) is the GRPO-trained structured-note policy, and MedGemma-27B-text-it writes the
final note. Perception (Endo-FM, MS-TCN, facts), datasets, rewards, watchdog, checkpoint selection
and evaluation are shared with `main` (Qwen3.5-9B policy), so the two variants are directly
comparable on the same held-out videos.

| | `main` | `medgemma` |
|---|---|---|
| Structured-note policy | Qwen3.5-9B + LoRA (GRPO) | MedGemma-4B + LoRA (GRPO), vision tower frozen |
| Final note writer | MedGemma-27B-text-it | MedGemma-27B-text-it |
| GRPO output / evaluation | `grpo_v2/`, `eval_v2/` | `grpo_medgemma/`, `eval_medgemma/` |
| Code copy in S3 | `operative-notes/pitvis-notegen-rl/` | `operative-notes/pitvis-notegen-rl-medgemma/` |

MedGemma models are gated on Hugging Face; jobs read the access token from
`control/hf_token.txt` in S3 and export it as `HF_TOKEN`.

## Pipeline

```
video (1 fps) ─► Endo-FM ─► MS-TCN ─► facts ─► MedGemma-4B + LoRA (GRPO) ─► plan ─► MedGemma-27B ─► verifier ─► operative note
                768-d/s    step &     segments,     structured note          phases   clinical prose   faithfulness
                           instrument  confidence,   (steps, times,                                     gate
                           probs / s   reliability   instruments)
```

| Stage | Component | Role |
|---|---|---|
| Visual encoder | Endo-FM, ViT-B/16 TimeSformer (frozen) | One 768-d feature per second of video |
| Temporal recogniser | MS-TCN: 2 stages × 10 dilated residual layers, joint step softmax (15) + instrument sigmoid (18) | Per-second step and instrument probabilities |
| Facts builder | Viterbi decoding with label-estimated step transitions, segmentation | Time-ordered segments with confidence and per-class reliability |
| Structured note policy | MedGemma-4B with LoRA (rank 32, language layers; vision tower frozen), trained with GRPO (ms-swift, vLLM generation) | Structured note: steps with times and confidences, instruments, closure |
| Final note writer | MedGemma-27B-text-it (FP8) | Clinical prose from a phase plan, one worked example in the prompt |
| Verifier | Deterministic checker | Every step, instrument and time in the prose must come from the structured note |

Training prompts are built from predicted facts (out-of-fold for training videos), while rewards are
computed against the PitVis labels, so the policy learns to write a correct note from perception output.

## Rewards (`notegen_rl/rewards.py`)

| ms-swift name | Weight | Measures |
|---|---|---|
| `note_grounding` | 1.0 | Mean of step F1 and instrument F1 against the labels; every allowed name in the note is a claim; recall weighted by per-class perception reliability |
| `note_calibration` | 0.5 | 1 − Brier score of each line's stated confidence against its correctness |
| `note_temporal` | 0.3 | IoU between each step's stated time range and its labelled time |
| `note_safety` | 0.3 | Findings and complications left for the surgeon (`[SURGEON TO COMPLETE]`) |
| `note_format` | 0.2 | Required sections present in order with parseable confidences |
| `note_concise` | 0.5 | Length and repetition control: full credit up to a soft token budget, falling to 0 at a hard budget; scaled down for consecutive repeated step lines and for text after `</note>` |

A truncated note (opened but not closed) keeps partial credit on the lines it completed
(`dataset.truncation_credit`). Generation stops at `</note>` during evaluation and inference.

GRPO settings (`config.yaml → grpo`): learning rate 5e-6 with cosine decay, KL coefficient 0.03,
gradient clipping 0.5, length-unbiased `dr_grpo` loss, 8 notes per prompt, 400 steps, full
checkpoints every 25 steps with all checkpoints kept.

## Stages

| Stage | Script | Output (under `work_dir`) |
|---|---|---|
| 0 | `scripts/00_prepare.py` — sync PitVis, validate layout, build taxonomy and split | `taxonomy.json`, `split.json` |
| 1 | `scripts/01_extract_features.py` — Endo-FM features per labelled second (pre-extracted frames or video decoding; resumable, multi-GPU) | `features/` |
| 2 | `scripts/02_train_step_model.py` — MS-TCN with 5-fold out-of-fold predictions and a final model | `predictions/`, `step_model/` |
| 3 | `scripts/03_build_facts.py` — smoothing, segmentation, per-class reliability, perception metrics | `facts/` |
| 4 | `scripts/04_build_datasets.py` — GRPO / SFT / evaluation prompts (full procedures + 15-min windows) | `datasets/` |
| 5 | `scripts/05_train.py --stage [sft\|grpo]` — LoRA training with ms-swift (`--resume`, `--set key=value` overrides) | `sft/`, `grpo/` |
| 6 | `scripts/06_evaluate.py` — template, base and GRPO systems scored with the reward functions; Markdown note per procedure | `eval/` |
| 7 | `scripts/07_generate_note.py --video X.mp4` — structured note for a new video | `notes/` |
| 8 | `scripts/08_smoke_data.py` — 10-procedure smoke-test datasets | `datasets/smoke_*.jsonl` |
| 9 | `scripts/09_final_notes.py` — plan → MedGemma → verify → final operative notes | `final_notes/` |
| 10 | `scripts/10_select_checkpoint.py` — evaluate GRPO checkpoints on the validation videos and record the best | `<grpo-dir>/best_checkpoint.json` |
| – | `scripts/grpo_watchdog.py` — follows the training log and stops training when KL, gradient norm, clipped completions, completion length or reward leave their configured range; records the last healthy checkpoint | `<grpo-dir>/.watchdog_stop` |

Videos 1–20 train the step model; within them, `data.val_video_ids` (5, 12, 18) are kept out of GRPO
and used for checkpoint selection; videos 21–25 are the held-out test set.

## Results

Step recognition with frozen Endo-FM features (held-out videos 21–25):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Per-frame linear probe | 0.535 | 0.366 |
| MS-TCN (this pipeline) | 0.721 | 0.537 |

Smoke test (10 full procedures, 40-step GRPO pilot): all generated notes parse into the required
format, training reward rose from 1.62 to 1.78, and the calibration error of the stated confidences
fell from 0.194 to 0.168.

## Setup and running

```bash
bash setup_env.sh                       # venv, requirements, Endo-FM clone + checkpoint, unit tests
bash run_pipeline.sh                    # stages 0-6 (STAGES="3 4" / NUM_GPUS=4 / CONFIG=... supported)
python scripts/05_train.py --stage grpo --dry-run      # print the ms-swift command
python scripts/09_final_notes.py --completions runs/pitvis-notegen/eval/grpo/completions.jsonl
python scripts/07_generate_note.py --video /path/to/case.mp4
```

All settings live in `config.yaml` (dataset location, Endo-FM, MS-TCN, facts, datasets, LLMs, SFT, GRPO,
final notes). MedGemma is a gated Hugging Face model; set `HF_TOKEN` for stage 9.

### Running on EC2 through S3 (`infra/`)

`infra/user_data.sh` installs a per-boot runner: on every boot the instance downloads
`control/job.sh` from S3, runs it, streams its log to `control/logs/`, writes `control/status.txt`
and shuts down (instance-initiated shutdown = stop, so the disk persists between jobs).
Job scripts: `job_prepare.sh`, `job_rebuild.sh`, `job_smoke.sh`, `job_smoke_grpo.sh`,
`job_format_check.sh`, `job_full_grpo.sh`, `job_grpo_v2.sh` and `job_final_notes.sh`.
`infra/status.sh` prints job status, current stage and GRPO progress; `infra/watch.sh` waits for
the next milestone.

`job_grpo_v2.sh` runs GRPO with the watchdog, selects the checkpoint on the validation videos and
evaluates template, base and the selected checkpoint on the held-out videos. Training resumes after
any interruption: the output directory is synced to S3 every 5 minutes and restored at start, and
training continues from the highest-step full checkpoint (or from the checkpoint named in
`<results_uri>/<grpo-dir>/resume_from.txt`). Manual resume on any machine:

```bash
python scripts/05_train.py --stage grpo --output-subdir grpo_v2 --resume                  # newest full checkpoint
python scripts/05_train.py --stage grpo --output-subdir grpo_v2 --resume-from <checkpoint-dir>
```

## Layout

```
config.yaml          all settings
setup_env.sh         environment, Endo-FM clone and checkpoint, unit tests
run_pipeline.sh      stages 0-6
notegen_rl/          data, endofm, step_model, facts, prompts, rewards, swift_plugin, llm, render, final_notes
scripts/00-10        one script per stage, plus grpo_watchdog.py
infra/               EC2 user-data, S3-driven job scripts, status and watch helpers
tests/               reward, facts, final-note, watchdog and checkpoint unit tests (pytest)
```

## References

- Endo-FM: Wang et al., MICCAI 2023 — https://github.com/med-air/Endo-FM
- PitVis-2023: Das et al. — https://arxiv.org/abs/2409.01184
- Operation notes from workflow recognition: Das et al. — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10958393/
- MS-TCN: Farha and Gall, CVPR 2019
- GRPO: Shao et al., DeepSeekMath, 2024
- Clinical summarisation with adapted LLMs: Van Veen et al., Nature Medicine 2024
- MedGemma-27B-text-it — https://huggingface.co/google/medgemma-27b-text-it
