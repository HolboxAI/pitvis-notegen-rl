# pitvis-notegen-rl — MedGemma text-only variant

Automated operative-note generation for endoscopic transsphenoidal pituitary surgery (PitVis-2023).
A frozen endoscopy video foundation model (Endo-FM) and a temporal recogniser (MS-TCN) turn the
surgical video into time-ordered steps and instruments. MedGemma 1.5 4B, trained with reinforcement
learning (GRPO), writes a structured note from these facts, and a medical language model turns the
structured note into a plain-language operative note ("The patient underwent ... The operation began
with ..."), checked by a deterministic verifier.

## Architecture

```
video (1 fps) ─► Endo-FM ─► MS-TCN ─► facts ─► MedGemma 1.5 4B + LoRA (GRPO) ─► merge ─► narrative writer ─► verifier ─► operative note
                768-d/s    step &     segments,   structured note              repeated    clinical prose    faithfulness
                           instrument  confidence,                              instrument                    gate
                           probs / s   reliability                              lines
```

| Component | Implementation | Output |
|---|---|---|
| Visual encoder | Endo-FM, ViT-B/16 TimeSformer (frozen) | 768-d feature per second |
| Temporal recogniser | MS-TCN, 2 stages × 10 dilated layers, step softmax + instrument sigmoid | Per-second step and instrument probabilities |
| Facts builder | Viterbi decoding, segmentation, per-class reliability from out-of-fold predictions | Time-ordered segments with confidence and detected instruments |
| Structured-note policy | MedGemma 1.5 4B + LoRA (rank 32), text-only prompts, GRPO with ms-swift and vLLM (SDPA attention) | Steps with times and confidences, instruments with confidences, closure |
| Post-processing | `rewards.read_bare_confidences`, `rewards.merge_repeated_instruments` | A bare number ending an item line read as its confidence; each instrument listed once, at its highest stated confidence |
| Final note writer | MedGemma (`final_notes.model`), reply starting "The patient underwent" | Narrative operative note; indication and postoperative course from optional case details, otherwise surgeon placeholders |
| Verifier | Deterministic checker (`notegen_rl/final_notes.py`) | Every step (by name or phrase, e.g. "the sella was opened") and instrument traced to the structured note; identified steps all mentioned; findings, complications and unobservable statements left to the surgeon |

**Prompt.** The note writer reads one line per detected segment, followed by one line listing every
detected instrument once with its highest detection confidence:

```
- [00:24:30-00:26:56] Sellotomy (confidence 0.34; historical reliability 0.52) | instruments: Kerrisons (0.83)
- [00:26:56-00:41:10] Tumour excision (confidence 0.71; historical reliability 0.75) | instruments: Pituitary rongeurs (0.88), Suction (0.80)
Instruments detected in this interval, each once (highest detection confidence): Pituitary rongeurs (0.88), Kerrisons (0.83), Suction (0.80)
```

The note lists each step occurrence and each instrument once; each step and instrument line carries
`(conf=…)`, the probability that the line is correct.

**Rewards** (`notegen_rl/rewards.py`), computed against the PitVis labels (maximum 2.8):

| Reward | Weight | Measures |
|---|---|---|
| Grounding | 1.0 | Step and instrument F1, reliability-weighted recall |
| Calibration | 0.5 | 1 − Brier score of the stated confidences |
| Timing | 0.3 | IoU of the stated step times with the labelled segments |
| Safety | 0.3 | Findings and complications left to the surgeon |
| Format | 0.2 | Sections in order, item lines with confidences |
| Conciseness | 0.5 | Length within budget × (1 − share of repeated step lines) × (1 − share of repeated instrument lines) |

**Training** (`config.yaml`, `grpo:`): up to 400 steps, 8 candidate notes per prompt, learning rate
5e-6 with 20 warm-up steps and cosine decay, KL coefficient 0.03, `dr_grpo` loss, gradient clipping 0.5,
1,500-token budget per note, repetition penalty 1.05. Full checkpoints every 25 steps (synced to S3,
training resumes from the newest one), a watchdog on KL, gradient norm, completion length, cut-off
notes and reward, and checkpoint selection every 50 steps on the validation videos (5, 12, 18).
Held-out evaluation on videos 21–25 scores the template, untrained MedGemma and the selected
checkpoint, each on its raw notes and post-processed (`<system>_merged`: bare confidences read, repeated
instrument lines merged).

**Narrative notes** (`scripts/09_final_notes.py`): a deterministic plan groups the structured note into
surgical phases; the writer turns it into prose; `[verify]` markers for uncertain steps and instruments
are placed from the plan; the verifier checks the prose against the plan, and prose that does not pass
is replaced by a rule-based narrative.

## Usage

```bash
bash setup_env.sh                                   # environment, Endo-FM checkpoint, unit tests
bash run_pipeline.sh                                # stages 0-6
python scripts/04_build_datasets.py                 # text-only prompts -> work_dir/<dataset.subdir>/
python scripts/05_train.py --stage grpo --resume    # GRPO (resumes from the newest full checkpoint)
python scripts/10_select_checkpoint.py --grpo-dir grpo_medgemma_v3
python scripts/06_evaluate.py                       # template, base and GRPO scored with the rewards (raw and post-processed)
python scripts/07_generate_note.py --video case.mp4 # note for a new video
python scripts/09_final_notes.py --completions <eval>/grpo/completions.jsonl [--case-details cases.json]
```

On EC2, `infra/job_grpo_v2.sh` runs training, checkpoint selection and held-out evaluation
(outputs `grpo_medgemma_v3/`, `eval_medgemma_v3/`), and `infra/job_smoke_medgemma.sh` runs a
40-step smoke test on the validation videos. All settings are in `config.yaml`, including generation
controls for MedGemma (`llm.response_prefix`, `llm.suppress_tokens`, `llm.repetition_penalty`,
`grpo.repetition_penalty`) and `eval.merge_repeated_instruments`. MedGemma models are gated on
Hugging Face (`HF_TOKEN`). `keyframes.enabled: true` adds keyframes from each detected segment to the
prompts (facts + keyframes variant).

## Layout

```
config.yaml     all settings
notegen_rl/     data, endofm, step_model, facts, keyframes, prompts, rewards, swift_plugin, llm, render, final_notes
scripts/        one script per stage (00-10) and the GRPO watchdog
infra/          EC2 user-data, job scripts, status helpers
tests/          unit tests (pytest)
```

## References

- Endo-FM: Wang et al., MICCAI 2023 — https://github.com/med-air/Endo-FM
- PitVis-2023: Das et al. — https://arxiv.org/abs/2409.01184
- MS-TCN: Farha and Gall, CVPR 2019
- GRPO: Shao et al., DeepSeekMath, 2024
- Dr. GRPO: Liu et al., Understanding R1-Zero-Like Training, 2025
- MedGemma 1.5 4B — https://huggingface.co/google/medgemma-1.5-4b-it
