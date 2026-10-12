# pitvis-notegen-rl — MedGemma variant

Automated operative-note generation for endoscopic transsphenoidal pituitary surgery (PitVis-2023).
A frozen endoscopy video foundation model (Endo-FM) and a temporal recogniser (MS-TCN) turn the
surgical video into time-ordered steps and instruments. MedGemma 1.5 4B, trained with reinforcement
learning (GRPO), writes a structured note from these facts together with keyframes taken from each
detected step, and MedGemma-27B-text-it turns the structured note into clinical prose, checked by a
deterministic verifier.

## Architecture

```
video (1 fps) ─► Endo-FM ─► MS-TCN ─► facts ──────────────┐
                768-d/s    step &     segments,             ├─► MedGemma 1.5 4B + LoRA (GRPO) ─► MedGemma-27B ─► verifier ─► operative note
                           instrument  confidence,          │   structured note                  clinical prose   faithfulness
                           probs / s   reliability          │                                                     gate
       frames ─► keyframe selection (per predicted segment) ┘
```

| Component | Implementation | Output |
|---|---|---|
| Visual encoder | Endo-FM, ViT-B/16 TimeSformer (frozen) | 768-d feature per second |
| Temporal recogniser | MS-TCN, 2 stages × 10 dilated layers, step softmax + instrument sigmoid | Per-second step and instrument probabilities |
| Facts builder | Viterbi decoding, segmentation, per-class reliability from out-of-fold predictions | Time-ordered segments with confidence |
| Keyframe selection | `notegen_rl/keyframes.py` | 1–2 frames per predicted segment, up to 16 per prompt |
| Structured-note policy | MedGemma 1.5 4B + LoRA (rank 32), image encoder frozen, GRPO with ms-swift and vLLM | Steps with times and confidences, instruments, closure |
| Final note writer | MedGemma-27B-text-it (FP8) | Clinical prose |
| Verifier | Deterministic checker | Every step, instrument and time in the prose traced to the structured note |

**Keyframes.** Each predicted segment in the prompt is followed by frames from inside it:

```
- [00:24:30-00:26:56] Sellotomy (confidence 0.34; historical reliability 0.52) | instruments: Kerrisons (0.83)
  keyframe 00:26:34: <image>
```

Frames are the seconds where the MS-TCN is most confident in the segment's step, skipping
out-of-patient, dark and blurred frames (measured inside the endoscope's circular view). Segments of
5 minutes or more get one frame from each half. Keyframes are chosen from predictions only, so
training and test prompts are built the same way. `keyframes.enabled: false` gives the text-only variant.

**Rewards** (`notegen_rl/rewards.py`), computed against the PitVis labels: grounding (step and
instrument F1, reliability-weighted recall) 1.0, calibration of stated confidences 0.5, temporal IoU
0.3, safety (findings and complications left to the surgeon) 0.3, format 0.2, conciseness 0.5.

## Usage

```bash
bash setup_env.sh                                   # environment, Endo-FM checkpoint, unit tests
bash run_pipeline.sh                                # stages 0-6
python scripts/04_build_datasets.py                 # prompts with keyframes -> work_dir/datasets_keyframes/
python scripts/05_train.py --stage grpo --resume    # GRPO (resumes from the newest full checkpoint)
python scripts/06_evaluate.py                       # template, base and GRPO scored with the rewards
python scripts/07_generate_note.py --video case.mp4 # note for a new video
python scripts/09_final_notes.py --completions <eval>/grpo/completions.jsonl
```

All settings are in `config.yaml`. MedGemma models are gated on Hugging Face (`HF_TOKEN`).
`infra/` holds the S3-driven EC2 job scripts.

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
- MedGemma 1.5 4B — https://huggingface.co/google/medgemma-1.5-4b-it
- MedGemma-27B-text-it — https://huggingface.co/google/medgemma-27b-text-it
