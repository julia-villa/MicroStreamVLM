# MicroStreamVLM

A two-stage vision-language model for real-time fitness coaching feedback from video streams, built on top of `meta-llama/Llama-3.2-3B-Instruct` with a frozen EfficientNet 3D-CNN stream encoder.

- **Stage 2** — cross-attention (xattn) alignment: only the inserted cross-attention layers are trained; everything else is frozen.
- **Stage 3** — LoRA fine-tuning on long-range video segments for temporal feedback generation.

---

## Requirements

- Python 3.10+
- A GPU is strongly recommended (tested on NVIDIA A40 and Apple Silicon MPS)
- A Hugging Face account with access to the gated [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) checkpoint
- The EfficientNet vision checkpoint — already included in the repo at:
  - `Stage2Llama3.2-3b-Instruct/ckpts_efficientnet/fitness_ally_hypermodel/efficientnet4Lite_1.8.3.checkpoint`
  - `Stage3Llama3.2-3b-Instruct/ckpts_efficientnet/fitness_ally_hypermodel/efficientnet4Lite_1.8.3.checkpoint`

---

## Installation

```bash
pip install -r requirements.txt
```

For QLoRA (4-bit quantization, CUDA only), `bitsandbytes` is included in `requirements.txt` but only used when `use_qlora=True` is passed at training time.

For evaluation metrics (METEOR, ROUGE-L, BERTScore), the following packages are required and are already listed:

```
evaluate  rouge_score  bert-score  nltk  datasets
```

---

## Pre-trained Checkpoints

The repo already includes trained outputs — you do not need to train from scratch to run evaluation or inference:

- **Stage 2 checkpoint** — `Stage2Llama3.2-3b-Instruct_Output&Weights/` contains a ready-to-use Stage 2 checkpoint (`xattn_state_dict.pt`, `stage2_config.json`, tokenizer files).
- **Stage 3 outputs** — `Stage3Llama3.2-4b-Instruct_Outputs/` contains `benchmark_manifest.json`, `benchmark_predictions_pilot32.json`, and `benchmark_metrics_pilot32.json`. You can run evaluation directly against these.

---

## Data Layout

The `data/` directory is **not included** in the repo and must be provided separately. Both stages expect it at `<repo_root>/data/combined/`:

```
data/combined/
  short_clips/                         # short video clips (Stage 2)
  fine_grained_labels.json
  feedbacks_short_clips.json
  questions.json
  long_range_videos_train/             # long-range videos + timestamp files (Stage 3)
  long_range_videos_benchmark/
  feedbacks_long_range_train.json
  feedbacks_long_range_benchmark.json
```

---

## Stage 2 — XAttn Alignment Training

Open and run the notebook:

```
Stage2Llama3.2-3b-Instruct/notebooks/stage2_xattn_llama32_a40_instruct.ipynb
```

> **Note:** The notebook hardcodes `repo_root` to a cluster path at the top of cell 1. Update it to your local path before running.

Key configuration at the top of the notebook:

| Variable | Description |
|---|---|
| `PROFILE` | `"cluster_a40"` or `"mac_m3_max"` |
| `repo_root` | **Must be updated** — set to the `Stage2Llama3.2-3b-Instruct/` directory |
| `config["llm_model_name_or_path"]` | HF model ID or local path to Llama 3.2 3B Instruct |
| `config["vision_checkpoint_path"]` | `repo_root / "ckpts_efficientnet/fitness_ally_hypermodel/efficientnet4Lite_1.8.3.checkpoint"` |
| `config["data_root"]` | Path to `data/combined/` directory (not included in repo) |
| `config["output_dir"]` | Where to save checkpoints |

The notebook trains for 2 epochs and saves per-epoch checkpoints plus a `final/` checkpoint. The final checkpoint directory contains:

```
final/
  stage2_config.json       # metadata including xattn_config and llm path
  xattn_state_dict.pt      # trained xattn weights
  tokenizer files
```

---

## Stage 3 — LoRA Fine-Tuning

Open and run the notebook:

```
Stage3Llama3.2-3b-Instruct/src/notebook/stage3_lora_long_range_a40_e15_instruct.ipynb
```

> **Note:** The notebook hardcodes `stage2_checkpoint_dir` to a cluster path. Update it to point to `Stage2Llama3.2-3b-Instruct_Output&Weights/` (the included Stage 2 checkpoint) or your own trained output.

Key configuration at the top of the notebook:

| Variable | Description |
|---|---|
| `PROFILE` | `"cluster_a40"` or `"mac_m3_max"` |
| `stage2_checkpoint_dir` | **Must be updated** — point to `Stage2Llama3.2-3b-Instruct_Output&Weights/` or a trained Stage 2 `final/` dir |
| `vision_checkpoint_path` | `Stage3Llama3.2-3b-Instruct/ckpts_efficientnet/fitness_ally_hypermodel/efficientnet4Lite_1.8.3.checkpoint` |
| `use_qlora` | `True` to enable QLoRA (CUDA only, requires `bitsandbytes`) |

The notebook will:
1. Cache encoded vision features for all train/val/benchmark segments
2. Build the Stage 3 LoRA model on top of the Stage 2 checkpoint
3. Train for the configured number of epochs
4. Generate benchmark predictions and run evaluation metrics

The final adapter is saved to `outputs/.../final_adapter/`.

---

## Evaluation

### Building a Segment Manifest

If you have raw long-range metadata, convert it to the segment manifest format first:

```bash
python Stage3Llama3.2-3b-Instruct/scripts/stage3_make_manifest.py \
  --metadata-path path/to/feedbacks_long_range_benchmark.json \
  --video-dir path/to/long_range_videos_benchmark \
  --split benchmark \
  --output-path outputs/benchmark_manifest.json
```

### Running Evaluation

To evaluate using the included pre-run outputs:

```bash
python Stage3Llama3.2-3b-Instruct/scripts/stage3_eval.py \
  --predictions Stage3Llama3.2-4b-Instruct_Outputs/benchmark_predictions_pilot32.json \
  --references Stage3Llama3.2-4b-Instruct_Outputs/benchmark_manifest.json \
  --output-path outputs/metrics.json \
  --tolerance 3.0
```

Or substitute your own prediction and manifest files:

```bash
python Stage3Llama3.2-3b-Instruct/scripts/stage3_eval.py \
  --predictions path/to/predictions.json \
  --references path/to/manifest.json \
  --output-path outputs/metrics.json \
  --tolerance 3.0
```

`--tolerance` controls the temporal matching window in seconds (default: 3.0).

**Output metrics** (`metrics.json`):

| Metric | Description |
|---|---|
| `temporal_f_score` | F-score for temporally-aligned feedback detection |
| `meteor` | METEOR score over matched feedback pairs |
| `rougeL` | ROUGE-L score over matched feedback pairs |
| `bert_score` | BERTScore F1 over matched feedback pairs |
| `mean_ttft_sec` | Mean time-to-first-token across predictions |
| `tokens_per_second` | Generation throughput |

---

## Live Webcam Inference

Run real-time coaching feedback from a webcam using a trained Stage 3 adapter:

```bash
python Stage3Llama3.2-3b-Instruct/scripts/stage3_webcam_infer.py \
  --stage2-checkpoint-dir Stage2Llama3.2-3b-Instruct_Output&Weights \
  --stage3-adapter-dir path/to/final_adapter \
  --vision-checkpoint Stage3Llama3.2-3b-Instruct/ckpts_efficientnet/fitness_ally_hypermodel/efficientnet4Lite_1.8.3.checkpoint \
  --device auto
```

Key optional flags:

| Flag | Default | Description |
|---|---|---|
| `--camera-index` | `0` | OpenCV camera index |
| `--capture-fps` | `8.0` | Live frame sampling rate |
| `--history-sec` | `12.0` | Rolling feature history length |
| `--decode-interval-sec` | `5.0` | How often to run the LLM decoder |
| `--max-feedback-tokens` | `32` | Max tokens per feedback span |
| `--save-log` | None | JSONL path to log emitted feedback events |
| `--rotate-90-cw` | off | Rotate frames 90° clockwise before encoding |

---

## HuggingFace Authentication

The Stage 2 and Stage 3 builds pull `meta-llama/Llama-3.2-3B-Instruct` from HuggingFace. Authenticate before running:

```bash
huggingface-cli login
```

Or set the environment variable:

```bash
export HF_TOKEN=hf_...
```
