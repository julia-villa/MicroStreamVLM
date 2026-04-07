# MicroStreamVLM

MicroStreamVLM is a two-stage vision-language pipeline for long-range fitness coaching from video. The final project system is built on top of `meta-llama/Llama-3.2-3B-Instruct`, a frozen EfficientNet-based 3D-CNN stream encoder, trainable cross-attention alignment in Stage 2, and LoRA-based long-range fine-tuning in Stage 3.

The repository contains the code needed to:
- fine-tune the Stage 2 xattn-only model,
- fine-tune the Stage 3 LoRA model,
- build benchmark manifests,
- run benchmark evaluation, and
- run live webcam inference.

This is a clean code release. Large external assets such as datasets and model weights are **not** bundled in the GitHub repository.

---

## Repository Structure

```text
MicroStreamVLM/
├── README.md
├── requirements.txt
├── ckpts_efficientnet/
│   └── fitness_ally_hypermodel/
├── data/
├── notebooks/
│   ├── stage2_xattn_llama32_a40_instruct.ipynb
│   └── stage3_lora_long_range_a40_e15_instruct-v2-1.ipynb
├── outputs/
├── scripts/
│   ├── stage3_eval.py
│   ├── stage3_make_manifest.py
│   └── stage3_webcam_infer.py
└── src/
    ├── chat_format.py
    ├── constants.py
    ├── utils.py
    ├── custom_llama/
    ├── stage2/
    ├── stage3/
    └── vision_modules/
```

Key code paths:
- Stage 2 training: `src/stage2/`
- Stage 3 training and inference: `src/stage3/`
- Instruct chat-format alignment: `src/chat_format.py`
- Vision backbone and cross-attention modules: `src/vision_modules/`, `src/custom_llama/`
- Standalone benchmark tooling: `scripts/stage3_make_manifest.py`, `scripts/stage3_eval.py`
- Live webcam demo: `scripts/stage3_webcam_infer.py`

---

## Requirements

- Python 3.10+
- A Hugging Face account with access to the gated [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) checkpoint
- A GPU is strongly recommended
  - tested on NVIDIA A40
  - tested on Apple Silicon MPS for local runs

Install dependencies with:

```bash
pip install -r requirements.txt
```

Notes:
- `bitsandbytes` is listed for QLoRA-style runs on CUDA only
- evaluation uses `evaluate`, `rouge_score`, `bert-score`, `datasets`, and `nltk`
- the repo pins `numpy<2` to avoid binary compatibility issues with parts of the stack

---

## What Is Not Included

The following assets are **not included** in this GitHub repository because they are too large to distribute through GitHub directly:

- raw QEVD / QEVD-FIT-COACH dataset files
- raw QEVD-FIT-COACH benchmark videos
- EfficientNet 3D-CNN checkpoint weights
- trained Stage 2 checkpoints
- trained Stage 3 LoRA adapter weights
- cached `.pt` feature files and other large training artifacts

### External assets you must provide manually

QEVD dataset and QEVD-FIT-COACH data:
- https://www.qualcomm.com/developer/software/qevd-dataset

Original Qualcomm FitCoach / Stream-VLM repository:
- https://github.com/Qualcomm-AI-research/FitCoach

Paper:
- https://arxiv.org/pdf/2407.08101v2

The EfficientNet 3D-CNN checkpoint must be downloaded manually using the original Qualcomm FitCoach repository instructions / release assets and placed under:

```text
ckpts_efficientnet/fitness_ally_hypermodel/
```

We also do **not** provide our trained Stage 2 or Stage 3 weights in this repository, because those checkpoint files are too large for GitHub.

---

## Expected Data Layout

The `data/` directory is intentionally empty in this repository. After downloading the required data, place it in the following structure:

```text
data/combined/
  short_clips/
  fine_grained_labels.json
  feedbacks_short_clips.json
  questions.json
  long_range_videos_train/
  long_range_videos_benchmark/
  feedbacks_long_range_train.json
  feedbacks_long_range_benchmark.json
```

Expected purpose of these assets:
- `short_clips/` and associated annotations are used for Stage 2
- `long_range_videos_train/` and `feedbacks_long_range_train.json` are used for Stage 3 training
- `long_range_videos_benchmark/` and `feedbacks_long_range_benchmark.json` are used for benchmark segmentation and evaluation

---

## Stage 2 Fine-Tuning

Notebook:
- `notebooks/stage2_xattn_llama32_a40_instruct.ipynb`

This notebook trains the xattn-only adaptation stage:
- frozen LLM backbone
- frozen EfficientNet stream encoder
- trainable cross-attention pathway only

You must supply:
- the Hugging Face `meta-llama/Llama-3.2-3B-Instruct` model
- the EfficientNet 3D-CNN checkpoint under `ckpts_efficientnet/fitness_ally_hypermodel/`
- the Stage 2 short-clip training data under `data/combined/`

Expected output location for a trained Stage 2 run:

```text
outputs/<stage2_run_name>/final/
```

A valid Stage 2 final checkpoint directory should contain at least:
- `stage2_config.json`
- `xattn_state_dict.pt`
- tokenizer files
- config files

Important note:
- notebook cells may still contain cluster-era example paths or profile presets; update them locally before running on your machine

---

## Stage 3 Fine-Tuning

Notebook:
- `notebooks/stage3_lora_long_range_a40_e15_instruct-v2-1.ipynb`

This notebook trains the long-range coaching stage:
- frozen vision encoder
- frozen Stage 2 xattn pathway
- trainable LoRA adapters on the LLM

You must supply:
- a valid Stage 2 checkpoint directory under `outputs/.../final/`
- the EfficientNet 3D-CNN checkpoint under `ckpts_efficientnet/fitness_ally_hypermodel/`
- the long-range training and benchmark data under `data/combined/`

Expected output location for a trained Stage 3 run:

```text
outputs/<stage3_run_name>/final_adapter/
```

A valid Stage 3 adapter directory is expected to contain the PEFT adapter files produced by training, along with the saved adapter metadata.

Important notes:
- notebook cells may still contain cluster-era example paths or profile presets and should be updated locally
- bundled JSON files under `outputs/` are archival examples only; they are not substitutes for runnable Stage 2 or Stage 3 weights

---

## Evaluation

### 1. Build a segment manifest

If you have raw long-range metadata and videos, first convert them into the benchmark/train manifest format:

```bash
python scripts/stage3_make_manifest.py \
  --metadata-path data/combined/feedbacks_long_range_benchmark.json \
  --video-dir data/combined/long_range_videos_benchmark \
  --split benchmark \
  --output-path outputs/benchmark_manifest.json
```

### 2. Run evaluation

Evaluate a predictions JSON file against a benchmark manifest:

```bash
python scripts/stage3_eval.py \
  --predictions path/to/predictions.json \
  --references path/to/benchmark_manifest.json \
  --output-path outputs/metrics.json \
  --tolerance 3.0
```

Outputs include metrics such as:
- temporal F-score
- METEOR
- ROUGE-L
- BERTScore
- generation timing / throughput metrics where available

Notes:
- archived benchmark JSON files under `outputs/` can be used as examples of the expected schema
- files under `outputs/` were created in an earlier training environment with different absolute filesystem paths
- manifest JSONs bundled in `outputs/` may therefore contain old absolute cache paths and should not be treated as fresh cache manifests for a new machine

---

## Live Webcam Inference

Run the webcam demo with your own Stage 2 checkpoint, Stage 3 adapter, and EfficientNet checkpoint:

```bash
python scripts/stage3_webcam_infer.py \
  --stage2-checkpoint-dir outputs/<stage2_run_name>/final \
  --stage3-adapter-dir outputs/<stage3_run_name>/final_adapter \
  --vision-checkpoint ckpts_efficientnet/fitness_ally_hypermodel/efficientnet4Lite_1.8.3.checkpoint \
  --device auto
```

Important note:
- the script currently includes legacy default path strings from the earlier FitCoach repo layout, so passing the three paths above explicitly is recommended

Useful optional flags:
- `--camera-index`
- `--capture-fps`
- `--history-sec`
- `--decode-interval-sec`
- `--max-feedback-tokens`
- `--save-log`
- `--rotate-90-cw`

---

## Hugging Face Authentication

Stage 2 and Stage 3 load `meta-llama/Llama-3.2-3B-Instruct` from Hugging Face. Authenticate before running:

```bash
huggingface-cli login
```

Or set:

```bash
export HF_TOKEN=hf_...
```

---

## Acknowledgements

This project was developed with substantial inspiration from Qualcomm AI Research’s FitCoach / Stream-VLM work:
- GitHub repository: https://github.com/Qualcomm-AI-research/FitCoach
- Paper: *Live Fitness Coaching as a Testbed for Situated Interaction*
- Dataset page: https://www.qualcomm.com/developer/software/qevd-dataset

Some files, implementation ideas, and architectural components in this repository were adapted from the original Qualcomm repository. Original copyright and notice headers should be preserved where applicable.

The upstream Qualcomm FitCoach repository is published under the **BSD-3-Clause-Clear** license, as indicated on the original repository. Reuse in this project follows that upstream licensing basis for the borrowed components.
