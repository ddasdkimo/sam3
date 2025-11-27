# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM 3 (Segment Anything Model 3) is Meta's unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts (points, boxes, masks). Key features include:
- Open-vocabulary concept segmentation via text prompts
- Visual prompts support (points, boxes, masks)
- Video tracking with temporal consistency
- 848M parameter model with DETR-based detector and SAM 2-based tracker

## Prerequisites

- Python 3.12+
- PyTorch 2.7+ with CUDA 12.6+
- Hugging Face authentication (model weights require access request at https://huggingface.co/facebook/sam3)

## Common Commands

### Installation
```bash
# Basic installation
pip install -e .

# With notebook support
pip install -e ".[notebooks]"

# For development (includes formatting tools and tests)
pip install -e ".[dev,train]"

# For video evaluation
pip install -e ".[veval]"
```

### Code Formatting
```bash
ufmt format .
```

### Running Tests
```bash
pytest
pytest tests/test_specific.py  # single test file
```

### Training
```bash
# Local single GPU
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 1

# Local multi-GPU
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 4

# SLURM cluster
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 1
```

### Evaluation
```bash
# SA-Co/Gold evaluation
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_metaclip_nps.yaml --use-cluster 0 --num-gpus 1

# Offline cgF1 evaluation
python scripts/eval/standalone_cgf1.py --pred_file /path/to/predictions.json --gt_files /path/to/gt.json
```

## Architecture Overview

### Core Model Components (`sam3/model/`)
- **Sam3Image**: Main image segmentation model combining detector components
- **Sam3VideoInferenceWithInstanceInteractivity**: Video model with tracking
- **ViT backbone** (`vitdet.py`): Vision Transformer for image encoding (1008x1008 input, patch size 14)
- **VETextEncoder** (`text_encoder_ve.py`): Text encoder for open-vocabulary prompts
- **SAM3VLBackbone** (`vl_combiner.py`): Vision-language backbone combining visual and text encoders
- **TransformerDecoder** (`decoder.py`): DETR-style decoder with presence token for text discrimination
- **Sam3TrackerPredictor** (`sam3_tracking_predictor.py`): Video tracking module based on SAM 2

### Model Builder (`model_builder.py`)
Entry point for constructing models:
- `build_sam3_image_model()`: Creates image segmentation model
- `build_sam3_video_model()`: Creates video tracking model
- `build_sam3_video_predictor()`: Creates multi-GPU video predictor wrapper

### Agent System (`sam3/agent/`)
- **agent_core.py**: Core agent logic for complex prompt processing
- **client_sam3.py**: SAM3 client interface
- **client_llm.py**: LLM client for agent reasoning
- **helpers/**: Utilities for masks, boxes, visualization, zoom-in operations

### Evaluation (`sam3/eval/`)
- **cgf1_eval.py**: Concept-grounded F1 evaluation metric
- **coco_eval.py**: COCO-style evaluation
- **saco_veval_eval.py**: SA-Co video evaluation
- **hota_eval_toolkit/**: HOTA tracking metrics
- **teta_eval_toolkit/**: TETA tracking metrics

### Training (`sam3/train/`)
- Uses Hydra configuration management
- Configs in `sam3/train/configs/`
- Supports SLURM cluster execution via submitit

## Key Configuration Files

- `sam3/train/configs/eval_base.yaml`: Base paths for evaluation datasets
- `sam3/train/configs/gold_image_evals/`: SA-Co/Gold evaluation configs
- `sam3/train/configs/silver_image_evals/`: SA-Co/Silver evaluation configs
- `sam3/train/configs/roboflow_v100/`: Roboflow dataset training configs
- `sam3/train/configs/odinw13/`: ODinW dataset training configs

## SA-Co Benchmark Datasets

- **SA-Co/Gold**: 7 image subsets with 3 independent human annotations each (oracle evaluation)
- **SA-Co/Silver**: 10 image subsets with single annotations
- **SA-Co/VEval**: 3 video domains (SA-V, YT-Temporal-1B, SmartGlasses)

Annotations use COCO-derived format with `text_input` field for noun phrases and RLE-encoded masks.
