# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of LEADER (Large Language Model Distilling Medication Recommendation) - a two-stage LLM-based medication recommendation system using knowledge distillation. Based on arXiv:2402.02803.

## Commands

### Setup
```bash
pip install -r requirements_fi.txt  # Full dependencies (recommended)
pip install -r requirements.txt     # Minimal dependencies
```

### Training Stage 1: LLM Teacher (LEADER(T))
```bash
# Single GPU (MIMIC-III)
bash experiment/mimic3/train_llm_cls.bash

# Multi-GPU with DeepSpeed (4 GPUs)
bash experiment/llm_cls.bash
```

### Training Stage 2: Knowledge Distillation (LEADER(S))
```bash
# Online distillation (LLM teacher runs during training)
bash experiment/mimic3/online_distill.bash

# Offline distillation (pre-cached LLM hidden states)
bash experiment/mimic3/offline_distill.bash
```

### Student Model Fine-tuning
```bash
python main_medrec.py [args]
```

### Evaluation
```bash
python evaluate.py
```

### Quick Validation
```bash
bash run_quick_test.sh
python test_single_input.py
```

## Architecture

### Two-Stage Pipeline

**Stage 1 - Teacher:** LLaMA-7B with a classification head (`LlamaForMedRec` in `llm/llama.py`) replaces the language generation head with a linear+sigmoid layer for multi-label medication prediction. Fine-tuned via LoRA on q/k/v/o/down/gate/up projections.

**Stage 2 - Student:** Compact transformer model (`LEADER` in `models/LEADER.py`) with parallel set encoders for diagnoses, procedures, and medications + a shared visit encoder. Trained with combined BCE + feature-level KD loss (distills hidden states, not predictions) + profile alignment loss (contrastive, for single-visit patients).

### Entry Points
- `main_llm_cls.py` - Train/test LLM teacher model
- `main_distill.py` - Knowledge distillation training
- `main_medrec.py` - Student model fine-tuning

### Key Modules
- `llm/` - LLaMA model adaptations, LoRA config, data processors, DeepSpeed config
- `models/LEADER.py` - Student model: PaddingEncoder, ProfileEncoder, LEADER class
- `generators/` - Dataset classes: `EHRTokenizer` (vocabularies for diag/proc/med), `DistillEHRDataset`
- `trainers/` - Training loops: base `Trainer`, `DistillTrainer`, `FinetuneTrainer`, `MedRecTrainer`
- `utils/evaluation.py` - Metrics: Jaccard, F1, PRAUC, DDI rate

### Data Flow
EHR records → prompt templates (`llm/data_processor/llama.py`) → LLaMA teacher → hidden states → distillation → student model → medication probabilities (thresholded)

### Data Requirements
- MIMIC-III/IV data in `data/mimic3/` or `data/mimic4/`
- Required files: `voc_final.pkl` (vocabulary), `ddi_A_final.pkl` (DDI adjacency matrix), `ehr_adj_final.pkl`, train/test JSON files
- Preprocessing notebooks in `data/mimic3/` and `data/mimic4/`

## Key Hyperparameters
- LoRA rank: 8, LR: 2e-4, batch size: 4, max seq length: 2048
- Distillation alpha: 0.4, alignment weight (beta): 0.005
- Training steps: 3000 (teacher), varies (student)
- Recommendation threshold gamma applied to sigmoid outputs

## Dependencies
Python 3.x, PyTorch, Transformers 4.28+, PEFT 0.10.0, DeepSpeed, RDKit (drug analysis), datasets, accelerate, bitsandbytes
