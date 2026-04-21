# Plan: DDI-Aware Medication Recommender (Capstone)

## Context

Improve LEADER model with DDI-aware drug recommendation. Code has DDI infrastructure (adjacency matrix, CLI args, eval metric) but **none wired into training loss**. Goal: reduce DDI rate while maintaining precision.

**Data source:** `improve/input/data4LLM_with_note.csv` (250 rows, 95 patients, clinical NOTE text)
**Pre-built artifacts:** `improve/output/mimic-iii/` (voc, DDI matrix, EHR adj, records, mappings — all ready)
**Execution:** Write code locally → run on Google Colab with GPU

---

## Data Directory Layout (what model expects)

```
data/mimic3/handled/
├── voc_final.pkl              ← copy from improve/output/mimic-iii/
├── profile_dict.json          ← BUILD from CSV
├── full/
│   ├── ddi_A_final.pkl        ← copy from improve/output/mimic-iii/
│   └── ehr_adj_final.pkl      ← copy from improve/output/mimic-iii/
├── train_leader.json          ← BUILD from CSV (JSONL, LEADER format)
├── val_leader.json            ← BUILD from CSV
└── test_leader.json           ← BUILD from CSV
```

Code reads: `--data_dir data/mimic3/handled/ --train_file leader`

---

## Phase 1: Data Preparation — `improve/prepare_data.py`

### What to build

Convert `data4LLM_with_note.csv` → LEADER JSONL format. Each JSON record needs:

```json
{
  "input": "The patient has N times ICU visits...[NOTE text]...",
  "target": "drug name 1, drug name 2, ...",
  "subject_id": 17,
  "drug_code": ["DB00364", "DB00465", ...],
  "records": {
    "diagnosis": [["4239", "5119"], ["7455", "2724"]],
    "procedure": [["3731", "8872"], ["3571"]],
    "medication": [["DB00986", "DB01400"], ["DB00364", "DB00465"]]
  },
  "profile": {"GENDER": "female", "AGE": "47"}
}
```

### Steps in `prepare_data.py`

1. Load `improve/output/mimic-iii/voc_final.pkl` — get `idx2word` mappings
2. Load `improve/input/data4LLM_with_note.csv` (250 rows)
3. Group by SUBJECT_ID, sort by ADMITTIME
4. For each patient:
   - Map `diag_id` → ICD-9 codes via `voc.diag_voc.idx2word`
   - Map `pro_id` → ICD-9 codes via `voc.pro_voc.idx2word`
   - Map `drug_id` → DrugBank IDs via `voc.med_voc.idx2word`
   - Build prompt using template: `"The patient has {N} times ICU visits.\n {history} In this visit..."`
   - Append NOTE text (truncated to ~200 tokens) to prompt where available
   - Build `drug_code` from last visit's DrugBank IDs
   - Build `target` from last visit's drug names
   - Build `profile` from GENDER, AGE
5. Split patients 80/10/10 → train/val/test JSONL
6. Build `profile_dict.json` with word2idx/idx2word for GENDER + AGE bins
7. Copy `voc_final.pkl`, `ddi_A_final.pkl`, `ehr_adj_final.pkl` to expected paths

**File to create:** `improve/prepare_data.py`

---

## Phase 2: DDI Loss in Student Model

### 2a. Pass DDI matrix to model
**File:** `trainers/distill_trainer.py` line 38, after `self.model = LEADER(...)`:
```python
ddi_adj_tensor = torch.FloatTensor(self.ddi_adj).to(self.device)
self.model.register_buffer('ddi_adj', ddi_adj_tensor)
```

### 2b. Add DDI loss method
**File:** `models/LEADER.py` — new method:
```python
def compute_ddi_loss(self, output):
    probs = torch.sigmoid(output)
    neg_pred = probs.unsqueeze(1)
    ddi_loss = torch.bmm(
        torch.bmm(neg_pred, self.ddi_adj.unsqueeze(0).expand(probs.size(0), -1, -1)),
        probs.unsqueeze(2)
    ).squeeze()
    return ddi_loss
```

### 2c. Wire into forward pass
**File:** `models/LEADER.py` lines 207-221, after BCE loss:
```python
if hasattr(self, 'ddi_adj') and self.ddi_flag:
    ddi_loss = self.compute_ddi_loss(output)
    ddi_weight = getattr(self, 'ddi_weight', self.ml_weight)
    loss = loss + ddi_weight * ddi_loss
```

### 2d. Store DDI flag in `__init__`
**File:** `models/LEADER.py` line ~85:
```python
self.ddi_flag = getattr(args, 'ddi', False)
```

**Run with:** `--ddi --ml_weight 0.05`

---

## Phase 3: Adaptive Safety Weighting

### 3a. Track beta in trainer
**File:** `trainers/distill_trainer.py` — in `__init__` add `self.ddi_beta = 0.0`

After each epoch eval (in parent `Trainer.train()` or override):
```python
current_ddi = acc_container.get("ddi", 0)
if current_ddi > self.args.target_ddi:
    self.ddi_beta = min(1.0, self.ddi_beta + (1.0 / self.args.ddi_temp))
else:
    self.ddi_beta = max(0.0, self.ddi_beta - (1.0 / self.args.ddi_temp))
self.model.ddi_weight = self.ddi_beta
```

**Uses existing args:** `--target_ddi 0.06 --ddi_temp 2.0`

---

## Phase 4: DDI Knowledge in Prompts

### 4a. Create `utils/ddi_context.py`
```python
def build_ddi_warnings(drug_names, drug_ids, ddi_adj):
    """Return list of DDI warning strings for known-interacting drug pairs."""
    warnings = []
    for i, di in enumerate(drug_ids):
        for j, dj in enumerate(drug_ids):
            if j <= i: continue
            if ddi_adj[di, dj] == 1 or ddi_adj[dj, di] == 1:
                warnings.append(f"{drug_names[i]} and {drug_names[j]}: known interaction")
    return warnings
```

### 4b. Enrich prompts in `prepare_data.py`
Append to each prompt:
```
Drug Safety Information:
- Acetaminophen and Warfarin: known interaction
- ...
```

---

## Phase 5: MDC Loss (stretch goal)

Build disease-drug contraindication matrix, add `compute_mdc_loss()` to LEADER.
Combined loss: `L = beta * (alpha * L_MDC + (1-alpha) * L_DDI) + (1-beta) * L_BCE`

---

## Phase 6: Evaluation

### 6a. Baseline → record Jaccard, F1, PRAUC, DDI rate
### 6b. Ablation table

| Run | Config |
|-----|--------|
| Baseline | No DDI loss |
| +DDI loss | `--ddi --ml_weight 0.05` |
| +Adaptive beta | `--ddi --target_ddi 0.06` |
| +DDI prompts | Knowledge-enriched prompts |
| +Clinical notes | NOTE text in prompts |

### 6c. Per-pair DDI analysis in `evaluate.py`

---

## Colab Notebook Structure: `improve/DDI_LEADER_Colab.ipynb`

### Cell 0: Setup Environment
```python
!pip install transformers==4.36.2 peft==0.10.0 deepspeed datasets accelerate bitsandbytes dill jsonlines
!pip install torch  # Colab has this pre-installed
```

### Cell 1: Mount Drive & Clone Repo
```python
from google.colab import drive
drive.mount('/content/drive')
# Option A: clone from git
# Option B: upload zip
!cp -r /content/drive/MyDrive/DDI-Drug-Recommender /content/project
%cd /content/project
```

### Cell 2: Upload/Copy Data Files
```python
import os, shutil
# Copy pre-built artifacts to expected locations
data_dir = "data/mimic3/handled/"
os.makedirs(data_dir + "full", exist_ok=True)

src = "improve/output/mimic-iii/"
shutil.copy(src + "voc_final.pkl", data_dir + "voc_final.pkl")
shutil.copy(src + "ddi_A_final.pkl", data_dir + "full/ddi_A_final.pkl")
shutil.copy(src + "ehr_adj_final.pkl", data_dir + "full/ehr_adj_final.pkl")
print("Artifacts copied!")
```

### Cell 3: Run Data Preparation
```python
!python improve/prepare_data.py
# Outputs: train_leader.json, val_leader.json, test_leader.json, profile_dict.json
# All placed in data/mimic3/handled/
```

### Cell 4 (Option A): Stage 2 Student Training — ONLINE distillation (no offline JSON needed)
Use this if you **did not** precompute offline hidden states/logits. This runs the teacher during student training.
```bash
python main_distill.py --dataset mimic3 \
  --model_name leader \
  --train_file leader \
  --data_dir data/mimic3/handled/ \
  --train_batch_size 4 \
  --gpu_id 0 \
  --num_train_epochs 100 \
  --distill \
  --check_path distill-online \
  --peft_path saved/lora-ddi/checkpoint-3000/ \
  --alpha 0.4 \
  --d_loss mse \
  --max_source_length 1024 \
  --profile \
  --num_workers 4 \
  --align \
  --align_weight 0.005 \
  --mark_name online \
  --log
```

### Cell 4 (Option B): Stage 2 Student Training — OFFLINE distillation (requires `offline_train_leader.json`)
If you pass `--offline`, the code expects:
`data/mimic3/handled/offline_train_leader.json`

To build it, run teacher prediction **on the train split** and then rename/copy the output:
```bash
# Teacher predict on TRAIN split (writes results/.../test_predictions.json)
deepspeed --num_gpus=1 main_llm_cls.py \
  --do_predict \
  --test_file data/mimic3/handled/train_leader.json \
  --cache_dir data/mimic3/handled/ \
  --prompt_column input \
  --response_column drug_code \
  --overwrite_cache \
  --model_name_or_path resources/llama-7b \
  --peft_path saved/lora-ddi/checkpoint-3000 \
  --output_dir results/teacher_lora_ddi_train \
  --overwrite_output_dir \
  --max_source_length 1024 \
  --max_target_length 256 \
  --per_device_eval_batch_size 4

# Move into the expected offline filename for student training
cp results/teacher_lora_ddi_train/test_predictions.json data/mimic3/handled/offline_train_leader.json
```

Then run student training with `--offline` (and optional DDI loss):
```bash
python main_distill.py --dataset mimic3 \
  --model_name leader \
  --train_file leader \
  --data_dir data/mimic3/handled/ \
  --train_batch_size 4 \
  --gpu_id 0 \
  --num_train_epochs 100 \
  --distill \
  --check_path distill-offline \
  --alpha 0.4 \
  --d_loss mse \
  --max_source_length 1024 \
  --profile \
  --offline \
  --num_workers 4 \
  --align \
  --align_weight 0.005 \
  --ddi \
  --ml_weight 0.05 \
  --target_ddi 0.06 \
  --ddi_temp 2.0 \
  --mark_name offline_ddi \
  --log
```

Notes:
- The legacy scripts under `experiment/mimic3/*.bash` are for older runs (different `train_file` naming). Prefer the notebook commands above for the `train_file=leader` flow.

### Cell 4: Verify Data
```python
import json, dill, numpy as np
voc = dill.load(open("data/mimic3/handled/voc_final.pkl", "rb"))
ddi = dill.load(open("data/mimic3/handled/full/ddi_A_final.pkl", "rb"))
train = [json.loads(l) for l in open("data/mimic3/handled/train_leader.json")]
print(f"Vocab: diag={len(voc['diag_voc'].word2idx)}, med={len(voc['med_voc'].word2idx)}, proc={len(voc['pro_voc'].word2idx)}")
print(f"DDI matrix: {ddi.shape}, pairs: {np.count_nonzero(ddi)//2}")
print(f"Train records: {len(train)}")
print(f"Sample prompt: {train[0]['input'][:200]}")
```

### Cell 5: Download LLaMA Model (or skip for offline distillation)
```python
# Option A: Use HuggingFace model
# !huggingface-cli login
# model will download automatically

# Option B: Skip teacher, use offline mode with pre-cached hidden states
# (requires running teacher first to cache outputs)
```

### Cell 6: Train LLM Teacher (Stage 1) — GPU Required
```bash
%%bash
# Only needed if training teacher from scratch
deepspeed --num_gpus=1 main_llm_cls.py \
    --deepspeed llm/ds.config \
    --do_train \
    --train_file data/mimic3/handled/train_leader.json \
    --cache_dir data/mimic3/handled/ \
    --prompt_column input \
    --response_column drug_code \
    --overwrite_cache \
    --model_name_or_path resources/llama-7b \
    --output_dir saved/lora-ddi \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 256 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 3000 \
    --logging_steps 100 \
    --save_steps 3000 \
    --learning_rate 2e-4 \
    --lora_rank 8 \
    --trainable q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --modules_to_save null \
    --lora_dropout 0.1 \
    --fp16
```

### Cell 7: Train Student with DDI Loss (Stage 2) — GPU Required
```bash
%%bash
# BASELINE (no DDI loss)
python main_distill.py --dataset mimic3 \
    --model_name leader \
    --train_file leader \
    --data_dir data/mimic3/handled/ \
    --train_batch_size 4 \
    --gpu_id 0 \
    --num_train_epochs 100 \
    --distill \
    --check_path distill-baseline \
    --peft_path saved/lora-ddi/checkpoint-3000/ \
    --alpha 0.4 \
    --d_loss mse \
    --max_source_length 1024 \
    --profile \
    --offline \
    --num_workers 4 \
    --align \
    --align_weight 0.005 \
    --mark_name baseline \
    --log
```

### Cell 8: Train with DDI Loss Enabled
```bash
%%bash
# WITH DDI LOSS
python main_distill.py --dataset mimic3 \
    --model_name leader \
    --train_file leader \
    --data_dir data/mimic3/handled/ \
    --train_batch_size 4 \
    --gpu_id 0 \
    --num_train_epochs 100 \
    --distill \
    --check_path distill-ddi \
    --peft_path saved/lora-ddi/checkpoint-3000/ \
    --alpha 0.4 \
    --d_loss mse \
    --max_source_length 1024 \
    --profile \
    --offline \
    --num_workers 4 \
    --align \
    --align_weight 0.005 \
    --ddi \
    --ml_weight 0.05 \
    --target_ddi 0.06 \
    --mark_name ddi_loss \
    --log
```

### Cell 9: Evaluate & Compare
```python
!python evaluate.py
# Compare baseline vs DDI-aware metrics
```

### Cell 10: DDI Weight Sweep (Trade-off Curve)
```python
import subprocess
weights = [0.01, 0.05, 0.1, 0.2, 0.5]
for w in weights:
    cmd = f"python main_distill.py --dataset mimic3 --model_name leader " \
          f"--train_file leader --data_dir data/mimic3/handled/ " \
          f"--train_batch_size 4 --gpu_id 0 --num_train_epochs 50 " \
          f"--distill --offline --profile --align --align_weight 0.005 " \
          f"--ddi --ml_weight {w} --mark_name sweep_{w} --check_path sweep-{w}"
    subprocess.run(cmd, shell=True)
```

### Cell 11: Plot Results
```python
import matplotlib.pyplot as plt
# Load results from each run and plot Jaccard vs DDI rate
```

---

## Implementation Order

```
1. Create improve/prepare_data.py        ← data pipeline
2. Create improve/DDI_LEADER_Colab.ipynb ← notebook
3. Modify models/LEADER.py              ← DDI loss (Phase 2)
4. Modify trainers/distill_trainer.py    ← pass DDI matrix + adaptive beta (Phase 2+3)
5. Create utils/ddi_context.py           ← DDI warnings for prompts (Phase 4)
6. Run notebook cells 0-4                ← verify data
7. Run notebook cells 5-8                ← train baseline + DDI
8. Run notebook cells 9-11               ← evaluate + plot
```

## Critical Files

| File | Action |
|------|--------|
| `improve/prepare_data.py` | CREATE — CSV → LEADER JSONL + copy artifacts |
| `improve/DDI_LEADER_Colab.ipynb` | CREATE — end-to-end notebook |
| `models/LEADER.py` | MODIFY — add `compute_ddi_loss()`, wire in forward (lines 85, 207-221) |
| `trainers/distill_trainer.py` | MODIFY — register DDI buffer (line 38), adaptive beta |
| `utils/ddi_context.py` | CREATE — DDI warning text builder |
| `evaluate.py` | MODIFY — add per-pair DDI analysis |

## Verification

1. Run `python improve/prepare_data.py` — should create train/val/test JSONs
2. Run Cell 4 in notebook — verify data loads correctly
3. Run baseline training — should get metrics similar to LEADER paper
4. Run DDI training — DDI rate should drop, Jaccard within ~2% of baseline
5. Compare ablation table across all runs
