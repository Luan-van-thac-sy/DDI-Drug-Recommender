# Plan: KELLM + Distill Integration

## Context

Teacher model hiện tại chỉ có BCE loss → kết quả thấp (Jaccard 0.2857 vs paper 0.5391). DDI warnings trong prompt bị truncate (LLaMA-1 limit 2048, prompt Leader ~3000 tokens). Cần tích hợp DDI/MDC loss **trực tiếp vào teacher loss function** theo KELLM paper, không phụ thuộc prompt text. Sau đó distill từ KELLM teacher sang student.

User xác nhận: knowledge context (DDI warnings) đã xây sẵn trong data. Code mới lưu trong `kellm+distill/`.

## Files to Create

| # | File | Purpose |
|---|------|---------|
| 1 | `kellm+distill/llama_kellm.py` | `LlamaForMedRecKELLM` — extends LlamaForMedRec + DDI/MDC loss |
| 2 | `kellm+distill/data_processor_kellm.py` | Data processors thêm `diag_indices` cho MDC loss |
| 3 | `kellm+distill/trainer_kellm.py` | `KELLMTeacherTrainer` — adaptive beta + DDI eval |
| 4 | `kellm+distill/main_kellm_cls.py` | Entry point (modified main_llm_cls.py) |
| 5 | `kellm+distill/train_kellm_teacher.bash` | Train KELLM teacher script |
| 6 | `kellm+distill/distill_kellm.bash` | Distill to student script (reuse main_distill.py) |

## Existing Code to Reuse (NOT duplicate)

- `llm/llama.py:LlamaForMedRec` — base class to extend
- `models/LEADER.py:240-256` — `compute_ddi_loss()` logic (copy formula, not import)
- `models/LEADER.py:258-284` — `compute_mdc_loss()` logic (simplified for teacher)
- `utils/mdc_context.py:build_mdc_matrix()` — build MDC matrix
- `llm/trainer_seq2seq.py:258-302` — `MedRecTrainer` base class
- `data/mimic3/handled/full/ddi_A_final.pkl` — DDI adjacency matrix

---

## Step-by-Step Implementation

### Step 1: `kellm+distill/llama_kellm.py`

```python
class LlamaForMedRecKELLM(LlamaForMedRec):
```

- `__init__`: Pop `ddi_adj`, `mdc_matrix` from kwargs, `register_buffer`, init `kellm_beta=0.0`, `kellm_alpha=0.5`
- `forward`: Call `super().forward()` → get BCE loss + logits → compute DDI/MDC loss → combine: `L = (1-β)·BCE + β·(α·MDC + (1-α)·DDI)` (KELLM eq 14)
- `compute_ddi_loss(logits)`: `probs = sigmoid(logits).float()`, `p^T @ ddi_adj @ p / num_pairs`
- `compute_mdc_loss(logits, diag_indices)`: Simplified — `diag_indices` is `(bs, max_diag)`, lookup `mdc_matrix[active_diags]`, dot with probs

Key: Cast to float32 for safety loss to avoid fp16 NaN.

### Step 2: `kellm+distill/data_processor_kellm.py`

Extend `llama_train_cls` and `llama_eval_cls`:
- Override `__call__` → call `super().__call__()` → extract `diag_indices` from `examples["records"]`
- Each sample: get last visit diagnoses → convert to `diag_voc.word2idx` indices → pad to `max_diag=50`
- Add `diag_indices` to `model_inputs`

Also handle the `records` field: HF `datasets.map()` removes columns after processing. Need to extract `diag_indices` DURING preprocessing since `records` column gets removed.

### Step 3: `kellm+distill/trainer_kellm.py`

```python
class KELLMTeacherTrainer(MedRecTrainer):
```

- Override `prediction_step`: Pass full inputs dict (including `diag_indices`) to model, not just `input_ids + labels`
- Add `_update_kellm_beta()`: After training, run eval → compute DDI rate → update `model.kellm_beta` (KELLM eq 13 with warmup: β=0 when DDI=0, β=ml_weight when DDI≤target, capped)
- Override `train()`: After `super().train()`, run beta update
- Add gradient clipping callback or override `training_step`

### Step 4: `kellm+distill/main_kellm_cls.py`

Modified copy of `main_llm_cls.py`:
- Import KELLM classes instead of originals
- Add KELLM args parsing (kellm_alpha, target_ddi, ddi_temp)
- Load `ddi_adj` from pickle: `data/{dataset}/handled/full/ddi_A_final.pkl`
- Build `mdc_matrix` via `build_mdc_matrix(diag_voc, med_voc)`
- Create `LlamaForMedRecKELLM.from_pretrained(..., ddi_adj=ddi_adj, mdc_matrix=mdc_matrix)`
- Use `llama_train_cls_kellm` / `llama_eval_cls_kellm` as preprocessors
- Use `KELLMTeacherTrainer` instead of `MedRecTrainer`
- Keep `max_source_length=1024` (fit LLaMA-1, avoid truncation of core prompt)

### Step 5: `kellm+distill/train_kellm_teacher.bash`

```bash
deepspeed --num_gpus=1 kellm+distill/main_kellm_cls.py \
    --deepspeed llm/ds.config \
    --do_train \
    --train_file data/mimic3/handled/train_leader.json \
    --model_name_or_path resources/llama-7b \
    --output_dir saved/kellm-teacher \
    --max_source_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 3000 \
    --learning_rate 2e-4 \
    --lora_rank 8 \
    --trainable q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --kellm_alpha 0.5 \
    --target_ddi 0.06 \
    --ddi_temp 2.0 \
    --fp16
```

### Step 6: `kellm+distill/distill_kellm.bash`

Reuse existing `main_distill.py` — point `--peft_path` to KELLM teacher:

```bash
python main_distill.py --dataset mimic3 \
    --model_name leader \
    --train_file leader \
    --distill \
    --peft_path saved/kellm-teacher/checkpoint-3000/ \
    --alpha 0.4 --d_loss mse \
    --max_source_length 1024 \
    --profile --align --align_weight 0.005 \
    --ddi --target_ddi 0.06 --ddi_temp 2.0 \
    --num_train_epochs 100 --train_batch_size 4
```

---

## Key Design Decisions

1. **max_source_length=1024** (not 2048): Core prompt fits 1024. DDI safety via loss function, not prompt text. Avoids truncation.
2. **β warmup**: β=0 initially (model learns BCE first), adaptive increase after model starts predicting.
3. **β cap at ml_weight=0.05**: Prevents BCE from vanishing (lesson from earlier NaN/zero-output bug).
4. **float32 for safety loss**: Avoid fp16 overflow in `p^T @ ddi_adj @ p`.
5. **Separate entry point** (`main_kellm_cls.py`): Don't modify original `main_llm_cls.py` — keep original pipeline working.

## Verification

1. **Unit test DDI loss**: Run `python test_ddi_fixes.py` (already exists)
2. **Train 1 epoch KELLM teacher**: Verify loss decreasing, no NaN
3. **Evaluate KELLM teacher**: Compare Jaccard/F1/DDI vs baseline teacher
4. **Distill**: Run 1 epoch distillation → verify student metrics
5. **Compare**: KELLM teacher vs original teacher vs KELLM student vs original student

```bash
# Quick test: 1 step train
deepspeed --num_gpus=1 kellm+distill/main_kellm_cls.py \
    --do_train --max_steps 10 \
    --train_file data/mimic3/handled/train_leader.json \
    --model_name_or_path resources/llama-7b \
    --output_dir saved/kellm-test \
    --max_source_length 1024 \
    --per_device_train_batch_size 2 \
    --fp16
```
