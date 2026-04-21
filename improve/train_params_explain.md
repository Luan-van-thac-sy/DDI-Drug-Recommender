---
title: "Giải thích Training Params — 2-Stage Pipeline (Teacher + Student)"
date: 2026-04-21
tags:
  - training
  - ddi-loss
  - knowledge-distillation
  - leader
  - lora
  - llama
---

# Giải thích Training Params — 2-Stage Pipeline

---

## Stage 1: Train LLM Teacher (LEADER(T)) với LoRA

### Command

```bash
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

### Model & Data

| Param | Value | Ý nghĩa |
|-------|-------|---------|
| `--model_name_or_path resources/llama-7b` | path | Base model = **LLaMA-7B**. Thay lm_head bằng classification head (linear+sigmoid) cho multi-label medication prediction |
| `--train_file data/mimic3/handled/train_leader.json` | path | File training — mỗi record chứa EHR prompt + medication labels |
| `--cache_dir data/mimic3/handled/` | path | Cache tokenized data |
| `--prompt_column input` | input | Cột chứa text prompt (EHR đã format thành natural language) |
| `--response_column drug_code` | drug_code | Cột chứa medication labels (multi-hot) |
| `--output_dir saved/lora-ddi` | path | Lưu LoRA checkpoint. **Stage 2 sẽ load từ đây** |

### DeepSpeed & Training

| Param | Value | Ý nghĩa |
|-------|-------|---------|
| `--deepspeed llm/ds.config` | config | DeepSpeed ZeRO config — tiết kiệm GPU memory cho LLaMA-7B |
| `--num_gpus 1` | 1 | Dùng 1 GPU (có thể tăng lên 4 cho multi-GPU) |
| `--per_device_train_batch_size 4` | 4 | 4 samples/GPU/step |
| `--gradient_accumulation_steps 2` | 2 | Tích lũy gradient 2 steps → effective batch size = 4×2 = **8** |
| `--max_steps 3000` | 3000 | Train đúng 3000 steps (không dùng epochs) |
| `--logging_steps 100` | 100 | Log loss mỗi 100 steps |
| `--save_steps 3000` | 3000 | Lưu checkpoint ở step 3000 (cuối training) |
| `--learning_rate 2e-4` | 2e-4 | LR cho LoRA params. Cao hơn fine-tune toàn bộ vì chỉ update ít params |
| `--fp16` | flag | Mixed precision FP16 — giảm memory, tăng tốc training |

### LoRA Config

| Param | Value | Ý nghĩa |
|-------|-------|---------|
| `--lora_rank 8` | 8 | Rank của LoRA adapter. Rank 8 = mỗi adapter thêm ~0.1% params so với full model |
| `--trainable q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj` | 7 modules | Apply LoRA lên **tất cả** attention projections (Q/K/V/O) + MLP layers (down/gate/up). Rộng hơn typical LoRA (thường chỉ Q/V) → expressiveness cao hơn |
| `--modules_to_save null` | null | Không save thêm module nào ngoài LoRA. Classification head được handle riêng bởi `LlamaForMedRec` |
| `--lora_dropout 0.1` | 0.1 | Dropout 10% trên LoRA weights — regularization |

### Sequence Length

| Param | Value | Ý nghĩa |
|-------|-------|---------|
| `--max_source_length 1024` | 1024 | Truncate EHR prompt tối đa 1024 tokens. Prompt chứa diagnosis/procedure/medication history |
| `--max_target_length 256` | 256 | Target (medication codes) tối đa 256 tokens. Dùng cho internal processing |

### Output
- Checkpoint tại `saved/lora-ddi/checkpoint-3000/` — chứa LoRA weights
- Teacher model = LLaMA-7B + LoRA → dùng làm teacher cho Stage 2

---

## Stage 2: Knowledge Distillation — Student Model (LEADER(S)) với DDI Loss

### Command

```bash
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
    --num_workers 4 \
    --align \
    --align_weight 0.005 \
    --ddi \
    --ml_weight 0.05 \
    --target_ddi 0.06 \
    --ddi_temp 2.0 \
    --mark_name ddi_loss
```

## Dataset & Model

| Param | Value | Ý nghĩa |
|-------|-------|---------|
| `--dataset mimic3` | mimic3 | Dùng dataset MIMIC-III |
| `--model_name leader` | leader | Student model = LEADER architecture |
| `--train_file leader` | leader | Load file `train_leader.json`, `val_leader.json`, `test_leader.json` |
| `--data_dir data/mimic3/handled/` | path | Thư mục chứa data đã xử lý (vocab, DDI matrix, EHR records) |

## Training Config

| Param | Value | Ý nghĩa |
|-------|-------|---------|
| `--train_batch_size 4` | 4 | 4 samples/batch. Nhỏ vì mỗi sample chứa nhiều visit sequences |
| `--gpu_id 0` | 0 | Dùng GPU 0 |
| `--num_train_epochs 100` | 100 | Train tối đa 100 epochs (có early stopping) |
| `--num_workers 4` | 4 | 4 DataLoader workers song song load data |
| `--check_path distill-ddi` | path | Lưu checkpoint vào `saved/distill-ddi/` |
| `--mark_name ddi_loss` | tag | Tên thí nghiệm — dùng cho logging/wandb |

## Knowledge Distillation (Stage 2 Core)

| Param | Value | Ý nghĩa |
|-------|-------|---------|
| `--distill` | flag | Bật knowledge distillation từ teacher → student |
| `--peft_path saved/lora-ddi/checkpoint-3000/` | path | LoRA weights của **teacher model** (LLaMA đã fine-tune ở Stage 1). Student học từ hidden states của teacher này |
| `--alpha 0.4` | 0.4 | Trọng số distillation loss. `total_loss = BCE + 0.4 × distill_loss + ...` |
| `--d_loss mse` | mse | Distillation dùng **MSE loss** giữa student hidden states và teacher hidden states (feature-based KD, không phải logit-based) |
| `--max_source_length 1024` | 1024 | Truncate LLM input prompt tối đa 1024 tokens. Teacher nhận prompt ≤1024 tokens |

## Profile & Alignment

| Param | Value | Ý nghĩa |
|-------|-------|---------|
| `--profile` | flag | Dùng **ProfileEncoder** thay vì PaddingEncoder. Encode thông tin bệnh nhân (tuổi, giới, ...) thành prompt vectors cho transformer |
| `--align` | flag | Bật **contrastive alignment loss** — căn chỉnh profile embedding với medication representation |
| `--align_weight 0.005` | 0.005 | Trọng số alignment loss. Nhỏ vì chỉ hỗ trợ, không phải loss chính |

## DDI Safety

| Param | Value | Ý nghĩa |
|-------|-------|---------|
| `--ddi` | flag | Bật **DDI penalty loss** — phạt model khi dự đoán cặp thuốc có tương tác xấu |
| `--ml_weight 0.05` | 0.05 | Trọng số ban đầu DDI loss. `loss += 0.05 × ddi_loss` |
| `--target_ddi 0.06` | 0.06 | DDI rate mục tiêu = 6%. Adaptive mechanism sẽ tăng/giảm DDI weight quanh mốc này |
| `--ddi_temp 2.0` | 2.0 | **Tốc độ điều chỉnh** adaptive beta. Mỗi eval: nếu DDI > target → `beta += 1/2.0 = 0.5`, nếu DDI < target → `beta -= 0.5`. Temp càng lớn → điều chỉnh càng chậm |

## Tổng Loss Function

**Không có MDC (case hiện tại, `mdc_flag=False`):**

```
L_total = L_BCE                               ← prediction loss (giữ nguyên, không bị scale)
        + β × (p^T @ ddi_adj @ p)             ← DDI penalty (cộng thêm)
        + 0.4 × MSE(student_h, teacher_h)     ← distillation
        + 0.005 × contrastive(prof, med)       ← alignment
```

- `β` bắt đầu = 0 (`ddi_beta`), adaptive tăng/giảm mỗi epoch dựa trên DDI rate so với target 6%
- Trước khi `β` được update lần đầu, `ddi_weight = ml_weight = 0.05`
- BCE **không bị giảm** khi β tăng — DDI chỉ là penalty phụ cộng thêm

**Nếu có MDC (`mdc_flag=True`):**

```
L_total = (1 - β) × L_BCE + β × (α_mdc × L_MDC + (1 - α_mdc) × L_DDI)
        + 0.4 × MSE(student_h, teacher_h)
        + 0.005 × contrastive(prof, med)
```

- BCE **bị scale xuống** khi β tăng — DDI/MDC thay thế phần BCE

## Flow Khi Train

```
Input data
  → LLM teacher (LoRA checkpoint-3000) → hidden states
  → Student LEADER model:
      - Profile → prompt vectors
      - Diag/Proc/Med sequences → transformer encoding
      - Output → medication probabilities
  → Loss = BCE + DDI penalty + distillation + alignment
  → Gradient clipping (max_norm=1.0)
  → Optimizer step
  → Mỗi epoch: eval → adaptive DDI beta update
```

---

## Tổng quan 2-Stage Pipeline

```
Stage 1: EHR data → LLaMA-7B + LoRA → fine-tune classification head
         Output: LoRA checkpoint (saved/lora-ddi/checkpoint-3000/)

Stage 2: EHR data → Teacher (Stage 1) → hidden states ─┐
         EHR data → Student (LEADER) ───────────────────┤
                                                         ↓
         Loss = BCE + DDI + Distill(MSE) + Align(Contrastive)
         Output: Compact student model cho deployment
```

Stage 1 tạo teacher mạnh (LLaMA-7B) nhưng nặng. Stage 2 chưng cất kiến thức sang student nhỏ (LEADER) giữ accuracy + thêm DDI safety.

---

## Bug Fix: NaN during Eval

### Root cause
DDI loss (`p^T @ ddi_adj @ p`) tạo gradient lớn → không có gradient clipping → weights bùng nổ → NaN output → `roc_auc_score` crash.

### Fix
1. `trainers/distill_trainer.py` — Thêm `clip_grad_norm_(max_norm=1.0)` sau `loss.backward()`
2. `utils/utils.py` — Thêm NaN guard trong `roc_auc` và `precision_auc`
