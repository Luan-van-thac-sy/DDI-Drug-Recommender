# Data Fields Usage — Các field trong JSON dùng ở step nào

## Tổng quan

| Field | Step | File | Mục đích |
|-------|------|------|---------|
| `"input"` | Stage 1 (Teacher) + Stage 2 (Distill) | `llm/data_processor/llama.py`, `generators/distill_generator.py` | Prompt text → tokenize cho LLaMA |
| `"drug_code"` | Stage 1 (Teacher) | `llm/data_processor/llama.py:160` | Target → multi-hot labels cho teacher |
| `"target"` | Stage 1b (Predict) | `llm/data_processor/llama.py:178` (eval class) | Drug names text — dùng khi evaluate |
| `"records"` | Stage 2 (Distill) | `generators/distill_generator.py:37-39` | `diagnosis`, `procedure`, `medication` sequences → student model input |
| `"profile"` | Stage 2 (Distill) | `generators/distill_generator.py:43-44` | Tuổi, giới → ProfileEncoder |
| `"drug_safety_info"` | **Không dùng** | — | Chỉ lưu metadata, không code nào đọc |

## Chi tiết theo stage

### Stage 1 — Train Teacher

```
"input"     → prompt cho LLaMA
"drug_code" → multi-hot labels
```

- `"input"`: Toàn bộ prompt text (visit history + current diagnosis + DDI warnings...)
- `"drug_code"`: Danh sách mã thuốc ATC (e.g., `["C10A", "A06A", "N02B"]`) → convert thành multi-hot vector kích thước `med_voc_size`

### Stage 1b — Predict Teacher

```
"input"     → prompt cho LLaMA
"drug_code" → ground truth để so sánh
```

### Stage 2 — Distill Student

```
"input"           → tokenize cho LLaMA teacher (lấy hidden states)
"records"         → diag/proc/med sequences → student LEADER model
  ├── diagnosis   → diag_seq
  ├── procedure   → proc_seq
  └── medication  → med_seq (visit trước = input, visit cuối = label)
"profile"         → ProfileEncoder (age, gender)
```

- `"input"`: Dùng để feed vào teacher model lấy hidden states cho knowledge distillation
- `"records"`: Structured EHR data — student model nhận trực tiếp (không qua LLM)
- `"profile"`: Demographics → ProfileEncoder tạo prompt vectors cho transformer

### Không dùng

```
"drug_safety_info" → chỉ metadata, không code nào load
```

Field `drug_safety_info` chứa `ddi_warnings` và `mdc_warnings` dạng structured list, nhưng không có code nào đọc field này. DDI info chỉ có tác dụng khi nằm trong `"input"` prompt text.
