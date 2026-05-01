# KELLM Comparison and Recommendations

→ Tất cả gộp thành "Knowledge Context" → nối vào input LLM

Khác biệt: KELLM dùng structured knowledge paths (nhân quả), không phải text warnings.

Tầng 2 — Loss function (trực tiếp trên predictions):

L_total = β · (α·L_MDC + (1-α)·L_DDI) + (1-β) · L_BCE

- L_DDI = ΣΣ M_ij · p_i · p_j — dùng DDI adjacency matrix trực tiếp
- L_MDC = ΣΣ Q_i,k · p_i — dùng MDC matrix trực tiếp
- Cả hai loss tính trên output probabilities của model, không phải từ text

## So sánh với implementation hiện tại

|                        | KELLM                         | Bạn hiện tại                              |
|------------------------|-------------------------------|-------------------------------------------|
| Input DDI              | Causal paths từ KG            | Text warnings trong prompt → bị truncate |
| Loss DDI (Teacher)     | Có — L_DDI trong teacher loss | Không — teacher chỉ có BCE               |
| Loss DDI (Student)     | N/A (KELLM không có student)  | Có — compute_ddi_loss                    |
| drug_safety_info field | N/A                           | Không dùng                                |

## Vấn đề chính: Teacher KHÔNG có DDI loss

KELLM apply DDI/MDC loss trực tiếp lên model output — không phụ thuộc vào prompt text. Bạn chỉ chèn text warnings vào prompt (bị truncate) mà teacher không có DDI loss.

## Recommend: Thêm DDI loss vào teacher

Thay vì chỉ dựa vào DDI text trong prompt, thêm DDI loss trực tiếp vào `LlamaForMedRec`:

```python
# llm/llama.py — forward()
loss = BCEWithLogitsLoss()(pooled_logits, labels.float())

# Thêm DDI loss (như KELLM)
if ddi_adj is not None:
    probs = torch.sigmoid(pooled_logits)
    ddi_loss = (probs @ ddi_adj) @ probs.T  # per sample
    loss = (1-beta) * loss + beta * ddi_loss
```

Như vậy teacher học tránh DDI pairs qua loss function — không cần text warnings trong prompt → không bị truncation, hiệu quả hơn.
