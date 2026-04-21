# Loss Functions trong LEADER

## 1. Main Loss (BCE)
```python
loss = self.loss_fct(output, labels)  # BCEWithLogitsLoss
```
- **Input:** `output` là logits từ model, `labels` là ground truth multi-label
- **Mục đích:** Binary Cross-Entropy cho bài toán multi-label medication classification

---

## 2. DDI Loss (Drug-Drug Interaction)

```python
def compute_ddi_loss(self, output):
    probs = torch.sigmoid(output)  # (bs, med_voc_size)
    ddi_loss = torch.bmm(
        torch.bmm(
            probs.unsqueeze(1),  # (bs, 1, V)
            self.ddi_adj.unsqueeze(0).expand(probs.size(0), -1, -1)  # (bs, V, V)
        ),  # (bs, 1, V)
        probs.unsqueeze(2)  # (bs, V, 1)
    ).squeeze()  # (bs,)
    return ddi_loss
```

**Công thức:**
```
L_DDI = Σ(i,j) ddi_adj[i,j] × p_i × p_j
```

- **Mục đích:** Penalize việc prescrib đồng thời 2 drug có adverse interaction
- `ddi_adj[i,j] = 1` nếu drug i và j có tương tác xấu (nguy hiểm)
- `p_i, p_j`: xác suất được prescrib (sigmoid output)
- **Ý nghĩa:** Nếu model predict cao cho cả 2 drug có interaction → loss tăng cao, model sẽ tránh điều này

---

## 3. MDC Loss (Major Diagnostic Category)

```python
def compute_mdc_loss(self, output, diag_seq, seq_mask):
    probs = torch.sigmoid(output)  # (bs, med_voc_size)
    bs = probs.size(0)

    # Get last valid visit index for each patient
    visit_counts = seq_mask.sum(dim=1).long()  # (bs,)
    last_visit_idx = (visit_counts - 1).clamp(min=0)  # (bs,)

    # Extract diagnosis indices from last visit
    last_diags = diag_seq[torch.arange(bs), last_visit_idx]  # (bs, max_set_len)

    # Build per-patient contraindication mask
    mdc_penalty = torch.zeros(bs, device=probs.device)
    for i in range(bs):
        active = last_diags[i]
        active = active[active > 0]  # filter padding
        if len(active) == 0:
            continue
        # Sum mdc_matrix rows for active diagnoses
        contra = self.mdc_matrix[active].sum(dim=0)  # (num_med,)
        # Penalize: sum of (contraindication_strength * predicted_probability)
        mdc_penalty[i] = (contra * probs[i]).sum()

    return mdc_penalty
```

**Công thức:**
```
mdc_penalty = Σ contra[diag, drug] × p_drug
```

- **Mục đích:** Penalize việc prescrib drug contraindicated cho bệnh chẩn hiện tại của bệnh nhân
- Lấy diagnosis của visit cuối cùng → tra `mdc_matrix` → tính penalty
- `mdc_matrix[diag, drug] = 1` nếu drug bị contraindicated với diagnosis đó

---

## 4. Profile Alignment Loss (Contrastive)

```python
align_loss = self.align_profile(multi_label, med_pp)
```

- **Mục đích:** Học representation tốt hơn cho single-visit patients
- Dùng contrastive learning để align medication representation
- Giúp model học được semantic relationship giữa các medications

---

## 5. Distillation Loss (Knowledge Distillation)

```python
mediator = self.medrec[1](self.medrec[0](torch.cat([diag_emb, proc_emb, med_emb], dim=1)))
mediator = self.projector(mediator)
pseudo_hidden = llm_output["hidden_states"].float().detach()
distill_loss = self.distill_loss_fct_mse(mediator, pseudo_hidden)
```

**Công thức:**
```
distill_loss = MSE(student_hidden, teacher_hidden)
```

- **Mục đích:** Distill knowledge từ LLM teacher xuống student model
- So sánh hidden states (feature-level KD), không phải predictions
- Giúp student học được representation từ teacher

---

## Tổng hợp Loss trong Training

```python
# Main task loss
loss = self.loss_fct(output, labels).mean(dim=-1)

# Profile alignment
if self.align:
    align_loss = self.align_profile(multi_label, med_pp.view(med_pp.shape[0], -1))
    loss += self.align_weight * align_loss.mean(dim=-1)

# Safety losses (DDI + MDC)
if self.ddi_flag:
    ddi_loss = self.compute_ddi_loss(output)
    ddi_weight = getattr(self, 'ddi_weight', self.ml_weight)
    
    if self.mdc_flag:
        mdc_loss = self.compute_mdc_loss(output, diag_seq, seq_mask)
        alpha_mdc = self.mdc_weight / (self.mdc_weight + self.ml_weight + 1e-8)
        safety_loss = alpha_mdc * mdc_loss + (1 - alpha_mdc) * ddi_loss
        loss = (1 - ddi_weight) * loss + ddi_weight * safety_loss
    else:
        loss = loss + ddi_weight * ddi_loss

# Distillation
if self.distill:
    loss = loss + self.alpha * distill_loss
```

---

## Bảng tổng kết

| Loss | Mục đích | Cơ chế |
|------|----------|--------|
| **BCE** | Học predict medication chính xác | Binary Cross-Entropy |
| **DDI** | Tránh drug interactions nguy hiểm | Penalize co-prescription của interacting drugs |
| **MDC** | Tránh prescrib drug contraindicated | Penalize drugs contraindicated với diagnosis |
| **Alignment** | Cải thiện representation | Contrastive learning |
| **Distillation** | Học từ LLM teacher | MSE trên hidden states |

---

## Tham khảo

- File gốc: `models/LEADER.py:240-284`
- Paper: LEADER (Large Language Model Distilling Medication Recommendation) - arXiv:2402.02803