# LEADER: Large Language Model Distilling Medication Recommendation Model

**Paper:** arXiv:2402.02803v2 (Jan 2025)
**Authors:** Qidong Liu, Xian Wu, Xiangyu Zhao, Yuanshao Zhu, Zijian Zhang, Feng Tian, Yefeng Zheng
**Code:** [LEADER-pytorch](https://github.com/liuqidong07/LEADER-pytorch)

---

## Main Idea

LEADER (LargE languAge moDel distilling mEdication Recommendation) addresses two key problems in medication recommendation:

1. **Lack of Semantic Understanding** -- Existing models rely on identity-based (ID) representations of diagnoses, procedures, and medications, ignoring the rich medical semantics embedded in their textual names.
2. **Single-Visit Patient Problem** -- Many state-of-the-art models require prescription history as input, making them unable to recommend medications for first-time hospital visitors.

LEADER solves both problems by leveraging Large Language Models (LLMs) and then distilling their knowledge into a lightweight student model suitable for deployment in resource-constrained healthcare settings.

---

## Method Overview (Two-Stage Framework)

### Stage 1: LLM-based Teacher Model -- LEADER(T)

- **Prompt Construction:** Patient EHR data (diagnoses, procedures, historical prescriptions) is formatted into natural language prompts that the LLM can understand.
- **Modified Output Layer:** The LLM's language generation head is replaced with a classification layer (linear + sigmoid) that outputs the probability of each medication, solving the **out-of-corpus problem** (LLMs generating drug names not in the valid drug set).
- **LoRA Fine-Tuning:** Only low-rank adapter matrices and the classification head are trained, keeping the LLM's pre-trained weights frozen. Uses binary cross-entropy loss.
- **Foundation Model:** LLaMA-7B.

### Stage 2: Knowledge Distillation -- LEADER(S)

- **Student Model Architecture:**
  - Three transformer-based set encoders for diagnoses, procedures, and medications (using ID embeddings, not text).
  - A shared visit encoder to capture temporal health history.
  - Profile features (age, gender, etc.) serve as pseudo medication records for single-visit patients.
- **Feature-Level Distillation:** Instead of distilling output probabilities (which are too close to ground truth), LEADER distills the **hidden state** from the LLM's last transformer layer into the student model via a learned projection.
- **Profile Alignment:** Contrastive learning aligns profile feature representations with medication set representations, improving single-visit patient performance.
- **Combined Loss:** `L = L_bce + alpha * L_KD + beta * L_align`

---

## Key Results

| Dataset   | Model      | PRAUC  | Jaccard | F1     |
|-----------|------------|--------|---------|--------|
| MIMIC-III | LEADER(T)  | 0.7816 | 0.5391  | 0.6921 |
| MIMIC-III | LEADER(S)  | 0.7795 | 0.5175  | 0.6737 |
| MIMIC-IV  | LEADER(T)  | 0.7120 | 0.4779  | 0.6296 |
| MIMIC-IV  | LEADER(S)  | 0.7020 | 0.4483  | 0.6005 |

- **LEADER(T)** achieves the best performance across all metrics on both datasets with statistically significant improvements over all baselines.
- **LEADER(S)** outperforms all medication recommendation baselines while being dramatically more efficient (small model, no LLM inference needed).
- LEADER(S) even surpasses LEADER(T) on single-visit patients under PRAUC, showing the benefit of combining collaborative signals with distilled semantic knowledge.

---

## Key Contributions

1. **First work** to integrate LLMs with medication recommendation via modified output layer and fine-tuning loss.
2. **Feature-level knowledge distillation** transfers LLM capabilities to a compact, deployable model.
3. **Profile alignment** via contrastive learning enables effective single-visit patient recommendation.
4. Consistent state-of-the-art results on MIMIC-III and MIMIC-IV datasets.

---

## Practical Significance

The two-stage approach is practical for healthcare deployment: the LLM is used only during training (as a teacher), while the lightweight student model handles inference -- addressing the **high inference cost problem** that makes direct LLM deployment impractical in hospitals with limited computing resources and strict privacy requirements.
