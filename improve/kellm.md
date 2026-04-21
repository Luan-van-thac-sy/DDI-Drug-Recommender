# KELLM: Knowledge-Enhanced Label-Wise Large Language Model for Safe and Interpretable Drug Recommendation

**Paper:** Electronics 2025, 14, 154 (Jan 2025)
**Authors:** Tianhan Xu, Bin Li (Yangzhou University)
**DOI:** https://doi.org/10.3390/electronics14010154

---

## Main Idea

KELLM addresses three shortcomings of existing drug recommendation models:

1. **Accuracy** -- Existing methods don't adequately integrate external medical knowledge; prompt-based LLM approaches can generate unpredictable/out-of-corpus drug labels.
2. **Safety** -- Most models only address DDIs (drug-drug interactions) but ignore **MDCs (multi-disease drug contraindications)**, which are critical for patients with comorbidities.
3. **Interpretability** -- Attention weights are insufficient to explain the causal mechanisms behind recommendations.

KELLM solves these via a three-step pipeline: medical entity recognition, causal knowledge integration from a knowledge graph, and a modified label-wise LLaMA architecture for multi-label drug classification.

---

## Method Overview (Three-Step Pipeline)

### Step 1: Medical Entity Recognition (MER)

- Uses fine-tuned DeBERTa on PubMed to extract medical entities (symptoms, signs, tests, diagnoses, procedures, medications) from raw EHR text.
- ChatGPT supplements the results to improve recall.

### Step 2: Causal Knowledge Integration (CKI)

- Extracted entities are linked to a **medical knowledge graph** (stored in Neo4j).
- **Causal Paths:** Multi-hop paths are mined from symptoms/tests to diseases/drugs (max path length L=6).
- **Causal Neighbors:** One-hop neighbors of key entities provide broader context.
- DDI and MDC knowledge is also retrieved.
- All combined into a **Knowledge Context** that enriches the LLM input, providing both accuracy improvements and interpretability via explicit causal chains.

### Step 3: Label-Wise LLaMA Architecture

- **Modified LLaMA-2-7B** adapted for multi-label classification (not autoregressive generation).
- Input = concatenation of extracted medical entities + knowledge context.
- **Unmasked variant** (bidirectional attention) outperforms the causal-masked variant for classification.
- **Max pooling** over hidden states for global feature aggregation.
- Output layer: linear + sigmoid for multi-label drug prediction.
- Fine-tuned with **LoRA** (rank=12, alpha=32).

### Loss Function

```
L_total = beta * (alpha * L_MDC + (1-alpha) * L_DDI) + (1-beta) * L_BCE
```

- **L_BCE:** Standard binary cross-entropy for drug classification.
- **L_DDI:** Penalizes co-prescription of drugs with known adverse interactions (via DDI adjacency matrix).
- **L_MDC:** Penalizes drugs contraindicated for the patient's diseases.
- **beta:** Adaptive safety adjustment factor -- when DDI/MDC risks exceed thresholds, the model shifts focus from accuracy to safety.

---

## Key Results

### Effectiveness (MIMIC-III / MIMIC-IV)

| Model       | PRAUC  | F1     | Jaccard | Effectiveness |
|-------------|--------|--------|---------|---------------|
| KELLM       | 0.7906 | 0.6784 | 0.5292  | 0.6661        |
| GraphCare   | 0.7851 | 0.6447 | 0.4958  | 0.6419        |
| COGNet      | 0.7771 | 0.6805 | 0.5275  | 0.6617        |
| LEADER(S)   | 0.7795 | 0.6737 | 0.5175  | 0.6572        |

KELLM achieves highest Effectiveness score on both datasets.

### Safety (MIMIC-III)

| Model    | DDI    | MDC    | Safety |
|----------|--------|--------|--------|
| KELLM    | 0.0718 | 0.0145 | 0.0348 |
| SafeDrug | 0.0686 | 0.0228 | 0.0383 |
| 4SDrug   | 0.0715 | 0.0213 | 0.0392 |

- KELLM achieves the **lowest overall Safety score** (lower = safer).
- Particularly strong on MDC reduction (0.0145 vs ground truth 0.0236).

### Ablation Study Highlights

- Zero-shot LLaMA-2-7B: PRAUC drops to 0.5725 (fine-tuning essential).
- Without MER: moderate degradation (entity extraction helps focus input).
- Without Causal Chains: significant drop in F1 and Jaccard (causal knowledge crucial).
- Without Label-wise LLaMA: large drop (classification head >> text generation for this task).

---

## Key Contributions

1. **Causal Knowledge Integration (CKI)** algorithm enriches LLM input with causal chains from medical KG, improving both accuracy and interpretability.
2. **Label-wise LLaMA** architecture with bidirectional attention and adaptive pooling for multi-label drug classification.
3. **Dual safety constraints** (DDI + MDC) with adaptive loss weighting -- first model to address MDCs in drug recommendation.
4. **Interpretable reasoning** via explicit causal chains linking symptoms to diagnoses to drug recommendations.

---

## Comparison with LEADER

| Aspect | LEADER | KELLM |
|--------|--------|-------|
| LLM Usage | Teacher-student distillation | Direct fine-tuned LLM inference |
| Knowledge Source | EHR data only | EHR + external medical KG |
| Safety | No explicit safety constraints | DDI + MDC loss terms |
| Interpretability | None | Causal chains from KG |
| Inference Efficiency | High (small student model) | Low (requires LLM at inference) |
| Single-visit Patients | Profile alignment technique | Not explicitly addressed |
| PRAUC (MIMIC-III) | 0.7795 | 0.7906 |
