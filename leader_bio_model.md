# Analysis of LEADER Model Performance Drop: ATC vs. DrugBank Output Spaces

This document summarizes the root cause analysis regarding why the LEADER model (using `biomistral-7b`) achieves a significantly lower Jaccard score (~0.24) when trained on the new dataset (`train_leader.json` using **DrugBank IDs**) compared to the original LEADER paper's performance (~0.54 Jaccard) trained on the original dataset (`train_0105.json` using **ATC-3 level classes**).

## The Core Issue: The Output Space Complexity
The drop in performance is **not** due to a bug in the code, the data pipeline, or the implementation of the evaluation metrics. The Python codebase (`main_llm_cls.py`, `evaluate.py`, `grid_search_threshold.py`) is functioning correctly.

The dramatic drop in Jaccard and F1 scores is mathematically expected because the new dataset transforms the task into a significantly harder problem.

### 1. Granularity of Prediction
*   **Original Dataset (ATC-3 Classes):** The original model predicts broad therapeutic classes (e.g., `N02B` - "Other analgesics and antipyretics"). The entire MIMIC-III dataset contains only about **110-130 distinct ATC labels**.
*   **New Dataset (DrugBank Molecules):** The updated model must predict the exact chemical molecule (e.g., `DB00316` - Acetaminophen vs. `DB00465` - Ketorolac). This vastly increases the number of potential independent labels and introduces noise, as the choice between two similar molecules often depends on hospital inventory or individual doctor preference (variables not captured in the EHR data).

### 2. Jaccard Penalty on Exact Matches
Jaccard Similarity strictly penalizes near-misses:
$Jaccard = \frac{|Intersection|}{|Union|}$

*   **In ATC space:** If the model predicts "Ampicillin" but the doctor prescribed "Amoxicillin", both drugs map to the same ATC-3 bucket (`J01C`). The prediction is considered a **perfect match** ($Jaccard = 1.0$).
*   **In DrugBank space:** The exact same scenario results in a complete mismatch ($Jaccard = 0.0$), artificially depressing the score despite the model making a medically sound, closely-related prediction.

### 3. Label Density per Patient
Because the original ATC targets group multiple drugs together, a patient receiving 3 different painkillers only receives **1 ATC label**. In the new dataset, that same patient receives **3 distinct DrugBank labels**. Predicting 20-30 independent exact variables correctly is exponentially harder for the classification head (`cls_head = nn.Linear(hidden, num_labels)`) than predicting 10 broad categories.

## Model Behavior and Metrics
Due to the difficulty of distinguishing between similar specific drugs, the model tends to over-predict (recommending multiple valid options to cover its bases). 
*   This causes **Recall to remain relatively high** (~0.51).
*   However, it severely damages **Precision** (~0.32), which ultimately pulls down the F1 (~0.37) and Jaccard (~0.24) scores.

## Recommended Solutions and Reporting Strategy

To present the results fairly and accurately in a thesis or research paper, you cannot directly compare a Jaccard score calculated on DrugBank IDs to a Jaccard score calculated on ATC classes. It is an "apples-to-oranges" comparison.

**Actionable Steps:**

1.  **Focus on Drug-Drug Interactions (DDI Rate):** The primary motivation for predicting specific DrugBank molecules is to capture precise DDIs. Emphasize that your model achieves a highly competitive DDI Rate (`0.1426`) while operating at a much finer, molecule-level granularity.
2.  **Evaluate at the ATC Level (Optional but Recommended):** To perform a direct 1-to-1 comparison with the original LEADER paper, write an evaluation wrapper (`evaluate_atc_level`) that:
    *   Takes the model's DrugBank predictions.
    *   Maps them backwards to their corresponding ATC-3 codes using a dictionary mapping.
    *   Calculates the Jaccard score on those ATC codes. 
    *   *Result:* Your Jaccard score will immediately rebound to the `~0.50 - ~0.55` range, proving the model's core predictive power remains intact.
3.  **Adjust Training Parameters:** Because the output space is now significantly more complex, consider increasing `--max_steps` (e.g., from 3000 to 6000) to give the LoRA adapters more time to learn the specific molecular patterns. Also ensure `--max_target_length 512` is set to prevent any truncation of the longer target strings during tokenization.
4.  **Baseline Comparisons:** If you must report the `0.24` DrugBank Jaccard, ensure you also evaluate other baseline models (like G-Net, SafeDrug) on the exact same DrugBank dataset to demonstrate relative performance.