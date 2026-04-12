"""
Multi-Disease Drug Contraindications (MDC) Matrix Builder

Builds a contraindication matrix mapping ICD-9 diagnosis codes to DrugBank drug codes.
Uses curated medical knowledge of common contraindications.

MDC matrix shape: (num_diagnoses, num_medications)
mdc_matrix[d, m] = 1 if drug m is contraindicated for diagnosis d
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional

# Curated MDC rules: ICD-9 code prefix → list of DrugBank IDs that are contraindicated
# Sources: FDA drug labels, clinical pharmacology references
# Format: { "ICD9_prefix": ["DrugBankID", ...] }
# ICD-9 prefix matching allows broader disease category coverage

CURATED_MDC_RULES = {
    # Renal failure / Chronic kidney disease (584-586) — contraindicated NSAIDs, nephrotoxic drugs
    "584": ["DB00945", "DB00465", "DB00328"],  # Aspirin, Ketorolac, Indomethacin
    "585": ["DB00945", "DB00465", "DB00328"],
    "586": ["DB00945", "DB00465", "DB00328"],
    "5849": ["DB00945", "DB00465", "DB00328"],

    # Heart failure (428) — contraindicated NSAIDs, calcium channel blockers (some)
    "428": ["DB00945", "DB00465", "DB00328"],  # NSAIDs worsen heart failure
    "4280": ["DB00945", "DB00465", "DB00328"],
    "4281": ["DB00945", "DB00465", "DB00328"],

    # Liver disease / Hepatic failure (570-573) — contraindicated hepatotoxic drugs
    "570": ["DB00316", "DB00795"],  # Acetaminophen, Sulfasalazine
    "571": ["DB00316", "DB00795"],
    "572": ["DB00316", "DB00795"],
    "573": ["DB00316", "DB00795"],

    # GI bleeding (578) — contraindicated anticoagulants, NSAIDs
    "578": ["DB00945", "DB00465", "DB01109", "DB00569"],  # Aspirin, Ketorolac, Heparin, Fondaparinux
    "5780": ["DB00945", "DB00465", "DB01109", "DB00569"],
    "5789": ["DB00945", "DB00465", "DB01109", "DB00569"],

    # Asthma (493) — contraindicated beta-blockers
    "493": ["DB00264", "DB01136"],  # Metoprolol, Carvedilol
    "4930": ["DB00264", "DB01136"],
    "4931": ["DB00264", "DB01136"],
    "4939": ["DB00264", "DB01136"],

    # Hyperkalemia (2767) — contraindicated potassium-sparing drugs, ACE inhibitors
    "2767": ["DB00722", "DB01344"],  # Lisinopril, Potassium chloride (if exists in voc)

    # Diabetes (250) — caution with corticosteroids (raise blood sugar)
    "250": ["DB00959", "DB00635"],  # Methylprednisolone, Prednisone
    "2500": ["DB00959", "DB00635"],

    # Peptic ulcer (531-534) — contraindicated NSAIDs
    "531": ["DB00945", "DB00465", "DB00328"],
    "532": ["DB00945", "DB00465", "DB00328"],
    "533": ["DB00945", "DB00465", "DB00328"],
    "534": ["DB00945", "DB00465", "DB00328"],

    # Coagulation disorders / thrombocytopenia (286-287) — contraindicated anticoagulants
    "286": ["DB01109", "DB00569", "DB00945"],  # Heparin, Fondaparinux, Aspirin
    "287": ["DB01109", "DB00569", "DB00945"],

    # Hypotension (458) — contraindicated vasodilators, some antihypertensives
    "458": ["DB00264", "DB01136"],  # Beta-blockers
    "4580": ["DB00264", "DB01136"],
    "45829": ["DB00264", "DB01136"],

    # Seizure disorders / epilepsy (345) — some drugs lower seizure threshold
    "345": ["DB00458"],  # Tramadol
    "3459": ["DB00458"],

    # Bradycardia (427.89, 427.81) — contraindicated beta-blockers, digoxin
    "42789": ["DB00264", "DB01136", "DB00390"],  # Metoprolol, Carvedilol, Digoxin
    "42781": ["DB00264", "DB01136", "DB00390"],
}


def build_mdc_matrix(
    diag_voc: object,
    med_voc: object,
    rules: Optional[Dict] = None,
) -> np.ndarray:
    """
    Build MDC matrix from curated rules.

    Parameters
    ----------
    diag_voc : Voc object with word2idx dict (ICD-9 code -> index)
    med_voc : Voc object with word2idx dict (DrugBank ID -> index)
    rules : dict mapping ICD-9 prefix -> list of contraindicated DrugBank IDs

    Returns
    -------
    mdc_matrix : np.ndarray of shape (num_diag, num_med)
    """
    if rules is None:
        rules = CURATED_MDC_RULES

    num_diag = len(diag_voc.word2idx)
    num_med = len(med_voc.word2idx)
    mdc = np.zeros((num_diag, num_med), dtype=np.float32)

    matched_rules = 0
    matched_pairs = 0

    for icd_prefix, drug_list in rules.items():
        # Find all diagnoses matching this ICD-9 prefix
        matching_diags = [
            idx for code, idx in diag_voc.word2idx.items()
            if str(code).startswith(icd_prefix)
        ]

        # Find matching drugs in vocabulary
        matching_drugs = [
            med_voc.word2idx[db_id]
            for db_id in drug_list
            if db_id in med_voc.word2idx
        ]

        if matching_diags and matching_drugs:
            matched_rules += 1
            for d_idx in matching_diags:
                for m_idx in matching_drugs:
                    mdc[d_idx, m_idx] = 1.0
                    matched_pairs += 1

    return mdc, matched_rules, matched_pairs


def build_mdc_warnings(
    diag_names: list,
    diag_ids: list,
    drug_names: list,
    drug_ids: list,
    mdc_matrix: np.ndarray,
    max_warnings: int = 10,
) -> list:
    """Build human-readable MDC warning strings."""
    warnings = []
    for i, di in enumerate(diag_ids):
        for j, mj in enumerate(drug_ids):
            try:
                if mdc_matrix[int(di), int(mj)] != 0:
                    warnings.append(
                        f"{drug_names[j]} is contraindicated for {diag_names[i]}"
                    )
                    if len(warnings) >= max_warnings:
                        return warnings
            except (IndexError, TypeError):
                continue
    return warnings
