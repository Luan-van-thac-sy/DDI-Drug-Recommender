from __future__ import annotations

from typing import List, Optional, Sequence


def build_ddi_warnings(
    drug_names: Sequence[str],
    drug_ids: Sequence[int],
    ddi_adj,
    *,
    max_warnings: Optional[int] = 20,
) -> List[str]:
    """
    Build human-readable warnings for known DDI (drug-drug interaction) pairs.

    Notes
    -----
    - `drug_ids` are expected to be integer indices aligned with the `ddi_adj` matrix
      (i.e., the same index space as the medication vocabulary used to build ddi_A_final.pkl).
    - Returns strings like: "<DrugA> and <DrugB>: known interaction"
    """
    if ddi_adj is None:
        return []

    try:
        ddi_n = int(ddi_adj.shape[0])
    except Exception:
        return []

    m = min(len(drug_names), len(drug_ids))
    if m < 2:
        return []

    # De-duplicate by id, keeping the first seen name for each id.
    id_to_name = {}
    for i in range(m):
        try:
            idx = int(drug_ids[i])
        except Exception:
            continue
        if idx < 0 or idx >= ddi_n:
            continue
        if idx not in id_to_name:
            id_to_name[idx] = str(drug_names[i])

    uniq_ids = sorted(id_to_name.keys())
    if len(uniq_ids) < 2:
        return []

    warnings: List[str] = []
    seen_pairs = set()

    for a_pos, di in enumerate(uniq_ids):
        for dj in uniq_ids[a_pos + 1 :]:
            try:
                interacts = ddi_adj[di, dj] != 0 or ddi_adj[dj, di] != 0
            except Exception:
                continue
            if not interacts:
                continue

            name_i = id_to_name[di]
            name_j = id_to_name[dj]
            pair_key = (name_i, name_j) if name_i <= name_j else (name_j, name_i)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            warnings.append(f"{name_i} and {name_j}: known interaction")
            if max_warnings is not None and len(warnings) >= max_warnings:
                return warnings

    return warnings

