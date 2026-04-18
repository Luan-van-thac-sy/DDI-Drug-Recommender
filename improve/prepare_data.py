"""
Phase 1: Data Preparation for DDI-Aware LEADER
Converts data4LLM_with_note.csv → LEADER JSONL format
Copies pre-built artifacts (voc, DDI, EHR) to expected paths
"""

import os
import csv
import ast
import json
import random
import shutil
import sys
import dill
import numpy as np

# ============================================================
# Config
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

INPUT_CSV = os.path.join(BASE_DIR, "input", "data4LLM_with_note.csv")
OUTPUT_ARTIFACTS = os.path.join(BASE_DIR, "output", "mimic-iii")
DATA_DIR = os.path.join(PROJECT_DIR, "data", "mimic3", "handled")
FULL_DIR = os.path.join(DATA_DIR, "full")

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO = 0.1 (remainder)

# Prompt templates (same as construction.ipynb cell-38)
MAIN_TEMPLATE = (
    "The patient has <VISIT_NUM> times ICU visits. \n "
    "<HISTORY> In this visit, he has diagnosis: <DIAGNOSIS>; "
    "procedures: <PROCEDURE>. Then, the patient should be prescribed: "
)
HIST_TEMPLATE = (
    "In <VISIT_NO> visit, the patient had diagnosis: <DIAGNOSIS>; "
    "procedures: <PROCEDURE>. The patient was prescribed drugs: <MEDICATION>. \n"
)

MAX_HISTORY_VISITS = 3  # keep at most last 3 historical visits

# Allow running this script from within `improve/` as well as repo root.
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.ddi_context import build_ddi_warnings  # noqa: E402
from utils.mdc_context import build_mdc_matrix, build_mdc_warnings  # noqa: E402


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def concat_str(str_list):
    return ", ".join(str_list)


def parse_list(s):
    """Parse a string like '[1, 2, 3]' or \"['a', 'b']\" into a Python list."""
    return ast.literal_eval(s)


def build_profile_dict(rows):
    """Build profile tokenizer dict from GENDER and AGE fields."""
    prof_dict = {"word2idx": {}, "idx2word": {}}

    # GENDER
    genders = sorted(set(r["GENDER"] for r in rows))
    prof_dict["word2idx"]["GENDER"] = {g: i for i, g in enumerate(genders)}
    prof_dict["idx2word"]["GENDER"] = {i: g for i, g in enumerate(genders)}

    # AGE — bin into decades
    ages = sorted(set(r["AGE"] for r in rows))
    prof_dict["word2idx"]["AGE"] = {a: i for i, a in enumerate(ages)}
    prof_dict["idx2word"]["AGE"] = {i: a for i, a in enumerate(ages)}

    return prof_dict


def build_patient_records(rows, voc):
    """Group CSV rows by patient, sort by time, build LEADER-format records."""
    diag_idx2word = voc["diag_voc"].idx2word
    pro_idx2word = voc["pro_voc"].idx2word
    med_idx2word = voc["med_voc"].idx2word

    # Group by patient
    patients = {}
    for r in rows:
        sid = r["SUBJECT_ID"]
        if sid not in patients:
            patients[sid] = []
        patients[sid].append(r)

    # Sort each patient's visits by ADMITTIME
    for sid in patients:
        patients[sid].sort(key=lambda x: x["ADMITTIME"])

    llm_data = []

    for sid, visits in patients.items():
        if len(visits) < 2:
            # Single-visit patients: still include them (LEADER supports this)
            # They have no history, only current visit
            pass

        visit_num = len(visits) - 1  # number of historical visits

        # Build history strings for prompt
        history_parts = []
        # Build records structure (all visits including current)
        records = {"diagnosis": [], "procedure": [], "medication": []}

        for visit_no, visit in enumerate(visits):
            diag_ids = parse_list(visit["diag_id"])
            pro_ids = parse_list(visit["pro_id"])
            drug_ids = parse_list(visit["drug_id"])
            drug_names = parse_list(visit["drug_name"])
            diag_names = parse_list(visit["diagnose"])
            proc_names = parse_list(visit["procedure"])

            # Map IDs back to standard codes for records
            diag_codes = [str(diag_idx2word[i]) for i in diag_ids if i in diag_idx2word]
            pro_codes = [str(pro_idx2word[i]) for i in pro_ids if i in pro_idx2word]
            med_codes = [str(med_idx2word[i]) for i in drug_ids if i in med_idx2word]

            records["diagnosis"].append(diag_codes)
            records["procedure"].append(pro_codes)
            records["medication"].append(med_codes)

            # Build history prompt (all visits except last)
            if visit_no < len(visits) - 1:
                hist_str = HIST_TEMPLATE.replace("<VISIT_NO>", str(visit_no + 1))
                hist_str = hist_str.replace("<DIAGNOSIS>", concat_str(diag_names))
                hist_str = hist_str.replace("<PROCEDURE>", concat_str(proc_names))
                hist_str = hist_str.replace("<MEDICATION>", concat_str(drug_names))
                history_parts.append(hist_str)

        # Keep at most last N historical visits
        if len(history_parts) > MAX_HISTORY_VISITS:
            history_parts = history_parts[-MAX_HISTORY_VISITS:]

        # Current visit (last one) — used for diagnosis/procedure in prompt and as target
        current = visits[-1]
        current_diag_names = parse_list(current["diagnose"])
        current_diag_ids = parse_list(current["diag_id"])
        current_proc_names = parse_list(current["procedure"])
        current_drug_names = parse_list(current["drug_name"])
        current_drug_ids = parse_list(current["drug_id"])
        current_drug_codes = [
            str(med_idx2word[i]) for i in current_drug_ids if i in med_idx2word
        ]

        # Build main prompt
        hist_str = "".join(history_parts)
        prompt = MAIN_TEMPLATE.replace("<VISIT_NUM>", str(visit_num))
        prompt = prompt.replace("<HISTORY>", hist_str)
        prompt = prompt.replace("<DIAGNOSIS>", concat_str(current_diag_names))
        prompt = prompt.replace("<PROCEDURE>", concat_str(current_proc_names))

        # Append clinical NOTE if available (truncated)
        note = current.get("NOTE", "").strip()
        if note:
            # Truncate to ~500 chars to stay within token limits
            truncated_note = note[:500]
            prompt += f"\nClinical Notes: {truncated_note}"

        # Append DDI knowledge (Phase 4) based on known interacting pairs in current meds
        warnings = build_ddi_warnings(
            current_drug_names, current_drug_ids, voc.get("ddi_adj")
        )
        prompt += "\nDrug Safety Information:\n"
        if warnings:
            prompt += "\n".join(f"- {w}" for w in warnings)
        else:
            prompt += "- No known interactions among the current medications."

        # Append MDC warnings (Phase 5) for disease-drug contraindications
        mdc_matrix = voc.get("mdc_matrix")
        mdc_warns = []
        if mdc_matrix is not None:
            mdc_warns = build_mdc_warnings(
                current_diag_names,
                current_diag_ids,
                current_drug_names,
                current_drug_ids,
                mdc_matrix,
                max_warnings=10,
            )
            if mdc_warns:
                prompt += "\nContraindication Warnings:\n"
                prompt += "\n".join(f"- {w}" for w in mdc_warns)

        # Build drug safety info field
        drug_safety_info = {
            "ddi_warnings": warnings if warnings else [],
            "mdc_warnings": mdc_warns if mdc_warns else [],
        }

        # Build target string (drug names)
        target = concat_str(current_drug_names)

        # Build profile
        profile = {
            "GENDER": current["GENDER"],
            "AGE": current["AGE"],
        }

        record = {
            "input": prompt,
            "target": target,
            "subject_id": int(sid),
            "drug_code": current_drug_codes,
            "records": records,
            "profile": profile,
            "drug_safety_info": drug_safety_info,
        }
        llm_data.append(record)

    return llm_data


def split_data(data, train_ratio, val_ratio, seed):
    """Split data by patient into train/val/test."""
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    n_train = int(len(data) * train_ratio)
    n_val = int(len(data) * val_ratio)

    train = [data[i] for i in indices[:n_train]]
    val = [data[i] for i in indices[n_train : n_train + n_val]]
    test = [data[i] for i in indices[n_train + n_val :]]

    return train, val, test


def save_jsonl(path, data):
    """Save list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    print("=" * 60)
    print("Phase 1: Data Preparation for DDI-Aware LEADER")
    print("=" * 60)

    # 1. Load vocabulary
    voc_path = os.path.join(OUTPUT_ARTIFACTS, "voc_final.pkl")
    print(f"\n[1] Loading vocabulary from {voc_path}")
    voc = dill.load(open(voc_path, "rb"))
    print(f"    diag_voc: {len(voc['diag_voc'].word2idx)} codes")
    print(f"    med_voc:  {len(voc['med_voc'].word2idx)} codes")
    print(f"    pro_voc:  {len(voc['pro_voc'].word2idx)} codes")

    # Load DDI adjacency for prompt enrichment (Phase 4)
    ddi_path = os.path.join(OUTPUT_ARTIFACTS, "ddi_A_final.pkl")
    print(f"\n[1b] Loading DDI adjacency from {ddi_path}")
    ddi_adj = dill.load(open(ddi_path, "rb"))
    voc["ddi_adj"] = ddi_adj
    print(f"    ddi_adj shape: {ddi_adj.shape}")

    # Load MDC matrix for prompt enrichment (Phase 5)
    print("\n[1c] Building MDC matrix...")
    mdc_matrix, matched_rules, matched_pairs = build_mdc_matrix(
        voc["diag_voc"], voc["med_voc"]
    )
    voc["mdc_matrix"] = mdc_matrix
    print(
        f"    MDC matrix shape: {mdc_matrix.shape}, rules matched: {matched_rules}, pairs: {matched_pairs}"
    )

    # 2. Load CSV
    print(f"\n[2] Loading CSV from {INPUT_CSV}")
    rows = load_csv(INPUT_CSV)
    print(f"    Rows: {len(rows)}")
    patients = set(r["SUBJECT_ID"] for r in rows)
    print(f"    Unique patients: {len(patients)}")

    # 3. Build LEADER-format records
    print("\n[3] Building LEADER-format records...")
    llm_data = build_patient_records(rows, voc)
    print(f"    Records built: {len(llm_data)}")

    # 4. Split data
    print(
        f"\n[4] Splitting data ({TRAIN_RATIO}/{VAL_RATIO}/{1 - TRAIN_RATIO - VAL_RATIO})..."
    )
    train, val, test = split_data(llm_data, TRAIN_RATIO, VAL_RATIO, SEED)
    print(f"    Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # 5. Create output directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FULL_DIR, exist_ok=True)

    # 6. Save JSONL files
    print(f"\n[5] Saving JSONL files to {DATA_DIR}")
    save_jsonl(os.path.join(DATA_DIR, "train_leader.json"), train)
    save_jsonl(os.path.join(DATA_DIR, "val_leader.json"), val)
    save_jsonl(os.path.join(DATA_DIR, "test_leader.json"), test)

    # 7. Build and save profile_dict.json
    print("\n[6] Building profile_dict.json...")
    profile_dict = build_profile_dict(rows)
    with open(os.path.join(DATA_DIR, "profile_dict.json"), "w") as f:
        json.dump(profile_dict, f, indent=2)
    print(f"    GENDER values: {list(profile_dict['word2idx']['GENDER'].keys())}")
    print(f"    AGE values: {len(profile_dict['word2idx']['AGE'])} unique ages")

    # 8. Copy pre-built artifacts
    print(f"\n[7] Copying artifacts to {DATA_DIR}")
    artifacts = [
        ("voc_final.pkl", DATA_DIR),
        ("ddi_A_final.pkl", FULL_DIR),
        ("ehr_adj_final.pkl", FULL_DIR),
    ]
    for fname, dest in artifacts:
        src = os.path.join(OUTPUT_ARTIFACTS, fname)
        dst = os.path.join(dest, fname)
        shutil.copy2(src, dst)
        print(f"    Copied {fname} → {dest}")

    # 9. Verify
    print("\n[8] Verification:")
    ddi = dill.load(open(os.path.join(FULL_DIR, "ddi_A_final.pkl"), "rb"))
    print(f"    DDI matrix shape: {ddi.shape}, pairs: {np.count_nonzero(ddi) // 2}")
    train_check = []
    with open(os.path.join(DATA_DIR, "train_leader.json")) as f:
        for line in f:
            train_check.append(json.loads(line))
    print(f"    Train file: {len(train_check)} records")
    print(f"    Sample drug_code: {train_check[0]['drug_code'][:5]}")
    print(f"    Sample prompt (first 150 chars): {train_check[0]['input'][:150]}")

    print("\n" + "=" * 60)
    print("Phase 1 COMPLETE!")
    print(f"Data ready at: {DATA_DIR}")
    print(f"Run with: --data_dir {DATA_DIR}/ --train_file leader")
    print("=" * 60)


if __name__ == "__main__":
    main()
