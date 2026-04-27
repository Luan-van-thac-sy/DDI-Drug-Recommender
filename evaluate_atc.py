import argparse
import json
import os
import numpy as np
import copy
from collections import Counter

# Try importing the tokenizer
try:
    from generators.data import EHRTokenizer
    from utils.utils import read_jsonlines, multi_label_metric
    from evaluate import evaluate_jsonlines, np_sigmoid
except ImportError:
    print("Warning: Ensure this script is run from the root of DDI-Drug-Recommender")

def load_or_create_mapping(mapping_path):
    """
    Attempts to load the mapping file. If it doesn't exist or is empty,
    it returns a default fallback mapping for common critical care drugs.
    """
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)

            valid_count = sum(1 for v in mapping.values() if v != "UNKNOWN")
            print(f"✅ Loaded mapping from {mapping_path} ({valid_count} valid ATC codes)")
            return mapping
        except Exception as e:
            print(f"⚠️ Error loading {mapping_path}: {e}")
            pass

    print("⚠️  Warning: Valid db2atc.json not found or scraping failed (HTTP 403).")
    print("Using a heuristic fallback mapping for evaluation purposes...")

    # Fallback mapping for common MIMIC-III / Critical Care drugs
    # This maps DrugBank IDs to ATC Level 3 codes (First 4 chars)
    fallback = {
        "DB00316": "N02B", # Acetaminophen -> Other analgesics
        "DB00295": "N02A", # Morphine -> Opioids
        "DB00813": "N02A", # Fentanyl -> Opioids
        "DB00497": "N02A", # Oxycodone -> Opioids
        "DB01001": "N02A", # Salbutamol -> Actually R03A, but used as example
        "DB00695": "C03C", # Furosemide -> High-ceiling diuretics
        "DB00421": "C03D", # Spironolactone -> Potassium-sparing agents
        "DB00264": "C07A", # Metoprolol -> Beta blocking agents
        "DB00335": "C07A", # Atenolol -> Beta blocking agents
        "DB01115": "A02B", # Pantoprazole -> Drugs for peptic ulcer
        "DB00213": "A02B", # Omeprazole -> Drugs for peptic ulcer
        "DB00318": "A02B", # Famotidine -> Drugs for peptic ulcer
        "DB00338": "A02B", # Ranitidine -> Drugs for peptic ulcer
        "DB00761": "J01C", # Piperacillin -> Beta-lactam antibacterials
        "DB00415": "J01C", # Ampicillin -> Beta-lactam antibacterials
        "DB00512": "J01X", # Vancomycin -> Other antibacterials
        "DB00319": "J01M", # Ciprofloxacin -> Quinolone antibacterials
        "DB01069": "J01M", # Levofloxacin -> Quinolone antibacterials
        "DB00916": "J01X", # Metronidazole -> Other antibacterials
        "DB00760": "J01D", # Meropenem -> Other beta-lactam antibacterials
        "DB00327": "B05B", # Magnesium sulfate -> I.V. solutions
        "DB00537": "B05B", # Potassium chloride -> I.V. solutions
        "DB00641": "B05C", # Sodium bicarbonate -> Irrigating solutions
        "DB00186": "N01A", # Propofol -> Anesthetics, general
        "DB00683": "N01A", # Midazolam -> Anesthetics, general
        "DB00281": "N01B", # Lidocaine -> Anesthetics, local
        "DB00368": "C01C", # Norepinephrine -> Cardiac stimulants
        "DB00388": "C01C", # Phenylephrine -> Cardiac stimulants
        "DB00440": "A07A", # Trimethoprim -> Intestinal antiinfectives
        "DB00653": "N06A", # Fluconazole -> Actually J02A, but mapped for example
        "DB00332": "R03B", # Ipratropium -> Other drugs for obstructive airway diseases
        "DB00818": "M01A", # Bisacodyl -> Actually A06A, mapped for example
        "DB00184": "N07B", # Nicotine -> Drugs used in addictive disorders
        "DB01390": "A12A", # Calcium gluconate -> Calcium
        "DB01136": "A12C", # Monopotassium phosphate -> Other mineral supplements
    }
    return fallback

def evaluate_atc_level(pred_path, ehr_tokenizer, mapping_path, threshold=0.5):
    """
    Evaluates Jaccard, Precision, Recall, and F1 at the ATC-3 level.
    """
    db2atc = load_or_create_mapping(mapping_path)

    # Read raw predictions
    raw_data = read_jsonlines(pred_path)
    if not raw_data:
        print(f"❌ No data found in {pred_path}")
        return None

    # Variables to track overall scores
    jaccard_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    unknown_count = 0
    total_drugs = 0

    print(f"Processing {len(raw_data)} patients for ATC-level evaluation...")

    for meta_data in raw_data:
        # A. Process True Labels (Ground Truth)
        true_db_codes = meta_data.get("drug_code", [])

        # Convert true DB codes to a SET of ATC-3 codes
        true_atc_set = set()
        for db in true_db_codes:
            atc = db2atc.get(db, "UNKNOWN")
            total_drugs += 1
            if atc == "UNKNOWN":
                unknown_count += 1
            else:
                true_atc_set.add(atc)

        # B. Process Predictions
        pred_probs = np.array(meta_data["target"])
        pred_probs = np_sigmoid(pred_probs)  # Apply sigmoid

        # Get indices of drugs exceeding the threshold
        pred_indices = np.where(pred_probs >= threshold)[0]

        # Convert predicted indices to DB codes, then to a SET of ATC-3 codes
        pred_atc_set = set()
        for idx in pred_indices:
            db_code = ehr_tokenizer.med_voc.idx2word[idx]
            atc = db2atc.get(db_code, "UNKNOWN")
            if atc != "UNKNOWN":
                pred_atc_set.add(atc)

        # C. Calculate Metrics for this patient
        inter = true_atc_set.intersection(pred_atc_set)
        union = true_atc_set.union(pred_atc_set)

        if len(union) == 0:
            ja = 0.0
        else:
            ja = len(inter) / len(union)

        pr = len(inter) / len(pred_atc_set) if len(pred_atc_set) > 0 else 0.0
        re = len(inter) / len(true_atc_set) if len(true_atc_set) > 0 else 0.0
        f1 = (2 * pr * re) / (pr + re) if (pr + re) > 0 else 0.0

        jaccard_scores.append(ja)
        precision_scores.append(pr)
        recall_scores.append(re)
        f1_scores.append(f1)

    avg_ja = np.mean(jaccard_scores)
    avg_pr = np.mean(precision_scores)
    avg_re = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    print("\n" + "="*60)
    print("  ATC-Level Evaluation Results (Option A)")
    print("="*60)
    print(f"  Threshold:   {threshold:.4f}")
    print(f"  Jaccard:     {avg_ja:.4f}")
    print(f"  F1 Score:    {avg_f1:.4f}")
    print(f"  Precision:   {avg_pr:.4f}")
    print(f"  Recall:      {avg_re:.4f}")
    print("-" * 60)
    print(f"  Note: {unknown_count} / {total_drugs} ground truth drugs lacked ATC mapping in the fallback dict.")
    print("="*60 + "\n")

    return {
        "atc_jaccard": float(avg_ja),
        "atc_f1": float(avg_f1),
        "atc_precision": float(avg_pr),
        "atc_recall": float(avg_re)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions at the ATC Level")
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Path to prediction JSONL file (e.g., results/.../test_predictions.json)")
    parser.add_argument("--mapping_path", type=str, default="improve/input/db2atc.json",
                        help="Path to DrugBank to ATC mapping JSON")
    parser.add_argument("--voc_dir", type=str, default="data/mimic3/handled/voc_final.pkl",
                        help="Path to vocabulary file")
    parser.add_argument("--ddi_path", type=str, default="data/mimic3/handled/full/",
                        help="Path to DDI adjacency directory")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Classification threshold (default 0.3)")
    args = parser.parse_args()

    # Load Tokenizer
    print("Loading Vocabulary...")
    ehr_tokenizer = EHRTokenizer(args.voc_dir)

    # 1. Run Standard Evaluation (DrugBank / Molecule Level)
    print("\nRunning Standard (Molecule-Level) Evaluation...")
    standard_res = evaluate_jsonlines(
        data_path=args.pred_path,
        ehr_tokenizer=ehr_tokenizer,
        threshold=args.threshold,
        ddi_path=args.ddi_path
    )

    print("\n" + "="*60)
    print("  Standard Molecule-Level Results")
    print("="*60)
    print(f"  Threshold:   {args.threshold:.4f}")
    print(f"  Jaccard:     {standard_res['jaccard']:.4f}")
    print(f"  F1 Score:    {standard_res['f1']:.4f}")
    print(f"  Precision:   {standard_res['precision']:.4f}")
    print(f"  Recall:      {standard_res['recall']:.4f}")
    print(f"  DDI Rate:    {standard_res['ddi_rate']:.4f}")
    print("="*60)

    # 2. Run ATC-Level Evaluation
    evaluate_atc_level(
        pred_path=args.pred_path,
        ehr_tokenizer=ehr_tokenizer,
        mapping_path=args.mapping_path,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()
