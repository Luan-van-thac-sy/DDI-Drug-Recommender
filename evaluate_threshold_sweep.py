import argparse
import os
import json
import numpy as np
from evaluate import evaluate_jsonlines
from generators.data import EHRTokenizer


def frange(start, stop, step):
    vals = []
    x = start
    while x <= stop + 1e-12:
        vals.append(round(x, 6))
        x += step
    return vals


def avg_pred_count(drug_code_results):
    n = len(drug_code_results["pred_drug_codes"])
    if n == 0:
        return 0.0
    return sum(len(x) for x in drug_code_results["pred_drug_codes"]) / n


def avg_true_count(drug_code_results):
    n = len(drug_code_results["true_drug_codes"])
    if n == 0:
        return 0.0
    return sum(len(x) for x in drug_code_results["true_drug_codes"]) / n


def main():
    parser = argparse.ArgumentParser(description="Sweep sigmoid threshold for MedRec evaluation.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to test_predictions.json (jsonl).")
    parser.add_argument("--voc_dir", type=str, required=True, help="Path to voc_final.pkl.")
    parser.add_argument("--ddi_path", type=str, default="data/mimic4/handled/full/", help="Dir containing ddi_A_final.pkl.")
    parser.add_argument("--start", type=float, default=0.5)
    parser.add_argument("--stop", type=float, default=0.9)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--select_by", type=str, default="jaccard", choices=["jaccard", "f1"])
    parser.add_argument("--out_json", type=str, default=None, help="Optional output json summary.")
    args = parser.parse_args()

    if not os.path.exists(args.pred_path):
        raise FileNotFoundError(f"pred_path not found: {args.pred_path}")
    if not os.path.exists(args.voc_dir):
        raise FileNotFoundError(f"voc_dir not found: {args.voc_dir}")

    tokenizer = EHRTokenizer(args.voc_dir)
    thresholds = frange(args.start, args.stop, args.step)

    rows = []
    print("\n=== Threshold Sweep ===")
    for th in thresholds:
        ja, prauc, avg_p, avg_r, avg_f1, drug_codes = evaluate_jsonlines(
            args.pred_path,
            tokenizer,
            threshold=th,
            ddi_path=args.ddi_path,
        )
        ap = avg_pred_count(drug_codes)
        at = avg_true_count(drug_codes)
        row = {
            "threshold": th,
            "jaccard": float(ja),
            "prauc": float(prauc),
            "avg_precision": float(avg_p),
            "avg_recall": float(avg_r),
            "avg_f1": float(avg_f1),
            "avg_pred_drugs": float(ap),
            "avg_true_drugs": float(at),
        }
        rows.append(row)
        print(
            f"th={th:.2f} | ja={ja:.4f} | f1={avg_f1:.4f} | prauc={prauc:.4f} | "
            f"avg_pred={ap:.2f} | avg_true={at:.2f}"
        )

    key = "jaccard" if args.select_by == "jaccard" else "avg_f1"
    best = max(rows, key=lambda x: x[key])

    print("\n=== Best Threshold ===")
    print(
        f"select_by={args.select_by} | threshold={best['threshold']:.2f} | "
        f"jaccard={best['jaccard']:.4f} | f1={best['avg_f1']:.4f} | "
        f"prauc={best['prauc']:.4f} | avg_pred={best['avg_pred_drugs']:.2f} | "
        f"avg_true={best['avg_true_drugs']:.2f}"
    )

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"select_by": args.select_by, "best": best, "all": rows}, f, indent=2)
        print(f"Saved summary to {args.out_json}")


if __name__ == "__main__":
    main()
