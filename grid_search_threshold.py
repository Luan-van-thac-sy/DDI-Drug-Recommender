"""
Grid Search for Classification Threshold

Scans a range of thresholds on an existing test_predictions.json
(output of main_llm_cls.py) and finds the best threshold for each metric.

Usage:
    python grid_search_threshold.py \
        --pred_path results/0105/test_predictions.json \
        --voc_dir data/mimic3/handled/voc_final.pkl \
        --ddi_path data/mimic3/handled/full/ \
        --metric jaccard \
        --output results/0105/threshold_search.json
"""

import argparse
import json
import os

import numpy as np

from evaluate import evaluate_jsonlines
from generators.data import EHRTokenizer


def grid_search_threshold(
    pred_path: str,
    ehr_tokenizer: EHRTokenizer,
    ddi_path: str,
    thresholds: list[float],
    primary_metric: str = "jaccard",
) -> dict:
    """
    Run evaluate_jsonlines for each threshold and collect results.

    Args:
        pred_path:       Path to test_predictions.json
        ehr_tokenizer:   EHRTokenizer instance
        ddi_path:        Path to DDI adjacency directory
        thresholds:      List of threshold values to search
        primary_metric:  Metric to optimise ('jaccard', 'f1', 'prauc')

    Returns:
        dict with keys:
          - 'all_results'  : {threshold -> metrics}
          - 'best_threshold': float
          - 'best_metrics'  : dict
    """
    all_results: dict[float, dict] = {}

    print(f"\n{'='*60}")
    print(f"  Grid Search: {len(thresholds)} thresholds")
    print(f"  Primary metric: {primary_metric}")
    print(f"{'='*60}")
    print(f"{'Threshold':>10} {'Jaccard':>9} {'F1':>9} {'PRAUC':>9} "
          f"{'Precision':>10} {'Recall':>9} {'DDI Rate':>10}")
    print(f"{'-'*60}")

    for t in thresholds:
        res = evaluate_jsonlines(
            pred_path,
            ehr_tokenizer,
            threshold=t,
            ddi_path=ddi_path,
        )
        # Strip heavy drug_code_results to keep dict lean
        res_lean = {k: v for k, v in res.items()
                    if k not in ("drug_code_results", "top_ddi_pairs")}
        all_results[t] = res_lean

        print(f"{t:>10.2f} {res['jaccard']:>9.4f} {res['f1']:>9.4f} "
              f"{res['prauc']:>9.4f} {res['precision']:>10.4f} "
              f"{res['recall']:>9.4f} {res['ddi_rate']:>10.4f}")

    print(f"{'='*60}\n")

    # Find best threshold by primary metric
    best_t = max(all_results, key=lambda t: all_results[t][primary_metric])
    best_metrics = all_results[best_t]

    print(f"✅ Best threshold for '{primary_metric}': {best_t:.2f}")
    print(f"   Jaccard  : {best_metrics['jaccard']:.4f}")
    print(f"   F1       : {best_metrics['f1']:.4f}")
    print(f"   PRAUC    : {best_metrics['prauc']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall   : {best_metrics['recall']:.4f}")
    print(f"   DDI Rate : {best_metrics['ddi_rate']:.4f}\n")

    return {
        "all_results": {str(k): v for k, v in all_results.items()},  # JSON-safe keys
        "best_threshold": best_t,
        "best_metrics": best_metrics,
        "primary_metric": primary_metric,
    }


def main():
    parser = argparse.ArgumentParser(description="Grid search for best classification threshold")
    parser.add_argument("--pred_path", type=str,
                        default="results/0105/test_predictions.json",
                        help="Path to test_predictions.json")
    parser.add_argument("--voc_dir", type=str,
                        default="data/mimic3/handled/voc_final.pkl",
                        help="Path to vocabulary file")
    parser.add_argument("--ddi_path", type=str,
                        default="data/mimic3/handled/full/",
                        help="Path to DDI adjacency directory")
    parser.add_argument("--metric", type=str, default="jaccard",
                        choices=["jaccard", "f1", "prauc", "precision", "recall"],
                        help="Primary metric to optimise")
    parser.add_argument("--min_t", type=float, default=0.1,
                        help="Minimum threshold to search (default: 0.1)")
    parser.add_argument("--max_t", type=float, default=0.6,
                        help="Maximum threshold to search (default: 0.6)")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Step size between thresholds (default: 0.05)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path to save JSON results (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.pred_path):
        print(f"❌ Prediction file not found: {args.pred_path}")
        return

    ehr_tokenizer = EHRTokenizer(args.voc_dir)

    thresholds = list(np.arange(args.min_t, args.max_t + 1e-9, args.step))
    thresholds = [round(t, 4) for t in thresholds]

    search_results = grid_search_threshold(
        pred_path=args.pred_path,
        ehr_tokenizer=ehr_tokenizer,
        ddi_path=args.ddi_path,
        thresholds=thresholds,
        primary_metric=args.metric,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(search_results, f, indent=2, default=str)
        print(f"📄 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
