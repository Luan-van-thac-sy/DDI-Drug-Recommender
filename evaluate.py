"""
Phase 6: Evaluation & Comparison for DDI-Aware LEADER

Features:
  - Standard metrics: Jaccard, F1, PRAUC, DDI rate
  - Single-visit vs Multi-visit breakdown
  - Per-pair DDI analysis (which drug pairs cause most violations)
  - MDC rate calculation
  - Comparison table across multiple runs
  - JSON result export for plotting
"""
import argparse
import copy
import glob
import json
import os
import pickle
from collections import Counter

import numpy as np

from utils.utils import read_jsonlines, multi_label_metric, ddi_rate_score, multi_test
from generators.data import Voc, EHRTokenizer


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def evaluate_jsonlines(data_path, ehr_tokenizer, threshold=0.5,
                       ddi_path='./data/mimic3/handled/full/'):
    """Evaluate predictions from a JSONL file. Returns full metrics dict."""

    pred_data_prob, pred_data = [], []
    raw_data = read_jsonlines(data_path)
    true_data = np.zeros((len(raw_data), len(ehr_tokenizer.med_voc.word2idx)))
    seq_len = []
    pred_label = []
    all_true_drug_codes = []
    all_pred_drug_codes = []
    all_subject_ids = []

    for row, meta_data in enumerate(raw_data):
        meta_pred_data_prob = np.array(meta_data["target"])
        pred_data_prob.append(np_sigmoid(meta_pred_data_prob))

        meta_pred_data = copy.deepcopy(np_sigmoid(meta_pred_data_prob))
        meta_pred_data[meta_pred_data >= threshold] = 1
        meta_pred_data[meta_pred_data < threshold] = 0
        pred_data.append(meta_pred_data)

        true_index = ehr_tokenizer.convert_med_tokens_to_ids(meta_data["drug_code"])
        true_data[row][true_index] = 1

        all_true_drug_codes.append(meta_data["drug_code"])
        all_subject_ids.append(meta_data.get("subject_id", row))

        try:
            visits = int(meta_data["input"].split("The patient has ")[1].split(" times ICU visits.")[0])
        except (IndexError, ValueError):
            visits = 0
        seq_len.append(visits)

        meta_label = np.where(meta_pred_data == 1)[0]
        pred_label.append([sorted(meta_label)])
        pred_drug_codes = [ehr_tokenizer.med_voc.idx2word[idx] for idx in meta_label]
        all_pred_drug_codes.append(pred_drug_codes)

    # Overall metrics
    ja, prauc, avg_p, avg_r, avg_f1, mean, std = multi_label_metric(
        true_data, np.array(pred_data), np.array(pred_data_prob))

    ddi_adj = pickle.load(open(os.path.join(ddi_path, 'ddi_A_final.pkl'), 'rb'))
    ddi = ddi_rate_score(pred_label, ddi_adj)

    results = {
        'jaccard': float(ja),
        'prauc': float(prauc),
        'precision': float(avg_p),
        'recall': float(avg_r),
        'f1': float(avg_f1),
        'ddi_rate': float(ddi),
        'num_samples': len(raw_data),
    }

    # Single-visit vs Multi-visit breakdown
    seq_len = np.array(seq_len)
    pred_data_arr = np.array(pred_data)
    pred_prob_arr = np.array(pred_data_prob)
    single_idx = (seq_len == 0)
    multi_idx = (seq_len >= 1)

    if single_idx.sum() > 0:
        s_ja, s_prauc, _, _, s_f1, _, _ = multi_label_metric(
            true_data[single_idx], pred_data_arr[single_idx], pred_prob_arr[single_idx])
        results['single_jaccard'] = float(s_ja)
        results['single_prauc'] = float(s_prauc)
        results['single_f1'] = float(s_f1)
        results['single_count'] = int(single_idx.sum())

    if multi_idx.sum() > 0:
        m_ja, m_prauc, _, _, m_f1, _, _ = multi_label_metric(
            true_data[multi_idx], pred_data_arr[multi_idx], pred_prob_arr[multi_idx])
        results['multi_jaccard'] = float(m_ja)
        results['multi_prauc'] = float(m_prauc)
        results['multi_f1'] = float(m_f1)
        results['multi_count'] = int(multi_idx.sum())

    # Per-pair DDI analysis
    ddi_pair_counts = Counter()
    ddi_total_pairs = 0
    for patient_preds in pred_label:
        for adm in patient_preds:
            for i, med_i in enumerate(adm):
                for j, med_j in enumerate(adm):
                    if j <= i:
                        continue
                    ddi_total_pairs += 1
                    if ddi_adj[med_i, med_j] == 1 or ddi_adj[med_j, med_i] == 1:
                        pair = tuple(sorted([med_i, med_j]))
                        ddi_pair_counts[pair] += 1

    # Convert to drug names for readability
    idx2word = ehr_tokenizer.med_voc.idx2word
    top_ddi_pairs = []
    for (di, dj), count in ddi_pair_counts.most_common(20):
        name_i = idx2word.get(di, f"Drug_{di}")
        name_j = idx2word.get(dj, f"Drug_{dj}")
        top_ddi_pairs.append({
            'drug_a': name_i, 'drug_b': name_j,
            'drug_a_idx': int(di), 'drug_b_idx': int(dj),
            'count': count
        })
    results['top_ddi_pairs'] = top_ddi_pairs
    results['total_ddi_violations'] = sum(ddi_pair_counts.values())
    results['total_drug_pairs'] = ddi_total_pairs

    # MDC rate (if MDC matrix available)
    mdc_path = os.path.join(ddi_path, '..', 'voc_final.pkl')
    if os.path.exists(mdc_path):
        try:
            from utils.mdc_context import build_mdc_matrix
            voc = pickle.load(open(mdc_path, 'rb'))
            mdc_matrix, _, _ = build_mdc_matrix(voc['diag_voc'], voc['med_voc'])

            # Calculate MDC rate from predictions
            # For each patient, check if any predicted drug is contraindicated for their diagnoses
            mdc_violations = 0
            mdc_total = 0
            for row_idx, patient_preds in enumerate(pred_label):
                for adm in patient_preds:
                    true_diags = np.where(true_data[row_idx] == 1)[0]  # not ideal, but approximate
                    for med_idx in adm:
                        mdc_total += 1
                        # Check all diagnoses in training data for this patient
                        # Use the raw data records if available
                        if row_idx < len(raw_data) and 'records' in raw_data[row_idx]:
                            diag_codes = raw_data[row_idx]['records'].get('diagnosis', [[]])[-1]
                            for diag_code in diag_codes:
                                if diag_code in voc['diag_voc'].word2idx:
                                    diag_idx = voc['diag_voc'].word2idx[diag_code]
                                    if diag_idx < mdc_matrix.shape[0] and med_idx < mdc_matrix.shape[1]:
                                        if mdc_matrix[diag_idx, med_idx] != 0:
                                            mdc_violations += 1
                                            break

            results['mdc_rate'] = float(mdc_violations / max(mdc_total, 1))
            results['mdc_violations'] = mdc_violations
            results['mdc_total_prescriptions'] = mdc_total
        except Exception as e:
            results['mdc_error'] = str(e)

    results['drug_code_results'] = {
        'true_drug_codes': all_true_drug_codes,
        'pred_drug_codes': all_pred_drug_codes,
        'subject_ids': all_subject_ids
    }

    return results


def print_results(results, name=""):
    """Pretty-print evaluation results."""
    header = f"=== Results: {name} ===" if name else "=== Evaluation Results ==="
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'=' * len(header)}")

    print(f"\n  Effectiveness Metrics:")
    print(f"    Jaccard:   {results['jaccard']:.4f}")
    print(f"    PRAUC:     {results['prauc']:.4f}")
    print(f"    F1:        {results['f1']:.4f}")
    print(f"    Precision: {results['precision']:.4f}")
    print(f"    Recall:    {results['recall']:.4f}")

    print(f"\n  Safety Metrics:")
    print(f"    DDI Rate:  {results['ddi_rate']:.4f}")
    if 'mdc_rate' in results:
        print(f"    MDC Rate:  {results['mdc_rate']:.4f}")

    if 'single_jaccard' in results:
        print(f"\n  Single-visit ({results.get('single_count', '?')} patients):")
        print(f"    Jaccard: {results['single_jaccard']:.4f}, "
              f"PRAUC: {results['single_prauc']:.4f}, "
              f"F1: {results['single_f1']:.4f}")

    if 'multi_jaccard' in results:
        print(f"\n  Multi-visit ({results.get('multi_count', '?')} patients):")
        print(f"    Jaccard: {results['multi_jaccard']:.4f}, "
              f"PRAUC: {results['multi_prauc']:.4f}, "
              f"F1: {results['multi_f1']:.4f}")

    if results.get('top_ddi_pairs'):
        print(f"\n  Top DDI Violating Pairs ({results['total_ddi_violations']} total violations):")
        for i, pair in enumerate(results['top_ddi_pairs'][:10]):
            print(f"    {i+1}. {pair['drug_a']} + {pair['drug_b']}: {pair['count']} times")

    print()


def compare_runs(results_dict):
    """Print comparison table across multiple runs."""
    if not results_dict:
        print("No results to compare.")
        return

    print("\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Run':<25} {'Jaccard':>8} {'PRAUC':>8} {'F1':>8} {'DDI Rate':>10} {'MDC Rate':>10}")
    print("-" * 90)
    for name, res in results_dict.items():
        mdc = f"{res.get('mdc_rate', 0):.4f}" if 'mdc_rate' in res else "N/A"
        print(f"{name:<25} {res['jaccard']:>8.4f} {res['prauc']:>8.4f} "
              f"{res['f1']:>8.4f} {res['ddi_rate']:>10.4f} {mdc:>10}")
    print("=" * 90)

    # Find best for each metric
    if len(results_dict) > 1:
        print("\n  Best Effectiveness (Jaccard):",
              max(results_dict.items(), key=lambda x: x[1]['jaccard'])[0])
        print("  Best Safety (DDI Rate):     ",
              min(results_dict.items(), key=lambda x: x[1]['ddi_rate'])[0])
        if any('mdc_rate' in r for r in results_dict.values()):
            mdc_runs = {k: v for k, v in results_dict.items() if 'mdc_rate' in v}
            if mdc_runs:
                print("  Best Safety (MDC Rate):     ",
                      min(mdc_runs.items(), key=lambda x: x[1]['mdc_rate'])[0])
    print()


def save_results(results, output_path):
    """Save results to JSON (exclude non-serializable drug_code_results)."""
    save_data = {k: v for k, v in results.items() if k != 'drug_code_results'}
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDI-Aware LEADER predictions")
    parser.add_argument("--pred_path", type=str, default=None,
                        help="Path to prediction JSONL file (single run evaluation)")
    parser.add_argument("--pred_dir", type=str, default=None,
                        help="Directory containing multiple prediction files (comparison mode)")
    parser.add_argument("--voc_dir", type=str, default="data/mimic3/handled/voc_final.pkl",
                        help="Path to vocabulary file")
    parser.add_argument("--ddi_path", type=str, default="data/mimic3/handled/full/",
                        help="Path to DDI adjacency directory")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Classification threshold")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    args = parser.parse_args()

    ehr_tokenizer = EHRTokenizer(args.voc_dir)

    if args.pred_path:
        # Single file evaluation
        print(f"Evaluating: {args.pred_path}")
        results = evaluate_jsonlines(args.pred_path, ehr_tokenizer,
                                     args.threshold, args.ddi_path)
        name = os.path.basename(os.path.dirname(args.pred_path))
        print_results(results, name)

        if args.output:
            save_results(results, args.output)

    elif args.pred_dir:
        # Multi-run comparison
        pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*/test_predictions.json")))
        if not pred_files:
            pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.json")))

        if not pred_files:
            print(f"No prediction files found in {args.pred_dir}")
            return

        all_results = {}
        for pf in pred_files:
            name = os.path.basename(os.path.dirname(pf))
            if name == os.path.basename(args.pred_dir):
                name = os.path.splitext(os.path.basename(pf))[0]
            print(f"\nEvaluating: {name} ({pf})")
            try:
                results = evaluate_jsonlines(pf, ehr_tokenizer,
                                             args.threshold, args.ddi_path)
                print_results(results, name)
                all_results[name] = results
            except Exception as e:
                print(f"  ERROR: {e}")

        compare_runs(all_results)

        if args.output:
            save_data = {k: {mk: mv for mk, mv in v.items() if mk != 'drug_code_results'}
                        for k, v in all_results.items()}
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            print(f"All results saved to {args.output}")

    else:
        # Default: try known prediction path
        default_path = "./results/0105/test_predictions.json"
        if os.path.exists(default_path):
            print(f"Evaluating default: {default_path}")
            results = evaluate_jsonlines(default_path, ehr_tokenizer,
                                         args.threshold, args.ddi_path)
            print_results(results, "default")
        else:
            print("Usage:")
            print("  Single run:  python evaluate.py --pred_path results/baseline/test_predictions.json")
            print("  Compare:     python evaluate.py --pred_dir results/")
            print(f"\nNo predictions found at {default_path}")


if __name__ == "__main__":
    main()
