# here put the import lib
import argparse
import copy
import csv
import json
import os
import pickle
import numpy as np
from utils.utils import read_jsonlines, multi_label_metric, ddi_rate_score, multi_test
from generators.data import Voc, EHRTokenizer


def evaluate_jsonlines(data_path, ehr_tokenizer, threshold=0.5, ddi_path='./data/mimic4/handled/full/'):

    pred_data_prob, pred_data = [], []
    true_data = np.zeros((len(read_jsonlines(data_path)), len(ehr_tokenizer.med_voc.word2idx)))
    seq_len = []
    pred_label = []

    # Collect drug codes
    all_true_drug_codes = []      # Ground truth drug codes
    all_pred_drug_codes = []      # Predicted drug codes
    all_subject_ids = []          # Subject IDs for tracking

    for row, meta_data in enumerate(read_jsonlines(data_path)):

        # noramlize the predicted scores by sigmoid, and get the prob
        meta_pred_data_prob = np.array(meta_data["target"])
        pred_data_prob.append(np_sigmoid(meta_pred_data_prob))

        # transform y to 0-1 by threshold
        meta_pred_data = copy.deepcopy(np_sigmoid(meta_pred_data_prob))
        meta_pred_data[meta_pred_data>=threshold] = 1
        meta_pred_data[meta_pred_data<threshold] = 0
        pred_data.append(meta_pred_data)

        # get the true data
        true_index = ehr_tokenizer.convert_med_tokens_to_ids(meta_data["drug_code"])
        true_data[row][true_index] = 1

        # Save true drug codes
        all_true_drug_codes.append(meta_data["drug_code"])
        all_subject_ids.append(meta_data.get("subject_id", row))

        seq_len.append(int(meta_data["input"].split("The patient has ")[1].split(" times ICU visits.")[0]))

        # prepare the labels for DDI calculation
        meta_label = np.where(meta_pred_data == 1)[0]
        pred_label.append([sorted(meta_label)])

        # Convert predicted indices back to drug codes
        pred_drug_codes = [ehr_tokenizer.med_voc.idx2word[idx] for idx in meta_label]
        all_pred_drug_codes.append(pred_drug_codes)

    ja, prauc, avg_p, avg_r, avg_f1, mean, std = multi_label_metric(true_data,
                                                         np.array(pred_data),
                                                         np.array(pred_data_prob))
    ddi_adj = pickle.load(open(os.path.join(ddi_path, 'ddi_A_final.pkl'), 'rb'))
    ddi = ddi_rate_score(pred_label, ddi_adj)

    print('\nJaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, DDI_rate: {:.4}\n'.format(
          ja, prauc, avg_p, avg_r, avg_f1, ddi
    ))
    # print("10-rounds PRAUC: %.5f + %.5f" % (mean[0], std[0]))
    # print("10-rounds Jaccard: %.5f + %.5f" % (mean[1], std[1]))
    # print("10-rounds F1-score: %.5f + %.5f" % (mean[2], std[2]))

    # seq_len = np.array(seq_len)
    # pred_data = np.array(pred_data)
    # pred_data_prob = np.array(pred_data_prob)
    # single_index = (seq_len == 0)
    # multi_index = (seq_len >= 1)
    # acc_container = {}
    # s_ja, s_prauc, s_avg_p, s_avg_r, s_avg_f1, s_mean, s_std = multi_label_metric(true_data[single_index],
    #                                                                pred_data[single_index],
    #                                                                pred_data_prob[single_index])
    # m_ja, m_prauc, m_avg_p, m_avg_r, m_avg_f1, m_mean, m_std = multi_label_metric(true_data[multi_index],
    #                                                                pred_data[multi_index],
    #                                                                pred_data_prob[multi_index])
    # acc_container['single-jaccard'] = s_ja
    # acc_container['single-f1'] = s_avg_f1
    # acc_container['single-prauc'] = s_prauc
    # acc_container['multiple-jaccard'] = m_ja
    # acc_container['multiple-f1'] = m_avg_f1
    # acc_container['multiple-prauc'] = m_prauc

    # for k, v in acc_container.items():
    #     print('%-10s : %-10.4f' % (k, v))

    # print("Single-visit 10-rounds PRAUC: %.5f + %.5f" % (s_mean[0], s_std[0]))
    # print("Single-vist 10-rounds Jaccard: %.5f + %.5f" % (s_mean[1], s_std[1]))
    # print("Single-visit 10-rounds F1-score: %.5f + %.5f" % (s_mean[2], s_std[2]))
    # print("Multi-visit 10-rounds PRAUC: %.5f + %.5f" % (m_mean[0], m_std[0]))
    # print("Multi-vist 10-rounds Jaccard: %.5f + %.5f" % (m_mean[1], m_std[1]))
    # print("Multi-visit 10-rounds F1-score: %.5f + %.5f" % (m_mean[2], m_std[2]))

    # Create dictionary containing drug codes
    drug_code_results = {
        'true_drug_codes': all_true_drug_codes,
        'pred_drug_codes': all_pred_drug_codes,
        'subject_ids': all_subject_ids
    }

    return ja, prauc, avg_p, avg_r, avg_f1, drug_code_results


def save_drug_codes_comparison(drug_code_results, output_dir):
    """Write drug_codes_comparison.json + .csv (same layout as main_llm_cls predict)."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "drug_codes_comparison.json")
    csv_path = os.path.join(output_dir, "drug_codes_comparison.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(drug_code_results, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "true_drug_codes", "pred_drug_codes"])
        for i in range(len(drug_code_results["subject_ids"])):
            subject_id = drug_code_results["subject_ids"][i]
            true_codes = "; ".join(sorted(str(x) for x in drug_code_results["true_drug_codes"][i]))
            pred_codes = "; ".join(sorted(str(x) for x in drug_code_results["pred_drug_codes"][i]))
            writer.writerow([subject_id, true_codes, pred_codes])

    n = len(drug_code_results["subject_ids"])
    avg_true = sum(len(codes) for codes in drug_code_results["true_drug_codes"]) / max(n, 1)
    avg_pred = sum(len(codes) for codes in drug_code_results["pred_drug_codes"]) / max(n, 1)

    print("\n✓ Drug codes saved to:")
    print(f"   - JSON: {json_path}")
    print(f"   - CSV:  {csv_path}")
    print(f"   - Total samples: {n}")
    print(f"   - Average true drugs per patient: {avg_true:.2f}")
    print(f"   - Average predicted drugs per patient: {avg_pred:.2f}")


def np_sigmoid(x):
    # sigmoid function using numpy
    return 1 / (1+np.exp(-x))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate test_predictions.json and export drug code comparison files.")
    parser.add_argument(
        "--pred_path",
        type=str,
        default="./results/0105/test_predictions.json",
        help="JSONL from main_llm_cls predict (must contain target logits + drug_code).",
    )
    parser.add_argument(
        "--voc_dir",
        type=str,
        default="data/mimic4/handled/voc_final.pkl",
        help="Path to voc_final.pkl (must match training/inference vocabulary).",
    )
    parser.add_argument("--threshold", type=float, default=0.3, help="Sigmoid threshold for binary predictions.")
    parser.add_argument(
        "--ddi_path",
        type=str,
        default="data/mimic4/handled/full/",
        help="Directory containing ddi_A_final.pkl.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write drug_codes_comparison.{json,csv}. Default: directory of --pred_path.",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(args.pred_path)) or "."

    ehr_tokenizer = EHRTokenizer(args.voc_dir)
    ja, prauc, avg_p, avg_r, avg_f1, drug_code_results = evaluate_jsonlines(
        args.pred_path,
        ehr_tokenizer,
        threshold=args.threshold,
        ddi_path=args.ddi_path,
    )
    save_drug_codes_comparison(drug_code_results, out_dir)
