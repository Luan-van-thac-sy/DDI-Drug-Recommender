"""
Evaluate student model with post-hoc DDI constraint.
Usage:
    python eval_student_posthoc.py \
        --model_path saved/mimic3/leader/distill-ddi/pytorch_model.bin \
        --data_dir data/mimic3/handled/ \
        --llm_path resources/biomistral-7b
"""
import torch
import numpy as np
import pickle
import json
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from generators.distill_generator import DistillEHRDataset
from models.LEADER import LEADER
from utils.config import BertConfig
from generators.data import EHRTokenizer
from utils.utils import read_jsonlines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to student pytorch_model.bin")
    parser.add_argument("--data_dir", type=str, default="data/mimic3/handled/")
    parser.add_argument("--llm_path", type=str, default="resources/biomistral-7b")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load tokenizers
    tokenizer = EHRTokenizer(f"{args.data_dir}/voc_final.pkl")
    profile_tokenizer = json.load(open(f"{args.data_dir}/profile_dict.json"))
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    llm_tokenizer.pad_token = llm_tokenizer.unk_token

    # Load test data
    test_data = read_jsonlines(f"{args.data_dir}/test_leader.json")

    # Build dataset
    model_args = argparse.Namespace(
        hidden_size=args.hidden_size, distill=True, d_loss='mse', alpha=0.1,
        align=True, align_weight=0.005, profile=True, prompt_num=1,
        num_trm_layers=1, ddi=True, ml_weight=0.05,
        mdc=False, mdc_weight=0.03, max_seq_length=100,
        max_record_num=10, max_source_length=1056,
        temperature=10, therhold=0.3, graph=False
    )
    test_dataset = DistillEHRDataset(test_data, tokenizer, profile_tokenizer, llm_tokenizer, model_args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Build model
    config = BertConfig(vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
    config.hidden_size = args.hidden_size
    model = LEADER(config, model_args, tokenizer, device, profile_tokenizer)

    # Load DDI adj
    ddi_adj = pickle.load(open(f"{args.data_dir}/full/ddi_A_final.pkl", "rb"))
    model.register_buffer('ddi_adj', torch.FloatTensor(ddi_adj))

    # Load saved weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Student model loaded from {args.model_path}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Predict
    all_preds = []
    all_labels = []
    for batch in tqdm(test_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            output = model(batch[0], batch[1], batch[2], batch[3], batch[4],
                           profile=batch[6], multi_label=batch[5])
            all_preds.append(torch.sigmoid(output).cpu().numpy())
            all_labels.append(batch[4].cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"Test samples: {len(all_preds)}")

    # === Results WITHOUT post-hoc ===
    print("\n" + "=" * 70)
    print("=== WITHOUT Post-hoc DDI Constraint ===")
    print("=" * 70)
    print(f"{'Thresh':>6} | {'Jaccard':>7} | {'F1':>6} | {'Prec':>6} | {'Recall':>6} | {'DDI':>6} | {'AvgDrugs':>8}")
    print("-" * 70)

    for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
        pred_bin = (all_preds > t).astype(int)
        jac, prec, rec, f1, ddi_rate, avg_d = calc_metrics(pred_bin, all_labels, ddi_adj)
        print(f"{t:>6.2f} | {jac:>7.4f} | {f1:>6.4f} | {prec:>6.4f} | {rec:>6.4f} | {ddi_rate:>6.4f} | {avg_d:>8.1f}")

    # === Results WITH post-hoc ===
    print("\n" + "=" * 70)
    print("=== WITH Post-hoc DDI Constraint ===")
    print("=" * 70)
    print(f"{'Thresh':>6} | {'Jaccard':>7} | {'F1':>6} | {'Prec':>6} | {'Recall':>6} | {'DDI':>6} | {'AvgDrugs':>8}")
    print("-" * 70)

    for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
        pred_bin = (all_preds > t).astype(int)
        pred_bin = apply_ddi_posthoc(pred_bin, all_preds, ddi_adj)
        jac, prec, rec, f1, ddi_rate, avg_d = calc_metrics(pred_bin, all_labels, ddi_adj)
        print(f"{t:>6.2f} | {jac:>7.4f} | {f1:>6.4f} | {prec:>6.4f} | {rec:>6.4f} | {ddi_rate:>6.4f} | {avg_d:>8.1f}")


def apply_ddi_posthoc(pred_bin, pred_probs, ddi_adj):
    """Greedy remove drugs causing most DDI violations."""
    pred_bin = pred_bin.copy()
    for i in range(len(pred_bin)):
        while True:
            drugs = np.where(pred_bin[i] == 1)[0]
            if len(drugs) < 2:
                break
            ddi_pairs = [(drugs[a], drugs[b])
                         for a in range(len(drugs)) for b in range(a + 1, len(drugs))
                         if ddi_adj[drugs[a]][drugs[b]] == 1]
            if not ddi_pairs:
                break
            counts = {}
            for d1, d2 in ddi_pairs:
                counts[d1] = counts.get(d1, 0) + 1
                counts[d2] = counts.get(d2, 0) + 1
            # Remove drug with most violations; break ties by lowest probability
            worst = max(counts, key=lambda d: (counts[d], -pred_probs[i][d]))
            pred_bin[i][worst] = 0
    return pred_bin


def calc_metrics(pred_bin, labels, ddi_adj):
    """Calculate Jaccard, F1, Precision, Recall, DDI rate, avg drugs."""
    jaccards = []
    precisions = []
    recalls = []
    for p, l in zip(pred_bin, labels):
        inter = (p * l).sum()
        union = ((p + l) > 0).sum()
        jaccards.append(inter / max(union, 1))
        precisions.append(inter / max(p.sum(), 1))
        recalls.append(inter / max(l.sum(), 1))

    jac = np.mean(jaccards)
    prec = np.mean(precisions)
    rec = np.mean(recalls)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    total_ddi, total_pairs = 0, 0
    for p in pred_bin:
        drugs = np.where(p == 1)[0]
        for a in range(len(drugs)):
            for b in range(a + 1, len(drugs)):
                total_pairs += 1
                if ddi_adj[drugs[a]][drugs[b]] == 1:
                    total_ddi += 1
    ddi_rate = total_ddi / max(total_pairs, 1)
    avg_d = np.mean([p.sum() for p in pred_bin])

    return jac, prec, rec, f1, ddi_rate, avg_d


if __name__ == "__main__":
    main()
