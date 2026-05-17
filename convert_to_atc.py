"""
Convert DrugBank-level ground truth / predictions JSONL to ATC-level JSONL.

Typical input formats in this repo:
  - Prediction JSONL (from main_llm_cls.py): each line contains:
      {
        "input": "...",
        "drug_code": ["DB....", ...],          # ground truth DrugBank IDs
        "target": [logit_0, logit_1, ...]      # logits over med vocab indices
      }
  - Dataset JSONL: each line may contain "drug_code" only (no "target").
  - LEADER test_leader JSONL: may include nested DrugBank lists under
      records["medication"] (one list per visit).

Default (--sidecar_atc_fields not set): overwrites DrugBank in place:
  - "drug_code"  -> list of ATC codes (DrugBank IDs removed)
  - records["medication"] -> same nesting, values are ATC per visit
  Removes legacy keys drug_code_atc / medication_atc if present in the input.

With --sidecar_atc_fields: keeps DrugBank in drug_code / medication and adds
  drug_code_atc and records.medication_atc (previous behavior).

Also when logits are present in "target":
  - "pred_atc", optional "pred_atc_scores" (same as before).

Note: string "target" (comma-separated drug names) is not DrugBank IDs and is left unchanged.

WARN: Default overwrite mode expects downstream vocab/evaluation to use ATC tokens in
drug_code if you run inference + evaluate_jsonlines; the stock DrugBank voc will not match.

Notes:
  - Default ATC granularity is ATC-3 (first 4 characters), consistent with evaluate_atc.py.
  - Mapping is read from a JSON file (DrugBank ID -> ATC code). If missing, UNKNOWNs are dropped.

Example (overwrite drug_code + medication with ATC, default):
  python3 convert_to_atc.py \\
    --input_jsonl data/mimic4/handled/test_leader.json \\
    --output_jsonl data/mimic4/handled/test_leader_atc.json \\
    --voc_dir data/mimic4/handled/voc_final.pkl \\
    --mapping_path improve/input/db2atc.json \\
    --atc_prefix_len 4

Example (keep DrugBank + sidecar *_atc fields):
  python3 convert_to_atc.py ... --sidecar_atc_fields
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import dill
import numpy as np


def load_med_vocab(voc_dir: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    with open(voc_dir, "rb") as f:
        voc_dict = dill.load(f)
    med_voc = voc_dict["med_voc"]
    # idx2word may be dict-like; enforce plain dict
    idx2word = dict(med_voc.idx2word)
    word2idx = dict(med_voc.word2idx)
    return idx2word, word2idx


def load_mapping(mapping_path: str) -> Dict[str, str]:
    if not os.path.exists(mapping_path):
        return {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        return {}
    return mapping


def atc_normalize(atc: str, atc_prefix_len: int) -> str:
    if not atc or atc == "UNKNOWN":
        return "UNKNOWN"
    atc = str(atc).strip().upper()
    if atc_prefix_len and len(atc) >= atc_prefix_len:
        return atc[:atc_prefix_len]
    return atc


def map_db_list_to_atc(
    db_codes: Iterable[str],
    db2atc: Dict[str, str],
    atc_prefix_len: int,
    drop_unknown: bool,
) -> List[str]:
    out: List[str] = []
    seen = set()
    for db in db_codes:
        atc = atc_normalize(db2atc.get(db, "UNKNOWN"), atc_prefix_len)
        if atc == "UNKNOWN" and drop_unknown:
            continue
        if atc not in seen:
            out.append(atc)
            seen.add(atc)
    return out


def convert_records_medication_to_atc(
    medication: Any,
    db2atc: Dict[str, str],
    atc_prefix_len: int,
    drop_unknown: bool,
) -> Any:
    """Mirror records['medication'] shape: list[list[DrugBank id]] -> list[list[ATC]]."""
    if not isinstance(medication, list):
        return medication
    out: List[Any] = []
    for visit in medication:
        if isinstance(visit, list):
            codes = [str(x) for x in visit]
            out.append(
                map_db_list_to_atc(codes, db2atc, atc_prefix_len, drop_unknown=drop_unknown)
            )
        else:
            out.append(visit)
    return out


def np_sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    x = x.astype(np.float32, copy=False)
    return 1.0 / (1.0 + np.exp(-x))


def aggregate_atc_scores(
    db_codes: List[str],
    db_scores: np.ndarray,
    db2atc: Dict[str, str],
    atc_prefix_len: int,
    agg: str,
    drop_unknown: bool,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for db, s in zip(db_codes, db_scores.tolist()):
        atc = atc_normalize(db2atc.get(db, "UNKNOWN"), atc_prefix_len)
        if atc == "UNKNOWN" and drop_unknown:
            continue
        if atc not in scores:
            scores[atc] = float(s)
            continue
        if agg == "max":
            scores[atc] = float(max(scores[atc], s))
        elif agg == "sum":
            scores[atc] = float(scores[atc] + s)
        elif agg == "mean":
            # store as (sum, count) temporarily using NaN sentinel in separate dict
            raise ValueError("mean aggregation is implemented via --score_agg mean (handled separately)")
        else:
            raise ValueError(f"Unknown score aggregation: {agg}")
    return scores


def aggregate_atc_scores_mean(
    db_codes: List[str],
    db_scores: np.ndarray,
    db2atc: Dict[str, str],
    atc_prefix_len: int,
    drop_unknown: bool,
) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    cnts: Dict[str, int] = {}
    for db, s in zip(db_codes, db_scores.tolist()):
        atc = atc_normalize(db2atc.get(db, "UNKNOWN"), atc_prefix_len)
        if atc == "UNKNOWN" and drop_unknown:
            continue
        sums[atc] = float(sums.get(atc, 0.0) + float(s))
        cnts[atc] = int(cnts.get(atc, 0) + 1)
    return {k: float(sums[k] / max(cnts[k], 1)) for k in sums}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert DrugBank JSONL to ATC-level JSONL")
    p.add_argument("--input_jsonl", required=True, help="Input JSONL path")
    p.add_argument("--output_jsonl", required=True, help="Output JSONL path")
    p.add_argument("--voc_dir", required=True, help="Path to voc_final.pkl (for mapping pred indices -> DrugBank)")
    p.add_argument(
        "--mapping_path",
        default="improve/input/db2atc.json",
        help="Path to DrugBank->ATC mapping JSON (default: improve/input/db2atc.json)",
    )
    p.add_argument("--threshold", type=float, default=0.3, help="Threshold for pred_atc from logits (default 0.3)")
    p.add_argument(
        "--atc_prefix_len",
        type=int,
        default=4,
        help="ATC prefix length (ATC-3 ~ first 4 chars; default 4)",
    )
    p.add_argument(
        "--drop_unknown",
        action="store_true",
        help="Drop UNKNOWN ATC codes instead of keeping them",
    )
    p.add_argument(
        "--add_atc_scores",
        action="store_true",
        help="Add pred_atc_scores (ATC -> aggregated probability) when predictions are available",
    )
    p.add_argument(
        "--score_agg",
        choices=["max", "sum", "mean"],
        default="max",
        help="Aggregation for multiple DrugBank codes mapping to same ATC (default max)",
    )
    p.add_argument(
        "--skip_records_medication",
        action="store_true",
        help="Do not convert records.medication (neither overwrite nor sidecar).",
    )
    p.add_argument(
        "--sidecar_atc_fields",
        action="store_true",
        help="Keep DrugBank in drug_code/medication; add drug_code_atc + medication_atc instead of overwriting.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    idx2db, _ = load_med_vocab(args.voc_dir)
    db2atc = load_mapping(args.mapping_path)

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    n = 0
    n_with_target = 0
    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row: Dict[str, Any] = json.loads(line)

            true_db = row.get("drug_code", []) or []
            rec = row.get("records")

            if args.sidecar_atc_fields:
                row["drug_code_atc"] = map_db_list_to_atc(
                    true_db, db2atc, args.atc_prefix_len, drop_unknown=args.drop_unknown
                )
                if not args.skip_records_medication and isinstance(rec, dict) and "medication" in rec:
                    rec["medication_atc"] = convert_records_medication_to_atc(
                        rec.get("medication"),
                        db2atc,
                        args.atc_prefix_len,
                        drop_unknown=args.drop_unknown,
                    )
            else:
                row["drug_code"] = map_db_list_to_atc(
                    true_db, db2atc, args.atc_prefix_len, drop_unknown=args.drop_unknown
                )
                row.pop("drug_code_atc", None)
                if not args.skip_records_medication and isinstance(rec, dict) and "medication" in rec:
                    rec["medication"] = convert_records_medication_to_atc(
                        rec.get("medication"),
                        db2atc,
                        args.atc_prefix_len,
                        drop_unknown=args.drop_unknown,
                    )
                    rec.pop("medication_atc", None)

            # Predictions: prefer logits vector in "target"; fallback to "pred_drug_codes" if present.
            if "target" in row and isinstance(row["target"], list) and len(row["target"]) > 0:
                n_with_target += 1
                logits = np.asarray(row["target"], dtype=np.float32)
                probs = np_sigmoid(logits)
                pred_indices = np.where(probs >= float(args.threshold))[0].tolist()
                pred_db = [idx2db[i] for i in pred_indices if i in idx2db]

                row["pred_atc"] = map_db_list_to_atc(
                    pred_db, db2atc, args.atc_prefix_len, drop_unknown=args.drop_unknown
                )

                if args.add_atc_scores:
                    # Aggregate ATC scores from probabilities at predicted indices.
                    pred_scores = probs[pred_indices] if len(pred_indices) > 0 else np.asarray([], dtype=np.float32)
                    if args.score_agg == "mean":
                        row["pred_atc_scores"] = aggregate_atc_scores_mean(
                            pred_db, pred_scores, db2atc, args.atc_prefix_len, drop_unknown=args.drop_unknown
                        )
                    else:
                        row["pred_atc_scores"] = aggregate_atc_scores(
                            pred_db, pred_scores, db2atc, args.atc_prefix_len, args.score_agg, drop_unknown=args.drop_unknown
                        )
            elif "pred_drug_codes" in row and isinstance(row["pred_drug_codes"], list):
                pred_db = row.get("pred_drug_codes") or []
                row["pred_atc"] = map_db_list_to_atc(
                    pred_db, db2atc, args.atc_prefix_len, drop_unknown=args.drop_unknown
                )

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"✅ Wrote: {args.output_jsonl}")
    print(f"   Lines: {n}")
    print(f"   With logits target: {n_with_target}")
    if not db2atc:
        print(f"⚠️  Mapping not found/empty at: {args.mapping_path} (ATC will be mostly UNKNOWN)")


if __name__ == "__main__":
    main()

