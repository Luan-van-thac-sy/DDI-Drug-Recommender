#!/usr/bin/env python3
"""
Simple test script to test LLM model with single input
Based on llm_cls_t4_gpu.bash and main_llm_cls.py
"""

import os
import json
import torch
from transformers import AutoTokenizer
from llm.llama import LlamaForMedRec
from llm.lora_cls import PeftModelForCLS
from generators.data import EHRTokenizer
import time

def test_single_input():
    """Test model with single input sample"""

    # Configuration (from llm_cls_t4_gpu.bash)
    model_name_or_path = "resources/llama-7b"
    peft_path = "saved/lora-0105/checkpoint-3000"
    voc_dir = "data/mimic3/handled/voc_final.pkl"
    max_source_length = 512
    max_target_length = 196

    # Sample input data
    sample_input = {
        "input": "The patient has 2 times ICU visits. \n In 1 visit, the patient had diagnosis: Oth spcf deformity head, Acute kidney failure NOS, Tracheostomy comp NEC, Hyperosmolality, Late ef-hemplga side NOS, Cerebral cysts, Hy kid NOS w cr kid I-IV, Chronic kidney dis NOS, DMI wo cmp nt st uncntrl, Old myocardial infarct, Crnry athrscl natve vssl, Status-post ptca, Hyperlipidemia NEC/NOS, Gastrostomy status, Abn react-external stoma, Depressive disorder NEC, Long-term use anticoagul, CHF NOS; procedures: Temporary tracheostomy, Percutan hrt assist syst, Pulsation balloon implan, Percutan aspiration gb, Ins nondrug elut cor st, Percu endosc gastrostomy, Closed bronchial biopsy, Cont inv mec ven 96+ hrs, Entral infus nutrit sub, Nasal sinus dx proc NEC, Resp tract intubat NEC. The patient was prescribed drugs: antithrombotic agents, stomatological preparations, belladonna and derivatives, plain, irrigating solutions, intestinal antiinfectives, insulins and analogues, other mineral supplements, hypnotics and sedatives, antidepressants, antacids, anesthetics, general, drugs for peptic ulcer and gastro-oesophageal reflux disease (gord), arteriolar smooth muscle, agents acting on, drugs for constipation, lipid modifying agents, plain, glycogenolytic hormones, vasodilators used in cardiac diseases, cardiac stimulants excl. cardiac glycosides, high-ceiling diuretics, muscle relaxants, peripherally acting agents, other beta-lactam antibacterials, low-ceiling diuretics, excl. thiazides, beta blocking agents, other cardiac preparations, potassium, quinolone antibacterials, other analgesics and antipyretics, antiepileptics, selective calcium channel blockers with mainly vascular effects, propulsives, anti-acne preparations for topical use, adrenergics, inhalants, decongestants and other nasal preparations for topical use, ace inhibitors, plain, aminoglycoside antibacterials, anxiolytics, antipsychotics, all other therapeutic products, low-ceiling diuretics, thiazides, opioids. \nIn 2 visit, the patient had diagnosis: Oth spcf deformity head, Acute kidney failure NOS, Tracheostomy comp NEC, Hyperosmolality, Late ef-hemplga side NOS, Cerebral cysts, Hy kid NOS w cr kid I-IV, Chronic kidney dis NOS, DMI wo cmp nt st uncntrl, Old myocardial infarct, Crnry athrscl natve vssl, Status-post ptca, Hyperlipidemia NEC/NOS, Gastrostomy status, Abn react-external stoma, Depressive disorder NEC, Long-term use anticoagul, CHF NOS; procedures: Cont inv mec ven <96 hrs, Entral infus nutrit sub. The patient was prescribed drugs: cardiac stimulants excl. cardiac glycosides, insulins and analogues, propulsives, antiepileptics, drugs for peptic ulcer and gastro-oesophageal reflux disease (gord), drugs for constipation, stomatological preparations, antithrombotic agents, irrigating solutions, glycogenolytic hormones, antidepressants, antiarrhythmics, class i and iii, other analgesics and antipyretics, quinolone antibacterials, beta blocking agents, high-ceiling diuretics, intestinal antiinfectives, other mineral supplements, hypnotics and sedatives, vitamin k and other hemostatics, beta-lactam antibacterials, penicillins, adrenergics, inhalants, expectorants, excl. combinations with cough suppressants, decongestants and other nasal preparations for topical use, selective calcium channel blockers with mainly vascular effects, vasodilators used in cardiac diseases, potassium, all other therapeutic products, lipid modifying agents, plain. \n In this visit, he has diagnosis: Oth spcf deformity head, Acute kidney failure NOS, Tracheostomy comp NEC, Hyperosmolality, Late ef-hemplga side NOS, Cerebral cysts, Hy kid NOS w cr kid I-IV, Chronic kidney dis NOS, DMI wo cmp nt st uncntrl, Old myocardial infarct, Crnry athrscl natve vssl, Status-post ptca, Hyperlipidemia NEC/NOS, Gastrostomy status, Abn react-external stoma, Depressive disorder NEC, Long-term use anticoagul, CHF NOS; procedures: Local destruc trach les, Other bronchoscopy, Replace trach tube, Entral infus nutrit sub. Then, the patient should be prescribed: ",
        "target": "lipid modifying agents, plain, drugs for constipation, other analgesics and antipyretics, drugs for peptic ulcer and gastro-oesophageal reflux disease (gord), antiepileptics, other mineral supplements, potassium, irrigating solutions, glycogenolytic hormones, arteriolar smooth muscle, agents acting on, insulins and analogues, antiemetics and antinauseants, beta blocking agents, propulsives, antithrombotic agents, antipsychotics, high-ceiling diuretics, anti-acne preparations for topical use, intestinal antiinfectives, stomatological preparations, antibiotics for topical use",
        "drug_code": ["C10A", "A06A", "N02B", "A02B", "N03A", "A12C", "A12B", "B05C", "H04A", "C02D", "A10A", "A04A", "C07A", "A03F", "B01A", "N05A", "C03C", "D10A", "A07A", "A01A", "D06A"]
    }

    print("=== LLM Model Single Input Test ===")
    print(f"Model path: {model_name_or_path}")
    print(f"PEFT path: {peft_path}")
    print(f"Vocab path: {voc_dir}")
    print()

    try:
        # Load EHR tokenizer
        print("Loading EHR tokenizer...")
        ehr_tokenizer = EHRTokenizer(voc_dir)
        print(f"✓ EHR tokenizer loaded. Med vocab size: {len(ehr_tokenizer.med_voc.word2idx)}")

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        print("✓ Tokenizer loaded")

        # Load model
        print("Loading base model...")
        model = LlamaForMedRec.from_pretrained(
            model_name_or_path,
            med_voc=len(ehr_tokenizer.med_voc.word2idx),
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("✓ Base model loaded")

        # Load PEFT model
        print("Loading PEFT model...")
        model = PeftModelForCLS.from_pretrained(
            model,
            peft_path,
            is_trainable=False
        )
        print("✓ PEFT model loaded")

        model.eval()

        # Prepare input
        print("\nPreparing input...")
        input_text = sample_input["input"]
        print(f"Input length: {len(input_text)} characters")

        # Tokenize input
        inputs = tokenizer(
            input_text,
            max_length=max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        print(f"Tokenized input shape: {inputs['input_ids'].shape}")

        # Run inference
        print("\nRunning inference...")
        start_time = time.time()

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.sigmoid(outputs)

        end_time = time.time()
        inference_time = end_time - start_time

        print(f"✓ Inference completed in {inference_time:.2f} seconds")

        # Process results
        print("\n=== RESULTS ===")

        # Get probabilities
        probabilities = predictions.cpu().numpy()[0]

        # Get ground truth drugs
        ground_truth_drugs = sample_input["drug_code"]
        print(f"Ground truth drugs ({len(ground_truth_drugs)}): {ground_truth_drugs}")

        # Get predicted drugs (threshold = 0.5)
        threshold = 0.5
        predicted_indices = torch.where(predictions > threshold)[1].cpu().numpy()
        predicted_drugs = [ehr_tokenizer.med_voc.idx2word[idx] for idx in predicted_indices]

        print(f"\nPredicted drugs (threshold={threshold}): {predicted_drugs}")

        # Get top 10 predictions
        top_indices = torch.topk(predictions, k=10, dim=1)[1].cpu().numpy()[0]
        top_drugs = [(ehr_tokenizer.med_voc.idx2word[idx], probabilities[idx])
                    for idx in top_indices]

        print(f"\nTop 10 predictions:")
        for i, (drug, prob) in enumerate(top_drugs, 1):
            print(f"  {i:2d}. {drug:<30} ({prob:.3f})")

        # Calculate accuracy metrics
        print(f"\n=== METRICS ===")

        # Convert ground truth to indices (unknown tokens e.g. ATC skipped)
        gt_indices = ehr_tokenizer.convert_med_tokens_to_ids(ground_truth_drugs)

        # Calculate precision, recall, F1
        predicted_set = set(predicted_indices)
        ground_truth_set = set(gt_indices)

        if len(predicted_set) > 0:
            precision = len(predicted_set & ground_truth_set) / len(predicted_set)
        else:
            precision = 0.0

        if len(ground_truth_set) > 0:
            recall = len(predicted_set & ground_truth_set) / len(ground_truth_set)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")

        # Save results
        output_file = "test_single_result.json"
        result = {
            "input": input_text,
            "ground_truth": ground_truth_drugs,
            "predicted_drugs": predicted_drugs,
            "probabilities": probabilities.tolist(),
            "top_predictions": top_drugs,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            },
            "inference_time": inference_time
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_input()
