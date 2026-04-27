import pickle
import json
import os
import time
import requests
from bs4 import BeautifulSoup

import urllib.request

def get_atc_from_drugbank(drugbank_id):
    """
    Uses the free, public NIH RxNorm API to map DrugBank -> RxCUI -> ATC.
    """
    try:
        # Step 1: DrugBank ID to RxCUI
        url1 = f'https://rxnav.nlm.nih.gov/REST/rxcui.json?idtype=Drugbank&id={drugbank_id}'
        req1 = urllib.request.Request(url1, headers={'User-Agent': 'Mozilla/5.0'})

        with urllib.request.urlopen(req1, timeout=5) as response1:
            data1 = json.loads(response1.read().decode('utf-8'))

            if 'idGroup' in data1 and 'rxnormId' in data1['idGroup']:
                rxcui = data1['idGroup']['rxnormId'][0]

                # Step 2: RxCUI to ATC Class
                url2 = f'https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui={rxcui}&relaSource=ATC'
                req2 = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})

                with urllib.request.urlopen(req2, timeout=5) as response2:
                    data2 = json.loads(response2.read().decode('utf-8'))

                    if 'rxclassDrugInfoList' in data2 and 'rxclassDrugInfo' in data2['rxclassDrugInfoList']:
                        classes = data2['rxclassDrugInfoList']['rxclassDrugInfo']

                        codes = []
                        for c in classes:
                            class_id = c['rxclassMinConceptItem']['classId']
                            # We want ATC-3 (4 characters long) or ATC-4 (5 chars)
                            if len(class_id) >= 4 and class_id[0].isalpha():
                                codes.append(class_id)

                        if codes:
                            codes.sort(key=len, reverse=True)
                            return codes[0][:4]

    except Exception as e:
        print(f"  [!] Error fetching {drugbank_id} from RxNorm: {e}")

    return None
def main():
    print("="*50)
    print("Fetching DrugBank to ATC-3 Mapping")
    print("="*50)

    voc_path = 'data/mimic3/handled/voc_final.pkl'
    out_path = 'improve/input/db2atc.json'

    if not os.path.exists(voc_path):
        print(f"Error: Vocabulary file not found at {voc_path}")
        return

    import dill
    with open(voc_path, 'rb') as f:
        voc = dill.load(f)

    med_voc = voc['med_voc']
    drug_ids = list(med_voc.word2idx.keys())
    print(f"Found {len(drug_ids)} unique DrugBank IDs in vocabulary.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load existing to resume if interrupted
    db2atc = {}
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            db2atc = json.load(f)
        print(f"Loaded {len(db2atc)} existing mappings.")

    new_fetches = 0
    for i, db_id in enumerate(drug_ids):
        if db_id in db2atc:
            continue

        print(f"[{i+1}/{len(drug_ids)}] Fetching ATC for {db_id}...")
        atc = get_atc_from_drugbank(db_id)

        if atc:
            print(f"  -> Found ATC-3: {atc}")
            db2atc[db_id] = atc
        else:
            print(f"  -> No ATC found.")
            db2atc[db_id] = "UNKNOWN"

        new_fetches += 1

        # Be nice to the server
        time.sleep(1.5)

        # Save every 10 fetches
        if new_fetches % 10 == 0:
            with open(out_path, 'w') as f:
                json.dump(db2atc, f, indent=2)

    # Final save
    with open(out_path, 'w') as f:
        json.dump(db2atc, f, indent=2)

    unknowns = sum(1 for v in db2atc.values() if v == "UNKNOWN")
    print("\n" + "="*50)
    print(f"Mapping complete! Saved to {out_path}")
    print(f"Total IDs: {len(db2atc)}")
    print(f"Successfully mapped: {len(db2atc) - unknowns}")
    print(f"Unknowns: {unknowns}")
    print("="*50)

if __name__ == "__main__":
    main()
