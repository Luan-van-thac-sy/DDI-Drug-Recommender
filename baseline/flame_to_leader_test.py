"""
Convert FLAME test dataset to LEADER model format
"""
import pandas as pd
import json
import ast
import dill

# Load the FLAME test dataset
flame_df = pd.read_csv("/content/DDI-Drug-Recommender/data/flame_test.csv")

# Load the vocabulary for tokenization
tokenizer = dill.load(open("/content/DDI-Drug-Recommender/data/mimic3/handled/voc_final.pkl", "rb"))
diag_voc = tokenizer["diag_voc"]
med_voc = tokenizer["med_voc"]
pro_voc = tokenizer["pro_voc"]

# Load ATC to drug name mapping
atc2drug = {}
drug2atc = {}
try:
    with open("/content/DDI-Drug-Recommender/data/mimic3/auxiliary/WHO ATC-DDD 2021-12-03.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                atc_code = parts[0]
                drug_name = parts[1].lower()
                if len(atc_code) == 4:
                    atc2drug[atc_code] = drug_name
                    drug2atc[drug_name] = atc_code
except:
    print("Warning: Could not load ATC mapping file")

# Load ICD9 mappings
icd2diag_dict = {}
icd2proc_dict = {}
try:
    icd2diag = pd.read_csv("/content/DDI-Drug-Recommender/data/mimic3/raw/D_ICD_DIAGNOSES.csv")
    icd2diag_dict = dict(zip(icd2diag["ICD9_CODE"].astype(str).values, icd2diag["SHORT_TITLE"].values))

    icd2proc = pd.read_csv("/content/DDI-Drug-Recommender/data/mimic3/raw/D_ICD_PROCEDURES.csv")
    icd2proc_dict = dict(zip(icd2proc["ICD9_CODE"].astype(str).values, icd2proc["SHORT_TITLE"].values))
except:
    print("Warning: Could not load ICD9 mapping files")


def parse_list_column(col_value):
    """Parse string representation of list"""
    if pd.isna(col_value):
        return []
    if isinstance(col_value, str):
        try:
            return ast.literal_eval(col_value)
        except:
            return []
    return col_value


def decode_codes(code_list, decoder_dict):
    """Decode a list of codes into corresponding names"""
    decoded = []
    for code in code_list:
        code_str = str(code)
        if code_str in decoder_dict:
            decoded.append(decoder_dict[code_str])
        else:
            # If not found, keep the original
            decoded.append(code_str)
    return decoded


def concat_str(str_list):
    """Concatenate a list of strings with commas"""
    if not str_list:
        return "none"
    return ", ".join(str_list)


# Template for LEADER format
hist_template = "In visit {visit_no}, the patient had diagnosis: {diagnosis}; procedures: {procedure}. The patient was prescribed drugs: {medication}. \n"
main_template = "The patient has {visit_num} times ICU visits. \n{history}In this visit, he has diagnosis: {diagnosis}; procedures: {procedure}. Then, the patient should be prescribed: "


# Process each row in FLAME dataset
leader_test_data = []

for idx, row in flame_df.iterrows():
    subject_id = row['SUBJECT_ID']
    hadm_id = row['HADM_ID']

    # Parse the list columns
    diag_ids = parse_list_column(row['diag_id'])
    proc_ids = parse_list_column(row['pro_id'])
    drug_ids = parse_list_column(row['drug_id'])

    # Get the string representations
    diagnoses = parse_list_column(row['diagnose'])
    procedures = parse_list_column(row['procedure'])
    drug_names = parse_list_column(row['drug_name'])

    # Convert to strings
    diag_str = concat_str(diagnoses)
    proc_str = concat_str(procedures)
    drug_str = concat_str(drug_names)

    # Get ATC3 codes from drug_id (these are indices in the vocabulary)
    # We need to map drug_id to ATC3 codes
    atc3_codes = []
    for drug_idx in drug_ids:
        if drug_idx in med_voc.idx2word:
            atc3_codes.append(med_voc.idx2word[drug_idx])

    # Since FLAME dataset appears to be single visits, we'll create a minimal history
    # In the real scenario, you might want to look up patient history from full MIMIC data
    visit_num = 1  # Single visit in FLAME
    history = ""  # No history available in FLAME

    # Create the input prompt
    input_prompt = main_template.format(
        visit_num=visit_num,
        history=history,
        diagnosis=diag_str,
        procedure=proc_str
    )

    # Create records structure (for compatibility with LEADER format)
    records = {
        "diagnosis": [[str(d) for d in diag_ids]],
        "procedure": [[str(p) for p in proc_ids]],
        "medication": [[str(med_voc.idx2word[m]) if m in med_voc.idx2word else str(m) for m in drug_ids]]
    }

    # Create the data entry
    data_entry = {
        "input": input_prompt,
        "target": drug_str,
        "subject_id": int(subject_id),
        "drug_code": atc3_codes,
        "records": records,
        "hadm_id": int(hadm_id),
        "original_index": int(row['original_index']) if 'original_index' in row else idx
    }

    leader_test_data.append(data_entry)

# Save the converted data
output_file = "/content/DDI-Drug-Recommender/data/mimic3/handled/flame_test.json"
with open(output_file, "w", encoding="utf-8") as f:
    for entry in leader_test_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Converted {len(leader_test_data)} samples from FLAME to LEADER format")
print(f"Saved to: {output_file}")

# Print a sample for verification
if leader_test_data:
    print("\nSample entry:")
    print(json.dumps(leader_test_data[0], indent=2, ensure_ascii=False))

