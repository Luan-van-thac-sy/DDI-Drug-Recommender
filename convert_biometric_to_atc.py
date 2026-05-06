import json
import argparse
import os

def convert_biometric_to_atc(input_file, mapping_file, output_file):
    """
    Converts DrugBank (DB) codes to ATC codes in the BioMistral output format 
    to match the LEADER baseline dataset format.
    """
    print(f"Loading mapping from {mapping_file}...")
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
        
    with open(mapping_file, 'r') as f:
        db2atc = json.load(f)
        
    print(f"Reading records from {input_file}...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    converted_records = []
    missing_mappings = set()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print("Warning: Skipping invalid JSON line.")
                continue
            
            # 1. Convert 'drug_code' list (ground truth / targets)
            new_drug_code = []
            for db_code in record.get('drug_code', []):
                if db_code in db2atc:
                    new_drug_code.append(db2atc[db_code])
                else:
                    missing_mappings.add(db_code)
            
            record['drug_code'] = new_drug_code
            
            # 2. Convert 'medication' lists inside 'records'
            if 'records' in record and 'medication' in record['records']:
                new_meds = []
                for med_list in record['records']['medication']:
                    new_med_list = []
                    for db_code in med_list:
                        if db_code in db2atc:
                            new_med_list.append(db2atc[db_code])
                        else:
                            missing_mappings.add(db_code)
                    new_meds.append(new_med_list)
                    
                record['records']['medication'] = new_meds
                
            converted_records.append(record)
            
    print(f"Writing {len(converted_records)} converted records to {output_file}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in converted_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print("Conversion complete!")
    if missing_mappings:
        print(f"Warning: {len(missing_mappings)} DB codes had no mapping in db2atc.json and were dropped.")
        print(f"Example missing codes: {list(missing_mappings)[:5]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert BioMistral DB codes to ATC codes")
    parser.add_argument('--input', type=str, default='data/mimic3/origin/test_biometric.json', 
                        help='Path to the input biometric JSON file')
    parser.add_argument('--mapping', type=str, default='data/mimic3/origin/db2atc.json', 
                        help='Path to the db2atc JSON mapping file')
    parser.add_argument('--output', type=str, default='data/mimic3/origin/test_biometric_atc.json', 
                        help='Path to save the converted JSON file')
    
    args = parser.parse_args()
    
    convert_biometric_to_atc(args.input, args.mapping, args.output)
