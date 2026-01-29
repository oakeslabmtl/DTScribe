import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import io
import seaborn as sns
import argparse
import json
import re


def load_data_from_jsons(exp_path, mode="oml"):
    """Load data directly from JSON files (oml or characteristics)."""
    
    if mode == "oml":
        json_dir = pathlib.Path(exp_path) / "oml_generation"
        glob_pattern = "*_oml.json"
    elif mode == "characteristics":
        json_dir = pathlib.Path(exp_path) / "characteristics_extraction"
        glob_pattern = "*_characteristics.json"
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'oml' or 'characteristics'")

    data = []
    
    # Look for files based on the pattern
    json_files = list(json_dir.glob(glob_pattern))
    
    if not json_files:
        print(f"No {mode} files found in {json_dir}")
        return pd.DataFrame()

    print(f"Found {len(json_files)} JSON files in {json_dir}")

    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                d = json.load(f)
                row = {}

                if 'experiment_id' in d:
                    eid = d['experiment_id']
                    # Handle legacy logging where _oml was appended to the ID
                    if mode == "oml" and isinstance(eid, str) and eid.endswith("_oml"):
                        eid = eid[:-4]
                    row['experiment_id'] = eid

                if mode == "characteristics" and "extracted_characteristics" in d:
                    chars = d["extracted_characteristics"]
                    if isinstance(chars, dict):
                        # Count "Not in Document" occurrences
                        nid_count = sum(1 for v in chars.values() if isinstance(v, str) and "not in document" in v.lower())
                        row['nid_count'] = nid_count
                    else:
                        row['nid_count'] = 0

                row['filename'] = jf.name
                row['filepath'] = str(jf.absolute())

                data.append(row)
                    
        except Exception as e:
            print(f"Warning: Error reading {jf.name}: {e}")
            
    return pd.DataFrame(data)


def _propose_deletion(df, orphan_ids, label):
    to_delete = df[df['experiment_id'].isin(orphan_ids)]
    print(f"\n=> Would you like to delete these {len(to_delete)} {label} files?")
    
    response = input("Type 'yes' to confirm deletion: ")
    if response.lower() == 'yes':
        for _, row in to_delete.iterrows():
            try:
                fpath = pathlib.Path(row['filepath'])
                fpath.unlink()
                print(f"Deleted: {row['filename']}")

                # Check for associated .oml file if this is an OML json file
                if row['filename'].endswith('_oml.json'):
                    oml_file_name = row['filename'].replace('_oml.json', '.oml')
                    oml_file_path = fpath.parent / oml_file_name
                    if oml_file_path.exists():
                        try:
                            oml_file_path.unlink()
                            print(f"Deleted associated OML file: {oml_file_name}")
                        except Exception as e:
                            print(f"Failed to delete associated OML file {oml_file_name}: {e}")

            except Exception as e:
                print(f"Failed to delete {row['filename']}: {e}")
    else:
        print("Skipping deletion.")


def analyze_experiment_integrity(exp_path, interactive_delete=False):
    print(f"\nAnalyzing experiment integrity in: {exp_path}")
    
    char_df = load_data_from_jsons(exp_path, mode="characteristics")
    oml_df = load_data_from_jsons(exp_path, mode="oml")
    
    if char_df.empty and oml_df.empty:
        print("No data found to analyze.")
        return

    # Handle missing IDs
    if not char_df.empty and 'experiment_id' in char_df.columns:
        missing_id_char = char_df[char_df['experiment_id'].isna()]
        if not missing_id_char.empty:
            print(f"Warning: {len(missing_id_char)} characteristic files are missing 'experiment_id'.")
            char_df = char_df.dropna(subset=['experiment_id'])
    
    if not oml_df.empty and 'experiment_id' in oml_df.columns:
        missing_id_oml = oml_df[oml_df['experiment_id'].isna()]
        if not missing_id_oml.empty:
            print(f"Warning: {len(missing_id_oml)} OML files are missing 'experiment_id'.")
            oml_df = oml_df.dropna(subset=['experiment_id'])

    # Get sets
    char_ids = set(char_df['experiment_id']) if not char_df.empty else set()
    oml_ids = set(oml_df['experiment_id']) if not oml_df.empty else set()
    
    orphaned_char = char_ids - oml_ids
    orphaned_oml = oml_ids - char_ids
    
    print(f"\n--- Report ---")
    print(f"Total valid Characteristics files: {len(char_ids)}")
    print(f"Total valid OML files: {len(oml_ids)}")
    print(f"Matched pairs: {len(char_ids & oml_ids)}")
    
    if orphaned_char:
        print(f"\n[!] Found {len(orphaned_char)} orphaned Characteristics (No corresponding OML)")
        print("Files:")
        for _, row in char_df[char_df['experiment_id'].isin(orphaned_char)].iterrows():
            print(f"  {row['filepath']}")

        if interactive_delete:
            _propose_deletion(char_df, orphaned_char, "orphaned Characteristics")
        else:
            print("    (Run with --fix to delete these files)")

    if orphaned_oml:
        print(f"\n[!] Found {len(orphaned_oml)} orphaned OML (No corresponding Characteristics)")
        print("Files:")
        for _, row in oml_df[oml_df['experiment_id'].isin(orphaned_oml)].iterrows():
            print(f"  {row['filepath']}")

        if interactive_delete:
            _propose_deletion(oml_df, orphaned_oml, "orphaned OML")
        else:
            print("    (Run with --fix to delete these files)")

    # Check for extractions with too many "Not in Document"
    if not char_df.empty and 'nid_count' in char_df.columns:
        poor_chars = char_df[char_df['nid_count'] >= 10]
        if not poor_chars.empty:
            print(f"\n[!] Found {len(poor_chars)} Characteristics files with >= 8 'Not in Document' fields")
            print("Files:")
            for _, row in poor_chars.iterrows():
                print(f"  {row['filepath']} (Count: {row['nid_count']})")
            
            if interactive_delete:
                poor_ids = set(poor_chars['experiment_id'])
                _propose_deletion(char_df, poor_ids, "poor quality characteristics")
            else:
                print("    (Run with --fix to delete these files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify experiment errors and inconsistencies.")
    parser.add_argument("--exp-path", default="experiments", help="Path to the experiments directory")
    parser.add_argument("--fix", action="store_true", help="Interactively fix issues (e.g. delete orphaned files)")
    
    args = parser.parse_args()
    
    analyze_experiment_integrity(args.exp_path, interactive_delete=args.fix)