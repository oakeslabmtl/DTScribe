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

                # Load config into row
                if 'config' in d:
                    config = d.get('config', {})
                    for k, v in config.items():
                        if k not in ['experiment_id', 'filename', 'filepath']:
                            row[k] = v
                
                # Ensure defaults are set if missing from config
                if 'max_judge_retries' not in row:
                    row['max_judge_retries'] = 0
                if 'baseline_full_doc' not in row:
                    row['baseline_full_doc'] = False

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


def check_redundant_runs(df, max_reps, interactive_delete):
    """Check for experiments with same config but excess repetitions."""
    if df.empty or max_reps is None:
        return

    print(f"\nChecking for redundancy (Max reps: {max_reps})...")
    
    # Identify columns to use for grouping (parameters)
    # Exclude metadata and potential non-config varying fields
    exclude_cols = {
        'experiment_id', 'filename', 'filepath', 'nid_count',  # added by loader
        'custom_params', # contains run metadata like batch, rep, etc.
        'experiment_batch', 'experiment_number', 'parameter_combination', 'repetition' # if they leaked into top level
    }
    
    config_cols = [c for c in df.columns if c not in exclude_cols]
    
    if not config_cols:
        print("Warning: No configuration columns found to group by.")
        return

    # Create a copy for grouping manipulations
    df_grouping = df.copy()
    
    # Handle unhashable types (lists, dicts) by converting to string for grouping
    for col in config_cols:
        if df_grouping[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df_grouping[col] = df_grouping[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
            
    # Try to extract repetition number for sorting
    if 'repetition' not in df_grouping.columns and 'custom_params' in df.columns:
         def get_rep(x):
             if isinstance(x, dict): return x.get('repetition', 999999)
             return 999999
         df_grouping['repetition'] = df['custom_params'].apply(get_rep)
    
    if 'repetition' not in df_grouping.columns:
        df_grouping['repetition'] = 999999

    grouped = df_grouping.groupby(config_cols, dropna=False)
    
    to_delete_ids = []
    
    for _, group in grouped:
        if len(group) > max_reps:
            # Sort by repetition (ascending), then experiment_id (as tie breaker)
            # This ensures we keep the first 'max_reps' repetitions
            sorted_group = group.sort_values(by=['repetition', 'experiment_id'], ascending=[True, True])
            
            # The ones to delete are at the end
            excess = sorted_group.iloc[max_reps:]
            to_delete_ids.extend(excess['experiment_id'].tolist())
            
    if to_delete_ids:
        print(f"\n[!] Found {len(to_delete_ids)} redundant experiments (exceeding {max_reps} reps).")
        # Identify the filenames corresponding to these IDs
        target_df = df[df['experiment_id'].isin(to_delete_ids)]
        
        if interactive_delete:
             to_delete_set = set(to_delete_ids)
             _propose_deletion(df, to_delete_set, "redundant runs")
        else:
            print("Files:")
            for _, row in target_df.iterrows():
                print(f"  {row['filepath']}")
            print("    (Run with --fix to delete these files)")
    else:
        print(f"No redundant runs found (all configs have <= {max_reps} runs).")


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


def check_custom_criteria(df, criteria_list, interactive_delete, label_prefix):
    """Filter dataframe by list of criteria dicts and propose deletion."""
    if df.empty or not criteria_list:
        return

    for i, criteria in enumerate(criteria_list):
        matches = df.copy()
        match_desc = ", ".join([f"{k}={v}" for k, v in criteria.items()])
        
        for k, v in criteria.items():
            if k not in matches.columns:
                # If the column doesn't exist, it can't match.
                matches = matches.iloc[0:0] 
                break
            matches = matches[matches[k] == v]
        
        if not matches.empty:
            print(f"\n[!] Found {len(matches)} {label_prefix} files matching: {{{match_desc}}}")
            
            if interactive_delete:
                ids = set(matches['experiment_id'])
                _propose_deletion(df, ids, f"{label_prefix} (Custom Match)")
            else:
                print("Files:")
                for _, row in matches.iterrows():
                    print(f"  {row['filepath']}")
                print("    (Run with --fix to delete these files)")


def analyze_experiment_integrity(exp_path, interactive_delete=False, custom_criteria=None, max_reps=3):
    print(f"\nAnalyzing experiment integrity in: {exp_path}")
    
    char_df = load_data_from_jsons(exp_path, mode="characteristics")
    oml_df = load_data_from_jsons(exp_path, mode="oml")
    
    if char_df.empty and oml_df.empty:
        print("No data found to analyze.")
        return

    # Check for redundant runs (duplicates beyond max_reps)
    if not oml_df.empty:
        print("\n--- Checking OML Redundancy ---")
        check_redundant_runs(oml_df, max_reps=max_reps, interactive_delete=interactive_delete)

    if not char_df.empty:
        print("\n--- Checking Characteristics Redundancy ---")
        check_redundant_runs(char_df, max_reps=max_reps, interactive_delete=interactive_delete)

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

    # Check Custom Criteria
    if custom_criteria:
        check_custom_criteria(char_df, custom_criteria, interactive_delete, "Characteristics")
        check_custom_criteria(oml_df, custom_criteria, interactive_delete, "OML")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify experiment errors and inconsistencies.")
    parser.add_argument("--exp-path", default="experiments", help="Path to the experiments directory")
    parser.add_argument("--fix", action="store_true", help="Interactively fix issues (e.g. delete orphaned files)")
    parser.add_argument("--max-reps", type=int, default=25, help="Maximum number of allowed repetitions for the same config (default: 3)")
    parser.add_argument("--delete-matching", type=str, help="JSON string of criteria to delete (e.g. '{\"model_name\": \"...\"}')")
    
    args = parser.parse_args()
    
    # Define custom deletion criteria here if you don't want to use CLI JSON
    # Add dictionaries to this list to target specific experiments for deletion.
    # Example: TARGET_CONFIGS_TO_DELETE = [{'model_name': 'gpt-3.5', 'baseline_full_doc': False}]
    TARGET_CONFIGS_TO_DELETE = [
        {
            'model_name': "gpt-oss:120b-cloud",
            "baseline_full_doc": False,
            "max_judge_retries": 2
        }
    ]

    if args.delete_matching:
        try:
            cli_criteria = json.loads(args.delete_matching)
            if isinstance(cli_criteria, dict):
                TARGET_CONFIGS_TO_DELETE.append(cli_criteria)
            elif isinstance(cli_criteria, list):
                TARGET_CONFIGS_TO_DELETE.extend(cli_criteria)
        except json.JSONDecodeError as e:
            print(f"Error parsing --delete-matching JSON: {e}")
            exit(1)

    analyze_experiment_integrity(args.exp_path, interactive_delete=args.fix, custom_criteria=TARGET_CONFIGS_TO_DELETE, max_reps=args.max_reps)