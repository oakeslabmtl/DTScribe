"""
Visualization script for characteristics extraction analysis.
Plots:
- Average processing time per block (with standard deviation)
- Input/output tokens per block
- Number of retries per block
"""

from httpx import head
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import pathlib
import re
from typing import List, Dict, Any
import seaborn as sns
import warnings

# Set style for publication-quality plots (double column)
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)  # Increase font sizes for readability

# Configure matplotlib for publication quality
plt.rcParams.update({
    'figure.figsize': (7, 4.5),  # Double column width (~3.5" each, 7" total)
    'figure.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,  # Set to True if LaTeX is available
    'axes.formatter.use_mathtext': True,
})

# Ground truth vectors for papers P1-P5
PAPERS_GROUND_TRUTH = {
    'P1': [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    'P2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    'P3': [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    'P4': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
    'P5': [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
}

# Ordered list of characteristic keys corresponding to the vector indices
CHARACTERISTIC_KEYS = [
    "system_under_study",
    "physical_acting_components",
    "physical_sensing_components",
    "physical_to_virtual_interaction",
    "virtual_to_physical_interaction",
    "dt_services",
    "twinning_time_scale",
    "multiplicities",
    "life_cycle_stages",
    "dt_models_and_data",
    "tooling_and_enablers",
    "dt_constellation",
    "twinning_process_and_dt_evolution",
    "fidelity_and_validity_considerations",
    "dt_technical_connection",
    "dt_hosting_deployment",
    "insights_and_decision_making",
    "horizontal_integration",
    "data_ownership_and_privacy",
    "standardization",
    "security_and_safety_considerations"
]

def load_extraction_data(data_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Load all characteristics extraction JSON files from a directory."""
    json_files = list(data_dir.glob("*_characteristics.json"))
    data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Loaded {len(data)} experiment files")
    return data


def calculate_accuracy(extracted_data: Dict[str, Any], ground_truth: List[int]) -> float:
    """Calculate accuracy (match proportion) between extracted data and ground truth."""
    matches = 0
    total = len(CHARACTERISTIC_KEYS)
    
    for idx, key in enumerate(CHARACTERISTIC_KEYS):
        extracted_val_str = extracted_data.get(key, "Not in Document")
        if not isinstance(extracted_val_str, str):
            extracted_val_str = "Not in Document" # Treat missing/None as absent
            
        # Determine presence (1) or absence (0)
        # Check against "Not in Document" (case-insensitive)
        is_present = 0 if "not in document" in extracted_val_str.lower() else 1
        
        if is_present == ground_truth[idx]:
            matches += 1
            
    return matches / total


def extract_block_metrics(data: List[Dict[str, Any]], ground_truth: List[int] = None) -> pd.DataFrame:
    """Extract block-level metrics from experiment data."""
    rows = []
    
    for experiment in data:
        config = experiment.get('config', {})
        
        # Extract model configuration
        model_name = config.get('model_name', 'unknown')
        max_judge_retries = config.get('max_judge_retries', 0)
        baseline_full_doc = config.get('baseline_full_doc', False)

        experiment_id = experiment.get('experiment_id', 'unknown')
        metadata = experiment.get('extraction_metadata', {})
        extracted_chars = experiment.get('extracted_characteristics', {})
        
        accuracy = None
        if ground_truth:
            accuracy = calculate_accuracy(extracted_chars, ground_truth)
        
        # Find all blocks by looking for block_N_processing_time patterns
        block_numbers = set()
        for key in metadata.keys():
            if key.startswith('block_') and key.endswith('_processing_time'):
                block_num = key.split('_')[1]
                block_numbers.add(int(block_num))
        
        # Pre-compute initial (retry_1) average scores for each block

        block_initial_scores = {}

        for block_num in block_numbers:

            # Case 1: retry_1 exists
            key = f"block_{block_num}_retry_preserve_info"
            block_info = metadata.get(key, {})

            retry_1 = block_info.get("retry_1")
            scores = []

            if isinstance(retry_1, dict):
                preserved = retry_1.get("preserved", [])
                retried = retry_1.get("retried", [])

                scores = [
                    item.get("score")
                    for item in preserved + retried
                    if isinstance(item, dict) and item.get("score") is not None
                ]

            # Case 2: no retry_1 → fallback to first judge pass
            if not scores:
                judge_data = metadata.get(f"block_{block_num}_judge", [])
                if isinstance(judge_data, list):
                    scores = [
                        item.get("score")
                        for item in judge_data
                        if isinstance(item, dict) and item.get("score") is not None
                    ]

            block_initial_scores[block_num] = (
                float(np.mean(scores)) if scores else np.nan
            )    

        # Handle case where no blocks are found (e.g. baseline full doc) but we want to record accuracy
        if not block_numbers and accuracy is not None:
            rows.append({
                'model_name': model_name,
                'experiment_id': experiment_id,
                'block_number': -1,
                'processing_time': None,
                'input_tokens': None,
                'output_tokens': None,
                'retries': 0,
                'average_score': None,
                'accuracy': accuracy,
                'max_judge_retries': max_judge_retries,
                'baseline_full_doc': baseline_full_doc
            })
        
        # Extract metrics for each block
        for block_num in sorted(block_numbers):
            # Calculate average score
            judge_data = metadata.get(f'block_{block_num}_judge', [])
            avg_score = None
            if judge_data and isinstance(judge_data, list):
                scores = [item.get('score', 0) for item in judge_data if isinstance(item, dict) and 'score' in item]
                # Filter out None values
                valid_scores = [s for s in scores if s is not None]
                if valid_scores:
                    avg_score = sum(valid_scores) / len(valid_scores)

            row = {
                'model_name': model_name,
                'experiment_id': experiment_id,
                'block_number': block_num,
                'processing_time': metadata.get(f'block_{block_num}_processing_time', None),
                'input_tokens': metadata.get(f'block_{block_num}_input_tokens', None),
                'output_tokens': metadata.get(f'block_{block_num}_output_tokens', None),
                'retries': metadata.get(f'block_{block_num}_retries', 0),
                'average_score': avg_score,
                "average_initial_score": block_initial_scores.get(block_num, np.nan),
                'accuracy': accuracy,
                'max_judge_retries': max_judge_retries,
                'baseline_full_doc': baseline_full_doc
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)

    # Ensure numeric columns are strictly numeric to avoid object dtypes
    numeric_cols = ['processing_time', 'input_tokens', 'output_tokens', 'retries', 'average_score', 'average_initial_score', 'accuracy']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def plot_processing_time_per_block(df: pd.DataFrame, save_path: pathlib.Path = None):
    """Plot processing time per block using box plots to show distribution."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Prepare data for box plot
    block_numbers = sorted(df['block_number'].unique())
    data_to_plot = [df[df['block_number'] == block]['processing_time'].dropna().values 
                    for block in block_numbers]
    
    # Create box plot with publication-quality styling
    bp = ax.boxplot(data_to_plot, tick_labels=block_numbers, patch_artist=True,
                     showmeans=True, meanline=True,
                     boxprops=dict(facecolor='steelblue', alpha=0.6, edgecolor='black', linewidth=1.2),
                     whiskerprops=dict(color='black', linewidth=1.2),
                     capprops=dict(color='black', linewidth=1.2),
                     medianprops=dict(color='darkred', linewidth=2),
                     meanprops=dict(color='darkgreen', linestyle='--', linewidth=2),
                     flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, 
                                   markeredgecolor='black', markeredgewidth=0.5, alpha=0.6))
    
    ax.set_xlabel('Block Number', fontsize=16, fontweight='bold')
    ax.set_ylabel('Processing Time (s)', fontsize=16, fontweight='bold')
    ax.set_title('Processing Time Distribution per Block', fontsize=16, fontweight='bold', pad=10)
    ax.tick_params(labelsize=12)
    ax.grid(axis='y', alpha=0.3, linewidth=0.8)
    
    # Add legend for median and mean
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkred', linewidth=2, label='Median'),
        Line2D([0], [0], color='darkgreen', linewidth=2, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        
    # Calculate and return statistics
    stats = df.groupby('block_number')['processing_time'].agg([
        'mean', 'std', 'count', 'min', 'max', 'median'
    ]).reset_index()
    
    return stats


def plot_tokens_per_block(df: pd.DataFrame, save_path: pathlib.Path = None):
    """Plot input and output tokens per block using box plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    
    # Prepare data
    block_numbers = sorted(df['block_number'].unique())
    
    # Input tokens
    input_data = [df[df['block_number'] == block]['input_tokens'].dropna().values 
                  for block in block_numbers]
    
    bp1 = ax1.boxplot(input_data, tick_labels=block_numbers, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(facecolor='coral', alpha=0.6, edgecolor='black', linewidth=1.0),
                       whiskerprops=dict(color='black', linewidth=1.0),
                       capprops=dict(color='black', linewidth=1.0),
                       medianprops=dict(color='darkred', linewidth=1.5),
                       meanprops=dict(color='darkgreen', linestyle='--', linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, 
                                     markeredgecolor='black', markeredgewidth=0.5, alpha=0.6))
    
    ax1.set_xlabel('Block Number', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Input Tokens', fontsize=16, fontweight='bold')
    ax1.set_title('(a) Input Tokens', fontsize=16, fontweight='bold', pad=8)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.7)
    ax1.tick_params(labelsize=12)
    
    # Output tokens
    output_data = [df[df['block_number'] == block]['output_tokens'].dropna().values 
                   for block in block_numbers]
    
    bp2 = ax2.boxplot(output_data, tick_labels=block_numbers, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(facecolor='lightseagreen', alpha=0.6, edgecolor='black', linewidth=1.0),
                       whiskerprops=dict(color='black', linewidth=1.0),
                       capprops=dict(color='black', linewidth=1.0),
                       medianprops=dict(color='darkred', linewidth=1.5),
                       meanprops=dict(color='darkgreen', linestyle='--', linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, 
                                     markeredgecolor='black', markeredgewidth=0.5, alpha=0.6))

    ax2.set_xlabel('Block Number', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Output Tokens', fontsize=16, fontweight='bold')
    ax2.set_title('(b) Output Tokens', fontsize=16, fontweight='bold', pad=8)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.7)
    ax2.tick_params(labelsize=12)
    
    # Add shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkred', linewidth=1.5, label='Median'),
        Line2D([0], [0], color='darkgreen', linewidth=1.5, linestyle='--', label='Mean')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=1, 
              fontsize=12, frameon=True, bbox_to_anchor=(0.55, 0.96))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        
    # Calculate and return statistics
    token_stats = df.groupby('block_number').agg({
        'input_tokens': ['mean', 'std', 'median', 'min', 'max'],
        'output_tokens': ['mean', 'std', 'median', 'min', 'max']
    }).reset_index()
    
    return token_stats


def plot_retries_per_block(df: pd.DataFrame, save_path: pathlib.Path = None):
    """Plot number of retries per block using box plots."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Prepare data
    block_numbers = sorted(df['block_number'].unique())
    retry_data = [df[df['block_number'] == block]['retries'].dropna().values 
                  for block in block_numbers]
    
    # Box plot for retry distribution
    bp = ax.boxplot(retry_data, tick_labels=block_numbers, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(facecolor='indianred', alpha=0.6, edgecolor='black', linewidth=1.0),
                       whiskerprops=dict(color='black', linewidth=1.0),
                       capprops=dict(color='black', linewidth=1.0),
                       medianprops=dict(color='darkred', linewidth=1.5),
                       meanprops=dict(color='darkgreen', linestyle='--', linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, 
                                     markeredgecolor='black', markeredgewidth=0.5, alpha=0.6))
    
    ax.set_xlabel('Block Number', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Retries', fontsize=16, fontweight='bold')
    ax.set_title('Retry Distribution per Block', fontsize=16, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linewidth=0.7)
    ax.tick_params(labelsize=12)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkred', linewidth=1.5, label='Median'),
        Line2D([0], [0], color='darkgreen', linewidth=1.5, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    # Calculate and return full statistics
    full_stats = df.groupby('block_number')['retries'].agg([
        'mean', 'std', 'sum', 'max', 'median', 'min'
    ]).reset_index()
    
    return full_stats


def plot_score_vs_retries(df: pd.DataFrame, save_path: pathlib.Path = None):
    """Plot average score against number of retries."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Filter data with scores
    plot_df = df.dropna(subset=['average_score', 'retries']).copy()
    
    if plot_df.empty:
        print("No score data available for plot_score_vs_retries")
        return

    # Create box plot grouping by number of retries
    retry_counts = sorted(plot_df['retries'].unique())
    data_to_plot = [plot_df[plot_df['retries'] == r]['average_score'].values for r in retry_counts]
    
    bp = ax.boxplot(data_to_plot, tick_labels=retry_counts, patch_artist=True,
                     showmeans=True, meanline=True,
                     boxprops=dict(facecolor='mediumpurple', alpha=0.6, edgecolor='black', linewidth=1.2),
                     whiskerprops=dict(color='black', linewidth=1.2),
                     capprops=dict(color='black', linewidth=1.2),
                     medianprops=dict(color='indigo', linewidth=2),
                     meanprops=dict(color='darkgreen', linestyle='--', linewidth=2),
                     flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, 
                                   markeredgecolor='black', markeredgewidth=0.5, alpha=0.6))
    
    ax.set_xlabel('Number of Retries', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=16, fontweight='bold')
    ax.set_title('Impact of Retries on Score', fontsize=16, fontweight='bold', pad=10)
    ax.tick_params(labelsize=12)
    ax.grid(axis='y', alpha=0.3, linewidth=0.8)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='indigo', linewidth=2, label='Median'),
        Line2D([0], [0], color='darkgreen', linewidth=2, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

def plot_correlation_heatmap(df: pd.DataFrame, save_path: pathlib.Path = None):
    """Plot correlation heatmap of numeric metrics."""
    # Select numeric columns
    cols = ['processing_time', 'input_tokens', 'output_tokens', 'retries', 'average_score']
    labels = ['Proc. Time', 'In Tokens', 'Out Tokens', 'Retries', 'Avg Score']

    if 'accuracy' in df.columns and df['accuracy'].notna().any():
        cols.append('accuracy')
        labels.append('Accuracy')

    # Filter columns that are present and have some data
    valid_cols = [c for c in cols if c in df.columns and df[c].notna().any()]
    
    # We need at least 2 columns to compute correlation
    if len(valid_cols) < 2:
        print("Not enough valid columns for correlation heatmap")
        return

    # Use only valid columns and drop rows with NaNs in those columns
    numeric_df = df[valid_cols].dropna()
    
    if numeric_df.empty:
        print("Not enough data for correlation heatmap")
        return

    # Update labels to match valid_cols
    valid_labels = [labels[cols.index(c)] for c in valid_cols]

    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                xticklabels=valid_labels, yticklabels=valid_labels, ax=ax)
    
    ax.set_title('Correlation Matrix of Metrics', fontsize=16, fontweight='bold', pad=10)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

def plot_avg_score_vs_oml_retries():
    pass

def plot_extraction_retries_vs_oml_retries(extraction_df: pd.DataFrame,
    exp_path: pathlib.Path,
    # model_name: str,
    save_path: pathlib.Path = None
):
    """
    Extraction retries vs OML retries
    OML retries are loaded directly from *_oml.json files.
    """

    # print(f"extraction_df shape: {extraction_df.shape}")
    # print(extraction_df.head())

    # ---------- Aggregate extraction retries ----------
    agg = (
        extraction_df.groupby("experiment_id")
        .agg(
            total_extraction_retries=("retries", "sum"),
            EqD=("average_score", "mean") 
        )
        .reset_index()
    )
    # print(agg.head())
    # print(f"Aggregated extraction data shape: {agg.shape}")

    # ---------- Load OML JSONs ----------
    json_dir = pathlib.Path(exp_path) / "oml_generation"
    rows = []

    for jf in json_dir.glob("*_oml.json"):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                d = json.load(f)

            row = {}

            stem = jf.stem.replace("_oml", "")
            parts = stem.split("_")

            #the experiment_id is always the last two components
            row["experiment_id"] = "_".join(parts[-2:])

            if "oml_repetition_count" in d:
                row["oml_repetition_count"] = d["oml_repetition_count"]
            else:
                continue

            if "model_name" in d["config"]:
                row["model_name"] = d["config"]["model_name"]
            else:
                row["model_name"] = "Unknown"

            rows.append(row)

        except Exception as e:
            print(f"Warning: could not parse {jf.name}: {e}")

    oml_df = pd.DataFrame(rows)

    if oml_df.empty:
        print("No OML JSON data found.")
        return

    # ---------- Filter by model ----------
    # oml_df = oml_df[oml_df["model_name"] == model_name]

    # ---------- Merge to add the oml rep count to each experiment ----------
    plot_df = agg.merge(
        oml_df[["experiment_id", "oml_repetition_count"]],
        on="experiment_id",
        how="inner"
    )

    # print(f"Merged data shape: {plot_df.shape}")
    # print(plot_df.head())
    # print(plot_df.info())

    plot_df = plot_df.dropna(subset=["oml_repetition_count", "EqD"])

    if plot_df.empty:
        print("No overlapping extraction / OML runs found.")
        
    else:
        total_extraction_retries = plot_df["total_extraction_retries"].values
        oml_repetition_count = plot_df["oml_repetition_count"].astype(int).values
        EqD = plot_df["EqD"].values

        # ---- Prepare categories ----
        categories = sorted(np.unique(oml_repetition_count))
        data_by_cat = [
            total_extraction_retries[oml_repetition_count == c]
            for c in categories
        ]

        # ---- Plot ----
        fig1, ax1 = plt.subplots(figsize=(7, 5))

        # Boxplot
        ax1.boxplot(
            data_by_cat,
            positions=np.arange(len(categories)),
            widths=0.6,
            patch_artist=False
        )

        # Stripplot (manual jitter)
        x_positions = []
        y_positions = []
        colors = []

        for idx, c in enumerate(categories):
            mask = oml_repetition_count == c
            n_c = mask.sum()
            jitter = (np.random.rand(n_c) - 0.5) * 0.25

            x_positions.extend(np.full(n_c, idx) + jitter)
            y_positions.extend(total_extraction_retries[mask])
            colors.extend(EqD[mask])

        scatter1 = ax1.scatter(
            x_positions,
            y_positions,
            c=colors,
            cmap="viridis",
            alpha=0.7
        )

        # ---- Labels ----
        ax1.set_xticks(np.arange(len(categories)))
        ax1.set_xticklabels([str(c) for c in categories])
        ax1.set_xlabel("OML retries")
        ax1.set_ylabel("Total extraction retries (sum blocks 1–6)")
        ax1.set_title("Figura 1 – Retries de extracción vs. retries de OML")

        # Colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label("Eq(D)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

def plot_double_lollipop_retry_behavior(model_df: pd.DataFrame, save_path: pathlib.Path = None):
    """
    Double lollipop chart:
    - initial average score (retry_1)
    - final average score
    """

    df = model_df.copy()

    grouped = (
        df.groupby("block_number")
        .agg(
            initial_score=("average_initial_score", "mean"),
            final_score=("average_score", "mean"),
        )
        .reset_index()
    )

    x = grouped["block_number"]

    fig, ax = plt.subplots()

    ax.set_ylim(1.0, 5.0)
    ax.set_yticks(np.arange(1.0, 5.1, 0.5))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # Vertical lines
    for _, row in grouped.iterrows():
        if not (np.isnan(row["initial_score"]) or np.isnan(row["final_score"])):
            ax.plot(
                [row["block_number"], row["block_number"]],
                [row["initial_score"], row["final_score"]],
                linewidth=2,
                alpha=0.7,
                color='black'
            )

    # Initial score dots
    ax.scatter(
        x,
        grouped["initial_score"],
        s=70,
        label="Initial (retry_1)",
        zorder=3,
        color='black'
    )

    # Final score dots
    ax.scatter(
        x,
        grouped["final_score"],
        s=70,
        label="Final (after retries)",
        zorder=3,
        facecolors="white",
        edgecolors="black",
    )

    ax.set_xlabel("Block")
    ax.set_ylabel("Average Judge Score")
    ax.set_xticks(x)
    ax.set_title("Retry Mechanism Effect on Judge Scores")

    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")


def plot_retry_lollipop_by_model(extraction_df: pd.DataFrame, save_path: pathlib.Path = None):
    """
    Double lollipop chart showing retry effect per block, grouped by model.

    X-axis: block_number (1..6)
    Y-axis: average_score
    Each lollipop: average_initial_score -> average_score
    Color: model_name
    """
    #filter out max_judge_retries = 0 since they have no retry data
    print(f"Original extraction_df shape: {extraction_df.shape}")
    extraction_df = extraction_df[extraction_df["max_judge_retries"] > 0]
    print(f"Filtered extraction_df shape (max_judge_retries > 0): {extraction_df.shape}")

    fig, ax = plt.subplots()

    blocks = sorted(extraction_df["block_number"].unique())
    models = sorted(extraction_df["model_name"].unique())

    # Color mapping per model
    cmap = plt.cm.get_cmap("tab10", len(models))
    model_colors = {m: cmap(i) for i, m in enumerate(models)}

    # Horizontal offsets so models don't overlap inside each block
    n_models = len(models)
    offset_width = 0.6
    offsets = np.linspace(
        -offset_width / 2,
        offset_width / 2,
        n_models
    )

    for m_idx, model in enumerate(models):
        model_df = extraction_df[extraction_df["model_name"] == model]

        # print(f"Plotting model: {model} with {len(model_df)} rows")
        # print(model_df.head(10))

        for block in blocks:
            row = model_df[model_df["block_number"] == block]
            if row.empty:
                continue

            y_initial = row["average_initial_score"].values[0]
            y_final = row["average_score"].values[0]

            # Skip if retry info does not exist
            if pd.isna(y_initial) or pd.isna(y_final):
                print(f"Skipping model {model} block {block} due to missing score data")
                continue

            x = block + offsets[m_idx]
            color = model_colors[model]

            # Stick
            ax.plot(
                [x, x],
                [y_initial, y_final],
                color=color,
                linewidth=2,
                zorder=1
            )

            # Initial point
            ax.scatter(
                x,
                y_initial,
                color=color,
                s=40,
                zorder=2
            )

            # Final point (hollow marker)
            ax.scatter(
                x,
                y_final,
                facecolors="none",
                edgecolors=color,
                s=60,
                linewidths=1.5,
                zorder=3
            )

    # Axes styling
    ax.set_xlabel("Block")
    ax.set_ylabel("Average Judge Score")
    ax.set_xticks(blocks)
    ax.set_ylim(1, 5)

    # Legend (one entry per model)
    legend_handles = [
        plt.Line2D(
            [0], [0],
            color=model_colors[m],
            lw=2,
            label=m
        )
        for m in models
    ]
    ax.legend(
        handles=legend_handles,
        title="Model",
        loc="best"
    )

    ax.set_title("Retry Mechanism Effect on Judge Scores")

    plt.tight_layout()

    if save_path:
        retry_path = save_path.with_name(save_path.stem + "_double_lollipop_all_models.png")
        plt.savefig(retry_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {retry_path}")

    # =========================
    # Grouped bar chart: deltas
    # =========================
    fig_delta, ax_delta = plt.subplots()

    for m_idx, model in enumerate(models):
        model_df = extraction_df[extraction_df["model_name"] == model]

        delta_vals = []
        x_positions = []

        for block in blocks:
            row = model_df[model_df["block_number"] == block]
            if row.empty:
                continue

            y_initial = row["average_initial_score"].values[0]
            y_final = row["average_score"].values[0]

            if pd.isna(y_initial) or pd.isna(y_final):
                continue

            delta = (y_final - y_initial) / (5 - y_initial) if y_initial != 5 else 0  # normalized delta
            # delta = y_final - y_initial  # raw delta
            x = block + offsets[m_idx]

            delta_vals.append(delta)
            x_positions.append(x)

        ax_delta.bar(
            x_positions,
            delta_vals,
            width=offset_width / n_models,
            color=model_colors[model],
            label=model
        )

    # Axes styling
    ax_delta.set_xlabel("Block")
    ax_delta.set_ylabel("Δ Average Judge Score (Final − Initial) / (5 − Initial)")
    ax_delta.set_xticks(blocks)
    # ax_delta.axhline(0, linewidth=1)  # baseline for positive/negative effect
    ax_delta.set_ylim(0, 1)

    ax_delta.set_title("Retry Gain per Block and Model")

    ax_delta.legend(title="Model", loc="best")
    plt.tight_layout()

    if save_path:
        delta_path = save_path.with_name(save_path.stem + "_delta_barchart.png")
        plt.savefig(delta_path, dpi=300, bbox_inches='tight')
        print(f"Saved delta bar chart to {delta_path}")

    # =========================
    # Percentage gain bar chart
    # =========================
    fig_pct, ax_pct = plt.subplots()

    for m_idx, model in enumerate(models):
        model_df = extraction_df[extraction_df["model_name"] == model]

        pct_vals = []
        x_positions = []

        for block in blocks:
            row = model_df[model_df["block_number"] == block]
            if row.empty:
                continue

            y_initial = row["average_initial_score"].values[0]
            y_final = row["average_score"].values[0]

            # Skip invalid or non-retry cases
            if pd.isna(y_initial) or pd.isna(y_final):
                continue
            if y_initial <= 0:
                continue

            pct_gain = ((y_final - y_initial) / y_initial) * 100.0
            x = block + offsets[m_idx]

            pct_vals.append(pct_gain)
            x_positions.append(x)

        ax_pct.bar(
            x_positions,
            pct_vals,
            width=offset_width / n_models,
            color=model_colors[model],
            label=model
        )

    # Axes styling
    ax_pct.set_xlabel("Block")
    ax_pct.set_ylabel("Percentage Gain (%)")
    ax_pct.set_xticks(blocks)
    ax_pct.axhline(0, linewidth=1)
    # ax_pct.set_ylim(0, 100)

    ax_pct.set_title("Relative Retry Gain per Block and Model")

    ax_pct.legend(title="Model", loc="best")
    plt.tight_layout()

    if save_path:
        pct_gain_path = save_path.with_name(save_path.stem + "_percentage_gain_barchart.png")
        plt.savefig(pct_gain_path, dpi=300, bbox_inches='tight')
        print(f"Saved percentage gain bar chart to {pct_gain_path}")



def plot_retry_lollipop_by_model_aggregated(
    extraction_df: pd.DataFrame,
    save_path: pathlib.Path = None
):
    """
    Double lollipop chart showing retry effect aggregated per model (w/o blocks separation)

    X-axis: model_name
    Y-axis: average judge score
    Each lollipop: mean(average_initial_score) -> mean(average_score)
    """

    print(f"Original extraction_df shape: {extraction_df.shape}")
    extraction_df = extraction_df[extraction_df["max_judge_retries"] > 0]
    print(f"Filtered extraction_df shape (max_judge_retries > 0): {extraction_df.shape}")

    fig, ax = plt.subplots(figsize=(5, 4))

    models = sorted(extraction_df["model_name"].unique())
    x_positions = np.arange(len(models))

    cmap = plt.cm.get_cmap("tab10", len(models))
    # model_colors = {m: cmap(i) for i, m in enumerate(models)}

    # Define weights based on number of characteristics per block
    block_weights = {
        1: 3,
        2: 4,
        3: 4,
        4: 4,
        5: 4,
        6: 2
    }

    delta_vals = []
    norm_delta_vals = []
    pct_vals = []
        
    for i, model in enumerate(models):
        model_df = extraction_df[extraction_df["model_name"] == model]

        # Calculate weighted averages
        model_df_copy = model_df.copy()
        model_df_copy['weight'] = model_df_copy['block_number'].map(block_weights)
        
        # Keep only blocks where both valid scores exist
        valid = model_df_copy[
            model_df_copy["average_initial_score"].notna() &
            model_df_copy["average_score"].notna()
        ]
        # print(f"Model {model} valid: {valid.shape[0]} rows after filtering for valid scores")

        if valid.empty:
            print(f"Skipping model {model}: no valid retry data")
            delta_vals.append(0)
            norm_delta_vals.append(0)
            pct_vals.append(0)
            continue

        total_weight = valid["weight"].sum()

        y_initial = (valid["average_initial_score"] * valid["weight"]).sum() / total_weight
        y_final = (valid["average_score"] * valid["weight"]).sum() / total_weight

        norm_delta = (y_final - y_initial) / (5 - y_initial) if y_initial != 5 else 0
        delta = (y_final - y_initial)

        delta_vals.append(delta)
        norm_delta_vals.append(norm_delta)

        if y_initial <= 0:
            pct_vals.append(0)
        else:
            pct_gain = ((y_final - y_initial) / y_initial) * 100.0
            pct_vals.append(pct_gain)

        x = x_positions[i]
        # color = model_colors[model]

        # Stick
        ax.plot(
            [x, x],
            [y_initial, y_final],
            color="black",
            linewidth=1.5,
            zorder=1
        )

        # Initial point
        ax.scatter(
            x,
            y_initial,
            color="black",
            s=60,
            zorder=2
        )

        # Final point
        ax.scatter(
            x,
            y_final,
            facecolors="white",
            edgecolors="black",
            s=80,
            linewidths=1,
            zorder=3
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Average Judge Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(1, 5)

    ax.set_title("Retry Mechanism Effect on Judge Scores per Model")

    plt.tight_layout()

    if save_path:
        agg_path = save_path.with_name(save_path.stem + "_double_lollipop_all_models.png")
        plt.savefig(agg_path, dpi=300, bbox_inches='tight')
        print(f"Saved aggregated lollipop chart to {agg_path}")

    # =========================
    # Aggregated delta bar chart
    # =========================
    fig_delta, ax_delta = plt.subplots(figsize=(6, 4))

    ax_delta.bar(x_positions, delta_vals, width=0.5, color='steelblue', edgecolor='black', linewidth=1)
    ax_delta.set_xlabel("Model", fontsize=12, fontweight='bold')
    ax_delta.set_ylabel("Δ Average Judge Score", fontsize=12, fontweight='bold')
    ax_delta.set_xticks(x_positions)
    ax_delta.set_xticklabels(models, rotation=30, ha="right")
    ax_delta.axhline(0, linewidth=1, color='black')
    # ax_delta.set_ylim(0, 1)
    ax_delta.set_title("Retry Gain per Model")
    plt.tight_layout()

    if save_path:
        delta_path = save_path.with_name(save_path.stem + "_delta_barchart.png")
        plt.savefig(delta_path, dpi=300, bbox_inches='tight')
        print(f"Saved delta bar chart to {delta_path}")        

    # =========================
    # Aggregated normalized delta bar chart
    # =========================
    fig_norm_delta, ax_norm_delta = plt.subplots(figsize=(6, 4))

    ax_norm_delta.bar(x_positions, norm_delta_vals, width=0.5, color='steelblue', edgecolor='black', linewidth=1)
    ax_norm_delta.set_xlabel("Model", fontsize=12, fontweight='bold')
    ax_norm_delta.set_ylabel("Δ Average Judge Score (normalized)", fontsize=12, fontweight='bold')
    ax_norm_delta.set_xticks(x_positions)
    ax_norm_delta.set_xticklabels(models, rotation=30, ha="right")
    ax_norm_delta.axhline(0, linewidth=1, color='black')
    ax_norm_delta.set_ylim(0, 1)
    ax_norm_delta.set_title("Retry Gain per Model")
    plt.tight_layout()

    if save_path:
        delta_path = save_path.with_name(save_path.stem + "_norm_delta_barchart.png")
        plt.savefig(delta_path, dpi=300, bbox_inches='tight')
        print(f"Saved normalized delta bar chart to {delta_path}")        


    # =========================
    # Aggregated percentage gain bar chart
    # =========================
    fig_pct, ax_pct = plt.subplots(figsize=(7, 4))

    ax_pct.bar(x_positions, pct_vals, width=0.6, color='lightseagreen', edgecolor='black', linewidth=1.2)
    ax_pct.set_xlabel("Model", fontsize=12, fontweight='bold')
    ax_pct.set_ylabel("Percentage Gain (%)", fontsize=12, fontweight='bold')
    ax_pct.set_xticks(x_positions)
    ax_pct.set_xticklabels(models, rotation=30, ha="right")
    ax_pct.axhline(0, linewidth=1, color='black')
    ax_pct.set_ylim(0, 100)
    ax_pct.set_title("Relative Retry Gain per Model")
    plt.tight_layout()

    if save_path:
        pct_path = save_path.with_name(save_path.stem + "_percentage_gain_barchart.png")
        plt.savefig(pct_path, dpi=300, bbox_inches='tight')
        print(f"Saved percentage gain bar chart to {pct_path}")


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive summary statistics."""
    agg_dict = {
        'processing_time': ['mean', 'std', 'min', 'max', 'median'],
        'input_tokens': ['mean', 'std', 'min', 'max', 'median'],
        'output_tokens': ['mean', 'std', 'min', 'max', 'median'],
        'retries': ['mean', 'std', 'min', 'max', 'sum'],
        'average_score': ['mean', 'std', 'min', 'max', 'median']
    }

    if 'accuracy' in df.columns and df['accuracy'].notna().any():
        agg_dict['accuracy'] = ['mean', 'std', 'min', 'max', 'median']

    # Suppress RuntimeWarning: Mean of empty slice from numpy when a group is all NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        summary = df.groupby('block_number').agg(agg_dict).round(2)
    
    return summary


def export_accuracy_table(df: pd.DataFrame, save_path: pathlib.Path):
    """Export a table of accuracy per model with mean and std dev, grouped by config."""
    if 'accuracy' not in df.columns or df['accuracy'].isna().all():
        return

    # Accuracy is per experiment, so we deduplicate by experiment_id
    cols = ['experiment_id', 'model_name', 'max_judge_retries', 'baseline_full_doc', 'accuracy']
    cols = [c for c in cols if c in df.columns]
    experiment_df = df[cols].drop_duplicates()
    
    # Group by model + config
    group_cols = ['model_name', 'max_judge_retries', 'baseline_full_doc']
    group_cols = [c for c in group_cols if c in df.columns]

    stats = experiment_df.groupby(group_cols)['accuracy'].agg(['mean', 'std', 'count']).reset_index()
    
    rows = []
    for _, row in stats.iterrows():
        mean_pct = row['mean'] * 100
        std_pct = row['std'] * 100 if pd.notna(row['std']) else 0.0
        
        # Latex format: \makecell{mean% $\pm$ std%}
        latex_str = r"\makecell{" + f"{mean_pct:.1f} $\pm$ {std_pct:.1f}" + r"}"
        
        entry = {
            'Accuracy_Mean': row['mean'],
            'Model': row.get('model_name', None),
            'Accuracy_Std': row['std'],
            'LaTeX_Cell': latex_str,
            'N_Experiments': row['count']
        }
        if 'max_judge_retries' in row:
            entry['Max_Judge_Retries'] = row['max_judge_retries']
        if 'baseline_full_doc' in row:
            entry['Baseline_Full_Doc'] = row['baseline_full_doc']
            
        rows.append(entry)
        
    result_df = pd.DataFrame(rows)
    result_df.to_csv(save_path, index=False)
    print(f"Accuracy summary table saved to {save_path}")


def export_latex_matrix(df: pd.DataFrame, paper_id: str, save_path: pathlib.Path):
    """Generates the LaTeX matrix table for accuracy comparison."""
    if 'accuracy' not in df.columns or df['accuracy'].isna().all():
        return

    # Filter relevant columns and deduplicate by experiment
    cols = ['experiment_id', 'model_name', 'max_judge_retries', 'baseline_full_doc', 'accuracy']
    # Filter only columns present
    cols = [c for c in cols if c in df.columns]
    experiment_df = df[cols].drop_duplicates()

    # Aggregate stats
    group_cols = ['model_name', 'max_judge_retries', 'baseline_full_doc']
    stats = experiment_df.groupby(group_cols)['accuracy'].agg(['mean', 'std']).reset_index()

    # Define model ordering and display names
    model_order = [
        'ministral-3:3b-cloud',
        'ministral-3:8b-cloud',
        'ministral-3:14b-cloud',
        'qwen3-next:80b-cloud',
        'gpt-oss:20b-cloud',
        'gpt-oss:120b-cloud'
    ]
    
    model_latex_headers = {
        'ministral-3:3b-cloud': r'\makecell{\texttt{ministral-3} \\ (3B)}',
        'ministral-3:8b-cloud': r'\makecell{\texttt{ministral-3} \\ (8B)}',
        'ministral-3:14b-cloud': r'\makecell{\texttt{ministral-3} \\ (14B)}',
        'qwen3-next:80b-cloud': r'\makecell{\texttt{qwen3-next} \\ (80B)}',
        'gpt-oss:20b-cloud': r'\makecell{\texttt{gpt-oss} \\ (20B)}',
        'gpt-oss:120b-cloud': r'\makecell{\texttt{gpt-oss} \\ (120B)}'
    }

    # Configurations: (Name, JudgeEnabled, FullDoc)
    configs = [
        ("Base", False, True),
        ("+Cluster", False, False),
        ("+Judge", True, True),
        ("+Cluster+Judge", True, False)
    ]

    def get_model_config_stats(model, judge_retries_gt_0, baseline_full_doc):
        matches = stats[
            (stats['model_name'] == model) & 
            (stats['baseline_full_doc'] == baseline_full_doc)
        ]
        
        if judge_retries_gt_0:
            matches = matches[matches['max_judge_retries'] > 0]
        else:
            matches = matches[matches['max_judge_retries'] == 0]
            
        if matches.empty:
            return None
        return matches.iloc[0]

    # Calculate max mean for each model to identify best results
    model_max_means = {}
    for model in model_order:
        means = []
        for _, judge_enabled, full_doc in configs:
            row = get_model_config_stats(model, judge_enabled, full_doc)
            if row is not None:
                means.append(row['mean'])
        if means:
            model_max_means[model] = max(means)
        else:
            model_max_means[model] = -1.0

    def get_cell_content(model, judge_retries_gt_0, baseline_full_doc):
        row = get_model_config_stats(model, judge_retries_gt_0, baseline_full_doc)
        
        if row is None:
            return "--"
        
        mean_val = row['mean']
        mean_pct = mean_val * 100
        std_pct = row['std'] * 100 if pd.notna(row['std']) else 0.0
        
        content = f"{mean_pct:.1f} $\\pm$ {std_pct:.1f}"
        
        # Bold if it's the best result for this model
        if mean_val >= model_max_means.get(model, -1.0) - 1e-9:
            return f"\\textbf{{{content}}}"
            
        return content

    paper_label = paper_id if paper_id else "Pn"

    latex_content = [
        r"\begin{table*}[htbp]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \caption{Extraction accuracy ($E_{\text{Acc}}$, $n=25$), with two judge retries.}",
        r"    \begin{tabular}{c|l|cccccc}",
        r"        \toprule",
        r"        \makecell{\textbf{Paper} \\ \textbf{ID}} & \textbf{Configuration}& " + 
        " & ".join([model_latex_headers.get(m, m.replace('_', r'\_')) for m in model_order]) + r" \\",
        r"        \hline",
        r"        "
    ]

    for i, (config_name, judge_enabled, full_doc) in enumerate(configs):
        row_str = ""
        if i == 0:
            row_str += f"        \\multirow{{4}}{{*}}{{\\textbf{{{paper_label}}}}} \n"
            row_str += f"          & \\textbf{{{config_name}}}"
        else:
            row_str += f"          & \\textbf{{{config_name}}}"
        
        for model in model_order:
            val = get_cell_content(model, judge_enabled, full_doc)
            row_str += f"& {val} "
        
        row_str += r"\\"
        latex_content.append(row_str)

    latex_content.extend([
        r"        \hline",
        r"        ",
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \label{tab:results_" + paper_label.lower() + r"}",
        r"\end{table*}"
    ])

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX matrix table saved to {save_path}")


def calculate_resource_stats(df: pd.DataFrame, title="Resource Usage Statistics", save_path: pathlib.Path = None):
    """Calculates and prints resource usage statistics per configuration."""
    if df.empty:
        return

    # Ensure columns exist
    if 'max_judge_retries' not in df.columns:
        df['max_judge_retries'] = 0
    if 'baseline_full_doc' not in df.columns:
        df['baseline_full_doc'] = False
        
    df['max_judge_retries'] = df['max_judge_retries'].fillna(0)

    # First aggregate by experiment to get totals
    # We sum tokens and processing time across all blocks for each experiment
    # Note: 'processing_time' here is per block, so sum gives total extraction time
    exp_totals = df.groupby(['experiment_id', 'baseline_full_doc', 'max_judge_retries'])[[
        'input_tokens', 'output_tokens', 'processing_time'
    ]].sum().reset_index()

    configs = [
        {
            'label': 'Base',
            'filter': lambda d: (d['baseline_full_doc'] == True) & (d['max_judge_retries'] == 0)
        },
        {
            'label': '+Cluster',
            'filter': lambda d: (d['baseline_full_doc'] == False) & (d['max_judge_retries'] == 0)
        },
        {
            'label': '+Judge',
            'filter': lambda d: (d['baseline_full_doc'] == True) & (d['max_judge_retries'] > 0)
        },
        {
            'label': '+Cluster+Judge',
            'filter': lambda d: (d['baseline_full_doc'] == False) & (d['max_judge_retries'] > 0)
        }
    ]

    print(f"\n--- {title} (Average per Experiment) ---")
    
    stats_data = []
    
    for cfg in configs:
        sub_df = exp_totals[cfg['filter'](exp_totals)]
        if not sub_df.empty:
            avg_in = sub_df['input_tokens'].mean()
            avg_out = sub_df['output_tokens'].mean()
            avg_time = sub_df['processing_time'].mean()
            print(f"[{cfg['label']}] ({len(sub_df)} experiments)")
            print(f"  Mean Input Tokens:  {avg_in:.0f}")
            print(f"  Mean Output Tokens: {avg_out:.0f}")
            print(f"  Mean Processing Time: {avg_time:.0f} s")
            
            stats_data.append({
                "Configuration": cfg['label'],
                "Mean_Input_Tokens": f"{avg_in:.0f}",
                "Mean_Output_Tokens": f"{avg_out:.0f}",
                "Mean_Time_Seconds": f"{avg_time:.0f}"
            })
            
    if save_path and stats_data:
        pd.DataFrame(stats_data).to_csv(save_path, index=False)
        print(f"Saved resource stats to {save_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Characteristics Extraction Visualization")
    parser.add_argument("--exp-path", default="experiments", help="Path to the experiments directory")
    parser.add_argument("--paper", choices=['P1', 'P2', 'P3', 'P4', 'P5'], help="Select ground truth paper for accuracy calculation", default=None)
    args = parser.parse_args()

    # PAPER ID MAPPING:
    # ------------------------------------------------------------
    # P1: A Digital Shadow for Accurate Robot Motion Control: Integrating Data with Friction Models
    # P2: Engineering Automotive Digital Twins on Standardized Architectures: A Case Study
    # P3: Lab-Scale Gantry Crane Digital Twin Exemplar
    # P4: Model-driven Digital Twins for AECO
    # P5: The Incubator Case Study for Digital Twin Engineering
    # ------------------------------------------------------------

    # Define paths
    base_dir = pathlib.Path(args.exp_path)
    data_dir = base_dir / "characteristics_extraction"
    viz_dir = base_dir / "analysis/visualizations"
    analysis_dir = base_dir / "analysis"

    viz_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading extraction data...")
    data = load_extraction_data(data_dir)

    # Determine ground truth
    ground_truth = PAPERS_GROUND_TRUTH.get(args.paper) if args.paper else None

    # Extract block metrics
    print("Extracting block metrics...")
    df = extract_block_metrics(data, ground_truth)
    
    print(f"\nData shape: {df.shape}")
    print(f"Blocks found: {sorted(df['block_number'].unique())}")
    print(f"Total experiments: {df['experiment_id'].nunique()}")

    # Get unique models
    unique_models = df['model_name'].unique()
    print(f"Found models: {unique_models}")


    for model_name in unique_models:
        print(f"\n{'='*40}")
        print(f"Generating statistics/plots for model: {model_name}")
        print(f"{'='*40}")

        safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)
        model_df = df[df['model_name'] == model_name].copy()

        # Calculate and print resource stats
        calculate_resource_stats(model_df, title=f"Resource Usage ({model_name})")

        if ground_truth and 'accuracy' in model_df.columns:
            print(f"\n--- Accuracy Analysis ({model_name} / Target Paper: {args.paper}) ---")
            
            # Accuracy is computed per experiment, so we should aggregate by experiment
            unique_exps = model_df[['experiment_id', 'accuracy']].drop_duplicates()
            avg_acc = unique_exps['accuracy'].mean()
            std_acc = unique_exps['accuracy'].std()
            max_acc = unique_exps['accuracy'].max()
            min_acc = unique_exps['accuracy'].min()
            
            print(f"Number of experiments analyzed: {len(unique_exps)}")
            print(f"Average Accuracy: {avg_acc:.2%}")
            print(f"Std Dev Accuracy: {std_acc:.2%}")
            print(f"Min Accuracy: {min_acc:.2%}")
            print(f"Max Accuracy: {max_acc:.2%}")

        # Generate plots
        print(f"\n--- Generating Processing Time Plot ({model_name}) ---")
        time_stats = plot_processing_time_per_block(
            model_df, 
            save_path=viz_dir / f"processing_time_per_block_{safe_model}.png"
        )
        
        print(f"\n--- Generating Token Usage Plots ({model_name}) ---")
        token_stats = plot_tokens_per_block(
            model_df,
            save_path=viz_dir / f"tokens_per_block_{safe_model}.png"
        )
        
        print(f"\n--- Generating Retry Analysis Plots ({model_name}) ---")
        retry_stats = plot_retries_per_block(
            model_df,
            save_path=viz_dir / f"retries_per_block_{safe_model}.png"
        )
        
        # Generate and save summary statistics
        print(f"\n--- Summary Statistics ({model_name}) ---")
        summary = generate_summary_statistics(model_df)
        print(summary)
        
        summary_path = analysis_dir / f"block_statistics_summary_{safe_model}.csv"
        summary.to_csv(summary_path)
        print(f"\nSummary statistics saved to {summary_path}")

        # Generate additional plots
        print(f"\n--- Generating Score vs Retries Plot ({model_name}) ---")
        plot_score_vs_retries(
            model_df,
            save_path=viz_dir / f"score_vs_retries_{safe_model}.png"
        )

        print(f"\n--- Generating Correlation Heatmap ({model_name}) ---")
        plot_correlation_heatmap(
            model_df,
            save_path=viz_dir / f"correlation_heatmap_{safe_model}.png"
        )

        print(f"\n--- Generating Double Lollipop Retry Behavior Plot ({model_name}) ---")
        plot_double_lollipop_retry_behavior(
            model_df,
            save_path=viz_dir / f"double_lollipop_retry_behavior_{safe_model}.png"
        )
        
        # Save detailed metrics
        model_df.to_csv(analysis_dir / f"detailed_block_metrics_{safe_model}.csv", index=False)
        print(f"Detailed metrics saved to {analysis_dir / f'detailed_block_metrics_{safe_model}.csv'}")


    print("\n--- Generating Correlation Heatmap ---")
    plot_correlation_heatmap(
        df,
        save_path=viz_dir / "correlation_heatmap.png"
    )

    print(f"\n--- Generating Double Lollipop Retry Behavior Plot (All Models) ---")
    plot_retry_lollipop_by_model(
        df, save_path=viz_dir / "retry_behavior"
    )
    
    print(f"\n--- Generating Aggregated Double Lollipop Retry Behavior Plot (All Models) ---")
    plot_retry_lollipop_by_model_aggregated(
        df, save_path=viz_dir / "retry_behavior_aggregated"
    )
    

    # Calculate global resource stats across all models
    calculate_resource_stats(df, title="GLOBAL RESOURCE USAGE STATISTICS (ALL MODELS)", save_path=analysis_dir / "resource_usage_extraction.csv")

    # Save detailed metrics
    df.to_csv(analysis_dir / "detailed_block_metrics.csv", index=False)
    print(f"Detailed metrics saved to {analysis_dir / 'detailed_block_metrics.csv'}")

    # Export accuracy table if applicable
    if ground_truth:
        print("\n--- Generating Accuracy Summary Table ---")
        export_accuracy_table(df, analysis_dir / "model_accuracy_summary.csv")
        
        print("\n--- Generating LaTeX Matrix Table ---")
        export_latex_matrix(df, args.paper, analysis_dir / "accuracy_matrix_table.tex")



if __name__ == "__main__":
    main()
