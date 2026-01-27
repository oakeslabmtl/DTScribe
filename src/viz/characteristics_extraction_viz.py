"""
Visualization script for characteristics extraction analysis.
Plots:
- Average processing time per block (with standard deviation)
- Input/output tokens per block
- Number of retries per block
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import pathlib
from typing import List, Dict, Any
import seaborn as sns

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
        # Skip experiments with baseline_full_doc set to true
        if experiment.get('config', {}).get('baseline_full_doc') is True:
            continue

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
                'experiment_id': experiment_id,
                'block_number': block_num,
                'processing_time': metadata.get(f'block_{block_num}_processing_time', None),
                'input_tokens': metadata.get(f'block_{block_num}_input_tokens', None),
                'output_tokens': metadata.get(f'block_{block_num}_output_tokens', None),
                'retries': metadata.get(f'block_{block_num}_retries', 0),
                'average_score': avg_score,
                'accuracy': accuracy
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
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

    summary = df.groupby('block_number').agg(agg_dict).round(2)
    
    return summary


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
    output_dir = base_dir / "analysis/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading extraction data...")
    data = load_extraction_data(data_dir)
    
    if not data:
        print("No data found. Please check the data directory.")
        return
    
    # Determine ground truth
    ground_truth = PAPERS_GROUND_TRUTH.get(args.paper) if args.paper else None

    # Extract block metrics
    print("Extracting block metrics...")
    df = extract_block_metrics(data, ground_truth)
    
    if ground_truth and 'accuracy' in df.columns:
        print(f"\n--- Accuracy Analysis (Target Paper: {args.paper}) ---")
        
        # Accuracy is computed per experiment, so we should aggregate by experiment
        unique_exps = df[['experiment_id', 'accuracy']].drop_duplicates()
        avg_acc = unique_exps['accuracy'].mean()
        std_acc = unique_exps['accuracy'].std()
        max_acc = unique_exps['accuracy'].max()
        min_acc = unique_exps['accuracy'].min()
        
        print(f"Number of experiments analyzed: {len(unique_exps)}")
        print(f"Average Accuracy: {avg_acc:.2%}")
        print(f"Std Dev Accuracy: {std_acc:.2%}")
        print(f"Min Accuracy: {min_acc:.2%}")
        print(f"Max Accuracy: {max_acc:.2%}")
    
    print(f"\nData shape: {df.shape}")
    print(f"Blocks found: {sorted(df['block_number'].unique())}")
    print(f"Total experiments: {df['experiment_id'].nunique()}")
    
    # Generate plots
    print("\n--- Generating Processing Time Plot ---")
    time_stats = plot_processing_time_per_block(
        df, 
        save_path=output_dir / "processing_time_per_block.png"
    )
    
    print("\n--- Generating Token Usage Plots ---")
    token_stats = plot_tokens_per_block(
        df,
        save_path=output_dir / "tokens_per_block.png"
    )
    
    print("\n--- Generating Retry Analysis Plots ---")
    retry_stats = plot_retries_per_block(
        df,
        save_path=output_dir / "retries_per_block.png"
    )
    
    # Generate and save summary statistics
    print("\n--- Summary Statistics ---")
    summary = generate_summary_statistics(df)
    print(summary)
    
    summary_path = output_dir / "block_statistics_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSummary statistics saved to {summary_path}")

    # Generate additional plots
    print("\n--- Generating Score vs Retries Plot ---")
    plot_score_vs_retries(
        df,
        save_path=output_dir / "score_vs_retries.png"
    )

    print("\n--- Generating Correlation Heatmap ---")
    plot_correlation_heatmap(
        df,
        save_path=output_dir / "correlation_heatmap.png"
    )
    
    # Save detailed metrics
    df.to_csv(output_dir / "detailed_block_metrics.csv", index=False)
    print(f"Detailed metrics saved to {output_dir / 'detailed_block_metrics.csv'}")


if __name__ == "__main__":
    main()
