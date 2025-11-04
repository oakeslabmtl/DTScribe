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


def extract_block_metrics(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract block-level metrics from experiment data."""
    rows = []
    
    for experiment in data:
        experiment_id = experiment.get('experiment_id', 'unknown')
        metadata = experiment.get('extraction_metadata', {})
        
        # Find all blocks by looking for block_N_processing_time patterns
        block_numbers = set()
        for key in metadata.keys():
            if key.startswith('block_') and key.endswith('_processing_time'):
                block_num = key.split('_')[1]
                block_numbers.add(int(block_num))
        
        # Extract metrics for each block
        for block_num in sorted(block_numbers):
            row = {
                'experiment_id': experiment_id,
                'block_number': block_num,
                'processing_time': metadata.get(f'block_{block_num}_processing_time', None),
                'input_tokens': metadata.get(f'block_{block_num}_input_tokens', None),
                'output_tokens': metadata.get(f'block_{block_num}_output_tokens', None),
                'retries': metadata.get(f'block_{block_num}_retries', 0),
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
    bp = ax.boxplot(data_to_plot, labels=block_numbers, patch_artist=True,
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
    
    # Maximize the window
    try:
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
    except:
        pass  # Ignore if backend doesn't support maximization
    
    plt.show()
    
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
    
    bp1 = ax1.boxplot(input_data, labels=block_numbers, patch_artist=True,
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
    
    bp2 = ax2.boxplot(output_data, labels=block_numbers, patch_artist=True,
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
    
    # Maximize the window
    try:
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
    except:
        pass  # Ignore if backend doesn't support maximization
    
    plt.show()
    
    # Calculate and return statistics
    token_stats = df.groupby('block_number').agg({
        'input_tokens': ['mean', 'std', 'median', 'min', 'max'],
        'output_tokens': ['mean', 'std', 'median', 'min', 'max']
    }).reset_index()
    
    return token_stats


def plot_retries_per_block(df: pd.DataFrame, save_path: pathlib.Path = None):
    """Plot number of retries per block using box plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    
    # Prepare data
    block_numbers = sorted(df['block_number'].unique())
    retry_data = [df[df['block_number'] == block]['retries'].dropna().values 
                  for block in block_numbers]
    
    # Box plot for retry distribution
    bp1 = ax1.boxplot(retry_data, labels=block_numbers, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(facecolor='indianred', alpha=0.6, edgecolor='black', linewidth=1.0),
                       whiskerprops=dict(color='black', linewidth=1.0),
                       capprops=dict(color='black', linewidth=1.0),
                       medianprops=dict(color='darkred', linewidth=1.5),
                       meanprops=dict(color='darkgreen', linestyle='--', linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, 
                                     markeredgecolor='black', markeredgewidth=0.5, alpha=0.6))
    
    ax1.set_xlabel('Block Number', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Number of Retries', fontsize=16, fontweight='bold')
    ax1.set_title('(a) Retry Distribution', fontsize=16, fontweight='bold', pad=8)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.7)
    ax1.tick_params(labelsize=12)
    
    # Total retries bar chart
    retry_stats = df.groupby('block_number')['retries'].agg(['sum', 'max']).reset_index()
    
    bars2 = ax2.bar(retry_stats['block_number'], retry_stats['sum'],
                    alpha=0.6, color='darkorange', edgecolor='black', linewidth=1.0)

    ax2.set_xlabel('Block Number', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Total Retries', fontsize=16, fontweight='bold')
    ax2.set_title('(b) Total Retries', fontsize=16, fontweight='bold', pad=8)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.7)
    ax2.tick_params(labelsize=12)
    
    # Add value labels with smaller font
    for bar, sum_val in zip(bars2, retry_stats['sum']):
        height = bar.get_height()
        if height > 0:  # Only add label if there are retries
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(sum_val)}',
                    ha='center', va='bottom', fontsize=7)
    
    # Add shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkred', linewidth=1.5, label='Median'),
        Line2D([0], [0], color='darkgreen', linewidth=1.5, linestyle='--', label='Mean')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
              fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    # Maximize the window
    try:
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
    except:
        pass  # Ignore if backend doesn't support maximization
    
    plt.show()
    
    # Calculate and return full statistics
    full_stats = df.groupby('block_number')['retries'].agg([
        'mean', 'std', 'sum', 'max', 'median', 'min'
    ]).reset_index()
    
    return full_stats


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive summary statistics."""
    summary = df.groupby('block_number').agg({
        'processing_time': ['mean', 'std', 'min', 'max', 'median'],
        'input_tokens': ['mean', 'std', 'min', 'max', 'median'],
        'output_tokens': ['mean', 'std', 'min', 'max', 'median'],
        'retries': ['mean', 'std', 'min', 'max', 'sum']
    }).round(2)
    
    return summary


def main():
    """Main execution function."""
    # Define paths
    data_dir = pathlib.Path("experiments/characteristics_extraction")
    output_dir = pathlib.Path("experiments/analysis/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading extraction data...")
    data = load_extraction_data(data_dir)
    
    if not data:
        print("No data found. Please check the data directory.")
        return
    
    # Extract block metrics
    print("Extracting block metrics...")
    df = extract_block_metrics(data)
    
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
    
    # Save detailed metrics
    df.to_csv(output_dir / "detailed_block_metrics.csv", index=False)
    print(f"Detailed metrics saved to {output_dir / 'detailed_block_metrics.csv'}")


if __name__ == "__main__":
    main()
