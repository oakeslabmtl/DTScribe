import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import io
import seaborn as sns
import argparse
import json
import re

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

plt.rcParams.update({
    'figure.figsize': (7, 4.5),
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
    'text.usetex': False,
    'axes.formatter.use_mathtext': True,
})

def load_data_from_jsons(exp_path):
    """Load data directly from OML generation JSON files."""
    json_dir = pathlib.Path(exp_path) / "oml_generation"
    data = []
    
    # Look for files ending in _oml.json based on the provided file list
    json_files = list(json_dir.glob("*_oml.json"))
    
    if not json_files:
        return pd.DataFrame()

    print(f"Found {len(json_files)} JSON files in {json_dir}")

    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                d = json.load(f)
                row = {}
                
                # Map JSON fields to expected DataFrame columns
                # Check for 'oml_valid' or 'valid'
                if 'oml_valid' in d:
                    row['oml_valid'] = d['oml_valid']
                elif 'valid' in d:
                    row['oml_valid'] = d['valid']
                
                # Check for 'oml_repetition_count' or 'repetition_count'
                if 'oml_repetition_count' in d:
                    row['oml_repetition_count'] = d['oml_repetition_count']
                elif 'repetition_count' in d:
                    row['oml_repetition_count'] = d['repetition_count']
                elif 'retries' in d:
                     row['oml_repetition_count'] = d['retries']

                # Check for max retries
                if 'max_oml_retries' in d['config']:
                    row['max_oml_retries'] = d['config']['max_oml_retries']
                if 'max_judge_retries' in d['config']:
                    row['max_judge_retries'] = d['config']['max_judge_retries']
                
                # Check for baseline_full_doc
                if 'baseline_full_doc' in d['config']:
                    row['baseline_full_doc'] = d['config']['baseline_full_doc']
                else:
                    row['baseline_full_doc'] = False

                # Check for model_name
                if 'model_name' in d['config']:
                    row['model_name'] = d['config']['model_name']
                elif 'model' in d['config']:
                    row['model_name'] = d['config']['model']
                else:
                    row['model_name'] = 'Unknown'

                data.append(row)
                    
        except Exception as e:
            print(f"Warning: Error reading {jf.name}: {e}")
            
    return pd.DataFrame(data)

def calculate_stats(df, max_retry_index):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Cumulated success rate calculation
    cumulated_results = []
    for i in range(max_retry_index + 1):
        # For each original experiment, does it have a success with retries <= i?
        success_within_i = df.apply(lambda r: int(r['oml_valid'] and r['oml_repetition_count'] <= i), axis=1)
        success_rate = success_within_i.sum() / len(df) * 100
        cumulated_results.append({"retry_index": i, "success_rate_pct": success_rate})

    cumulated_df = pd.DataFrame(cumulated_results)

    # Conditional success rate calculation
    step_results = []
    for i in range(max_retry_index + 1):
        # For each original experiment, does it have a success exactly at retry i?
        success_at_i = df.apply(lambda r: int(r['oml_valid'] and r['oml_repetition_count'] == i), axis=1)
        
        # Failures at step i:
        # 1. Experiments that continued beyond i (oml_repetition_count > i)
        # 2. Experiments that stopped at i but failed (oml_repetition_count == i and not oml_valid)
        failures_at_i = df.apply(lambda r: int(r['oml_repetition_count'] > i or (r['oml_repetition_count'] == i and not r['oml_valid'])), axis=1)
        
        # Success rate at this specific step (out of attempts that reached this step)
        total_at_step = success_at_i.sum() + failures_at_i.sum()
        success_rate = success_at_i.sum() / total_at_step * 100 if total_at_step > 0 else 0
        step_results.append({"retry_index": i, "success_rate_pct": success_rate, 
                             "successes": success_at_i.sum(), "failures": failures_at_i.sum()})

    step_df = pd.DataFrame(step_results)
    # Filter out steps where no experiments were present (total_at_step == 0)
    step_df = step_df[(step_df['successes'] + step_df['failures']) > 0]
    
    return cumulated_df, step_df

def generate_plots_for_model(model_df, model_name, output_dir, max_retry_index):
    # Split data
    df_false = model_df[model_df['baseline_full_doc'] == False]
    df_true = model_df[model_df['baseline_full_doc'] == True]

    print(f"--- Stats for model: {model_name} ---")
    print(f"Data points with baseline_full_doc=False: {len(df_false)}")
    print(f"Data points with baseline_full_doc=True: {len(df_true)}")

    cumul_false, step_false = calculate_stats(df_false, max_retry_index)
    cumul_true, step_true = calculate_stats(df_true, max_retry_index)

    # Safe filename
    safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)

    # ==================== GRAPH 1: Cumulated Success Rate ====================
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    fig1.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)

    if not cumul_false.empty:
        ax1.plot(cumul_false['retry_index'], cumul_false['success_rate_pct'], 
                 marker='o', markersize=8, linewidth=2.5, color='steelblue',
                 markerfacecolor='steelblue', markeredgecolor='black', markeredgewidth=1.2,
                 label='Baseline Full Doc: False')
        for idx, row in cumul_false.iterrows():
            ax1.annotate(f"{row['success_rate_pct']:.1f}%", 
                         (row['retry_index'], row['success_rate_pct']),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='steelblue')

    if not cumul_true.empty:
        ax1.plot(cumul_true['retry_index'], cumul_true['success_rate_pct'], 
                 marker='^', markersize=8, linewidth=2.5, color='forestgreen',
                 markerfacecolor='forestgreen', markeredgecolor='black', markeredgewidth=1.2,
                 label='Baseline Full Doc: True')
        for idx, row in cumul_true.iterrows():
            ax1.annotate(f"{row['success_rate_pct']:.1f}%", 
                         (row['retry_index'], row['success_rate_pct']),
                         textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='forestgreen')
    
    if not cumul_false.empty:
        ax1.fill_between(cumul_false['retry_index'], 0, cumul_false['success_rate_pct'], alpha=0.3, color='steelblue')
    if not cumul_true.empty:
        ax1.fill_between(cumul_true['retry_index'], 0, cumul_true['success_rate_pct'], alpha=0.3, color='forestgreen')

    ax1.set_xlabel("Retry Index (i)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Cumulative Success Rate $R_i$ (%)", fontsize=14, fontweight='bold')
    ax1.set_title(f"Cumulative Success Rate ({model_name})", fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylim(0, 115)
    ax1.set_xticks(range(0, max_retry_index + 1))
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3, linewidth=0.8)
    ax1.legend()

    plt.tight_layout(pad=2.0)
    save_path1 = output_dir / f"oml_cumulative_success_rate_{safe_model}.png"
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"Saved cumulative plot to {save_path1}")
    plt.close(fig1)

    # ==================== GRAPH 2: Conditional Success Rate ====================
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)

    if not step_false.empty:
        ax2.plot(step_false['retry_index'], step_false['success_rate_pct'], 
                 marker='s', markersize=8, linewidth=2.5, color='coral',
                 markerfacecolor='coral', markeredgecolor='black', markeredgewidth=1.2,
                 label='Baseline Full Doc: False')
        for idx, row in step_false.iterrows():
            ax2.annotate(f"{row['success_rate_pct']:.1f}%\n({row['successes']}/{row['successes'] + row['failures']})",
                         (row['retry_index'], row['success_rate_pct']),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='coral')

    if not step_true.empty:
        ax2.plot(step_true['retry_index'], step_true['success_rate_pct'], 
                 marker='D', markersize=8, linewidth=2.5, color='purple',
                 markerfacecolor='purple', markeredgecolor='black', markeredgewidth=1.2,
                 label='Baseline Full Doc: True')
        for idx, row in step_true.iterrows():
            ax2.annotate(f"{row['success_rate_pct']:.1f}%\n({row['successes']}/{row['successes'] + row['failures']})", 
                         (row['retry_index'], row['success_rate_pct']),
                         textcoords="offset points", xytext=(0, -25), ha='center', fontsize=8, color='purple')

    ax2.set_xlabel("Retry Index (i)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Conditional Success Rate $H_i$ (%)", fontsize=14, fontweight='bold')
    ax2.set_title(f"Conditional Success Rate ({model_name})", fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylim(0, 115)
    ax2.set_xticks(range(0, max_retry_index + 1))
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3, linewidth=0.8, axis='y')
    ax2.legend()

    plt.tight_layout(pad=2.0)
    save_path2 = output_dir / f"oml_conditional_success_rate_{safe_model}.png"
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Saved conditional plot to {save_path2}")
    plt.close(fig2)

    # ==================== GRAPH 3: Combined View ====================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 5))
    fig3.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.12, wspace=0.25)

    # Left: Cumulative
    if not cumul_false.empty:
        ax3a.plot(cumul_false['retry_index'], cumul_false['success_rate_pct'], 
                  marker='o', markersize=8, linewidth=2.5, color='steelblue',
                  markerfacecolor='steelblue', markeredgecolor='black', markeredgewidth=1.2,
                  label='False')
    if not cumul_true.empty:
        ax3a.plot(cumul_true['retry_index'], cumul_true['success_rate_pct'], 
                  marker='^', markersize=8, linewidth=2.5, color='forestgreen',
                  markerfacecolor='forestgreen', markeredgecolor='black', markeredgewidth=1.2,
                  label='True')
    
    if not cumul_false.empty:
        ax3a.fill_between(cumul_false['retry_index'], 0, cumul_false['success_rate_pct'], alpha=0.3, color='steelblue')
    if not cumul_true.empty:
        ax3a.fill_between(cumul_true['retry_index'], 0, cumul_true['success_rate_pct'], alpha=0.3, color='forestgreen')

    for idx, row in cumul_false.iterrows():
        ax3a.annotate(f"{row['success_rate_pct']:.1f}%", 
                     (row['retry_index'], row['success_rate_pct']),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='steelblue')
    for idx, row in cumul_true.iterrows():
        ax3a.annotate(f"{row['success_rate_pct']:.1f}%", 
                     (row['retry_index'], row['success_rate_pct']),
                     textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8, color='forestgreen')

    ax3a.set_xlabel("Retry Index (i)", fontsize=12, fontweight='bold')
    ax3a.set_ylabel("Cumulative Success Rate $R_i$ (%)", fontsize=12, fontweight='bold')
    ax3a.set_title(f"Cumulative Success Rate ({model_name})", fontsize=12, fontweight='bold')
    ax3a.set_ylim(0, 115)
    ax3a.set_xticks(range(0, max_retry_index + 1))
    ax3a.tick_params(labelsize=10)
    ax3a.grid(True, alpha=0.3, linewidth=0.8)
    ax3a.legend(title="Baseline Full Doc")

    # Right: Conditional
    if not step_false.empty:
        ax3b.plot(step_false['retry_index'], step_false['success_rate_pct'], 
                  marker='s', markersize=8, linewidth=2.5, color='coral',
                  markerfacecolor='coral', markeredgecolor='black', markeredgewidth=1.2,
                  label='False')
    if not step_true.empty:
        ax3b.plot(step_true['retry_index'], step_true['success_rate_pct'], 
                  marker='D', markersize=8, linewidth=2.5, color='purple',
                  markerfacecolor='purple', markeredgecolor='black', markeredgewidth=1.2,
                  label='True')

    for idx, row in step_false.iterrows():
        ax3b.annotate(f"{row['success_rate_pct']:.1f}%\n({row['successes']}/{row['successes'] + row['failures']})", 
                     (row['retry_index'], row['success_rate_pct']),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='coral')
    for idx, row in step_true.iterrows():
        ax3b.annotate(f"{row['success_rate_pct']:.1f}%\n({row['successes']}/{row['successes'] + row['failures']})", 
                     (row['retry_index'], row['success_rate_pct']),
                     textcoords="offset points", xytext=(0, -25), ha='center', fontsize=8, color='purple')

    ax3b.set_xlabel("Retry Index (i)", fontsize=12, fontweight='bold')
    ax3b.set_ylabel("Conditional Success Rate $H_i$ (%)", fontsize=12, fontweight='bold')
    ax3b.set_title(f"Conditional Success Rate ({model_name})", fontsize=12, fontweight='bold')
    ax3b.set_ylim(0, 115)
    ax3b.set_xticks(range(0, max_retry_index + 1))
    ax3b.tick_params(labelsize=10)
    ax3b.grid(True, alpha=0.3, linewidth=0.8)
    ax3b.legend(title="Baseline Full Doc")

    plt.tight_layout(pad=2.5)
    save_path3 = output_dir / f"oml_success_rate_combined_{safe_model}.png"
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {save_path3}")
    plt.close(fig3)

    # Display summary tables
    print(f"\n[Model: {model_name}] CUMULATIVE SUCCESS RATE SUMMARY (False)")
    print(cumul_false.to_string(index=False))
    print(f"\n[Model: {model_name}] CUMULATIVE SUCCESS RATE SUMMARY (True)")
    print(cumul_true.to_string(index=False))
    print(f"\n[Model: {model_name}] CONDITIONAL SUCCESS RATE SUMMARY (False)")
    print(step_false.to_string(index=False))
    print(f"\n[Model: {model_name}] CONDITIONAL SUCCESS RATE SUMMARY (True)")
    print(step_true.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oml Success Retry Visualization")
    parser.add_argument("--exp-path", default="experiments", help="Path to the experiments directory")
    args = parser.parse_args()

    print("Attempting to load raw JSON data...")
    df = load_data_from_jsons(args.exp_path)

    if df.empty:
        print("No valid JSON data found. Exiting.")
        exit(1)

    # Ensure oml_valid is boolean
    if df['oml_valid'].dtype == 'object':
        df['oml_valid'] = df['oml_valid'].map({'True': True, 'False': False, True: True, False: False})

    # Create output directory
    output_dir = pathlib.Path(args.exp_path) / "analysis" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique models
    unique_models = df['model_name'].unique()
    print(f"Found models: {unique_models}")

    for model_name in unique_models:
        print(f"\n{'='*40}")
        print(f"Generating plots for model: {model_name}")
        print(f"{'='*40}")
        
        model_df = df[df['model_name'] == model_name]
        
        # Determine max retries for this model
        max_oml_retries = int(model_df['max_oml_retries'].iloc[0])
        max_judge_retries = int(model_df['max_judge_retries'].iloc[0])

        generate_plots_for_model(model_df, model_name, output_dir, max_oml_retries)

    print("\nProcessing complete.")