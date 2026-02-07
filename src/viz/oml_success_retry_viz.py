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
                
                # Extract resource usage metrics
                row['total_input_tokens'] = d.get('total_input_tokens', 0)
                row['total_output_tokens'] = d.get('total_output_tokens', 0)
                row['generation_time_seconds'] = d.get('generation_time_seconds', 0.0)

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

def generate_plots_for_model(model_df, model_name, viz_dir, max_retry_index, summary_list=None):
    # Ensure max_judge_retries column exists and fill NaNs
    if 'max_judge_retries' not in model_df.columns:
        model_df['max_judge_retries'] = 0
    
    model_df['max_judge_retries'] = model_df['max_judge_retries'].fillna(0)

    # Define configuration for the 4 possible lines
    configs = [
        {
            'label': 'Base',
            'filter': lambda df: (df['baseline_full_doc'] == True) & (df['max_judge_retries'] == 0),
            'color': 'forestgreen', 'marker': '^', 'offset': -15
        },
        {
            'label': '+Cluster',
            'filter': lambda df: (df['baseline_full_doc'] == False) & (df['max_judge_retries'] == 0),
            'color': 'steelblue', 'marker': 'o', 'offset': 10
        },
        {
            'label': '+Judge',
            'filter': lambda df: (df['baseline_full_doc'] == True) & (df['max_judge_retries'] > 0),
            'color': 'purple', 'marker': 'D', 'offset': -25
        },
        {
            'label': '+Cluster+Judge',
            'filter': lambda df: (df['baseline_full_doc'] == False) & (df['max_judge_retries'] > 0),
            'color': 'coral', 'marker': 's', 'offset': 20
        }
    ]

    active_lines = []
    print(f"--- Stats for model: {model_name} ---")
    
    for cfg in configs:
        sub_df = model_df[cfg['filter'](model_df)]
        if not sub_df.empty:
            print(f"Data points for [{cfg['label']}]: {len(sub_df)}")
            
            # Compute and print average resource usage
            avg_input_tokens = sub_df['total_input_tokens'].mean()
            avg_output_tokens = sub_df['total_output_tokens'].mean()
            avg_gen_time = sub_df['generation_time_seconds'].mean()
            print(f"  [Stats] Mean Input Tokens: {avg_input_tokens:.2f}")
            print(f"  [Stats] Mean Output Tokens: {avg_output_tokens:.2f}")
            print(f"  [Stats] Mean Generation Time: {avg_gen_time:.2f} s")

            cumul, step = calculate_stats(sub_df, max_retry_index)
            active_lines.append({
                'cfg': cfg,
                'cumul': cumul,
                'step': step
            })

            if summary_list is not None:
                # Extract stats
                first_succ = cumul.loc[cumul['retry_index'] == 0, 'success_rate_pct'].values[0] if not cumul.empty else 0.0
                last_succ = cumul['success_rate_pct'].iloc[-1] if not cumul.empty else 0.0
                
                # Map labels to short configuration names
                label_map = {
                    'w/o Clustering, w/o Judge': 'Base',
                    'w/ Clustering, w/o Judge': '+Cluster',
                    'w/o Clustering, w/ Judge': '+Judge',
                    'w/ Clustering, w/ Judge': '+Cluster+Judge'
                }
                short_conf = label_map.get(cfg['label'], cfg['label'])
                
                # \makecell{first success $~/~$ last success}
                latex_str = r"\makecell{" + f"{first_succ:.1f} $~/~$ {last_succ:.1f}" + r"}"
                
                summary_list.append({
                    'Model': model_name,
                    'Configuration': short_conf,
                    'First_Success': first_succ,
                    'Last_Success': last_succ,
                    'LaTeX_Cell': latex_str
                })

    # Safe filename
    safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)

    # ==================== GRAPH 1: Cumulated Success Rate ====================
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    fig1.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)
    
    has_data = False
    for line in active_lines:
        df_stats = line['cumul']
        cfg = line['cfg']
        if not df_stats.empty:
            has_data = True
            ax1.plot(df_stats['retry_index'], df_stats['success_rate_pct'], 
                    marker=cfg['marker'], markersize=8, linewidth=2.5, color=cfg['color'],
                    markerfacecolor=cfg['color'], markeredgecolor='black', markeredgewidth=1.2,
                    label=cfg['label'])
            
            # Fill
            ax1.fill_between(df_stats['retry_index'], 0, df_stats['success_rate_pct'], alpha=0.3, color=cfg['color'])

            for _, row in df_stats.iterrows():
                ax1.annotate(f"{row['success_rate_pct']:.1f}%", 
                             (row['retry_index'], row['success_rate_pct']),
                             textcoords="offset points", xytext=(0, cfg['offset']), ha='center', fontsize=9, color=cfg['color'])

    ax1.set_xlabel("Retry Index (i)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Cumulative Success Rate $R_i$ (%)", fontsize=14, fontweight='bold')
    ax1.set_title(f"Cumulative Success Rate ({model_name})", fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(range(0, max_retry_index + 1))
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3, linewidth=0.8)
    if has_data:
        ax1.legend(fontsize=8)

    plt.tight_layout(pad=2.0)
    save_path1 = viz_dir / f"oml_cumulative_success_rate_{safe_model}.png"
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"Saved cumulative plot to {save_path1}")
    plt.close(fig1)

    # ==================== GRAPH 2: Conditional Success Rate ====================
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)

    has_data = False
    for line in active_lines:
        df_stats = line['step']
        cfg = line['cfg']
        if not df_stats.empty:
            has_data = True
            ax2.plot(df_stats['retry_index'], df_stats['success_rate_pct'], 
                    marker=cfg['marker'], markersize=8, linewidth=2.5, color=cfg['color'],
                    markerfacecolor=cfg['color'], markeredgecolor='black', markeredgewidth=1.2,
                    label=cfg['label'])
            
            for _, row in df_stats.iterrows():
                ax2.annotate(f"{row['success_rate_pct']:.1f}%\n({int(row['successes'])}/{int(row['successes'] + row['failures'])})", 
                             (row['retry_index'], row['success_rate_pct']),
                             textcoords="offset points", xytext=(0, cfg['offset']), ha='center', fontsize=8, color=cfg['color'])

    ax2.set_xlabel("Retry Index (i)", fontsize=14, fontweight='bold')
    save_path2 = viz_dir / f"oml_conditional_success_rate_{safe_model}.png"
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Saved conditional plot to {save_path2}")
    plt.close(fig2)

    # ==================== GRAPH 3: Combined View ====================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    fig3.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.12, wspace=0.25)

    # Left: Cumulative
    for line in active_lines:
        df_stats = line['cumul']
        cfg = line['cfg']
        if not df_stats.empty:
            ax3a.plot(df_stats['retry_index'], df_stats['success_rate_pct'], 
                    marker=cfg['marker'], markersize=8, linewidth=2.5, color=cfg['color'],
                    markerfacecolor=cfg['color'], markeredgecolor='black', markeredgewidth=1.2,
                    label=cfg['label'])
            ax3a.fill_between(df_stats['retry_index'], 0, df_stats['success_rate_pct'], alpha=0.3, color=cfg['color'])
            # for _, row in df_stats.iterrows():
            #     ax3a.annotate(f"{row['success_rate_pct']:.1f}%", 
            #                  (row['retry_index'], row['success_rate_pct']),
            #                  textcoords="offset points", xytext=(0, cfg['offset']), ha='center', fontsize=8, color=cfg['color'])

    ax3a.set_xlabel("Retry Index (i)", fontsize=12, fontweight='bold')
    ax3a.set_ylabel("Cumulative Success Rate $R_i$ (%)", fontsize=12, fontweight='bold')
    ax3a.set_title(f"Cumulative Success Rate ({model_name})", fontsize=12, fontweight='bold')
    ax3a.set_ylim(0, 100)
    ax3a.set_xticks(range(0, max_retry_index + 1))
    ax3a.tick_params(labelsize=10)
    ax3a.grid(True, alpha=0.3, linewidth=0.8)
    ax3a.legend(title="Configurations", fontsize=8)

    # Right: Conditional
    for line in active_lines:
        df_stats = line['step']
        cfg = line['cfg']
        if not df_stats.empty:
            ax3b.plot(df_stats['retry_index'], df_stats['success_rate_pct'], 
                    marker=cfg['marker'], markersize=8, linewidth=2.5, color=cfg['color'],
                    markerfacecolor=cfg['color'], markeredgecolor='black', markeredgewidth=1.2,
                    label=cfg['label'])
            for _, row in df_stats.iterrows():
                ax3b.annotate(f"{row['success_rate_pct']:.1f}%\n({int(row['successes'])}/{int(row['successes'] + row['failures'])})", 
                             (row['retry_index'], row['success_rate_pct']),
                             textcoords="offset points", xytext=(0, cfg['offset']), ha='center', fontsize=8, color=cfg['color'])

    ax3b.set_xlabel("Retry Index (i)", fontsize=12, fontweight='bold')
    ax3b.set_ylabel("Conditional Success Rate $H_i$ (%)", fontsize=12, fontweight='bold')
    ax3b.set_title(f"Conditional Success Rate ({model_name})", fontsize=12, fontweight='bold')
    ax3b.set_ylim(0, 115)
    ax3b.set_xticks(range(0, max_retry_index + 1))
    ax3b.tick_params(labelsize=10)
    ax3b.grid(True, alpha=0.3, linewidth=0.8)
    ax3b.legend(title="Configurations", fontsize=8)

    plt.tight_layout(pad=2.5)
    save_path3 = viz_dir / f"oml_success_rate_combined_{safe_model}.png"
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {save_path3}")
    plt.close(fig3)

    # Display summary tables
    for line in active_lines:
        print(f"\n[Model: {model_name}] CUMULATIVE SUCCESS RATE SUMMARY ({line['cfg']['label']})")
        print(line['cumul'].to_string(index=False))
        print(f"\n[Model: {model_name}] CONDITIONAL SUCCESS RATE SUMMARY ({line['cfg']['label']})")
        print(line['step'].to_string(index=False))

def calculate_global_resource_stats(df, save_path=None):
    """Calculates and prints global resource usage statistics across all models."""
    if df.empty:
        return

    # Ensure max_judge_retries column exists and fill NaNs
    if 'max_judge_retries' not in df.columns:
        df['max_judge_retries'] = 0
    df['max_judge_retries'] = df['max_judge_retries'].fillna(0)

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

    print(f"\n{'='*40}")
    print("GLOBAL RESOURCE USAGE STATISTICS (ALL MODELS)")
    print(f"{'='*40}")

    stats_data = []

    for cfg in configs:
        sub_df = df[cfg['filter'](df)]
        if not sub_df.empty:
            avg_in = sub_df['total_input_tokens'].mean()
            avg_out = sub_df['total_output_tokens'].mean()
            avg_time = sub_df['generation_time_seconds'].mean()
            
            print(f"[{cfg['label']}] (Total samples: {len(sub_df)})")
            print(f"  Mean Input Tokens:     {avg_in:.0f}")
            print(f"  Mean Output Tokens:    {avg_out:.0f}")
            print(f"  Mean Generation Time:  {avg_time:.0f} s")
            
            stats_data.append({
                "Configuration": cfg['label'],
                "Mean_Input_Tokens": f"{avg_in:.0f}",
                "Mean_Output_Tokens": f"{avg_out:.0f}",
                "Mean_Time_Seconds": f"{avg_time:.0f}"
            })
            
    if save_path and stats_data:
        pd.DataFrame(stats_data).to_csv(save_path, index=False)
        print(f"Saved global resource stats to {save_path}")


def export_latex_matrix(df: pd.DataFrame, paper_id: str, save_path: pathlib.Path):
    """Generates the LaTeX matrix table for success rate comparison."""
    if df.empty:
        return

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

    # Configuration definitions matching generate_plots_for_model
    # We need to handle NaN in max_judge_retries
    if 'max_judge_retries' not in df.columns:
        df['max_judge_retries'] = 0
    df['max_judge_retries'] = df['max_judge_retries'].fillna(0)

    configs = [
        ("Base", lambda d: (d['baseline_full_doc'] == True) & (d['max_judge_retries'] == 0)),
        ("+Cluster", lambda d: (d['baseline_full_doc'] == False) & (d['max_judge_retries'] == 0)),
        ("+Judge", lambda d: (d['baseline_full_doc'] == True) & (d['max_judge_retries'] > 0)),
        ("+Cluster+Judge", lambda d: (d['baseline_full_doc'] == False) & (d['max_judge_retries'] > 0))
    ]

    # Helper to calculate metrics
    def calculate_s1_r4(sub_df):
        if sub_df.empty:
            return None
        
        # S1: Success at repetition 0 (First try)
        # oml_valid=True AND oml_repetition_count <= 0
        s1_count = sub_df.apply(lambda r: int(r['oml_valid'] and r['oml_repetition_count'] <= 0), axis=1).sum()
        s1 = s1_count / len(sub_df) * 100

        # R4: Cumulative success at repetition <= 4
        # oml_valid=True AND oml_repetition_count <= 4
        # Note: We check if it succeeded at any retry <= 4.
        r4_count = sub_df.apply(lambda r: int(r['oml_valid'] and r['oml_repetition_count'] <= 4), axis=1).sum()
        r4 = r4_count / len(sub_df) * 100
        
        return s1, r4

    # Pre-calculate all values to find max for bolding
    results = {} # (model, config_name) -> (s1, r4)
    model_max_r4 = {} # model -> max_r4
    model_max_s1 = {} # model -> max_s1

    for model in model_order:
        model_df = df[df['model_name'] == model]
        if model_df.empty:
            continue
            
        max_r4 = -1.0
        max_s1 = -1.0
        
        for config_name, filter_func in configs:
            sub_df = model_df[filter_func(model_df)]
            metrics = calculate_s1_r4(sub_df)
            
            if metrics:
                s1, r4 = metrics
                results[(model, config_name)] = (s1, r4)
                if r4 > max_r4:
                    max_r4 = r4
                if s1 > max_s1:
                    max_s1 = s1
            else:
                results[(model, config_name)] = None
        
        model_max_r4[model] = max_r4
        model_max_s1[model] = max_s1

    paper_label = paper_id if paper_id else "Pn"

    latex_content = [
        r"\begin{table*}[htbp]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \caption{Performance comparison on OML generation ($S_1(\%)~/~R_4(\%)$, $n=25$). $S_1$ denotes the first round success rate, while $R_4$ denotes the cumulative success rate after four retries.}",
        r"    \begin{tabular}{c|l|cccccc}",
        r"        \hline",
        r"        \makecell{\textbf{Paper} \\ \textbf{ID}} & \textbf{Configuration} & " + 
        " & ".join([model_latex_headers.get(m, m.replace('_', r'\_')) for m in model_order]) + r" \\",
        r"        \hline",
        r"        "
    ]

    for i, (config_name, _) in enumerate(configs):
        row_str = ""
        if i == 0:
            row_str += f"        \\multirow{{4}}{{*}}{{\\textbf{{{paper_label}}}}} \n"
            row_str += f"          & \\textbf{{{config_name}}}"
        else:
            row_str += f"          & \\textbf{{{config_name}}}"
        
        for model in model_order:
            val = results.get((model, config_name))
            if val is None:
                row_str += "& -- "
            else:
                s1, r4 = val
                
                s1_str = f"{s1:.1f}"
                if s1 >= model_max_s1.get(model, -1.0) - 1e-9:
                     s1_str = f"\\textbf{{{s1_str}}}"

                r4_str = f"{r4:.1f}"
                if r4 >= model_max_r4.get(model, -1.0) - 1e-9:
                     r4_str = f"\\textbf{{{r4_str}}}"

                cell_text = f"{s1_str} $~/~$ {r4_str}"
                
                row_str += f"& {cell_text} "
        
        row_str += r"\\"
        latex_content.append(row_str)

    latex_content.extend([
        r"        \hline",
        r"        ",
        r"    \end{tabular}",
        r"    \label{tab:success_rates_" + paper_label.lower() + r"}",
        r"\end{table*}"
    ])

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX matrix table saved to {save_path}")


summary_data = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oml Success Retry Visualization")
    parser.add_argument("--exp-path", default="experiments", help="Path to the experiments directory")
    parser.add_argument("--paper", default=None, help="Paper ID (e.g. P1) for LaTeX table generation")
    args = parser.parse_args()

    print("Attempting to load raw JSON data...")
    df = load_data_from_jsons(args.exp_path)

    if df.empty:
        print("No valid JSON data found. Exiting.")
        exit(1)

    # Ensure oml_valid is boolean
    if df['oml_valid'].dtype == 'object':
        df['oml_valid'] = df['oml_valid'].map({'True': True, 'False': False, True: True, False: False})

    # Create output directories
    base_dir = pathlib.Path(args.exp_path)
    viz_dir = base_dir / "analysis" / "visualizations"
    analysis_dir = base_dir / "analysis"
    
    viz_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

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

        generate_plots_for_model(model_df, model_name, viz_dir, max_oml_retries, summary_list=summary_data)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # Sort for better consistency if needed
        # summary_df.sort_values(by=['Model', 'Configuration'], inplace=True)
        out_csv = analysis_dir / "oml_success_summary.csv"
        summary_df.to_csv(out_csv, index=False)
        print(f"\nSaved success summary table to {out_csv}")
    
    # Calculate global resource stats
    calculate_global_resource_stats(df, save_path=analysis_dir / "resource_usage_oml.csv")
    
    # Generate LaTeX Matrix Table if requested (or always if paper is not strictly required but we can default)
    print("\n--- Generating LaTeX Matrix Table ---")
    export_latex_matrix(df, args.paper, analysis_dir / "oml_success_matrix_table.tex")

    print("\nProcessing complete.")