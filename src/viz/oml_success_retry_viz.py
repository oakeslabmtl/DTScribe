import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import io
import seaborn as sns
import argparse
import json

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
                if 'oml_max_retries' in d:
                    row['oml_max_retries'] = d['oml_max_retries']
                elif 'max_retries' in d:
                    row['oml_max_retries'] = d['max_retries']
                
                # Only add if we have the essential data
                if 'oml_valid' in row and 'oml_repetition_count' in row:
                    data.append(row)
                    
        except Exception as e:
            print(f"Warning: Error reading {jf.name}: {e}")
            
    return pd.DataFrame(data)

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

# Expand rows into attempts (failed repetitions + final success)
expanded_rows = []
for idx, row in df.iterrows():
    for rep in range(int(row['oml_repetition_count'])):
        expanded_rows.append({"retry_index": rep, "success": 0})
    expanded_rows.append({"retry_index": int(row['oml_repetition_count']), "success": int(row['oml_valid'])})

expanded_df = pd.DataFrame(expanded_rows)

# Percentage of success if attempting up to k retries
if 'oml_max_retries' in df.columns:
    max_retry_index = int(df['oml_max_retries'].max())
else:
    max_retry_index = int(expanded_df['retry_index'].max())

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
    
    # print(f"Retry index: {i}, Successes at index: {success_at_i.sum()}, Failures at index: {failures_at_i.sum()}")
    # Success rate at this specific step (out of attempts that reached this step)
    total_at_step = success_at_i.sum() + failures_at_i.sum()
    success_rate = success_at_i.sum() / total_at_step * 100 if total_at_step > 0 else 0
    step_results.append({"retry_index": i, "success_rate_pct": success_rate, 
                         "successes": success_at_i.sum(), "failures": failures_at_i.sum()})

step_df = pd.DataFrame(step_results)
# Filter out steps where no experiments were present (total_at_step == 0)
step_df = step_df[(step_df['successes'] + step_df['failures']) > 0]

# Create output directory
output_dir = pathlib.Path(args.exp_path) / "analysis" / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

# ==================== GRAPH 1: Cumulated Success Rate ====================
fig1, ax1 = plt.subplots(figsize=(7, 4))
fig1.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)

ax1.plot(cumulated_df['retry_index'], cumulated_df['success_rate_pct'], 
         marker='o', markersize=8, linewidth=2.5, color='steelblue',
         markerfacecolor='steelblue', markeredgecolor='black', markeredgewidth=1.2)

# Add data labels on points
for idx, row in cumulated_df.iterrows():
    ax1.annotate(f"{row['success_rate_pct']:.1f}%", 
                 (row['retry_index'], row['success_rate_pct']),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
ax1.fill_between(cumulated_df['retry_index'], 0, cumulated_df['success_rate_pct'], 
                alpha=0.3, color='steelblue')
ax1.set_xlabel("Retry Index (i)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Cumulative Success Rate $R_i$ (%)", fontsize=14, fontweight='bold')
ax1.set_title("Cumulative Success Rate vs Retry Index", 
              fontsize=14, fontweight='bold', pad=10)
ax1.set_ylim(0, 115)
ax1.set_xticks(range(0, int(cumulated_df['retry_index'].max()) + 1))
ax1.tick_params(labelsize=12)
ax1.grid(True, alpha=0.3, linewidth=0.8)

plt.tight_layout(pad=2.0)
save_path1 = output_dir / "oml_cumulative_success_rate.png"
plt.savefig(save_path1, dpi=300, bbox_inches='tight')
print(f"Saved cumulative plot to {save_path1}")

# ==================== GRAPH 2: Conditional Success Rate ====================
fig2, ax2 = plt.subplots(figsize=(7, 4))
fig2.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)

ax2.plot(step_df['retry_index'], step_df['success_rate_pct'], 
         marker='s', markersize=8, linewidth=2.5, color='coral',
         markerfacecolor='coral', markeredgecolor='black', markeredgewidth=1.2)

# Add data labels on points
for idx, row in step_df.iterrows():
    ax2.annotate(f"{row['success_rate_pct']:.1f}%\n({row['successes']}/{row['successes'] + row['failures']})", 
                 (row['retry_index'], row['success_rate_pct']),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

ax2.set_xlabel("Retry Index (i)", fontsize=14, fontweight='bold')
ax2.set_ylabel("Conditional Success Rate $H_i$ (%)", fontsize=14, fontweight='bold')
ax2.set_title("Conditional Success Rate", 
              fontsize=14, fontweight='bold', pad=10)
ax2.set_ylim(0, 115)
ax2.set_xticks(range(0, max_retry_index + 1))
ax2.tick_params(labelsize=12)
ax2.grid(True, alpha=0.3, linewidth=0.8, axis='y')

plt.tight_layout(pad=2.0)
save_path2 = output_dir / "oml_conditional_success_rate.png"
plt.savefig(save_path2, dpi=300, bbox_inches='tight')
print(f"Saved conditional plot to {save_path2}")

# ==================== GRAPH 3: Combined View ====================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(8, 5))
fig3.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.12, wspace=0.25)

# Left: Cumulative
ax3a.plot(cumulated_df['retry_index'], cumulated_df['success_rate_pct'], 
          marker='o', markersize=8, linewidth=2.5, color='steelblue',
          markerfacecolor='steelblue', markeredgecolor='black', markeredgewidth=1.2)
ax3a.fill_between(cumulated_df['retry_index'], 0, cumulated_df['success_rate_pct'], 
                  alpha=0.3, color='steelblue')
ax3a.set_xlabel("Retry Index (i)", fontsize=12, fontweight='bold')
ax3a.set_ylabel("Cumulative Success Rate $R_i$ (%)", fontsize=12, fontweight='bold')
ax3a.set_title("Cumulative Success Rate", fontsize=12, fontweight='bold')
ax3a.set_ylim(0, 115)
ax3a.set_xticks(range(0, int(cumulated_df['retry_index'].max()) + 1))
ax3a.tick_params(labelsize=10)
ax3a.grid(True, alpha=0.3, linewidth=0.8)

# Add data labels on points for Cumulative
for idx, row in cumulated_df.iterrows():
    ax3a.annotate(f"{row['success_rate_pct']:.1f}%", 
                 (row['retry_index'], row['success_rate_pct']),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

# Right: Conditional
ax3b.plot(step_df['retry_index'], step_df['success_rate_pct'], 
          marker='s', markersize=8, linewidth=2.5, color='coral',
          markerfacecolor='coral', markeredgecolor='black', markeredgewidth=1.2)
ax3b.set_xlabel("Retry Index (i)", fontsize=12, fontweight='bold')
ax3b.set_ylabel("Conditional Success Rate $H_i$ (%)", fontsize=12, fontweight='bold')
ax3b.set_title("Conditional Success Rate", fontsize=12, fontweight='bold')
ax3b.set_ylim(0, 115)
ax3b.set_xticks(range(0, max_retry_index + 1))
ax3b.tick_params(labelsize=10)
ax3b.grid(True, alpha=0.3, linewidth=0.8, axis='y')

# Add data labels on points for Conditional
for idx, row in step_df.iterrows():
    ax3b.annotate(f"{row['success_rate_pct']:.1f}%\n({row['successes']}/{row['successes'] + row['failures']})", 
                 (row['retry_index'], row['success_rate_pct']),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=7)

plt.tight_layout(pad=2.5)
save_path3 = output_dir / "oml_success_rate_combined.png"
plt.savefig(save_path3, dpi=300, bbox_inches='tight')
print(f"Saved combined plot to {save_path3}")

# Maximize the window
try:
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
except:
    pass  # Ignore if backend doesn't support maximization

plt.show()

# Display summary tables
print("\n" + "="*60)
print("CUMULATIVE SUCCESS RATE SUMMARY")
print("="*60)
print(cumulated_df.to_string(index=False))

print("\n" + "="*60)
print("CONDITIONAL SUCCESS RATE SUMMARY")
print("="*60)
print(step_df.to_string(index=False))