import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import io
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

# Load the provided CSV data into a DataFrame
oml_file = pathlib.Path("experiments/analysis/oml_summary.csv")
if oml_file.exists():
    df = pd.read_csv(oml_file)
csv_data = df.to_csv(index=False)

df = pd.read_csv(io.StringIO(csv_data))

# Expand rows into attempts (failed repetitions + final success)
expanded_rows = []
for idx, row in df.iterrows():
    for rep in range(row['oml_repetition_count']):
        expanded_rows.append({"retry_index": rep, "success": 0})
    expanded_rows.append({"retry_index": int(row['oml_repetition_count']), "success": int(row['oml_valid'])})

expanded_df = pd.DataFrame(expanded_rows)

# Percentage of success if attempting up to k retries
results = []
max_retry_index = expanded_df['retry_index'].max()

for k in range(max_retry_index + 1):
    # For each original experiment, does it have a success with retries <= k?
    success_within_k = df.apply(lambda r: int(r['oml_valid'] and r['oml_repetition_count'] <= k), axis=1)
    success_rate = success_within_k.sum() / len(df) * 100
    results.append({"max_retries_allowed": k, "success_rate_pct": success_rate})

results_df = pd.DataFrame(results)

# Plot with publication-quality styling
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(results_df['max_retries_allowed'], results_df['success_rate_pct'], 
        marker='o', markersize=8, linewidth=2.5, color='steelblue',
        markerfacecolor='steelblue', markeredgecolor='black', markeredgewidth=1.2)

ax.set_xlabel("OML Generation Retries", fontsize=16, fontweight='bold')
ax.set_ylabel("Success Rate (%)", fontsize=16, fontweight='bold')
ax.set_title("Success Rate vs Number of OML Generation Retries", fontsize=16, fontweight='bold', pad=10)
ax.set_ylim(0, 105)
ax.set_xticks(range(0, int(results_df['max_retries_allowed'].max()) + 1))
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3, linewidth=0.8)

plt.tight_layout()

# Save the plot
output_dir = pathlib.Path("experiments/analysis/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)
save_path = output_dir / "oml_success_rate_vs_retries.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {save_path}")

# Maximize the window
try:
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
except:
    pass  # Ignore if backend doesn't support maximization

plt.show()

results_df