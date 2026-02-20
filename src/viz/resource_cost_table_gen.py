import argparse
import pandas as pd
import pathlib
import sys

def load_data(experiment_paths, filename):
    dfs = []
    print(f"Loading {filename} from {len(experiment_paths)} locations...", file=sys.stderr)
    for path in experiment_paths:
        p = pathlib.Path(path) / "analysis" / filename
        if p.exists():
            try:
                df = pd.read_csv(p)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {p}: {e}", file=sys.stderr)
        else:
            print(f"Warning: {p} does not exist", file=sys.stderr)
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs)
    # Average across the different experiments
    averaged = combined.groupby('Configuration').mean().reset_index()
    return averaged

def generate_latex(extraction_df, oml_df):
    # configurations order
    configs_order = ["Base", "+Cluster", "+Judge", "+Cluster+Judge"]
    
    # Merge on Configuration
    # suffixes: _ext, _oml
    merged = pd.merge(extraction_df, oml_df, on='Configuration', suffixes=('_ext', '_oml'))
    
    # Calculate totals
    merged['Mean_Input_Tokens_total'] = merged['Mean_Input_Tokens_ext'] + merged['Mean_Input_Tokens_oml']
    merged['Mean_Output_Tokens_total'] = merged['Mean_Output_Tokens_ext'] + merged['Mean_Output_Tokens_oml']
    merged['Mean_Time_Seconds_total'] = merged['Mean_Time_Seconds_ext'] + merged['Mean_Time_Seconds_oml']
    
    # Set index for easy lookup
    merged.set_index('Configuration', inplace=True)
    
    latex_template = r"""\begin{table*}[h!]
\centering
\caption{Computational cost analysis across pipeline stages averaged over P1--P5. Results show token usage and processing time for the Extraction and Generation phases, alongside the cumulative total.}
\label{tab:pipeline_costs}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
 & \multicolumn{3}{c}{\textbf{Extraction Phase}} & \multicolumn{3}{c}{\textbf{Generation Phase}} & \multicolumn{3}{c}{\textbf{Total}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
\textbf{Configuration} & \makecell{\textbf{Input}\\\textbf{Tokens}} & \makecell{\textbf{Output}\\\textbf{Tokens}} & \textbf{Time (s)} & \makecell{\textbf{Input}\\\textbf{Tokens}} & \makecell{\textbf{Output}\\\textbf{Tokens}} & \textbf{Time (s)} & \makecell{\textbf{Input}\\\textbf{Tokens}} & \makecell{\textbf{Output}\\\textbf{Tokens}} & \textbf{Time (s)} \\
\midrule
"""
    
    rows = []
    for config in configs_order:
        if config not in merged.index:
            continue
            
        row_data = merged.loc[config]
        
        # Format values
        # Tokens as integers (comma separated for thousands)
        # Time with 1 decimal place
        
        def fmt_int(x): return f"{x:,.0f}"
        def fmt_time(x): return f"{x:.1f}"
        
        ext_in = fmt_int(row_data['Mean_Input_Tokens_ext'])
        ext_out = fmt_int(row_data['Mean_Output_Tokens_ext'])
        ext_time = fmt_time(row_data['Mean_Time_Seconds_ext'])
        
        oml_in = fmt_int(row_data['Mean_Input_Tokens_oml'])
        oml_out = fmt_int(row_data['Mean_Output_Tokens_oml'])
        oml_time = fmt_time(row_data['Mean_Time_Seconds_oml'])
        
        tot_in = fmt_int(row_data['Mean_Input_Tokens_total'])
        tot_out = fmt_int(row_data['Mean_Output_Tokens_total'])
        tot_time = fmt_time(row_data['Mean_Time_Seconds_total'])
        
        # Latex row
        # \textbf{Base} & ...
        line = f"\\textbf{{{config}}} & {ext_in} & {ext_out} & {ext_time} & {oml_in} & {oml_out} & {oml_time} & {tot_in} & {tot_out} & {tot_time} \\\\"
        rows.append(line)
        
    latex_footer = r"""\bottomrule
\end{tabular}%
}
\end{table*}"""

    return latex_template + "\n".join(rows) + latex_footer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Latex table for resource costs")
    parser.add_argument('experiments', nargs='+', help="List of experiment directory paths")
    parser.add_argument('--output', '-o', help="Output file path (optional)")
    
    args = parser.parse_args()
    
    ext_df = load_data(args.experiments, "resource_usage_extraction.csv")
    oml_df = load_data(args.experiments, "resource_usage_oml.csv")
    
    if ext_df.empty or oml_df.empty:
        print("Error: Could not load data from one or both phases.", file=sys.stderr)
        sys.exit(1)
        
    latex_code = generate_latex(ext_df, oml_df)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(latex_code)
        print(f"Table saved to {args.output}")
    else:
        print(latex_code)
