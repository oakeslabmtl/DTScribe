"""Unified Results CLI: CSV visualization + experiment analysis with clean, extensible CLI.

This module merges the previous `results_visualizer.py` and `results_analyzer.py` into one
cohesive tool with subcommands and a simple visualization registry so you can easily add
or modify plots without touching the CLI wiring.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


# ---------------------------- CSV Visualizer ---------------------------- #
class ExperimentResultsVisualizer:
    """Reads an experiment CSV and produces textual + graphical summaries."""

    def __init__(self, csv_file_path: str):
        self.csv_path = Path(csv_file_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        if self.df.empty:
            raise ValueError("CSV file is empty – nothing to visualize.")
        self.successful_df = self._filter_success(self.df)
        if "experiment_id" in self.successful_df.columns:
            self.successful_df["config_hash"] = self.successful_df["experiment_id"].str.split("_").str[0]

    # ------------------------------------------------------------------
    def _filter_success(self, df: pd.DataFrame) -> pd.DataFrame:
        if "success" in df.columns:
            return df[df["success"] == True].copy()  # noqa: E712
        return df.copy()

    # ------------------------------------------------------------------
    def print_summary_statistics(self) -> None:
        if self.successful_df.empty:
            print("No successful experiment rows to summarize.")
            return
        print("\n" + "=" * 70)
        print("📈 SUMMARY STATISTICS")
        print("=" * 70)
        for col in ["extraction_rate", "total_time", "extracted_count"]:
            if col in self.successful_df.columns:
                series = self.successful_df[col].dropna()
                if not series.empty:
                    print(f"• {col}: mean={series.mean():.2f} | std={series.std():.2f} | min={series.min():.2f} | max={series.max():.2f}")
        print(f"Rows (total / successful): {len(self.df)} / {len(self.successful_df)}")

    def find_best_configurations(self) -> None:
        if self.successful_df.empty:
            return
        print("\n" + "=" * 70)
        print("🏆 BEST CONFIGURATIONS")
        print("=" * 70)
        sdf = self.successful_df
        if "extraction_rate" in sdf.columns:
            best = sdf.loc[sdf["extraction_rate"].idxmax()]
            print("Best extraction rate:")
            self._print_config(best)
        if "total_time" in sdf.columns:
            fast = sdf.loc[sdf["total_time"].idxmin()]
            print("\nFastest configuration:")
            self._print_config(fast)
        if {"extracted_count", "extraction_rate"}.issubset(sdf.columns):
            most = sdf.loc[sdf["extracted_count"].idxmax()]
            print("\nMost extracted characteristics:")
            self._print_config(most)

    def _print_config(self, row: pd.Series) -> None:
        for p in ["chunk_size", "chunk_overlap", "temperature"]:
            if p in row.index:
                print(f"  - {p}: {row[p]}")
        for m in ["extraction_rate", "total_time", "extracted_count"]:
            if m in row.index:
                print(f"  - {m}: {row[m]}")

    # ------------------------------------------------------------------
    def create_overview_dashboard(self, save_path: Optional[str] = None, show: bool = False) -> Optional[Path]:
        if self.successful_df.empty:
            print("Skipping dashboard – no successful rows.")
            return None
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("Experiment Results Overview", fontsize=16, fontweight="bold")
        df = self.successful_df

        # 1 Extraction Rate distribution
        ax = axes[0, 0]
        if "extraction_rate" in df.columns:
            ax.hist(df["extraction_rate"].dropna(), bins=15, color="steelblue", edgecolor="black")
            ax.set_title("Extraction Rate Distribution")
            ax.set_xlabel("Extraction Rate (%)")
        else:
            ax.axis("off")

        # 2 Time distribution
        ax = axes[0, 1]
        if "total_time" in df.columns:
            ax.hist(df["total_time"].dropna(), bins=15, color="seagreen", edgecolor="black")
            ax.set_title("Processing Time Distribution (s)")
            ax.set_xlabel("Seconds")
        else:
            ax.axis("off")

        # 3 Trade-off
        ax = axes[0, 2]
        if {"extraction_rate", "total_time"}.issubset(df.columns):
            scatter = ax.scatter(df["total_time"], df["extraction_rate"], c=df.get("chunk_size", pd.Series([0]*len(df))), cmap="viridis", alpha=0.75)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Extraction Rate (%)")
            ax.set_title("Rate vs Time (color=chunk_size)")
            if "chunk_size" in df.columns:
                plt.colorbar(scatter, ax=ax, label="chunk_size")
        else:
            ax.axis("off")

        # 4 Chunk size impact
        ax = axes[1, 0]
        if {"chunk_size", "extraction_rate"}.issubset(df.columns):
            sns.boxplot(x="chunk_size", y="extraction_rate", data=df, ax=ax)
            ax.set_title("Impact of chunk_size")
        else:
            ax.axis("off")

        # 5 Temperature impact
        ax = axes[1, 1]
        if {"temperature", "extraction_rate"}.issubset(df.columns):
            sns.boxplot(x="temperature", y="extraction_rate", data=df, ax=ax)
            ax.set_title("Impact of temperature")
        else:
            ax.axis("off")

        # 6 Correlation heatmap
        ax = axes[1, 2]
        numeric_cols = df.select_dtypes(include=[np.number])
        if numeric_cols.shape[1] >= 2:
            sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix")
        else:
            ax.axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path: Optional[Path] = None
        if save_path:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150)
            print(f"💾 Dashboard saved to {out_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return out_path

    def create_all_visualizations(self, output_dir: Optional[str] = None, show: bool = False) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        if output_dir:
            base = Path(output_dir)
            base.mkdir(parents=True, exist_ok=True)
            dash_path = base / "overview_dashboard.png"
            self.create_overview_dashboard(save_path=str(dash_path), show=show)
            outputs["overview_dashboard"] = dash_path
        else:
            self.create_overview_dashboard(show=show)
        return outputs


def visualize_csv(csv_file_path: str, summary_only: bool = False, output_dir: Optional[str] = None, show: bool = False) -> bool:
    try:
        viz = ExperimentResultsVisualizer(csv_file_path)
        viz.print_summary_statistics()
        viz.find_best_configurations()
        if not summary_only:
            viz.create_all_visualizations(output_dir, show=show)
        return True
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False

# ---------------------------- Results Analyzer ---------------------------- #
class ResultsAnalyzer:
    """Comprehensive analysis of experiment results."""

    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.analysis_dir = self.experiments_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True, parents=True)

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available experiment data."""
        data: Dict[str, pd.DataFrame] = {}

        # Load characteristics summary
        char_file = self.analysis_dir / "characteristics_summary.csv"
        if char_file.exists():
            df = pd.read_csv(char_file)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                data['characteristics'] = df

        # Load OML summary
        oml_file = self.analysis_dir / "oml_summary.csv"
        if oml_file.exists():
            df = pd.read_csv(oml_file)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                data['oml'] = df

        return data

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""

        data = self.load_all_data()
        if not data:
            return "No experiment data found. Run some experiments first."

        report: List[str] = []
        report.append("# Digital Twin Characteristics Extraction - Experiment Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Characteristics Extraction Analysis
        if 'characteristics' in data:
            char_df = data['characteristics']
            report.append("## Characteristics Extraction Analysis")
            report.append(f"- **Total Experiments**: {len(char_df)}")
            if 'extraction_rate' in char_df.columns:
                report.append(f"- **Average Extraction Rate**: {char_df['extraction_rate'].mean():.2f}% ± {char_df['extraction_rate'].std():.2f}%")
                report.append(f"- **Best Extraction Rate**: {char_df['extraction_rate'].max():.2f}%")
            if 'processing_time_seconds' in char_df.columns:
                report.append(f"- **Average Processing Time**: {char_df['processing_time_seconds'].mean():.2f}s")
            if 'average_description_length' in char_df.columns:
                report.append(f"- **Average Description Length**: {char_df['average_description_length'].mean():.0f} characters")
            report.append("")

            # Model performance comparison
            if 'model_name' in char_df.columns and len(char_df['model_name'].unique()) > 1:
                report.append("### Model Performance Comparison")
                model_stats = char_df.groupby('model_name').agg({
                    'extraction_rate': ['mean', 'std', 'count'] if 'extraction_rate' in char_df.columns else 'count',
                    'processing_time_seconds': ['mean', 'std'] if 'processing_time_seconds' in char_df.columns else 'count',
                    'average_description_length': 'mean' if 'average_description_length' in char_df.columns else 'count'
                }).round(2)
                report.append(model_stats.to_string())
                report.append("")

            # Hyperparameter impact analysis
            report.append("### Hyperparameter Impact Analysis")

            # Chunk size analysis
            if 'chunk_size' in char_df.columns and len(char_df['chunk_size'].unique()) > 1 and 'extraction_rate' in char_df.columns:
                chunk_analysis = char_df.groupby('chunk_size')['extraction_rate'].agg(['mean', 'std', 'count'])
                report.append("#### Chunk Size Impact on Extraction Rate")
                report.append(chunk_analysis.to_string())
                report.append("")

            # Temperature analysis
            if 'temperature' in char_df.columns and len(char_df['temperature'].unique()) > 1 and 'extraction_rate' in char_df.columns:
                temp_analysis = char_df.groupby('temperature')['extraction_rate'].agg(['mean', 'std', 'count'])
                report.append("#### Temperature Impact on Extraction Rate")
                report.append(temp_analysis.to_string())
                report.append("")

        # OML Generation Analysis
        if 'oml' in data:
            oml_df = data['oml']
            report.append("## OML Generation Analysis")
            report.append(f"- **Total OML Generations**: {len(oml_df)}")
            if 'oml_syntax_valid' in oml_df.columns and len(oml_df) > 0:
                report.append(f"- **Syntax Valid Rate**: {(oml_df['oml_syntax_valid'].sum() / len(oml_df) * 100):.1f}%")
            if 'oml_completeness_score' in oml_df.columns:
                report.append(f"- **Average Completeness Score**: {oml_df['oml_completeness_score'].mean():.2f}")
            if 'oml_instance_count' in oml_df.columns:
                report.append(f"- **Average Instance Count**: {oml_df['oml_instance_count'].mean():.1f}")
            if 'generation_time_seconds' in oml_df.columns:
                report.append(f"- **Average Generation Time**: {oml_df['generation_time_seconds'].mean():.2f}s")
            report.append("")

        # Quality trends over time
        if 'characteristics' in data:
            char_df = data['characteristics']
            if len(char_df) > 1 and 'extraction_rate' in char_df.columns and 'timestamp' in char_df.columns:
                report.append("## Quality Trends Over Time")
                recent_threshold = datetime.now() - timedelta(days=7)
                recent_df = char_df[char_df['timestamp'] > recent_threshold]
                older_df = char_df[char_df['timestamp'] <= recent_threshold]
                if len(recent_df) > 0 and len(older_df) > 0:
                    report.append(f"- **Recent (7 days) avg extraction rate**: {recent_df['extraction_rate'].mean():.2f}%")
                    report.append(f"- **Older experiments avg extraction rate**: {older_df['extraction_rate'].mean():.2f}%")
                    improvement = recent_df['extraction_rate'].mean() - older_df['extraction_rate'].mean()
                    if improvement > 0:
                        report.append(f"- **Improvement**: +{improvement:.2f}% 📈")
                    else:
                        report.append(f"- **Change**: {improvement:.2f}% 📉")
                    report.append("")

        # Recommendations
        report.append("## Recommendations")
        if 'characteristics' in data:
            char_df = data['characteristics']
            if 'extraction_rate' in char_df.columns and len(char_df) > 0:
                best_exp = char_df.loc[char_df['extraction_rate'].idxmax()]
                report.append("### Best Performing Configuration")
                if 'model_name' in best_exp:
                    report.append(f"- **Model**: {best_exp['model_name']}")
                for p in ['chunk_size', 'chunk_overlap', 'temperature']:
                    if p in best_exp:
                        report.append(f"- **{p.replace('_', ' ').title()}**: {best_exp[p]}")
                report.append(f"- **Extraction Rate**: {best_exp['extraction_rate']:.2f}%")
                report.append("")

                # Correlation analysis
                numeric_cols = [c for c in ['chunk_size', 'chunk_overlap', 'temperature', 'extraction_rate'] if c in char_df.columns]
                if len(numeric_cols) > 2:
                    corr_with_extraction = char_df[numeric_cols].corr()['extraction_rate'].abs().sort_values(ascending=False)
                    report.append("### Factors most correlated with extraction rate:")
                    for factor, correlation in corr_with_extraction.items():
                        if factor != 'extraction_rate':
                            report.append(f"- **{factor}**: {correlation:.3f}")
                    report.append("")

            # Error analysis
            if 'error_count' in char_df.columns and len(char_df) > 0:
                error_rate = (char_df['error_count'] > 0).sum() / len(char_df) * 100
                report.append(f"### Error Analysis")
                report.append(f"- **Experiments with errors**: {error_rate:.1f}%")
                if error_rate > 0:
                    report.append("- **Recommendation**: Review error logs for common issues")
                report.append("")

        return "\n".join(report)

    # --- Dashboards ---
    def _create_characteristics_dashboard(self, char_df: pd.DataFrame, show: bool = False) -> Path:
        """Create characteristics extraction dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Characteristics Extraction Performance Dashboard', fontsize=16)

        # 1. Extraction rate over time
        if 'timestamp' in char_df.columns and 'extraction_rate' in char_df.columns:
            axes[0, 0].plot(char_df['timestamp'], char_df['extraction_rate'], marker='o', alpha=0.7)
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Extraction Rate (%)')
            axes[0, 0].set_title('Extraction Rate Over Time')
            axes[0, 0].tick_params(axis='x', rotation=45)
        else:
            axes[0, 0].axis('off')

        # 2. Processing time vs extraction rate
        if 'processing_time_seconds' in char_df.columns and 'extraction_rate' in char_df.columns and 'chunk_size' in char_df.columns:
            scatter = axes[0, 1].scatter(char_df['processing_time_seconds'], char_df['extraction_rate'],
                                         c=char_df['chunk_size'], cmap='viridis', alpha=0.7)
            axes[0, 1].set_xlabel('Processing Time (seconds)')
            axes[0, 1].set_ylabel('Extraction Rate (%)')
            axes[0, 1].set_title('Processing Time vs Extraction Rate')
            plt.colorbar(scatter, ax=axes[0, 1], label='Chunk Size')
        else:
            axes[0, 1].axis('off')

        # 3. Model comparison (if multiple models)
        if 'model_name' in char_df.columns and 'extraction_rate' in char_df.columns and len(char_df['model_name'].unique()) > 1:
            model_stats = char_df.groupby('model_name')['extraction_rate'].mean().sort_values(ascending=True)
            axes[0, 2].barh(model_stats.index, model_stats.values)
            axes[0, 2].set_xlabel('Average Extraction Rate (%)')
            axes[0, 2].set_title('Model Performance Comparison')
        else:
            axes[0, 2].text(0.5, 0.5, 'Single Model or missing columns', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Model Performance')

        # 4. Hyperparameter impact - Chunk size
        if 'chunk_size' in char_df.columns and 'extraction_rate' in char_df.columns and len(char_df['chunk_size'].unique()) > 1:
            chunk_data = [char_df[char_df['chunk_size'] == cs]['extraction_rate'].values
                          for cs in sorted(char_df['chunk_size'].unique())]
            axes[1, 0].boxplot(chunk_data, labels=sorted(char_df['chunk_size'].unique()))
            axes[1, 0].set_xlabel('Chunk Size')
            axes[1, 0].set_ylabel('Extraction Rate (%)')
            axes[1, 0].set_title('Chunk Size Impact')
        else:
            axes[1, 0].text(0.5, 0.5, 'Single Chunk Size or missing columns', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Chunk Size Impact')

        # 5. Temperature impact
        if 'temperature' in char_df.columns and 'extraction_rate' in char_df.columns and len(char_df['temperature'].unique()) > 1:
            temp_stats = char_df.groupby('temperature')['extraction_rate'].mean()
            axes[1, 1].plot(temp_stats.index, temp_stats.values, marker='o', linewidth=2)
            axes[1, 1].set_xlabel('Temperature')
            axes[1, 1].set_ylabel('Average Extraction Rate (%)')
            axes[1, 1].set_title('Temperature Impact')
        else:
            axes[1, 1].text(0.5, 0.5, 'Single Temperature or missing columns', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Temperature Impact')

        # 6. Quality distribution
        if 'extraction_rate' in char_df.columns:
            axes[1, 2].hist(char_df['extraction_rate'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 2].axvline(char_df['extraction_rate'].mean(), color='red', linestyle='--', label='Mean')
            axes[1, 2].set_xlabel('Extraction Rate (%)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Extraction Rate Distribution')
            axes[1, 2].legend()
        else:
            axes[1, 2].axis('off')

        plt.tight_layout()
        dashboard_file = self.analysis_dir / "characteristics_dashboard.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        print(f"📊 Characteristics dashboard saved to: {dashboard_file}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return dashboard_file

    def _create_oml_dashboard(self, oml_df: pd.DataFrame, show: bool = False) -> Path:
        """Create OML generation dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('OML Generation Performance Dashboard', fontsize=16)

        # 1. Syntax validity over time
        if 'timestamp' in oml_df.columns and 'oml_syntax_valid' in oml_df.columns:
            axes[0, 0].plot(oml_df['timestamp'], oml_df['oml_syntax_valid'].astype(int), marker='o', alpha=0.7)
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Syntax Valid (1=Yes, 0=No)')
            axes[0, 0].set_title('OML Syntax Validity Over Time')
            axes[0, 0].tick_params(axis='x', rotation=45)
        else:
            axes[0, 0].axis('off')

        # 2. Completeness score distribution
        if 'oml_completeness_score' in oml_df.columns:
            axes[0, 1].hist(oml_df['oml_completeness_score'], bins=15, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(oml_df['oml_completeness_score'].mean(), color='red', linestyle='--', label='Mean')
            axes[0, 1].set_xlabel('Completeness Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('OML Completeness Score Distribution')
            axes[0, 1].legend()
        else:
            axes[0, 1].axis('off')

        # 3. Instance count vs completeness
        if 'oml_instance_count' in oml_df.columns and 'oml_completeness_score' in oml_df.columns:
            axes[1, 0].scatter(oml_df['oml_instance_count'], oml_df['oml_completeness_score'], alpha=0.7)
            axes[1, 0].set_xlabel('Instance Count')
            axes[1, 0].set_ylabel('Completeness Score')
            axes[1, 0].set_title('Instance Count vs Completeness')
        else:
            axes[1, 0].axis('off')

        # 4. Generation time distribution
        if 'generation_time_seconds' in oml_df.columns:
            axes[1, 1].hist(oml_df['generation_time_seconds'], bins=15, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(oml_df['generation_time_seconds'].mean(), color='red', linestyle='--', label='Mean')
            axes[1, 1].set_xlabel('Generation Time (seconds)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('OML Generation Time Distribution')
            axes[1, 1].legend()
        else:
            axes[1, 1].axis('off')

        plt.tight_layout()
        dashboard_file = self.analysis_dir / "oml_dashboard.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        print(f"📊 OML dashboard saved to: {dashboard_file}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return dashboard_file

    def _create_comparison_dashboard(self, char_df: pd.DataFrame, oml_df: pd.DataFrame, show: bool = False) -> Optional[Path]:
        """Create comparison dashboard between characteristics and OML tasks."""
        if 'characteristics_experiment_id' not in oml_df.columns:
            return None
        merged_df = char_df.merge(
            oml_df,
            left_on='experiment_id',
            right_on='characteristics_experiment_id',
            how='inner',
            suffixes=('_char', '_oml')
        )
        if len(merged_df) == 0:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Characteristics vs OML Performance Comparison', fontsize=16)

        # 1. Extraction rate vs OML completeness
        if 'extraction_rate' in merged_df.columns and 'oml_completeness_score' in merged_df.columns:
            axes[0].scatter(merged_df['extraction_rate'], merged_df['oml_completeness_score'], alpha=0.7)
            axes[0].set_xlabel('Characteristics Extraction Rate (%)')
            axes[0].set_ylabel('OML Completeness Score')
            axes[0].set_title('Extraction Quality vs OML Quality')
            corr = merged_df['extraction_rate'].corr(merged_df['oml_completeness_score'])
            axes[0].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[0].transAxes,
                         bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        else:
            axes[0].axis('off')

        # 2. Processing time comparison
        if 'processing_time_seconds' in merged_df.columns and 'generation_time_seconds' in merged_df.columns:
            time_comparison = merged_df[['processing_time_seconds', 'generation_time_seconds']].mean()
            axes[1].bar(time_comparison.index, time_comparison.values)
            axes[1].set_ylabel('Average Time (seconds)')
            axes[1].set_title('Processing Time Comparison')
            axes[1].set_xticklabels(['Characteristics\nExtraction', 'OML\nGeneration'], rotation=45)
        else:
            axes[1].axis('off')

        plt.tight_layout()
        comparison_file = self.analysis_dir / "comparison_dashboard.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison dashboard saved to: {comparison_file}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return comparison_file

    # Public API for dashboards
    def create_dashboard_visualizations(self, which: Optional[List[str]] = None, show: bool = False) -> Dict[str, Optional[Path]]:
        """Create selected dashboards. which can include: ['characteristics', 'oml', 'comparison']."""
        outputs: Dict[str, Optional[Path]] = {}
        data = self.load_all_data()
        if not data:
            print("No experiment data found. Run some experiments first.")
            return outputs
        which = which or ['characteristics', 'oml', 'comparison']
        if 'characteristics' in which and 'characteristics' in data:
            outputs['characteristics'] = self._create_characteristics_dashboard(data['characteristics'], show=show)
        if 'oml' in which and 'oml' in data:
            outputs['oml'] = self._create_oml_dashboard(data['oml'], show=show)
        if 'comparison' in which and 'characteristics' in data and 'oml' in data:
            outputs['comparison'] = self._create_comparison_dashboard(data['characteristics'], data['oml'], show=show)
        return outputs

    def export_research_summary(self) -> str:
        report = self.generate_comprehensive_report()
        report_file = self.analysis_dir / f"research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Research summary saved to: {report_file}")
        return str(report_file)


# ---------------------------- Visualization Registry ---------------------------- #
# This makes it easy to add new CSV visualizations without touching CLI code.
CSV_VISUALIZATIONS: Dict[str, Callable[[ExperimentResultsVisualizer, Path, bool], Optional[Path]]] = {}

def register_csv_viz(name: str, fn: Callable[[ExperimentResultsVisualizer, Path, bool], Optional[Path]]) -> None:
    CSV_VISUALIZATIONS[name] = fn


def _viz_overview(viz: ExperimentResultsVisualizer, out_dir: Path, show: bool) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return viz.create_overview_dashboard(save_path=str(out_dir / 'overview_dashboard.png'), show=show)


# Register default visualizations
register_csv_viz('overview', _viz_overview)


# ---------------------------- CLI ---------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Unified Results Tool: CSV visualization and experiment analysis')
    sub = parser.add_subparsers(dest='command', required=True)

    # csv visualize subcommand
    p_csv = sub.add_parser('csv', help='Visualize a single experiment CSV')
    p_csv.add_argument('csv_file', help='Path to the CSV file containing experiment results')
    p_csv.add_argument('--output-dir', '-o', help='Directory to save plots (optional)')
    p_csv.add_argument('--summary-only', '-s', action='store_true', help='Only print summary statistics')
    p_csv.add_argument('--show', action='store_true', help='Show plots interactively')
    p_csv.add_argument('--which', nargs='*', default=['overview'], help=f"CSV visualizations to generate (default: overview). Available: {list(CSV_VISUALIZATIONS.keys())}")

    # analyze subcommand
    p_an = sub.add_parser('analyze', help='Analyze experiments/analysis summaries and create report/dashboards')
    p_an.add_argument('--experiments-dir', '-e', default='experiments', help='Experiments root directory (default: experiments)')
    p_an.add_argument('--report', action='store_true', help='Export research summary markdown')
    p_an.add_argument('--dashboards', nargs='*', help="Dashboards to create: characteristics oml comparison (default: all)")
    p_an.add_argument('--show', action='store_true', help='Show plots interactively')

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'csv':
        # Run CSV textual summary + selected visualizations
        ok = visualize_csv(args.csv_file, args.summary_only, args.output_dir, show=args.show)
        if not ok:
            return 1
        if not args.summary_only and args.output_dir and args.which:
            viz = ExperimentResultsVisualizer(args.csv_file)
            out_dir = Path(args.output_dir)
            for name in args.which:
                fn = CSV_VISUALIZATIONS.get(name)
                if fn is None:
                    print(f"⚠️  Unknown CSV visualization '{name}'. Skipping.")
                    continue
                fn(viz, out_dir, args.show)
        return 0

    if args.command == 'analyze':
        analyzer = ResultsAnalyzer(args.experiments_dir)
        # Report
        if args.report:
            analyzer.export_research_summary()
        # Dashboards
        which_dash = args.dashboards or ['characteristics', 'oml', 'comparison']
        analyzer.create_dashboard_visualizations(which=which_dash, show=args.show)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
