"""Experiment results visualization utilities (SOLID-aligned minimal design)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    def create_overview_dashboard(self, save_path: Optional[str] = None) -> Optional[Path]:
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
        return out_path

    def create_all_visualizations(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        if output_dir:
            base = Path(output_dir)
            base.mkdir(parents=True, exist_ok=True)
            dash_path = base / "overview_dashboard.png"
            self.create_overview_dashboard(save_path=str(dash_path))
            outputs["overview_dashboard"] = dash_path
        else:
            self.create_overview_dashboard()
        return outputs


def visualize_csv(csv_file_path: str, summary_only: bool = False, output_dir: Optional[str] = None) -> bool:
    try:
        viz = ExperimentResultsVisualizer(csv_file_path)
        viz.print_summary_statistics()
        viz.find_best_configurations()
        if not summary_only:
            viz.create_all_visualizations(output_dir)
        return True
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False


def main() -> int:
    """Main function for standalone usage."""
    parser = argparse.ArgumentParser(description='Visualize hyperparameter experiment results')
    parser.add_argument('csv_file', help='Path to the CSV file containing experiment results')
    parser.add_argument('--output-dir', '-o', help='Directory to save plots (optional)')
    parser.add_argument('--summary-only', '-s', action='store_true', help='Only print summary statistics')
    args = parser.parse_args()
    success = visualize_csv(args.csv_file, args.summary_only, args.output_dir)
    return 0 if success else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
