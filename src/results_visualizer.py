"""
Standalone results visualizer for hyperparameter experiment data.
Easily load and visualize previous experiment results from CSV files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse


class ExperimentResultsVisualizer:
    """Visualize and analyze experiment results from CSV files."""
    
    def __init__(self, csv_file_path: str):
        """Initialize with path to CSV file containing experiment results."""
        self.csv_path = Path(csv_file_path)
        self.df = None
        self.successful_df = None
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the experiment data."""
        print(f"📊 Loading experiment data from: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
        # Filter successful experiments
        if 'success' in self.df.columns:
            self.successful_df = self.df[self.df['success'] == True].copy()
        else:
            # Assume all experiments are successful if no success column
            self.successful_df = self.df.copy()

        # If experiment_id present, allow base hash extraction for grouping
        if 'experiment_id' in self.successful_df.columns:
            self.successful_df['experiment_base_hash'] = self.successful_df['experiment_id'].str.split('_').str[0]
        
        print(f"✅ Loaded {len(self.df)} total experiments")
        print(f"✅ {len(self.successful_df)} successful experiments")
        
        if len(self.successful_df) == 0:
            print("⚠️  No successful experiments found for visualization")
    
    def print_summary_statistics(self):
        """Print summary statistics of the experiments."""
        if len(self.successful_df) == 0:
            print("❌ No successful experiments to analyze")
            return
        
        print("\n" + "="*60)
        print("📈 EXPERIMENT SUMMARY STATISTICS")
        print("="*60)
        
        # Basic metrics
        if 'extraction_rate' in self.successful_df.columns:
            print(f"📈 Extraction Rate:")
            print(f"   • Mean: {self.successful_df['extraction_rate'].mean():.2f}%")
            print(f"   • Std:  {self.successful_df['extraction_rate'].std():.2f}%")
            print(f"   • Min:  {self.successful_df['extraction_rate'].min():.2f}%")
            print(f"   • Max:  {self.successful_df['extraction_rate'].max():.2f}%")
        
        if 'total_time' in self.successful_df.columns:
            print(f"\n⏱️  Processing Time:")
            print(f"   • Mean: {self.successful_df['total_time'].mean():.2f}s")
            print(f"   • Std:  {self.successful_df['total_time'].std():.2f}s")
            print(f"   • Min:  {self.successful_df['total_time'].min():.2f}s")
            print(f"   • Max:  {self.successful_df['total_time'].max():.2f}s")
        
        if 'extracted_count' in self.successful_df.columns:
            print(f"\n🔍 Extracted Characteristics:")
            print(f"   • Mean: {self.successful_df['extracted_count'].mean():.1f}")
            print(f"   • Std:  {self.successful_df['extracted_count'].std():.1f}")
            print(f"   • Min:  {self.successful_df['extracted_count'].min()}")
            print(f"   • Max:  {self.successful_df['extracted_count'].max()}")
    
    def find_best_configurations(self):
        """Find and display the best configurations."""
        if len(self.successful_df) == 0:
            return
        
        print("\n" + "="*60)
        print("🏆 BEST CONFIGURATIONS")
        print("="*60)
        
        # Best extraction rate
        if 'extraction_rate' in self.successful_df.columns:
            best_extraction = self.successful_df.loc[self.successful_df['extraction_rate'].idxmax()]
            print(f"\n🏆 Best Extraction Rate: {best_extraction['extraction_rate']:.2f}%")
            self._print_config(best_extraction)
        
        # Fastest configuration
        if 'total_time' in self.successful_df.columns:
            fastest_config = self.successful_df.loc[self.successful_df['total_time'].idxmin()]
            print(f"\n⚡ Fastest Configuration: {fastest_config['total_time']:.2f}s")
            self._print_config(fastest_config)
        
        # Most extractions
        if 'extracted_count' in self.successful_df.columns:
            most_extractions = self.successful_df.loc[self.successful_df['extracted_count'].idxmax()]
            print(f"\n🔍 Most Characteristics Extracted: {most_extractions['extracted_count']}")
            self._print_config(most_extractions)
    
    def _print_config(self, config_row):
        """Print configuration parameters."""
        param_columns = ['chunk_size', 'chunk_overlap',  
                         'temperature']
        for param in param_columns:
            if param in config_row:
                print(f"   • {param}: {config_row[param]}")
    
    def create_overview_dashboard(self, save_path: Optional[str] = None):
        """Create a comprehensive overview dashboard."""
        if len(self.successful_df) == 0:
            print("❌ No data to visualize")
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Experiment Results Overview Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Extraction Rate Distribution
        if 'extraction_rate' in self.successful_df.columns:
            axes[0, 0].hist(self.successful_df['extraction_rate'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Extraction Rate (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Extraction Rates')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Processing Time Distribution
        if 'total_time' in self.successful_df.columns:
            axes[0, 1].hist(self.successful_df['total_time'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 1].set_xlabel('Processing Time (seconds)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Processing Times')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Extraction Rate vs Time Trade-off
        if 'extraction_rate' in self.successful_df.columns and 'total_time' in self.successful_df.columns:
            scatter = axes[0, 2].scatter(self.successful_df['total_time'], 
                                       self.successful_df['extraction_rate'],
                                       alpha=0.7, c=self.successful_df.get('chunk_size', 'blue'), 
                                       cmap='viridis', s=60)
            axes[0, 2].set_xlabel('Processing Time (seconds)')
            axes[0, 2].set_ylabel('Extraction Rate (%)')
            axes[0, 2].set_title('Extraction Rate vs Processing Time')
            axes[0, 2].grid(True, alpha=0.3)
            if 'chunk_size' in self.successful_df.columns:
                plt.colorbar(scatter, ax=axes[0, 2], label='Chunk Size')
        
        # 4. Parameter Impact - Chunk Size
        if 'chunk_size' in self.successful_df.columns and 'extraction_rate' in self.successful_df.columns:
            chunk_sizes = sorted(self.successful_df['chunk_size'].unique())
            extraction_by_chunk = [self.successful_df[self.successful_df['chunk_size'] == cs]['extraction_rate'].values 
                                 for cs in chunk_sizes]
            axes[1, 0].boxplot(extraction_by_chunk, labels=chunk_sizes)
            axes[1, 0].set_xlabel('Chunk Size')
            axes[1, 0].set_ylabel('Extraction Rate (%)')
            axes[1, 0].set_title('Extraction Rate by Chunk Size')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Parameter Impact - Temperature
        if 'temperature' in self.successful_df.columns and 'extraction_rate' in self.successful_df.columns:
            temperatures = sorted(self.successful_df['temperature'].unique())
            extraction_by_temp = [self.successful_df[self.successful_df['temperature'] == t]['extraction_rate'].values 
                                for t in temperatures]
            axes[1, 1].boxplot(extraction_by_temp, labels=temperatures)
            axes[1, 1].set_xlabel('Temperature')
            axes[1, 1].set_ylabel('Extraction Rate (%)')
            axes[1, 1].set_title('Extraction Rate by Temperature')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Correlation Heatmap
        numeric_cols = self.successful_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 2:
            corr_matrix = self.successful_df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1, 2], fmt='.2f', square=True)
            axes[1, 2].set_title('Parameter Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Dashboard saved to: {save_path}")
        
        plt.show()
    
    def create_parameter_analysis(self, save_path: Optional[str] = None):
        """Create detailed parameter analysis plots."""
        if len(self.successful_df) == 0:
            print("❌ No data to visualize")
            return
        
        # Check available parameters
        param_columns = ['chunk_size', 'chunk_overlap', 'temperature']
        available_params = [p for p in param_columns if p in self.successful_df.columns]
        
        if not available_params:
            print("❌ No parameter columns found")
            return
        
        n_params = len(available_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle('Parameter Impact Analysis', fontsize=16, fontweight='bold')
        
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, param in enumerate(available_params):
            if 'extraction_rate' in self.successful_df.columns:
                param_values = sorted(self.successful_df[param].unique())
                extraction_by_param = [
                    self.successful_df[self.successful_df[param] == val]['extraction_rate'].values 
                    for val in param_values
                ]
                
                # Create box plot
                box_plot = axes[i].boxplot(extraction_by_param, labels=param_values)
                axes[i].set_xlabel(param.replace('_', ' ').title())
                axes[i].set_ylabel('Extraction Rate (%)')
                axes[i].set_title(f'Extraction Rate by {param.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)
                
                # Add mean values as red diamonds
                means = [np.mean(data) for data in extraction_by_param]
                axes[i].scatter(range(1, len(means) + 1), means, 
                              color='red', marker='D', s=50, zorder=10, label='Mean')
                axes[i].legend()
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Parameter analysis saved to: {save_path}")
        
        plt.show()
    
    def create_time_series_analysis(self, save_path: Optional[str] = None):
        """Create time series analysis if experiment numbers are available."""
        if 'experiment_number' not in self.successful_df.columns:
            print("⚠️  No experiment_number column found for time series analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Time Series Analysis of Experiments', fontsize=16, fontweight='bold')
        
        # Sort by experiment number
        sorted_df = self.successful_df.sort_values('experiment_number')
        
        # 1. Extraction Rate Over Time
        if 'extraction_rate' in sorted_df.columns:
            axes[0, 0].plot(sorted_df['experiment_number'], sorted_df['extraction_rate'], 
                           marker='o', linestyle='-', alpha=0.7)
            axes[0, 0].set_xlabel('Experiment Number')
            axes[0, 0].set_ylabel('Extraction Rate (%)')
            axes[0, 0].set_title('Extraction Rate Over Experiments')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Processing Time Over Time
        if 'total_time' in sorted_df.columns:
            axes[0, 1].plot(sorted_df['experiment_number'], sorted_df['total_time'], 
                           marker='s', linestyle='-', alpha=0.7, color='red')
            axes[0, 1].set_xlabel('Experiment Number')
            axes[0, 1].set_ylabel('Processing Time (seconds)')
            axes[0, 1].set_title('Processing Time Over Experiments')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rolling Average of Extraction Rate
        if 'extraction_rate' in sorted_df.columns:
            window_size = min(5, len(sorted_df))
            rolling_avg = sorted_df['extraction_rate'].rolling(window=window_size).mean()
            axes[1, 0].plot(sorted_df['experiment_number'], sorted_df['extraction_rate'], 
                           'o-', alpha=0.5, label='Individual')
            axes[1, 0].plot(sorted_df['experiment_number'], rolling_avg, 
                           'r-', linewidth=2, label=f'Rolling Avg (window={window_size})')
            axes[1, 0].set_xlabel('Experiment Number')
            axes[1, 0].set_ylabel('Extraction Rate (%)')
            axes[1, 0].set_title('Extraction Rate with Rolling Average')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cumulative Best Performance
        if 'extraction_rate' in sorted_df.columns:
            cumulative_best = sorted_df['extraction_rate'].cummax()
            axes[1, 1].plot(sorted_df['experiment_number'], cumulative_best, 
                           marker='D', linestyle='-', color='green', linewidth=2)
            axes[1, 1].set_xlabel('Experiment Number')
            axes[1, 1].set_ylabel('Best Extraction Rate (%)')
            axes[1, 1].set_title('Cumulative Best Performance')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Time series analysis saved to: {save_path}")
        
        plt.show()
    
    def create_all_visualizations(self, output_dir: Optional[str] = None):
        """Create all available visualizations."""
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        else:
            output_path = self.csv_path.parent
        
        print("\n🎨 Creating comprehensive visualizations...")
        
        # Print summary statistics
        self.print_summary_statistics()
        
        # Find best configurations
        self.find_best_configurations()
        
        # Create visualizations
        self.create_overview_dashboard(output_path / "overview_dashboard.png")
        self.create_parameter_analysis(output_path / "parameter_analysis.png")
        self.create_time_series_analysis(output_path / "time_series_analysis.png")
        
        print(f"\n✅ All visualizations saved to: {output_path}")


def visualize_csv(csv_file_path: str, summary_only: bool = False, output_dir: Optional[str] = None):
    """
    Quick visualization function for CSV files - similar to visualize_results.py
    
    Args:
        csv_file_path: Path to the CSV file containing experiment results
        summary_only: If True, only show summary statistics without creating plots
        output_dir: Optional output directory for visualizations
    """
    try:
        print(f"📊 Loading experiment results from: {csv_file_path}")
        print("=" * 80)
        
        # Create visualizer
        visualizer = ExperimentResultsVisualizer(csv_file_path)
        
        if summary_only:
            # Only show summary and best configurations
            visualizer.print_summary_statistics()
            visualizer.find_best_configurations()
        else:
            # Create all visualizations
            if output_dir is None:
                output_dir = Path(csv_file_path).parent / "visualizations"
            visualizer.create_all_visualizations(str(output_dir))
        
        print("\n✅ Visualization completed!")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ CSV file not found: {csv_file_path}")
        print("\n💡 Available options:")
        print("   1. Run experiments first to generate results")
        print("   2. Specify the correct path to your CSV file")
        print("   3. Use the default path: hyperparameter_tuning/hyperparameter_experiments.csv")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main function for standalone usage."""
    parser = argparse.ArgumentParser(description='Visualize hyperparameter experiment results')
    parser.add_argument('csv_file', help='Path to the CSV file containing experiment results')
    parser.add_argument('--output-dir', '-o', help='Output directory for visualizations')
    parser.add_argument('--summary-only', '-s', action='store_true', 
                       help='Only print summary statistics, no visualizations')
    
    args = parser.parse_args()
    
    # Use the new visualize_csv function
    success = visualize_csv(args.csv_file, args.summary_only, args.output_dir)
    return 0 if success else 1


if __name__ == "__main__":
    main()
