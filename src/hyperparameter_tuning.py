"""
Hyperparameter tuning and evaluation script for research purposes.
Demonstrates how to run multiple experiments with different configurations.
"""

import itertools
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import time

from main import ExtractionPipelineFactory
from experiment_tracking import ExperimentConfig


class HyperparameterTuner:
    """Runs multiple experiments with different hyperparameter configurations."""
    
    def __init__(self, pdf_path: str, output_dir: str = "hyperparameter_tuning"):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create orchestrator with experiment tracking
        self.orchestrator = ExtractionPipelineFactory.create_orchestrator(with_experiment_tracking=True)
    
    def define_hyperparameter_grid(self) -> List[Dict[str, Any]]:
        """Define the hyperparameter grid to search."""
        
        # Define parameter ranges
        param_grid = {
            'model': ["llama 3.2", "qwen3:4b", "qwen3:8b"],
            'chunk_size': [2000, 2500, 3000],
            'temperature': [0.1, 0.3],
        }
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        return combinations
    
     # def run_experiment_batch(self, max_experiments: int = 10, use_individual_extraction: bool = False) -> List[Dict[str, Any]]:
     #     """Run a batch of experiments with different hyperparameters and extraction strategies."""

    def run_experiment_batch(self, max_experiments: int = 10) -> List[Dict[str, Any]]:
        """Run a batch of experiments with different hyperparameters."""     
   
        hyperparameter_combinations = self.define_hyperparameter_grid()
        # Limit the number of experiments for demonstration
        if max_experiments > 0:
            hyperparameter_combinations = hyperparameter_combinations[:max_experiments]
        
        print(f"🧪 Running {len(hyperparameter_combinations)} experiments...")

        experiment_results = []
        
        for i, params in enumerate(hyperparameter_combinations, 1):
            print(f"\n{'='*60}")
            print(f"🔬 Experiment {i}/{len(hyperparameter_combinations)}")
            print(f"📊 Parameters: {params}")
            print(f"{'='*60}")

            try:
                # Create configuration
                config = ExtractionPipelineFactory.create_config(
                    **params,
                    custom_params={
                        "experiment_batch": "hyperparameter_tuning",
                        "experiment_number": i,
                        # "extraction_strategy": "individual" if use_individual_extraction else "batch"
                    }
                )

                # Run extraction
                start_time = time.time()

                results = self.orchestrator.run_extraction(
                    self.pdf_path,
                    config=config,
                    save_results=True,
                    # use_individual_extraction=use_individual_extraction
                )
                experiment_time = time.time() - start_time
                
                # Analyze results
                quality_metrics = self.orchestrator.analyze_characteristic_extraction(results)
                
                # Store experiment result
                experiment_result = {
                    'experiment_number': i,
                    'total_time': experiment_time,
                    **params,
                    **quality_metrics,
                    'success': True,
                    'error': None,
                    # 'extraction_strategy': "individual" if use_individual_extraction else "batch"
                }
                
                experiment_results.append(experiment_result)
                print(f"✅ Experiment {i} completed successfully (id stored with hash_timestamp format)")
                print(f"   📈 Extraction Rate: {quality_metrics['extraction_rate']:.1f}%")
                print(f"   ⏱️  Total Time: {experiment_time:.2f}s")
            except Exception as e:
                print(f"❌ Experiment {i} failed: {str(e)}")
                experiment_result = {
                    'experiment_number': i,
                    **params,
                    'success': False,
                    'error': str(e),
                    'extraction_rate': 0.0,
                    'total_time': 0.0,
                    # 'extraction_strategy': "individual" if use_individual_extraction else "batch"
                }
                experiment_results.append(experiment_result)
        
        return experiment_results
    
    def analyze_and_visualize_results(self, experiment_results: List[Dict[str, Any]]):
        """Analyze experiment results and create visualizations."""
        
        # Convert to DataFrame
        df = pd.DataFrame(experiment_results)
        
        # Save detailed results
        results_file = self.output_dir / "hyperparameter_experiments.csv"
        df.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Filter successful experiments
        successful_df = df[df['success'] == True]
        
        if len(successful_df) == 0:
            print("❌ No successful experiments to analyze")
            return
        
        print(f"\n📊 Analysis of {len(successful_df)} successful experiments:")
        
        # Basic statistics
        print(f"   📈 Extraction Rate: {successful_df['extraction_rate'].mean():.2f}% ± {successful_df['extraction_rate'].std():.2f}%")
        print(f"   ⏱️  Average Time: {successful_df['total_time'].mean():.2f}s ± {successful_df['total_time'].std():.2f}s")
        print(f"   🏆 Best Extraction Rate: {successful_df['extraction_rate'].max():.2f}%")
        print(f"   ⚡ Fastest Time: {successful_df['total_time'].min():.2f}s")
        
        # Find best configuration
        best_extraction = successful_df.loc[successful_df['extraction_rate'].idxmax()]
        fastest_config = successful_df.loc[successful_df['total_time'].idxmin()]
        
        print(f"\n🏆 Best Extraction Rate Configuration:")
        for param in ['chunk_size', 'model', 'temperature']:
            print(f"   • {param}: {best_extraction[param]}")
        print(f"   • Extraction Rate: {best_extraction['extraction_rate']:.2f}%")

        print(f"\n⚡ Fastest Configuration:")
        for param in ['chunk_size', 'model', 'temperature']:
            print(f"   • {param}: {fastest_config[param]}")
        print(f"   • Time: {fastest_config['total_time']:.2f}s")
        
        # Create visualizations
        self._create_visualizations(successful_df)
    
    def _create_visualizations(self, df: pd.DataFrame):
        """Create various visualizations of the results."""
        
        plt.style.use('default')
        
        # 1. Extraction Rate vs Time Trade-off
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hyperparameter Tuning Results', fontsize=16)
        
        # Scatter plot: Extraction Rate vs Time
        axes[0, 0].scatter(df['total_time'], df['extraction_rate'], alpha=0.7, c=df['chunk_size'], cmap='viridis')
        axes[0, 0].set_xlabel('Total Time (seconds)')
        axes[0, 0].set_ylabel('Extraction Rate (%)')
        axes[0, 0].set_title('Extraction Rate vs Processing Time')
        cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
        cbar.set_label('Chunk Size')
        
        # Box plot: Extraction Rate by Chunk Size
        chunk_sizes = sorted(df['chunk_size'].unique())
        extraction_by_chunk = [df[df['chunk_size'] == cs]['extraction_rate'].values for cs in chunk_sizes]
        axes[0, 1].boxplot(extraction_by_chunk, labels=chunk_sizes)
        axes[0, 1].set_xlabel('Chunk Size')
        axes[0, 1].set_ylabel('Extraction Rate (%)')
        axes[0, 1].set_title('Extraction Rate by Chunk Size')
        
        # Box plot: Processing Time by Retrieval K
        # Removed box plot for Retrieval K since retrieval_k is no longer a parameter
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / "hyperparameter_analysis.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualizations saved to: {viz_file}")
        
        plt.show()
        
        # 2. Correlation Analysis
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Select numeric columns for correlation
        numeric_cols = ['chunk_size', 'temperature', 'extraction_rate', 'total_time', 'extracted_count']
        corr_matrix = df[numeric_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Hyperparameter Correlation Matrix')
        
        plt.tight_layout()
        corr_file = self.output_dir / "correlation_matrix.png"
        plt.savefig(corr_file, dpi=300, bbox_inches='tight')
        print(f"🔗 Correlation matrix saved to: {corr_file}")
        
        plt.show()


def visualize_existing_results(csv_file_path: str = None, output_dir: str = None):
    """Convenience function to visualize existing experiment results."""
    from results_visualizer import ExperimentResultsVisualizer
    
    if csv_file_path is None:
        csv_file_path = "hyperparameter_tuning/hyperparameter_experiments.csv"
    
    print(f"📊 Visualizing existing results from: {csv_file_path}")
    print("=" * 80)
    
    try:
        visualizer = ExperimentResultsVisualizer(csv_file_path)
        visualizer.create_all_visualizations(output_dir)
        print("\n✅ Visualization completed!")
        
    except FileNotFoundError:
        print(f"❌ CSV file not found: {csv_file_path}")
        print("💡 Run experiments first or check the file path.")
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")


def main():
    """Main function for hyperparameter tuning."""
    import sys
    
    # Check if user wants to visualize existing results
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        csv_path = sys.argv[2] if len(sys.argv) > 2 else None
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        visualize_existing_results(csv_path, output_dir)
        return
    
    # Configuration
    pdf_path = "data/papers/The Incubator Case Study for Digital Twin Engineering.pdf"
    
    print("🧪 Starting Hyperparameter Tuning for Digital Twin Characteristics Extraction")
    print("=" * 80)
    
    # Create tuner
    tuner = HyperparameterTuner(pdf_path)

    # Run experiments
    # experiment_results = tuner.run_experiment_batch(max_experiments=-1)  # Run all combinations
    experiment_results = tuner.run_experiment_batch(max_experiments=-1) 
    
    # Analyze and visualize results
    tuner.analyze_and_visualize_results(experiment_results)
    
    print("\n✅ Hyperparameter tuning completed!")
    print("💡 Use the results to select optimal hyperparameters for your specific use case.")


if __name__ == "__main__":
    main()
