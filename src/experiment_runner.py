"""
Runs multiple experiments with different configurations.
"""

import itertools
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import time

from main import ExtractionPipelineFactory, ExtractionOrchestrator

class ExperimentRunner:
    """Runs multiple experiments with different parameter configurations."""
    
    def __init__(self, pdf_path: str, orchestrator: ExtractionOrchestrator):
        self.pdf_path = pdf_path
        self.orchestrator = orchestrator

    def define_combinations_from_parameter_grid(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Define the parameter grid to search."""
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        return combinations

    def run_experiment_batch(self, max_experiments: int = 10, experiment_name: str = "default", param_grid: Dict[str, Any] = None, repeat_experiments: int = 1, mode: str = "both") -> List[Dict[str, Any]]:
        """Run a batch of experiments with different parameters."""

        self.output_dir = Path(experiment_name)
        self.output_dir.mkdir(exist_ok=True)

        if param_grid is None:
            print("❌ No parameter grid provided for experiments.")
            return []
        parameter_combinations = self.define_combinations_from_parameter_grid(param_grid)

        # Limit the number of experiments for demonstration
        if max_experiments > 0:
            parameter_combinations = parameter_combinations[:max_experiments]

        print(f"🧪 Running {len(parameter_combinations)} experiments...")

        experiment_results = []
        total_experiments = len(parameter_combinations) * repeat_experiments
        experiment_counter = 0

        for combo_idx, params in enumerate(parameter_combinations, 1):
            for rep in range(repeat_experiments):
                experiment_counter += 1
                print(f"\n{'='*60}")
                print(f"🔬 Experiment {combo_idx}/{len(parameter_combinations)}")
                print(f"➕ Experiment counter: {experiment_counter}/{total_experiments}")
                print(f"📊 Parameters: {params}")
                print(f"🔄 Repetition: {rep + 1}/{repeat_experiments}")
                print(f"{'='*60}")

                try:                   
                    # Create configuration
                    config = ExtractionPipelineFactory.create_config(
                        **params,
                        custom_params={
                            "experiment_batch": experiment_name,
                            "experiment_number": experiment_counter,
                            "parameter_combination": combo_idx,
                            "repetition": rep + 1,
                        }
                    )

                    # Initialize pipeline
                    self.orchestrator.initialize_pipeline(self.pdf_path, config=config)

                    # Run extraction
                    start_time = time.time()

                    extraction_results = {}
                    oml_results = {}
                    experiment_id = None
                    if mode in ("extraction", "both"):
                        extraction_results = self.orchestrator.run_extraction(
                            pdf_path=self.pdf_path,
                            config=config,
                            save_results=True,
                            experiment_id=None
                        )
                        experiment_id = self.orchestrator.last_experiment_id

                    if mode in ("oml", "both"):
                        oml_results = self.orchestrator.run_oml_generation(
                            config=config,
                            experiment_id=experiment_id,
                            save_results=True,
                            source_experiment_id=experiment_id
                        )
                        oml_output = oml_results.get("oml_output")
                        if not oml_output:
                            print("No OML generated.")
                    
                    experiment_time = time.time() - start_time

                    # Analyze results
                    if extraction_results.get("extracted_characteristics"):
                        quality_metrics = self.orchestrator.analyze_characteristic_extraction(extraction_results)
                        print("\n📈 Extraction Quality:")
                        print(f" - Extraction Rate: {quality_metrics['extraction_rate']:.2f}%")
                        print(f" - Extracted: {quality_metrics['extracted_count']}/{quality_metrics['total_characteristics']}")
                                
                        # Store experiment result
                        experiment_result = {
                            'experiment_number': experiment_counter,
                            'parameter_combination': combo_idx,
                            'repetition': rep + 1,
                            'total_time': experiment_time,
                            **params,
                            **quality_metrics,
                            'success': True,
                            'error': None,
                        }
                        
                        experiment_results.append(experiment_result)
                        print(f"✅ Experiment {experiment_counter} completed successfully (id stored with hash_timestamp format)")
                        print(f"   📈 Extraction Rate: {quality_metrics['extraction_rate']:.1f}%")
                        print(f"   ⏱️  Total Time: {experiment_time:.2f}s")
                except Exception as e:
                    print(f"❌ Experiment {experiment_counter} failed: {str(e)}")
                    experiment_result = {
                        'experiment_number': experiment_counter,
                        'parameter_combination': combo_idx,
                        'repetition': rep + 1,
                        **params,
                        'success': False,
                        'error': str(e),
                        'extraction_rate': 0.0,
                        'total_time': 0.0,
                    }
                    experiment_results.append(experiment_result)
        print(f"\n✅ Parameter tuning named {experiment_name} completed!")
        print(f"📊 Ran {len(parameter_combinations)} parameter combinations × {repeat_experiments} repetitions = {total_experiments} total experiments")
        return experiment_results
    
    def analyze_and_visualize_results(self, experiment_results: List[Dict[str, Any]]):
        """Analyze experiment results and create visualizations."""
        
        # Convert to DataFrame
        df = pd.DataFrame(experiment_results)
        
        # Save detailed results
        results_file = self.output_dir / "experiments.csv"
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
        print(f"   ⏱️ Average Time: {successful_df['total_time'].mean():.2f}s ± {successful_df['total_time'].std():.2f}s")
        print(f"   🏆 Best Extraction Rate: {successful_df['extraction_rate'].max():.2f}%")
        print(f"   ⚡ Fastest Time: {successful_df['total_time'].min():.2f}s")
        
        # Find best configuration
        best_extraction = successful_df.loc[successful_df['extraction_rate'].idxmax()]
        fastest_config = successful_df.loc[successful_df['total_time'].idxmin()]
        
        print(f"\n🏆 Best Extraction Rate Configuration:")
        for param in ['chunk_size', 'model_name', 'temperature']:
            if param in best_extraction:
                print(f"   • {param}: {best_extraction[param]}")
        print(f"   • Extraction Rate: {best_extraction['extraction_rate']:.2f}%")

        print(f"\n⚡ Fastest Configuration:")
        for param in ['chunk_size', 'model_name', 'temperature']:
            if param in fastest_config:
                print(f"   • {param}: {fastest_config[param]}")
        print(f"   • Time: {fastest_config['total_time']:.2f}s")
        
        # Create visualizations
        self._create_visualizations(successful_df)
    
    def _create_visualizations(self, df: pd.DataFrame):
        """Create various visualizations of the results."""
        
        plt.style.use('default')
        
        # 1. Extraction Rate vs Time Trade-off
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parameter Tuning Results', fontsize=16)
        
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
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / "analysis.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualizations saved to: {viz_file}")
        
        plt.show()
        
        # 2. Correlation Analysis
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Select numeric columns for correlation
        numeric_cols = ['chunk_size', 'temperature', 'extraction_rate', 'total_time', 'extracted_count']
        corr_matrix = df[numeric_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Parameter Correlation Matrix')
        
        plt.tight_layout()
        corr_file = self.output_dir / "correlation_matrix.png"
        plt.savefig(corr_file, dpi=300, bbox_inches='tight')
        print(f"🔗 Correlation matrix saved to: {corr_file}")
        
        plt.show()


def visualize_results(csv_file_path: str = None, output_dir: str = None):
    """Convenience function to visualize existing experiment results."""
    from results_visualizer import ExperimentResultsVisualizer
    
    if csv_file_path is None:
        print("❌ Please provide the path to the CSV file containing experiment results.")
        return
    
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
    import sys
    
    # Configuration
    pdf_path = "data/papers/The Incubator Case Study for Digital Twin Engineering.pdf"

    print("🧪 Starting Parameter Exploration for Digital Twin Characteristics Extraction")
    print("=" * 80)
    
    # Create experiment runner
    orchestrator = ExtractionPipelineFactory.create_orchestrator(with_experiment_tracking=True)
    runner = ExperimentRunner(pdf_path, orchestrator=orchestrator)

    # Run experiments
    # param_grid = {
    #     'model_name': ["qwen3:8b"],
    #     'chunk_size': [1000, 1500, 2000, 2500, 3000, 3500],
    #     'temperature': [0.1, 0.15, 0.2, 0.25, 0.3],
    #     "embedding_model": ["embeddinggemma"],
    #     "chunk_overlap": [200, 300, 400],
    # }
    experiment_name = "hyperparameter_tuning"
    param_grid = {
        'model_name': ["qwen3:8b"],
        'chunk_size': [2000, 2500, 3000],
        'temperature': [0.1, 0.2, 0.3],
        "embedding_model": ["embeddinggemma"],
        "chunk_overlap": [100, 200, 300, 400],
        "max_judge_retries": [0],
        "max_oml_retries": [0],
    }
    experiment_results = runner.run_experiment_batch(
        max_experiments=-1,
        repeat_experiments=2,
        experiment_name=experiment_name,
        param_grid=param_grid,
        mode="extraction",
    )

    # Check if user wants to visualize existing results
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        output_dir = Path(experiment_name)
        csv_path = output_dir / "experiments.csv"
        visualize_results(csv_path, output_dir)

    # TODO: Fix this part
    # Analyze and visualize results
    # runner.analyze_and_visualize_results(experiment_results)

    print("\n✅ Parameter tuning completed!")


if __name__ == "__main__":
    main()
