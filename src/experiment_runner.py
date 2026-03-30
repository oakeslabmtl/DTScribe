"""
Runs multiple experiments with different configurations.
"""

import argparse
import itertools
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import time
import concurrent.futures
import traceback

from main import ExtractionPipelineFactory, ExtractionOrchestrator

def run_experiment_task(
    input_path: str,
    output_dir_path: str,
    params: Dict[str, Any],
    experiment_name: str,
    experiment_counter: int,
    combo_idx: int,
    rep: int,
    mode: str,
    exp_id: str,
    characteristics_file_path: str = None
):
    """Worker function to run a single experiment in a separate process."""
    try:
        # Re-create orchestrator within the process
        orchestrator = ExtractionPipelineFactory.create_orchestrator(
            with_experiment_tracking=True, 
            output_dir=output_dir_path
        )
        
        print(f" Experiment {combo_idx} (Rep {rep+1}, Total {experiment_counter}) started with params: {params}")

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
        orchestrator.initialize_pipeline(input_path, config=config)

        # Run extraction
        start_time = time.time()

        extraction_results = {}
        oml_results = {}
        experiment_id = exp_id
        
        if mode in ("extraction", "both"):
            extraction_results = orchestrator.run_extraction(
                input_path=input_path,
                config=config,
                save_results=True,
                experiment_id=None
            )
            experiment_id = orchestrator.last_experiment_id

        if mode in ("oml", "both"):
             # Load characteristics if in OML-only mode and file path is provided
            if mode == "oml" and characteristics_file_path:
                try:
                    with open(characteristics_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "extracted_characteristics" in data:
                             orchestrator._state_manager.update_state({"extracted_characteristics": data["extracted_characteristics"]})
                             # Use the original experiment ID if available to link results
                             if "experiment_id" in data and not experiment_id:
                                 experiment_id = data["experiment_id"]
                             print(f" Loaded characteristics from {Path(characteristics_file_path).name}")
                except Exception as e:
                    print(f" Failed to load characteristics from {characteristics_file_path}: {e}")

            oml_results = orchestrator.run_oml_generation(
                config=config,
                experiment_id=experiment_id,
                save_results=True,
                source_experiment_id=experiment_id
            )
            oml_output = oml_results.get("oml_output")
            if not oml_output:
                print(f"  Experiment {experiment_counter}: No OML generated.")
        
        experiment_time = time.time() - start_time

        # Analyze results
        result_data = {
            'experiment_number': experiment_counter,
            'parameter_combination': combo_idx,
            'repetition': rep + 1,
            'total_time': experiment_time,
            **params,
            'success': True,
            'error': None,
        }

        if extraction_results.get("extracted_characteristics"):
            quality_metrics = orchestrator.analyze_characteristic_extraction(extraction_results)
            result_data.update(quality_metrics)
            print(f" Experiment {experiment_counter} completed. Rate: {quality_metrics['extraction_rate']:.1f}%")
        else:
             print(f" Experiment {experiment_counter} completed (no characteristics).")

        return result_data

    except Exception as e:
        print(f" Experiment {experiment_counter} failed: {str(e)}")
        # traceback.print_exc()
        return {
            'experiment_number': experiment_counter,
            'parameter_combination': combo_idx,
            'repetition': rep + 1,
            **params,
            'success': False,
            'error': str(e),
            'extraction_rate': 0.0,
            'total_time': 0.0,
        }

class ExperimentRunner:
    """Runs multiple experiments with different parameter configurations."""
    
    def __init__(self, input_path: str, orchestrator: ExtractionOrchestrator):
        self.input_path = input_path
        self.orchestrator = orchestrator

    def define_combinations_from_parameter_grid(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Define the parameter grid to search."""
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        return combinations

    def _map_completed_experiments(self, output_dir: Path) -> Dict[tuple[str, int, int], str]:
        """Map completed experiments to their characteristics file path."""
        file_map = {}
        results_dir = output_dir / "characteristics_extraction"
        if not results_dir.exists():
            return file_map
        
        for file_path in results_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Extract identification info
                    params = data.get("config", {}).get("custom_params", {}).get("custom_params", {})
                    batch = params.get("experiment_batch")
                    combo = params.get("parameter_combination")
                    rep = params.get("repetition")
                    
                    if batch and combo is not None and rep is not None:
                        file_map[(batch, int(combo), int(rep))] = str(file_path)
            except Exception as e:
                print(f" Error reading {file_path}: {e}")
        
        return file_map

    def _get_completed_experiments(self, output_dir: Path, subdir: str = "characteristics_extraction") -> set[tuple[str, int, int]]:
        """Identify completed experiments from results directory."""
        completed = set()
        results_dir = output_dir / subdir
        if not results_dir.exists():
            return completed
        
        for file_path in results_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Extract identification info
                    config = data.get("config", {})
                    params = config.get("custom_params", {})
                    
                    # Handle potential nested custom_params
                    if "custom_params" in params and isinstance(params["custom_params"], dict):
                        params = params["custom_params"]

                    batch = params.get("experiment_batch")
                    combo = params.get("parameter_combination")
                    rep = params.get("repetition")
                    
                    if batch and combo is not None and rep is not None:
                        completed.add((batch, int(combo), int(rep)))
            except Exception as e:
                print(f" Error reading {file_path}: {e}")
        
        return completed

    def run_experiment_batch(
            self, 
            max_experiments: int = 10, 
            experiment_name: str = "default", 
            param_grid: Dict[str, Any] = None, 
            repeat_experiments: int = 1, 
            mode: str = "both", 
            exp_id: str = None,
            workers: int = 1,
            resume: bool = False
            ) -> List[Dict[str, Any]]:
        """Run a batch of experiments with different parameters."""

        self.output_dir = Path(experiment_name)
        self.output_dir.mkdir(exist_ok=True)

        completed_experiments = set()
        completed_oml = set()
        
        if resume:
            completed_experiments = self._get_completed_experiments(self.output_dir, "characteristics_extraction")
            if mode in ["oml", "both"]:
                completed_oml = self._get_completed_experiments(self.output_dir, "oml_generation")
            
            print(f" Resuming experiment batch. Found {len(completed_experiments)} completed extractions and {len(completed_oml)} completed OML generations.")

        if param_grid is None:
            print(" No parameter grid provided for experiments.")
            return []
        
        parameter_combinations = self.define_combinations_from_parameter_grid(param_grid)

        # Limit the number of experiments for demonstration
        if max_experiments > 0:
            parameter_combinations = parameter_combinations[:max_experiments]

        print(f" Preparing {len(parameter_combinations)} configurations × {repeat_experiments} repetitions...")
        
        tasks_args = []

        experiment_counter = 0

        existing_files_map = {}
        if mode == "oml":
             existing_files_map = self._map_completed_experiments(self.output_dir)
             print(f" Found {len(existing_files_map)} existing characteristic files for OML generation.")

        for combo_idx, params in enumerate(parameter_combinations, 1):
            for rep in range(repeat_experiments):
                key = (experiment_name, combo_idx, rep + 1)
                
                if resume:
                    if mode == "extraction" and key in completed_experiments:
                        continue
                    elif mode == "oml" and key in completed_oml:
                        continue
                    elif mode == "both" and key in completed_experiments and key in completed_oml:
                        continue
                
                characteristics_file = None
                if mode == "oml":
                    characteristics_file = existing_files_map.get((experiment_name, combo_idx, rep + 1))
                    if not characteristics_file:
                         continue

                experiment_counter += 1
                tasks_args.append((
                    self.input_path,
                    str(self.output_dir.absolute()),
                    params,
                    experiment_name,
                    experiment_counter,
                    combo_idx,
                    rep,
                    mode,
                    exp_id,
                    characteristics_file
                ))
        
        total_experiments = len(tasks_args)
        experiment_results = []
        failed_tasks_args = []
        
        # Helper to process futures
        def process_futures(futures_map):
            results = []
            failures = []
            for future in concurrent.futures.as_completed(futures_map):
                args = futures_map[future]
                try:
                    result = future.result()
                    if result.get('success', False):
                        results.append(result)
                    else:
                        print(f" Experiment {result.get('experiment_number', '?')} failed temporarily (Error: {result.get('error')}). Will retry.")
                        failures.append(args)
                except Exception as e:
                    print(f" Worker exception for Experiment {args[4]}: {e}")
                    failures.append(args)
            return results, failures

        # 1. Initial Execution
        if workers > 1:
            print(f" Running {total_experiments} experiments in parallel with {workers} workers...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_args = {executor.submit(run_experiment_task, *args): args for args in tasks_args}
                initial_results, failed_tasks_args = process_futures(future_to_args)
                experiment_results.extend(initial_results)
        else:
             print(" Running sequentially...")
             for args in tasks_args:
                 result = run_experiment_task(*args)
                 if result.get('success', False):
                     experiment_results.append(result)
                 else:
                     print(f" Experiment {result.get('experiment_number')} failed. Will retry.")
                     failed_tasks_args.append(args)

        # 2. Retry Logic (Sequential)
        if failed_tasks_args:
            print(f"\n Retrying {len(failed_tasks_args)} failed experiments sequentially to mitigate rate limits (429/503)...")
            
            for args in failed_tasks_args:
                exp_num = args[4]
                print(f"    Retrying Experiment {exp_num}...") 
                
                # We can add a small sleep before retry
                time.sleep(2.0)
                
                result = run_experiment_task(*args)
                experiment_results.append(result) # Append result regardless of success/failure this time
                
                if result.get('success', False):
                    print(f"    Retry successful for Experiment {exp_num}")
                else:
                    print(f"    Retry failed for Experiment {exp_num}: {result.get('error')}")

        # Sort results
        experiment_results.sort(key=lambda x: x.get('experiment_number', 0))

        print(f"\n Experiments {experiment_name} completed!")
        print(f" Processed {len(experiment_results)}/{total_experiments} experiments.")
        return experiment_results
    
    def analyze_and_visualize_results(self, experiment_results: List[Dict[str, Any]]):
        """Analyze experiment results and create visualizations."""
        
        # Convert to DataFrame
        df = pd.DataFrame(experiment_results)
        
        # Save detailed results
        results_file = self.output_dir / "experiments.csv"
        df.to_csv(results_file, index=False)
        print(f"\n Detailed results saved to: {results_file}")
        
        # Filter successful experiments
        successful_df = df[df['success'] == True]
        
        if len(successful_df) == 0:
            print(" No successful experiments to analyze")
            return
        
        print(f"\n Analysis of {len(successful_df)} successful experiments:")
        
        # Basic statistics
        print(f"    Extraction Rate: {successful_df['extraction_rate'].mean():.2f}% ± {successful_df['extraction_rate'].std():.2f}%")
        print(f"    Average Time: {successful_df['total_time'].mean():.2f}s ± {successful_df['total_time'].std():.2f}s")
        print(f"    Best Extraction Rate: {successful_df['extraction_rate'].max():.2f}%")
        print(f"    Fastest Time: {successful_df['total_time'].min():.2f}s")
        
        # Find best configuration
        best_extraction = successful_df.loc[successful_df['extraction_rate'].idxmax()]
        fastest_config = successful_df.loc[successful_df['total_time'].idxmin()]
        
        print(f"\n Best Extraction Rate Configuration:")
        for param in ['chunk_size', 'model_name', 'temperature']:
            if param in best_extraction:
                print(f"   • {param}: {best_extraction[param]}")
        print(f"   • Extraction Rate: {best_extraction['extraction_rate']:.2f}%")

        print(f"\n Fastest Configuration:")
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
        print(f" Visualizations saved to: {viz_file}")
        
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
        print(f" Correlation matrix saved to: {corr_file}")
        
        plt.show()

def main():
    import sys

    # Configuration
    parser = argparse.ArgumentParser(description="Experiment Runner for Digital Twin Characteristics Extraction")
    parser.add_argument("--output-dir", nargs="+", default=["experiments"], help="Path(s) to the experiments directory")
    parser.add_argument("--input-path", nargs="+", default=["data/papers/Ramdhan et al. - 2025 - Engineering Automotive Digital Twins on Standardized Architectures A Case Study.pdf"], help="Path(s) to the input PDF or directory")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true", help="Resume from failed/missing experiments")
    parser.add_argument("--mode", default="both", choices=["both", "extraction", "oml"], help="Experiment mode")
    
    args = parser.parse_args()
    
    input_paths = args.input_path
    output_dirs = args.output_dir
    workers = args.workers

    # Handle broadcasting of single output dir to multiple inputs
    if len(output_dirs) == 1 and len(input_paths) > 1:
        output_dirs = output_dirs * len(input_paths)

    if len(input_paths) != len(output_dirs):
        print(f" Configuration Error: Provided {len(input_paths)} inputs but {len(output_dirs)} outputs.")
        return

    param_grid = {
        # 'model_name': ["qwen3-next:80b-cloud"],
        'model_name': ["gpt-oss:120b-cloud", "qwen3-next:80b-cloud", "gpt-oss:20b-cloud", "ministral-3:14b-cloud", "ministral-3:3b-cloud", "ministral-3:8b-cloud"],
        "temperature": [None],
        "top_p": [None],
        "top_k": [None],
        "embedding_model": ["embeddinggemma"],
        'chunk_size': [3000],
        "chunk_overlap": [500],
        'judge_model_name': ["glm-4.7:cloud"],
        "max_judge_retries": [0, 2],
        "max_oml_retries": [4],
        "baseline_full_doc": [False, True],
        "baseline_max_chars": [24000],
    }

    print(f" Parameter grid defined with {len(param_grid)} keys.")

    # Process each configured input/output pair
    for cfg_idx, (in_path_str, out_dir_str) in enumerate(zip(input_paths, output_dirs)):
        input_arg = Path(in_path_str)
        base_output_dir = Path(out_dir_str)
        
        print(f"\n" + "=" * 80)
        print(f" Configuration {cfg_idx+1}/{len(input_paths)}")
        print(f" Input: {input_arg}")
        print(f" Output: {base_output_dir}")

        files_to_process = []
        if input_arg.is_dir():
            files_to_process = list(input_arg.glob("*.pdf"))
            print(f" Found {len(files_to_process)} PDF files in {input_arg}")
        elif input_arg.exists():
            files_to_process = [input_arg]
        else:
            print(f" Input path not found: {input_arg}")
            continue

        for i, file_path in enumerate(files_to_process):
            print("\n" + "-" * 40)
            print(f" Processing file {i+1}/{len(files_to_process)}: {file_path.name}")
            
            # Determine output directory for this specific experiment
            if len(files_to_process) > 1:
                experiment_name = str(base_output_dir / file_path.stem)
            else:
                experiment_name = str(base_output_dir)
                
            print(f" Output directory: {experiment_name}")
        
            # Create experiment runner
            orchestrator = ExtractionPipelineFactory.create_orchestrator(with_experiment_tracking=True, output_dir=experiment_name)
            runner = ExperimentRunner(str(file_path), orchestrator=orchestrator)

            runner.run_experiment_batch(
                max_experiments=-1,
                repeat_experiments=25,
                experiment_name=experiment_name,
                param_grid=param_grid,
                mode=args.mode,
                exp_id=None,
                workers=workers,
                resume=args.resume
            )

            # TODO: Fix this part
            # Analyze and visualize results
            # runner.analyze_and_visualize_results(experiment_results)

    print("\n All experiments completed!")


if __name__ == "__main__":
    main()
