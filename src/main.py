"""
Main orchestrator following SOLID principles.
"""

from typing import Dict, Any, List
from pathlib import Path
import shutil
import pandas as pd
import time
from datetime import datetime

from abstractions import (
    IBlockProcessor, IPipelineInitializer, IOMLGenerator, 
    IQualityAnalyzer, IStateManager, IDocumentRetriever, ICharacteristicsExtractor
)
from implementations import (
    StateManager, DocumentRetriever, CharacteristicsExtractor,
    PipelineInitializer, OMLGenerator, QualityAnalyzer,
    Block1Processor, Block2Processor, Block3Processor,
    Block4Processor, Block5Processor, Block6Processor
)
from experiment_tracking import (
    ExperimentConfig, ResultsSaver, ExperimentTracker,
    CharacteristicsExtractionResult, OMLGenerationResult
)
from results_analyzer import ResultsAnalyzer


class ExtractionOrchestrator:
    """Main orchestrator that coordinates the extraction pipeline."""
    
    def __init__(self, 
                 initializer: IPipelineInitializer,
                 state_manager: IStateManager,
                 oml_generator: IOMLGenerator,
                 quality_analyzer: IQualityAnalyzer,
                 block_processors: List[IBlockProcessor],
                 experiment_tracker: ExperimentTracker = None):
        
        self._initializer = initializer
        self._state_manager = state_manager
        self._oml_generator = oml_generator
        self._quality_analyzer = quality_analyzer
        self._block_processors = block_processors
        self._experiment_tracker = experiment_tracker
        
        self._retriever: IDocumentRetriever = None
        self._extractor: ICharacteristicsExtractor = None
    
    def run_extraction(self, pdf_path: str, experiment_id: str = None, config: ExperimentConfig = None, save_results: bool = True) -> Dict[str, Any]:
        """Run the complete extraction pipeline with optional experiment tracking."""
        print("🚀 Starting Enhanced Digital Twin Characteristics Extraction")
        print("=" * 60)

        overall_start_time = time.time()
        experiment_id = None

        # Start experiment tracking if configured
        if self._experiment_tracker and config:
            experiment_id = self._experiment_tracker.start_experiment(config)
            print(f"📊 Experiment ID: {experiment_id}")

        # Only process PDF and create vector DB if it does not exist

        vector_db_path = Path("vector_db")
        if not vector_db_path.exists() or not any(vector_db_path.iterdir()):
            print("📁 Vector DB not found, processing PDF and creating vector DB...")
            init_result = self._initializer.initialize(pdf_path)
            self._state_manager.update_state(init_result)
        else:
            print("📂 Vector DB found, loading existing vector DB...")
            # Ensure RAG pipeline and vector DB are loaded into state manager
            init_result = self._initializer.load_existing_vector_db(str(vector_db_path))
            self._state_manager.update_state(init_result)

        # Set up retriever and extractor
        rag_pipeline = self._state_manager.get_state("rag_pipeline")
        vectordb = self._state_manager.get_state("vectordb")

        self._retriever = DocumentRetriever(rag_pipeline, vectordb)
        self._extractor = CharacteristicsExtractor(rag_pipeline)

        # Track block processing
        block_metrics = {
            'processing_times': {},
            'success_rates': {},
        }
        errors = []
        warnings = []

        # Process all blocks
        for i, processor in enumerate(self._block_processors, 1):
            print(f"\n--- Processing Block {i} ---")
            result = processor.process(self._retriever, self._extractor)

            # Track block metrics
            block_name = f"block_{i}"
            block_metrics['success_rates'][block_name] = result.success
            if f"{block_name}_processing_time" in result.metadata:
                block_metrics['processing_times'][block_name] = result.metadata[f"{block_name}_processing_time"]

            if result.success:
                self._state_manager.merge_characteristics(result.characteristics)

                # Update metadata
                existing_metadata = self._state_manager.get_state("extraction_metadata") or {}
                existing_metadata.update(result.metadata)
                self._state_manager.update_state({"extraction_metadata": existing_metadata})
            else:
                error_msg = f"Block {i} processing failed: {result.error_message}"
                print(f"❌ {error_msg}")
                errors.append(error_msg)

        # Calculate characteristics extraction metrics
        characteristics_processing_time = time.time() - overall_start_time

        # Save characteristics extraction results if tracking is enabled
        characteristics_result = None
        if self._experiment_tracker and save_results:
            results = self._state_manager.get_all_state()
            quality_metrics = self.analyze_characteristic_extraction(results)

            print("==" * 60)
            print(quality_metrics.get('extraction_rate', 0.0), "% characteristics extracted")
            print("==" * 60)

            characteristics_result = CharacteristicsExtractionResult(
                experiment_id=experiment_id,
                pdf_path=pdf_path,
                extracted_characteristics=results.get("extracted_characteristics", {}),
                extraction_metadata=results.get("extraction_metadata", {}),
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now(),
                config=config,
                total_characteristics=quality_metrics.get('total_characteristics', 0),
                extracted_count=quality_metrics.get('extracted_count', 0),
                not_found_count=quality_metrics.get('not_found_count', 0),
                extraction_rate=quality_metrics.get('extraction_rate', 0.0),
                average_description_length=quality_metrics.get('average_description_length', 0.0),
                total_docs_retrieved=quality_metrics.get('total_docs_retrieved', 0),
                total_chunks=quality_metrics.get('total_chunks', 0),
                processing_time_seconds=characteristics_processing_time,
                block_processing_times=block_metrics.get('processing_times', {}),
                block_success_rates=block_metrics.get('success_rates', {}),
            )

            saved_path = self._experiment_tracker.results_saver.save_characteristics_results(characteristics_result)
            print(f"💾 Characteristics results saved to: {saved_path}")
        
        return self._state_manager.get_all_state()
        
        
    def run_oml_generation(self, experiment_id: str = None, save_results: bool = True) -> Dict[str, Any]:
        """Run OML generation as a separate task with optional experiment tracking."""
        print("\n--- Generating OML ---")
        oml_start_time = time.time()
        oml_errors = []
        oml_warnings = []

        try:
            characteristics = self._state_manager.get_state("extracted_characteristics")
            vocab_files = {
                "DTDFVocab": "data/oml/DTDF/vocab/DTDFVocab.oml",
                "base": "data/oml/DTDF/vocab/base.oml"
            }

            oml_output = self._oml_generator.generate(characteristics, vocab_files)
            self._state_manager.update_state({"oml_output": oml_output})

        except Exception as e:
            oml_error = f"OML generation failed: {str(e)}"
            print(f"❌ {oml_error}")
            oml_errors.append(oml_error)
            oml_output = ""

        oml_processing_time = time.time() - oml_start_time

        # Save OML generation results if tracking is enabled
        if self._experiment_tracker and save_results and oml_output:
            oml_result = OMLGenerationResult(
                experiment_id=f"{experiment_id}_oml" if experiment_id else f"manual_{int(time.time())}_oml",
                characteristics_experiment_id=experiment_id or f"manual_{int(time.time())}",
                generated_oml=oml_output,
                oml_metadata={},
                generation_time_seconds=oml_processing_time,
                errors=oml_errors,
                warnings=oml_warnings,
                timestamp=datetime.now(),
                oml_syntax_valid=True,
                oml_completeness_score=1.0,
                oml_line_count=len(oml_output.splitlines()),
                oml_instance_count=oml_output.count("instance")
            )

            saved_oml_path = self._experiment_tracker.results_saver.save_oml_results(oml_result)
            print(f"💾 OML results saved to: {saved_oml_path}")

        return self._state_manager.get_all_state()

    def analyze_characteristic_extraction(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of extraction results."""
        return self._quality_analyzer.analyze_characteristics(results)


class ExtractionPipelineFactory:
    """Factory for creating extraction pipeline components."""
    
    @staticmethod
    def create_orchestrator(with_experiment_tracking: bool = True) -> ExtractionOrchestrator:
        """Create a fully configured extraction orchestrator."""
        
        # Create core components
        state_manager = StateManager()
        initializer = PipelineInitializer()
        quality_analyzer = QualityAnalyzer()
        
        # Create experiment tracking if requested
        experiment_tracker = None
        if with_experiment_tracking:
            results_saver = ResultsSaver()
            experiment_tracker = ExperimentTracker(results_saver)
        
        # Create block processors
        block_processors = [
            Block1Processor(),
            Block2Processor(),
            Block3Processor(),
            Block4Processor(),
            Block5Processor(),
            Block6Processor()
        ]
        
        # Create a wrapper that handles the OML generator creation
        class DeferredOMLGenerator:
            def __init__(self, state_manager):
                self._state_manager = state_manager
                self._generator = None
            
            def generate(self, characteristics: Dict[str, Any], vocab_files: Dict[str, str]) -> str:
                if self._generator is None:
                    rag_pipeline = self._state_manager.get_state("rag_pipeline")
                    self._generator = OMLGenerator(rag_pipeline)
                return self._generator.generate(characteristics, vocab_files)
        
        oml_generator = DeferredOMLGenerator(state_manager)
        
        return ExtractionOrchestrator(
            initializer=initializer,
            state_manager=state_manager,
            oml_generator=oml_generator,
            quality_analyzer=quality_analyzer,
            block_processors=block_processors,
            experiment_tracker=experiment_tracker
        )
    
    @staticmethod
    def create_config(
        model_name: str = "qwen3:4b",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        # retrieval_k: int = 6,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 20,
        # repeat_penalty: float = 1.1,
        max_pages: int = None,
        **custom_params
    ) -> ExperimentConfig:
        """Create an experiment configuration."""
        return ExperimentConfig(
            model_name=model_name,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # retrieval_k=retrieval_k,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # repeat_penalty=repeat_penalty,
            max_pages=max_pages,
            custom_params=custom_params
        )


def main():
    """Main function demonstrating SOLID principles with experiment tracking."""
    
    # Configuration
    pdf_path = "data/papers/The Incubator Case Study for Digital Twin Engineering.pdf"
    
    # Create experiment configuration
    config = ExtractionPipelineFactory.create_config(
        model_name="qwen3:4b",
        embedding_model="nomic-embed-text",
        chunk_size=1500,
        chunk_overlap=200,
        # retrieval_k=6,
        temperature=0.1,
        custom_params={"experiment_name": "baseline_run"}
    )
    
    # Create orchestrator with experiment tracking
    orchestrator = ExtractionPipelineFactory.create_orchestrator(with_experiment_tracking=True)

    # Define experimental conditions
    experiments = [
        {"chunk_size": 2500, "temperature": 0.2},
    ]

    # Run systematic experiments
    for exp_params in experiments:
        # Create configuration for this experiment
        config = ExtractionPipelineFactory.create_config(**exp_params)

        print(f"\n🔬 Running experiment with parameters: {exp_params}")

        experiment_id = None
        # Start experiment tracking if configured
        if orchestrator._experiment_tracker and config:
            # Only create one experiment per unique configuration
            experiment_id = orchestrator._experiment_tracker.start_experiment(config)
            print(f"📊 Experiment ID: {experiment_id}")

        extraction_results = orchestrator.run_extraction(pdf_path, experiment_id=experiment_id, config=config, save_results=True)
        oml_results = orchestrator.run_oml_generation(experiment_id=experiment_id, save_results=True)
        
        # Detailed results
        print(f"\n📋 Extracted Characteristics:")
        df = pd.DataFrame(list(extraction_results["extracted_characteristics"].items()), 
                        columns=['Characteristic', 'Description'])
        print(df.to_markdown(index=False, tablefmt="grid"))
        
        print(f"\n🏗️ Generated OML:")
        print("-" * 40)
        oml_output = oml_results.get("oml_output", "Not generated")
        print(oml_output)

        # Analyze and display results
        print("\n" + "=" * 60)
        print("📊 EXTRACTION RESULTS")
        print("=" * 60)

        quality_metrics = orchestrator.analyze_characteristic_extraction(extraction_results)
        print(f"📈 Quality Metrics:")
        print(f"   • Extraction Rate: {quality_metrics['extraction_rate']:.1f}%")
        print(f"   • Characteristics Extracted: {quality_metrics['extracted_count']}/{quality_metrics['total_characteristics']}")
        print(f"   • Average Description Length: {quality_metrics['average_description_length']:.0f} characters")
        print(f"   • Total Documents Retrieved: {quality_metrics['total_docs_retrieved']}")
        print(f"   • Total Chunks in Vector DB: {quality_metrics['total_chunks']}")
        
        # Show experiment tracking info
        if orchestrator._experiment_tracker:
            print(f"\n📊 Experiment Tracking:")
            print(f"   • Results saved to: experiments/")
            print(f"   • Characteristics summary: experiments/analysis/characteristics_summary.csv")
            print(f"   • OML summary: experiments/analysis/oml_summary.csv")
            
            # # Show recent experiments summary
            # char_summary = orchestrator._experiment_tracker.results_saver.get_characteristics_summary()
            # if not char_summary.empty:
            #     print(f"   • Total experiments: {len(char_summary)}")
            #     print(f"   • Latest extraction rate: {char_summary.iloc[-1]['extraction_rate']:.1f}%")
    
    print("\n✅ Enhanced extraction completed successfully!")

    # Generate comprehensive analysis
    analyzer = ResultsAnalyzer()
    report = analyzer.generate_comprehensive_report()

    # Create visualizations
    analyzer.create_dashboard_visualizations()

    # Export for publications
    analyzer.export_research_summary()


if __name__ == "__main__":
    main()