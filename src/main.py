"""Main orchestrator & CLI.

Enhancement: allow standalone OML generation from previously saved characteristics
without re-running extraction (faster iteration). Use --mode oml and provide
--source-experiment-id.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import argparse
import sys

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
# from utils.memory import get_memory_usage_mb


class ExtractionOrchestrator:
    """Main orchestrator that coordinates the extraction pipeline."""
    
    def __init__(
        self,
        initializer: IPipelineInitializer,
        state_manager: IStateManager,
        oml_generator: IOMLGenerator,
        quality_analyzer: IQualityAnalyzer,
        block_processors: List[IBlockProcessor],
        experiment_tracker: ExperimentTracker = None,
    ):
        """Initialize orchestrator with its collaborating components."""
        self._initializer = initializer
        self._state_manager = state_manager
        self._oml_generator = oml_generator
        self._quality_analyzer = quality_analyzer
        self._block_processors = block_processors
        self._experiment_tracker = experiment_tracker

        self._retriever: IDocumentRetriever = None
        self._extractor: ICharacteristicsExtractor = None
        self.last_experiment_id: Optional[str] = None
    
    def run_extraction(self, pdf_path: str, experiment_id: Optional[str] = None, config: Optional[ExperimentConfig] = None, save_results: bool = True) -> Dict[str, Any]:
        """Run the complete extraction pipeline with optional experiment tracking."""
        print("🚀 Starting Enhanced Digital Twin Characteristics Extraction")
        print("=" * 60)

        overall_start_time = time.time()
        # Start experiment only if not externally provided
        if self._experiment_tracker and config and not experiment_id:
            experiment_id = self._experiment_tracker.start_experiment(config)
            print(f"📊 Experiment ID (started): {experiment_id}")
        elif experiment_id:
            print(f"📊 Using provided Experiment ID: {experiment_id}")
        self.last_experiment_id = experiment_id

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
            'docs_retrieved': {},
            # 'memory_usages': {},
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

            if  f"{block_name}_docs_retrieved" in result.metadata:
                block_metrics['docs_retrieved'][block_name] = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in result.metadata[f"{block_name}_docs_retrieved"]]
            
            # if f"{block_name}_memory_usage_mb" in result.metadata:
            #     block_metrics['memory_usages'][block_name] = result.metadata[f"{block_name}_memory_usage_mb"]

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
            print(f"{quality_metrics.get('extraction_rate', 0.0):.2f}% characteristics extracted")
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
                # total_docs_retrieved=quality_metrics.get('total_docs_retrieved', 0),
                total_chunks=quality_metrics.get('total_chunks', 0),
                processing_time_seconds=characteristics_processing_time,
                block_docs_retrieved=block_metrics.get('docs_retrieved', {}),
                block_processing_times=block_metrics.get('processing_times', {}),
                # block_memory_usages=block_metrics.get('memory_usages', {}),
                block_success_rates=block_metrics.get('success_rates', {}),
            )

            saved_path = self._experiment_tracker.results_saver.save_characteristics_results(characteristics_result)
            print(f"💾 Characteristics results saved to: {saved_path}")

        # Include experiment_id in state for downstream consumers
        state = self._state_manager.get_all_state()
        state['experiment_id'] = experiment_id
        return state
        
        
    def run_oml_generation(self, experiment_id: str = None, save_results: bool = True,
                            source_experiment_id: str = None) -> Dict[str, Any]:
        """Run OML generation; optionally load characteristics from a saved experiment.

        Parameters:
            experiment_id: (optional) ID for this OML generation run.
            save_results: persist results if tracking enabled.
            source_experiment_id: load characteristics from this prior extraction experiment.
        """
        print("\n--- Generating OML ---")
        oml_start_time = time.time()
        oml_errors: List[str] = []
        oml_warnings: List[str] = []

        characteristics = None
        if source_experiment_id and self._experiment_tracker:
            loaded = self._experiment_tracker.results_saver.load_characteristics_results(source_experiment_id)
            if loaded:
                print(f"📄 Loaded characteristics from experiment {source_experiment_id}")
                characteristics = loaded.extracted_characteristics
                # use source experiment id as lineage if no explicit experiment id passed
                if experiment_id is None:
                    experiment_id = source_experiment_id
            else:
                print(f"⚠️ No saved characteristics found for experiment id {source_experiment_id}; falling back to in-memory state")
        if characteristics is None:
            characteristics = self._state_manager.get_state("extracted_characteristics") or {}
        if not characteristics:
            print("❌ No characteristics available to generate OML. Run extraction first or specify --source-experiment-id.")
            return self._state_manager.get_all_state()

        # Ensure RAG pipeline is available (load existing DB if necessary for standalone mode)
        if self._state_manager.get_state("rag_pipeline") is None:
            vector_db_path = Path("vector_db")
            if vector_db_path.exists() and any(vector_db_path.iterdir()):
                print("📂 RAG pipeline missing. Loading existing vector DB for OML generation...")
                try:
                    init_result = self._initializer.load_existing_vector_db(str(vector_db_path))
                    self._state_manager.update_state(init_result)
                except Exception as e:
                    print(f"⚠️ Failed to load existing vector DB (continuing): {e}")
            else:
                print("⚠️ No vector DB directory found; proceeding without RAG context (may reduce OML quality).")

        try:
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
        # mem_after = get_memory_usage_mb()

        if self._experiment_tracker and save_results and oml_output:
            oml_result = OMLGenerationResult(
                experiment_id=f"{experiment_id}_oml" if experiment_id else f"manual_{int(time.time())}_oml",
                characteristics_experiment_id=source_experiment_id or experiment_id or f"manual_{int(time.time())}",
                generated_oml=oml_output,
                oml_metadata={},
                generation_time_seconds=oml_processing_time,
                # oml_memory_usage_mb=mem_after - mem_before,
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
                    if rag_pipeline is None:
                        # Attempt deferred load failed; provide clearer message
                        raise RuntimeError("RAG pipeline still not available after attempted load.")
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
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 20,
        max_pages: int = None,
        **custom_params
    ) -> ExperimentConfig:
        """Create an experiment configuration."""
        return ExperimentConfig(
            model_name=model_name,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_pages=max_pages,
            custom_params=custom_params
        )


def main():
    parser = argparse.ArgumentParser(description="Extraction / OML generation pipeline")
    parser.add_argument("--mode", choices=["both", "extraction", "oml"], default="both", help="Run extraction, OML generation, or both")
    parser.add_argument("--pdf", default="data/papers/The Incubator Case Study for Digital Twin Engineering.pdf", help="PDF path for extraction")
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--model-name", default="qwen3:4b")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--source-experiment-id", help="Existing experiment id (hash_timestamp or just hash for latest) containing characteristics for standalone OML generation")
    parser.add_argument("--no-save", action="store_true", help="Do not persist results")
    args = parser.parse_args()

    orchestrator = ExtractionPipelineFactory.create_orchestrator(with_experiment_tracking=True)

    extraction_results: Dict[str, Any] = {}
    experiment_id = None

    if args.mode in ("extraction", "both"):
        config = ExtractionPipelineFactory.create_config(
            model_name=args.model_name,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            temperature=args.temperature,
            custom_params={"cli": True}
        )
        extraction_results = orchestrator.run_extraction(args.pdf, experiment_id=None, config=config, save_results=not args.no_save)
        experiment_id = orchestrator.last_experiment_id

    if args.mode in ("oml", "both"):
        oml_results = orchestrator.run_oml_generation(experiment_id=experiment_id, save_results=not args.no_save, source_experiment_id=args.source_experiment_id)
        oml_output = oml_results.get("oml_output")
        if oml_output:
            print("\n🏗️ Generated OML (first 40 lines):")
            print("-" * 40)
            for i, line in enumerate(oml_output.splitlines()[:40], 1):
                print(f"{i:03}: {line}")
        else:
            print("No OML generated.")

    if extraction_results.get("extracted_characteristics"):
        quality_metrics = orchestrator.analyze_characteristic_extraction(extraction_results)
        print("\n📈 Extraction Quality:")
        print(f" - Extraction Rate: {quality_metrics['extraction_rate']:.2f}%")
        print(f" - Extracted: {quality_metrics['extracted_count']}/{quality_metrics['total_characteristics']}")

    print("\n✅ Completed mode:", args.mode)
    print("💡 For standalone OML: python -m src.main --mode oml --source-experiment-id <hash_timestamp | hash>")
    print("   - Provide full ID (e.g. a1b2c3d4e5f6_20250812143055) to target an exact run")
    print("   - Or just the 12-char hash to use the latest run with that configuration")


if __name__ == "__main__":
    main()