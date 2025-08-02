"""
Main orchestrator following SOLID principles.
"""

from typing import Dict, Any, List
from pathlib import Path
import shutil
import pandas as pd

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


class ExtractionOrchestrator:
    """Main orchestrator that coordinates the extraction pipeline."""
    
    def __init__(self, 
                 initializer: IPipelineInitializer,
                 state_manager: IStateManager,
                 oml_generator: IOMLGenerator,
                 quality_analyzer: IQualityAnalyzer,
                 block_processors: List[IBlockProcessor]):
        
        self._initializer = initializer
        self._state_manager = state_manager
        self._oml_generator = oml_generator
        self._quality_analyzer = quality_analyzer
        self._block_processors = block_processors
        
        self._retriever: IDocumentRetriever = None
        self._extractor: ICharacteristicsExtractor = None
    
    def run_extraction(self, pdf_path: str) -> Dict[str, Any]:
        """Run the complete extraction pipeline."""
        
        # Initialize pipeline
        print("🚀 Starting Enhanced Digital Twin Characteristics Extraction")
        print("=" * 60)
        
        init_result = self._initializer.initialize(pdf_path)
        self._state_manager.update_state(init_result)
        
        # Set up retriever and extractor
        rag_pipeline = self._state_manager.get_state("rag_pipeline")
        vectordb = self._state_manager.get_state("vectordb")
        
        self._retriever = DocumentRetriever(rag_pipeline, vectordb)
        self._extractor = CharacteristicsExtractor(rag_pipeline)
        
        # Process all blocks
        for i, processor in enumerate(self._block_processors, 1):
            print(f"\n--- Processing Block {i} ---")
            result = processor.process(self._retriever, self._extractor)
            
            if result.success:
                self._state_manager.merge_characteristics(result.characteristics)
                
                # Update metadata
                existing_metadata = self._state_manager.get_state("extraction_metadata") or {}
                existing_metadata.update(result.metadata)
                self._state_manager.update_state({"extraction_metadata": existing_metadata})
            else:
                print(f"❌ Block {i} processing failed: {result.error_message}")
        
        # Generate OML
        print("\n--- Generating OML ---")
        characteristics = self._state_manager.get_state("extracted_characteristics")
        vocab_files = {
            "DTDFVocab": "data/oml/DTDF/vocab/DTDFVocab.oml",
            "base": "data/oml/DTDF/vocab/base.oml"
        }
        
        oml_output = self._oml_generator.generate(characteristics, vocab_files)
        self._state_manager.update_state({"oml_output": oml_output})
        
        return self._state_manager.get_all_state()
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of extraction results."""
        return self._quality_analyzer.analyze(results)


class ExtractionPipelineFactory:
    """Factory for creating extraction pipeline components."""
    
    @staticmethod
    def create_orchestrator() -> ExtractionOrchestrator:
        """Create a fully configured extraction orchestrator."""
        
        # Create core components
        state_manager = StateManager()
        initializer = PipelineInitializer()
        
        # These will be created after initialization
        oml_generator = None
        quality_analyzer = QualityAnalyzer()
        
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
            block_processors=block_processors
        )


def main():
    """Main function demonstrating SOLID principles."""
    
    # Configuration
    pdf_path = "data/papers/The Incubator Case Study for Digital Twin Engineering.pdf"
    
    # Clean up previous vector database
    vector_db_path = Path("vector_db")
    if vector_db_path.exists():
        shutil.rmtree(vector_db_path)
        print("🧹 Cleaned up previous vector database")
    
    # Create orchestrator using factory
    orchestrator = ExtractionPipelineFactory.create_orchestrator()
    
    # Run extraction
    results = orchestrator.run_extraction(pdf_path)
    
    # Analyze and display results
    print("\n" + "=" * 60)
    print("📊 EXTRACTION RESULTS")
    print("=" * 60)
    
    quality_metrics = orchestrator.analyze_results(results)
    print(f"📈 Quality Metrics:")
    print(f"   • Extraction Rate: {quality_metrics['extraction_rate']:.1f}%")
    print(f"   • Characteristics Extracted: {quality_metrics['extracted_count']}/{quality_metrics['total_characteristics']}")
    print(f"   • Average Description Length: {quality_metrics['average_description_length']:.0f} characters")
    print(f"   • Total Documents Retrieved: {quality_metrics['total_docs_retrieved']}")
    print(f"   • Total Chunks in Vector DB: {quality_metrics['total_chunks']}")
    
    # Detailed results
    print(f"\n📋 Extracted Characteristics:")
    df = pd.DataFrame(list(results["extracted_characteristics"].items()), 
                     columns=['Characteristic', 'Description'])
    print(df.to_markdown(index=False, tablefmt="grid"))
    
    print(f"\n🏗️ Generated OML:")
    print("-" * 40)
    print(results.get("oml_output", "Not generated"))
    
    print("\n✅ Enhanced extraction completed successfully!")


if __name__ == "__main__":
    main()
