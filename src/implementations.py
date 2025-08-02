"""
Concrete implementations following SOLID principles.
"""

from typing import Dict, Any, List, Type, Optional
from pydantic import BaseModel
from pathlib import Path
import pandas as pd

from abstractions import (
    IDocumentRetriever, ICharacteristicsExtractor, IBlockProcessor,
    IPipelineInitializer, IOMLGenerator, IQualityAnalyzer, IStateManager,
    ExtractionConfig, ExtractionResult
)
from utils.enhanced_rag_config import EnhancedRAGPipeline
from models.schemas import (
    Block1Characteristics, Block2Characteristics, Block3Characteristics,
    Block4Characteristics, Block5Characteristics, Block6Characteristics
)


class StateManager(IStateManager):
    """Manages pipeline state following SRP."""
    
    def __init__(self):
        self._state: Dict[str, Any] = {}
    
    def get_state(self, key: str) -> Any:
        return self._state.get(key)
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        self._state.update(updates)
    
    def merge_characteristics(self, new_characteristics: Dict[str, Any]) -> None:
        existing = self._state.get("extracted_characteristics", {})
        existing.update(new_characteristics)
        self._state["extracted_characteristics"] = existing
    
    def get_all_state(self) -> Dict[str, Any]:
        return self._state.copy()


class DocumentRetriever(IDocumentRetriever):
    """Handles document retrieval operations."""
    
    def __init__(self, rag_pipeline: EnhancedRAGPipeline, vectordb):
        self._rag_pipeline = rag_pipeline
        self._vectordb = vectordb
    
    def retrieve_documents(self, query: str, k: int = 5, **kwargs) -> List[Any]:
        use_mmr = kwargs.get('use_mmr', True)
        diversity_factor = kwargs.get('diversity_factor', 0.3)
        
        return self._rag_pipeline.enhanced_retrieval(
            self._vectordb, query, k=k, use_mmr=use_mmr, diversity_factor=diversity_factor
        )


class CharacteristicsExtractor(ICharacteristicsExtractor):
    """Handles characteristics extraction with fallback mechanisms."""
    
    def __init__(self, rag_pipeline: EnhancedRAGPipeline):
        self._rag_pipeline = rag_pipeline
    
    def extract(self, description: str, documents: List[Any], schema: Type[BaseModel]) -> BaseModel:
        try:
            return self._rag_pipeline.generate_with_cot_and_validation(
                description, documents, schema
            )
        except Exception as e:
            print(f"⚠️  Structured output failed, trying manual parsing: {e}")
            return self._rag_pipeline.generate_with_manual_parsing(
                description, documents, schema
            )


class PipelineInitializer(IPipelineInitializer):
    """Initializes the RAG pipeline."""
    
    def __init__(self):
        self._rag_pipeline = None
    
    def initialize(self, pdf_path: str) -> Dict[str, Any]:
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print("🚀 Initializing enhanced RAG pipeline...")
        self._rag_pipeline = EnhancedRAGPipeline()
        
        # Show PDF information
        print("📊 Analyzing PDF...")
        pdf_info = self._rag_pipeline.get_pdf_info(pdf_path)
        if "error" not in pdf_info:
            print(f"   📄 Pages: {pdf_info['total_pages']}")
            print(f"   💾 Size: {pdf_info['file_size_mb']:.2f} MB")
            print(f"   📝 Title: {pdf_info.get('title', 'Unknown')}")
        
        print("📄 Processing PDF with enhanced chunking...")
        vectordb = self._rag_pipeline.enhanced_pdf_processing(pdf_path)
        
        print("✅ Pipeline initialized successfully")
        return {
            "vectordb": vectordb,
            "rag_pipeline": self._rag_pipeline,
            "extraction_metadata": {
                "total_chunks": vectordb._collection.count(),
                "pdf_info": pdf_info
            }
        }


class OMLGenerator(IOMLGenerator):
    """Generates OML from extracted characteristics."""
    
    def __init__(self, rag_pipeline: EnhancedRAGPipeline):
        self._rag_pipeline = rag_pipeline
    
    def generate(self, characteristics: Dict[str, Any], vocab_files: Dict[str, str]) -> str:
        print("🏗️ Generating enhanced OML description...")
        oml_output = self._rag_pipeline.generate_enhanced_oml(characteristics, vocab_files)
        print("✅ OML generation completed")
        return oml_output


class QualityAnalyzer(IQualityAnalyzer):
    """Analyzes extraction quality."""
    
    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        characteristics = results.get("extracted_characteristics", {})
        metadata = results.get("extraction_metadata", {})
        
        # Count extracted vs not found
        total_characteristics = len(characteristics)
        not_found_count = sum(1 for v in characteristics.values() if v == "Not Found")
        extracted_count = total_characteristics - not_found_count
        
        # Calculate average description length for extracted characteristics
        extracted_values = [v for v in characteristics.values() if v != "Not Found"]
        avg_length = sum(len(str(v)) for v in extracted_values) / len(extracted_values) if extracted_values else 0
        
        return {
            "total_characteristics": total_characteristics,
            "extracted_count": extracted_count,
            "not_found_count": not_found_count,
            "extraction_rate": extracted_count / total_characteristics * 100,
            "average_description_length": avg_length,
            "total_docs_retrieved": sum(v for k, v in metadata.items() if k.endswith("_docs_retrieved")),
            "total_chunks": metadata.get("total_chunks", 0)
        }


# Block processors following Open/Closed Principle
class Block1Processor(IBlockProcessor):
    """Processes Block 1: Purpose characteristics."""
    
    def get_config(self) -> ExtractionConfig:
        return ExtractionConfig(
            query="""
            Digital Twin system purpose and objectives: What is the Physical Twin system being studied?
            What specific services does the Digital Twin provide to users and stakeholders?
            What tools, technologies, and enablers support the Digital Twin's functionality and services?
            System architecture and implementation approach for achieving Digital Twin goals.
            """,
            description="""
            system_under_study: Provide a comprehensive description of the Physical Twin system being studied. Include the type of system, its main components, operational domain, and key characteristics that make it suitable for digital twinning.
            
            dt_services: Detail the specific services that the Digital Twin provides to users and the physical system. Include service types (monitoring, optimization, prediction, control, visualization), target users, and service capabilities.
            
            tooling_and_enablers: Describe the specific tools, technologies, frameworks, and enablers used to implement the Digital Twin. Include software platforms, development tools, simulation engines, databases, and any specialized technologies with their roles and functionalities.
            """,
            k=6
        )
    
    def get_schema(self) -> Type[BaseModel]:
        return Block1Characteristics
    
    def process(self, retriever: IDocumentRetriever, extractor: ICharacteristicsExtractor) -> ExtractionResult:
        config = self.get_config()
        print("🔍 Retrieving documents for Block 1 (Purpose) characteristics...")
        
        try:
            retrieved_docs = retriever.retrieve_documents(
                config.query, k=config.k, use_mmr=config.use_mmr
            )
            
            print("🧠 Extracting Block 1 characteristics with enhanced reasoning...")
            output = extractor.extract(config.description, retrieved_docs, self.get_schema())
            
            return ExtractionResult(
                characteristics=output.model_dump(exclude_none=True),
                metadata={"block1_docs_retrieved": len(retrieved_docs)},
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                characteristics={},
                metadata={},
                success=False,
                error_message=str(e)
            )


class Block2Processor(IBlockProcessor):
    """Processes Block 2: Orchestration characteristics."""
    
    def get_config(self) -> ExtractionConfig:
        return ExtractionConfig(
            query="""
            Digital Twin temporal aspects: synchronization frequencies, time scales, and temporal requirements.
            System multiplicities and hierarchies: multiple twins, distributed architectures, coordination mechanisms.
            Digital Twin constellation orchestration: system-wide coordination, component integration, service orchestration.
            External system integration: data exchange with other systems, interoperability, horizontal integration patterns.
            """,
            description="""
            twinning_time_scale: Describe the temporal aspects of the Digital Twin including synchronization frequencies between physical and digital components, time scales for different services, latency requirements, and temporal granularity needs.
            
            multiplicities: Describe the multiplicities and hierarchical structure including multiple digital twin instances, centralized vs decentralized architectures, coordination mechanisms between twins, and scope of responsibilities for each instance.
            
            dt_constellation: Describe the overall orchestration and coordination of the Digital Twin system including architecture patterns, component integration, resource management, and system-wide coordination mechanisms.
            
            horizontal_integration: Describe integration with external systems including data exchange protocols, interoperability standards, dependencies on external systems, and integration patterns with other Digital Twins or enterprise systems.
            """,
            k=7
        )
    
    def get_schema(self) -> Type[BaseModel]:
        return Block2Characteristics
    
    def process(self, retriever: IDocumentRetriever, extractor: ICharacteristicsExtractor) -> ExtractionResult:
        config = self.get_config()
        print("🔍 Retrieving documents for Block 2 (Orchestration) characteristics...")
        
        try:
            retrieved_docs = retriever.retrieve_documents(
                config.query, k=config.k, use_mmr=config.use_mmr
            )
            
            print("🧠 Extracting Block 2 characteristics with enhanced reasoning...")
            output = extractor.extract(config.description, retrieved_docs, self.get_schema())
            
            return ExtractionResult(
                characteristics=output.model_dump(exclude_none=True),
                metadata={"block2_docs_retrieved": len(retrieved_docs)},
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                characteristics={},
                metadata={},
                success=False,
                error_message=str(e)
            )


class Block3Processor(IBlockProcessor):
    """Processes Block 3: Components characteristics."""
    
    def get_config(self) -> ExtractionConfig:
        return ExtractionConfig(
            query="""
            Digital Twin models and data: computational models, data sources, model types, data management, model validation.
            Physical acting components: actuators, control mechanisms, remote control capabilities, actuation systems.
            Physical sensing components: sensors, measurement systems, data collection mechanisms, sensing technologies.
            Model fidelity and validation: accuracy requirements, validation methods, uncertainty quantification, verification processes.
            """,
            description="""
            dt_models_and_data: Describe the Digital Twin's computational models and data components including model types (geometric, behavioral, analytical), data sources, data management approaches, model relationships, and their specific roles in the Digital Twin constellation.
            
            physical_acting_components: Describe the physical actuators and control mechanisms including types of actuators, control capabilities, remote control interfaces, actuation ranges and limitations, safety constraints, and addressing mechanisms.
            
            physical_sensing_components: Describe the sensing infrastructure including sensor types, measurement capabilities, spatial distribution, sampling frequencies, data transmission rates, accuracy and precision specifications, and data collection mechanisms.
            
            fidelity_and_validity_considerations: Describe model fidelity requirements, validation and verification methods, uncertainty sources and quantification, error detection and correction mechanisms, and quality assurance processes for models and data.
            """,
            k=6
        )
    
    def get_schema(self) -> Type[BaseModel]:
        return Block3Characteristics
    
    def process(self, retriever: IDocumentRetriever, extractor: ICharacteristicsExtractor) -> ExtractionResult:
        config = self.get_config()
        print("🔍 Retrieving documents for Block 3 (Components) characteristics...")
        
        try:
            retrieved_docs = retriever.retrieve_documents(
                config.query, k=config.k, use_mmr=config.use_mmr
            )
            
            print("🧠 Extracting Block 3 characteristics with enhanced reasoning...")
            output = extractor.extract(config.description, retrieved_docs, self.get_schema())
            
            return ExtractionResult(
                characteristics=output.model_dump(exclude_none=True),
                metadata={"block3_docs_retrieved": len(retrieved_docs)},
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                characteristics={},
                metadata={},
                success=False,
                error_message=str(e)
            )


class Block4Processor(IBlockProcessor):
    """Processes Block 4: Connectivity characteristics."""
    
    def get_config(self) -> ExtractionConfig:
        return ExtractionConfig(
            query="""
            Physical to digital data flows: sensor data transmission, communication protocols, data preprocessing, event handling.
            Digital to physical control flows: command transmission, control signals, feedback mechanisms, command validation.
            Network infrastructure: communication protocols, network topology, bandwidth requirements, connection reliability.
            Deployment infrastructure: hosting platforms, computational resources, cloud/edge deployment, scalability considerations.
            """,
            description="""
            physical_to_virtual_interaction: Describe data flows from physical to digital including data types transmitted, communication protocols, transmission frequencies, event triggers, data quality assurance, preprocessing steps, and data integration mechanisms.
            
            virtual_to_physical_interaction: Describe control flows from digital to physical including command types, control message formats, validation and authentication mechanisms, feedback systems, conflict resolution, and command execution confirmation.
            
            dt_technical_connection: Describe the technical network infrastructure including communication protocols, network topology and architecture, bandwidth and latency requirements, reliability and redundancy mechanisms, and security protocols for network communication.
            
            dt_hosting_deployment: Describe the deployment infrastructure including hosting platforms (cloud/edge/on-premise), computational and storage requirements, scalability mechanisms, deployment models supported, and availability/reliability assurance.
            """,
            k=7
        )
    
    def get_schema(self) -> Type[BaseModel]:
        return Block4Characteristics
    
    def process(self, retriever: IDocumentRetriever, extractor: ICharacteristicsExtractor) -> ExtractionResult:
        config = self.get_config()
        print("🔍 Retrieving documents for Block 4 (Connectivity) characteristics...")
        
        try:
            retrieved_docs = retriever.retrieve_documents(
                config.query, k=config.k, use_mmr=config.use_mmr
            )
            
            print("🧠 Extracting Block 4 characteristics with enhanced reasoning...")
            output = extractor.extract(config.description, retrieved_docs, self.get_schema())
            
            return ExtractionResult(
                characteristics=output.model_dump(exclude_none=True),
                metadata={"block4_docs_retrieved": len(retrieved_docs)},
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                characteristics={},
                metadata={},
                success=False,
                error_message=str(e)
            )


class Block5Processor(IBlockProcessor):
    """Processes Block 5: Lifecycle characteristics."""
    
    def get_config(self) -> ExtractionConfig:
        return ExtractionConfig(
            query="""
            Digital Twin lifecycle phases: development stages, deployment phases, evolution processes, maintenance activities.
            Engineering processes: development methodologies, quality assurance, testing approaches, version control.
            Decision support: insight generation, analytics capabilities, decision making processes, user interactions.
            Standards and compliance: industry standards, certification requirements, compliance frameworks, interoperability standards.
            """,
            description="""
            life_cycle_stages: Describe the lifecycle phases of the Digital Twin including development stages, representation types at different phases, transition management between phases, and lifecycle governance approaches.
            
            twinning_process_and_dt_evolution: Describe the engineering and development processes including methodologies used, quality assurance practices, testing and validation approaches, version control, and evolution management strategies.
            
            insights_and_decision_making: Describe the decision support capabilities including types of insights generated, analytics and processing capabilities, decision making processes, user interaction methods, and confidence measures for recommendations.
            
            standardization: Describe standards compliance including relevant industry standards, specifications followed, certification requirements, compliance frameworks, and interoperability standards adopted.
            """,
            k=6
        )
    
    def get_schema(self) -> Type[BaseModel]:
        return Block5Characteristics
    
    def process(self, retriever: IDocumentRetriever, extractor: ICharacteristicsExtractor) -> ExtractionResult:
        config = self.get_config()
        print("🔍 Retrieving documents for Block 5 (Lifecycle) characteristics...")
        
        try:
            retrieved_docs = retriever.retrieve_documents(
                config.query, k=config.k, use_mmr=config.use_mmr
            )
            
            print("🧠 Extracting Block 5 characteristics with enhanced reasoning...")
            output = extractor.extract(config.description, retrieved_docs, self.get_schema())
            
            return ExtractionResult(
                characteristics=output.model_dump(exclude_none=True),
                metadata={"block5_docs_retrieved": len(retrieved_docs)},
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                characteristics={},
                metadata={},
                success=False,
                error_message=str(e)
            )


class Block6Processor(IBlockProcessor):
    """Processes Block 6: Governance characteristics."""
    
    def get_config(self) -> ExtractionConfig:
        return ExtractionConfig(
            query="""
            Data governance: data ownership policies, privacy regulations, data protection measures, consent management.
            Security considerations: cybersecurity measures, access control, authentication, threat mitigation, security protocols.
            Safety requirements: safety-critical considerations, fail-safe mechanisms, risk assessment, hazard analysis.
            Compliance frameworks: regulatory compliance, data protection laws, industry regulations, audit requirements.
            """,
            description="""
            data_ownership_and_privacy: Describe data governance including data ownership policies, privacy protection measures, compliance with data protection regulations, consent management, data anonymization, and data sharing agreements.
            
            security_and_safety_considerations: Describe security and safety measures including cybersecurity protocols, access control mechanisms, authentication systems, threat mitigation strategies, fail-safe mechanisms, risk assessment approaches, and safety-critical system considerations.
            """,
            k=5
        )
    
    def get_schema(self) -> Type[BaseModel]:
        return Block6Characteristics
    
    def process(self, retriever: IDocumentRetriever, extractor: ICharacteristicsExtractor) -> ExtractionResult:
        config = self.get_config()
        print("🔍 Retrieving documents for Block 6 (Governance) characteristics...")
        
        try:
            retrieved_docs = retriever.retrieve_documents(
                config.query, k=config.k, use_mmr=config.use_mmr
            )
            
            print("🧠 Extracting Block 6 characteristics with enhanced reasoning...")
            output = extractor.extract(config.description, retrieved_docs, self.get_schema())
            
            return ExtractionResult(
                characteristics=output.model_dump(exclude_none=True),
                metadata={"block6_docs_retrieved": len(retrieved_docs)},
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                characteristics={},
                metadata={},
                success=False,
                error_message=str(e)
            )
