"""
Enhanced Digital Twin Characteristics Extraction Pipeline
Uses advanced RAG techniques for improved generation quality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Dict, Any
import pandas as pd
import shutil
from pathlib import Path

from utils.enhanced_rag_config import EnhancedRAGPipeline
from models.schemas import (
    Block1Characteristics, 
    Block2Characteristics, 
    Block3Characteristics, 
    Block4Characteristics, 
    Block5Characteristics, 
    Block6Characteristics,
)

# ---- Config ----
load_dotenv()
os.environ['LANGSMITH_TRACING'] = os.getenv("LANGSMITH_TRACING", "false")
os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY", "")
os.environ['USER_AGENT'] = os.getenv("USER_AGENT", "LLM2DTDF/1.0")


# ---- Graph State setup ----
class EnhancedState(TypedDict):
    pdf_path: str
    vectordb: object
    rag_pipeline: EnhancedRAGPipeline
    extracted_characteristics: Dict[str, Any]
    oml_output: str
    extraction_metadata: Dict[str, Any]


# ---- Enhanced extraction functions ----
def initialize_pipeline(state: EnhancedState) -> EnhancedState:
    """Initialize the enhanced RAG pipeline and process the PDF."""
    if "pdf_path" not in state:
        raise ValueError("PDF path must be provided.")

    print("🚀 Initializing enhanced RAG pipeline...")
    rag_pipeline = EnhancedRAGPipeline()
    
    # Show PDF information
    print("📊 Analyzing PDF...")
    pdf_info = rag_pipeline.get_pdf_info(state["pdf_path"])
    if "error" not in pdf_info:
        print(f"   📄 Pages: {pdf_info['total_pages']}")
        print(f"   💾 Size: {pdf_info['file_size_mb']:.2f} MB")
        print(f"   📝 Title: {pdf_info.get('title', 'Unknown')}")
    
    print("📄 Processing PDF with enhanced chunking...")
    vectordb = rag_pipeline.enhanced_pdf_processing(state["pdf_path"])
    
    print("✅ Pipeline initialized successfully")
    return {
        "vectordb": vectordb,
        "rag_pipeline": rag_pipeline,
        "extraction_metadata": {
            "total_chunks": vectordb._collection.count(),
            "pdf_info": pdf_info
        }
    }


def enhanced_extractor_block1(state: EnhancedState) -> EnhancedState:
    """Enhanced extraction for Block 1: Purpose characteristics."""
    
    vectordb = state["vectordb"]
    rag_pipeline = state["rag_pipeline"]
    
    block1_query = """
    Digital Twin system purpose and objectives: What is the Physical Twin system being studied?
    What specific services does the Digital Twin provide to users and stakeholders?
    What tools, technologies, and enablers support the Digital Twin's functionality and services?
    System architecture and implementation approach for achieving Digital Twin goals.
    """
    
    print("🔍 Retrieving documents for Block 1 (Purpose) characteristics...")
    retrieved_docs = rag_pipeline.enhanced_retrieval(vectordb, block1_query, k=6, use_mmr=True)
    
    block1_description = """
    system_under_study: Provide a comprehensive description of the Physical Twin system being studied. Include the type of system, its main components, operational domain, and key characteristics that make it suitable for digital twinning.
    
    dt_services: Detail the specific services that the Digital Twin provides to users and the physical system. Include service types (monitoring, optimization, prediction, control, visualization), target users, and service capabilities.
    
    tooling_and_enablers: Describe the specific tools, technologies, frameworks, and enablers used to implement the Digital Twin. Include software platforms, development tools, simulation engines, databases, and any specialized technologies with their roles and functionalities.
    """

    print("🧠 Extracting Block 1 characteristics with enhanced reasoning...")
    try:
        output = rag_pipeline.generate_with_cot_and_validation(
            block1_description, retrieved_docs, Block1Characteristics
        )
    except Exception as e:
        print(f"⚠️  Structured output failed, trying manual parsing: {e}")
        output = rag_pipeline.generate_with_manual_parsing(
            block1_description, retrieved_docs, Block1Characteristics
        )

    return {
        "extracted_characteristics": output.model_dump(exclude_none=True),
        "extraction_metadata": {
            **state.get("extraction_metadata", {}),
            "block1_docs_retrieved": len(retrieved_docs)
        }
    }


def enhanced_extractor_block2(state: EnhancedState) -> EnhancedState:
    """Enhanced extraction for Block 2: Orchestration characteristics."""
    
    vectordb = state["vectordb"]
    rag_pipeline = state["rag_pipeline"]
    
    block2_query = """
    Digital Twin temporal aspects: synchronization frequencies, time scales, and temporal requirements.
    System multiplicities and hierarchies: multiple twins, distributed architectures, coordination mechanisms.
    Digital Twin constellation orchestration: system-wide coordination, component integration, service orchestration.
    External system integration: data exchange with other systems, interoperability, horizontal integration patterns.
    """
    
    print("🔍 Retrieving documents for Block 2 (Orchestration) characteristics...")
    retrieved_docs = rag_pipeline.enhanced_retrieval(vectordb, block2_query, k=7, use_mmr=True)
    
    block2_description = """
    twinning_time_scale: Describe the temporal aspects of the Digital Twin including synchronization frequencies between physical and digital components, time scales for different services, latency requirements, and temporal granularity needs.
    
    multiplicities: Describe the multiplicities and hierarchical structure including multiple digital twin instances, centralized vs decentralized architectures, coordination mechanisms between twins, and scope of responsibilities for each instance.
    
    dt_constellation: Describe the overall orchestration and coordination of the Digital Twin system including architecture patterns, component integration, resource management, and system-wide coordination mechanisms.
    
    horizontal_integration: Describe integration with external systems including data exchange protocols, interoperability standards, dependencies on external systems, and integration patterns with other Digital Twins or enterprise systems.
    """

    print("🧠 Extracting Block 2 characteristics with enhanced reasoning...")
    try:
        output = rag_pipeline.generate_with_cot_and_validation(
            block2_description, retrieved_docs, Block2Characteristics
        )
    except Exception as e:
        print(f"⚠️  Structured output failed, trying manual parsing: {e}")
        output = rag_pipeline.generate_with_manual_parsing(
            block2_description, retrieved_docs, Block2Characteristics
        )

    # Merge with existing characteristics
    existing_characteristics = state.get("extracted_characteristics", {})
    existing_characteristics.update(output.model_dump(exclude_none=True))
    
    return {
        "extracted_characteristics": existing_characteristics,
        "extraction_metadata": {
            **state.get("extraction_metadata", {}),
            "block2_docs_retrieved": len(retrieved_docs)
        }
    }


def enhanced_extractor_block3(state: EnhancedState) -> EnhancedState:
    """Enhanced extraction for Block 3: Components characteristics."""
    
    vectordb = state["vectordb"]
    rag_pipeline = state["rag_pipeline"]
    
    block3_query = """
    Digital Twin models and data: computational models, data sources, model types, data management, model validation.
    Physical acting components: actuators, control mechanisms, remote control capabilities, actuation systems.
    Physical sensing components: sensors, measurement systems, data collection mechanisms, sensing technologies.
    Model fidelity and validation: accuracy requirements, validation methods, uncertainty quantification, verification processes.
    """
    
    print("🔍 Retrieving documents for Block 3 (Components) characteristics...")
    retrieved_docs = rag_pipeline.enhanced_retrieval(vectordb, block3_query, k=6, use_mmr=True)
    
    block3_description = """
    dt_models_and_data: Describe the Digital Twin's computational models and data components including model types (geometric, behavioral, analytical), data sources, data management approaches, model relationships, and their specific roles in the Digital Twin constellation.
    
    physical_acting_components: Describe the physical actuators and control mechanisms including types of actuators, control capabilities, remote control interfaces, actuation ranges and limitations, safety constraints, and addressing mechanisms.
    
    physical_sensing_components: Describe the sensing infrastructure including sensor types, measurement capabilities, spatial distribution, sampling frequencies, data transmission rates, accuracy and precision specifications, and data collection mechanisms.
    
    fidelity_and_validity_considerations: Describe model fidelity requirements, validation and verification methods, uncertainty sources and quantification, error detection and correction mechanisms, and quality assurance processes for models and data.
    """

    print("🧠 Extracting Block 3 characteristics with enhanced reasoning...")
    try:
        output = rag_pipeline.generate_with_cot_and_validation(
            block3_description, retrieved_docs, Block3Characteristics
        )
    except Exception as e:
        print(f"⚠️  Structured output failed, trying manual parsing: {e}")
        output = rag_pipeline.generate_with_manual_parsing(
            block3_description, retrieved_docs, Block3Characteristics
        )

    # Merge with existing characteristics
    existing_characteristics = state.get("extracted_characteristics", {})
    existing_characteristics.update(output.model_dump(exclude_none=True))
    
    return {
        "extracted_characteristics": existing_characteristics,
        "extraction_metadata": {
            **state.get("extraction_metadata", {}),
            "block3_docs_retrieved": len(retrieved_docs)
        }
    }


def enhanced_extractor_block4(state: EnhancedState) -> EnhancedState:
    """Enhanced extraction for Block 4: Connectivity characteristics."""
    
    vectordb = state["vectordb"]
    rag_pipeline = state["rag_pipeline"]
    
    block4_query = """
    Physical to digital data flows: sensor data transmission, communication protocols, data preprocessing, event handling.
    Digital to physical control flows: command transmission, control signals, feedback mechanisms, command validation.
    Network infrastructure: communication protocols, network topology, bandwidth requirements, connection reliability.
    Deployment infrastructure: hosting platforms, computational resources, cloud/edge deployment, scalability considerations.
    """
    
    print("🔍 Retrieving documents for Block 4 (Connectivity) characteristics...")
    retrieved_docs = rag_pipeline.enhanced_retrieval(vectordb, block4_query, k=7, use_mmr=True)
    
    block4_description = """
    physical_to_virtual_interaction: Describe data flows from physical to digital including data types transmitted, communication protocols, transmission frequencies, event triggers, data quality assurance, preprocessing steps, and data integration mechanisms.
    
    virtual_to_physical_interaction: Describe control flows from digital to physical including command types, control message formats, validation and authentication mechanisms, feedback systems, conflict resolution, and command execution confirmation.
    
    dt_technical_connection: Describe the technical network infrastructure including communication protocols, network topology and architecture, bandwidth and latency requirements, reliability and redundancy mechanisms, and security protocols for network communication.
    
    dt_hosting_deployment: Describe the deployment infrastructure including hosting platforms (cloud/edge/on-premise), computational and storage requirements, scalability mechanisms, deployment models supported, and availability/reliability assurance.
    """

    print("🧠 Extracting Block 4 characteristics with enhanced reasoning...")
    try:
        output = rag_pipeline.generate_with_cot_and_validation(
            block4_description, retrieved_docs, Block4Characteristics
        )
    except Exception as e:
        print(f"⚠️  Structured output failed, trying manual parsing: {e}")
        output = rag_pipeline.generate_with_manual_parsing(
            block4_description, retrieved_docs, Block4Characteristics
        )

    # Merge with existing characteristics
    existing_characteristics = state.get("extracted_characteristics", {})
    existing_characteristics.update(output.model_dump(exclude_none=True))
    
    return {
        "extracted_characteristics": existing_characteristics,
        "extraction_metadata": {
            **state.get("extraction_metadata", {}),
            "block4_docs_retrieved": len(retrieved_docs)
        }
    }


def enhanced_extractor_block5(state: EnhancedState) -> EnhancedState:
    """Enhanced extraction for Block 5: Lifecycle characteristics."""
    
    vectordb = state["vectordb"]
    rag_pipeline = state["rag_pipeline"]
    
    block5_query = """
    Digital Twin lifecycle phases: development stages, as-designed vs as-operated representations, lifecycle management.
    Engineering and development processes: requirements engineering, quality assurance, development methodologies, evolution planning.
    Decision support and insights: analytics, predictions, recommendations, decision making processes, insight generation.
    Standards and compliance: industry standards, specifications, compliance verification, certification processes, interoperability standards.
    """
    
    print("🔍 Retrieving documents for Block 5 (Lifecycle) characteristics...")
    retrieved_docs = rag_pipeline.enhanced_retrieval(vectordb, block5_query, k=8, use_mmr=True)
    
    block5_description = """
    life_cycle_stages: Describe the lifecycle phases where the Digital Twin is active including development, deployment, operation phases, representation types (as-designed, as-built, as-operated), lifecycle transitions, and historical data preservation across stages.
    
    twinning_process_and_dt_evolution: Describe the engineering processes including development methodologies, requirements capture and validation, quality assurance procedures, evolution and versioning management, milestone tracking, and upgrade criteria and procedures.
    
    insights_and_decision_making: Describe the decision support capabilities including types of insights generated, communication methods to decision makers, prediction and recommendation generation, confidence levels and reliability measures, and analytics processes.
    
    standardization: Describe standards compliance including industry standards followed, specification adherence, compliance verification methods, interoperability standards supported, certification requirements, and standard update management processes.
    """

    print("🧠 Extracting Block 5 characteristics with enhanced reasoning...")
    try:
        output = rag_pipeline.generate_with_cot_and_validation(
            block5_description, retrieved_docs, Block5Characteristics
        )
    except Exception as e:
        print(f"⚠️  Structured output failed, trying manual parsing: {e}")
        output = rag_pipeline.generate_with_manual_parsing(
            block5_description, retrieved_docs, Block5Characteristics
        )

    # Merge with existing characteristics
    existing_characteristics = state.get("extracted_characteristics", {})
    existing_characteristics.update(output.model_dump(exclude_none=True))
    
    return {
        "extracted_characteristics": existing_characteristics,
        "extraction_metadata": {
            **state.get("extraction_metadata", {}),
            "block5_docs_retrieved": len(retrieved_docs)
        }
    }


def enhanced_extractor_block6(state: EnhancedState) -> EnhancedState:
    """Enhanced extraction for Block 6: Governance characteristics."""
    
    vectordb = state["vectordb"]
    rag_pipeline = state["rag_pipeline"]
    
    block6_query = """
    Data ownership and privacy: data ownership policies, privacy regulations, data sharing agreements, consent management.
    Security and safety: cybersecurity measures, safety mechanisms, access control, threat mitigation, fail-safe systems.
    """
    
    print("🔍 Retrieving documents for Block 6 (Governance) characteristics...")
    retrieved_docs = rag_pipeline.enhanced_retrieval(vectordb, block6_query, k=5, use_mmr=True)
    
    block6_description = """
    data_ownership_and_privacy: Describe data ownership policies, privacy regulations and compliance, data sharing agreements and constraints, consent management mechanisms, personal and sensitive data protection measures, and ethical considerations for data usage.
    
    security_and_safety_considerations: Describe security threats and vulnerabilities, access control and authentication systems, safety mechanisms for preventing harmful operations, security incident detection and response, fail-safe mechanisms for critical operations, and human safety assurance during remote operations.
    """

    print("🧠 Extracting Block 6 characteristics with enhanced reasoning...")
    try:
        output = rag_pipeline.generate_with_cot_and_validation(
            block6_description, retrieved_docs, Block6Characteristics
        )
    except Exception as e:
        print(f"⚠️  Structured output failed, trying manual parsing: {e}")
        output = rag_pipeline.generate_with_manual_parsing(
            block6_description, retrieved_docs, Block6Characteristics
        )

    # Merge with existing characteristics
    existing_characteristics = state.get("extracted_characteristics", {})
    existing_characteristics.update(output.model_dump(exclude_none=True))
    
    return {
        "extracted_characteristics": existing_characteristics,
        "extraction_metadata": {
            **state.get("extraction_metadata", {}),
            "block6_docs_retrieved": len(retrieved_docs)
        }
    }


def enhanced_generate_oml(state: EnhancedState) -> EnhancedState:
    """Enhanced OML generation with better context and validation."""
    
    if "extracted_characteristics" not in state:
        raise ValueError("Extracted characteristics must be provided.")
    
    characteristics = state["extracted_characteristics"]
    rag_pipeline = state["rag_pipeline"]
    
    # Define vocabulary files
    vocab_files = {
        "DTDFVocab": "data/oml/DTDF/vocab/DTDFVocab.oml",
        "base": "data/oml/DTDF/vocab/base.oml"
    }
    
    print("🏗️ Generating enhanced OML description...")
    oml_output = rag_pipeline.generate_enhanced_oml(characteristics, vocab_files)
    
    print("✅ OML generation completed")
    return {"oml_output": oml_output}


# ---- Enhanced Graph setup ----
def create_enhanced_workflow() -> StateGraph:
    """Create the enhanced workflow graph."""
    
    graph = StateGraph(EnhancedState)
    
    # Add nodes
    graph.add_node("initialize_pipeline", initialize_pipeline)
    graph.add_node("enhanced_extractor_block1", enhanced_extractor_block1)
    graph.add_node("enhanced_extractor_block2", enhanced_extractor_block2)
    graph.add_node("enhanced_extractor_block3", enhanced_extractor_block3)
    graph.add_node("enhanced_extractor_block4", enhanced_extractor_block4)
    graph.add_node("enhanced_extractor_block5", enhanced_extractor_block5)
    graph.add_node("enhanced_extractor_block6", enhanced_extractor_block6)
    graph.add_node("enhanced_generate_oml", enhanced_generate_oml)
    
    # Set edges
    graph.set_entry_point("initialize_pipeline")
    graph.add_edge("initialize_pipeline", "enhanced_extractor_block1")
    graph.add_edge("enhanced_extractor_block1", "enhanced_extractor_block2")
    graph.add_edge("enhanced_extractor_block2", "enhanced_extractor_block3")
    graph.add_edge("enhanced_extractor_block3", "enhanced_extractor_block4")
    graph.add_edge("enhanced_extractor_block4", "enhanced_extractor_block5")
    graph.add_edge("enhanced_extractor_block5", "enhanced_extractor_block6")
    graph.add_edge("enhanced_extractor_block6", "enhanced_generate_oml")
    graph.set_finish_point("enhanced_generate_oml")
    
    return graph.compile()


def analyze_extraction_quality(result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the quality of the extraction process."""
    characteristics = result.get("extracted_characteristics", {})
    metadata = result.get("extraction_metadata", {})
    
    # Count extracted vs not found
    total_characteristics = len(characteristics)
    not_found_count = sum(1 for v in characteristics.values() if v == "Not Found")
    extracted_count = total_characteristics - not_found_count
    
    # Calculate average description length for extracted characteristics
    extracted_values = [v for v in characteristics.values() if v != "Not Found"]
    avg_length = sum(len(str(v)) for v in extracted_values) / len(extracted_values) if extracted_values else 0
    
    quality_metrics = {
        "total_characteristics": total_characteristics,
        "extracted_count": extracted_count,
        "not_found_count": not_found_count,
        "extraction_rate": extracted_count / total_characteristics * 100,
        "average_description_length": avg_length,
        "total_docs_retrieved": sum(v for k, v in metadata.items() if k.endswith("_docs_retrieved")),
        "total_chunks": metadata.get("total_chunks", 0)
    }
    
    return quality_metrics


if __name__ == "__main__":
    # Configuration
    pdf_path = "data/papers/The Incubator Case Study for Digital Twin Engineering.pdf"
    # pdf_path = "data/case_studies/DT_book-276-289_incubator.pdf"
    
    # Clean up previous vector database
    vector_db_path = Path("vector_db")
    if vector_db_path.exists():
        shutil.rmtree(vector_db_path)
        print("🧹 Cleaned up previous vector database")
    
    # Create and run enhanced workflow
    print("🚀 Starting Enhanced Digital Twin Characteristics Extraction")
    print("=" * 60)
    
    workflow = create_enhanced_workflow()
    result = workflow.invoke({"pdf_path": pdf_path})
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 EXTRACTION RESULTS")
    print("=" * 60)
    
    # Quality analysis
    quality_metrics = analyze_extraction_quality(result)
    print(f"📈 Quality Metrics:")
    print(f"   • Extraction Rate: {quality_metrics['extraction_rate']:.1f}%")
    print(f"   • Characteristics Extracted: {quality_metrics['extracted_count']}/{quality_metrics['total_characteristics']}")
    print(f"   • Average Description Length: {quality_metrics['average_description_length']:.0f} characters")
    print(f"   • Total Documents Retrieved: {quality_metrics['total_docs_retrieved']}")
    print(f"   • Total Chunks in Vector DB: {quality_metrics['total_chunks']}")
    
    # Detailed results
    print(f"\n📋 Extracted Characteristics:")
    df = pd.DataFrame(list(result["extracted_characteristics"].items()), 
                     columns=['Characteristic', 'Description'])
    print(df.to_markdown(index=False, tablefmt="grid"))
    
    print(f"\n🏗️ Generated OML:")
    print("-" * 40)
    print(result.get("oml_output", "Not generated"))
    
    print("\n✅ Enhanced extraction completed successfully!")
