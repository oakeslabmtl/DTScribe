"""
Advanced Digital Twin Characteristics Extraction Pipeline
Integrates advanced prompting techniques for unified, high-quality extraction.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Dict, Any, List
import pandas as pd
import shutil
from pathlib import Path
import json
from datetime import datetime

from utils.enhanced_rag_config import EnhancedRAGPipeline
from utils.advanced_prompting import AdvancedPromptEngineer
from models.enhanced_schemas import DTCharacteristics
from models.schemas import DTCharacteristics as SimpleDTCharacteristics

# ---- Config ----
load_dotenv()
# os.environ['LANGSMITH_TRACING'] = os.getenv("LANGSMITH_TRACING", "false")
# os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY", "")
# os.environ['USER_AGENT'] = os.getenv("USER_AGENT", "LLM2DTDF/1.0")


# ---- Advanced Graph State setup ----
class AdvancedState(TypedDict):
    pdf_path: str
    vectordb: object
    rag_pipeline: EnhancedRAGPipeline
    prompt_engineer: AdvancedPromptEngineer
    extracted_characteristics: Dict[str, Any]
    extraction_metadata: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    oml_output: str


# ---- Advanced extraction functions ----
def initialize_advanced_pipeline(state: AdvancedState) -> AdvancedState:
    """Initialize the advanced pipeline with all components."""
    if "pdf_path" not in state:
        raise ValueError("PDF path must be provided.")

    print("🚀 Initializing advanced RAG pipeline with prompting techniques...")
    
    # Initialize core components
    rag_pipeline = EnhancedRAGPipeline()
    prompt_engineer = AdvancedPromptEngineer()
    
    # Analyze PDF
    print("📊 Analyzing PDF...")
    pdf_info = rag_pipeline.get_pdf_info(state["pdf_path"])
    if "error" not in pdf_info:
        print(f"   📄 Pages: {pdf_info['total_pages']}")
        print(f"   💾 Size: {pdf_info['file_size_mb']:.2f} MB")
        print(f"   📝 Title: {pdf_info.get('title', 'Unknown')}")
    
    print("📄 Processing PDF with enhanced chunking...")
    vectordb = rag_pipeline.enhanced_pdf_processing(state["pdf_path"])
    
    print("✅ Advanced pipeline initialized successfully")
    return {
        "vectordb": vectordb,
        "rag_pipeline": rag_pipeline,
        "prompt_engineer": prompt_engineer,
        "extraction_metadata": {
            "total_chunks": vectordb._collection.count(),
            "pdf_info": pdf_info,
            "start_time": datetime.now().isoformat()
        }
    }


def unified_advanced_extraction(state: AdvancedState) -> AdvancedState:
    """
    Unified extraction using advanced prompting techniques.
    Extracts all characteristics in a single, comprehensive pass.
    """
    vectordb = state["vectordb"]
    rag_pipeline = state["rag_pipeline"]
    prompt_engineer = state["prompt_engineer"]
    
    print("🔍 Performing unified retrieval for all Digital Twin characteristics...")
    
    # Comprehensive query covering all aspects
    unified_query = """
    Digital Twin comprehensive characteristics extraction:
    - Physical Twin system description, components, and operational domain
    - Digital Twin services, capabilities, and user interactions
    - Tools, technologies, frameworks, and implementation platforms
    - Temporal aspects, synchronization, and time scales
    - System multiplicities, hierarchies, and coordination mechanisms
    - System orchestration, architecture patterns, and resource management
    - External system integration and interoperability
    - Models, data types, sources, and management approaches
    - Physical actuators, control mechanisms, and actuation capabilities
    - Sensing infrastructure, sensors, and measurement systems
    - Data flows from physical to digital and vice versa
    - Network infrastructure, protocols, and communication mechanisms
    - Deployment infrastructure, hosting, and scalability
    - Lifecycle phases, development processes, and evolution management
    - Decision support, insights generation, and analytics capabilities
    - Standards compliance, specifications, and certification
    - Data governance, ownership, privacy, and consent management
    - Security measures, safety mechanisms, and threat mitigation
    - Model fidelity, validation, verification, and quality assurance
    """
    
    # Enhanced retrieval with higher k for comprehensive coverage
    retrieved_docs = rag_pipeline.enhanced_retrieval(
        vectordb, unified_query, k=15, use_mmr=True, diversity_factor=0.4
    )
    
    print(f"📚 Retrieved {len(retrieved_docs)} diverse document chunks")
    
    # Create comprehensive description for all characteristics
    all_characteristics_description = """
    Extract detailed information for ALL of the following Digital Twin characteristics:
    
    system_under_study: Comprehensive description of the Physical Twin system including type, components, operational domain, boundaries, functional requirements, and key characteristics that make it suitable for digital twinning.
    
    dt_services: Detailed description of specific services provided by the Digital Twin including service types (monitoring, optimization, prediction, control, visualization), target users, input/output requirements, performance specifications, and service interactions.
    
    tooling_and_enablers: Comprehensive description of tools, technologies, frameworks, and enablers used to implement the Digital Twin including software platforms, development tools, simulation engines, databases, ML frameworks, and their specific roles and functionalities.
    
    twinning_time_scale: Detailed description of temporal aspects including synchronization frequencies between physical and digital components, time scales for different services, latency requirements, temporal granularity needs, and time stamp management.
    
    multiplicities: Comprehensive description of system multiplicities including multiple digital twin instances, hierarchical structures, centralized vs decentralized architectures, coordination mechanisms, scope of responsibilities, and conflict resolution.
    
    dt_constellation: Detailed description of Digital Twin system orchestration including overall architecture patterns, component integration mechanisms, resource allocation and management, interfaces between components, and system-wide coordination.
    
    horizontal_integration: Comprehensive description of integration with external systems including data exchange protocols, interoperability standards, dependencies on external systems, integration patterns, and data consistency maintenance.
    
    dt_models_and_data: Comprehensive description of Digital Twin models and data including model types (geometric, behavioral, analytical), data sources, data management approaches, model relationships, validation strategies, metadata management, and provenance tracking.
    
    physical_acting_components: Detailed description of physical actuators and control mechanisms including actuator types, control capabilities, remote control interfaces, actuation ranges and limitations, safety constraints, addressing mechanisms, and response characteristics.
    
    physical_sensing_components: Comprehensive description of sensing infrastructure including sensor types, measurement capabilities, spatial distribution, sampling frequencies, data transmission rates, accuracy and precision specifications, calibration requirements, and data collection mechanisms.
    
    physical_to_virtual_interaction: Comprehensive description of data flows from physical to digital including data types transmitted, communication protocols, transmission frequencies, event triggers, data quality assurance mechanisms, preprocessing steps, and integration processes.
    
    virtual_to_physical_interaction: Detailed description of control flows from digital to physical including command types, control message formats, validation and authentication mechanisms, feedback systems, conflict resolution strategies, and execution confirmation.
    
    dt_technical_connection: Comprehensive description of network infrastructure including communication protocols, network topology and architecture, bandwidth and latency requirements, reliability and redundancy mechanisms, and security protocols.
    
    dt_hosting_deployment: Detailed description of deployment infrastructure including hosting platforms (cloud/edge/on-premise), computational and storage requirements, scalability mechanisms, deployment models, availability and reliability assurance, and resource optimization.
    
    life_cycle_stages: Comprehensive description of lifecycle phases including development, deployment, and operation stages, representation types (as-designed, as-built, as-operated), lifecycle transition management, and historical data preservation strategies.
    
    twinning_process_and_dt_evolution: Detailed description of engineering processes including development methodologies, requirements capture and validation, quality assurance procedures, evolution and versioning management, milestone tracking, and upgrade criteria.
    
    insights_and_decision_making: Comprehensive description of decision support capabilities including insight types generated, communication methods to decision makers, prediction and recommendation generation, confidence levels and reliability measures, and analytics processes.
    
    standardization: Detailed description of standards compliance including industry standards followed, specification adherence, compliance verification methods, interoperability standards supported, certification requirements, and update management.
    
    data_ownership_and_privacy: Comprehensive description of data governance including ownership policies, privacy regulations and compliance (GDPR, CCPA), data sharing agreements and constraints, consent management mechanisms, and ethical considerations.
    
    security_and_safety_considerations: Detailed description of security and safety measures including threat identification and mitigation, access control and authentication systems, safety mechanisms for preventing harmful operations, incident response procedures, and fail-safe mechanisms.
    
    fidelity_and_validity_considerations: Detailed description of model fidelity requirements, validation and verification methods, uncertainty sources and quantification, error detection and correction mechanisms, quality assurance processes, and confidence measures.
    """
    
    print("🧠 Applying advanced prompting techniques...")
    
    # Try multiple advanced prompting strategies
    extraction_attempts = []
    
    # Strategy 1: Chain of Thought with Domain Expertise
    print("   🔗 Attempting Chain of Thought extraction...")
    try:
        docs_content = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)])
        cot_prompt = prompt_engineer.create_chain_of_thought_prompt(
            all_characteristics_description, docs_content, DTCharacteristics
        )
        
        output_cot = rag_pipeline.llm.invoke(cot_prompt)
        parsed_cot = rag_pipeline._clean_llm_response(output_cot.content)
        result_cot = DTCharacteristics.model_validate_json(parsed_cot)
        extraction_attempts.append(("chain_of_thought", result_cot))
        print("   ✅ Chain of Thought extraction successful")
        
    except Exception as e:
        print(f"   ⚠️  Chain of Thought failed: {e}")
    
    # Strategy 2: Role-based prompt (Systems Engineer perspective)
    print("   👨‍💼 Attempting Systems Engineer role-based extraction...")
    try:
        role_prompt = prompt_engineer.create_role_based_prompt(
            all_characteristics_description, docs_content, DTCharacteristics, "systems_engineer"
        )
        
        output_role = rag_pipeline.llm.invoke(role_prompt)
        parsed_role = rag_pipeline._clean_llm_response(output_role.content)
        result_role = DTCharacteristics.model_validate_json(parsed_role)
        extraction_attempts.append(("systems_engineer", result_role))
        print("   ✅ Systems Engineer extraction successful")
        
    except Exception as e:
        print(f"   ⚠️  Systems Engineer role failed: {e}")
    
    # Strategy 3: Fallback to enhanced RAG with self-consistency
    print("   🎯 Attempting self-consistency extraction...")
    try:
        # Create multiple prompt variants
        base_prompt = prompt_engineer.create_chain_of_thought_prompt(
            all_characteristics_description, docs_content, DTCharacteristics
        )
        prompt_variants = prompt_engineer.create_self_consistency_prompts(base_prompt, 2)
        
        consistency_results = []
        for i, variant_prompt in enumerate(prompt_variants):
            try:
                output_var = rag_pipeline.llm.invoke(variant_prompt)
                parsed_var = rag_pipeline._clean_llm_response(output_var.content)
                result_var = DTCharacteristics.model_validate_json(parsed_var)
                consistency_results.append(result_var)
            except Exception as var_e:
                print(f"   ⚠️  Variant {i+1} failed: {var_e}")
        
        if consistency_results:
            # Use the first successful result for now (could implement voting mechanism)
            extraction_attempts.append(("self_consistency", consistency_results[0]))
            print("   ✅ Self-consistency extraction successful")
            
    except Exception as e:
        print(f"   ⚠️  Self-consistency failed: {e}")
    
    # Strategy 4: Enhanced RAG fallback
    if not extraction_attempts:
        print("   🔄 Falling back to enhanced RAG...")
        try:
            output = rag_pipeline.generate_with_cot_and_validation(
                all_characteristics_description, retrieved_docs, DTCharacteristics
            )
            extraction_attempts.append(("enhanced_rag", output))
            print("   ✅ Enhanced RAG extraction successful")
        except Exception as e:
            print(f"   ⚠️  Enhanced RAG failed: {e}")
    
    # Final fallback
    if not extraction_attempts:
        print("   🆘 Using manual parsing fallback...")
        output = rag_pipeline.generate_with_manual_parsing(
            all_characteristics_description, retrieved_docs, DTCharacteristics
        )
        extraction_attempts.append(("manual_parsing", output))
    
    # Select best extraction result
    if extraction_attempts:
        best_method, best_result = extraction_attempts[0]  # Use first successful
        print(f"🎯 Selected extraction from: {best_method}")
        
        # Convert to dict
        if hasattr(best_result, 'model_dump'):
            extracted_characteristics = best_result.model_dump(exclude_none=True)
        else:
            extracted_characteristics = best_result
            
    else:
        print("❌ All extraction methods failed")
        extracted_characteristics = {}
    
    return {
        "extracted_characteristics": extracted_characteristics,
        "extraction_metadata": {
            **state.get("extraction_metadata", {}),
            "docs_retrieved": len(retrieved_docs),
            "extraction_attempts": len(extraction_attempts),
            "successful_methods": [method for method, _ in extraction_attempts],
            "extraction_method": extraction_attempts[0][0] if extraction_attempts else "failed"
        }
    }

def generate_enhanced_oml(state: AdvancedState) -> AdvancedState:
    """Enhanced OML generation with quality validation."""
    
    if "extracted_characteristics" not in state:
        raise ValueError("Extracted characteristics must be provided.")
    
    characteristics = state["extracted_characteristics"]
    rag_pipeline = state["rag_pipeline"]
    prompt_engineer = state["prompt_engineer"]
    
    print("🏗️ Generating enhanced OML description...")
    
    # Define vocabulary files
    vocab_files = {
        "DTDFVocab": "data/oml/DTDF/vocab/DTDFVocab.oml",
        "base": "data/oml/DTDF/vocab/base.oml", 
        "baseDesc": "data/oml/DTDF/desc/baseDesc.oml"
    }
    
    # Load vocabulary context
    vocab_context = ""
    for name, path in vocab_files.items():
        vocab_path = Path(path)
        if vocab_path.exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_context += f"\n\n{name}.oml:\n{f.read()}"
    
    # Generate OML using advanced prompting
    oml_prompt = prompt_engineer.create_oml_generation_prompt(
        characteristics, vocab_context
    )
    
    try:
        oml_response = rag_pipeline.llm.invoke(oml_prompt)
        oml_output = oml_response.content
        print("✅ OML generation completed successfully")
    except Exception as e:
        print(f"⚠️  OML generation failed: {e}")
        oml_output = "// OML generation failed"
    
    return {"oml_output": oml_output}


# ---- Advanced Graph setup ----
def create_advanced_workflow() -> StateGraph:
    """Create the advanced workflow graph with unified extraction."""
    
    graph = StateGraph(AdvancedState)
    
    # Add nodes
    graph.add_node("initialize_advanced_pipeline", initialize_advanced_pipeline)
    graph.add_node("unified_advanced_extraction", unified_advanced_extraction)
    graph.add_node("generate_enhanced_oml", generate_enhanced_oml)
    
    # Set edges - simplified linear flow
    graph.set_entry_point("initialize_advanced_pipeline")
    graph.add_edge("initialize_advanced_pipeline", "unified_advanced_extraction")
    graph.add_edge("unified_advanced_extraction", "generate_enhanced_oml")
    graph.set_finish_point("generate_enhanced_oml")
    
    return graph.compile()


def analyze_advanced_results(result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the results from the advanced extraction process."""
    characteristics = result.get("extracted_characteristics", {})
    metadata = result.get("extraction_metadata", {})
    quality_metrics = result.get("quality_metrics", {})
    
    # Count extracted vs not found
    total_characteristics = len(characteristics)
    not_found_count = sum(1 for v in characteristics.values() if v == "Not Found")
    extracted_count = total_characteristics - not_found_count
    
    # Calculate technical content indicators
    extracted_values = [v for v in characteristics.values() if v != "Not Found"]
    avg_length = sum(len(str(v)) for v in extracted_values) / len(extracted_values) if extracted_values else 0
    
    # Technical term count
    technical_terms = [
        "sensor", "actuator", "protocol", "API", "cloud", "edge", "IoT", "ML", "AI",
        "synchronization", "fidelity", "validation", "OPC UA", "MQTT", "REST", "TCP"
    ]
    total_tech_terms = sum(
        sum(1 for term in technical_terms if term.lower() in str(v).lower())
        for v in extracted_values
    )
    
    analysis = {
        "extraction_summary": {
            "total_characteristics": total_characteristics,
            "extracted_count": extracted_count,
            "not_found_count": not_found_count,
            "extraction_rate": extracted_count / total_characteristics * 100 if total_characteristics > 0 else 0,
        },
        "content_analysis": {
            "average_description_length": avg_length,
            "total_technical_terms": total_tech_terms,
            "avg_technical_terms_per_field": total_tech_terms / extracted_count if extracted_count > 0 else 0,
        },
        "methodology_analysis": {
            "extraction_method": metadata.get("extraction_method", "unknown"),
            "successful_methods": metadata.get("successful_methods", []),
            "docs_retrieved": metadata.get("docs_retrieved", 0),
            "total_chunks": metadata.get("total_chunks", 0),
        },
        "quality_metrics": quality_metrics,
        "processing_time": {
            "start_time": metadata.get("start_time"),
            "evaluation_time": metadata.get("evaluation_time"),
        }
    }
    
    return analysis


def save_results(result: Dict[str, Any], output_dir: str = "outputs"):
    """Save extraction results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save extracted characteristics
    with open(output_path / f"characteristics_{timestamp}.json", 'w') as f:
        json.dump(result.get("extracted_characteristics", {}), f, indent=2)
    
    # Save quality metrics
    with open(output_path / f"quality_metrics_{timestamp}.json", 'w') as f:
        json.dump(result.get("quality_metrics", {}), f, indent=2)
    
    # Save metadata
    with open(output_path / f"metadata_{timestamp}.json", 'w') as f:
        json.dump(result.get("extraction_metadata", {}), f, indent=2)
    
    # Save OML output
    with open(output_path / f"output_{timestamp}.oml", 'w') as f:
        f.write(result.get("oml_output", ""))
    
    # Save quality report
    quality_report = result.get("extraction_metadata", {}).get("quality_report", "")
    if quality_report:
        with open(output_path / f"quality_report_{timestamp}.md", 'w') as f:
            f.write(quality_report)
    
    print(f"📁 Results saved to {output_path}")


if __name__ == "__main__":
    # Configuration
    pdf_path = "data/papers/The Incubator Case Study for Digital Twin Engineering.pdf"
    # pdf_path = "data/case_studies/DT_book-276-289_incubator.pdf"
    
    # Clean up previous vector database
    vector_db_path = Path("vector_db")
    if vector_db_path.exists():
        shutil.rmtree(vector_db_path)
        print("🧹 Cleaned up previous vector database")
    
    # Create and run advanced workflow
    print("🚀 Starting Advanced Digital Twin Characteristics Extraction")
    print("🔬 Using Unified Extraction with Advanced Prompting Techniques")
    print("=" * 80)
    
    workflow = create_advanced_workflow()
    result = workflow.invoke({"pdf_path": pdf_path})
    
    # Analyze results
    analysis = analyze_advanced_results(result)
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 ADVANCED EXTRACTION RESULTS")
    print("=" * 80)
    
    # Extraction summary
    summary = analysis["extraction_summary"]
    print(f"📈 Extraction Summary:")
    print(f"   • Extraction Rate: {summary['extraction_rate']:.1f}%")
    print(f"   • Characteristics Extracted: {summary['extracted_count']}/{summary['total_characteristics']}")
    
    # Content analysis
    content = analysis["content_analysis"]
    print(f"\n📝 Content Analysis:")
    print(f"   • Average Description Length: {content['average_description_length']:.0f} characters")
    print(f"   • Total Technical Terms Found: {content['total_technical_terms']}")
    print(f"   • Avg Technical Terms per Field: {content['avg_technical_terms_per_field']:.1f}")
    
    # Methodology analysis
    method = analysis["methodology_analysis"]
    print(f"\n🔬 Methodology Analysis:")
    print(f"   • Selected Extraction Method: {method['extraction_method']}")
    print(f"   • Successful Methods: {', '.join(method['successful_methods'])}")
    print(f"   • Documents Retrieved: {method['docs_retrieved']}")
    print(f"   • Total Chunks in Vector DB: {method['total_chunks']}")
    
    # Quality metrics
    if analysis["quality_metrics"]:
        quality = analysis["quality_metrics"]
        print(f"\n🎯 Quality Metrics:")
        print(f"   • Overall Quality Score: {quality.get('overall_quality_score', 0):.1f}/100")
        print(f"   • Technical Depth Score: {quality.get('technical_depth_score', 0):.1f}/100")
        print(f"   • Semantic Coherence: {quality.get('semantic_coherence_score', 0):.1f}/100")
        print(f"   • Completeness Score: {quality.get('completeness_score', 0):.1f}/100")
        print(f"   • Specificity Score: {quality.get('specificity_score', 0):.1f}/100")
    
    # Detailed results table
    print(f"\n📋 Extracted Characteristics:")
    df = pd.DataFrame(list(result["extracted_characteristics"].items()), 
                     columns=['Characteristic', 'Description'])
    print(df.to_markdown(index=False, tablefmt="grid"))
    
    # OML output
    print(f"\n🏗️ Generated OML:")
    print("-" * 50)
    print(result.get("oml_output", "Not generated"))
    
    # Save results
    save_results(result)
    
    print("\n✅ Advanced extraction completed successfully!")
    print("🎉 Check the 'outputs' directory for detailed results and reports!")
