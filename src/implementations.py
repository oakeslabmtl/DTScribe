"""
Concrete implementations following SOLID principles.
"""

from typing import Dict, Any, List, Type
from abc import ABC, abstractmethod
from pydantic import BaseModel
from pathlib import Path
import time

from abstractions import (
    IDocumentRetriever, ICharacteristicsExtractor, IBlockProcessor,
    IPipelineInitializer, IOMLGenerator, IQualityAnalyzer, IStateManager,
    ExtractionConfig, ExtractionResult
)
from utils.enhanced_rag_config import EnhancedRAGPipeline
from models.schemas import (
    DTCharacteristics,
    Block1Characteristics, Block2Characteristics, Block3Characteristics,
    Block4Characteristics, Block5Characteristics, Block6Characteristics
)
from judge_evaluator import JudgeEvaluator
from langchain_core.documents import Document

BASELINE_MAX_CHARS = 24_000 # size limit for document ingestion w/o RAG, (approx. 6000 tokens, assuming 4 chars/token)

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


# class DocumentRetriever(IDocumentRetriever):
#     """Handles document retrieval operations."""
    
#     def __init__(self, rag_pipeline: EnhancedRAGPipeline, vectordb):
#         self._rag_pipeline = rag_pipeline
#         self._vectordb = vectordb
    
#     def retrieve_documents(self, query: str, k: int = 5) -> List[Any]:
        
#         return self._rag_pipeline.enhanced_retrieval(
#             self._vectordb, query, k=k
#         )
    

class DocumentRetriever(IDocumentRetriever):
    """Handles document retrieval operations."""
    
    def __init__(self, rag_pipeline: EnhancedRAGPipeline, vectordb):
        self._rag_pipeline = rag_pipeline
        self._vectordb = vectordb
    
    def retrieve_documents(self, query: str, k: int = 5) -> List[Any]:

        full_doc = getattr(self._rag_pipeline, "full_corpus_doc", None)

        # Baseline
        if self._vectordb is None and full_doc is not None:
            # pipeline expects a list of tuples (Document, score), score being the similarity score (1  here as a dummy value)
            return [(full_doc, 1.0)]
        
        # Original
        return self._rag_pipeline.enhanced_retrieval(
            self._vectordb, query, k=k
        )


class CharacteristicsExtractor(ICharacteristicsExtractor):
    """Handles characteristics extraction with fallback mechanisms."""
    
    def __init__(self, rag_pipeline: EnhancedRAGPipeline):
        self._rag_pipeline = rag_pipeline

    def extract(self, description: str, documents: List[Any], schema: Type[BaseModel], judge_results: List[dict[str, Any]]) -> BaseModel:
        # import traceback
        try:
            return self._rag_pipeline.generate_with_cot_and_validation(
                description, documents, schema, judge_results
            )
        except Exception as e:
            print(f"⚠️  Structured output failed, trying manual parsing: {e}")
            # traceback.print_exc() 
            return self._rag_pipeline.generate_with_manual_parsing(
                description, documents, schema, judge_results
            )


class PipelineInitializer(IPipelineInitializer):
    """Initializes the pipeline."""
    
    def __init__(self):
        self._rag_pipeline = None

    def initialize(self, input_path: str, config: ExtractionConfig, model_name: str, embedding_model: str) -> Dict[str, Any]:
        if not Path(input_path).exists():
            raise FileNotFoundError(f"PDF source not found: {input_path}")
        
        raw_custom = getattr(config, "custom_params", {}) or {}
        cli_custom = raw_custom.get("custom_params", raw_custom)

        baseline_full_doc = bool(cli_custom.get("baseline_full_doc", False))
        baseline_max_chars = cli_custom.get("baseline_max_chars", BASELINE_MAX_CHARS)

        self._rag_pipeline = EnhancedRAGPipeline(model_name=model_name, embedding_model=embedding_model)
        
        #BASELINE:
        if baseline_full_doc:
            print("🚀 Initializing BASELINE full-doc pipeline (no chunking, no vector DB)...")

            print("📄 Processing source (full document)...")
            docs = self._rag_pipeline.load_documents(input_path)

            combined_text = "\n\n".join(doc.page_content for doc in docs)
            original_len = len(combined_text)

            if original_len > baseline_max_chars:
                truncated_text = combined_text[:baseline_max_chars]
                print(
                    f"✂️ Baseline truncation applied: {original_len} → "
                    f"{len(truncated_text)} characters (limit={baseline_max_chars})."
                )
            else:
                truncated_text = combined_text
                print(f"✅ No truncation needed (len={original_len} chars).")

            # create a single docuemnt with the full/truncated content
            baseline_doc = Document(
                page_content=truncated_text,
                metadata={
                    "source": str(input_path),
                    "type": "baseline_full_document",
                    "baseline_truncated": original_len > baseline_max_chars,
                    "baseline_max_chars": baseline_max_chars,
                    "original_char_count": original_len,
                },
            )

            # store in rag_pipeline to be used by DocumentRetriever
            self._rag_pipeline.full_corpus_doc = baseline_doc

            # print(f"\n\n\n{'---'*10}BASELINE DOC:{'---'*10}\n{baseline_doc}{'---'*30}\n\n\n")

            print("✅ Baseline full-doc pipeline initialized successfully")
            return {
                "vectordb": None,
                "rag_pipeline": self._rag_pipeline,
                "extraction_metadata": {
                    "total_chunks": 1,
                    "baseline_original_chars": original_len,
                    "baseline_truncated_chars": len(truncated_text),
                    "baseline_max_chars": baseline_max_chars,
                },
            }

        # RAG
        print("🚀 Initializing RAG pipeline...")
        
        # # Show PDF information
        # print("📊 Analyzing PDF...")
        # pdf_info = self._rag_pipeline.get_pdf_info(input_path)
        # if "error" not in pdf_info:
        #     print(f"   📄 Pages: {pdf_info['total_pages']}")
        #     print(f"   💾 Size: {pdf_info['file_size_mb']:.2f} MB")
        #     print(f"   📝 Title: {pdf_info.get('title', 'Unknown')}")
        
        print("📄 Processing source...")

        docs = self._rag_pipeline.load_documents(input_path)
        vectordb = self._rag_pipeline.chunk_and_store(docs, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        
        print("✅ Pipeline initialized successfully")
        return {
            "vectordb": vectordb,
            "rag_pipeline": self._rag_pipeline,
            "extraction_metadata": {
                "total_chunks": vectordb._collection.count(),
                # "pdf_info": pdf_info
            }
        }


class OMLGenerator(IOMLGenerator):
    """Generates OML from extracted characteristics."""
    
    def __init__(self, rag_pipeline: EnhancedRAGPipeline):
        self._rag_pipeline = rag_pipeline

    def generate(self, characteristics: Dict[str, Any], vocab_files: Dict[str, str], max_retries: int = 3):
        print("🏗️ Generating OML description...")
        oml_output, oml_repetition_count, validation_status = self._rag_pipeline.generate_oml(characteristics, vocab_files, max_retries=max_retries)
        if oml_output and oml_output.strip() != "":
            print("OML generation completed")
        else:
            print("⚠️ OML generation failed or produced empty output")
        return oml_output, oml_repetition_count, validation_status


class QualityAnalyzer(IQualityAnalyzer):
    """Analyzes extraction quality."""

    EXPECTED_FIELDS = list(DTCharacteristics.model_fields.keys())

    def analyze_characteristics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        characteristics = results.get("extracted_characteristics", {})
        metadata = results.get("extraction_metadata", {})
        
        # total_characteristics = len(characteristics)
        total_characteristics = len(self.EXPECTED_FIELDS)
        valid_extractions = 0
        total_length = 0

        # for v in characteristics.values():
        #     if v and v != "Not in Document" and v.strip() != "":
        #         valid_extractions += 1
        #         total_length += len(v)

        for name in self.EXPECTED_FIELDS:
            v = characteristics.get(name)

            # counts as extracted if: is a string, not empty or not "Not in Document"
            if isinstance(v, str) and v.strip() and v.strip() != "Not in Document":
                valid_extractions += 1
                total_length += len(v)

        avg_length = total_length / valid_extractions if valid_extractions > 0 else 0.0

        return {
            "total_characteristics": total_characteristics,
            "extracted_count": valid_extractions,
            "not_found_count": total_characteristics - valid_extractions,
            "extraction_rate": (valid_extractions / total_characteristics * 100) if total_characteristics > 0 else 0.0,
            "average_description_length": avg_length,
            "total_chunks": metadata.get("total_chunks", 0)
        }


# Block processors following Open/Closed Principle
class BaseBlockProcessor(IBlockProcessor, ABC):
    """Template base class for block processors (reduces duplication)."""

    block_index: int = 0  # override in subclasses for consistent metadata keys
    block_label: str = ""

    @abstractmethod
    def get_config(self) -> ExtractionConfig:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def get_schema(self) -> Type[BaseModel]:  # pragma: no cover - abstract
        ...

    def process(
        self,
        retriever: IDocumentRetriever,
        extractor: ICharacteristicsExtractor,
        judge: "JudgeEvaluator" = None,
        max_retries: int = 2
    ) -> ExtractionResult:
        
        config = self.get_config()
        label = self.block_label or f"Block {self.block_index}"

        print(f"🔍 Retrieving documents for {label} characteristics...")

        start_time = time.time()
        meta_prefix = f"block_{self.block_index}"

        try:
            retrieved_docs = retriever.retrieve_documents(config.query, k=config.k)
            retries = 0
            last_output = None
            last_metadata = {}
            judge_results = []
            preserved_info = {}

            # retry/lock state
            initial_low_set = None          # set[str]
            retry_candidates = set()        # set[str]
            locked = set()                  # set[str]
            last_scores = {}                # dict[str, int]

            while True:
                print(f"🧠 Extracting {label} (attempt {retries+1})...")
                output, response_metadata = extractor.extract(
                    config.description, retrieved_docs, self.get_schema(), judge_results
                )
                extracted_dict = output.model_dump()

                # Clean up escaped quotes in strings
                for key, value in extracted_dict.items():
                    if isinstance(value, str):
                        extracted_dict[key] = value.replace('\"', "'")

                # Update output with cleaned values
                output = self.get_schema()(**extracted_dict)

                # if it's a retry preserve locked characteristics from previous output
                # only allow retry_candidates to be replaced by the new extraction
                if retries > 0 and last_output is not None:
                    prev_dict = last_output.model_dump()
                    preserved = []
                    retried = []

                    for cname in extracted_dict.keys():
                        score = int(last_scores.get(cname, 0))
                        item = {"characteristic": cname, "score": score}

                        if cname in locked and cname in prev_dict:
                            extracted_dict[cname] = prev_dict[cname]
                            preserved.append(item)
                        elif cname in retry_candidates:
                            retried.append(item)
                        else:
                            preserved.append(item)

                    preserved_info[f"retry_{retries}"] = {"preserved": preserved, "retried": retried}
                    output = self.get_schema()(**extracted_dict)

                last_output = output
                last_metadata = response_metadata

                # Judge step
                if judge is not None:
                    print(f"🔬 Performing LLM evaluation of {label} (attempt {retries+1})...")
                    judge_results = judge.evaluate(extracted_dict, retrieved_docs, config.description)

                    # --- debug prints ---
                    # print(f"\n\n Extracted dict:")
                    # for k, v in extracted_dict.items():
                    #     print(f" - {k}: {v}\n")

                    # print(f"💡 LLM evaluation results for {label}:")
                    # for res in judge_results:
                    #     print(f"  - Characteristic: {res.get('characteristic')}")
                    #     print(f"    Score: {res.get('score')}")
                    #     print(f"    Reasoning: {res.get('reasoning')}\n")
                    # -------------------

                    score_map = {
                        r.get("characteristic"): int(r.get("score", 0))
                        for r in judge_results
                        if r.get("characteristic")
                    }

                    # Fallback: if judge retutns ALL_BLOCK (parse fail) retry all characteristics
                    if any(r.get("characteristic") == "ALL_BLOCK" for r in judge_results):
                        if initial_low_set is None:
                            initial_low_set = set(extracted_dict.keys())
                            locked = set()
                            retry_candidates = set(extracted_dict.keys())
                    else:
                        # first retry: build initial low set and locked set
                        if initial_low_set is None:
                            all_names = list(extracted_dict.keys())

                            initial_low_set = {c for c in all_names if score_map.get(c, 0) < 4}
                            locked = {c for c in all_names if c not in initial_low_set}  # >=4
                            retry_candidates = set(initial_low_set)
                        # only retry for initial_low_set and lock those that reached >=4
                        else:
                            for c in list(retry_candidates):
                                if score_map.get(c, 0) >= 4:
                                    locked.add(c)
                                    retry_candidates.remove(c)

                    retry_candidates = {c for c in retry_candidates if score_map.get(c, 0) < 4}

                    # save scores for next iteration
                    last_scores = score_map

                    if retry_candidates and retries < max_retries:
                        print(
                            f"⚠️ Low judge score detected in {label}. "
                            f"Retrying extraction for: {sorted(retry_candidates)}"
                        )
                        retries += 1
                        continue

                break  # acceptable quality, no judge, or retries exhausted

            processing_time = time.time() - start_time
            meta = {
                "block_name": meta_prefix,
                f"{meta_prefix}_processing_time": processing_time,
                f"{meta_prefix}_docs_retrieved": [str(doc[0].page_content) for doc in retrieved_docs],
                f"{meta_prefix}_input_tokens": last_metadata.get('prompt_eval_count', 0),
                f"{meta_prefix}_output_tokens": last_metadata.get('eval_count', 0),
                f"{meta_prefix}_retries": retries,
            }
            if judge_results:
                meta[f"{meta_prefix}_judge"] = judge_results
            if preserved_info:
                meta[f"{meta_prefix}_retry_preserve_info"] = preserved_info

            return ExtractionResult(
                characteristics=last_output.model_dump(),
                metadata=meta,
                success=True
            )
    
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Error processing {meta_prefix}: {e}")
            # import traceback
            # traceback.print_exc() 
            return ExtractionResult(
                characteristics={},
                metadata={
                    f"{meta_prefix}_processing_time": processing_time,
                    f"{meta_prefix}_docs_retrieved": None
                },
                success=False,
                error_message=str(e)
            )

class Block1Processor(BaseBlockProcessor):
    block_index = 1
    block_label = "Block 1 (Purpose)"

    def get_config(self) -> ExtractionConfig:
        return ExtractionConfig(
            query="""
System under study: system purpose and objectives, twin description, operational context, performance objectives, stakeholder goals
Digital Twin services: monitoring and simulation, prediction and optimization, decision-support tools, control and actuation, visualization and interaction
Digital Twin enablers: technologies, tools, data acquisition pipeline, modeling and simulation tools, AI/ML components
            """,
            description="""
system_under_study: Describe the Physical Twin system being studied. Include the type of system, its main components, operational domain, and key characteristics that make it suitable for digital twinning.

dt_services: Describe specific services that the Digital Twin provides to users and the physical system. Include service types (monitoring, optimization, prediction, control, visualization), target users, and service capabilities.

tooling_and_enablers: Describe the specific tools, technologies, frameworks, and enablers used to implement the Digital Twin. Include software platforms, development tools, simulation engines, databases, and any specialized technologies with their roles and functionalities.
            """,
            k=6
        )
    
    def get_schema(self) -> Type[BaseModel]:
        return Block1Characteristics

class Block2Processor(BaseBlockProcessor):
    block_index = 2
    block_label = "Block 2 (Orchestration)"

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

class Block3Processor(BaseBlockProcessor):
    block_index = 3
    block_label = "Block 3 (Components)"

    def get_config(self) -> ExtractionConfig:
        return ExtractionConfig(
            query="""
Digital Twin models and data: computational models, data sources, model types, data management, model validation.
Physical acting components: actuators, control mechanisms, remote control capabilities, actuation systems.
Physical sensing components: sensors, measurement systems, data collection mechanisms, sensing technologies.
Model fidelity and validation: accuracy requirements, validation methods, uncertainty quantification, verification processes.
            """,
            description="""
dt_models_and_data: Describe the computational models and data components including model types (geometric, behavioral, analytical), data sources, data management approaches, model relationships, and their specific roles in the Digital Twin constellation.

physical_acting_components: Describe the physical actuators and control mechanisms including types of actuators, control capabilities, remote control interfaces, actuation ranges and limitations, safety constraints, and addressing mechanisms.

physical_sensing_components: Describe the sensing infrastructure including sensor types, measurement capabilities, spatial distribution, sampling frequencies, data transmission rates, accuracy and precision specifications, and data collection mechanisms.

fidelity_and_validity_considerations: Describe model fidelity requirements, validation and verification methods, uncertainty sources and quantification, error detection and correction mechanisms, and quality assurance processes for models and data.
            """,
            k=6
        )

    def get_schema(self) -> Type[BaseModel]:
        return Block3Characteristics


class Block4Processor(BaseBlockProcessor):
    block_index = 4
    block_label = "Block 4 (Connectivity)"

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


class Block5Processor(BaseBlockProcessor):
    block_index = 5
    block_label = "Block 5 (Lifecycle)"

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


class Block6Processor(BaseBlockProcessor):
    block_index = 6
    block_label = "Block 6 (Governance)"

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


class DTCharacteristicsProcessor(BaseBlockProcessor):
    """
    Single-block processor that extracts all 21 Digital Twin characteristics
    at once using the DTCharacteristics schema.
    """

    block_index = 1
    block_label = "All 21 DT characteristics (baseline)"

    def get_config(self) -> ExtractionConfig:
        return ExtractionConfig(
            query="",
            description=basic_description,
            k=0,
        )

    def get_schema(self) -> Type[BaseModel]:

        return DTCharacteristics

detailed_description = """
You must extract a structured description for the following 21 Digital Twin
characteristics for a single Digital Twin system. For each characteristic,
write one coherent description in English that is as specific and technical
as possible and grounded ONLY in the provided documents. If the information
for a characteristic is not available in the documents, return exactly the
string "Not in Document" for that field.

system_under_study: Describe the physical system being twinned (purpose,
  main components, operating environment, key variables, and performance
  objectives).

physical_acting_components: Describe the actuators, machines, or devices
  that apply actions in the physical system, including what they control
  and any relevant technical specifications.

physical_sensing_components: Describe sensors and measurement devices,
  what they measure, where they are located, sampling characteristics,
  and important limitations.

physical_to_virtual_interaction: Explain how data flows from the physical
  system to the Digital Twin (signals, preprocessing, communication
  protocols, data pipelines, and update frequencies).

virtual_to_physical_interaction: Explain how the Digital Twin can influence
  or control the physical system (control loops, setpoints, decision logic,
  human-in-the-loop interactions, etc.).

dt_services: Describe the services and functionalities provided by the
  Digital Twin to users or other systems (monitoring, prediction,
  optimization, diagnostics, visualization, decision support, etc.).

twinning_time_scale: Describe the time scales and synchronization between
  physical and virtual (real-time, near-real-time, offline, batch) and any
  latency or freshness requirements.

multiplicities: Describe whether there are multiple twins, components,
  or hierarchical levels, and how they relate (e.g., fleet twins,
  line-cell-machine hierarchies, multi-level or distributed twins).

life_cycle_stages: Describe in which lifecycle stages of the physical
  system the Digital Twin is used (design, engineering, commissioning,
  operation, maintenance, end-of-life).

dt_models_and_data: Describe the models, simulations, and data structures
  used in the Digital Twin (types of models, inputs/outputs, data sources,
  storage approaches, and schemas).

tooling_and_enablers: Describe software tools, platforms, middleware, and
  other enablers used to build and operate the Digital Twin (simulation
  tools, cloud platforms, data platforms, integration frameworks).

dt_constellation: Describe how this Digital Twin is related to other twins
  or systems in a broader constellation or ecosystem (federations, networks
  of twins, or system-of-systems structures).

twinning_process_and_dt_evolution: Describe how the Digital Twin is created,
  calibrated, updated, and evolved over time, including configuration
  management, model updates, and data-driven refinement procedures.

fidelity_and_validity_considerations: Describe model fidelity and accuracy,
  validation and verification procedures, uncertainty handling, and known
  limitations of the Digital Twin representations.

dt_technical_connection: Describe the technical connectivity between
  physical and virtual (network technologies, fieldbuses, protocols,
  interfaces, APIs, and any security mechanisms applied to the connection).

dt_hosting_deployment: Describe how and where the Digital Twin is deployed
  (on-premise, cloud, edge, hybrid), including runtime infrastructure and
  deployment patterns (microservices, containers, etc.).

insights_and_decision_making: Describe the kinds of insights, KPIs, or
  decisions that the Digital Twin supports, who uses them, and how they
  are integrated into operational or strategic decision processes.

horizontal_integration: Describe how the Digital Twin integrates horizontally
  with other enterprise systems (MES, ERP, PLM, other DTs, external services)
  and the main data or process flows involved.

data_ownership_and_privacy: Describe who owns the data, how it is governed,
  any privacy policies or constraints mentioned, and how data access and
  sharing are managed.

standardization: Describe any standards, or guidelines that the Digital Twin design claims to follow.

security_and_safety_considerations: Describe cybersecurity and safety
  considerations related to the Digital Twin (threats, mitigations, safety
  functions, risk assessment approaches, and safety-critical aspects).
            """

basic_description = """
You must extract a structured description for the following 21 Digital Twin
characteristics for a single Digital Twin system. For each characteristic,
write one coherent description in English that is as specific and technical
as possible and grounded ONLY in the provided documents. If the information
for a characteristic is not available in the documents, return exactly the
string "Not in Document" for that field.

System under study: Describes the SUS, i.e., the PT, of the system of interest.

Physical acting components: Describes the available acting components in the DT constellation, i.e., the mechanisms the DT can use to act on the PT.

Physical sensing components: Describes the available sensing components in the DT constellation, i.e., the mechanisms the PT can use to transfer data to the DT.

Physical-to-virtual interaction: Describes the interactions from the physical world to the virtual world, i.e., the data transmitted from PT to DT, including inputs and events that the DT processes.

Virtual-to-physical interaction: Describes the interactions from the virtual world to the physical world, i.e., the data transmitted from DT to PT, including outputs the DT generates as part of its services.

DT services: Describes the services, such as optimization, task planning, and visualization, which the DT provides to the users and the physical system.

Twinning time-scale: Describes the time-scale use and the time rates for the DT services and DT-to-PT synchronization.

Multiplicities: Describes the multiplicities, i.e., the internal twins that compose the DT system, which can be implemented in a centralized or decentralized way.

Life-cycle stages: Describes the lifecycle phases in which the DT takes place. It also informs which representation phase the DT covers of its physical counterpart, i.e., as designed (ideal), as manufactured, or as operated.

DT models and data: Describes the DT components, including available models and data, and their role in the DT constellation"
Tooling and enablers: Describes the tools or enablers that are used to achieve the goals of the DT, i.e., they enable the DT to provide the DT services.

DT constellation: Describes the orchestration of the DT system, components, and services as a whole.

Twinning process and DT evolution: Describes the engineering process involved in the DT implementation, including the development process, quality assurance, and definition of requirements. It also informs on the milestones of the DT engineering process over time and intended upgrades.

Fidelity and validity considerations: Describes the fidelity and validity considerations behind the models that constitute the DT, including verification and validation mechanisms, uncertainty, and errors.

DT technical connection: Describes the technical network connection details between PT and DT, including the network protocols and architectures.

DT hosting/deployment: Describes the technical hosting aspects of the DT and the associated technology.

Insights and decision making: Defines the insights and decision making, i.e., indirect outputs of the DT, which have no direct effect on the PT, such as update of parameters, plans, and so on.

Horizontal integration: Describes the information exchange with external information systems not limited to other DTs.

Data ownership and privacy: Refers to the ethical and technical aspects regarding data ownership and data privacy. Is the data owned by the PT owner or by the DT service provider?

Standardization: Refers to the standards being followed for the engineering of the DT and its components.

Security and safety considerations: Refers to the ethical and technical aspects regarding data cybersecurity and safety on operation. Can a DT execute operations remotely on a PT where there may be accidents with humans?
"""
