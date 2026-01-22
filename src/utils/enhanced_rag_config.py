"""
Enhanced RAG Configuration with modern LLM techniques for improved Digital Twin characteristics extraction.
"""

import subprocess
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from chromadb.config import Settings
from typing import Type, List, Dict, Any
from pydantic import BaseModel
import pymupdf4llm
import pymupdf
import re
import json
from pathlib import Path
import time

from .oml_writer import IOMLWriter, OMLFileWriter

guiding_syntax = """
```oml                                          
// C2: Acting Components
instance <ActuatorName> : DTDFVocab:ActingComponent [
    base:desc "<specific description from characteristics>"
]

// C3: Physical sensing components
instance <SensorName> : DTDFVocab:SensingComponent [
    base:desc "<specific description from characteristics>"
    DTDFVocab:producedData <DataTransmittedName>
]

// C4: Physical-to-virtual interaction                                           
instance <DataTransmittedName> : DTDFVocab:DataTransmitted [
    DTDFVocab:producedFrom <SensorName1>, <SensorName2>
]

// C10: Models/Data
instance <ModelName> : DTDFVocab:Model [
    base:desc "<specific model description from characteristics>"
    DTDFVocab:inputTo <EnablerName1>, <EnablerName2>
]

instance <DataName> : DTDFVocab:Data [
    base:desc "<specific data description from characteristics>"
    DTDFVocab:inputTo <EnablerName1>, <EnablerName2>
    DTDFVocab:fromData <DataTransmittedName>
]

// C11: Enablers
instance <EnablerName> : DTDFVocab:Enabler [
    base:desc "<specific enabler description from characteristics>"
    DTDFVocab:enables <ServiceName1>, <ServiceName2>
]

// C6: Services
instance <ServiceName> : DTDFVocab:Service [
    base:desc "<specific service description from characteristics>"
    DTDFVocab:provides <InsightName1>, <ActionName2>
]

// C17: Insights/Actions
instance <InsightName> : DTDFVocab:Insight [
    base:desc "<specific insight description from characteristics>"
]

instance <ActionName> : DTDFVocab:Action [
    base:desc "<specific action description from characteristics>"
    DTDFVocab:IsAutomatic <true|false>
]

instance <EnvironmentName> : DTDFVocab:Environment [
    base:contains <ComponentName1>, <ComponentName2>
]

instance <ComponentName> : DTDFVocab:Component [
    base:desc "<specific component description>"
]
```
"""

class EnhancedRAGPipeline:
    """Enhanced RAG Pipeline with improved techniques for better generation quality."""
    
    def __init__(self, model_name, embedding_model):
        self.llm = ChatOllama(
            model=model_name,
            # temperature=0.1,
            # top_p=0.9,
            # top_k=20,
            # repeat_penalty=1.1,
            # num_ctx=8192,
            # num_predict=8192,
        )
        
        self.embeddings = OllamaEmbeddings(model=embedding_model)
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """Get information about the PDF before processing."""
        try:
            doc = pymupdf.open(pdf_path)
            info = {
                "total_pages": len(doc),
                "title": doc.metadata.get("title", "Unknown"),
                "author": doc.metadata.get("author", "Unknown"),
                "creator": doc.metadata.get("creator", "Unknown"),
                "file_size_mb": Path(pdf_path).stat().st_size / (1024 * 1024),
                "is_encrypted": doc.is_encrypted
            }
            doc.close()
            return info
        except Exception as e:
            return {"error": str(e), "total_pages": 0}

    def load_documents(self, input_path: str) -> List[Document]:
        """
        Load generic documents (PDF, DOCX, TXT, etc.).
        Converts PDFs to Markdown with pymupdf4llm.
        Returns a list of LangChain Document objects.
        """
        path = Path(input_path)

        if path.is_dir():
            print(f"📂 Loading documents in directory: {path}")
            loader = DirectoryLoader(
                str(path),
                glob="**/*.*",
                show_progress=True,
                silent_errors=True,
                use_multithreading=True,
                loader_cls=UnstructuredLoader,
                loader_kwargs={"mode": "elements", "strategy": "fast"},
            )

            docs = loader.load()
            # print(f"\n\n DEBUG: docs[0].metadata: {docs[0].metadata} \n\n")
            print(f"✅ Loaded {len(docs)} documents from directory.")
            return docs

        elif path.is_file():
            ext = path.suffix.lower()
            print(f"📄 Loading single file: {path.name} ({ext})")

            # Case 1: PDF -> process as Markdown
            if ext == ".pdf":
                try:
                    # print("🧾 Extracting PDF as Markdown using pymupdf4llm...")
                    md_text = pymupdf4llm.to_markdown(str(path))

                    m = re.search(r"\n#+\s*\**references\**\s*\n", md_text, flags=re.I)
                    if m: md_text = md_text[:m.start()]

                    # creating a single LangChain document with clean metadata
                    doc = Document(
                        metadata={
                            "source": str(path),
                            "type": "technical_document",
                            "format": "pdf",
                            "filename": path.name,
                        },
                        page_content=md_text,
                    )
                    print(f"✅ Loaded 1 Markdown document from PDF ({len(md_text)} chars).")
                    # print(f"\n\n DEBUG: docs[0].page_content: {doc.page_content} \n\n")
                    return [doc]

                except Exception as e:
                    print(f"⚠️ Markdown conversion failed ({e}), falling back to UnstructuredLoader.")
                    loader = UnstructuredLoader(str(path), mode="paged", strategy="fast")
                    docs = loader.load()
                    # print(f"\n\n DEBUG: docs[0].metadata: {docs[0].metadata} \n\n")
                    return docs

            # Case 2: any other file (DOCX, TXT, PPTX, etc.)
            else:
                loader = UnstructuredLoader(str(path), mode="paged", strategy="fast")
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = str(path)
                    d.metadata["format"] = ext.strip(".")
                print(f"✅ Loaded {len(docs)} elements from {ext} file.")
                # print(f"\n\n DEBUG: docs[0].metadata: {docs[0].metadata} \n\n")
                return docs

        else:
            raise FileNotFoundError(f"❌ Path not found: {input_path}")
        
    
    def chunk_and_store(self, docs: List, chunk_size: int, overlap: int) -> Chroma:
        """Split all documents into chunks and store them in Chroma with metadata."""

        print(f"✂️ Chunking {len(docs)} documents (chunk_size={chunk_size}, overlap={overlap})...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
            length_function=len,
        )

        chunks = splitter.split_documents(docs)

        # print(f"\n\n DEBUG: first chunk with complex metadata : {chunks[0]} \n\n")

        chunks = filter_complex_metadata(chunks)

        # print(f"\n\n DEBUG: first chunk after filter complex metadata : {chunks[0]} \n\n")

        # Add metadata
        for i, doc in enumerate(chunks):
            doc.metadata.update({
                "chunk_id": i,
                "word_count": len(doc.page_content.split()),
            })

        print(f"✅ Created {len(chunks)} chunks total.")

        # print(f"\n\n DEBUG: first chunk after filter and add id and count: {chunks[0]} \n\n")

        # Crear vector store
        vectordb = Chroma.from_documents(
            chunks,
            embedding=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=Settings(allow_reset=True),
        )

        print("📦 Vector DB created and ready for retrieval.")
        return vectordb

    def chunk_retrieval(self, vectordb: Chroma, query: str, k: int = 5) -> List:
      # Standard similarity search
        docs = vectordb.similarity_search_with_relevance_scores(query=query, k=k) # output relevance scores, they are already ranked by relevance
        # print(f"\n\nsimilarity_search_with_relevance_scores. \ntype(docs):{type(docs)} \n\nDOCS:{docs}\n\n") # debug print
        
        # 3. Re-rank documents based on relevance and technical content
        # ranked_docs = self._rerank_documents(all_docs, query)
        
        # 4. Remove duplicates while preserving order
        seen_content = set()
        unique_docs = []
        # for doc in ranked_docs:
        for doc in docs:
            content_hash = hash(doc[0].page_content) # first element is the document, second is the score
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs[:k]

    #     return sorted(docs, key=score_document, reverse=True)
    
    def extract_characteristics_with_schema(self, description: str, retrieved_docs: List, 
                                       schema: Type[BaseModel], judge_results: List[dict[str, Any]]) -> tuple[BaseModel, dict]:
        """
        LLM generation with Chain of Thought prompting and validation.
        """
        
        docs_content = "\n\n".join([
            f"## Document {i+1}:\n{str(doc[0].page_content)}" 
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=schema)

        # print(f"\n\njudge_results:\n{judge_results}\n\n")  # debug print
        
        # If judge_results is empty [] (first attempt) or any characteristic is "ALL_BLOCK" (when judge parsing fails), set judge_results to "Not provided"
        if not judge_results or any(res.get('characteristic') == "ALL_BLOCK" for res in judge_results):
            judge_results_str = "Not provided"
        else:
            judge_results_str = "\n".join([
            f"\n- Characteristic: {res.get('characteristic')}\nScore: {res.get('score')}\nReasoning: {res.get('reasoning')}\n\n"
            for res in judge_results
            ])

        # print(f"\n\njudge_results_str:\n{judge_results_str}\n\n")  # debug print

        cot_prompt = PromptTemplate.from_template("""
You are an expert in Digital Twin systems and ontology modeling.
                                                  
You must extract a structured description for the following 21 Digital Twin
characteristics for a single Digital Twin system. For each characteristic,
write one coherent description in English that is as specific and technical
as possible and grounded ONLY in the provided documents. If the information
for a characteristic is not available in the documents, return exactly the
string "Not in Document" for that field.

CONTEXT DOCUMENTS:
{docs_content}

CHARACTERISTICS TO EXTRACT:
{description}
                                                  
JUDGE FEEDBACK:
{judge_results}  
                                                  
INSTRUCTIONS:
1. REASONING PHASE: First, analyze the documents step by step:
   - Identify which documents contain relevant information for each characteristic
   - Note any technical details, technologies, or methodologies mentioned
   - Consider the specific context and domain of the use case

2. EXTRACTION PHASE: For each characteristic:
   - Provide specific, detailed descriptions based ONLY on the provided documents
   - Include concrete technical details (tools, technologies, protocols, methods)
   - Be precise about quantities, frequencies, and specifications when mentioned
   - If no evidence is found, state "Not in Document"
   - Implement the information from the judge feedback (if provided) to preserve high-quality characteristics and enhance characteristics scored less than 4.

3. VALIDATION PHASE: Review your extracted information:
   - Ensure all details come from the provided documents
   - Check that technical terms are used correctly
   - Verify completeness of the description
   - Ensure the extraction of characteristics is consistent with the judge feedback (if provided)

IMPORTANT: You MUST respond with ONLY valid JSON. Do not include any explanations, thinking tags, or additional text outside the JSON structure.

FOLLOW THESE FORMAT INSTRUCTIONS:
{format_instructions}

Remember: Be highly specific and technical. Include exact technologies, methods, and specifications mentioned in the documents. Return ONLY the JSON object with no additional text.
""")
        
        formatted_prompt = cot_prompt.format(
            docs_content=docs_content,
            description=description,
            judge_results=judge_results_str,
            format_instructions=parser.get_format_instructions(),
        )

        # print(f"\n\nDOCS CONTENT in prompt (enhanced_rag_config.py:299):\n{docs_content}\n\n")  # debug print
        # print(f"\n\nFORMATTED PROMPT (enhanced_rag_config.py:299):\n{formatted_prompt}\n\n")  # debug print
        
        # Generate with retry mechanism for better reliability
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Generate with direct LLM call first, then clean and parse
                response = self.llm.invoke(formatted_prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                response_metadata = getattr(response, 'response_metadata', {})

                # print(f"\n\n {'---'*10}🔴RAW RESPONSE🔴{'---'*10}:\n{response_text}\n\n{'---'*10}\n\n")  # debug print

                # Clean the response to remove thinking tags
                cleaned_text = self._clean_llm_response(response_text)

                # Pre-process JSON to handle complex types where strings are expected
                try:
                    data = json.loads(cleaned_text)
                    if isinstance(data, dict):
                        modified = False
                        for key, value in data.items():
                            if isinstance(value, (dict, list)):
                                data[key] = json.dumps(value)
                                modified = True
                        if modified:
                            cleaned_text = json.dumps(data)
                except json.JSONDecodeError:
                    pass

                # print(f"\n\n {'---'*10}🔴CLEANED TEXT🔴{'---'*10}:\n{cleaned_text}\n\n{'---'*10}\n\n") 
                
                # Parse the cleaned response
                output = parser.parse(cleaned_text)

                # print(f"\n\n {'---'*10}🔴PARSED OUTPUT🔴{'---'*10}:\n{output}\n\n{'---'*10}\n\n")  # debug print

                return output, response_metadata
                
            except Exception as e:
                print(f"Warning: Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    # Try with a simpler prompt on retry
                    parser = PydanticOutputParser(pydantic_object=schema)
                    simpler_prompt = f"""
Extract Digital Twin characteristics from the provided documents.

DOCUMENTS:
{docs_content}

TASK: Extract the following characteristics and return as valid JSON only:
{description}

For each characteristic, provide a detailed description based on the documents, or "Not in Document" if no information exists.

RETURN ONLY VALID JSON IN THIS FORMAT:
{parser.get_format_instructions()}

JSON OUTPUT:
"""
                    try:
                        response = self.llm.invoke(simpler_prompt)
                        response_text = response.content if hasattr(response, 'content') else str(response)
                        response_metadata = getattr(response, 'response_metadata', {})
                        cleaned_text = self._clean_llm_response(response_text)

                        # Pre-process JSON to handle complex types where strings are expected
                        try:
                            data = json.loads(cleaned_text)
                            if isinstance(data, dict):
                                modified = False
                                for key, value in data.items():
                                    if isinstance(value, (dict, list)):
                                        data[key] = json.dumps(value)
                                        modified = True
                                if modified:
                                    cleaned_text = json.dumps(data)
                        except json.JSONDecodeError:
                            pass

                        output = parser.parse(cleaned_text)
                        validated_output = self._self_validate_output(output, retrieved_docs)
                        print(f"Success with fallback prompt on attempt {attempt + 1}")
                        return validated_output, response_metadata
                    except Exception as fallback_error:
                        print(f"Fallback also failed: {str(fallback_error)}")
                        continue
                else:
                    # Final fallback: create a basic output with "Not Found" values
                    print("All attempts failed, creating fallback output...")
                    fb = self._create_fallback_output(schema)
                    return fb, {}
    
    def _clean_llm_response(self, response_text: str) -> str:
        """
        Clean LLM response by removing thinking tags and extracting JSON content.
        Also normalizes null/None values to the sentinel string "Not in Document".
        """
        
        # Strip whitespace
        response_text = response_text.strip()

        # Remove thinking tags and their content completely
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)

        # Also handle unclosed thinking tags
        response_text = re.sub(r'<think>.*', '', response_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Strip again after cleaning
        response_text = response_text.strip()
        
        # Remove code block markers
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```oml'):
            response_text = response_text[6:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        # Extract JSON content - look for the first complete JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        
            json_str = re.sub(
                r'(:\s*)null(\s*[,\}])',
                r'\1"Not in Document"\2',
                json_str,
                flags=re.IGNORECASE
            )
            json_str = re.sub(
                r'(:\s*)None(\s*[,\}])',
                r'\1"Not in Document"\2',
                json_str
            )
            return json_str
        
        # If no JSON object found, return the cleaned text as-is
        return response_text


    def _create_fallback_output(self, schema: Type[BaseModel]) -> BaseModel:
        """Create a fallback output when all parsing attempts fail."""
        # Get field names from schema
        field_names = list(schema.model_fields.keys())
        
        # Create a dict with "Not Found" for all fields
        fallback_data = {field: "Not in Document" for field in field_names}
        
        # Create and return the schema instance
        return schema(**fallback_data)
    
    def generate_with_manual_parsing(self, description: str, retrieved_docs: List, 
                                   schema: Type[BaseModel], judge_results: List[dict[str, Any]]) -> tuple[BaseModel, dict]:
        """
        Alternative generation method with manual JSON parsing as fallback.
        """
        docs_content = "\n\n".join([
            f"Document {i+1}:\n{getattr(doc, 'page_content', str(doc))}" 
            for i, doc in enumerate(retrieved_docs)
        ])

        if not judge_results or any(res.get('characteristic') == "ALL_BLOCK" for res in judge_results):
            judge_results_str = "Not provided"
        else:
            judge_results_str = "\n".join([
            f"\n- Characteristic: {res.get('characteristic')}\nScore: {res.get('score')}\nReasoning: {res.get('reasoning')}\n\n"
            for res in judge_results
            ])        
        
        # Simple prompt without structured output parser
        prompt = f"""
You are an expert in Digital Twin systems. Extract characteristics from the provided documents.

DOCUMENTS:
{docs_content}

EXTRACT THESE CHARACTERISTICS:
{description}

USE THESE JUDGE EVALUATIONS TO IMPROVE YOUR EXTRACTION (if provided):
{judge_results_str}

For each characteristic, provide a detailed description based ONLY on the documents, or "Not in Document" if no information exists.

IMPORTANT: Respond with ONLY a valid JSON object. No additional text, explanations, or formatting.

Example format:
{{
    "system_under_study": "description here or Not in Document",
    "dt_services": "description here or Not in Document",
    "tooling_and_enablers": "description here or Not in Document"
}}

JSON:
"""
        
        try:
            # Direct LLM call without structured output
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            response_metadata = getattr(response, 'response_metadata', {})
            
            # Use the same cleaning function for consistency
            cleaned_text = self._clean_llm_response(response_text)
            
            # Parse JSON manually
            parsed_data = json.loads(cleaned_text)
            
            # Create schema instance
            return schema(**parsed_data), response_metadata
            
        except Exception as e:
            print(f"Manual parsing failed: {e}")
            return self._create_fallback_output(schema), {}

    
    def generate_oml(self, characteristics: Dict[str, Any], 
                        vocab_files: Dict[str, str],
                        output_path: Path = Path(r"data\DTDF\src\oml\bentleyjoakes.github.io\LLM_described_DT\llm_dt.oml"),
                        catalog_parent_path: Path = Path(r"data\DTDF\\"),
                        writer: IOMLWriter = None,
                        max_retries: int = 3
                        ) -> tuple[str, int, int]:
        """
        OML generation workflow with retries and validation.
        """
        if writer is None:
            writer = OMLFileWriter()

        start_time = time.perf_counter()
        total_input_tokens = 0
        total_output_tokens = 0

        # --- 1. PREPARATION ---
        # Define description-based vocab mapping
        description_based_vocab_mapping = {
            "virtual_to_physical_interaction": "VirtualToPhysical",
            "twinning_time_scale": "TimeScale",
            "dt_constellation": "Constellation",
            "life_cycle_stages": "EvolutionStage",
            "fidelity_and_validity_considerations": "FidelityConsideration",
            "dt_technical_connection": "TechnicalConnection",
            "dt_hosting_deployment": "Deployment",
            "horizontal_integration": "HoriIntegration",
            "data_ownership_and_privacy": "DataOwnershipPrivacy",
            "standardization": "Standardization",
            "security_and_safety_considerations": "SecuritySafety",
        }
        comma_separated_description_based_vocab_mapping_keys = ", ".join(description_based_vocab_mapping.keys())
        
        # Partition characteristics
        component_based_characteristics_keys = {key for key in characteristics if key not in description_based_vocab_mapping}
        description_based_characteristics = {key: characteristics[key] for key in description_based_vocab_mapping if key in characteristics}
        component_based_characteristics = {key: characteristics[key] for key in component_based_characteristics_keys if key in characteristics}

        # --- 2. INITIAL GENERATION ---
        description_based_oml = self.generate_description_based_oml(description_based_characteristics, description_based_vocab_mapping)
        component_based_oml, response_metadata = self.generate_component_based_oml(component_based_characteristics, vocab_files, comma_separated_description_based_vocab_mapping_keys)
        
        total_input_tokens += response_metadata.get('prompt_eval_count', 0)
        total_output_tokens += response_metadata.get('eval_count', 0)

        oml_repetition_count = 0
        
        # --- 3. VALIDATION & REPAIR LOOP ---
        # We loop (max_retries + 1) times. 
        # e.g., if max_retries=3, we try: Initial, Retry 1, Retry 2, Retry 3.
        total_attempts = max_retries + 1
        
        print(f"🔄 Starting OML Generation & Validation Loop (Max Attempts: {total_attempts})...")

        for attempt in range(total_attempts):
            loop_start = time.perf_counter()
            print(f"📝 Attempt {attempt + 1}/{total_attempts}...")

            # A. Sanity Check: Ensure OML isn't empty
            if component_based_characteristics and (not component_based_oml or not component_based_oml.strip()):
                print(f"❌ Attempt {attempt + 1}: Component-based OML is empty. Regenerating...")
                component_based_oml, response_metadata = self.generate_component_based_oml(component_based_characteristics, vocab_files, comma_separated_description_based_vocab_mapping_keys)
                total_input_tokens += response_metadata.get('prompt_eval_count', 0)
                total_output_tokens += response_metadata.get('eval_count', 0)
                # We don't 'continue' here because we want to try validating this new generation immediately in this loop iteration

            # B. Combine OML
            combined_oml = f"{component_based_oml}\n\n{description_based_oml}"
            combined_oml = self._clean_llm_response(combined_oml)

            # C. Syntax Check (Brackets)
            if combined_oml.count('[') != combined_oml.count(']'):
                print(f"❌ Attempt {attempt + 1}: Syntax error (mismatched brackets). Repairing...")
                # Regenerate and force a retry
                component_based_oml, response_metadata = self.generate_component_based_oml(component_based_characteristics, vocab_files, comma_separated_description_based_vocab_mapping_keys)
                total_input_tokens += response_metadata.get('prompt_eval_count', 0)
                total_output_tokens += response_metadata.get('eval_count', 0)
                continue 

            # D. Write to File
            if not writer.write_oml(combined_oml, output_path):
                print(f"❌ Attempt {attempt + 1}: File write failed.")
                continue

            # E. Validate (OpenCAESAR)
            is_valid, validation_output = self._validate_oml_with_opencaesar(catalog_parent_path)

            if is_valid:
                # --- PHASE: DEPLOYMENT (SUCCESS) ---
                print(f"✅ Attempt {attempt + 1}: Validation Successful! (⏱️ {time.perf_counter() - loop_start:.2f}s)")
                
                # Deploy to Fuseki immediately
                fuseki_ok, fuseki_output = self._load_oml_into_fuseki(catalog_parent_path)
                
                if fuseki_ok:
                    print("🚀 Fuseki start & OML load successful.")
                    if combined_oml and combined_oml.strip() != "":
                        print("OML generation completed")
                    else:
                        print("⚠️ OML generation failed or produced empty output")
                    return combined_oml, oml_repetition_count, 1, total_input_tokens, total_output_tokens
                else:
                    print("⚠️ Validation passed, but Fuseki load failed:")
                    print(fuseki_output)
                    # Even if Fuseki fails, the OML is valid, so we might return success (1) or failure (0)
                    if combined_oml and combined_oml.strip() != "":
                        print("OML generation completed")
                    else:
                        print("⚠️ OML generation failed or produced empty output")
                    return combined_oml, oml_repetition_count, 0, total_input_tokens, total_output_tokens

            # --- PHASE: REPAIR (FAILURE) ---
            print(f"❌ Attempt {attempt + 1}: Validation failed.")
            
            # If this was the last attempt, we fail gracefully
            if attempt == total_attempts - 1:
                print("❌ Retries exhausted. Final validation failed.")
                break # Exit loop to return failure

            # Otherwise, use the Reasoner report to repair
            print("🔧 Repairing OML with reasoner feedback...")
            try:
                component_based_oml, response_metadata = self._fix_oml_with_feedback(
                    component_based_oml, 
                    validation_output, 
                    component_based_characteristics, 
                    vocab_files,
                    writer=writer
                )
                total_input_tokens += response_metadata.get('prompt_eval_count', 0)
                total_output_tokens += response_metadata.get('eval_count', 0)
                oml_repetition_count += 1
            except Exception as e:
                print(f"❌ Error during repair: {e}")
                # Try a raw regeneration if repair crashes
                component_based_oml, response_metadata = self.generate_component_based_oml(component_based_characteristics, vocab_files, comma_separated_description_based_vocab_mapping_keys)
                total_input_tokens += response_metadata.get('prompt_eval_count', 0)
                total_output_tokens += response_metadata.get('eval_count', 0)

        # --- 4. FAILURE EXIT ---
        # If we reach here, we exhausted retries without returning success
        print("⚠️ OML generation failed or produced empty output")
        return None, oml_repetition_count, 0, total_input_tokens, total_output_tokens

    def generate_description_based_oml(self, characteristics: Dict[str, Any], vocab_mapping: Dict[str, str]) -> str:
        """Generate OML of description characteristics programmatically."""
        print("🏗️ Generating description-based OML...")
        # Generate OML for each characteristic
        oml_parts = []
        for key in characteristics:
            value = characteristics.get(key)
            if value and value != "Not in Document":
                oml_parts.append(f"instance {key} : DTDFVocab:{vocab_mapping.get(key, 'Unknown')} [\n    base:desc \"{value}\"\n]")
        joined_parts = "\n\n".join(oml_parts)
        return self._clean_llm_response(joined_parts)
    
    def generate_component_based_oml(self, characteristics: Dict[str, Any], 
                                    vocab_files: Dict[str, str], description_based_vocab_mapping: str) -> tuple[str, dict]:
        """Generate OML of component characteristics with an LLM."""
        print("🏗️ Generating component-based OML...")
        # Load vocabulary files
        vocab_context = ""
        for name, path in vocab_files.items():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    vocab_context += f"\n\n{name}:\n```oml\n{content}\n```"
            except FileNotFoundError:
                print(f"Warning: Could not find '{path}'")
        
        # Enhanced OML generation prompt
        oml_prompt = PromptTemplate.from_template("""
You are an expert in OML (Ontological Modeling Language) and Digital Twin modeling.

TASK: Generate complete, syntactically correct OML code for a Digital Twin based on the extracted characteristics.

EXTRACTED CHARACTERISTICS:
{characteristics}

LIST OF ALREADY GENERATED CHARACTERISTICS (do not generate again):
{description_based_vocab_mapping}

## STRICT NAMING CONVENTIONS:
- Instance names must be CamelCase without spaces (e.g., TemperatureSensor, not Temperature_Sensor)
- Use descriptive technical names based on actual content from characteristics
- All instance names must be unique across the entire OML code
- Avoid generic names like "sensor1", "data1" - use specific names like "PermafrostTemperatureSensor"

## SYNTAX EXAMPLES:
{guiding_syntax}

## PROCESSING RULES:

### 1. Characteristic Analysis:
- Only generate instances for characteristics with meaningful content (ignore "Not in Document")
- For characteristics containing multiple items, create separate instances for each distinct item
- Extract specific technical details from characteristic descriptions for instance names and descriptions

### 2. Mandatory Data Flow Pattern:
Follow this exact sequence for data flow:
```
SensingComponent → producedData → DataTransmitted → fromData → Data → inputTo → Enabler → enables → Service → provides → Insight/Action
```

### 3. Multiple Instance Guidelines:
Create separate instances when characteristics mention:
- Multiple sensors/components (e.g., "temperature sensors and pressure sensors")
- Multiple services (e.g., "monitoring and visualization services")
- Multiple models (e.g., "thermal model and structural model")
- Multiple data types (e.g., "temperature data and historical data")
- Multiple enablers/tools (e.g., "RabbitMQ, InfluxDB, Godot")

### 4. Relationship Establishment:
- Every DTDFVocab:SensingComponent MUST have DTDFVocab:producedData pointing to a DTDFVocab:DataTransmitted
- Every DTDFVocab:DataTransmitted MUST have DTDFVocab:producedFrom pointing back to its source sensor(s)
- Every DTDFVocab:Data MUST have DTDFVocab:fromData pointing to a DTDFVocab:DataTransmitted
- Every DTDFVocab:Model and DTDFVocab:Data MUST have DTDFVocab:inputTo pointing to enabler(s)
- Every DTDFVocab:Enabler MUST have DTDFVocab:enables pointing to service(s)
- Every DTDFVocab:Service MUST have DTDFVocab:provides pointing to insight(s) or action(s)

### 5. Description Quality:
- Use direct quotes or paraphrases from the characteristics
- Be specific about technical details (protocols, frequencies, capabilities)
- Avoid generic descriptions like "provides monitoring"

## VALIDATION CHECKLIST:
Before generating, ensure:
- [ ] All instance names are unique and CamelCase
- [ ] All relationships follow the mandatory data flow pattern
- [ ] Descriptions are specific and extracted from characteristics
- [ ] No "Not in Document" characteristics are included
- [ ] Multiple items in characteristics are split into separate instances
- [ ] All OML syntax is correct (brackets, colons, commas)

Generate ONLY the OML code with no additional explanations:
""")
        
        formatted_prompt = oml_prompt.format(
            characteristics=json.dumps(characteristics, indent=2),
            vocab_context=vocab_context,
            description_based_vocab_mapping=description_based_vocab_mapping,
            guiding_syntax=guiding_syntax
        )
        
        response = self.llm.invoke(formatted_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_metadata = getattr(response, 'response_metadata', {})
        response_text = self._clean_llm_response(response_text)
        response_text = response_text.replace("<", "").replace(">", "")  # Remove any stray angle brackets for components
        return response_text, response_metadata

    def _fix_oml_with_feedback(self, oml_content: str, validation_output: str, 
                              characteristics: Dict[str, Any], vocab_files: Dict[str, str], writer: IOMLWriter = None,) -> str:
        """
        Fix OML content based on OpenCAESAR validation feedback.
        """
        print("🔧 Attempting to fix OML based on validation feedback...")
        
        # Load vocabulary context
        vocab_context = ""
        for name, path in vocab_files.items():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    vocab_context += f"\n\n{name}:\n```oml\n{content}\n```"
            except FileNotFoundError:
                print(f"Warning: Could not find '{path}'")

        # Combine OML content and validation output
        oml_content = writer._combine_oml_with_validation_errors(oml_content, validation_output)
        print("🔍 Validation errors integrated into OML content for context")
        print(oml_content)

        # Create fix prompt with validation feedback
        fix_prompt = PromptTemplate.from_template("""
You are an expert in OML (Ontological Modeling Language) debugging and fixing syntax errors.

TASK: Fix the OML code based on the validation errors provided, given the set of characteristics this ontology is based on.

OML CODE WITH CONTEXTUAL VALIDATION ERRORS TO FIX MARKED AS 'TODO':
```oml
{oml_content}
```

EXTRACTED CHARACTERISTICS FOR CONTEXT:
{characteristics}

## ERROR ANALYSIS FRAMEWORK:

### 1. Error Pattern Recognition:
Analyze each error by:
- **Error type**: Identify the specific issue (reference, syntax, semantic)
- **Root cause**: Determine why the error occurred
- **Impact scope**: Assess which parts of code are affected

### 2. Common Error Types and Fixes:

#### Reference Resolution Errors:
- `Couldn't resolve reference to SemanticProperty 'DTDFVocab:X'`
  - **Cause**: Invalid or non-existent vocabulary property
  - **Fix**: Replace with correct DTDFVocab property or remove if invalid

#### Syntax Errors:
- Missing brackets, commas, or colons
- Malformed instance declarations
- Incorrect namespace usage

#### Semantic Errors:
- Circular dependencies
- Type mismatches
- Invalid relationships

### 3. Valid DTDFVocab Properties Reference:
```oml
// Core Properties (use only these):
DTDFVocab:producedData        // SensingComponent → DataTransmitted
DTDFVocab:producedFrom        // DataTransmitted → SensingComponent(s)
DTDFVocab:fromData           // Data → DataTransmitted
DTDFVocab:inputTo            // Model/Data → Enabler
DTDFVocab:enables            // Enabler → Service
DTDFVocab:provides           // Service → Insight/Action
DTDFVocab:IsAutomatic        // Action → boolean
base:contains                // Environment → Components
base:desc                    // Any instance → description string
base:isContainedIn          // Component → Environment
DTDFVocab:hasEnvironment       // SystemUnderStudy → Environment
```

### 4. Systematic Fixing Process:

#### Step 1: Validate All Property References
- Check each DTDFVocab property against valid properties list
- Replace invalid properties with correct ones or remove entirely
- Ensure namespace prefixes are correct
- Make sure the capitalization matches exactly

#### Step 2: Fix Syntax Issues
- Verify bracket matching: `[ ]` pairs
- Confirm colon usage in instance declarations

#### Step 3: Resolve Relationship Consistency
- Ensure referenced instances exist
- Validate relationship directions match vocabulary semantics
- Fix any circular dependencies

#### Step 4: Preserve Original Intent
- Keep all meaningful instances from original code
- Maintain descriptive content and relationships where valid
- Only remove or modify elements that cause errors

### 5. Validation Checklist:
Before outputting fixed code, verify:
- [ ] All DTDFVocab properties are from the valid list above
- [ ] All referenced instances are defined in the code
- [ ] All brackets and colons are properly placed
- [ ] No circular references exist
- [ ] Instance names are unique and properly formatted
- [ ] Relationships follow correct vocabulary semantics
- [ ] Capitalization and spelling of properties are exact

## SYNTAX EXAMPLES:
{guiding_syntax}

## FIXING STRATEGY:
1. **Minimal Changes**: Make only necessary changes to fix errors
2. **Preserve Intent**: Keep original structure and content where possible
3. **Valid Properties Only**: Replace invalid properties with correct ones from the reference list
4. **Complete Relationships**: Ensure all referenced instances exist
5. **Proper Syntax**: Fix brackets, capitalization, and formatting issues

## CRITICAL RULES:
- NEVER invent new DTDFVocab properties - use only the valid ones listed above exactly
- ALWAYS preserve the original instance names unless they cause syntax errors
- ONLY remove code if it's completely invalid and cannot be corrected
- MAINTAIN all meaningful relationships and descriptions from the original

Generate ONLY the corrected OML code with no explanations or comments:
""")
        
        formatted_prompt = fix_prompt.format(
            oml_content=oml_content,
            validation_errors=validation_output,
            characteristics=json.dumps(characteristics, indent=2),
            vocab_context=vocab_context,
            guiding_syntax=guiding_syntax,
        )
        
        try:
            response = self.llm.invoke(formatted_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            response_metadata = getattr(response, 'response_metadata', {})
            response_text = self._clean_llm_response(response_text)
            response_text = response_text.replace("<", "").replace(">", "").replace(";", "")  # Remove any stray characters

            print("🔧 Fixed OML generated based on validation feedback")
            return response_text, response_metadata
        except Exception as e:
            print(f"❌ Error fixing OML with feedback: {e}")
            return oml_content  # Return original if fixing fails

    def _validate_oml_with_opencaesar(self, catalog_parent_path: Path, output_path: str = "report.txt") -> tuple[bool, str]:
        """
        Validate OML content using OpenCAESAR's OML Validate.
        Returns: (is_valid, validation_output) where validation_output contains error details if validation fails
        """
        try:
            # Convert to absolute path to avoid relative path issues
            abs_catalog_path = catalog_parent_path.resolve()
            
            # Check if the directory exists
            if not abs_catalog_path.exists():
                error_msg = f"Error: Directory {abs_catalog_path} does not exist"
                print(error_msg)
                return False, error_msg
            
            # Check if gradlew.bat exists in the directory
            gradlew_script = abs_catalog_path / ("gradlew.bat" if os.name == "nt" else "gradlew")
            if not gradlew_script.exists():
                error_msg = f"Error: {gradlew_script} not found"
                print(error_msg)
                print(f"Looking for gradlew in: {abs_catalog_path}")
                # List files in directory for debugging
                try:
                    files = list(abs_catalog_path.iterdir())
                    print(f"Files in directory: {[f.name for f in files[:10]]}")  # Show first 10 files
                except Exception as e:
                    print(f"Cannot list directory contents: {e}")
                return False, error_msg
            
            # Run the validation
            print("⚙️ Validating written OML file with OpenCAESAR...")
            print("Parameters:")
            print(f" - Working Directory: {abs_catalog_path}")
            print(f" - Gradlew Script: {gradlew_script}")
            print(f" - Output Report: {output_path}")
            
            # Execute the command with absolute path
            result = subprocess.run([
                str(gradlew_script),
                "owlReason"
            ],
            cwd=str(abs_catalog_path),
            capture_output=True,
            text=True,
            timeout=300)

            print("📤 OpenCAESAR OML validation result:")
            print(f"Return code: {result.returncode}")
            # print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")

            # Prepare validation output for feedback
            validation_output = f"Return code: {result.returncode}\n"
            if result.stderr:
                # regex to match the diagnostic lines with line/col and unresolved reference
                pattern = re.compile(r"\[\d+,\s*\d+\]: Couldn't resolve reference to .*")
                relevant = pattern.findall(result.stderr)
                relevant = "\n".join(relevant)
                print(f"Relevant stderr lines:\n{relevant}")
                validation_output += f"Stderr:\n{relevant}\n"

            return result.returncode == 0, validation_output
            
        except subprocess.TimeoutExpired:
            error_msg = "Error: OpenCAESAR validation timed out (300 seconds)"
            print(error_msg)
            return False, error_msg
        except FileNotFoundError as e:
            error_msg = f"Error: File not found - {e}. This usually means gradlew.bat is not in the expected location"
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Error during OpenCAESAR validation: {e}"
            print(error_msg)
            return False, error_msg

    def _load_oml_into_fuseki(self, catalog_parent_path: Path, timeout: int = 300) -> tuple[bool, str]:
        """Start Fuseki (triple store) and load OML using gradle tasks.

        Sequence:
          1. gradlew.bat startFuseki (or ./gradlew startFuseki on *nix)
          2. gradlew.bat owlLoad

        Returns: (success, combined_output)
        """
        try:
            abs_catalog_path = catalog_parent_path.resolve()
            if not abs_catalog_path.exists():
                msg = f"Error: Directory {abs_catalog_path} does not exist"
                print(msg)
                return False, msg
            gradlew_script = abs_catalog_path / ("gradlew.bat" if os.name == "nt" else "gradlew")
            if not gradlew_script.exists():
                msg = f"Error: {gradlew_script} not found (cannot start Fuseki / load OML)"
                print(msg)
                return False, msg

            print("🚀 Starting Fuseki server via Gradle...")
            start_cmd = [str(gradlew_script), "startFuseki"]
            start_proc = subprocess.run(
                start_cmd,
                cwd=str(abs_catalog_path),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            start_output = f"StartFuseki Return code: {start_proc.returncode}\nSTDOUT:\n{start_proc.stdout}\nSTDERR:\n{start_proc.stderr}\n"
            print(start_output)
            if start_proc.returncode != 0:
                return False, start_output

            print("📥 Loading OML into Fuseki (owlLoad)...")
            load_cmd = [str(gradlew_script), "owlLoad"]
            load_proc = subprocess.run(
                load_cmd,
                cwd=str(abs_catalog_path),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            load_output = f"owlLoad Return code: {load_proc.returncode}\nSTDOUT:\n{load_proc.stdout}\nSTDERR:\n{load_proc.stderr}\n"
            print(load_output)
            success = start_proc.returncode == 0 and load_proc.returncode == 0
            combined = start_output + "\n" + load_output
            return success, combined
        except subprocess.TimeoutExpired:
            msg = f"Error: Fuseki start or load timed out ({timeout}s)"
            print(msg)
            return False, msg
        except Exception as e:
            msg = f"Error during Fuseki load sequence: {e}"
            print(msg)
            return False, msg
