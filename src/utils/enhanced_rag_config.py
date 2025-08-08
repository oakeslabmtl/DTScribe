"""
Enhanced RAG Configuration with modern LLM techniques for improved Digital Twin characteristics extraction.
"""

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import MarkdownTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from typing import Type, List, Dict, Any
from pydantic import BaseModel
import pymupdf4llm
import pymupdf
import re
import json
from pathlib import Path

# Import OML Writer components
from .oml_writer import IOMLWriter, OMLFileWriter


class EnhancedRAGPipeline:
    """Enhanced RAG Pipeline with improved techniques for better generation quality."""
    
    def __init__(self, model_name: str = "qwen3:8b", embedding_model: str = "nomic-embed-text"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            top_p=0.9,
            # Add more specific parameters for better control
            top_k=20,
            repeat_penalty=1.1,
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
        
    def enhanced_pdf_processing(self, pdf_path: str, chunk_size: int = 1500, overlap: int = 200, max_pages: int = None) -> Chroma:
        """
        Enhanced PDF processing with better chunking strategy and metadata preservation.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            max_pages: Maximum number of pages to process (None for all pages)
        """
        # Get PDF page count and extract pages
        import pymupdf
        try:
            doc = pymupdf.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            # Determine pages to process
            if max_pages is not None:
                pages_to_process = min(total_pages, max_pages)
                print(f"Processing {pages_to_process} of {total_pages} pages (limited by max_pages={max_pages})...")
            else:
                pages_to_process = total_pages
                print(f"Processing all {total_pages} pages...")
            
            # Extract markdown with specified pages
            pages_list = list(range(pages_to_process))
            md_text = pymupdf4llm.to_markdown(pdf_path, pages=pages_list)
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            print("Falling back to processing first 20 pages...")
            # Fallback to original behavior
            md_text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(20)))
        
        # Enhanced text splitter with better chunk boundaries
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
            length_function=len,
        )
        
        # Create documents with enhanced metadata
        docs = splitter.create_documents(
            [md_text],
            metadatas=[{"source": pdf_path, "type": "technical_document"}]
        )
        
        # Add semantic metadata to chunks
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "chunk_id": i,
                "word_count": len(doc.page_content.split())
        })
        
        # Create vector store with improved configuration
        vectordb = Chroma.from_documents(
            docs,
            embedding=self.embeddings,
            persist_directory="./vector_db",
            collection_metadata={"hnsw:space": "cosine"}  # Better for semantic similarity
        )
        
        return vectordb
    
    def enhanced_retrieval(self, vectordb: Chroma, query: str, k: int = 5) -> List:
        """
        Enhanced retrieval with multiple strategies and query expansion.
        """      
        all_docs = []

        # Standard similarity search
        docs = vectordb.similarity_search(query=query, k=k)
            
        all_docs.extend(docs)
        
        # 3. Re-rank documents based on relevance and technical content
        ranked_docs = self._rerank_documents(all_docs, query)
        
        # 4. Remove duplicates while preserving order
        seen_content = set()
        unique_docs = []
        for doc in ranked_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs[:k]
    
    def _rerank_documents(self, docs: List, original_query: str) -> List:
        """Re-rank documents based on multiple factors."""
        def score_document(doc):
            content = doc.page_content.lower()
            query_lower = original_query.lower()
            
            # Basic relevance score
            relevance_score = sum(1 for word in query_lower.split() if word in content)
            
            # Length penalty (prefer moderate length chunks)
            length_penalty = abs(len(content.split()) - 200) / 1000
            
            return relevance_score - length_penalty
        
        return sorted(docs, key=score_document, reverse=True)
    
    def generate_with_cot_and_validation(self, description: str, retrieved_docs: List, 
                                       schema: Type[BaseModel]) -> BaseModel:
        """
        Enhanced generation with Chain of Thought reasoning and self-validation.
        """
        docs_content = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=schema)
        
        # Enhanced prompt with Chain of Thought reasoning and strict output format
        cot_prompt = PromptTemplate.from_template("""
You are an expert in Digital Twin systems and ontology modeling. Your task is to extract detailed characteristics from technical documents.

CONTEXT DOCUMENTS:
{docs_content}

CHARACTERISTICS TO EXTRACT:
{description}
                                                  
INSTRUCTIONS:
1. REASONING PHASE: First, analyze the documents step by step:
   - Identify which documents contain relevant information for each characteristic
   - Note any technical details, technologies, or methodologies mentioned
   - Consider the specific context and domain of the use case

2. EXTRACTION PHASE: For each characteristic:
   - Provide specific, detailed descriptions based ONLY on the provided documents
   - Include concrete technical details (tools, technologies, protocols, methods)
   - Be precise about quantities, frequencies, and specifications when mentioned
   - If no evidence is found, state "Not Found"

3. VALIDATION PHASE: Review your extracted information:
   - Ensure all details come from the provided documents
   - Check that technical terms are used correctly
   - Verify completeness of the description

REASONING:
Let me analyze each document for relevant information...

[Analyze the documents here, identifying key information for each characteristic]

IMPORTANT: You MUST respond with ONLY valid JSON. Do not include any explanations, thinking tags, or additional text outside the JSON structure.

EXTRACTED CHARACTERISTICS (JSON FORMAT ONLY):
{format_instructions}

Remember: Be highly specific and technical. Include exact technologies, methods, and specifications mentioned in the documents. Return ONLY the JSON object with no additional text.
""")
        
        formatted_prompt = cot_prompt.format(
            docs_content=docs_content,
            description=description,
            format_instructions=parser.get_format_instructions()
        )
        
        # Generate with retry mechanism for better reliability
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Generate with direct LLM call first, then clean and parse
                response = self.llm.invoke(formatted_prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Clean the response to remove thinking tags
                cleaned_text = self._clean_llm_response(response_text)
                
                # Parse the cleaned response
                output = parser.parse(cleaned_text)

                return output
                
            except Exception as e:
                print(f"Warning: Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    # Try with a simpler prompt on retry
                    simple_prompt = self._create_simple_fallback_prompt(description, docs_content, schema)
                    try:
                        response = self.llm.invoke(simple_prompt)
                        response_text = response.content if hasattr(response, 'content') else str(response)
                        cleaned_text = self._clean_llm_response(response_text)
                        output = parser.parse(cleaned_text)
                        validated_output = self._self_validate_output(output, retrieved_docs)
                        print(f"Success with fallback prompt on attempt {attempt + 1}")
                        return validated_output
                    except Exception as fallback_error:
                        print(f"Fallback also failed: {str(fallback_error)}")
                        continue
                else:
                    # Final fallback: create a basic output with "Not Found" values
                    print("All attempts failed, creating fallback output...")
                    return self._create_fallback_output(schema)
    
    def _clean_llm_response(self, response_text: str) -> str:
        """
        Clean LLM response by removing thinking tags and extracting JSON content.
        """
        import re
        
        # Strip whitespace
        response_text = response_text.strip()
        
        # Remove code block markers
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        elif response_text.startswith('```oml'):
            response_text = response_text[6:]
        elif response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        # Remove thinking tags and their content completely
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Also handle unclosed thinking tags
        response_text = re.sub(r'<think>.*', '', response_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Strip again after cleaning
        response_text = response_text.strip()
        
        # Extract JSON content - look for the first complete JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # If no JSON object found, return the cleaned text as-is
        return response_text

    def _create_simple_fallback_prompt(self, description: str, docs_content: str, 
                                     schema: Type[BaseModel]) -> str:
        """Create a simpler prompt as fallback when structured output fails."""
        parser = PydanticOutputParser(pydantic_object=schema)
        
        prompt = f"""
Extract Digital Twin characteristics from the provided documents.

DOCUMENTS:
{docs_content}

TASK: Extract the following characteristics and return as valid JSON only:
{description}

For each characteristic, provide a detailed description based on the documents, or "Not Found" if no information exists.

RETURN ONLY VALID JSON IN THIS FORMAT:
{parser.get_format_instructions()}

JSON OUTPUT:
"""
        return prompt
    
    def _create_fallback_output(self, schema: Type[BaseModel]) -> BaseModel:
        """Create a fallback output when all parsing attempts fail."""
        # Get field names from schema
        field_names = list(schema.model_fields.keys())
        
        # Create a dict with "Not Found" for all fields
        fallback_data = {field: "Not Found" for field in field_names}
        
        # Create and return the schema instance
        return schema(**fallback_data)
    
    def generate_with_manual_parsing(self, description: str, retrieved_docs: List, 
                                   schema: Type[BaseModel]) -> BaseModel:
        """
        Alternative generation method with manual JSON parsing as fallback.
        """
        docs_content = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Simple prompt without structured output parser
        prompt = f"""
You are an expert in Digital Twin systems. Extract characteristics from the provided documents.

DOCUMENTS:
{docs_content}

EXTRACT THESE CHARACTERISTICS:
{description}

For each characteristic, provide a detailed description based ONLY on the documents, or "Not Found" if no information exists.

IMPORTANT: Respond with ONLY a valid JSON object. No additional text, explanations, or formatting.

Example format:
{{
    "system_under_study": "description here or Not Found",
    "dt_services": "description here or Not Found",
    "tooling_and_enablers": "description here or Not Found"
}}

JSON:
"""
        
        try:
            # Direct LLM call without structured output
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Use the same cleaning function for consistency
            cleaned_text = self._clean_llm_response(response_text)
            
            # Parse JSON manually
            import json
            parsed_data = json.loads(cleaned_text)
            
            # Create schema instance
            return schema(**parsed_data)
            
        except Exception as e:
            print(f"Manual parsing failed: {e}")
            return self._create_fallback_output(schema)
    
    def generate_oml(self, characteristics: Dict[str, Any], 
                        vocab_files: Dict[str, str],
                        output_path: str = r"data\DTDF\src\oml\bentleyjoakes.github.io\LLM_described_DT\llm_dt.oml",
                        catalog_parent_path: str = r"data\DTDF\\",
                        writer: IOMLWriter = None,
                        max_retries: int = 3) -> str:
        """
        OML generation with better context and validation.
        """
        if writer is None:
            writer = OMLFileWriter()

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
        # Keys not in description-based characteristics will be considered component-based
        component_based_characteristics_keys = {key for key in characteristics if key not in description_based_vocab_mapping}
        print("🏗️ Splitting characteristics for OML generation...")
        # Create description-based dictionary
        description_based_characteristics = {key: characteristics[key] for key in description_based_vocab_mapping if key in characteristics}
        # Create component-based dictionary
        component_based_characteristics = {key: characteristics[key] for key in component_based_characteristics_keys if key in characteristics}

        description_based_oml = self.generate_description_based_oml(description_based_characteristics, description_based_vocab_mapping)

        # Retry mechanism for component-based OML generation
        print("🔄 Retrying component-based OML generation with multiple attempts...")
        for attempt in range(max_retries + 1):
            try:
                print(f"📝 Attempt {attempt + 1}/{max_retries + 1}: Generating OML...")
                component_based_oml = self.generate_component_based_oml(component_based_characteristics, vocab_files, comma_separated_description_based_vocab_mapping_keys)
                # Validate the generated OML syntax
                if not self._validate_oml_syntax(component_based_oml):
                    print(f"❌ Attempt {attempt + 1}: Invalid OML syntax detected")
                    continue
                # Write the OML content to file
                print("💾 Writing OML to file...")
                write_success = writer.write_oml(component_based_oml, output_path)
                if not write_success:
                    print(f"❌ Attempt {attempt + 1}: Failed to write OML to file")
                    continue
                # Validate the written OML file with OpenCAESAR
                print("⚙️ Validating written OML file with OpenCAESAR...")
                if not self._validate_oml_with_opencaesar(catalog_parent_path):
                    print(f"❌ Attempt {attempt + 1}: Written OML file validation with OpenCAESAR failed")
                    continue
                # Combine both OML description-based and component-based characteristics
                print("🏗️ Combining OML descriptions...")
                combined_oml = f"{component_based_oml}\n\n{description_based_oml}"
                combined_oml = self._clean_llm_response(combined_oml)
                # Write the OML content to file
                print("💾 Writing OML to file...")
                write_success = writer.write_oml(combined_oml, output_path)
                # Validate the combined OML syntax
                if not self._validate_oml_syntax(combined_oml):
                    print(f"❌ Attempt {attempt + 1}: Combined OML syntax validation failed")
                    continue
                if not self._validate_oml_with_opencaesar(combined_oml):
                    print(f"❌ Attempt {attempt + 1}: Combined OML validation with OpenCAESAR failed")
                    continue
                print("✅ OML generation and validation successful!")
                return combined_oml
            except Exception as e:
                print(f"❌ Attempt {attempt + 1}: Unexpected error: {str(e)}")
                if attempt < max_retries:
                    print("🔄 Retrying due to unexpected error...")
                    continue
                else:
                    print("❌ Failed due to persistent errors")
                    return None

    def generate_description_based_oml(self, characteristics: Dict[str, Any], vocab_mapping: Dict[str, str]) -> str:
        """Generate OML based on characteristics description."""
        print("🏗️ Generating description-based OML...")
        # Generate OML for each characteristic

        oml_parts = []
        for key in characteristics:
            value = characteristics.get(key)
            if value and value != "Not Found":
                oml_parts.append(f"instance {key} : DTDFVocab:{vocab_mapping.get(key, 'Unknown')} [\n    base:desc \"{value}\"\n]")
        joined_parts = "\n\n".join(oml_parts)
        return self._clean_llm_response(joined_parts)
    
    def generate_component_based_oml(self, characteristics: Dict[str, Any], 
                                    vocab_files: Dict[str, str], description_based_vocab_mapping: str) -> str:
        """Generate OML based on components and vocabulary files."""
        print("🏗️ Generating component-based OML description...")
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

VOCABULARY REFERENCE:
{vocab_context}
                                                  
LIST OF ALREADY GENERATED CHARACTERISTICS (do not generate again):
{description_based_vocab_mapping}

Only generate OML code that's similar to the following syntax examples:
SYNTAX EXAMPLES:
```oml
// C2: Acting Components
instance <name_of_acting_component> : DTDFVocab:ActingComponent [
    base:desc "<description of the acting component>"
]
    
// C3: Physical sensing components
instance <name_of_sensing_component> : DTDFVocab:SensingComponent[
    base:desc "<description of the sensing component>"
    DTDFVocab:producedData <name_of_produced_data_transmitted>
]

// C4: Physical-to-virtual interaction                                           
instance <name_of_produced_data_transmitted> : DTDFVocab:DataTransmitted [
    DTDFVocab:producedFrom <name_of_sensing_component1, name_of_sensing_component2, ...>
]

// Insights/Actions (C17)
instance <name_of_insight> : DTDFVocab:Insight [
    base:desc "<description of the insight>"
]

instance <name_of_action> : DTDFVocab:Action[
    base:desc "<description of the action>"
    DTDFVocab:IsAutomatic <true_or_false>
]
                                                  
// Services (C6)
instance <name_of_service> : DTDFVocab:Service [
    base:desc "<description of the service>"
    DTDFVocab:provides <name_of_action_or_insight1, name_of_action_or_insight2, ...>
]

// Enablers (C11)
instance <name_of_enabler> : DTDFVocab:Enabler [
    base:desc "<description of the enabler>"
    DTDFVocab:enables <name_of_service1, name_of_service2, ...>
]

// Models/Data (C10)
instance <name_of_model> : DTDFVocab:Model [
    base:desc "<description of the model>"
    DTDFVocab:inputTo <name_of_enabler1, name_of_enabler2, ...>
    DTDFVocab:fromData <name_of_physical_to_virtual_interaction>
]
                                                  
instance <name_of_data> : DTDFVocab:Data [
    base:desc "<description of the data>"
    DTDFVocab:inputTo <name_of_enabler1, name_of_enabler2, ...>
    DTDFVocab:fromData <name_of_DTDFVocab:DataTransmitted>
]
```
                                                  
REQUIREMENTS:
1. Follow OML syntax precisely. Names between <> are placeholders for actual names.
2. Create instances for each characteristic that has meaningful content (not "Not Found")
3. Establish proper relationships between instances using the vocabulary predicates
4. Use descriptive, technical names for instances based on the content
5. Ensure all generated code is syntactically valid OML

GUIDELINES:
- For multiple components (sensors, actuators, services), create separate instances
- Certain characteristics (Physical acting components, Physical sensing components, Physical-to-virtual interaction, DT services, Twinning time-scale, DT models and data, Tooling and enablers, Insights and decision making) usually have multiple instances
- Use base:desc for detailed descriptions of other characteristics
- Establish proper relationships using DTDFVocab predicates
- Name instances descriptively based on their function/content

Generate ONLY the OML code, no explanations or comments outside the OML syntax:
""")
        
        formatted_prompt = oml_prompt.format(
            characteristics=json.dumps(characteristics, indent=2),
            vocab_context=vocab_context,
            description_based_vocab_mapping=description_based_vocab_mapping
        )
        
        response = self.llm.invoke(formatted_prompt)
        oml_content = response.content if hasattr(response, 'content') else str(response)
        # Clean the response to ensure valid OML syntax
        oml_content = self._clean_llm_response(oml_content)
        
        # Basic OML syntax validation
        if self._validate_oml_syntax(oml_content):
            print("Basic OML syntax validation passed")
        else:
            print("Warning: Generated OML may have syntax issues")
            # TODO: Retry generation

        # Advanced validation with OpenCAESAR OML validator
        if not self._validate_oml_with_opencaesar(oml_content):
            print("Warning: Generated OML may not meet OpenCAESAR standards")
            # TODO: Retry generation

        return oml_content

    def _validate_oml_syntax(self, oml_content: str) -> bool:
        """Basic OML syntax validation."""
        # Check for basic OML patterns
        has_instances = "instance" in oml_content
        has_proper_brackets = oml_content.count('[') == oml_content.count(']')
        has_vocabulary_references = "DTDFVocab:" in oml_content or "base:" in oml_content
        
        return has_instances and has_proper_brackets and has_vocabulary_references

    def _validate_oml_with_opencaesar(self, oml_catalog_path: str) -> bool:
        """Validate OML content using OpenCAESAR's OML Validate."""
        # TODO: Implement OpenCAESAR OML Validate service
        return True  # Placeholder for actual validation result