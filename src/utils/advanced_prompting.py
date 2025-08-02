"""
Advanced Prompt Engineering Techniques for Digital Twin Ontology Generation
Implements state-of-the-art prompting strategies for improved LLM performance.
"""

from typing import List, Dict, Any, Type
from pydantic import BaseModel
import json
import re


class AdvancedPromptEngineer:
    """Advanced prompt engineering techniques for Digital Twin characteristics extraction."""
    
    def create_chain_of_thought_prompt(self, description: str, docs_content: str, 
                                     schema: Type[BaseModel]) -> str:
        """Create an advanced Chain of Thought prompt with domain expertise."""
        
        schema_fields = list(schema.model_fields.keys())
        
        prompt = f"""
You are a Digital Twin ontology expert with deep knowledge of cyber-physical systems, IoT architectures, and industrial automation. Your task is to extract precise technical characteristics from documentation.

DOMAIN EXPERTISE CONTEXT:
- Digital Twins create bidirectional connections between physical and digital worlds
- They involve real-time data exchange, model synchronization, and decision support
- Technical implementation requires specific protocols, architectures, and methodologies
- Fidelity and validation are critical for trustworthy Digital Twin operations

ANALYSIS TASK:
Extract the following characteristics: {', '.join(schema_fields)}

DOCUMENT ANALYSIS:
{docs_content}

STEP-BY-STEP REASONING:

Step 1: Document Scanning and Technical Term Identification
Let me scan the documents for technical terms, system descriptions, and implementation details...

[Identify key technical terms, system components, and methodologies mentioned]

Step 2: Characteristic-Specific Analysis
For each characteristic, I'll analyze the relevant document sections:

{self._generate_characteristic_analysis_steps(schema_fields)}

Step 3: Technical Detail Extraction
For each found characteristic, I'll extract specific technical details including:
- Technologies and tools mentioned
- Specifications and parameters
- Protocols and standards
- Implementation approaches
- Performance metrics

Step 4: Validation and Completeness Check
I'll verify that extracted information is:
- Technically accurate and grounded in the documents
- Sufficiently detailed for ontology modeling
- Free from assumptions not supported by the text

EXTRACTION RESULTS:
Based on my analysis, here are the extracted characteristics in the required JSON format:

{self._get_format_instructions(schema)}

Note: I will only include information explicitly mentioned or clearly implied in the documents. If no evidence exists for a characteristic, I will indicate "Not Found".
"""
        return prompt
    
    def _generate_characteristic_analysis_steps(self, fields: List[str]) -> str:
        """Generate analysis steps for each characteristic field."""
        steps = []
        for i, field in enumerate(fields, 1):
            field_formatted = field.replace('_', ' ').title()
            steps.append(f"   {i}. {field_formatted}: Search for information about {field.replace('_', ' ')}")
        return '\n'.join(steps)
    
    def _get_format_instructions(self, schema: Type[BaseModel]) -> str:
        """Get formatting instructions for the schema."""
        fields = {}
        for field_name, field_info in schema.model_fields.items():
            fields[field_name] = "string or 'Not Found'"
        
        return json.dumps(fields, indent=2)
    
    def create_self_consistency_prompts(self, base_prompt: str, num_variants: int = 3) -> List[str]:
        """Create multiple prompt variants for self-consistency."""
        variants = []
        
        # Original prompt
        variants.append(base_prompt)
        
        # Variant 1: Focus on technical implementation
        variant1 = base_prompt.replace(
            "STEP-BY-STEP REASONING:",
            "TECHNICAL IMPLEMENTATION FOCUS:\nAnalyze the documents with emphasis on technical implementation details, system architecture, and technology stack.\n\nSTEP-BY-STEP REASONING:"
        )
        variants.append(variant1)
        
        # Variant 2: Focus on operational aspects
        variant2 = base_prompt.replace(
            "STEP-BY-STEP REASONING:",
            "OPERATIONAL PERSPECTIVE:\nExamine the documents from an operational standpoint, focusing on how the system works in practice and real-world deployment.\n\nSTEP-BY-STEP REASONING:"
        )
        variants.append(variant2)
        
        # Variant 3: Focus on system integration
        if num_variants > 3:
            variant3 = base_prompt.replace(
                "STEP-BY-STEP REASONING:",
                "SYSTEM INTEGRATION ANALYSIS:\nAnalyze how different components integrate and interact within the Digital Twin ecosystem.\n\nSTEP-BY-STEP REASONING:"
            )
            variants.append(variant3)
        
        return variants[:num_variants]
    
    def create_role_based_prompt(self, description: str, docs_content: str, 
                                schema: Type[BaseModel], role: str = "systems_engineer") -> str:
        """Create role-based prompts for different expert perspectives."""
        
        roles = {
            "systems_engineer": {
                "description": "You are a senior systems engineer with 15+ years of experience in cyber-physical systems and Digital Twin implementations.",
                "focus": "system architecture, technical specifications, and engineering requirements"
            },
            "data_scientist": {
                "description": "You are a data scientist specializing in Industrial IoT and Digital Twin analytics.",
                "focus": "data flows, model validation, and analytics capabilities"
            },
            "operations_manager": {
                "description": "You are an operations manager responsible for Digital Twin deployment and maintenance.",
                "focus": "operational aspects, deployment considerations, and practical implementation"
            },
            "security_expert": {
                "description": "You are a cybersecurity expert specializing in Industrial IoT and Digital Twin security.",
                "focus": "security protocols, data protection, and safety mechanisms"
            }
        }
        
        role_info = roles.get(role, roles["systems_engineer"])
        
        prompt = f"""
EXPERT ROLE: {role_info['description']}

EXPERTISE FOCUS: Your analysis should emphasize {role_info['focus']}.

TASK: Extract Digital Twin characteristics from the provided technical documentation, focusing on aspects most relevant to your expertise while maintaining comprehensive coverage.

DOCUMENT CONTENT:
{docs_content}

TARGET CHARACTERISTICS:
{description}

EXPERT ANALYSIS:
Drawing from your professional experience in {role_info['focus']}, analyze the documents and extract precise, technically accurate information for each characteristic.

Pay special attention to:
- Technical specifications and implementation details
- Industry standards and best practices mentioned
- Performance requirements and constraints
- Practical deployment considerations

OUTPUT FORMAT:
{self._get_format_instructions(schema)}

Ensure your extraction reflects the depth of analysis expected from someone with your expertise level.
"""
        return prompt
    
    def create_verification_prompt(self, extracted_data: Dict[str, Any], 
                                 original_docs: str) -> str:
        """Create a verification prompt to validate extracted information."""
        
        prompt = f"""
VERIFICATION TASK: You are a quality assurance expert reviewing extracted Digital Twin characteristics for accuracy and grounding.

EXTRACTED CHARACTERISTICS:
{json.dumps(extracted_data, indent=2)}

ORIGINAL DOCUMENTS:
{original_docs}

VERIFICATION CRITERIA:
1. FACTUAL ACCURACY: Is each extracted detail actually mentioned in the documents?
2. TECHNICAL PRECISION: Are technical terms and specifications used correctly?
3. COMPLETENESS: Are there missed details that should be included?
4. SPECIFICITY: Are descriptions sufficiently detailed and specific?

VERIFICATION PROCESS:
For each characteristic, verify:
- Quote the specific document text that supports the extraction
- Identify any technical inaccuracies or misinterpretations
- Note any important details that were missed
- Assess the level of detail and specificity

VERIFICATION RESULTS:
Provide a structured assessment indicating:
- ✅ VERIFIED: Information is accurately extracted and well-grounded
- ⚠️ PARTIAL: Information is generally correct but lacks detail or has minor issues
- ❌ INCORRECT: Information is not supported by the documents or contains errors
- 📝 MISSING: Important information was not extracted

Format your response as:
[Characteristic Name]: [Status] - [Explanation and specific document quotes]
"""
        return prompt
    
    def create_iterative_refinement_prompt(self, initial_extraction: Dict[str, Any], 
                                         feedback: str, docs_content: str) -> str:
        """Create a prompt for iterative refinement based on feedback."""
        
        prompt = f"""
ITERATIVE REFINEMENT TASK: Improve the extracted Digital Twin characteristics based on quality feedback.

INITIAL EXTRACTION:
{json.dumps(initial_extraction, indent=2)}

FEEDBACK FOR IMPROVEMENT:
{feedback}

ORIGINAL DOCUMENTS:
{docs_content}

REFINEMENT OBJECTIVES:
1. Address specific feedback points
2. Increase technical detail and specificity
3. Ensure all claims are properly grounded in the documents
4. Improve clarity and precision of descriptions

REFINEMENT PROCESS:
1. Analyze the feedback to understand improvement areas
2. Re-examine the documents with refined focus
3. Enhance descriptions with additional technical details
4. Verify that all improvements are document-supported

REFINED EXTRACTION:
Provide the improved extraction in the same JSON format, with enhanced detail and accuracy based on the feedback.
"""
        return prompt
    
    def create_oml_generation_prompt(self, characteristics: Dict[str, Any], 
                                   vocab_context: str, examples: str = "") -> str:
        """Create an advanced prompt for OML generation with better structure."""
        
        prompt = f"""
ONTOLOGY MODELING EXPERT: You are a senior ontology engineer specializing in Digital Twin formal representations using OML (Ontological Modeling Language).

TASK: Generate syntactically correct and semantically rich OML code for a Digital Twin system based on extracted characteristics.

EXTRACTED CHARACTERISTICS:
{json.dumps(characteristics, indent=2)}

VOCABULARY REFERENCE:
{vocab_context}

{examples}

OML GENERATION GUIDELINES:

1. INSTANCE NAMING CONVENTIONS:
   - Use descriptive, camelCase names reflecting functionality
   - Example: temperatureSensor, predictionService, cloudDeployment

2. RELATIONSHIP MODELING:
   - Model data flows using appropriate predicates (inputTo, outputFrom, etc.)
   - Establish service relationships (enables, provides, supports)
   - Connect physical and virtual components appropriately

3. DESCRIPTION DETAIL:
   - Use base:desc for rich, technical descriptions
   - Include specific technologies, protocols, and specifications
   - Maintain traceability to extracted characteristics

4. ARCHITECTURAL PATTERNS:
   - Group related instances logically
   - Model hierarchical relationships where appropriate
   - Ensure consistency with Digital Twin architectural principles

5. SYNTAX VALIDATION:
   - Proper bracket matching and indentation
   - Correct predicate usage from vocabulary
   - Valid OML instance declaration syntax

GENERATION PROCESS:
1. Analyze characteristics for instance candidates
2. Determine relationships between instances
3. Generate OML code with proper syntax
4. Validate semantic consistency

GENERATED OML CODE:
"""
        return prompt


# Factory function
def create_advanced_prompt_engineer() -> AdvancedPromptEngineer:
    """Create an advanced prompt engineer instance."""
    return AdvancedPromptEngineer()
