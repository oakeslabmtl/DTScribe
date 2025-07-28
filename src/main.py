import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Dict, Any

import pandas as pd

from utils.rag_config import load_and_split_pdf, generate_desc, llm
import shutil
from models.schemas import (
    Block1Characteristics, 
    Block2Characteristics, 
    Block3Characteristics, 
    Block4Characteristics, 
    Block5Characteristics, 
    Block6Characteristics,
    # DTCharacteristics
    )

# ---- Config ----
load_dotenv()
os.environ['LANGSMITH_TRACING'] = os.getenv("LANGSMITH_TRACING")
os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ['USER_AGENT'] = os.getenv("USER_AGENT")


# ---- Graph State setup ----
class State(TypedDict):
    pdf_path: str
    vectordb: object
    extracted_characteristics: Dict[str, Any]
    oml_output: str


# --------- functions for eahc node -------
def preprocess(state: State) -> State:
    """
    Load a PDF, split it into chunks, and store them in a vector database.
    """
    if "pdf_path" not in state:
        raise ValueError("PDF path must be provided.")

    print("Loading and splitting PDF...")
    vectordb = load_and_split_pdf(state["pdf_path"])
    print("Vector DB created...")

    return {"vectordb": vectordb}


#######################################################3333####################
###############################################################################

# NOTE for Prompt engineering
# - broaden the scope of the retrieval query by making it less specific
# - Insist on extracting the characteristics in a more detailed way, focusing on the specific use case

###############################################################################
###############################################################################

def extractor_block1(state: State) -> State:
    """Extract characterrstics from the first block of the Digital Twin description."""

    if "vectordb" not in state:
        raise ValueError("Vector database must be provided.")
    
    block1_retrieval_query= """(
        The System under study, the Physical Twin (PT) of the system of interest.
        The services provided by the Digital Twin (DT), such as optimization, task planning, and visualization.
        The tools or enablers used to achieve the goals of the Digital Twin (DT) and provide services.
        """
    
    print("Retrieving documents for Block 1 characteristics...")
    retrieved_docs = state["vectordb"].similarity_search(block1_retrieval_query, k=3)  

    block1_description = """
        System_under_study: What is the System under study in this case?, i.e., the Physical Twin of the system of interest in this case.
        dt_services: What are the specific services which the DT provides to the users and the physical system?
        tooling_and_enablers: What are specific the tools or enablers that are used to achieve the goals of the DT? i.e., they enable the DT to provide the DT services. Mention concisely, the technologies, tools and techniques and thei functionality.
        """

    print("Extracting Block 1 characteristics...")
    output = generate_desc(block1_description, retrieved_docs, Block1Characteristics)

    return {"extracted_characteristics": output.model_dump(exclude_none=True)}
    # return {"extracted_characteristics": output}


def extractor_block2(state: State) -> State:
    """Extract characterrstics from the second block of the Digital Twin description."""

    if "vectordb" not in state:
        raise ValueError("Vector database must be provided.")
    
    block2_retrieval_query= """
        time-scale use and the time rates for services and synchronization.
        multiplicities, or the internal twins in the system.
        orchestration of the DT system, components, and services as a whole.
        information exchange with external information systems.
        """
    
    print("Retrieving documents for Block 2 characteristics...")
    retrieved_docs = state["vectordb"].similarity_search(block2_retrieval_query, k=5)  

    block2_description = """
        twinning_time_scale: Describe the time-scale use and the time rates for the Digital Twin (DT) services and DT-to-PT synchronization. 
        multiplicities: Describe the multiplicities, i.e., the internal twins that compose the Digital Twin (DT) system, which can be implemented in a centralized or decentralized way.
        dt_constellation: Describe the orchestration of the Digital Twin (DT) system, components, and services as a whole.
        horizontal_integration: Describe the information exchange with external information systems not limited to other Digital Twins (DTs).
        """

    print("Extracting Block 2 characteristics...")
    output = generate_desc(block2_description, retrieved_docs, Block2Characteristics)

    # merge the output with the previous characteristics
    if "extracted_characteristics" in state:
        existing_characteristics = state["extracted_characteristics"]
        existing_characteristics.update(output.model_dump(exclude_none=True))
        return {"extracted_characteristics": existing_characteristics}

    return {"extracted_characteristics": output.model_dump(exclude_none=True)}


def extractor_block3(state: State) -> State:
    """Extract characterrstics from the third block of the Digital Twin description."""

    if "vectordb" not in state:
        raise ValueError("Vector database must be provided.")
    
    block3_retrieval_query= """
        The Digital Twin (DT) components, including available models and data.
        The available acting components in the Digital Twin (DT), the mechanisms the Digital Twin (DT) can use to act on the Physical Twin (PT).
        The available sensing components in the Digital Twin (DT), the mechanisms the Physical Twin (PT) can use to transfer data to the Digtal Twin (DT).
        The fidelity and validity considerations behind the models that constitute the Digital Twin (DT), including verification and validation mechanisms, uncertainty, and errors.
        """
    
    print("Retrieving documents for Block 3 characteristics...")
    retrieved_docs = state["vectordb"].similarity_search(block3_retrieval_query, k=3)  

    block3_description = """
            dt_models_and_data: Describes the Digital Twin (DT) components, including available models and data, and what is their role in the Digital Twin (DT) constellation.
            physical_acting_components: Describes the available acting components in the Digital Twin (DT) constellation, i.e., the mechanisms the Digital Twin (DT) can use to act on the Physical Twin (PT).
            physical_sensing_components: Describes the available sensing components in the Digital Twin (DT) constellation, i.e., the mechanisms the Physical Twin (PT) can use to transfer data to the Digtal Twin (DT).
            fidelity_and_validity_considerations: Describes the fidelity and validity considerations behind the models that constitute the Digital Twin (DT), including verification and validation mechanisms, uncertainty, and errors.
      """

    print("Extracting Block 3 characteristics...")
    output = generate_desc(block3_description, retrieved_docs, Block3Characteristics)

    # merge the output with the previous characteristics
    if "extracted_characteristics" in state:
        existing_characteristics = state["extracted_characteristics"]
        existing_characteristics.update(output.model_dump(exclude_none=True))
        return {"extracted_characteristics": existing_characteristics}

    return {"extracted_characteristics": output.model_dump(exclude_none=True)}


def extractor_block4(state: State) -> State:

    """Extract characterrstics from the fourth block of the Digital Twin description."""

    if "vectordb" not in state:
        raise ValueError("Vector database must be provided.")
    
    block4_retrieval_query= """
        Data transmission from PT to DT, including inputs and events that the Digital Twin (DT) processes.
        Data transmission from Digital Twin (DT) to Physical Twin (PT), including outputs the Digital Twin (DT) generates as part of its services.
        The technical network connection details between Physical Twin (PT) and Digital Twin (DT), including the network protocols and architectures.
        The technical hosting aspects of the Digital Twin (DT) and the associated technology."
    """
    
    print("Retrieving documents for Block 4 characteristics...")
    retrieved_docs = state["vectordb"].similarity_search(block4_retrieval_query, k=5)  

    block4_description = """
            physical_to_virtual_interaction: Describes the interactions from the physical world to the virtual world, i.e., the data transmitted from Physical Twin (PT) to Digital Twin (DT), including inputs and events that the Digital Twin (DT) processes.
            virtual_to_physical_interaction: Describes the interactions from the virtual world to the physical world, i.e., the data transmitted from Digital Twin (DT) to Physical Twin (PT), including outputs the Digital Twin (DT) generates as part of its services.
            dt_technical_connection: Describes the technical network connection details between Physical Twin (PT) and Digital Twin (DT), including the network protocols and architectures."
            dt_hosting_deployment: Describes the technical hosting aspects of the Digital Twin (DT) and the associated technology."
    """

    print("Extracting Block 4 characteristics...")
    output = generate_desc(block4_description, retrieved_docs, Block4Characteristics)

    # merge the output with the previous characteristics
    if "extracted_characteristics" in state:
        existing_characteristics = state["extracted_characteristics"]
        existing_characteristics.update(output.model_dump(exclude_none=True))
        return {"extracted_characteristics": existing_characteristics}

    return {"extracted_characteristics": output.model_dump(exclude_none=True)}


def extractor_block5(state: State) -> State:

    """Extract characterrstics from the fifth block of the Digital Twin description."""

    if "vectordb" not in state:
        raise ValueError("Vector database must be provided.")
    
    block5_retrieval_query= """
        The lifecycle phases in which the Digital Twin (DT) takes place. Which representation phase the Digital Twin (DT) covers of its physical counterpart, i.e., as designed (ideal), as manufactured, or as operated.
        The engineering process of Digital Twin (DT) implementation, including the development process, quality assurance, and definition of requirements. Milestones of the Digital Twin (DT) engineering process over time and intended upgrades. 
        The insights and decision making, i.e., indirect outputs of the Digital Twin (DT), which have no direct effect on the Physical Twin (PT) such as update of parameters, plans, and so on.
        The standards being followed for the engineering of the Digital Twin (DT) and its components.
    """
    
    print("Retrieving documents for Block 5 characteristics...")
    retrieved_docs = state["vectordb"].similarity_search(block5_retrieval_query, k=7)  

    block5_description = """
        life_cycle_stages: Describes the lifecycle phases in which the Digital Twin (DT) takes place. It also informs which representation phase the Digital Twin (DT) covers of its physical counterpart, i.e., as designed (ideal), as manufactured, or as operated.
        twinning_process_and_dt_evolution: Describes the engineering process involved in the Digital Twin (DT) implementation, including the development process, quality assurance, and definition of requirements. It also informs on the milestones of the Digital Twin (DT) engineering process over time and intended upgrades. 
        insights_and_decision_making: Defines the insights and decision making, i.e., indirect outputs of the Digital Twin (DT), which have no direct effect on the Physical Twin (PT) such as update of parameters, plans, and so on.
        standardization: Refers to the standards being followed for the engineering of the Digital Twin (DT) and its components.
       
 """

    print("Extracting Block 5 characteristics...")
    output = generate_desc(block5_description, retrieved_docs, Block5Characteristics)

    # merge the output with the previous characteristics
    if "extracted_characteristics" in state:
        existing_characteristics = state["extracted_characteristics"]
        existing_characteristics.update(output.model_dump(exclude_none=True))
        return {"extracted_characteristics": existing_characteristics}

    return {"extracted_characteristics": output.model_dump(exclude_none=True)}


def extractor_block6(state: State) -> State:

    """Extract characterrstics from the fifth block of the Digital Twin description."""

    if "vectordb" not in state:
        raise ValueError("Vector database must be provided.")
    
    block6_retrieval_query= """
        Data ownership and data privacy, data cybersecurity and safety on operation.
    """
    
    print("Retrieving documents for Block 6 characteristics...")
    retrieved_docs = state["vectordb"].similarity_search(block6_retrieval_query, k=3)  

    block6_description = """
        data_ownership_and_privacy: Refers to the ethical and technical aspects regarding data ownership and data privacy. Is the data owned by the PT owner or by the DT service provider?"
        security_and_safety_considerations: Refers to the ethical and technical aspects regarding data cybersecurity and safety on operation. Can a DT execute operations remotely on a PT where there may be accidents with humans?"
 """

    print("Extracting Block 6 characteristics...")
    output = generate_desc(block6_description, retrieved_docs, Block6Characteristics)

    # merge the output with the previous characteristics
    if "extracted_characteristics" in state:
        existing_characteristics = state["extracted_characteristics"]
        existing_characteristics.update(output.model_dump(exclude_none=True))
        return {"extracted_characteristics": existing_characteristics}

    return {"extracted_characteristics": output.model_dump(exclude_none=True)}



# FIXME: Too many instructions, simplify prompt
def generate_oml(state: State) -> State:
    """
    Generate OML description based on extracted characteristics.
    """
    if "extracted_characteristics" not in state:
        raise ValueError("Extracted characteristics must be provided.")
        
    characteristics = state["extracted_characteristics"]
    
    # Read the vocabulary for context
    dtdf_vocab = ""
    try:
        with open("data/oml/DTDF/vocab/DTDFVocab.oml", "r", encoding="utf-8") as f:
            dtdf_vocab = f.read()
    except FileNotFoundError:
        print("Warning: Could not find 'DTDFVocab.oml' file")
    
    base_oml = ""
    try:
        with open("data/oml/DTDF/vocab/base.oml", "r", encoding="utf-8") as f:
            base_oml = f.read()
    except FileNotFoundError:
        print("Warning: Could not find 'base.oml' file")

    baseDesc = ""
    try:
        with open("data/oml/DTDF/desc/baseDesc.oml", "r", encoding="utf-8") as f:
            baseDesc = f.read()
    except FileNotFoundError:
        print("Warning: Could not find 'baseDesc.oml' file")
    
    prompt = f"""
You are an expert in OML (Ontological Modeling Language) tasked with generating a complete OML description for a Digital Twin based on the following extracted characteristics:

EXTRACTED CHARACTERISTICS:
\n{characteristics}\n

You are also provided with a vocabulary reference to use in your OML description. The vocabulary is structured in a way that allows you to create instances and relationships between them, following the OML syntax.
VOCABULARY REFERENCE:

DTDFVocab.oml: ´´´\n{dtdf_vocab}\n´´´
---
base.oml:´´´\n{base_oml}\n´´´
--- 
baseDesc.oml:´´´\n{baseDesc}\n´´´
---

Only generate OML code that’s similar to the following syntax examples: 
// enabler (C11)
instance <name_of_enabler>: DTDFVocab:Enabler [
        DTDFVocab:enables <name_of_service_enabled_1>
        DTDFVocab:enables <name_of_service_enabled_2>
        DTDFVocab:enables <name_of_service_enabled_3>
]

// DT hosting/deployment (C16)
instance deployment : DTDFVocab:Deployment [
        base:desc "<Description of the DT hosting/deployment characteristic>"
]

// Models/Data (C10)
instance <name_of_model_or_data>: DTDFVocab:Data
    [
        DTDFVocab:inputTo <name_of_enabler_1>
        DTDFVocab:inputTo <name_of_enabler_2>
        DTDFVocab:fromData <name_of-physical_to_virtual_interaction>
    ]

IMPORTANT NOTE: Certain characteristics can have multiple instances: Physical acting components, Physical sensing components, Physical-to-virtual interaction, DT services, Twinning time-scale, DT models and data, Tooling and enablers, Insights and decision making.

---
INSTRUCTIONS:
- Generate a complete OML description following the structure shown in the example
- Generate only the OML code, no additional explanation.
"""
    
    response = llm.invoke(prompt)
    
    # Extract content from the response
    oml_content = response.content if hasattr(response, 'content') else str(response)
    
    return {"oml_output": oml_content}



# ---- Graph setup ----
graph = StateGraph(State)

# set nodes
graph.add_node("preprocess", preprocess)
graph.add_node("extractor_block1", extractor_block1)
graph.add_node("extractor_block2", extractor_block2)
graph.add_node("extractor_block3", extractor_block3)
graph.add_node("extractor_block4", extractor_block4)
graph.add_node("extractor_block5", extractor_block5)
graph.add_node("extractor_block6", extractor_block6)
graph.add_node("generate_oml", generate_oml)

# set edges
graph.set_entry_point("preprocess")
graph.add_edge("preprocess", "extractor_block1")
graph.add_edge("extractor_block1", "extractor_block2")
graph.add_edge("extractor_block2", "extractor_block3")
graph.add_edge("extractor_block3", "extractor_block4")
graph.add_edge("extractor_block4", "extractor_block5")
graph.add_edge("extractor_block5", "extractor_block6")
# graph.set_finish_point("extractor_block6")
graph.add_edge("extractor_block6", "generate_oml")
graph.set_finish_point("generate_oml")

workflow = graph.compile()


if __name__ == "__main__":
    # pdf_path = "data/case_studies/DT_book-276-289_incubator.pdf"
    pdf_path = "data/papers/The Incubator Case Study for Digital Twin Engineering.pdf"
    # pdf_path = "data/papers/Gil et al. - 2025 - Toward a systematic reporting framework for Digital Twins a cooperative robotics case study-1-12.pdf"
    

    # Delete ../vector_db if it already exists
    vector_db_path = os.path.join(os.path.dirname(__file__), "..", "vector_db")
    if os.path.exists(vector_db_path):
        shutil.rmtree(vector_db_path)

    # Run the workflow
    result = workflow.invoke({"pdf_path": pdf_path})

    print("\n Extracted characteristics:\n", result["extracted_characteristics"])

    df = pd.DataFrame(result["extracted_characteristics"], index=[1])
    df = df.transpose().reset_index()
    df.columns = ['Characteristic', 'Description']

    print(f"Generated output:\n{df.to_markdown(index=False)}")
    # Count how many extracted characteristics are 'Not Found'
    not_found_count = sum(
        1 for v in result["extracted_characteristics"].values() if v == "Not Found"
    )
    print(f"\nNumber of 'Not Found' characteristics: {not_found_count}")
    print("\n OML description generated:\n", result.get("oml_output", "Not generated"))


