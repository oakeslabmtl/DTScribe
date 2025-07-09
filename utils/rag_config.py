# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
# from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from typing import Type
from pydantic import BaseModel


# --- LLM setup ---
llm = ChatOllama(
    model="llama3.2:latest", 
    temperature=0.1, 
    top_p=0.9,
)


# --- Document loader and text splitter ---
def load_and_split_pdf(pdf_path: str):
    """
    Load a PDF, split it into chunks, and store them in a vector database.
    """

    loader = PyMuPDFLoader(pdf_path)

    documents = loader.load()

    print(documents)

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=30)
    chunks = splitter.split_documents(documents)

    print(f"Loaded {len(documents)} documents and split them into {len(chunks)} chunks.")

    vectordb = Chroma.from_documents(
            chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory="./vector_db"
    )

    return vectordb


# For the prompt ⬇️
# NOTE: if not acceptable, provide example of the output format
# NOTE: we can tailor HOW detailed
# FIXME: force it to be even more detailed,

def generate_desc(description: str, retrieved_docs, schema: Type[BaseModel]) -> BaseModel:
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
                You are an ontology expert tasked with extracting detailed descriptions of the following Digital Twin (DT) characteristics:

                \n{description}\n

                Based only on the following documents:

                \n{docs_content}\n

                Remember to extract the following characteristics:
                \n{description}\n
                ---

                IMPORTANT:
                - Provide a detailed and factual description for each characteristic for the specific use case.
                - Use ONLY the provided information to infer the contents to be filled for each characteristic.
                - Be VERY specific of the tecnologies mentioned and assume you are presenting it to someone without any context of the use case.
                - Insert "Not Found" into the field if no evidence is found in the text.
                """

    output = llm.with_structured_output(schema).invoke(prompt)
    # output = llm.invoke(prompt)
    return output



################### NOTE: Compare with old generation function #########################

# def generate_desc(retrieved_docs, characteristics):

#     docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

#     query = f"""
#     You are an ontology expert tasked with extracting detailed descriptions of the characteristics 
#     of a Digital Twin (DT) based on provided documents.

#     Given the following information:\n\n{docs_content}\n\n

#     Your task is to provide detailed descriptions for each of the following characteristics 
#     of a Digital Twin: \n\n{characteristics}\n\n

#     IMPORTANT:
#     - Provide a detailed description for each characteristic.
#     - Use the information from the provided documents to inform your descriptions.
#     - ONLY include information that is directly related to the characteristics, nothing else.
#     """
    
#     response = llm.invoke({"query": query})

#     return response


################### NOTE: This is most likely trash: ###################################3

# def retrieve_documents(state, vectordb):

#     """
#     Retrieve documents from the vector database based on a query.
#     """
#     if "query" not in state:
#         raise ValueError("Query must be provided.")

    #  retrieved_docs = vectordb.similarity_search(state["query"], k=5)

#     return retrieved_docs