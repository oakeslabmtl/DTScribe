# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
# from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from typing import Type
from pydantic import BaseModel


from langchain.text_splitter import MarkdownTextSplitter
import pymupdf4llm



# --- LLM setup ---
llm = ChatOllama(
    model="qwen3:8b", 
    # model="qwen3:4b", 
    temperature=0.1, 
    top_p=0.9,
)


# --- Document loader and text splitter ---
def load_and_split_pdf(pdf_path: str):
    """
    Load a PDF, split it into chunks, and store them in a vector database.
    """

    # loader = PyMuPDFLoader(pdf_path)
    # documents = loader.load()
    # print(documents)

    # splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=90)
    # chunks = splitter.split_documents(documents)

    md_text = pymupdf4llm.to_markdown(pdf_path, pages=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    splitter = MarkdownTextSplitter(chunk_size=2500, chunk_overlap=100)
    docs = splitter.create_documents([md_text])

    # print(f"Loaded {len(documents)} documents and split them into {len(chunks)} chunks.")

    vectordb = Chroma.from_documents(
            docs,
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
- Provide a very detailed and factual description for each characteristic for the specific use case.
- Use ONLY the provided information to fill the content for each characteristic.
- Be VERY specific of the tecnologies mentioned and assume you are presenting it to someone without any context of the use case.
- State the facts without making any specific references to the documents.
- Insert "Not Found" into the field if no evidence is found in the text.
        """

    output = llm.with_structured_output(schema).invoke(prompt)
    # output = llm.invoke(prompt)
    return output
