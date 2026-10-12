from ollama_usage.OllamaAgent import Models, OllamaAgent
from ollama_usage.OllamaAgent import Models
from langchain_core.tools import create_retriever_tool
from RAG_usage.OllamaEmbed import create_storer, was_index_saved
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from RAG_usage.OllamaEmbed import create_retriever
from pathlib import Path
from RAG_usage.DocumentsFromThesis import create_documents
from RAG_usage.DocumentsFromThesis import create_chunks
from langchain_core.documents import Document

## PLACE YOUR PDF IN THE documents FOLDER
PDF_PATH: Path = Path("documents/main.pdf")

PDF_SYS: str = """
    You are helpful assistant specialized in reading academic thesis for knowledge purpose.
    The user will require informations about arguments in the thesis.
    First call the pdf_search tool to get the needed informations and then answer based on them.
    You can use the pdf_search tool multiple times if the retrieved informations are not pertinent
    to the question.
    DO NOT use external knowledge for answer.
    ONLY answer based on the obtained informations.
"""

def askRAG(prompt: str, model: Models = Models.LLAMA) -> str:
    """
        Ask a question to a RAG system. It will create the retriever, chunks and will provide 
        the output
    """
    retriever: VectorStoreRetriever = obtain_retriever()

    retriever_tool = create_retriever_tool(retriever, name = "pdf_search", \
        description="Search the information from the thesis")
    
    agent = OllamaAgent(model, PDF_SYS, [retriever_tool])

    return agent.askLLM(prompt=prompt)


def obtain_retriever() -> VectorStoreRetriever:
    """
        Create the retriever. Does NOT recreate chunks if they have been saved already
    """
    chunks: list[Document] = []
    if not was_index_saved():
        chunks = create_chunks(create_documents(PDF_PATH))
    index: FAISS = create_storer(chunks)
    retriever: VectorStoreRetriever = create_retriever(index=index)
    return retriever