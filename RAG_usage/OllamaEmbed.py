from pathlib import Path
from typing import Optional
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

from langchain_ollama import OllamaEmbeddings, embeddings
from langchain_community.vectorstores import FAISS

MAX_RETRIEVED_CHUNKS: int = 4
INDEX_SAVE_LOCATION: str = "saved_index"

def get_embed() -> OllamaEmbeddings:
    """
        Obtain the embeddings. As standard, use BGE large from Ollama
    """
    embeddings = OllamaEmbeddings(model="bge-large")
    return embeddings

def create_storer(chunks: list[Document] = []) -> FAISS:
    """
        Create and return a FAISS index. If it was saved in the default path, it won't create a new one.
    """
    if was_index_saved():
        index: FAISS = FAISS.load_local(
            INDEX_SAVE_LOCATION,
            get_embed(),
            allow_dangerous_deserialization=True,
        )
    else:
        index: FAISS = FAISS.from_documents(chunks, get_embed())
        index.save_local(INDEX_SAVE_LOCATION)
    return index

def was_index_saved() -> bool:
    """
        Returns True if the FAISS index is saved in the default path
    """
    index_path = Path(INDEX_SAVE_LOCATION)
    return index_path.exists() and (index_path / "index.faiss").exists()

def create_retriever(index: FAISS) -> VectorStoreRetriever:
    """
        Create a retriever from a given FAISS index
    """
    retriever = index.as_retriever(search_kwargs={"k": MAX_RETRIEVED_CHUNKS})
    return retriever

