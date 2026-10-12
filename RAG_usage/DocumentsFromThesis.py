"""
    Extract documents and chunks from a given PDF
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader

SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]
CHUNKS_SIZE: int = 1000

def create_documents(path: Path) -> list[Document]:
    """
        Create the Document class from a given Path.
        If the path do not exist, raise an error to stop execution.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF not found at: {path.resolve()}")
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs

def create_chunks(documents: list[Document]) -> list[Document]:
    """
        Create chunks from given Documents.
    """
    text_splitter: RecursiveCharacterTextSplitter = \
            RecursiveCharacterTextSplitter(
                chunk_size=CHUNKS_SIZE,
                separators=SEPARATORS
            )
    
    chunks: list[Document] = text_splitter.split_documents(documents)
    return chunks


