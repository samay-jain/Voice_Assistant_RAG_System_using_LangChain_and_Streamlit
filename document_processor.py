from typing import List
import os
import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS
import faiss  # Facebook AI Similarity Search library

class DocumentProcessor:
    """
    DocumentProcessor handles the ingestion of raw documents from disk, 
    splits them into smaller chunks, generates embeddings for those chunks,
    and stores/retrieves them using a FAISS vector store.
    """

    def __init__(self):
        """
        Initializes the processor with a recursive text splitter for chunking
        and Ollama embeddings (via a local Ollama server) for vector generation.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,                  # Target max size of each chunk
            chunk_overlap=200,                # Amount of overlap between chunks
            separators=["\n\n", "\n", ". ", " ", ""]  # Order of splitting priority
        )

        # Embedding model via Ollama server (must be running locally)
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )

    def load_documents(self, directory: str) -> List[Document]:
        """
        Loads .pdf, .txt, and .md files from the specified directory using 
        LangChain's document loaders.

        Args:
            directory (str): Path to the folder containing documents.

        Returns:
            List[Document]: A list of LangChain Document objects.
        """
        # Configure supported loaders by file extension
        loaders = {
            ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        }

        documents = []

        # Try loading each supported file type
        for file_type, loader in loaders.items():
            try:
                loaded = loader.load()
                documents.extend(loaded)
                print(f"Loaded {len(loaded)} {file_type} documents")
            except Exception as e:
                print(f"Error loading {file_type} documents: {str(e)}")

        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits each document into smaller overlapping chunks to improve embedding
        quality and support better semantic retrieval.

        Args:
            documents (List[Document]): Raw documents.

        Returns:
            List[Document]: Chunked documents ready for embedding.
        """
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document], persist_directory: str) -> FAISS:
        """
        Creates or loads a FAISS vector store using the chunked documents and
        associated embeddings. Stores both the FAISS index and its metadata.

        Args:
            documents (List[Document]): Pre-processed document chunks.
            persist_directory (str): Where to save/load the FAISS index and metadata.

        Returns:
            FAISS: Vector store that can be used for semantic search.
        """
        index_path = os.path.join(persist_directory, "index.faiss")

        if os.path.exists(index_path):
            # Load previously saved vector store
            print(f"Loading existing FAISS vector store from {persist_directory}")
            return FAISS.load_local(
                persist_directory, 
                self.embeddings, 
                allow_dangerous_deserialization=True  # Required to unpickle safely
            )
        else:
            # Create new vector store from documents
            print(f"Creating new FAISS vector store in {persist_directory}")
            os.makedirs(persist_directory, exist_ok=True)

            vector_store = FAISS.from_documents(documents, embedding=self.embeddings)
            vector_store.save_local(persist_directory)  # Persist index to disk

            return vector_store
