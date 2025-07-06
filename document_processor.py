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
import faiss  # FAISS library to handle saving/loading index

class DocumentProcessor:
    """
    DocumentProcessor handles loading documents from disk, splitting them into chunks,
    generating embeddings, and storing or retrieving them from a FAISS vector database.
    """

    def __init__(self):
        """
        Initialize the document processor with a recursive text splitter and Ollama embeddings.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,                  # Max characters per chunk
            chunk_overlap=200,                # Overlap between chunks to preserve context
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"  # Local Ollama server
        )

    def load_documents(self, directory: str) -> List[Document]:
        """
        Load documents from a given directory.
        Supports .pdf, .txt, and .md files.

        Args:
            directory (str): Path to the directory containing documents.

        Returns:
            List[Document]: List of LangChain Document objects.
        """
        loaders = {
            ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        }

        documents = []
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
        Split loaded documents into smaller chunks using a recursive strategy.

        Args:
            documents (List[Document]): List of loaded documents.

        Returns:
            List[Document]: List of split document chunks.
        """
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document], persist_directory: str) -> FAISS:
        """
        Create or load a FAISS vector store using document embeddings.

        Args:
            documents (List[Document]): List of chunked documents.
            persist_directory (str): Directory where FAISS index and metadata are stored.

        Returns:
            FAISS: A LangChain-compatible FAISS vector store.
        """
        if os.path.exists(os.path.join(persist_directory, "index.faiss")):
            # Load existing FAISS vector store using LangChain's built-in method
            print(f"Loading existing FAISS vector store from {persist_directory}")
            return FAISS.load_local(persist_directory, self.embeddings, allow_dangerous_deserialization=True)
        else:
            # Create new vector store
            print(f"Creating new FAISS vector store in {persist_directory}")
            os.makedirs(persist_directory, exist_ok=True)

            vector_store = FAISS.from_documents(documents, embedding=self.embeddings)
            vector_store.save_local(persist_directory)
            return vector_store

