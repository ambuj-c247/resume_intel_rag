"""
Document Loading Abstractions and Implementations.

This module provides a unified interface for loading different resume file formats
(PDF, TXT, MD) and converting them into LangChain Document objects.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Type

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


class BaseLoader(ABC):
    """
    Abstract Base Class for resume loaders.
    
    Adheres to the Dependency Inversion Principle (SOLID) so that the core
    ingestion system depends on abstractions rather than concrete file parser libraries.
    """

    @abstractmethod
    def load(self, file_path: Path) -> List[Document]:
        """
        Load a document file and parse it into a list of LangChain Document objects.

        Args:
            file_path: Path to the document file.

        Returns:
            A list of Document objects with extracted text and metadata.
        """
        pass


class PDFLoader(BaseLoader):
    """Loads and parses PDF resumes using the pypdf library."""

    def load(self, file_path: Path) -> List[Document]:
        # PyPDFLoader parses page-by-page, preserving page numbers in metadata.
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        
        # Clean text content to remove duplicate/excessive whitespaces
        for doc in documents:
            doc.page_content = self._clean_text(doc.page_content)
            doc.metadata["file_type"] = "pdf"
            doc.metadata["source"] = str(file_path)
            
        return documents

    def _clean_text(self, text: str) -> str:
        """Helper to clean extracted text for better embedding quality."""
        # Replace multiple spaces and newlines with a single space/newline
        lines = [line.strip() for line in text.splitlines()]
        cleaned = "\n".join([line for line in lines if line])
        return cleaned


class TXTLoader(BaseLoader):
    """Loads and parses plain text resumes."""

    def load(self, file_path: Path) -> List[Document]:
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()
        
        for doc in documents:
            doc.metadata["file_type"] = "txt"
            doc.metadata["source"] = str(file_path)
            
        return documents


class MarkdownLoader(BaseLoader):
    """Loads and parses Markdown resumes."""

    def load(self, file_path: Path) -> List[Document]:
        # Markdown is a text-based format. We load it as text and preserve its
        # layout (which contains headers that are useful for chunking).
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()
        
        for doc in documents:
            doc.metadata["file_type"] = "markdown"
            doc.metadata["source"] = str(file_path)
            
        return documents


class LoaderFactory:
    """
    Factory class to dynamically instantiate the correct loader based on file extension.
    
    Follows the Open-Closed Principle (SOLID) - support for new file extensions
    can be added by registering new loaders without changing the calling code.
    """

    _registry: Dict[str, Type[BaseLoader]] = {
        ".pdf": PDFLoader,
        ".txt": TXTLoader,
        ".md": MarkdownLoader,
    }

    @classmethod
    def get_loader(cls, file_path: Path) -> BaseLoader:
        """
        Resolve and return the appropriate loader for the given file extension.

        Args:
            file_path: Path to the target file.

        Returns:
            An instance of BaseLoader.
            
        Raises:
            ValueError: If the file extension is not supported.
        """
        suffix = file_path.suffix.lower()
        loader_class = cls._registry.get(suffix)
        
        if not loader_class:
            supported = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unsupported file format '{suffix}' for file: {file_path.name}. "
                f"Supported formats: {supported}"
            )
            
        return loader_class()
