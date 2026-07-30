"""
Document Chunking and Splitting Module.

This module exposes abstractions for splitting full documents into smaller chunks
optimized for semantic search.
"""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.config import settings


class BaseChunker(ABC):
    """
    Abstract Base Class for text chunkers.
    
    Provides an interface for splitting larger documents into smaller pieces
    suitable for generating embeddings and retrieving context.
    """

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Document objects into a list of chunked Document objects.

        Args:
            documents: List of input Document objects.

        Returns:
            A list of smaller, chunked Document objects.
        """
        pass


class RecursiveChunker(BaseChunker):
    """
    Splits documents recursively using RecursiveCharacterTextSplitter.
    
    Why Chunking Exists:
    - LLMs have limit boundaries (context windows). We cannot pass hundreds of pages
      of text to the LLM prompt without hitting limits, high latency, and increased costs.
    - Large text segments dilute semantic meaning. An embedding represents the average
      semantic vector of the text. Generating one vector for a 10-page document misses
      fine-grained details (like a single specific skill or project).
    - Chunking splits the text into small, self-contained paragraphs/sections, allowing
      us to create high-precision vector embeddings for each detail.
      
    Why Chunk Overlap Exists:
    - Document splitters split text at boundaries. If a critical detail (like "Developed
      RAG pipelines using LangChain") gets split exactly in half (e.g., "Developed RAG"
      in Chunk 1 and "pipelines using LangChain" in Chunk 2), both chunks lose the full context.
    - Overlap ensures that text at boundaries is duplicated across consecutive chunks,
      preserving local context and transitions between sentences.
      
    Parameters:
    - chunk_size: Target character length of each chunk.
    - chunk_overlap: Number of characters to overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the RecursiveChunker with configuration values.

        Args:
            chunk_size: Optional custom chunk size. Defaults to config settings.
            chunk_overlap: Optional custom chunk overlap. Defaults to config settings.
        """
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        
        # RecursiveCharacterTextSplitter attempts to split by the first separator in the list
        # (e.g., paragraphs '\n\n'), then by sentences ('\n'), then by words (' '), falling back to
        # characters (''). This keeps semantically related sentences together.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,  # Tracks character offset where the chunk starts in the original document
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split the input documents into smaller chunks.

        Args:
            documents: List of input Document objects.

        Returns:
            A list of smaller, chunked Document objects.
        """
        chunks = self.splitter.split_documents(documents)
        
        # Add metadata for chunk index relative to original document
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            
        return chunks
