"""
Embedding Abstraction Interface.

Defines the abstract interface for generating text embeddings.
This allows switching from Google Gemini to other providers (OpenAI, HuggingFace, etc.)
with zero modifications to the core ingestion and retrieval logic.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddings(ABC):
    """
    Abstract Base Class for generating vector embeddings of text.
    
    Why Embeddings are Needed:
    - Raw text cannot be compared mathematically by computers.
    - Embeddings translate words, sentences, or paragraphs into high-dimensional vectors 
      (lists of numbers) where semantically similar phrases are mapped closer together 
      in the vector space.
    - For example, "competence in writing Python scripts" and "proficient Python developer" 
      are syntactically different (different words) but semantically identical. Vector search 
      detects this similarity.
    """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single query string.

        Args:
            text: The text query.

        Returns:
            A list of floats representing the embedding vector.
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of document strings.

        Args:
            texts: List of document strings.

        Returns:
            A list of embedding vectors (each vector is a list of floats).
        """
        pass
