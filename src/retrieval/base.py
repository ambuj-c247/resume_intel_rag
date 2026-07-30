"""
Retrieval Abstraction Interface.

Defines the interface for context retrievers. This enables pluggable search 
mechanisms, making it easy to swap simple vector search with advanced search 
strategies (Hybrid Search, MMR, Parent Document Retrieval, etc.).
"""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseRetriever(ABC):
    """
    Abstract Base Class for context retrieval.
    
    Provides an interface to retrieve relevant resume context chunks based on 
    a natural language query.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Search the database and return the most relevant document chunks.

        Args:
            query: The user search query.
            top_k: Number of relevant chunks to retrieve. Defaults to settings.top_k.

        Returns:
            A list of LangChain Document objects containing the text and metadata.
        """
        pass
