"""
Google Gemini Embedding Implementation.

Concrete implementation of the BaseEmbeddings interface using the Google Gemini 
Embedding API (defaulting to models/gemini-embedding-001).
"""

from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config.config import settings
from src.embeddings.base import BaseEmbeddings


class GeminiEmbeddings(BaseEmbeddings):
    """
    Google Gemini API wrapper for embedding generation.
    
    Uses LangChain's GoogleGenerativeAIEmbeddings internally to handle HTTP calls 
    and rate limit backoffs.
    """

    def __init__(self) -> None:
        """Initialize the Gemini Embeddings client using settings config."""
        # Note: LangChain requires the models/ prefix in front of gemini-embedding-001
        model_name = settings.embedding_model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
            
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=settings.gemini_api_key
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding vector for a query using Google Gemini.

        Args:
            text: The user search query.

        Returns:
            768-dimensional float vector.
        """
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple chunks using Google Gemini.

        Args:
            texts: List of chunk texts.

        Returns:
            List of 768-dimensional float vectors.
        """
        return self._embeddings.embed_documents(texts)
