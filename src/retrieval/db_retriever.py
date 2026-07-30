"""
Database Vector Retriever.

Implements semantic similarity search in PostgreSQL using the pgvector 
extension.
"""

from typing import List, Optional

from langchain_core.documents import Document
from sqlalchemy import select

from src.config.config import settings
from src.database.connection import get_db
from src.database.models import DocumentChunk
from src.embeddings.base import BaseEmbeddings
from src.retrieval.base import BaseRetriever


class PostgresVectorRetriever(BaseRetriever):
    """
    Semantic search retriever using PostgreSQL and pgvector.
    
    Future-ready architecture allows inheriting from this class or adding 
    strategies like Hybrid Search, MMR, or Parent Document Retrieval.
    """

    def __init__(self, embeddings_service: BaseEmbeddings) -> None:
        """
        Initialize with the injected embeddings service.

        Args:
            embeddings_service: Service to generate query embeddings.
        """
        self.embeddings_service = embeddings_service

    def retrieve(
        self, 
        query: str, 
        resume_id: Optional[int] = None, 
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve chunks using pgvector Cosine similarity.
        
        How Vector Similarity Search Works:
        1. Embed the search query into a 768-dimensional vector.
        2. Query the PostgreSQL database.
        3. Use the pgvector cosine distance operator `<=>` to compute the cosine distance 
           between the query vector and all stored chunk vectors.
        4. Sort by distance ascending (closest direction first) and limit to Top K.

        Args:
            query: User's question or search terms.
            resume_id: Optional filter to restrict search to a single resume.
            top_k: Number of chunks to return.

        Returns:
            List of LangChain Document objects.
        """
        top_k = top_k if top_k is not None else settings.top_k
        
        # Step 1: Embed query
        query_vector = self.embeddings_service.embed_query(query)
        
        # Step 2: Query PostgreSQL
        with get_db() as db:
            # We select the chunk content and its metadata
            stmt = select(DocumentChunk)
            
            # Apply resume filter if specified
            if resume_id is not None:
                stmt = stmt.filter(DocumentChunk.resume_id == resume_id)
                
            # Order by Cosine Distance (<=> in SQL, mapped to cosine_distance in SQLAlchemy)
            stmt = stmt.order_by(DocumentChunk.embedding.cosine_distance(query_vector))
            stmt = stmt.limit(top_k)
            
            chunk_records = db.scalars(stmt).all()
            
            # Step 3: Map to LangChain Document structure
            documents = []
            for chunk in chunk_records:
                metadata = chunk.meta_info.copy() if chunk.meta_info else {}
                metadata["resume_id"] = chunk.resume_id
                metadata["chunk_index"] = chunk.chunk_index
                
                documents.append(Document(
                    page_content=chunk.content,
                    metadata=metadata
                ))
                
            return documents

    # --- Future Retrieval Strategies (Architecture Placeholders) ---

    def retrieve_hybrid(self, query: str, resume_id: Optional[int] = None, top_k: int = 5) -> List[Document]:
        """
        Future Extension: Hybrid Search.
        
        Combines keyword search (BM25 or PostgreSQL Full-Text Search tsvector) with 
        semantic search. Merges results using Reciprocal Rank Fusion (RRF).
        """
        # Placeholder for future implementation
        raise NotImplementedError("Hybrid search will be supported in future versions.")

    def retrieve_mmr(self, query: str, resume_id: Optional[int] = None, top_k: int = 5, lambda_mult: float = 0.5) -> List[Document]:
        """
        Future Extension: Maximal Marginal Relevance (MMR).
        
        Retrieves a larger set of candidates, then reranks them to balance relevancy 
        against diversity, preventing the LLM from receiving repetitive information.
        """
        # Placeholder for future implementation
        raise NotImplementedError("Maximal Marginal Relevance (MMR) will be supported in future versions.")

    def retrieve_parent_document(self, query: str, resume_id: Optional[int] = None, top_k: int = 5) -> List[Document]:
        """
        Future Extension: Parent Document Retrieval.
        
        Embeds small chunks to optimize vector similarity searches, but returns the larger
        "parent" section (or full document) to the LLM to provide richer context for generation.
        """
        # Placeholder for future implementation
        raise NotImplementedError("Parent Document Retrieval will be supported in future versions.")
