"""
Database Models.

Defines the SQLAlchemy ORM models for Resume and DocumentChunk records.
"""

from datetime import datetime
from typing import Any, Dict, List

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """
    Base class for SQLAlchemy ORM models.
    
    Use SQLAlchemy 2.0 DeclarativeBase where mapping is defined using 
    type annotations (Mapped[type]) and mapped_column().
    """
    pass


class Resume(Base):
    """
    ORM Model representing an ingested Resume file.
    
    Stores the general information of the resume file including its filename, 
    path, unique content hash (to avoid duplicate parsing), and generic metadata.
    """
    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    
    # Store SHA-256 hash of document content.
    # Why document hashing:
    # Instead of re-parsing, chunking, embedding, and uploading a document on every 
    # run, we check if the file hash matches what is already stored. If it does, 
    # we skip it, which reduces Gemini API embedding call costs and database overhead.
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    # meta_info contains optional fields parsed or specified (like author, tags, size, etc.)
    # We use PostgreSQL JSONB for fast indexing and query capabilities on unstructured fields.
    meta_info: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)

    # Establish cascade deletion so deleting a Resume automatically cleans up all its chunks.
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk", 
        back_populates="resume", 
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    def __repr__(self) -> str:
        return f"<Resume(id={self.id}, filename='{self.filename}', uploaded_at={self.uploaded_at})>"


class DocumentChunk(Base):
    """
    ORM Model representing a text chunk split from a Resume.
    
    Contains the text slice, the page number, the index, the start offset, 
    and the high-dimensional vector embedding.
    """
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    resume_id: Mapped[int] = mapped_column(
        ForeignKey("resumes.id", ondelete="CASCADE"), 
        nullable=False
    )
    
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Why store embeddings in pgvector Vector(3072):
    # Vector embeddings are 1D arrays of floating-point numbers mapping semantic meaning.
    # Google's embedding model outputs 3072-dimensional vectors. 
    # pgvector's Vector type maps this array natively to a binary format in PostgreSQL.
    # This enables high-performance indexing (e.g. HNSW/IVFFlat) and operations:
    # - '<=>' Cosine Distance: measures direction of vectors (best for text embeddings)
    # - '<->' L2/Euclidean Distance: measures straight-line distance
    # - '<#>' Negative Inner Product: useful for dot product operations
    embedding: Mapped[List[float]] = mapped_column(Vector(3072), nullable=False)
    
    # Metadata tracking where the chunk came from
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_index: Mapped[int] = mapped_column(Integer, nullable=True)
    meta_info: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)

    resume: Mapped["Resume"] = relationship("Resume", back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk(id={self.id}, resume_id={self.resume_id}, "
            f"chunk_index={self.chunk_index}, length={len(self.content)})>"
        )


class QueryLog(Base):
    """
    ORM Model representing a user Q&A log.
    
    Stores the query text, the query embedding generated for similarity search, 
    the final generated response, and related metadata (e.g. latency, model used).
    """
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    resume_id: Mapped[int] = mapped_column(
        ForeignKey("resumes.id", ondelete="SET NULL"), 
        nullable=True
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Store the query embedding vector (3072 dimensions)
    query_embedding: Mapped[List[float]] = mapped_column(Vector(3072), nullable=False)
    generated_response: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    meta_info: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)

    resume: Mapped["Resume"] = relationship("Resume")

    def __repr__(self) -> str:
        return f"<QueryLog(id={self.id}, query_text='{self.query_text[:30]}...', created_at={self.created_at})>"

