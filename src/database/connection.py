"""
Database Connection Manager.

This module configures the SQLAlchemy engine and provides sessions for 
database operations using the modern SQLAlchemy 2.0 context manager pattern.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.config import settings

# Create engine with psycopg3 driver.
# Why configuration-driven connections matter:
# Keeping connection details in configuration allows us to easily point to local dev, 
# Docker containers, staging, or production databases without modifying the codebase.
# echo=False prevents spamming SQL commands in output, but can be set to True for debugging.
engine = create_engine(
    settings.database_url,
    echo=False,
    future=True
)

# SessionLocal is our factory for creating database sessions
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True
)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Yields:
        An active SQLAlchemy Session.
        
    Ensures that the session is closed when the context exits, even if 
    an exception is raised during database operations.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
