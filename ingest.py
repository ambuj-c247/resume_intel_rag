"""
Ingestion Pipeline Script.

Loads resumes from the resumes/ directory, parses, chunks, generates 
Google Gemini embeddings, and saves them to the PostgreSQL database using pgvector.
"""

import hashlib
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Ensure the root directory is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import settings
from src.database.connection import get_db, engine
from src.database.models import Base, Resume, DocumentChunk
from src.ingestion.loader import LoaderFactory
from src.ingestion.chunker import RecursiveChunker
from src.embeddings.gemini import GeminiEmbeddings

console = Console()


def compute_file_hash(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file's content to support idempotency."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_ingestion():
    """Main ingestion runner."""
    console.print("[bold green]Loading resumes...[/bold green]")
    
    resumes_dir = PROJECT_ROOT / "resumes"
    if not resumes_dir.exists():
        console.print("[bold red]Error: resumes/ directory does not exist![/bold red]")
        sys.exit(1)
        
    resume_files = [
        p for p in resumes_dir.iterdir() 
        if p.is_file() and p.suffix.lower() in [".pdf", ".txt", ".md"]
    ]
    
    if not resume_files:
        console.print("[yellow]No resumes (.pdf, .txt, .md) found in resumes/ directory.[/yellow]")
        return
        
    console.print(f"Found [bold]{len(resume_files)}[/bold] resumes to process.")
    
    # Initialize the embeddings generator (needs GEMINI_API_KEY)
    if not settings.gemini_api_key:
        console.print("[bold red]Error: GEMINI_API_KEY is not set in .env![/bold red]")
        console.print("[yellow]Please obtain a Gemini API key and set it before proceeding.[/yellow]")
        sys.exit(1)
        
    try:
        embeddings_service = GeminiEmbeddings()
    except Exception as e:
        console.print(f"[bold red]Failed to initialize Gemini Embeddings: {e}[/bold red]")
        sys.exit(1)

    chunker = RecursiveChunker()

    # Ensure tables exist.
    # Why this fallback exists:
    # Although Alembic handles schema migrations, executing create_all() acts as a safety valve.
    # If the user spins up the database container and runs ingest.py without applying migrations first,
    # create_all() automatically configures the database schema so the app runs out-of-the-box.
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        console.print(f"[bold red]Database connection failed: {e}[/bold red]")
        console.print("[yellow]Ensure that your PostgreSQL container is running using:[/yellow]")
        console.print("  [bold cyan]docker compose -f docker/docker-compose.yml up -d[/bold cyan]")
        sys.exit(1)

    with get_db() as db:
        for file_path in resume_files:
            console.print(f"\n[bold blue]Processing Resume:[/bold blue] {file_path.name}")
            
            file_hash = compute_file_hash(file_path)
            
            # Check if this resume was already ingested with the exact same content
            existing_resume = db.query(Resume).filter_by(file_hash=file_hash).first()
            if existing_resume:
                console.print(f"[green]Resume '{file_path.name}' is already up-to-date (matching hash). Skipping Ingestion.[/green]")
                continue
                
            # Check if a resume with the same filename exists, but has a different hash
            # in which case we delete the old record and overwrite it
            old_resume = db.query(Resume).filter_by(filename=file_path.name).first()
            if old_resume:
                console.print(f"[yellow]Hash mismatch for '{file_path.name}' (content changed). Deleting old chunks...[/yellow]")
                db.delete(old_resume)
                db.commit()

            try:
                # 1. Loading & Extracting
                console.print("Extracting text...")
                loader = LoaderFactory.get_loader(file_path)
                documents = loader.load(file_path)
                
                # 2. Chunking
                console.print("Chunking...")
                chunks = chunker.split_documents(documents)
                console.print(f"Generated {len(chunks)} chunks.")
                
                # 3. Embedding Generation
                console.print("Generating embeddings...")
                texts = [chunk.page_content for chunk in chunks]
                
                # Generate embeddings in batch for speed and cost efficiency
                embeddings = embeddings_service.embed_documents(texts)
                
                # 4. Saving to Database
                console.print("Saving to PostgreSQL...")
                
                resume_record = Resume(
                    filename=file_path.name,
                    file_path=str(file_path),
                    file_hash=file_hash,
                    meta_info={"file_size": file_path.stat().st_size}
                )
                db.add(resume_record)
                db.flush()  # Populates resume_record.id

                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_record = DocumentChunk(
                        resume_id=resume_record.id,
                        content=chunk.page_content,
                        embedding=embedding,
                        chunk_index=i,
                        start_index=chunk.metadata.get("start_index"),
                        meta_info=chunk.metadata
                    )
                    db.add(chunk_record)
                
                db.commit()
                console.print(f"[bold green]Successfully ingested {file_path.name}![/bold green]")
                
            except Exception as e:
                db.rollback()
                console.print(f"[bold red]Failed to ingest {file_path.name}: {e}[/bold red]")
                import traceback
                console.print(traceback.format_exc(), style="dim red")
                
    console.print("\n[bold green]Completed.[/bold green]")


if __name__ == "__main__":
    run_ingestion()
