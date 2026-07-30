"""
Verification Script for Milestone 1.

This script tests the document loading and chunking modules by loading 
the resumes in the `resumes/` directory, splitting them into chunks, 
and printing the outputs using the Rich library for clean visual presentation.
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Ensure that the src directory is in the python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import settings
from src.ingestion.loader import LoaderFactory
from src.ingestion.chunker import RecursiveChunker

console = Console()


def verify_loader_and_chunker():
    """Load sample resumes and run chunking verification."""
    console.print(Panel.fit(
        "[bold blue]Milestone 1 Verification Script[/bold blue]\n"
        "Testing: Configuration, Document Loading, and Chunking",
        border_style="blue"
    ))
    
    # 1. Print Configuration Settings
    config_table = Table(title="Configuration Parameters", style="cyan")
    config_table.add_column("Setting", style="bold")
    config_table.add_column("Value")
    config_table.add_row("Chunk Size", str(settings.chunk_size))
    config_table.add_row("Chunk Overlap", str(settings.chunk_overlap))
    config_table.add_row("Default LLM Model", settings.gemini_model)
    config_table.add_row("Embedding Model", settings.embedding_model)
    console.print(config_table)
    console.print()
    
    # 2. Check resumes folder
    resumes_dir = PROJECT_ROOT / "resumes"
    if not resumes_dir.exists() or not any(resumes_dir.iterdir()):
        console.print("[bold red]Error: No resumes found in the resumes/ directory.[/bold red]")
        return
        
    resume_files = list(resumes_dir.glob("*"))
    console.print(f"[bold green]Found {len(resume_files)} resumes to ingest:[/bold green]")
    
    chunker = RecursiveChunker()
    
    # Process each resume
    for file_path in resume_files:
        if file_path.suffix.lower() not in [".pdf", ".txt", ".md"]:
            console.print(f"[yellow]Skipping unsupported file: {file_path.name}[/yellow]")
            continue
            
        console.print(Panel(
            f"[bold magenta]Processing File:[/bold magenta] {file_path.name}\n"
            f"[bold magenta]Type:[/bold magenta] {file_path.suffix.upper()}",
            expand=False
        ))
        
        try:
            # Load document
            console.print("  [dim]Loading document...[/dim]")
            loader = LoaderFactory.get_loader(file_path)
            documents = loader.load(file_path)
            
            console.print(f"  [green]Success![/green] Loaded [bold]{len(documents)}[/bold] document page(s)/segment(s).")
            
            # Print document info
            for idx, doc in enumerate(documents):
                console.print(f"    Page/Segment {idx+1} characters: {len(doc.page_content)}")
                console.print(f"    Metadata: {doc.metadata}")
            
            # Split document
            console.print("  [dim]Chunking document...[/dim]")
            chunks = chunker.split_documents(documents)
            console.print(f"  [green]Success![/green] Generated [bold]{len(chunks)}[/bold] chunks.")
            
            # Print preview of the first chunk
            if chunks:
                first_chunk = chunks[0]
                chunk_panel = Panel(
                    first_chunk.page_content,
                    title=f"[bold]First Chunk Preview (Index: {first_chunk.metadata.get('chunk_index')})[/bold]",
                    border_style="yellow",
                    subtitle=f"Length: {len(first_chunk.page_content)} characters | Start Index: {first_chunk.metadata.get('start_index')}"
                )
                console.print(chunk_panel)
                
        except Exception as e:
            console.print(f"  [bold red]Failed to process {file_path.name}: {str(e)}[/bold red]", style="red")
            import traceback
            console.print(traceback.format_exc(), style="dim red")
            
        console.print("-" * 50)
        
    console.print("[bold green]Milestone 1 Verification completed successfully.[/bold green]")


if __name__ == "__main__":
    verify_loader_and_chunker()
