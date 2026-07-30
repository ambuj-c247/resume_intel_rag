"""
Interactive Terminal Chat CLI.

Allows users to select an ingested resume and interactively query it, 
leveraging RAG and Google Gemini 2.5 Flash. Includes shortcut commands
for specialized Resume Intelligence features and logs query vector embeddings.
"""

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# Ensure the root directory is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import settings
from src.database.connection import get_db
from src.database.models import Resume, QueryLog
from src.embeddings.gemini import GeminiEmbeddings
from src.retrieval.db_retriever import PostgresVectorRetriever
from src.llm.base import BaseLLM
from src.llm.factory import get_llm
from src.prompts import templates

app = typer.Typer(help="Resume Intelligence AI Chat interface.")
console = Console()


def get_all_resumes() -> List[Resume]:
    """Fetch list of all ingested resumes from the database."""
    with get_db() as db:
        return db.query(Resume).order_by(Resume.id).all()


def display_pipeline_step(step_name: str, details: Optional[str] = None):
    """Utility to print the current stage of the RAG pipeline."""
    console.print(f"[bold yellow]↓ {step_name}[/bold yellow]")
    if details:
        console.print(Panel(details, border_style="dim", expand=False))


def log_query_execution(
    embeddings: GeminiEmbeddings,
    resume_id: int,
    query_text: str,
    response: str,
    retrieved_docs: List
):
    """
    Log the user query, generated query embedding, and LLM response 
    to the query_logs table in PostgreSQL.
    """
    try:
        # Step 1: Embed the search query (matching the exact embedding logic)
        query_vector = embeddings.embed_query(query_text)
        
        # Step 2: Save log details to PostgreSQL
        with get_db() as db:
            log_record = QueryLog(
                resume_id=resume_id,
                query_text=query_text,
                query_embedding=query_vector,
                generated_response=response,
                meta_info={
                    "model": settings.gemini_model,
                    "retrieved_chunks_count": len(retrieved_docs),
                    "chunk_indices": [doc.metadata.get("chunk_index") for doc in retrieved_docs]
                }
            )
            db.add(log_record)
            db.commit()
    except Exception as e:
        console.print(f"[dim red]Warning: Failed to save query log: {e}[/dim red]")


def handle_feature_shortcut(
    embeddings: GeminiEmbeddings,
    retriever: PostgresVectorRetriever,
    llm: BaseLLM,
    resume_id: int,
    feature_name: str,
    prompt_instruction: str
):
    """Handles running a specialized intelligence feature prompt."""
    query = f"Provide information regarding the candidate's {feature_name}."
    
    # 1. Similarity Search
    display_pipeline_step("Similarity Search", f"Retrieving context for feature: {feature_name}")
    retrieved_docs = retriever.retrieve(query, resume_id=resume_id, top_k=settings.top_k)
    
    # 2. Retrieved Context
    context_text = "\n\n".join([f"--- Chunk {i+1} ---\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)])
    display_pipeline_step("Retrieved Context", f"Retrieved {len(retrieved_docs)} chunks (total length: {len(context_text)} chars)")
    
    # 3. Prompt Construction
    full_prompt = f"{prompt_instruction}\n\nContext from resume:\n{context_text}"
    display_pipeline_step("Prompt Construction", "Combined template instructions with retrieved context")
    
    # 4. Gemini Response
    display_pipeline_step("Gemini Response", "Invoking gemini-2.5-flash...")
    with console.status("[bold green]Analyzing resume...", spinner="dots"):
        response = llm.generate(
            prompt=full_prompt,
            system_instruction=templates.RAG_SYSTEM_INSTRUCTION
        )
        
    display_pipeline_step("Final Answer")
    console.print(Panel(Markdown(response), title=f"[bold green]Resume {feature_name.capitalize()}[/bold green]", border_style="green"))
    
    # Log query and query embedding
    log_query_execution(embeddings, resume_id, query, response, retrieved_docs)


def handle_job_matching(
    embeddings: GeminiEmbeddings,
    retriever: PostgresVectorRetriever,
    llm: BaseLLM,
    resume_id: int
):
    """Interactive job matching interface."""
    console.print("\n[bold cyan]Paste target Job Description (Press Enter, then Ctrl-D or Ctrl-Z to submit):[/bold cyan]")
    jd_lines = []
    try:
        while True:
            line = input()
            jd_lines.append(line)
    except EOFError:
        pass
        
    job_description = "\n".join(jd_lines).strip()
    if not job_description:
        console.print("[bold red]Job description cannot be empty.[/bold red]")
        return
        
    # We query the retriever using the key skills listed in the JD
    display_pipeline_step("Similarity Search", "Retrieving context based on Job Description requirements")
    retrieved_docs = retriever.retrieve(job_description[:1000], resume_id=resume_id, top_k=settings.top_k)
    
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    display_pipeline_step("Retrieved Context", f"Retrieved {len(retrieved_docs)} chunks matching JD")
    
    # Construct prompt
    prompt = templates.MATCH_PROMPT_TEMPLATE.format(
        job_description=job_description,
        context=context_text
    )
    display_pipeline_step("Prompt Construction")
    
    display_pipeline_step("Gemini Response", "Evaluating job fit...")
    with console.status("[bold green]Comparing resume with Job Description...", spinner="dots"):
        response = llm.generate(
            prompt=prompt,
            system_instruction=templates.RAG_SYSTEM_INSTRUCTION
        )
        
    display_pipeline_step("Final Answer")
    console.print(Panel(Markdown(response), title="[bold green]Job Match Report[/bold green]", border_style="green"))
    
    # Log query and query embedding
    log_query_execution(embeddings, resume_id, "Compare resume against Job Description", response, retrieved_docs)


@app.command()
def chat():
    """Start interactive terminal chat session."""
    console.print(Panel.fit(
        "[bold green]Welcome to Resume Intelligence AI Chat[/bold green]\n"
        "Ask questions about resumes or use shortcuts to extract intelligence.",
        border_style="green"
    ))
    
    # Ensure database is accessible and has resumes
    try:
        resumes = get_all_resumes()
    except Exception as e:
        console.print(f"[bold red]Database Connection Failed:[/bold red] {e}")
        console.print("[yellow]Please ensure your PostgreSQL container is running on port 5454.[/yellow]")
        raise typer.Exit(1)
        
    if not resumes:
        console.print("[bold yellow]No resumes found in the database.[/bold yellow]")
        console.print("Please ingest resumes first by running:")
        console.print("  [bold cyan]python ingest.py[/bold cyan]")
        raise typer.Exit()
        
    # Present resumes for selection
    console.print("\n[bold]Select a resume to chat about:[/bold]")
    for idx, r in enumerate(resumes):
        console.print(f"  [bold cyan][{idx + 1}][/bold cyan] {r.filename} (Ingested: {r.uploaded_at.strftime('%Y-%m-%d %H:%M')})")
        
    selection = Prompt.ask("\nEnter selection number", choices=[str(i+1) for i in range(len(resumes))])
    selected_resume = resumes[int(selection) - 1]
    
    console.print(Panel(
        f"Active Chat Resume: [bold green]{selected_resume.filename}[/bold green]\n"
        f"Type your questions naturally. Type [bold yellow]/help[/bold yellow] to see available shortcuts.\n"
        f"Type [bold red]/exit[/bold red] to end session.",
        border_style="cyan"
    ))
    
    # Initialize RAG components
    embeddings = GeminiEmbeddings()
    retriever = PostgresVectorRetriever(embeddings_service=embeddings)
    llm = get_llm("gemini")
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold green]User[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nExiting chat session.")
            break
            
        if not user_input:
            continue
            
        # Handle Exit
        if user_input in ["/exit", "/quit"]:
            console.print("[bold red]Goodbye![/bold red]")
            break
            
        # Handle Help
        if user_input == "/help":
            help_text = (
                "**Available Commands:**\n"
                "- `/help` : Show this help message\n"
                "- `/exit` : Close the chat session\n"
                "- `/summary` : Summarize the candidate's background\n"
                "- `/skills` : Extract technical skills\n"
                "- `/experience` : Extract work experience\n"
                "- `/projects` : Extract details of project work\n"
                "- `/education` : Extract academic qualifications\n"
                "- `/certifications` : Extract training certificates\n"
                "- `/strengths` : Analyze candidate strengths\n"
                "- `/weaknesses` : Analyze profile gaps & weaknesses\n"
                "- `/questions` : Generate tailored interview questions\n"
                "- `/match` : Perform Job Description matching\n"
            )
            console.print(Panel(Markdown(help_text), title="Help Menu", border_style="yellow"))
            continue
            
        # Handle shortcuts
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/summary":
                handle_feature_shortcut(embeddings, retriever, llm, selected_resume.id, "summary", templates.SUMMARY_PROMPT)
            elif cmd == "/skills":
                handle_feature_shortcut(embeddings, retriever, llm, selected_resume.id, "skills", templates.SKILLS_PROMPT)
            elif cmd == "/projects":
                handle_feature_shortcut(embeddings, retriever, llm, selected_resume.id, "projects", templates.PROJECTS_PROMPT)
            elif cmd == "/education":
                handle_feature_shortcut(embeddings, retriever, llm, selected_resume.id, "education", templates.EDUCATION_PROMPT)
            elif cmd == "/experience":
                handle_feature_shortcut(embeddings, retriever, llm, selected_resume.id, "experience", templates.EXPERIENCE_PROMPT)
            elif cmd == "/certifications":
                handle_feature_shortcut(embeddings, retriever, llm, selected_resume.id, "certifications", templates.CERTIFICATIONS_PROMPT)
            elif cmd == "/strengths":
                handle_feature_shortcut(embeddings, retriever, llm, selected_resume.id, "strengths", templates.STRENGTHS_PROMPT)
            elif cmd == "/weaknesses":
                handle_feature_shortcut(embeddings, retriever, llm, selected_resume.id, "weaknesses", templates.WEAKNESSES_PROMPT)
            elif cmd == "/questions":
                handle_feature_shortcut(embeddings, retriever, llm, selected_resume.id, "interview questions", templates.INTERVIEW_QUESTIONS_PROMPT)
            elif cmd == "/match":
                handle_job_matching(embeddings, retriever, llm, selected_resume.id)
            else:
                console.print(f"[bold red]Unknown command: {user_input}[/bold red]. Type `/help` for options.")
            continue

        # Standard RAG Query
        # 1. Similarity Search
        display_pipeline_step("Similarity Search", f"Searching chunks matching query: '{user_input}'")
        retrieved_docs = retriever.retrieve(user_input, resume_id=selected_resume.id, top_k=settings.top_k)
        
        # 2. Retrieved Context
        context_text = "\n\n".join([f"--- Chunk {i+1} ---\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)])
        display_pipeline_step("Retrieved Context", f"Retrieved {len(retrieved_docs)} chunks")
        
        # 3. Prompt Construction
        prompt = templates.RAG_PROMPT_TEMPLATE.format(context=context_text, question=user_input)
        display_pipeline_step("Prompt Construction")
        
        # 4. Gemini Response
        display_pipeline_step("Gemini Response", "Generating grounded answer...")
        with console.status("[bold green]Thinking...", spinner="dots"):
            try:
                response = llm.generate(
                    prompt=prompt,
                    system_instruction=templates.RAG_SYSTEM_INSTRUCTION
                )
            except Exception as e:
                console.print(f"[bold red]Gemini generation failed:[/bold red] {e}")
                continue
                
        display_pipeline_step("Final Answer")
        console.print(Panel(Markdown(response), title="[bold green]Assistant[/bold green]", border_style="green"))
        
        # Log standard query
        log_query_execution(embeddings, selected_resume.id, user_input, response, retrieved_docs)


if __name__ == "__main__":
    app()
