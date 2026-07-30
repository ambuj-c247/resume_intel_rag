"""
RAG Evaluation Runner.

Runs the evaluation pipeline over the QA dataset. Performs similarity 
search, generates LLM responses, evaluates using Ragas (Gemini judge), 
prints a Rich report, and saves results in evaluation/reports/.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Ensure the root directory is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import settings
from src.database.connection import get_db
from src.database.models import Resume
from src.embeddings.gemini import GeminiEmbeddings
from src.retrieval.db_retriever import PostgresVectorRetriever
from src.llm.factory import get_llm
from src.prompts import templates
from src.evaluation.evaluator import RagasEvaluator

console = Console()


def run_evaluation():
    """Execute Ragas evaluation on the ingested resume dataset."""
    console.print(Panel.fit(
        "[bold green]Resume Intelligence AI - Ragas Evaluator[/bold green]\n"
        "Measuring Faithfulness, Relevancy, Precision, and Recall using Google Gemini.",
        border_style="green"
    ))

    # 1. Check for API key
    if not settings.gemini_api_key:
        console.print("[bold red]Error: GEMINI_API_KEY is not set in .env![/bold red]")
        sys.exit(1)

    # 2. Check for database records
    with get_db() as db:
        alice_resume = db.query(Resume).filter(Resume.filename.ilike("%alice%")).first()
        if not alice_resume:
            console.print("[bold red]Error: 'alice_developer.pdf' is not found in the database.[/bold red]")
            console.print("Please ingest resumes first by running:")
            console.print("  [bold cyan]python ingest.py[/bold cyan]")
            sys.exit(1)
            
        resume_id = alice_resume.id
        filename = alice_resume.filename

    # 3. Load dataset
    dataset_path = PROJECT_ROOT / "evaluation" / "dataset.json"
    if not dataset_path.exists():
        console.print(f"[bold red]Error: Evaluation dataset not found at {dataset_path}![/bold red]")
        sys.exit(1)
        
    with open(dataset_path, "r") as f:
        qa_pairs = json.load(f)
        
    console.print(f"Loaded [bold]{len(qa_pairs)}[/bold] evaluation QA pairs for [cyan]{filename}[/cyan].\n")

    # Check if Groq API is available and not rate-limited
    use_groq = False
    if settings.groq_api_key:
        try:
            temp_llm = get_llm("groq")
            temp_llm.generate("Hello")
            use_groq = True
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e) or "quota" in str(e).lower():
                console.print("[yellow]Groq API is rate-limited or quota exceeded. Automatically falling back to Google Gemini for this run.[/yellow]\n")
            else:
                console.print(f"[yellow]Groq API check failed ({e}). Falling back to Google Gemini for this run.[/yellow]\n")

    # Initialize RAG components
    embeddings = GeminiEmbeddings()
    retriever = PostgresVectorRetriever(embeddings_service=embeddings)
    llm = get_llm("groq" if use_groq else "gemini")

    questions = []
    contexts = []
    answers = []
    ground_truths = []

    # 4. Generate LLM answers and collect contexts
    with console.status("[bold green]Executing Q&A over resume chunks...", spinner="dots"):
        for pair in qa_pairs:
            q = pair["question"]
            gt = pair["ground_truth"]
            
            # Retrieve context chunks
            retrieved_docs = retriever.retrieve(q, resume_id=resume_id, top_k=settings.top_k)
            retrieved_texts = [doc.page_content for doc in retrieved_docs]
            
            # Build RAG prompt and generate answer
            context_str = "\n\n".join([f"--- Chunk {i+1} ---\n{t}" for i, t in enumerate(retrieved_texts)])
            prompt = templates.RAG_PROMPT_TEMPLATE.format(context=context_str, question=q)
            
            answer = llm.generate(
                prompt=prompt,
                system_instruction=templates.RAG_SYSTEM_INSTRUCTION
            )
            
            questions.append(q)
            contexts.append(retrieved_texts)
            answers.append(answer)
            ground_truths.append(gt)

    console.print("[green]Generated answers for all QA pairs. Starting Ragas evaluation...[/green]")

    # 5. Execute Ragas evaluation
    try:
        evaluator = RagasEvaluator(provider="groq" if use_groq else "gemini")
        with console.status("[bold green]Evaluating with Ragas (Google Gemini Judge)...", spinner="dots"):
            scores = evaluator.evaluate_pipeline(
                questions=questions,
                contexts=contexts,
                answers=answers,
                ground_truths=ground_truths
            )
    except Exception as e:
        console.print(f"[bold red]Evaluation failed:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc(), style="dim red")
        sys.exit(1)

    # 6. Format and Display Report
    console.print("\n[bold green]=== Evaluation Report ===[/bold green]")

    scores_dict = scores._repr_dict

    score_table = Table(title="Ragas Core Metric Scores", style="cyan")
    score_table.add_column("Metric Name", style="bold")
    score_table.add_column("Score (0.0 - 1.0)", justify="right")
    
    score_table.add_row("Faithfulness (Groundedness)", f"{scores_dict.get('faithfulness', 0.0):.4f}")
    score_table.add_row("Context Precision", f"{scores_dict.get('context_precision', 0.0):.4f}")
    score_table.add_row("Context Recall", f"{scores_dict.get('context_recall', 0.0):.4f}")
    score_table.add_row("Answer Relevancy", f"{scores_dict.get('answer_relevancy', 0.0):.4f}")
    
    console.print(score_table)

    # Convert results to DataFrame to dump detailed outputs
    df = scores.to_pandas()
    
    # Map DataFrame columns to output keys to support newer Ragas versions
    q_col = "user_input" if "user_input" in df.columns else "question"
    c_col = "retrieved_contexts" if "retrieved_contexts" in df.columns else "contexts"
    a_col = "response" if "response" in df.columns else "answer"
    gt_col = "reference" if "reference" in df.columns else "ground_truth"
    
    # 7. Persist evaluation results
    reports_dir = PROJECT_ROOT / "evaluation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json_path = reports_dir / f"eval_report_{timestamp}.json"
    report_md_path = reports_dir / f"eval_report_{timestamp}.md"
    
    # Save detailed JSON report
    detailed_data = []
    for idx, row in df.iterrows():
        detailed_data.append({
            "question": row[q_col],
            "contexts": row[c_col],
            "answer": row[a_col],
            "ground_truth": row[gt_col],
            "scores": {
                "faithfulness": float(row.get("faithfulness", 0.0)),
                "context_precision": float(row.get("context_precision", 0.0)),
                "context_recall": float(row.get("context_recall", 0.0)),
                "answer_relevancy": float(row.get("answer_relevancy", 0.0))
            }
        })
        
    report_payload = {
        "timestamp": timestamp,
        "resume_evaluated": filename,
        "overall_scores": {k: float(v) for k, v in scores_dict.items()},
        "detailed_results": detailed_data
    }
    
    with open(report_json_path, "w") as f:
        json.dump(report_payload, f, indent=2)

    # Save detailed Markdown report
    md_lines = [
        f"# RAG Pipeline Evaluation Report",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Candidate Resume**: {filename}",
        f"",
        f"## Overall Scores",
        f"| Metric | Score (0.0 - 1.0) | Description |",
        f"| :--- | :---: | :--- |",
        f"| **Faithfulness** | {scores_dict.get('faithfulness', 0.0):.4f} | Measures if the generated answer is strictly supported by the retrieved context. |",
        f"| **Context Precision** | {scores_dict.get('context_precision', 0.0):.4f} | Measures if the retriever ranked relevant context higher than irrelevant context. |",
        f"| **Context Recall** | {scores_dict.get('context_recall', 0.0):.4f} | Measures if the retriever fetched all necessary info to match the ground truth. |",
        f"| **Answer Relevancy** | {scores_dict.get('answer_relevancy', 0.0):.4f} | Measures if the answer directly addresses the question. |",
        f"",
        f"## Detailed Row Analysis"
    ]
    
    for idx, row in df.iterrows():
        md_lines.append(f"### Question {idx + 1}: {row[q_col]}")
        md_lines.append(f"- **Ground Truth**: {row[gt_col]}")
        md_lines.append(f"- **Generated Answer**: {row[a_col]}")
        md_lines.append(f"- **Scores**:")
        md_lines.append(f"  - Faithfulness: {row.get('faithfulness', 0.0):.4f}")
        md_lines.append(f"  - Context Precision: {row.get('context_precision', 0.0):.4f}")
        md_lines.append(f"  - Context Recall: {row.get('context_recall', 0.0):.4f}")
        md_lines.append(f"  - Answer Relevancy: {row.get('answer_relevancy', 0.0):.4f}")
        md_lines.append(f"")
        
    with open(report_md_path, "w") as f:
        f.write("\n".join(md_lines))

    console.print(f"\n[green]Saved detailed JSON report to:[/green] {report_json_path.name}")
    console.print(f"[green]Saved detailed Markdown report to:[/green] {report_md_path.name}")


if __name__ == "__main__":
    run_evaluation()
