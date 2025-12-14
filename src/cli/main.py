"""
main.py

Click-based CLI interface for Stage 2 NLP Processing Service.
Provides command-line access to document processing, job management, and administrative functions.

Features:
- Single document processing with pretty output
- Batch document processing from JSONL files
- Job status monitoring and result retrieval
- Health checks across all services
- Service discovery and listing
"""

import click
import httpx
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.panel import Panel
from rich.json import JSON as RichJSON

# Initialize Rich console
console = Console()

# Default API base URL
DEFAULT_API_URL = "http://localhost:8000"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_api_url() -> str:
    """Get API base URL from environment or use default."""
    import os
    return os.getenv("NLP_API_URL", DEFAULT_API_URL)


def make_http_request(
    method: str,
    endpoint: str,
    json_data: Optional[Dict[str, Any]] = None,
    timeout: float = 300.0
) -> Dict[str, Any]:
    """
    Make HTTP request to the orchestrator service.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        json_data: Request body as dictionary
        timeout: Request timeout in seconds

    Returns:
        Response JSON as dictionary

    Raises:
        click.ClickException: On network or API errors
    """
    url = f"{get_api_url()}{endpoint}"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(method, url, json=json_data)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        raise click.ClickException(
            f"Failed to connect to API at {get_api_url()}\n"
            "Make sure the orchestrator service is running: ./run.sh start"
        )
    except httpx.HTTPStatusError as e:
        error_detail = ""
        try:
            error_detail = e.response.json().get("message", str(e))
        except:
            error_detail = e.response.text
        raise click.ClickException(f"API Error: {error_detail}")
    except Exception as e:
        raise click.ClickException(f"Request failed: {str(e)}")


def read_jsonl_file(file_path: str) -> list:
    """
    Read JSONL file with Stage1Document objects.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of document dictionaries

    Raises:
        click.ClickException: On file read or parse errors
    """
    path = Path(file_path)
    if not path.exists():
        raise click.ClickException(f"File not found: {file_path}")

    documents = []
    try:
        with open(path, 'r') as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    try:
                        doc = json.loads(line)
                        documents.append(doc)
                    except json.JSONDecodeError as e:
                        raise click.ClickException(
                            f"Invalid JSON on line {line_no}: {str(e)}"
                        )
    except IOError as e:
        raise click.ClickException(f"Failed to read file: {str(e)}")

    return documents


def save_results_to_file(data: Any, output_file: str) -> None:
    """
    Save results to JSON file with pretty formatting.

    Args:
        data: Data to save
        output_file: Output file path

    Raises:
        click.ClickException: On write errors
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        console.print(f"[green]Results saved to:[/green] {output_path.absolute()}")
    except IOError as e:
        raise click.ClickException(f"Failed to write output file: {str(e)}")


# =============================================================================
# MAIN CLI GROUP
# =============================================================================

@click.group()
@click.version_option(version="1.0.0", prog_name="nlp")
def cli():
    """
    Stage 2 NLP Processing Service CLI.

    Process documents for event and entity extraction.
    """
    pass


# =============================================================================
# DOCUMENTS COMMAND GROUP
# =============================================================================

@cli.group()
def documents():
    """Document processing commands."""
    pass


@documents.command(name="process")
@click.argument("text")
@click.option(
    "--output",
    "-o",
    type=str,
    default=None,
    help="Save results to JSON file"
)
@click.option(
    "--document-id",
    "-d",
    type=str,
    default=None,
    help="Document identifier (optional)"
)
def process_single(text: str, output: Optional[str], document_id: Optional[str]):
    """
    Process a single text document.

    Extracts entities, events, and relationships from the provided text.
    Returns structured results with confidence scores.

    Example:
        nlp documents process "Apple announced a new product today"
    """
    console.print("[cyan]Processing document...[/cyan]")

    # Prepare request payload
    payload = {
        "text": text,
        "document_id": document_id or f"cli_doc_{id(text)}"
    }

    try:
        with console.status("[bold cyan]Calling orchestrator service..."):
            response = make_http_request("POST", "/v1/documents", payload)

        # Handle response
        if response.get("success"):
            result = response.get("result", {})
            processing_time = response.get("processing_time_ms", 0)

            # Display summary
            console.print("\n")
            console.print(Panel(
                f"[green]Processing completed successfully[/green]\n"
                f"Document ID: {response.get('document_id')}\n"
                f"Processing time: {processing_time:.2f}ms",
                title="Success",
                border_style="green"
            ))

            # Display extraction summary
            entities = result.get("extracted_entities", [])
            events = result.get("events", [])
            triplets = result.get("extracted_soa_triplets", [])

            summary_table = Table(title="Extraction Summary")
            summary_table.add_column("Artifact Type", style="cyan")
            summary_table.add_column("Count", style="magenta")
            summary_table.add_row("Entities", str(len(entities)))
            summary_table.add_row("Events", str(len(events)))
            summary_table.add_row("SOA Triplets", str(len(triplets)))
            console.print(summary_table)

            # Display entities
            if entities:
                console.print("\n[bold]Extracted Entities:[/bold]")
                entity_table = Table()
                entity_table.add_column("Text", style="green")
                entity_table.add_column("Type", style="yellow")
                entity_table.add_column("Confidence", style="blue")
                for ent in entities[:10]:  # Show first 10
                    conf = f"{ent.get('confidence', 1.0):.2f}"
                    entity_table.add_row(ent.get("text", ""), ent.get("type", ""), conf)
                if len(entities) > 10:
                    entity_table.add_row("[dim]...[/dim]", f"[dim]+{len(entities)-10} more[/dim]", "")
                console.print(entity_table)

            # Display events
            if events:
                console.print("\n[bold]Extracted Events:[/bold]")
                event_table = Table()
                event_table.add_column("Type", style="green")
                event_table.add_column("Trigger", style="yellow")
                event_table.add_column("Arguments", style="blue")
                for evt in events[:5]:  # Show first 5
                    trigger = evt.get("trigger", {}).get("text", "")
                    args_count = len(evt.get("arguments", []))
                    event_table.add_row(
                        evt.get("event_type", ""),
                        trigger,
                        f"{args_count} args"
                    )
                if len(events) > 5:
                    event_table.add_row("[dim]...[/dim]", f"[dim]+{len(events)-5} more[/dim]", "")
                console.print(event_table)

            # Save to file if requested
            if output:
                save_results_to_file(result, output)

            # Show full JSON option
            console.print(
                "\n[dim]Tip: Use --output to save full results to file[/dim]"
            )
        else:
            error_msg = response.get("error", "Unknown error")
            raise click.ClickException(f"Processing failed: {error_msg}")

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Unexpected error: {str(e)}")


@documents.command(name="batch")
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=str,
    default=None,
    help="Save results to JSON file"
)
@click.option(
    "--batch-id",
    "-b",
    type=str,
    default=None,
    help="Batch identifier (optional)"
)
def process_batch(file_path: str, output: Optional[str], batch_id: Optional[str]):
    """
    Process a batch of documents from JSONL file.

    Submits documents to the asynchronous batch processing pipeline.
    Returns a job ID that can be used to track progress and retrieve results.

    File format: One JSON document per line (Stage1Document format)

    Example:
        nlp documents batch /path/to/documents.jsonl
    """
    console.print(f"[cyan]Reading documents from:[/cyan] {file_path}")

    try:
        documents_list = read_jsonl_file(file_path)
    except click.ClickException:
        raise

    if not documents_list:
        raise click.ClickException("No documents found in file")

    console.print(f"[cyan]Found {len(documents_list)} documents[/cyan]")

    # Prepare request payload
    payload = {
        "documents": documents_list,
        "batch_id": batch_id or f"cli_batch_{id(file_path)}"
    }

    try:
        with console.status("[bold cyan]Submitting batch to orchestrator..."):
            response = make_http_request("POST", "/v1/documents/batch", payload)

        # Handle response
        if response.get("success"):
            job_id = response.get("job_id")
            batch_id_returned = response.get("batch_id")

            console.print("\n")
            console.print(Panel(
                f"[green]Batch submitted successfully[/green]\n"
                f"Batch ID: {batch_id_returned}\n"
                f"Job ID: [bold]{job_id}[/bold]\n"
                f"Documents: {response.get('document_count')}",
                title="Submission Success",
                border_style="green"
            ))

            # Save job ID for later reference
            if output:
                save_results_to_file({"job_id": job_id, "batch_id": batch_id_returned}, output)

            # Next steps
            console.print(
                f"\n[cyan]Next steps:[/cyan]\n"
                f"  1. Check progress: [bold]nlp jobs status {job_id}[/bold]\n"
                f"  2. Get results: [bold]nlp jobs results {job_id}[/bold]"
            )
        else:
            error_msg = response.get("error", "Unknown error")
            raise click.ClickException(f"Batch submission failed: {error_msg}")

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Unexpected error: {str(e)}")


# =============================================================================
# JOBS COMMAND GROUP
# =============================================================================

@cli.group()
def jobs():
    """Job management commands."""
    pass


@jobs.command(name="status")
@click.argument("job_id")
def job_status(job_id: str):
    """
    Get status of a processing job.

    Shows job progress, number of documents processed, and overall status.

    Example:
        nlp jobs status abc-def-123
    """
    try:
        with console.status("[bold cyan]Fetching job status..."):
            response = make_http_request("GET", f"/v1/jobs/{job_id}")

        # Extract status information
        status = response.get("status", "UNKNOWN")
        progress = response.get("progress", 0)
        docs_processed = response.get("documents_processed", 0)
        docs_total = response.get("documents_total", 0)
        error = response.get("error")

        # Display status
        console.print("\n")
        status_color = {
            "PENDING": "yellow",
            "STARTED": "cyan",
            "SUCCESS": "green",
            "FAILURE": "red"
        }.get(status, "white")

        console.print(Panel(
            f"[{status_color}]Status: {status}[/{status_color}]\n"
            f"Progress: {progress:.1f}%\n"
            f"Documents processed: {docs_processed}/{docs_total}",
            title=f"Job {job_id}",
            border_style=status_color
        ))

        # Display progress bar if in progress
        if status in ["PENDING", "STARTED"] and docs_total > 0:
            with Progress() as progress_bar:
                task = progress_bar.add_task(
                    "[cyan]Processing...",
                    total=docs_total,
                    completed=docs_processed
                )
                progress_bar.update(task, completed=docs_processed)

        # Display error if present
        if error:
            console.print(f"\n[red]Error:[/red] {error}")

        # Display result summary if completed
        if status == "SUCCESS" and response.get("result"):
            result = response.get("result", {})
            summary_table = Table(title="Result Summary")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="magenta")

            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        summary_table.add_row(str(key), str(value))

            console.print(summary_table)

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to get job status: {str(e)}")


@jobs.command(name="results")
@click.argument("job_id")
@click.option(
    "--output",
    "-o",
    type=str,
    required=True,
    help="Output file path for results"
)
def job_results(job_id: str, output: str):
    """
    Retrieve results for a completed job.

    Downloads and saves the processing results to a JSON file.

    Example:
        nlp jobs results abc-def-123 --output results.json
    """
    try:
        with console.status("[bold cyan]Fetching job results..."):
            response = make_http_request("GET", f"/v1/jobs/{job_id}")

        status = response.get("status")
        if status != "SUCCESS":
            raise click.ClickException(
                f"Job is still {status.lower()}. Wait for completion before retrieving results."
            )

        result = response.get("result")
        if not result:
            raise click.ClickException("No results found for this job")

        # Save results
        save_results_to_file(result, output)
        console.print(f"[green]Successfully retrieved {len(result) if isinstance(result, list) else 1} result(s)[/green]")

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to get job results: {str(e)}")


# =============================================================================
# ADMIN COMMAND GROUP
# =============================================================================

@cli.group()
def admin():
    """Administrative commands."""
    pass


@admin.command(name="health")
def health_check():
    """
    Health check for all services.

    Verifies that all microservices (NER, DP, Event LLM, Orchestrator) are running
    and responding correctly.

    Example:
        nlp admin health
    """
    try:
        with console.status("[bold cyan]Running health checks..."):
            response = make_http_request("GET", "/health")

        overall_status = response.get("status", "unknown")
        services = response.get("services", {})
        timestamp = response.get("timestamp")

        # Display overall status
        status_color = "green" if overall_status == "ok" else "yellow" if overall_status == "degraded" else "red"
        console.print("\n")
        console.print(Panel(
            f"[{status_color}]Overall Status: {overall_status.upper()}[/{status_color}]\n"
            f"Timestamp: {timestamp}",
            title="Health Check",
            border_style=status_color
        ))

        # Display service status table
        if services:
            service_table = Table(title="Service Status")
            service_table.add_column("Service", style="cyan")
            service_table.add_column("Status", style="green")
            service_table.add_column("Port", style="yellow")
            service_table.add_column("Details", style="blue")

            for service_name, service_info in services.items():
                svc_status = service_info.get("status", "unknown")
                port = service_info.get("port", "-")
                details = service_info.get("message", "")

                status_emoji = "✓" if svc_status == "ok" else "⚠" if svc_status == "degraded" else "✗"
                service_table.add_row(
                    f"{status_emoji} {service_name}",
                    svc_status,
                    str(port),
                    details[:40] + "..." if len(details) > 40 else details
                )

            console.print(service_table)

            # Show warnings for non-healthy services
            unhealthy = [s for s, i in services.items() if i.get("status") != "ok"]
            if unhealthy:
                console.print(
                    f"\n[yellow]Warning:[/yellow] "
                    f"{len(unhealthy)} service(s) not healthy: {', '.join(unhealthy)}"
                )

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Health check failed: {str(e)}")


@admin.command(name="services")
def list_services():
    """
    List all available services with their URLs and ports.

    Shows configuration for all microservices in the system.

    Example:
        nlp admin services
    """
    # Get service information from config or use defaults
    services_info = {
        "Orchestrator": {"url": "http://localhost:8000", "port": 8000, "description": "Main API and job management"},
        "NER Service": {"url": "http://localhost:8001", "port": 8001, "description": "Named Entity Recognition"},
        "DP Service": {"url": "http://localhost:8002", "port": 8002, "description": "Dependency Parsing"},
        "Event LLM Service": {"url": "http://localhost:8003", "port": 8003, "description": "Event extraction with LLM"},
        "Redis": {"url": "redis://localhost:6379", "port": 6379, "description": "Job queue and caching"},
    }

    console.print("\n")
    services_table = Table(title="Available Services")
    services_table.add_column("Service", style="cyan", width=20)
    services_table.add_column("URL", style="green")
    services_table.add_column("Port", style="yellow")
    services_table.add_column("Description", style="blue")

    for service_name, info in services_info.items():
        services_table.add_row(
            service_name,
            info["url"],
            str(info["port"]),
            info["description"]
        )

    console.print(services_table)

    console.print(
        "\n[dim]Tip: Set NLP_API_URL environment variable to use different orchestrator URL[/dim]"
    )


# =============================================================================
# ERROR HANDLING & ENTRY POINT
# =============================================================================

def main():
    """Entry point for CLI."""
    try:
        cli()
    except click.ClickException as e:
        console.print(f"[red]Error:[/red] {e.format_message()}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
