"""Typer-based CLI entry point for bug_triage.

Defines three commands:

* ``triage`` — fetch issues (from GitHub or a local file) and run the full
  triage pipeline, producing a Markdown or JSON report.
* ``fetch``  — fetch and display issues without running triage.
* ``report`` — generate a rendered report from a previously saved triage
  JSON file.

Configuration is read from environment variables or a ``.env`` file in the
working directory.  CLI flags always override environment settings.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich import box

from bug_triage import __version__
from bug_triage.fetcher import FetcherError, fetch_issues
from bug_triage.llm_client import LLMClient, LLMError
from bug_triage.models import (
    Issue,
    LLMProvider,
    OutputFormat,
    ReportOutput,
)
from bug_triage.reporter import Reporter, ReporterError, render_report
from bug_triage.triage import TriageEngine, TriageError

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

# Load .env from the current working directory (if present) so that env-vars
# are available before Typer parses arguments.
load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

# Rich console used for all CLI output.
_console = Console(stderr=False)
_err_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="bug-triage",
    help=(
        "AI-powered bug triage CLI tool.\n\n"
        "Ingests GitHub issues or local bug-report files and uses an LLM "
        "(OpenAI or Anthropic) to classify, deduplicate, and prioritize them, "
        "producing a structured triage report in Markdown or JSON format."
    ),
    add_completion=False,
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=False,
)


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:  # noqa: FBT001
    """Print the package version and exit."""
    if value:
        _console.print(f"bug-triage version [bold cyan]{__version__}[/bold cyan]")
        raise typer.Exit()


@app.callback()
def _main(
    version: Optional[bool] = typer.Option(  # noqa: UP007
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show the application version and exit.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose (DEBUG-level) logging.",
    ),
) -> None:
    """bug-triage: AI-powered GitHub issue triage using LLMs."""
    _configure_logging(verbose=verbose)


# ---------------------------------------------------------------------------
# Helper — logging setup
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool = False) -> None:
    """Configure the root logger with Rich formatting.

    Args:
        verbose: If ``True``, set the log level to DEBUG; otherwise WARNING.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=_err_console,
                rich_tracebacks=True,
                show_path=False,
            )
        ],
        force=True,
    )


# ---------------------------------------------------------------------------
# Helper — build LLMClient from CLI flags + env
# ---------------------------------------------------------------------------


def _build_llm_client(
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
) -> LLMClient:
    """Construct an :class:`~bug_triage.llm_client.LLMClient` from CLI options.

    Resolution order for each setting (first wins):
    1. Explicit CLI flag value.
    2. ``BUG_TRIAGE_LLM_PROVIDER`` / ``BUG_TRIAGE_MODEL`` environment variable.
    3. Package defaults.

    Args:
        provider: LLM provider string from CLI (``openai`` or ``anthropic``).
        model: Model name string from CLI.
        api_key: Explicit API key from CLI.

    Returns:
        A configured :class:`~bug_triage.llm_client.LLMClient`.

    Raises:
        typer.BadParameter: If the provider value is not recognised.
    """
    resolved_provider_str = (
        provider
        or os.environ.get("BUG_TRIAGE_LLM_PROVIDER", "")
        or "openai"
    ).lower().strip()

    try:
        resolved_provider = LLMProvider(resolved_provider_str)
    except ValueError:
        _err_console.print(
            f"[bold red]Error:[/bold red] Unknown LLM provider '{resolved_provider_str}'. "
            "Choose 'openai' or 'anthropic'."
        )
        raise typer.Exit(code=1)

    resolved_model = (
        model
        or os.environ.get("BUG_TRIAGE_MODEL", "")
        or None  # let LLMClient pick the default
    )

    # API key: prefer explicit flag, then provider-specific env var.
    resolved_key: Optional[str] = api_key or None
    if not resolved_key:
        if resolved_provider == LLMProvider.OPENAI:
            resolved_key = os.environ.get("OPENAI_API_KEY") or None
        else:
            resolved_key = os.environ.get("ANTHROPIC_API_KEY") or None

    return LLMClient(
        provider=resolved_provider,
        model=resolved_model,
        api_key=resolved_key,
    )


# ---------------------------------------------------------------------------
# Helper — resolve OutputFormat
# ---------------------------------------------------------------------------


def _resolve_output_format(fmt: Optional[str]) -> OutputFormat:
    """Resolve an output format string to an :class:`~bug_triage.models.OutputFormat`.

    Resolution order:
    1. ``fmt`` argument (CLI flag).
    2. ``BUG_TRIAGE_OUTPUT_FORMAT`` environment variable.
    3. Default: ``markdown``.

    Args:
        fmt: Output format string from CLI option.

    Returns:
        A resolved :class:`OutputFormat` enum value.
    """
    raw = (
        fmt
        or os.environ.get("BUG_TRIAGE_OUTPUT_FORMAT", "")
        or "markdown"
    ).lower().strip()
    try:
        return OutputFormat(raw)
    except ValueError:
        _err_console.print(
            f"[bold red]Error:[/bold red] Unknown output format '{raw}'. "
            "Choose 'markdown' or 'json'."
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Helper — resolve max_issues
# ---------------------------------------------------------------------------


def _resolve_max_issues(max_issues: Optional[int]) -> int:
    """Resolve the maximum number of issues to process.

    Resolution order:
    1. ``max_issues`` CLI flag.
    2. ``BUG_TRIAGE_MAX_ISSUES`` environment variable.
    3. Default: ``0`` (unlimited).

    Args:
        max_issues: Value from CLI option.

    Returns:
        An integer (``0`` means unlimited).
    """
    if max_issues is not None:
        return max(0, max_issues)
    env_val = os.environ.get("BUG_TRIAGE_MAX_ISSUES", "0")
    try:
        return max(0, int(env_val))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Helper — display issues table
# ---------------------------------------------------------------------------


def _print_issues_table(issues: list[Issue], console: Console) -> None:
    """Render a Rich table of issues to the given console.

    Args:
        issues: List of :class:`~bug_triage.models.Issue` objects to display.
        console: The Rich console to print to.
    """
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=False,
        title=f"Fetched Issues ({len(issues)} total)",
    )
    table.add_column("#", style="dim", width=6, justify="right")
    table.add_column("Title", min_width=30)
    table.add_column("Labels", min_width=12)
    table.add_column("State", width=6)
    table.add_column("Comments", width=9, justify="right")
    table.add_column("URL", min_width=20)

    for issue in issues:
        labels_str = ", ".join(issue.labels) if issue.labels else "-"
        table.add_row(
            str(issue.id),
            issue.title[:80] + ("…" if len(issue.title) > 80 else ""),
            labels_str[:40],
            issue.state,
            str(issue.comments_count),
            issue.url[:60] if issue.url else "-",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Command: triage
# ---------------------------------------------------------------------------


@app.command("triage")
def cmd_triage(
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="GitHub repository in 'owner/repo' format.",
        metavar="OWNER/REPO",
    ),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to a local JSON or CSV bug-report file.",
        exists=False,  # We do our own check for better error messages.
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the rendered report to this file path.",
        resolve_path=True,
    ),
    fmt: Optional[str] = typer.Option(
        None,
        "--format",
        help="Output format: 'markdown' (default) or 'json'.",
        metavar="FORMAT",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="LLM provider: 'openai' (default) or 'anthropic'.",
        metavar="PROVIDER",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model name (e.g. 'gpt-4o', 'claude-3-5-sonnet-20241022').",
        metavar="MODEL",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="LLM provider API key (overrides OPENAI_API_KEY / ANTHROPIC_API_KEY).",
        metavar="KEY",
        envvar="BUG_TRIAGE_API_KEY",
    ),
    github_token: Optional[str] = typer.Option(
        None,
        "--github-token",
        help="GitHub personal access token (overrides GITHUB_TOKEN).",
        metavar="TOKEN",
        envvar="GITHUB_TOKEN",
    ),
    max_issues: Optional[int] = typer.Option(
        None,
        "--max-issues",
        help="Maximum number of issues to fetch and triage (0 = unlimited).",
        metavar="N",
    ),
    batch_size: int = typer.Option(
        20,
        "--batch-size",
        help="Number of issues per LLM classification call.",
        metavar="N",
    ),
    save_raw: Optional[Path] = typer.Option(
        None,
        "--save-raw",
        help="Save the raw triage JSON output to this path (useful for later 'report' runs).",
        resolve_path=True,
    ),
    no_summary: bool = typer.Option(
        False,
        "--no-summary",
        help="Skip printing the Rich summary table to the terminal.",
    ),
) -> None:
    """Fetch issues and run the full triage pipeline.

    Fetch open GitHub issues (via --repo) or parse a local file (via --file),
    classify them with an LLM, deduplicate and cluster related issues, estimate
    fix complexity, and produce a prioritised triage report.

    \b
    Examples:
        bug-triage triage --repo owner/repo
        bug-triage triage --file issues.json --format json --output report.json
        bug-triage triage --repo owner/repo --provider anthropic --model claude-3-5-sonnet-20241022
    """
    # ---- Validate source args -------------------------------------------
    if repo and file:
        _err_console.print(
            "[bold red]Error:[/bold red] Specify either --repo or --file, not both."
        )
        raise typer.Exit(code=1)
    if not repo and not file:
        _err_console.print(
            "[bold red]Error:[/bold red] You must specify either --repo or --file."
        )
        raise typer.Exit(code=1)

    output_format = _resolve_output_format(fmt)
    resolved_max = _resolve_max_issues(max_issues)
    resolved_token = github_token or os.environ.get("GITHUB_TOKEN") or None

    # ---- Fetch issues ------------------------------------------------------
    _console.print("[cyan]⟳[/cyan] Fetching issues…")
    try:
        issues = fetch_issues(
            repo=repo,
            file=str(file) if file else None,
            github_token=resolved_token,
            max_issues=resolved_max,
        )
    except FetcherError as exc:
        _err_console.print(f"[bold red]Fetch error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if not issues:
        _console.print("[yellow]No issues found.  Nothing to triage.[/yellow]")
        raise typer.Exit(code=0)

    _console.print(
        f"[green]✔[/green] Fetched [bold]{len(issues)}[/bold] issue(s)."
    )

    # ---- Build LLM client --------------------------------------------------
    try:
        llm_client = _build_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
        )
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _err_console.print(
            f"[bold red]LLM client initialisation error:[/bold red] {exc}"
        )
        raise typer.Exit(code=1)

    _console.print(
        f"[cyan]⟳[/cyan] Running triage with "
        f"[bold]{llm_client.provider.value}[/bold] / [bold]{llm_client.model}[/bold]…"
    )

    # ---- Run triage --------------------------------------------------------
    engine = TriageEngine(
        llm_client=llm_client,
        batch_size=max(1, batch_size),
        repository=repo or "",
        source_file=str(file) if file else "",
        output_format=output_format,
    )

    try:
        report = engine.run(issues)
    except TriageError as exc:
        _err_console.print(f"[bold red]Triage error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except LLMError as exc:
        _err_console.print(f"[bold red]LLM error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    _console.print(
        f"[green]✔[/green] Triage complete: "
        f"[bold]{report.metadata.total_groups}[/bold] group(s) from "
        f"[bold]{report.metadata.total_issues}[/bold] issue(s)."
    )

    # ---- Save raw triage JSON if requested ---------------------------------
    if save_raw:
        try:
            raw_json = report.model_dump_json(indent=2)
            save_raw.parent.mkdir(parents=True, exist_ok=True)
            save_raw.write_text(raw_json, encoding="utf-8")
            _console.print(
                f"[green]✔[/green] Raw triage JSON saved to [bold]{save_raw}[/bold]."
            )
        except OSError as exc:
            _err_console.print(
                f"[yellow]Warning:[/yellow] Could not save raw triage JSON: {exc}"
            )

    # ---- Render and output report ------------------------------------------
    reporter = Reporter(
        output_format=output_format,
        console=_console,
    )

    try:
        rendered = reporter.render(report)
    except ReporterError as exc:
        _err_console.print(f"[bold red]Report rendering error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if output:
        try:
            reporter.save(report, output)
            _console.print(
                f"[green]✔[/green] Report written to [bold]{output}[/bold]."
            )
        except ReporterError as exc:
            _err_console.print(f"[bold red]Save error:[/bold red] {exc}")
            raise typer.Exit(code=1)
    else:
        # Print the rendered report to stdout.
        _console.print(rendered)

    # ---- Print Rich summary to terminal ------------------------------------
    if not no_summary:
        reporter.print_summary(report)


# ---------------------------------------------------------------------------
# Command: fetch
# ---------------------------------------------------------------------------


@app.command("fetch")
def cmd_fetch(
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="GitHub repository in 'owner/repo' format.",
        metavar="OWNER/REPO",
    ),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to a local JSON or CSV bug-report file.",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save fetched issues as JSON to this file path.",
        resolve_path=True,
    ),
    github_token: Optional[str] = typer.Option(
        None,
        "--github-token",
        help="GitHub personal access token (overrides GITHUB_TOKEN).",
        metavar="TOKEN",
        envvar="GITHUB_TOKEN",
    ),
    max_issues: Optional[int] = typer.Option(
        None,
        "--max-issues",
        help="Maximum number of issues to fetch (0 = unlimited).",
        metavar="N",
    ),
) -> None:
    """Fetch and display issues without running triage.

    Fetches open GitHub issues or parses a local file and displays them in a
    Rich table.  Optionally saves the fetched issues as a JSON file that can
    later be used with 'bug-triage triage --file'.

    \b
    Examples:
        bug-triage fetch --repo owner/repo
        bug-triage fetch --repo owner/repo --output issues.json
        bug-triage fetch --file ./my_bugs.csv
    """
    if repo and file:
        _err_console.print(
            "[bold red]Error:[/bold red] Specify either --repo or --file, not both."
        )
        raise typer.Exit(code=1)
    if not repo and not file:
        _err_console.print(
            "[bold red]Error:[/bold red] You must specify either --repo or --file."
        )
        raise typer.Exit(code=1)

    resolved_max = _resolve_max_issues(max_issues)
    resolved_token = github_token or os.environ.get("GITHUB_TOKEN") or None

    _console.print("[cyan]⟳[/cyan] Fetching issues…")
    try:
        issues = fetch_issues(
            repo=repo,
            file=str(file) if file else None,
            github_token=resolved_token,
            max_issues=resolved_max,
        )
    except FetcherError as exc:
        _err_console.print(f"[bold red]Fetch error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if not issues:
        _console.print("[yellow]No issues found.[/yellow]")
        raise typer.Exit(code=0)

    _console.print(
        f"[green]✔[/green] Fetched [bold]{len(issues)}[/bold] issue(s)."
    )

    # Display issues table.
    _print_issues_table(issues, _console)

    # Optionally save as JSON.
    if output:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            serialised = [
                json.loads(issue.model_dump_json()) for issue in issues
            ]
            output.write_text(
                json.dumps(serialised, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            _console.print(
                f"[green]✔[/green] Issues saved to [bold]{output}[/bold]."
            )
        except OSError as exc:
            _err_console.print(
                f"[bold red]Save error:[/bold red] Could not write to '{output}': {exc}"
            )
            raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Command: report
# ---------------------------------------------------------------------------


@app.command("report")
def cmd_report(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to a previously saved triage JSON file (from 'triage --save-raw').",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the rendered report to this file path.",
        resolve_path=True,
    ),
    fmt: Optional[str] = typer.Option(
        None,
        "--format",
        help="Output format: 'markdown' (default) or 'json'.",
        metavar="FORMAT",
    ),
    no_summary: bool = typer.Option(
        False,
        "--no-summary",
        help="Skip printing the Rich summary table to the terminal.",
    ),
) -> None:
    """Generate a report from a previously saved triage JSON file.

    Reads a triage JSON file produced by 'bug-triage triage --save-raw' and
    renders it as Markdown or JSON, optionally saving to a file.

    \b
    Examples:
        bug-triage report --input triage_results.json
        bug-triage report --input triage_results.json --format json
        bug-triage report --input triage_results.json --output final_report.md
    """
    output_format = _resolve_output_format(fmt)

    # ---- Load the saved triage JSON ----------------------------------------
    _console.print(f"[cyan]⟳[/cyan] Loading triage data from [bold]{input_path}[/bold]…")
    try:
        raw_json = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        _err_console.print(
            f"[bold red]Error:[/bold red] Could not read '{input_path}': {exc}"
        )
        raise typer.Exit(code=1)

    try:
        report = ReportOutput.model_validate_json(raw_json)
    except Exception as exc:  # noqa: BLE001
        _err_console.print(
            f"[bold red]Error:[/bold red] Failed to parse triage JSON from "
            f"'{input_path}': {exc}"
        )
        raise typer.Exit(code=1)

    _console.print(
        f"[green]✔[/green] Loaded report: "
        f"[bold]{report.metadata.total_groups}[/bold] group(s), "
        f"[bold]{report.metadata.total_issues}[/bold] issue(s)."
    )

    # Override the output format in metadata to match the requested format.
    report.metadata.output_format = output_format

    # ---- Render ------------------------------------------------------------
    reporter = Reporter(
        output_format=output_format,
        console=_console,
    )

    try:
        rendered = reporter.render(report)
    except ReporterError as exc:
        _err_console.print(f"[bold red]Rendering error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if output:
        try:
            reporter.save(report, output)
            _console.print(
                f"[green]✔[/green] Report written to [bold]{output}[/bold]."
            )
        except ReporterError as exc:
            _err_console.print(f"[bold red]Save error:[/bold red] {exc}")
            raise typer.Exit(code=1)
    else:
        _console.print(rendered)

    # ---- Print Rich summary ------------------------------------------------
    if not no_summary:
        reporter.print_summary(report)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    app()
