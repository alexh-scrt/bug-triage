# bug-triage 🐛

> Cut through issue backlog noise — let an LLM do the triage.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/your-org/bug-triage)

**bug-triage** is a developer CLI tool that ingests open GitHub issues or raw bug report files and uses an LLM (OpenAI or Anthropic) to automatically classify, deduplicate, and prioritize them by severity and impact. It produces a structured triage report in Markdown or JSON format — complete with a ranked fix order, complexity estimates, and duplicate groupings — so your engineering team can stop drowning in backlog noise and start shipping fixes that matter.

---

## Quick Start

### Install

```bash
pip install bug-triage
```

Or install from source:

```bash
git clone https://github.com/your-org/bug-triage.git
cd bug-triage
pip install -e .
```

### Configure

Create a `.env` file in your working directory:

```env
# Required: choose one LLM provider
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...

# Required for GitHub fetching
GITHUB_TOKEN=ghp_...

# Optional defaults
BUG_TRIAGE_PROVIDER=openai        # openai | anthropic
BUG_TRIAGE_MODEL=gpt-4o
BUG_TRIAGE_OUTPUT_FORMAT=markdown  # markdown | json
```

### Run

```bash
# Triage open issues from a GitHub repo
bug-triage triage --repo owner/repo-name

# Triage a local bug report file
bug-triage triage --file ./issues.json

# Output a JSON report instead of Markdown
bug-triage triage --repo owner/repo-name --format json --output report.json
```

That's it. A ranked triage report will be printed to your terminal and optionally saved to a file.

---

## Features

- **GitHub & Local File Ingestion** — Fetch open issues directly from any public or private GitHub repository, or point the tool at a local JSON or CSV file of bug reports.
- **Dual LLM Backend Support** — Seamlessly switch between OpenAI (`gpt-4o`, `gpt-4-turbo`) and Anthropic (`claude-3-5-sonnet`) with a single config flag. Includes automatic exponential-backoff retry logic.
- **LLM-Powered Classification** — Each issue is classified by severity (`critical` / `high` / `medium` / `low`) and impact category (`crash`, `performance`, `security`, `ux`, etc.) using a structured rubric prompt.
- **Automatic Deduplication & Clustering** — Related and duplicate issues are grouped into clusters, reducing noise and helping teams avoid redundant fix efforts.
- **Structured Report Output** — Produces a ranked Markdown or JSON report with suggested fix order, per-group complexity scores (`low` / `medium` / `high`), and cross-references between related issues.

---

## Usage Examples

### Triage a GitHub Repository

```bash
# Basic triage — prints Markdown report to stdout
bug-triage triage --repo acme-corp/backend-api

# Limit to the 50 most recent issues
bug-triage triage --repo acme-corp/backend-api --limit 50

# Use Anthropic instead of OpenAI
bug-triage triage --repo acme-corp/backend-api --provider anthropic --model claude-3-5-sonnet-20241022

# Save the report to a file
bug-triage triage --repo acme-corp/backend-api --output triage-report.md
```

### Triage a Local Bug Report File

```bash
# JSON file
bug-triage triage --file ./exports/issues.json

# CSV file
bug-triage triage --file ./exports/issues.csv --format json --output triage.json
```

### Fetch Issues Without Triaging

```bash
# Preview fetched issues as a table before running triage
bug-triage fetch --repo acme-corp/backend-api
```

### Re-render a Report from a Saved Triage JSON

```bash
# Generate a Markdown report from a previously saved JSON triage result
bug-triage report --input triage.json --format markdown --output report.md
```

### Example Report Output (Markdown)

```markdown
# Bug Triage Report
Generated: 2024-01-15 | Repo: acme-corp/backend-api | Issues analysed: 24

---

## 🔴 Critical (Fix Immediately)

### Group 1 — JWT Authentication Vulnerabilities
**Complexity:** High | **Issues:** #1, #2, #7

| Issue | Title | Severity | Impact |
|-------|-------|----------|--------|
| #1 | App crashes on login with malformed JWT | critical | security |
| #2 | Authentication bypass via null token header | critical | security |
| #7 | JWT secret exposed in error logs | critical | security |

**Suggested fix order:** #2 → #1 → #7
**Similar past issues:** #45, #61

---

## 🟠 High (Fix This Sprint)

### Group 2 — Checkout Flow Errors
**Complexity:** Medium | **Issues:** #5, #12
...
```

---

## Project Structure

```
bug-triage/
├── pyproject.toml                    # Project metadata, dependencies, CLI entry point
├── README.md
├── .env.example                      # Example environment variable configuration
│
├── bug_triage/
│   ├── __init__.py                   # Package init, exposes version
│   ├── cli.py                        # Typer CLI: triage, fetch, report commands
│   ├── fetcher.py                    # GitHub API fetcher + local JSON/CSV parser
│   ├── llm_client.py                 # OpenAI/Anthropic abstraction with retry logic
│   ├── triage.py                     # Core triage pipeline: classify, deduplicate, rank
│   ├── models.py                     # Pydantic models: Issue, TriageResult, ReportOutput
│   ├── reporter.py                   # Jinja2 + Rich report renderer (Markdown/JSON)
│   └── prompts/
│       ├── triage_prompt.j2          # LLM prompt: severity/impact classification
│       └── complexity_prompt.j2      # LLM prompt: fix complexity estimation
│
└── tests/
    ├── test_triage.py                # Unit tests: classification, deduplication, scoring
    ├── test_fetcher.py               # Tests: GitHub fetcher + local file parsing
    ├── test_reporter.py              # Tests: Markdown and JSON report rendering
    └── fixtures/
        └── sample_issues.json        # Sample GitHub issues fixture (no live API needed)
```

---

## Configuration

All configuration can be provided via environment variables or a `.env` file. CLI flags always take precedence.

| Variable | CLI Flag | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | — | — | OpenAI API key (required if using OpenAI) |
| `ANTHROPIC_API_KEY` | — | — | Anthropic API key (required if using Anthropic) |
| `GITHUB_TOKEN` | — | — | GitHub personal access token (required for private repos; recommended for rate limits) |
| `BUG_TRIAGE_PROVIDER` | `--provider` | `openai` | LLM provider: `openai` or `anthropic` |
| `BUG_TRIAGE_MODEL` | `--model` | `gpt-4o` | Model name passed to the LLM API |
| `BUG_TRIAGE_OUTPUT_FORMAT` | `--format` | `markdown` | Report format: `markdown` or `json` |
| `BUG_TRIAGE_MAX_ISSUES` | `--limit` | `100` | Maximum number of issues to fetch and triage |
| `BUG_TRIAGE_OUTPUT_FILE` | `--output` | stdout | Path to write the report file |

### Severity Rubric Weights

The triage prompt uses the following rubric by default:

| Severity | Criteria |
|---|---|
| `critical` | Data loss, security vulnerability, authentication bypass, complete service outage |
| `high` | Major feature broken, significant performance degradation, crash affecting many users |
| `medium` | Feature partially broken, moderate impact, reasonable workaround exists |
| `low` | Minor cosmetic issue, rare edge case, enhancement, or nice-to-have |

The rubric is defined in `bug_triage/prompts/triage_prompt.j2` and can be customised directly.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All tests use mocked GitHub API and LLM responses — no live API calls or credentials required.

---

## License

MIT © Bug Triage Contributors

---

*Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.*
