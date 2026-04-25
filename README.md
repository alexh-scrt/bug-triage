# bug-triage

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**bug-triage** is a developer CLI tool that ingests open GitHub issues or raw bug report files and uses an LLM (OpenAI or Anthropic) to automatically classify, deduplicate, and prioritize them by severity and impact.

It produces a structured triage report in Markdown or JSON format, complete with:
- Suggested fix order ranked by priority
- Estimated complexity scores per issue group
- Groupings of duplicate or related issues
- Links to similar past issues

---

## Features

- **GitHub Integration** — Fetch open issues directly from any public or private GitHub repository using the GitHub API.
- **Local File Ingestion** — Alternatively, ingest a local JSON or CSV bug report file.
- **Dual LLM Support** — Works with both OpenAI (`gpt-4o`, `gpt-4-turbo`, etc.) and Anthropic (`claude-3-5-sonnet`, etc.) backends.
- **Severity Classification** — Each issue is classified as `critical`, `high`, `medium`, or `low` severity with an impact category (`crash`, `performance`, `security`, `ux`, `other`).
- **Deduplication & Clustering** — Related and duplicate issues are automatically grouped into clusters to reduce noise.
- **Structured Reports** — Output in Markdown (human-readable) or JSON (machine-readable) format.
- **Configurable** — All settings configurable via environment variables or a `.env` file.

---

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/example/bug-triage.git
cd bug-triage
pip install -e ".[dev]"
```

### Using pip

```bash
pip install bug-triage
```

---

## Configuration

Create a `.env` file in your working directory (or export variables in your shell):

```env
# LLM Backend: "openai" or "anthropic"
BUG_TRIAGE_LLM_PROVIDER=openai

# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Model selection (optional, defaults provided)
BUG_TRIAGE_MODEL=gpt-4o

# GitHub Personal Access Token (required for private repos or higher rate limits)
GITHUB_TOKEN=ghp_...

# Output format: "markdown" or "json"
BUG_TRIAGE_OUTPUT_FORMAT=markdown

# Maximum number of issues to fetch (0 = unlimited)
BUG_TRIAGE_MAX_ISSUES=50
```

---

## Usage

### Commands Overview

```
bug-triage --help
```

```
Usage: bug-triage [OPTIONS] COMMAND [ARGS]...

  AI-powered bug triage CLI tool.

Commands:
  triage  Fetch issues and run full triage pipeline.
  fetch   Fetch and display issues without triaging.
  report  Generate a report from a previously saved triage JSON.
```

---

### `triage` — Full triage pipeline

Fetch open GitHub issues and produce a triage report:

```bash
# Triage issues from a GitHub repository
bug-triage triage --repo owner/repo-name

# Triage from a local JSON file
bug-triage triage --file ./issues.json

# Output as JSON instead of Markdown
bug-triage triage --repo owner/repo-name --format json

# Save the report to a file
bug-triage triage --repo owner/repo-name --output triage_report.md

# Limit the number of issues processed
bug-triage triage --repo owner/repo-name --max-issues 30

# Use Anthropic instead of OpenAI
bug-triage triage --repo owner/repo-name --provider anthropic --model claude-3-5-sonnet-20241022
```

---

### `fetch` — Fetch issues only

Fetch and display issues without running triage:

```bash
# Fetch issues from GitHub and display them
bug-triage fetch --repo owner/repo-name

# Fetch and save to a local file
bug-triage fetch --repo owner/repo-name --output issues.json

# Fetch from a local file and display
bug-triage fetch --file ./my_bugs.csv
```

---

### `report` — Generate report from saved triage data

Generate a report from a previously saved triage JSON output:

```bash
# Render a Markdown report from a triage JSON
bug-triage report --input triage_results.json

# Render as JSON
bug-triage report --input triage_results.json --format json

# Save rendered report to file
bug-triage report --input triage_results.json --output final_report.md
```

---

## Example Output

### Markdown Report

```markdown
# Bug Triage Report

**Generated:** 2024-01-15 14:32:00 UTC  
**Repository:** acme-corp/backend-api  
**Total Issues Analyzed:** 47  
**Issue Groups (after deduplication):** 31  

---

## 🔴 Critical (2 groups)

### Group 1 — Authentication bypass vulnerability
**Severity:** Critical | **Impact:** Security  
**Priority Score:** 98/100 | **Estimated Complexity:** High  
**Issues in group:** #234, #241, #255  

> Users can bypass JWT validation by sending a malformed token...

**Suggested Fix Order:** 1st  
**Related Issues:** #189 (closed, similar auth issue from Q3)

---

### Group 2 — Database connection pool exhaustion
**Severity:** Critical | **Impact:** Crash  
**Priority Score:** 95/100 | **Estimated Complexity:** Medium  
**Issues in group:** #267  

> Under high load, the connection pool is exhausted causing 500 errors...

**Suggested Fix Order:** 2nd  

---

## 🟠 High (5 groups)

...
```

### JSON Report (excerpt)

```json
{
  "generated_at": "2024-01-15T14:32:00Z",
  "repository": "acme-corp/backend-api",
  "total_issues": 47,
  "total_groups": 31,
  "groups": [
    {
      "id": "group_001",
      "title": "Authentication bypass vulnerability",
      "severity": "critical",
      "impact_category": "security",
      "priority_score": 98,
      "complexity": "high",
      "fix_order": 1,
      "issue_ids": [234, 241, 255],
      "summary": "Users can bypass JWT validation...",
      "similar_closed_issues": [189]
    }
  ]
}
```

---

## Local File Formats

### JSON format

The JSON file should contain an array of issue objects:

```json
[
  {
    "id": 1,
    "title": "App crashes on login",
    "body": "When I click the login button the app crashes with a NullPointerException.",
    "labels": ["bug", "crash"],
    "created_at": "2024-01-10T09:00:00Z",
    "url": "https://github.com/example/repo/issues/1"
  }
]
```

### CSV format

The CSV file should have the following columns (header row required):

```
id,title,body,labels,created_at,url
1,App crashes on login,"When I click login...","bug,crash",2024-01-10T09:00:00Z,https://...
```

---

## Development

### Setup

```bash
git clone https://github.com/example/bug-triage.git
cd bug-triage
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest

# With coverage
pytest --cov=bug_triage --cov-report=term-missing
```

### Project Structure

```
bug_triage/
├── __init__.py          # Package version
├── cli.py               # Typer CLI entry point
├── fetcher.py           # GitHub API + local file ingestion
├── llm_client.py        # OpenAI/Anthropic abstraction layer
├── triage.py            # Core triage pipeline
├── models.py            # Pydantic data models
├── reporter.py          # Report rendering (Markdown/JSON)
└── prompts/
    ├── triage_prompt.j2     # LLM triage prompt template
    └── complexity_prompt.j2 # LLM complexity estimation template
tests/
├── test_triage.py
├── test_fetcher.py
├── test_reporter.py
└── fixtures/
    └── sample_issues.json
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
