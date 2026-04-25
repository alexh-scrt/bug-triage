"""Issue fetcher for bug_triage.

This module fetches open GitHub issues via PyGithub or parses local
JSON/CSV bug report files and normalises them into the shared Issue model.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from github import Github, GithubException
from github.Repository import Repository

from bug_triage.models import Issue

logger = logging.getLogger(__name__)


class FetcherError(Exception):
    """Raised when issue fetching fails in an unrecoverable way."""


class GitHubFetcher:
    """Fetches open issues from a GitHub repository using PyGithub.

    Args:
        token: GitHub personal access token.  Pass ``None`` to use the
               unauthenticated client (lower rate limits).
        max_issues: Maximum number of issues to return.  ``0`` means unlimited.
    """

    def __init__(self, token: Optional[str] = None, max_issues: int = 0) -> None:
        self._client = Github(token) if token else Github()
        self.max_issues = max_issues

    def fetch(self, repo_name: str) -> list[Issue]:
        """Fetch open issues from the given repository.

        Args:
            repo_name: Repository identifier in ``owner/repo`` format.

        Returns:
            A list of :class:`~bug_triage.models.Issue` objects.

        Raises:
            FetcherError: If the repository cannot be accessed or the API call
                fails.
        """
        logger.info("Fetching issues from GitHub repository: %s", repo_name)
        try:
            repo: Repository = self._client.get_repo(repo_name)
            paginated = repo.get_issues(state="open")
        except GithubException as exc:
            raise FetcherError(
                f"Failed to access GitHub repository '{repo_name}': {exc}"
            ) from exc

        issues: list[Issue] = []
        for gh_issue in paginated:
            # PyGithub returns pull-requests via get_issues(); skip them.
            if gh_issue.pull_request is not None:
                continue

            labels = [lbl.name for lbl in gh_issue.labels]
            issue = Issue(
                id=gh_issue.number,
                title=gh_issue.title,
                body=gh_issue.body or "",
                labels=labels,
                created_at=gh_issue.created_at,
                updated_at=gh_issue.updated_at,
                url=gh_issue.html_url,
                author=gh_issue.user.login if gh_issue.user else "",
                state=gh_issue.state,
                comments_count=gh_issue.comments,
                repository=repo_name,
            )
            issues.append(issue)

            if self.max_issues and len(issues) >= self.max_issues:
                logger.debug(
                    "Reached max_issues limit (%d); stopping fetch.", self.max_issues
                )
                break

        logger.info("Fetched %d issues from %s.", len(issues), repo_name)
        return issues


class LocalFileFetcher:
    """Parses a local JSON or CSV file containing bug reports.

    JSON files must contain a top-level array of issue objects.
    CSV files must contain a header row with at minimum an ``id`` and
    ``title`` column.

    Args:
        max_issues: Maximum number of issues to return.  ``0`` means unlimited.
    """

    _DATETIME_FORMATS = (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    )

    def __init__(self, max_issues: int = 0) -> None:
        self.max_issues = max_issues

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, file_path: str | Path) -> list[Issue]:
        """Parse a local JSON or CSV file and return a list of issues.

        Args:
            file_path: Path to the JSON or CSV file.

        Returns:
            A list of :class:`~bug_triage.models.Issue` objects.

        Raises:
            FetcherError: If the file cannot be read or has an unsupported
                format.
        """
        path = Path(file_path)
        if not path.exists():
            raise FetcherError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            raw_records = self._parse_json(path)
        elif suffix == ".csv":
            raw_records = self._parse_csv(path)
        else:
            raise FetcherError(
                f"Unsupported file format '{suffix}'.  Expected .json or .csv."
            )

        issues = self._records_to_issues(raw_records, source=str(path))
        if self.max_issues:
            issues = issues[: self.max_issues]

        logger.info("Loaded %d issues from '%s'.", len(issues), path)
        return issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_json(self, path: Path) -> list[dict]:
        """Read and parse a JSON file returning a list of dicts."""
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise FetcherError(f"Failed to parse JSON file '{path}': {exc}") from exc

        if not isinstance(data, list):
            raise FetcherError(
                f"JSON file '{path}' must contain a top-level array of issue objects."
            )
        return data

    def _parse_csv(self, path: Path) -> list[dict]:
        """Read and parse a CSV file returning a list of dicts."""
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        except OSError as exc:
            raise FetcherError(f"Failed to read CSV file '{path}': {exc}") from exc

        if not rows:
            return []

        required = {"id", "title"}
        headers = set(rows[0].keys())
        missing = required - headers
        if missing:
            raise FetcherError(
                f"CSV file '{path}' is missing required columns: {missing}"
            )
        return rows

    def _records_to_issues(self, records: list[dict], source: str) -> list[Issue]:
        """Convert raw dicts to :class:`Issue` objects, skipping invalid rows."""
        issues: list[Issue] = []
        for idx, record in enumerate(records):
            try:
                issue = self._dict_to_issue(record, source=source)
                issues.append(issue)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping record %d in '%s' due to error: %s", idx, source, exc
                )
        return issues

    def _dict_to_issue(self, record: dict, source: str) -> Issue:
        """Map a raw dict record to an :class:`Issue`."""
        raw_id = record.get("id") or record.get("number")
        try:
            issue_id = int(raw_id)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid issue id '{raw_id}': {exc}") from exc

        created_at = self._parse_datetime(record.get("created_at"))
        updated_at = self._parse_datetime(record.get("updated_at"))

        return Issue(
            id=issue_id,
            title=str(record.get("title", "")).strip(),
            body=str(record.get("body") or record.get("description") or ""),
            labels=record.get("labels", []),
            created_at=created_at,
            updated_at=updated_at,
            url=str(record.get("url") or record.get("html_url") or ""),
            author=str(record.get("author") or record.get("user") or ""),
            state=str(record.get("state", "open")),
            comments_count=int(record.get("comments_count") or record.get("comments") or 0),
            repository=str(record.get("repository", source)),
        )

    def _parse_datetime(self, value: object) -> Optional[datetime]:
        """Attempt to parse a datetime string using known formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        raw = str(value).strip()
        if not raw:
            return None
        for fmt in self._DATETIME_FORMATS:
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        logger.debug("Could not parse datetime value '%s'; ignoring.", raw)
        return None


def fetch_issues(
    repo: Optional[str] = None,
    file: Optional[str | Path] = None,
    github_token: Optional[str] = None,
    max_issues: int = 0,
) -> list[Issue]:
    """Convenience function that dispatches to the appropriate fetcher.

    Exactly one of ``repo`` or ``file`` must be provided.

    Args:
        repo: GitHub repository in ``owner/repo`` format.
        file: Path to a local JSON or CSV file.
        github_token: GitHub personal access token (optional).
        max_issues: Maximum number of issues to return (``0`` = unlimited).

    Returns:
        A list of :class:`~bug_triage.models.Issue` objects.

    Raises:
        FetcherError: If neither or both source arguments are supplied, or if
            fetching/parsing fails.
    """
    if repo and file:
        raise FetcherError("Specify either 'repo' or 'file', not both.")
    if not repo and not file:
        raise FetcherError("Either 'repo' or 'file' must be specified.")

    if repo:
        fetcher = GitHubFetcher(token=github_token, max_issues=max_issues)
        return fetcher.fetch(repo)

    local_fetcher = LocalFileFetcher(max_issues=max_issues)
    return local_fetcher.fetch(file)  # type: ignore[arg-type]
