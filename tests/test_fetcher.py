"""Tests for bug_triage.fetcher — GitHub fetcher and local file parser."""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bug_triage.fetcher import (
    FetcherError,
    GitHubFetcher,
    LocalFileFetcher,
    fetch_issues,
)
from bug_triage.models import Issue


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_gh_issue(
    number: int = 1,
    title: str = "Test issue",
    body: str = "Body text",
    labels: list[str] | None = None,
    state: str = "open",
    comments: int = 0,
    is_pr: bool = False,
) -> MagicMock:
    """Build a mock PyGithub Issue object."""
    mock = MagicMock()
    mock.number = number
    mock.title = title
    mock.body = body
    mock.labels = [
        MagicMock(name=lbl) for lbl in (labels or [])
    ]
    # Make label.name return the string correctly
    for lbl_mock, lbl_str in zip(mock.labels, labels or []):
        lbl_mock.name = lbl_str
    mock.created_at = datetime(2024, 1, 10, 9, 0, 0)
    mock.updated_at = datetime(2024, 1, 11, 9, 0, 0)
    mock.html_url = f"https://github.com/example/repo/issues/{number}"
    mock.user = MagicMock()
    mock.user.login = "alice"
    mock.state = state
    mock.comments = comments
    mock.pull_request = MagicMock() if is_pr else None
    return mock


# ---------------------------------------------------------------------------
# GitHubFetcher tests
# ---------------------------------------------------------------------------


class TestGitHubFetcher:
    def test_fetch_returns_issues(self, mocker):
        """Happy-path: fetching open issues from a repo."""
        gh_issue = _make_gh_issue(number=42, title="Crash on login", labels=["bug"])

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [gh_issue]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher(token="fake-token")
        issues = fetcher.fetch("example/repo")

        assert len(issues) == 1
        issue = issues[0]
        assert isinstance(issue, Issue)
        assert issue.id == 42
        assert issue.title == "Crash on login"
        assert issue.labels == ["bug"]
        assert issue.repository == "example/repo"

    def test_fetch_skips_pull_requests(self, mocker):
        """Pull-request objects returned by get_issues() are skipped."""
        pr = _make_gh_issue(number=10, title="Add feature", is_pr=True)
        real_issue = _make_gh_issue(number=11, title="Real bug")

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [pr, real_issue]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher()
        issues = fetcher.fetch("example/repo")

        assert len(issues) == 1
        assert issues[0].id == 11

    def test_fetch_respects_max_issues(self, mocker):
        """max_issues limits the number of issues returned."""
        gh_issues = [_make_gh_issue(number=i, title=f"Issue {i}") for i in range(1, 11)]

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = gh_issues

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher(max_issues=3)
        issues = fetcher.fetch("example/repo")

        assert len(issues) == 3

    def test_fetch_raises_fetcher_error_on_github_exception(self, mocker):
        """GithubException is wrapped in FetcherError."""
        from github import GithubException

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(404, "Not Found")
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher()
        with pytest.raises(FetcherError, match="Failed to access GitHub repository"):
            fetcher.fetch("nonexistent/repo")

    def test_fetch_none_body_becomes_empty_string(self, mocker):
        """Issues with None body are converted to empty string."""
        gh_issue = _make_gh_issue(number=1, title="Empty body", body=None)  # type: ignore

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [gh_issue]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher()
        issues = fetcher.fetch("example/repo")
        assert issues[0].body == ""


# ---------------------------------------------------------------------------
# LocalFileFetcher — JSON tests
# ---------------------------------------------------------------------------


class TestLocalFileFetcherJSON:
    def test_parse_valid_json(self, tmp_path: Path):
        """A well-formed JSON file is parsed into Issue objects."""
        data = [
            {
                "id": 1,
                "title": "App crashes on login",
                "body": "Steps to reproduce...",
                "labels": ["bug", "crash"],
                "created_at": "2024-01-10T09:00:00Z",
                "url": "https://github.com/example/repo/issues/1",
                "state": "open",
            }
        ]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)

        assert len(issues) == 1
        assert issues[0].id == 1
        assert issues[0].title == "App crashes on login"
        assert issues[0].labels == ["bug", "crash"]

    def test_parse_json_with_comma_separated_labels(self, tmp_path: Path):
        """Labels stored as a comma-separated string are split correctly."""
        data = [{"id": 2, "title": "Perf issue", "labels": "performance,backend"}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)

        assert issues[0].labels == ["performance", "backend"]

    def test_parse_json_missing_optional_fields(self, tmp_path: Path):
        """Issues with only id and title are accepted; optionals default."""
        data = [{"id": 3, "title": "Minimal issue"}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)

        assert len(issues) == 1
        assert issues[0].body == ""
        assert issues[0].labels == []
        assert issues[0].state == "open"

    def test_parse_json_invalid_top_level(self, tmp_path: Path):
        """A JSON file with a top-level object instead of array raises FetcherError."""
        json_file = tmp_path / "bad.json"
        json_file.write_text(json.dumps({"id": 1, "title": "oops"}), encoding="utf-8")

        fetcher = LocalFileFetcher()
        with pytest.raises(FetcherError, match="top-level array"):
            fetcher.fetch(json_file)

    def test_parse_json_invalid_json(self, tmp_path: Path):
        """A malformed JSON file raises FetcherError."""
        json_file = tmp_path / "broken.json"
        json_file.write_text("{not valid json}", encoding="utf-8")

        fetcher = LocalFileFetcher()
        with pytest.raises(FetcherError, match="Failed to parse JSON"):
            fetcher.fetch(json_file)

    def test_max_issues_limits_json_results(self, tmp_path: Path):
        """max_issues trims the result list for JSON files."""
        data = [{"id": i, "title": f"Issue {i}"} for i in range(1, 10)]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher(max_issues=4)
        issues = fetcher.fetch(json_file)
        assert len(issues) == 4

    def test_invalid_records_are_skipped(self, tmp_path: Path):
        """Records with invalid ids are skipped with a warning."""
        data = [
            {"id": "not-an-int", "title": "Bad record"},
            {"id": 99, "title": "Good record"},
        ]
        json_file = tmp_path / "mixed.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)
        assert len(issues) == 1
        assert issues[0].id == 99


# ---------------------------------------------------------------------------
# LocalFileFetcher — CSV tests
# ---------------------------------------------------------------------------


class TestLocalFileFetcherCSV:
    def test_parse_valid_csv(self, tmp_path: Path):
        """A well-formed CSV file is parsed into Issue objects."""
        content = textwrap.dedent("""\
            id,title,body,labels,created_at,url,state,comments_count
            1,App crash,"Steps to reproduce","bug,crash",2024-01-10T09:00:00Z,https://example.com/1,open,3
        """)
        csv_file = tmp_path / "issues.csv"
        csv_file.write_text(content, encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(csv_file)

        assert len(issues) == 1
        assert issues[0].id == 1
        assert issues[0].title == "App crash"
        assert issues[0].comments_count == 3

    def test_parse_csv_missing_required_columns(self, tmp_path: Path):
        """CSV without 'id' or 'title' columns raises FetcherError."""
        content = "description,body\nsome bug,some body\n"
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text(content, encoding="utf-8")

        fetcher = LocalFileFetcher()
        with pytest.raises(FetcherError, match="missing required columns"):
            fetcher.fetch(csv_file)

    def test_parse_empty_csv(self, tmp_path: Path):
        """An empty CSV file returns an empty list."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(csv_file)
        assert issues == []


# ---------------------------------------------------------------------------
# LocalFileFetcher — edge cases
# ---------------------------------------------------------------------------


class TestLocalFileFetcherEdgeCases:
    def test_file_not_found_raises_fetcher_error(self):
        """A non-existent file path raises FetcherError."""
        fetcher = LocalFileFetcher()
        with pytest.raises(FetcherError, match="File not found"):
            fetcher.fetch("/nonexistent/path/issues.json")

    def test_unsupported_extension_raises_fetcher_error(self, tmp_path: Path):
        """An unsupported file extension raises FetcherError."""
        xml_file = tmp_path / "issues.xml"
        xml_file.write_text("<issues/>", encoding="utf-8")

        fetcher = LocalFileFetcher()
        with pytest.raises(FetcherError, match="Unsupported file format"):
            fetcher.fetch(xml_file)


# ---------------------------------------------------------------------------
# fetch_issues convenience function
# ---------------------------------------------------------------------------


class TestFetchIssues:
    def test_raises_when_neither_repo_nor_file(self):
        with pytest.raises(FetcherError, match="Either 'repo' or 'file' must be specified"):
            fetch_issues()

    def test_raises_when_both_repo_and_file(self):
        with pytest.raises(FetcherError, match="not both"):
            fetch_issues(repo="owner/repo", file="issues.json")

    def test_dispatches_to_github_fetcher(self, mocker):
        mock_fetch = mocker.patch(
            "bug_triage.fetcher.GitHubFetcher.fetch",
            return_value=[],
        )
        mocker.patch("bug_triage.fetcher.Github", return_value=MagicMock())
        result = fetch_issues(repo="owner/repo")
        mock_fetch.assert_called_once_with("owner/repo")
        assert result == []

    def test_dispatches_to_local_file_fetcher(self, tmp_path: Path):
        data = [{"id": 1, "title": "Test"}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        result = fetch_issues(file=str(json_file))
        assert len(result) == 1
        assert result[0].id == 1
