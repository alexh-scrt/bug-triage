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
    label_mocks = []
    for lbl_str in (labels or []):
        lbl_mock = MagicMock()
        lbl_mock.name = lbl_str
        label_mocks.append(lbl_mock)
    mock.labels = label_mocks
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
        gh_issue = _make_gh_issue(number=1, title="Empty body")
        gh_issue.body = None

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [gh_issue]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher()
        issues = fetcher.fetch("example/repo")
        assert issues[0].body == ""

    def test_fetch_without_token_uses_unauthenticated_client(self, mocker):
        """GitHubFetcher with no token still works (unauthenticated client)."""
        gh_issue = _make_gh_issue(number=1, title="Public issue")

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [gh_issue]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher(token=None)
        issues = fetcher.fetch("example/repo")
        assert len(issues) == 1

    def test_fetch_multiple_labels(self, mocker):
        """Multiple labels on a GitHub issue are all captured."""
        gh_issue = _make_gh_issue(
            number=5, title="Multi-label issue", labels=["bug", "security", "high-priority"]
        )

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [gh_issue]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher()
        issues = fetcher.fetch("example/repo")
        assert issues[0].labels == ["bug", "security", "high-priority"]

    def test_fetch_preserves_author_and_comments(self, mocker):
        """Author login and comments count are correctly captured."""
        gh_issue = _make_gh_issue(number=7, title="Issue with comments", comments=5)
        gh_issue.user.login = "bob"

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [gh_issue]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher()
        issues = fetcher.fetch("example/repo")
        assert issues[0].author == "bob"
        assert issues[0].comments_count == 5

    def test_fetch_max_issues_zero_means_unlimited(self, mocker):
        """max_issues=0 means no limit — all issues are returned."""
        gh_issues = [_make_gh_issue(number=i, title=f"Issue {i}") for i in range(1, 21)]

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = gh_issues

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher(max_issues=0)
        issues = fetcher.fetch("example/repo")
        assert len(issues) == 20

    def test_fetch_skips_all_pull_requests(self, mocker):
        """All pull requests in the paginated list are skipped."""
        prs = [_make_gh_issue(number=i, title=f"PR {i}", is_pr=True) for i in range(1, 4)]
        real_issues = [_make_gh_issue(number=i, title=f"Issue {i}") for i in range(10, 13)]

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = prs + real_issues

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher()
        issues = fetcher.fetch("example/repo")
        assert len(issues) == 3
        assert all(i.id >= 10 for i in issues)

    def test_fetch_html_url_stored_in_url_field(self, mocker):
        """html_url from PyGithub is stored in the Issue.url field."""
        gh_issue = _make_gh_issue(number=99, title="URL test")
        gh_issue.html_url = "https://github.com/owner/repo/issues/99"

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [gh_issue]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher()
        issues = fetcher.fetch("owner/repo")
        assert issues[0].url == "https://github.com/owner/repo/issues/99"

    def test_fetch_issue_state_captured(self, mocker):
        """Issue state (open/closed) is correctly mapped."""
        gh_issue = _make_gh_issue(number=3, title="Open issue", state="open")

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [gh_issue]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher()
        issues = fetcher.fetch("example/repo")
        assert issues[0].state == "open"

    def test_fetch_empty_repo_returns_empty_list(self, mocker):
        """A repo with no open issues returns an empty list."""
        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = []

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher()
        issues = fetcher.fetch("example/empty-repo")
        assert issues == []

    def test_fetch_no_user_object_on_issue(self, mocker):
        """Issues where user is None use empty string for author."""
        gh_issue = _make_gh_issue(number=1, title="Anonymous issue")
        gh_issue.user = None

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [gh_issue]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher()
        issues = fetcher.fetch("example/repo")
        assert issues[0].author == ""

    def test_fetch_timestamps_captured(self, mocker):
        """created_at and updated_at timestamps are correctly mapped."""
        gh_issue = _make_gh_issue(number=1, title="Timestamped issue")
        gh_issue.created_at = datetime(2024, 3, 15, 10, 0, 0)
        gh_issue.updated_at = datetime(2024, 3, 16, 12, 0, 0)

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [gh_issue]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mocker.patch("bug_triage.fetcher.Github", return_value=mock_github)

        fetcher = GitHubFetcher()
        issues = fetcher.fetch("example/repo")
        assert issues[0].created_at == datetime(2024, 3, 15, 10, 0, 0)
        assert issues[0].updated_at == datetime(2024, 3, 16, 12, 0, 0)


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

    def test_parse_json_with_null_labels(self, tmp_path: Path):
        """Issues with null labels field default to empty list."""
        data = [{"id": 4, "title": "No labels", "labels": None}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)
        assert issues[0].labels == []

    def test_parse_json_with_body_from_description_field(self, tmp_path: Path):
        """'description' field is used as body if 'body' is absent."""
        data = [{"id": 5, "title": "Issue with description", "description": "Some description."}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)
        assert issues[0].body == "Some description."

    def test_parse_json_url_from_html_url_field(self, tmp_path: Path):
        """'html_url' field is used as url if 'url' is absent."""
        data = [
            {
                "id": 6,
                "title": "URL from html_url",
                "html_url": "https://github.com/owner/repo/issues/6",
            }
        ]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)
        assert issues[0].url == "https://github.com/owner/repo/issues/6"

    def test_parse_json_number_field_as_id(self, tmp_path: Path):
        """'number' field is accepted as an alternative to 'id'."""
        data = [{"number": 7, "title": "Issue with number field"}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)
        assert issues[0].id == 7

    def test_parse_json_empty_array(self, tmp_path: Path):
        """An empty JSON array returns an empty list."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("[]", encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)
        assert issues == []

    def test_parse_json_comments_count_field(self, tmp_path: Path):
        """comments_count is parsed correctly from JSON."""
        data = [{"id": 8, "title": "Commented issue", "comments_count": 7}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)
        assert issues[0].comments_count == 7

    def test_parse_json_comments_field_alternative(self, tmp_path: Path):
        """'comments' field is accepted as alternative to 'comments_count'."""
        data = [{"id": 9, "title": "Alt comments field", "comments": 3}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)
        assert issues[0].comments_count == 3

    def test_parse_json_author_from_user_field(self, tmp_path: Path):
        """'user' field is accepted as alternative to 'author'."""
        data = [{"id": 10, "title": "User field test", "user": "bob"}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)
        assert issues[0].author == "bob"

    def test_parse_json_multiple_records(self, tmp_path: Path):
        """Multiple records are all parsed correctly."""
        data = [{"id": i, "title": f"Issue {i}"} for i in range(1, 6)]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)
        assert len(issues) == 5
        assert [i.id for i in issues] == [1, 2, 3, 4, 5]

    def test_parse_json_max_issues_zero_is_unlimited(self, tmp_path: Path):
        """max_issues=0 means no limit."""
        data = [{"id": i, "title": f"Issue {i}"} for i in range(1, 15)]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher(max_issues=0)
        issues = fetcher.fetch(json_file)
        assert len(issues) == 14

    def test_parse_json_various_datetime_formats(self, tmp_path: Path):
        """Various datetime string formats are parsed without error."""
        data = [
            {"id": 1, "title": "ISO with Z", "created_at": "2024-01-10T09:00:00Z"},
            {"id": 2, "title": "ISO without Z", "created_at": "2024-01-10T09:00:00"},
            {"id": 3, "title": "Date only", "created_at": "2024-01-10"},
            {"id": 4, "title": "Space separator", "created_at": "2024-01-10 09:00:00"},
        ]
        json_file = tmp_path / "dates.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)
        assert len(issues) == 4
        # All should have created_at set (not None)
        for issue in issues:
            assert issue.created_at is not None

    def test_parse_json_invalid_datetime_is_none(self, tmp_path: Path):
        """Unparseable datetime values are set to None without raising."""
        data = [{"id": 1, "title": "Bad date", "created_at": "not-a-date"}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)
        assert len(issues) == 1
        assert issues[0].created_at is None

    def test_parse_json_accepts_path_object(self, tmp_path: Path):
        """fetch() accepts a pathlib.Path object."""
        data = [{"id": 1, "title": "Path object test"}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)  # Path object, not str
        assert len(issues) == 1

    def test_parse_json_accepts_string_path(self, tmp_path: Path):
        """fetch() accepts a string path."""
        data = [{"id": 1, "title": "String path test"}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(str(json_file))  # str, not Path
        assert len(issues) == 1


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

    def test_parse_csv_multiple_rows(self, tmp_path: Path):
        """Multiple CSV rows are all parsed correctly."""
        content = textwrap.dedent("""\
            id,title,body
            1,Issue One,Body one
            2,Issue Two,Body two
            3,Issue Three,Body three
        """)
        csv_file = tmp_path / "multi.csv"
        csv_file.write_text(content, encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(csv_file)
        assert len(issues) == 3
        assert issues[0].id == 1
        assert issues[1].id == 2
        assert issues[2].id == 3

    def test_parse_csv_only_required_columns(self, tmp_path: Path):
        """CSV with only id and title columns is parsed without error."""
        content = "id,title\n10,Minimal CSV issue\n"
        csv_file = tmp_path / "minimal.csv"
        csv_file.write_text(content, encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(csv_file)
        assert len(issues) == 1
        assert issues[0].id == 10
        assert issues[0].title == "Minimal CSV issue"
        assert issues[0].body == ""
        assert issues[0].labels == []

    def test_parse_csv_comma_separated_labels(self, tmp_path: Path):
        """CSV labels stored as 'bug,crash' are split into a list."""
        content = textwrap.dedent("""\
            id,title,labels
            5,Label test,"bug,crash,performance"
        """)
        csv_file = tmp_path / "labels.csv"
        csv_file.write_text(content, encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(csv_file)
        assert issues[0].labels == ["bug", "crash", "performance"]

    def test_parse_csv_max_issues_limits_results(self, tmp_path: Path):
        """max_issues trims the result list for CSV files."""
        rows = "\n".join([f"{i},Issue {i}" for i in range(1, 11)])
        content = f"id,title\n{rows}\n"
        csv_file = tmp_path / "many.csv"
        csv_file.write_text(content, encoding="utf-8")

        fetcher = LocalFileFetcher(max_issues=5)
        issues = fetcher.fetch(csv_file)
        assert len(issues) == 5

    def test_parse_csv_invalid_id_row_skipped(self, tmp_path: Path):
        """A CSV row with an invalid ID is skipped; valid rows are returned."""
        content = textwrap.dedent("""\
            id,title
            bad-id,Bad row
            42,Good row
        """)
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text(content, encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(csv_file)
        assert len(issues) == 1
        assert issues[0].id == 42

    def test_parse_csv_state_column(self, tmp_path: Path):
        """State column is correctly captured from CSV."""
        content = "id,title,state\n1,Test,closed\n"
        csv_file = tmp_path / "state.csv"
        csv_file.write_text(content, encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(csv_file)
        assert issues[0].state == "closed"

    def test_parse_csv_missing_id_column_only(self, tmp_path: Path):
        """CSV missing only 'id' column raises FetcherError."""
        content = "title,body\nSome issue,Some body\n"
        csv_file = tmp_path / "no_id.csv"
        csv_file.write_text(content, encoding="utf-8")

        fetcher = LocalFileFetcher()
        with pytest.raises(FetcherError, match="missing required columns"):
            fetcher.fetch(csv_file)

    def test_parse_csv_missing_title_column_only(self, tmp_path: Path):
        """CSV missing only 'title' column raises FetcherError."""
        content = "id,body\n1,Some body\n"
        csv_file = tmp_path / "no_title.csv"
        csv_file.write_text(content, encoding="utf-8")

        fetcher = LocalFileFetcher()
        with pytest.raises(FetcherError, match="missing required columns"):
            fetcher.fetch(csv_file)


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

    def test_txt_extension_raises_fetcher_error(self, tmp_path: Path):
        """A .txt file raises FetcherError with unsupported format."""
        txt_file = tmp_path / "issues.txt"
        txt_file.write_text("some text", encoding="utf-8")

        fetcher = LocalFileFetcher()
        with pytest.raises(FetcherError, match="Unsupported file format"):
            fetcher.fetch(txt_file)

    def test_json_file_with_all_invalid_records_returns_empty(self, tmp_path: Path):
        """JSON file where every record is invalid returns an empty list."""
        data = [
            {"id": "bad", "title": "Record 1"},
            {"id": None, "title": "Record 2"},
        ]
        json_file = tmp_path / "all_bad.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(json_file)
        assert issues == []

    def test_parse_datetime_none_returns_none(self):
        """_parse_datetime with None input returns None."""
        fetcher = LocalFileFetcher()
        result = fetcher._parse_datetime(None)
        assert result is None

    def test_parse_datetime_empty_string_returns_none(self):
        """_parse_datetime with empty string returns None."""
        fetcher = LocalFileFetcher()
        result = fetcher._parse_datetime("")
        assert result is None

    def test_parse_datetime_with_datetime_object(self):
        """_parse_datetime with a datetime object returns it directly."""
        fetcher = LocalFileFetcher()
        dt = datetime(2024, 6, 15, 12, 0, 0)
        result = fetcher._parse_datetime(dt)
        assert result == dt

    def test_parse_datetime_iso_z_format(self):
        """_parse_datetime correctly parses ISO 8601 with Z suffix."""
        fetcher = LocalFileFetcher()
        result = fetcher._parse_datetime("2024-01-10T09:00:00Z")
        assert result == datetime(2024, 1, 10, 9, 0, 0)

    def test_parse_datetime_date_only_format(self):
        """_parse_datetime correctly parses date-only strings."""
        fetcher = LocalFileFetcher()
        result = fetcher._parse_datetime("2024-06-15")
        assert result == datetime(2024, 6, 15)

    def test_parse_datetime_garbage_returns_none(self):
        """_parse_datetime with garbage input returns None (no exception)."""
        fetcher = LocalFileFetcher()
        result = fetcher._parse_datetime("not-a-date-at-all!!!")
        assert result is None

    def test_csv_file_header_only_returns_empty(self, tmp_path: Path):
        """A CSV file with only a header row returns an empty list."""
        content = "id,title,body\n"
        csv_file = tmp_path / "header_only.csv"
        csv_file.write_text(content, encoding="utf-8")

        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(csv_file)
        assert issues == []


# ---------------------------------------------------------------------------
# fetch_issues convenience function
# ---------------------------------------------------------------------------


class TestFetchIssues:
    def test_raises_when_neither_repo_nor_file(self):
        """Calling without repo or file raises FetcherError."""
        with pytest.raises(FetcherError, match="Either 'repo' or 'file' must be specified"):
            fetch_issues()

    def test_raises_when_both_repo_and_file(self):
        """Calling with both repo and file raises FetcherError."""
        with pytest.raises(FetcherError, match="not both"):
            fetch_issues(repo="owner/repo", file="issues.json")

    def test_dispatches_to_github_fetcher(self, mocker):
        """With repo= argument, dispatches to GitHubFetcher."""
        mock_fetch = mocker.patch(
            "bug_triage.fetcher.GitHubFetcher.fetch",
            return_value=[],
        )
        mocker.patch("bug_triage.fetcher.Github", return_value=MagicMock())
        result = fetch_issues(repo="owner/repo")
        mock_fetch.assert_called_once_with("owner/repo")
        assert result == []

    def test_dispatches_to_local_file_fetcher(self, tmp_path: Path):
        """With file= argument, dispatches to LocalFileFetcher."""
        data = [{"id": 1, "title": "Test"}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        result = fetch_issues(file=str(json_file))
        assert len(result) == 1
        assert result[0].id == 1

    def test_github_token_forwarded_to_fetcher(self, mocker):
        """github_token is passed through to GitHubFetcher."""
        mock_init = mocker.patch("bug_triage.fetcher.GitHubFetcher.__init__", return_value=None)
        mocker.patch("bug_triage.fetcher.GitHubFetcher.fetch", return_value=[])

        fetch_issues(repo="owner/repo", github_token="ghp_test_token")
        mock_init.assert_called_once_with(token="ghp_test_token", max_issues=0)

    def test_max_issues_forwarded_to_github_fetcher(self, mocker):
        """max_issues is passed through to GitHubFetcher."""
        mock_init = mocker.patch("bug_triage.fetcher.GitHubFetcher.__init__", return_value=None)
        mocker.patch("bug_triage.fetcher.GitHubFetcher.fetch", return_value=[])

        fetch_issues(repo="owner/repo", max_issues=10)
        mock_init.assert_called_once_with(token=None, max_issues=10)

    def test_max_issues_forwarded_to_local_file_fetcher(self, mocker, tmp_path: Path):
        """max_issues is passed through to LocalFileFetcher."""
        mock_init = mocker.patch(
            "bug_triage.fetcher.LocalFileFetcher.__init__", return_value=None
        )
        mocker.patch("bug_triage.fetcher.LocalFileFetcher.fetch", return_value=[])

        dummy_file = tmp_path / "issues.json"
        dummy_file.write_text("[]", encoding="utf-8")

        fetch_issues(file=str(dummy_file), max_issues=5)
        mock_init.assert_called_once_with(max_issues=5)

    def test_returns_list_of_issue_objects(self, tmp_path: Path):
        """fetch_issues returns a list of Issue model instances."""
        data = [
            {"id": 1, "title": "Issue One"},
            {"id": 2, "title": "Issue Two"},
        ]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        issues = fetch_issues(file=str(json_file))
        assert len(issues) == 2
        assert all(isinstance(i, Issue) for i in issues)

    def test_fetch_issues_propagates_fetcher_error(self, mocker):
        """FetcherError from GitHubFetcher propagates to the caller."""
        mocker.patch(
            "bug_triage.fetcher.GitHubFetcher.fetch",
            side_effect=FetcherError("API error"),
        )
        mocker.patch("bug_triage.fetcher.Github", return_value=MagicMock())

        with pytest.raises(FetcherError, match="API error"):
            fetch_issues(repo="owner/repo")

    def test_fetch_issues_with_path_object_for_file(self, tmp_path: Path):
        """fetch_issues accepts a Path object for the file argument."""
        data = [{"id": 42, "title": "Path object test"}]
        json_file = tmp_path / "issues.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        issues = fetch_issues(file=json_file)  # Path object, not str
        assert len(issues) == 1
        assert issues[0].id == 42


# ---------------------------------------------------------------------------
# Sample fixture — tests/fixtures/sample_issues.json
# ---------------------------------------------------------------------------


class TestSampleIssuesFixture:
    """Tests that use the bundled sample_issues.json fixture file."""

    _FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_issues.json"

    def test_fixture_file_exists(self):
        """The sample fixture file is present in the test tree."""
        assert self._FIXTURE_PATH.exists(), (
            f"Fixture file not found at {self._FIXTURE_PATH}. "
            "Create tests/fixtures/sample_issues.json."
        )

    def test_fixture_file_is_valid_json(self):
        """The sample fixture file contains valid JSON."""
        if not self._FIXTURE_PATH.exists():
            pytest.skip("Fixture file not found.")
        content = self._FIXTURE_PATH.read_text(encoding="utf-8")
        data = json.loads(content)  # Should not raise
        assert isinstance(data, list)

    def test_fixture_parses_into_issues(self):
        """The sample fixture can be parsed into Issue objects."""
        if not self._FIXTURE_PATH.exists():
            pytest.skip("Fixture file not found.")
        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(self._FIXTURE_PATH)
        # There should be at least one issue in the fixture
        assert len(issues) >= 1
        assert all(isinstance(i, Issue) for i in issues)

    def test_fixture_issues_have_required_fields(self):
        """All issues parsed from the fixture have non-empty id and title."""
        if not self._FIXTURE_PATH.exists():
            pytest.skip("Fixture file not found.")
        fetcher = LocalFileFetcher()
        issues = fetcher.fetch(self._FIXTURE_PATH)
        for issue in issues:
            assert issue.id > 0
            assert issue.title.strip() != ""
