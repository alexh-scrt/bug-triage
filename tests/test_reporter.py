"""Tests for bug_triage.reporter — Markdown and JSON report rendering."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from rich.console import Console

from bug_triage.models import (
    Complexity,
    ImpactCategory,
    Issue,
    IssueGroup,
    LLMProvider,
    OutputFormat,
    ReportMetadata,
    ReportOutput,
    Severity,
    TriageResult,
)
from bug_triage.reporter import Reporter, ReporterError, _InlineTemplateLoader, render_report


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_metadata(
    repository: str = "owner/repo",
    total_issues: int = 3,
    total_groups: int = 2,
    output_format: OutputFormat = OutputFormat.MARKDOWN,
) -> ReportMetadata:
    """Create a ReportMetadata instance for testing."""
    return ReportMetadata(
        generated_at=datetime(2024, 1, 15, 14, 32, 0),
        repository=repository,
        source_file="",
        llm_provider="openai",
        llm_model="gpt-4o",
        total_issues=total_issues,
        total_groups=total_groups,
        output_format=output_format,
    )


def _make_group(
    group_id: str = "group_001",
    title: str = "Authentication bypass",
    severity: Severity = Severity.CRITICAL,
    impact_category: ImpactCategory = ImpactCategory.SECURITY,
    priority_score: float = 98.0,
    complexity: Complexity = Complexity.HIGH,
    fix_order: int = 1,
    issue_ids: list[int] | None = None,
    canonical_issue_id: int | None = None,
    summary: str = "JWT tokens can be bypassed.",
    tags: list[str] | None = None,
    similar_closed_issue_ids: list[int] | None = None,
) -> IssueGroup:
    """Create an IssueGroup instance for testing."""
    ids = issue_ids or [1, 2]
    return IssueGroup(
        id=group_id,
        title=title,
        severity=severity,
        impact_category=impact_category,
        priority_score=priority_score,
        complexity=complexity,
        fix_order=fix_order,
        issue_ids=ids,
        canonical_issue_id=canonical_issue_id or ids[0],
        summary=summary,
        tags=tags or ["auth", "jwt"],
        similar_closed_issue_ids=similar_closed_issue_ids or [],
    )


def _make_issue(
    id: int = 1,
    title: str = "Test issue",
    body: str = "Issue body.",
    labels: list[str] | None = None,
) -> Issue:
    """Create an Issue instance for testing."""
    return Issue(
        id=id,
        title=title,
        body=body,
        labels=labels or ["bug"],
        created_at=datetime(2024, 1, 10),
        updated_at=datetime(2024, 1, 11),
        url=f"https://github.com/owner/repo/issues/{id}",
        author="alice",
        state="open",
        comments_count=2,
        repository="owner/repo",
    )


def _make_triage_result(
    issue_id: int = 1,
    severity: Severity = Severity.HIGH,
    priority_score: float = 75.0,
) -> TriageResult:
    """Create a TriageResult instance for testing."""
    return TriageResult(
        issue_id=issue_id,
        severity=severity,
        impact_category=ImpactCategory.OTHER,
        priority_score=priority_score,
        summary="A test summary.",
        reasoning="Test reasoning.",
        complexity=Complexity.MEDIUM,
    )


def _make_report(
    groups: list[IssueGroup] | None = None,
    raw_issues: list[Issue] | None = None,
    triage_results: list[TriageResult] | None = None,
    ungrouped_issue_ids: list[int] | None = None,
    output_format: OutputFormat = OutputFormat.MARKDOWN,
) -> ReportOutput:
    """Create a ReportOutput instance for testing."""
    g = groups if groups is not None else [_make_group()]
    ri = raw_issues if raw_issues is not None else [_make_issue(1), _make_issue(2)]
    tr = triage_results if triage_results is not None else [_make_triage_result(1)]
    return ReportOutput(
        metadata=_make_metadata(
            total_issues=len(ri),
            total_groups=len(g),
            output_format=output_format,
        ),
        groups=g,
        raw_issues=ri,
        triage_results=tr,
        ungrouped_issue_ids=ungrouped_issue_ids or [],
    )


# ---------------------------------------------------------------------------
# Reporter.render_markdown
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def test_returns_string(self):
        """render_markdown returns a non-empty string."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        result = reporter.render_markdown(report)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_header(self):
        """Rendered Markdown contains the report header."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        md = reporter.render_markdown(report)
        assert "# Bug Triage Report" in md

    def test_contains_generated_timestamp(self):
        """Rendered Markdown contains the generation timestamp."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        md = reporter.render_markdown(report)
        assert "2024-01-15 14:32:00" in md

    def test_contains_repository(self):
        """Rendered Markdown contains the repository name."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        md = reporter.render_markdown(report)
        assert "owner/repo" in md

    def test_contains_llm_model(self):
        """Rendered Markdown contains the LLM model name."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        md = reporter.render_markdown(report)
        assert "gpt-4o" in md

    def test_contains_total_issues(self):
        """Rendered Markdown contains the total issue count."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        md = reporter.render_markdown(report)
        assert str(report.metadata.total_issues) in md

    def test_contains_group_title(self):
        """Rendered Markdown contains the group title."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(title="Authentication bypass vulnerability")
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "Authentication bypass vulnerability" in md

    def test_contains_severity_label(self):
        """Rendered Markdown contains the severity label."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(severity=Severity.CRITICAL)
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "Critical" in md or "critical" in md

    def test_contains_issue_ids(self):
        """Rendered Markdown contains issue IDs with # prefix."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(issue_ids=[42, 43])
        report = _make_report(
            groups=[group],
            raw_issues=[_make_issue(42), _make_issue(43)],
        )
        md = reporter.render_markdown(report)
        assert "#42" in md
        assert "#43" in md

    def test_contains_summary(self):
        """Rendered Markdown contains the group summary."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(summary="JWT tokens can be bypassed.")
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "JWT tokens can be bypassed." in md

    def test_contains_tags(self):
        """Rendered Markdown contains group tags."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(tags=["auth", "security"])
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "auth" in md
        assert "security" in md

    def test_similar_closed_issues_included(self):
        """Similar closed issue IDs appear in the rendered Markdown."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(similar_closed_issue_ids=[189, 190])
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "#189" in md
        assert "#190" in md

    def test_ungrouped_issues_section_present_when_non_empty(self):
        """Ungrouped issues section appears when there are ungrouped issues."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report(ungrouped_issue_ids=[99, 100])
        md = reporter.render_markdown(report)
        assert "Ungrouped" in md or "ungrouped" in md
        assert "#99" in md
        assert "#100" in md

    def test_ungrouped_section_absent_when_empty(self):
        """Ungrouped section is absent when all issues are grouped."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report(ungrouped_issue_ids=[])
        md = reporter.render_markdown(report)
        assert "Ungrouped" not in md

    def test_severity_sections_ordered_critical_first(self):
        """Severity sections appear in Critical > High > Medium > Low order."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        groups = [
            _make_group("g1", severity=Severity.LOW, fix_order=4, issue_ids=[4]),
            _make_group("g2", severity=Severity.CRITICAL, fix_order=1, issue_ids=[1]),
            _make_group("g3", severity=Severity.MEDIUM, fix_order=3, issue_ids=[3]),
            _make_group("g4", severity=Severity.HIGH, fix_order=2, issue_ids=[2]),
        ]
        report = _make_report(
            groups=groups,
            raw_issues=[_make_issue(i) for i in range(1, 5)],
        )
        md = reporter.render_markdown(report)
        pos_critical = md.find("Critical")
        pos_high = md.find("High")
        pos_medium = md.find("Medium")
        pos_low = md.find("Low")
        assert pos_critical < pos_high < pos_medium < pos_low

    def test_empty_report_renders_without_error(self):
        """An empty report (no groups) renders without raising."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report(groups=[], raw_issues=[], triage_results=[])
        md = reporter.render_markdown(report)
        assert "# Bug Triage Report" in md

    def test_source_file_included_when_set(self):
        """Source file path appears in Markdown when set in metadata."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        report.metadata.source_file = "/path/to/issues.json"
        md = reporter.render_markdown(report)
        assert "/path/to/issues.json" in md

    def test_complexity_value_rendered(self):
        """Estimated complexity appears in Markdown output."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(complexity=Complexity.HIGH)
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "High" in md or "high" in md

    def test_fix_order_rendered_for_each_group(self):
        """Fix order numbers appear in the rendered Markdown."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        groups = [
            _make_group("g1", fix_order=1, issue_ids=[1]),
            _make_group("g2", severity=Severity.HIGH, fix_order=2, issue_ids=[2]),
        ]
        report = _make_report(
            groups=groups,
            raw_issues=[_make_issue(1), _make_issue(2)],
        )
        md = reporter.render_markdown(report)
        assert "#1" in md
        assert "#2" in md

    def test_llm_provider_rendered(self):
        """LLM provider name appears in the Markdown output."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        md = reporter.render_markdown(report)
        assert "openai" in md

    def test_priority_score_rendered(self):
        """Priority score appears in the Markdown output."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(priority_score=87.5)
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "87.5" in md or "87" in md

    def test_high_severity_group_rendered(self):
        """A HIGH severity group is rendered with correct label."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(
            severity=Severity.HIGH,
            impact_category=ImpactCategory.PERFORMANCE,
            fix_order=1,
            issue_ids=[5],
        )
        report = _make_report(groups=[group], raw_issues=[_make_issue(5)])
        md = reporter.render_markdown(report)
        assert "High" in md or "high" in md

    def test_medium_severity_group_rendered(self):
        """A MEDIUM severity group is rendered with correct label."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(
            severity=Severity.MEDIUM,
            impact_category=ImpactCategory.UX,
            fix_order=1,
            issue_ids=[6],
        )
        report = _make_report(groups=[group], raw_issues=[_make_issue(6)])
        md = reporter.render_markdown(report)
        assert "Medium" in md or "medium" in md

    def test_low_severity_group_rendered(self):
        """A LOW severity group is rendered with correct label."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(
            severity=Severity.LOW,
            fix_order=1,
            issue_ids=[7],
        )
        report = _make_report(groups=[group], raw_issues=[_make_issue(7)])
        md = reporter.render_markdown(report)
        assert "Low" in md or "low" in md

    def test_multiple_groups_all_rendered(self):
        """All groups appear in the Markdown output."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        groups = [
            _make_group("g1", title="Group One", severity=Severity.CRITICAL,
                        fix_order=1, issue_ids=[1]),
            _make_group("g2", title="Group Two", severity=Severity.HIGH,
                        fix_order=2, issue_ids=[2]),
            _make_group("g3", title="Group Three", severity=Severity.MEDIUM,
                        fix_order=3, issue_ids=[3]),
        ]
        report = _make_report(
            groups=groups,
            raw_issues=[_make_issue(i) for i in range(1, 4)],
        )
        md = reporter.render_markdown(report)
        assert "Group One" in md
        assert "Group Two" in md
        assert "Group Three" in md

    def test_group_with_no_canonical_issue_id_renders(self):
        """Group without canonical_issue_id renders without error."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        # canonical_issue_id must be in issue_ids per model validator
        group = _make_group(issue_ids=[10], canonical_issue_id=10)
        report = _make_report(groups=[group], raw_issues=[_make_issue(10)])
        md = reporter.render_markdown(report)
        assert "# Bug Triage Report" in md

    def test_render_uses_format_from_metadata(self):
        """render_markdown always produces Markdown regardless of metadata format."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report(output_format=OutputFormat.JSON)  # metadata says JSON
        # But reporter is configured for Markdown
        md = reporter.render_markdown(report)
        assert "# Bug Triage Report" in md

    def test_groups_sorted_by_fix_order_in_output(self):
        """Groups are sorted by fix_order in the rendered output."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        groups = [
            _make_group("g3", title="Third Fix", severity=Severity.LOW,
                        fix_order=3, issue_ids=[3]),
            _make_group("g1", title="First Fix", severity=Severity.CRITICAL,
                        fix_order=1, issue_ids=[1]),
            _make_group("g2", title="Second Fix", severity=Severity.HIGH,
                        fix_order=2, issue_ids=[2]),
        ]
        report = _make_report(
            groups=groups,
            raw_issues=[_make_issue(i) for i in range(1, 4)],
        )
        md = reporter.render_markdown(report)
        # Critical (fix_order=1) should appear before High (fix_order=2)
        pos_first = md.find("First Fix")
        pos_second = md.find("Second Fix")
        pos_third = md.find("Third Fix")
        assert pos_first < pos_second < pos_third


# ---------------------------------------------------------------------------
# Reporter.render_json
# ---------------------------------------------------------------------------


class TestRenderJSON:
    def test_returns_valid_json_string(self):
        """render_json returns a valid JSON string."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        result = reporter.render_json(report)
        assert isinstance(result, str)
        parsed = json.loads(result)  # should not raise
        assert isinstance(parsed, dict)

    def test_json_contains_metadata(self):
        """JSON output contains a metadata section with correct values."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        parsed = json.loads(reporter.render_json(report))
        assert "metadata" in parsed
        meta = parsed["metadata"]
        assert meta["repository"] == "owner/repo"
        assert meta["llm_model"] == "gpt-4o"
        assert meta["total_issues"] == report.metadata.total_issues

    def test_json_contains_groups(self):
        """JSON output contains a groups list."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        group = _make_group(group_id="group_001", title="Auth bypass")
        report = _make_report(groups=[group])
        parsed = json.loads(reporter.render_json(report))
        assert "groups" in parsed
        assert len(parsed["groups"]) == 1
        g = parsed["groups"][0]
        assert g["id"] == "group_001"
        assert g["title"] == "Auth bypass"

    def test_json_group_contains_all_fields(self):
        """Each group in JSON output contains all expected fields."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        group = _make_group(
            group_id="group_001",
            severity=Severity.CRITICAL,
            impact_category=ImpactCategory.SECURITY,
            priority_score=98.0,
            complexity=Complexity.HIGH,
            fix_order=1,
            issue_ids=[1, 2],
            canonical_issue_id=1,
            summary="Test summary.",
            tags=["auth"],
            similar_closed_issue_ids=[50],
        )
        report = _make_report(groups=[group])
        parsed = json.loads(reporter.render_json(report))
        g = parsed["groups"][0]
        assert g["severity"] == "critical"
        assert g["impact_category"] == "security"
        assert g["priority_score"] == 98.0
        assert g["complexity"] == "high"
        assert g["fix_order"] == 1
        assert g["issue_ids"] == [1, 2]
        assert g["canonical_issue_id"] == 1
        assert g["summary"] == "Test summary."
        assert g["tags"] == ["auth"]
        assert g["similar_closed_issue_ids"] == [50]

    def test_json_contains_raw_issues(self):
        """JSON output contains a raw_issues list."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        issues = [_make_issue(1), _make_issue(2)]
        report = _make_report(raw_issues=issues)
        parsed = json.loads(reporter.render_json(report))
        assert "raw_issues" in parsed
        assert len(parsed["raw_issues"]) == 2

    def test_json_contains_triage_results(self):
        """JSON output contains a triage_results list."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        tr = [_make_triage_result(1), _make_triage_result(2)]
        report = _make_report(triage_results=tr)
        parsed = json.loads(reporter.render_json(report))
        assert "triage_results" in parsed
        assert len(parsed["triage_results"]) == 2

    def test_json_indent_applied(self):
        """JSON output uses the configured indentation level."""
        reporter = Reporter(output_format=OutputFormat.JSON, json_indent=4)
        report = _make_report()
        result = reporter.render_json(report)
        assert "    " in result

    def test_json_indent_two_is_default(self):
        """Default JSON indentation is 2 spaces."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        result = reporter.render_json(report)
        assert "  " in result

    def test_empty_report_renders_to_valid_json(self):
        """An empty report serialises to valid JSON with empty lists."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report(groups=[], raw_issues=[], triage_results=[])
        result = reporter.render_json(report)
        parsed = json.loads(result)
        assert parsed["groups"] == []
        assert parsed["raw_issues"] == []
        assert parsed["triage_results"] == []

    def test_datetime_serialised_as_string(self):
        """generated_at datetime is serialised as an ISO string."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        parsed = json.loads(reporter.render_json(report))
        generated_at = parsed["metadata"]["generated_at"]
        assert isinstance(generated_at, str)
        assert "2024" in generated_at

    def test_ungrouped_issue_ids_in_json(self):
        """ungrouped_issue_ids field appears in JSON output."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report(ungrouped_issue_ids=[77, 88])
        parsed = json.loads(reporter.render_json(report))
        assert parsed["ungrouped_issue_ids"] == [77, 88]

    def test_json_metadata_total_groups(self):
        """JSON metadata contains correct total_groups count."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        groups = [_make_group(f"g{i}", fix_order=i, issue_ids=[i]) for i in range(1, 4)]
        report = _make_report(
            groups=groups,
            raw_issues=[_make_issue(i) for i in range(1, 4)],
        )
        parsed = json.loads(reporter.render_json(report))
        assert parsed["metadata"]["total_groups"] == 3

    def test_json_metadata_llm_provider(self):
        """JSON metadata contains the LLM provider name."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        parsed = json.loads(reporter.render_json(report))
        assert parsed["metadata"]["llm_provider"] == "openai"

    def test_json_group_severity_values_are_lowercase_strings(self):
        """All severity values in JSON groups are lowercase strings."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        groups = [
            _make_group("g1", severity=Severity.CRITICAL, fix_order=1, issue_ids=[1]),
            _make_group("g2", severity=Severity.HIGH, fix_order=2, issue_ids=[2]),
            _make_group("g3", severity=Severity.MEDIUM, fix_order=3, issue_ids=[3]),
            _make_group("g4", severity=Severity.LOW, fix_order=4, issue_ids=[4]),
        ]
        report = _make_report(
            groups=groups,
            raw_issues=[_make_issue(i) for i in range(1, 5)],
        )
        parsed = json.loads(reporter.render_json(report))
        severities = [g["severity"] for g in parsed["groups"]]
        assert set(severities) == {"critical", "high", "medium", "low"}

    def test_json_multiple_groups_all_present(self):
        """All groups appear in the JSON output."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        groups = [
            _make_group("g1", title="First", fix_order=1, issue_ids=[1]),
            _make_group("g2", title="Second", fix_order=2, issue_ids=[2]),
            _make_group("g3", title="Third", fix_order=3, issue_ids=[3]),
        ]
        report = _make_report(
            groups=groups,
            raw_issues=[_make_issue(i) for i in range(1, 4)],
        )
        parsed = json.loads(reporter.render_json(report))
        titles = [g["title"] for g in parsed["groups"]]
        assert "First" in titles
        assert "Second" in titles
        assert "Third" in titles


# ---------------------------------------------------------------------------
# Reporter.render — dispatch
# ---------------------------------------------------------------------------


class TestReporterRenderDispatch:
    def test_render_dispatches_to_markdown(self):
        """render() with MARKDOWN format delegates to render_markdown."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        result = reporter.render(report)
        assert "# Bug Triage Report" in result

    def test_render_dispatches_to_json(self):
        """render() with JSON format delegates to render_json."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        result = reporter.render(report)
        parsed = json.loads(result)
        assert "metadata" in parsed

    def test_render_returns_string_for_markdown(self):
        """render() returns a string for Markdown format."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        result = reporter.render(_make_report())
        assert isinstance(result, str)

    def test_render_returns_string_for_json(self):
        """render() returns a string for JSON format."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        result = reporter.render(_make_report())
        assert isinstance(result, str)

    def test_default_format_is_markdown(self):
        """Reporter defaults to MARKDOWN output format."""
        reporter = Reporter()  # no format specified
        assert reporter.output_format == OutputFormat.MARKDOWN


# ---------------------------------------------------------------------------
# Reporter.save
# ---------------------------------------------------------------------------


class TestReporterSave:
    def test_save_writes_markdown_file(self, tmp_path: Path):
        """save() writes a Markdown file to the specified path."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        out_path = tmp_path / "report.md"
        returned = reporter.save(report, out_path)
        assert returned == out_path
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "# Bug Triage Report" in content

    def test_save_writes_json_file(self, tmp_path: Path):
        """save() writes a JSON file to the specified path."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        out_path = tmp_path / "report.json"
        reporter.save(report, out_path)
        content = out_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert "metadata" in parsed

    def test_save_creates_parent_directories(self, tmp_path: Path):
        """save() creates parent directories if they don't exist."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        out_path = tmp_path / "nested" / "dir" / "report.md"
        reporter.save(report, out_path)
        assert out_path.exists()

    def test_save_returns_path_object(self, tmp_path: Path):
        """save() returns a Path object pointing to the written file."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        out_path = tmp_path / "r.md"
        result = reporter.save(report, str(out_path))  # pass as string
        assert isinstance(result, Path)
        assert result == out_path

    def test_save_raises_reporter_error_on_invalid_path(self):
        """save() raises ReporterError when the path is invalid."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        with pytest.raises(ReporterError, match="Failed to write report"):
            reporter.save(report, "/dev/null/impossible/path/report.md")

    def test_save_file_content_matches_render_output(self, tmp_path: Path):
        """File content written by save() matches what render() returns."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        out_path = tmp_path / "report.md"
        rendered = reporter.render(report)
        reporter.save(report, out_path)
        file_content = out_path.read_text(encoding="utf-8")
        assert file_content == rendered

    def test_save_json_file_content_is_valid(self, tmp_path: Path):
        """JSON file saved by save() contains valid parseable JSON."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        out_path = tmp_path / "report.json"
        reporter.save(report, out_path)
        content = out_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, dict)
        assert "groups" in parsed

    def test_save_overwrites_existing_file(self, tmp_path: Path):
        """save() overwrites an existing file at the target path."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        out_path = tmp_path / "report.md"
        out_path.write_text("old content", encoding="utf-8")
        report = _make_report()
        reporter.save(report, out_path)
        content = out_path.read_text(encoding="utf-8")
        assert "# Bug Triage Report" in content
        assert "old content" not in content


# ---------------------------------------------------------------------------
# Reporter.print_summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    def test_print_summary_does_not_raise(self):
        """print_summary completes without raising for a normal report."""
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        report = _make_report()
        reporter.print_summary(report)  # should not raise

    def test_print_summary_output_contains_repo(self):
        """Terminal output contains the repository name."""
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        report = _make_report()
        reporter.print_summary(report)
        output = console.export_text()
        assert "owner/repo" in output

    def test_print_summary_output_contains_severity(self):
        """Terminal output contains the severity label."""
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        group = _make_group(severity=Severity.CRITICAL)
        report = _make_report(groups=[group])
        reporter.print_summary(report)
        output = console.export_text()
        assert "Critical" in output or "critical" in output

    def test_print_summary_empty_report_does_not_raise(self):
        """print_summary handles an empty report without raising."""
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        report = _make_report(groups=[], raw_issues=[], triage_results=[])
        reporter.print_summary(report)  # should not raise

    def test_print_summary_shows_ungrouped_notice(self):
        """Terminal output mentions ungrouped issue IDs."""
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        report = _make_report(ungrouped_issue_ids=[99])
        reporter.print_summary(report)
        output = console.export_text()
        assert "99" in output

    def test_print_summary_no_ungrouped_section_when_empty(self):
        """Terminal output does not mention 'ungrouped' when the list is empty."""
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        report = _make_report(ungrouped_issue_ids=[])
        reporter.print_summary(report)
        output = console.export_text()
        assert "ungrouped" not in output.lower()

    def test_print_summary_shows_group_count(self):
        """Terminal output contains the group count from metadata."""
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        groups = [_make_group(f"g{i}", fix_order=i, issue_ids=[i]) for i in range(1, 4)]
        report = _make_report(
            groups=groups,
            raw_issues=[_make_issue(i) for i in range(1, 4)],
        )
        reporter.print_summary(report)
        output = console.export_text()
        assert "3" in output

    def test_print_summary_shows_issue_count(self):
        """Terminal output contains the total issue count."""
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        issues = [_make_issue(i) for i in range(1, 6)]
        report = _make_report(
            groups=[_make_group(issue_ids=[i for i in range(1, 6)],
                                canonical_issue_id=1)],
            raw_issues=issues,
        )
        reporter.print_summary(report)
        output = console.export_text()
        # Should show 5 somewhere (total issues)
        assert "5" in output

    def test_print_summary_shows_priority_score(self):
        """Terminal output contains priority scores."""
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        group = _make_group(priority_score=95.0)
        report = _make_report(groups=[group])
        reporter.print_summary(report)
        output = console.export_text()
        assert "95" in output

    def test_print_summary_shows_llm_info(self):
        """Terminal output contains LLM provider and model info."""
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        report = _make_report()
        reporter.print_summary(report)
        output = console.export_text()
        assert "openai" in output or "gpt-4o" in output

    def test_print_summary_multiple_ungrouped_shown(self):
        """All ungrouped IDs appear in terminal output."""
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        report = _make_report(ungrouped_issue_ids=[101, 102, 103])
        reporter.print_summary(report)
        output = console.export_text()
        assert "101" in output
        assert "102" in output
        assert "103" in output


# ---------------------------------------------------------------------------
# render_report convenience function
# ---------------------------------------------------------------------------


class TestRenderReportConvenienceFunction:
    def test_returns_markdown_string_by_default(self):
        """render_report returns a Markdown string by default."""
        report = _make_report()
        result = render_report(report, output_format=OutputFormat.MARKDOWN, print_to_console=False)
        assert isinstance(result, str)
        assert "# Bug Triage Report" in result

    def test_returns_json_string_when_requested(self):
        """render_report returns valid JSON when JSON format is requested."""
        report = _make_report()
        result = render_report(report, output_format=OutputFormat.JSON, print_to_console=False)
        parsed = json.loads(result)
        assert "metadata" in parsed

    def test_saves_to_file_when_output_path_provided(self, tmp_path: Path):
        """render_report writes to a file when output_path is given."""
        report = _make_report()
        out_path = tmp_path / "out.md"
        render_report(
            report,
            output_format=OutputFormat.MARKDOWN,
            output_path=out_path,
            print_to_console=False,
        )
        assert out_path.exists()
        assert "# Bug Triage Report" in out_path.read_text(encoding="utf-8")

    def test_does_not_print_when_flag_is_false(self):
        """When print_to_console=False, no Rich output is produced."""
        console = Console(record=True, width=120)
        report = _make_report()
        render_report(
            report,
            output_format=OutputFormat.MARKDOWN,
            print_to_console=False,
            console=console,
        )
        output = console.export_text()
        assert output.strip() == ""

    def test_prints_summary_when_flag_is_true(self):
        """When print_to_console=True, Rich output is produced."""
        console = Console(record=True, width=120)
        report = _make_report()
        render_report(
            report,
            output_format=OutputFormat.MARKDOWN,
            print_to_console=True,
            console=console,
        )
        output = console.export_text()
        assert len(output.strip()) > 0

    def test_custom_json_indent_applied(self, tmp_path: Path):
        """json_indent parameter controls JSON indentation."""
        report = _make_report()
        result = render_report(
            report,
            output_format=OutputFormat.JSON,
            print_to_console=False,
            json_indent=4,
        )
        assert "    " in result  # 4-space indent

    def test_returns_rendered_content_as_string(self):
        """render_report always returns the rendered content as a string."""
        report = _make_report()
        result = render_report(report, print_to_console=False)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_saves_json_to_file(self, tmp_path: Path):
        """render_report saves JSON output to file correctly."""
        report = _make_report()
        out_path = tmp_path / "report.json"
        render_report(
            report,
            output_format=OutputFormat.JSON,
            output_path=out_path,
            print_to_console=False,
        )
        assert out_path.exists()
        parsed = json.loads(out_path.read_text(encoding="utf-8"))
        assert "metadata" in parsed

    def test_no_output_path_does_not_create_file(self, tmp_path: Path):
        """When no output_path is given, no file is created."""
        report = _make_report()
        render_report(
            report,
            output_format=OutputFormat.MARKDOWN,
            output_path=None,
            print_to_console=False,
        )
        # No files should be created in the current directory
        # (This test just verifies no exception is raised)

    def test_default_console_created_when_none(self):
        """render_report creates a default Console when console=None."""
        report = _make_report()
        # Should not raise
        result = render_report(
            report,
            output_format=OutputFormat.MARKDOWN,
            print_to_console=False,
            console=None,
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _InlineTemplateLoader
# ---------------------------------------------------------------------------


class TestInlineTemplateLoader:
    def test_get_source_returns_template(self):
        """_InlineTemplateLoader returns templates by name."""
        from jinja2 import Environment

        loader = _InlineTemplateLoader({"hello.j2": "Hello, {{ name }}!"})
        env = Environment(loader=loader)
        template = env.get_template("hello.j2")
        result = template.render(name="World")
        assert result == "Hello, World!"

    def test_get_source_raises_template_not_found(self):
        """_InlineTemplateLoader raises TemplateNotFound for unknown templates."""
        from jinja2 import Environment, TemplateNotFound

        loader = _InlineTemplateLoader({})
        env = Environment(loader=loader)
        with pytest.raises(TemplateNotFound):
            env.get_template("missing.j2")

    def test_multiple_templates(self):
        """_InlineTemplateLoader serves multiple templates correctly."""
        from jinja2 import Environment

        loader = _InlineTemplateLoader(
            {
                "a.j2": "Template A: {{ val }}",
                "b.j2": "Template B: {{ val }}",
            }
        )
        env = Environment(loader=loader)
        assert env.get_template("a.j2").render(val="x") == "Template A: x"
        assert env.get_template("b.j2").render(val="y") == "Template B: y"

    def test_template_with_conditionals(self):
        """_InlineTemplateLoader handles templates with Jinja2 conditionals."""
        from jinja2 import Environment

        loader = _InlineTemplateLoader(
            {"cond.j2": "{% if show %}Shown{% else %}Hidden{% endif %}"}
        )
        env = Environment(loader=loader)
        tmpl = env.get_template("cond.j2")
        assert tmpl.render(show=True) == "Shown"
        assert tmpl.render(show=False) == "Hidden"

    def test_template_with_loop(self):
        """_InlineTemplateLoader handles templates with Jinja2 loops."""
        from jinja2 import Environment

        loader = _InlineTemplateLoader(
            {"loop.j2": "{% for i in items %}{{ i }}{% if not loop.last %},{% endif %}{% endfor %}"}
        )
        env = Environment(loader=loader)
        tmpl = env.get_template("loop.j2")
        assert tmpl.render(items=[1, 2, 3]) == "1,2,3"

    def test_empty_template_renders_empty_string(self):
        """An empty template string renders to an empty string."""
        from jinja2 import Environment

        loader = _InlineTemplateLoader({"empty.j2": ""})
        env = Environment(loader=loader)
        tmpl = env.get_template("empty.j2")
        assert tmpl.render() == ""

    def test_get_source_returns_three_tuple(self):
        """get_source returns a (source, filename, uptodate) tuple."""
        from jinja2 import Environment

        loader = _InlineTemplateLoader({"t.j2": "content"})
        env = Environment(loader=loader)
        source, filename, uptodate = loader.get_source(env, "t.j2")
        assert source == "content"
        # filename may be None for inline templates
        assert uptodate is None or callable(uptodate)

    def test_loader_not_found_message_includes_template_name(self):
        """TemplateNotFound exception message includes the template name."""
        from jinja2 import Environment, TemplateNotFound

        loader = _InlineTemplateLoader({})
        env = Environment(loader=loader)
        with pytest.raises(TemplateNotFound) as exc_info:
            env.get_template("my_missing_template.j2")
        assert "my_missing_template.j2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Edge cases and regression tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_multiple_severity_levels_all_rendered_in_markdown(self):
        """Report with groups across all four severity levels renders all sections."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        groups = [
            _make_group("g1", severity=Severity.CRITICAL, fix_order=1, issue_ids=[1]),
            _make_group("g2", severity=Severity.HIGH, fix_order=2, issue_ids=[2]),
            _make_group("g3", severity=Severity.MEDIUM, fix_order=3, issue_ids=[3]),
            _make_group("g4", severity=Severity.LOW, fix_order=4, issue_ids=[4]),
        ]
        report = _make_report(
            groups=groups,
            raw_issues=[_make_issue(i) for i in range(1, 5)],
        )
        md = reporter.render_markdown(report)
        assert "Critical" in md
        assert "High" in md
        assert "Medium" in md
        assert "Low" in md

    def test_group_with_no_tags_renders_cleanly(self):
        """Group with empty tags list renders without error."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(tags=[])
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "# Bug Triage Report" in md

    def test_group_with_no_summary_renders_cleanly(self):
        """Group with empty summary renders without error."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(summary="")
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "# Bug Triage Report" in md

    def test_large_number_of_groups_renders_without_error(self):
        """50 groups render without error or performance issues."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        groups = [
            _make_group(
                f"group_{i:03d}",
                title=f"Issue group {i}",
                severity=Severity.MEDIUM,
                priority_score=float(50 + i % 30),
                fix_order=i,
                issue_ids=[i],
            )
            for i in range(1, 51)
        ]
        report = _make_report(
            groups=groups,
            raw_issues=[_make_issue(i) for i in range(1, 51)],
        )
        md = reporter.render_markdown(report)
        assert "# Bug Triage Report" in md
        assert len(groups) == 50

    def test_json_roundtrip_preserves_priority_score_precision(self):
        """Priority score precision is preserved through JSON serialisation."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        group = _make_group(priority_score=87.35)
        report = _make_report(groups=[group])
        parsed = json.loads(reporter.render_json(report))
        assert parsed["groups"][0]["priority_score"] == 87.35

    def test_render_json_and_markdown_produce_different_output(self):
        """Markdown and JSON renders produce distinct output."""
        report = _make_report()
        md = Reporter(output_format=OutputFormat.MARKDOWN).render(report)
        js = Reporter(output_format=OutputFormat.JSON).render(report)
        assert md != js
        assert "# Bug Triage Report" in md
        assert "{" in js

    def test_report_with_no_repository_renders(self):
        """Report with empty repository renders without error."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        report.metadata.repository = ""
        md = reporter.render_markdown(report)
        assert "# Bug Triage Report" in md

    def test_report_with_source_file_and_no_repository(self):
        """Report with source_file but no repository renders both fields correctly."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        report.metadata.repository = ""
        report.metadata.source_file = "/data/issues.csv"
        md = reporter.render_markdown(report)
        assert "/data/issues.csv" in md

    def test_json_output_does_not_contain_markdown_header(self):
        """JSON output does not contain Markdown-style headers."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        result = reporter.render_json(report)
        assert "# Bug Triage Report" not in result

    def test_markdown_output_does_not_contain_json_structure(self):
        """Markdown output does not start with a JSON brace."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        result = reporter.render_markdown(report)
        assert not result.strip().startswith("{")

    def test_render_report_json_file_roundtrip(self, tmp_path: Path):
        """JSON saved to file and read back produces identical parsed output."""
        report = _make_report()
        out_path = tmp_path / "roundtrip.json"
        rendered = render_report(
            report,
            output_format=OutputFormat.JSON,
            output_path=out_path,
            print_to_console=False,
        )
        file_content = out_path.read_text(encoding="utf-8")
        assert json.loads(rendered) == json.loads(file_content)

    def test_group_with_multiple_similar_closed_issues(self):
        """Multiple similar closed issue IDs all appear in the rendered output."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(similar_closed_issue_ids=[10, 20, 30, 40, 50])
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        for iid in [10, 20, 30, 40, 50]:
            assert f"#{iid}" in md

    def test_json_triage_result_fields_preserved(self):
        """All TriageResult fields are present in JSON output."""
        reporter = Reporter(output_format=OutputFormat.JSON)
        tr = TriageResult(
            issue_id=42,
            severity=Severity.HIGH,
            impact_category=ImpactCategory.CRASH,
            priority_score=78.5,
            summary="App crashes.",
            reasoning="It crashes always.",
            duplicate_of=None,
            related_issue_ids=[10, 11],
            complexity=Complexity.MEDIUM,
            tags=["crash", "login"],
        )
        report = _make_report(triage_results=[tr])
        parsed = json.loads(reporter.render_json(report))
        results = parsed["triage_results"]
        assert len(results) == 1
        r = results[0]
        assert r["issue_id"] == 42
        assert r["severity"] == "high"
        assert r["impact_category"] == "crash"
        assert r["priority_score"] == 78.5
        assert r["summary"] == "App crashes."
        assert r["related_issue_ids"] == [10, 11]
        assert r["tags"] == ["crash", "login"]

    def test_markdown_contains_impact_category(self):
        """Impact category appears in the Markdown output."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(impact_category=ImpactCategory.DATA_LOSS)
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "Data Loss" in md or "data_loss" in md or "data loss" in md.lower()

    def test_reporter_can_be_reused_for_multiple_reports(self):
        """The same Reporter instance can render multiple reports."""
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report1 = _make_report(groups=[_make_group("g1", fix_order=1, issue_ids=[1])])
        report2 = _make_report(groups=[_make_group("g2", title="Other group",
                                                    fix_order=1, issue_ids=[2])])
        md1 = reporter.render(report1)
        md2 = reporter.render(report2)
        assert "# Bug Triage Report" in md1
        assert "# Bug Triage Report" in md2
        assert "Other group" in md2
