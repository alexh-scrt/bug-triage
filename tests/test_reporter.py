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
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        result = reporter.render_markdown(report)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_header(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        md = reporter.render_markdown(report)
        assert "# Bug Triage Report" in md

    def test_contains_generated_timestamp(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        md = reporter.render_markdown(report)
        assert "2024-01-15 14:32:00" in md

    def test_contains_repository(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        md = reporter.render_markdown(report)
        assert "owner/repo" in md

    def test_contains_llm_model(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        md = reporter.render_markdown(report)
        assert "gpt-4o" in md

    def test_contains_total_issues(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        md = reporter.render_markdown(report)
        assert str(report.metadata.total_issues) in md

    def test_contains_group_title(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(title="Authentication bypass vulnerability")
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "Authentication bypass vulnerability" in md

    def test_contains_severity_label(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(severity=Severity.CRITICAL)
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "Critical" in md or "critical" in md

    def test_contains_issue_ids(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(issue_ids=[42, 43])
        report = _make_report(groups=[group], raw_issues=[_make_issue(42), _make_issue(43)])
        md = reporter.render_markdown(report)
        assert "#42" in md
        assert "#43" in md

    def test_contains_summary(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(summary="JWT tokens can be bypassed.")
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "JWT tokens can be bypassed." in md

    def test_contains_tags(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(tags=["auth", "security"])
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "auth" in md
        assert "security" in md

    def test_similar_closed_issues_included(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(similar_closed_issue_ids=[189, 190])
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "#189" in md
        assert "#190" in md

    def test_ungrouped_issues_section_present_when_non_empty(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report(ungrouped_issue_ids=[99, 100])
        md = reporter.render_markdown(report)
        assert "Ungrouped" in md or "ungrouped" in md
        assert "#99" in md
        assert "#100" in md

    def test_ungrouped_section_absent_when_empty(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report(ungrouped_issue_ids=[])
        md = reporter.render_markdown(report)
        # Should not mention ungrouped when there are none
        assert "Ungrouped" not in md

    def test_severity_sections_ordered_critical_first(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        groups = [
            _make_group("g1", severity=Severity.LOW, fix_order=4, issue_ids=[4]),
            _make_group("g2", severity=Severity.CRITICAL, fix_order=1, issue_ids=[1]),
            _make_group("g3", severity=Severity.MEDIUM, fix_order=3, issue_ids=[3]),
            _make_group("g4", severity=Severity.HIGH, fix_order=2, issue_ids=[2]),
        ]
        report = _make_report(groups=groups, raw_issues=[_make_issue(i) for i in range(1, 5)])
        md = reporter.render_markdown(report)
        # Critical section should appear before High, High before Medium, etc.
        pos_critical = md.find("Critical")
        pos_high = md.find("High")
        pos_medium = md.find("Medium")
        pos_low = md.find("Low")
        assert pos_critical < pos_high < pos_medium < pos_low

    def test_empty_report_renders_without_error(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report(groups=[], raw_issues=[], triage_results=[])
        md = reporter.render_markdown(report)
        assert "# Bug Triage Report" in md

    def test_source_file_included_when_set(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        report.metadata.source_file = "/path/to/issues.json"
        md = reporter.render_markdown(report)
        assert "/path/to/issues.json" in md

    def test_complexity_value_rendered(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(complexity=Complexity.HIGH)
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "High" in md or "high" in md

    def test_fix_order_rendered_for_each_group(self):
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
        assert "Fix #1" in md or "#1" in md
        assert "Fix #2" in md or "#2" in md


# ---------------------------------------------------------------------------
# Reporter.render_json
# ---------------------------------------------------------------------------


class TestRenderJSON:
    def test_returns_valid_json_string(self):
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        result = reporter.render_json(report)
        assert isinstance(result, str)
        parsed = json.loads(result)  # should not raise
        assert isinstance(parsed, dict)

    def test_json_contains_metadata(self):
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        parsed = json.loads(reporter.render_json(report))
        assert "metadata" in parsed
        meta = parsed["metadata"]
        assert meta["repository"] == "owner/repo"
        assert meta["llm_model"] == "gpt-4o"
        assert meta["total_issues"] == report.metadata.total_issues

    def test_json_contains_groups(self):
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
        reporter = Reporter(output_format=OutputFormat.JSON)
        issues = [_make_issue(1), _make_issue(2)]
        report = _make_report(raw_issues=issues)
        parsed = json.loads(reporter.render_json(report))
        assert "raw_issues" in parsed
        assert len(parsed["raw_issues"]) == 2

    def test_json_contains_triage_results(self):
        reporter = Reporter(output_format=OutputFormat.JSON)
        tr = [_make_triage_result(1), _make_triage_result(2)]
        report = _make_report(triage_results=tr)
        parsed = json.loads(reporter.render_json(report))
        assert "triage_results" in parsed
        assert len(parsed["triage_results"]) == 2

    def test_json_indent_applied(self):
        reporter = Reporter(output_format=OutputFormat.JSON, json_indent=4)
        report = _make_report()
        result = reporter.render_json(report)
        # 4-space indented JSON should have 4-space lines
        assert "    " in result

    def test_empty_report_renders_to_valid_json(self):
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report(groups=[], raw_issues=[], triage_results=[])
        result = reporter.render_json(report)
        parsed = json.loads(result)
        assert parsed["groups"] == []
        assert parsed["raw_issues"] == []
        assert parsed["triage_results"] == []

    def test_datetime_serialised_as_string(self):
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        parsed = json.loads(reporter.render_json(report))
        generated_at = parsed["metadata"]["generated_at"]
        assert isinstance(generated_at, str)
        # Should be ISO-8601 format
        assert "2024" in generated_at

    def test_ungrouped_issue_ids_in_json(self):
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report(ungrouped_issue_ids=[77, 88])
        parsed = json.loads(reporter.render_json(report))
        assert parsed["ungrouped_issue_ids"] == [77, 88]


# ---------------------------------------------------------------------------
# Reporter.render — dispatch
# ---------------------------------------------------------------------------


class TestReporterRenderDispatch:
    def test_render_dispatches_to_markdown(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        result = reporter.render(report)
        assert "# Bug Triage Report" in result

    def test_render_dispatches_to_json(self):
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        result = reporter.render(report)
        parsed = json.loads(result)
        assert "metadata" in parsed


# ---------------------------------------------------------------------------
# Reporter.save
# ---------------------------------------------------------------------------


class TestReporterSave:
    def test_save_writes_markdown_file(self, tmp_path: Path):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        out_path = tmp_path / "report.md"
        returned = reporter.save(report, out_path)
        assert returned == out_path
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "# Bug Triage Report" in content

    def test_save_writes_json_file(self, tmp_path: Path):
        reporter = Reporter(output_format=OutputFormat.JSON)
        report = _make_report()
        out_path = tmp_path / "report.json"
        reporter.save(report, out_path)
        content = out_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert "metadata" in parsed

    def test_save_creates_parent_directories(self, tmp_path: Path):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        out_path = tmp_path / "nested" / "dir" / "report.md"
        reporter.save(report, out_path)
        assert out_path.exists()

    def test_save_returns_path_object(self, tmp_path: Path):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        out_path = tmp_path / "r.md"
        result = reporter.save(report, str(out_path))  # pass as string
        assert isinstance(result, Path)
        assert result == out_path

    def test_save_raises_reporter_error_on_invalid_path(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        report = _make_report()
        # Use a path that cannot be created (root-level impossible path on
        # most systems when running as a non-root user).
        with pytest.raises(ReporterError, match="Failed to write report"):
            reporter.save(report, "/dev/null/impossible/path/report.md")


# ---------------------------------------------------------------------------
# Reporter.print_summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    def test_print_summary_does_not_raise(self):
        """print_summary should complete without raising for a normal report."""
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        report = _make_report()
        reporter.print_summary(report)  # should not raise

    def test_print_summary_output_contains_repo(self):
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        report = _make_report()
        reporter.print_summary(report)
        output = console.export_text()
        assert "owner/repo" in output

    def test_print_summary_output_contains_severity(self):
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        group = _make_group(severity=Severity.CRITICAL)
        report = _make_report(groups=[group])
        reporter.print_summary(report)
        output = console.export_text()
        assert "Critical" in output or "critical" in output

    def test_print_summary_empty_report_does_not_raise(self):
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        report = _make_report(groups=[], raw_issues=[], triage_results=[])
        reporter.print_summary(report)  # should not raise

    def test_print_summary_shows_ungrouped_notice(self):
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        report = _make_report(ungrouped_issue_ids=[99])
        reporter.print_summary(report)
        output = console.export_text()
        assert "99" in output

    def test_print_summary_no_ungrouped_section_when_empty(self):
        console = Console(record=True, width=120)
        reporter = Reporter(console=console)
        report = _make_report(ungrouped_issue_ids=[])
        reporter.print_summary(report)
        output = console.export_text()
        # Should not mention "ungrouped" when the list is empty
        assert "ungrouped" not in output.lower()


# ---------------------------------------------------------------------------
# render_report convenience function
# ---------------------------------------------------------------------------


class TestRenderReportConvenienceFunction:
    def test_returns_markdown_string_by_default(self):
        report = _make_report()
        result = render_report(report, output_format=OutputFormat.MARKDOWN, print_to_console=False)
        assert isinstance(result, str)
        assert "# Bug Triage Report" in result

    def test_returns_json_string_when_requested(self):
        report = _make_report()
        result = render_report(report, output_format=OutputFormat.JSON, print_to_console=False)
        parsed = json.loads(result)
        assert "metadata" in parsed

    def test_saves_to_file_when_output_path_provided(self, tmp_path: Path):
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
        """When print_to_console=False, no Rich output should be produced."""
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
        console = Console(record=True, width=120)
        report = _make_report()
        render_report(
            report,
            output_format=OutputFormat.MARKDOWN,
            print_to_console=True,
            console=console,
        )
        output = console.export_text()
        # Should have produced some output
        assert len(output.strip()) > 0

    def test_custom_json_indent_applied(self, tmp_path: Path):
        report = _make_report()
        result = render_report(
            report,
            output_format=OutputFormat.JSON,
            print_to_console=False,
            json_indent=4,
        )
        assert "    " in result  # 4-space indent


# ---------------------------------------------------------------------------
# _InlineTemplateLoader
# ---------------------------------------------------------------------------


class TestInlineTemplateLoader:
    def test_get_source_returns_template(self):
        from jinja2 import Environment

        loader = _InlineTemplateLoader({"hello.j2": "Hello, {{ name }}!"})
        env = Environment(loader=loader)
        template = env.get_template("hello.j2")
        result = template.render(name="World")
        assert result == "Hello, World!"

    def test_get_source_raises_template_not_found(self):
        from jinja2 import Environment, TemplateNotFound

        loader = _InlineTemplateLoader({})
        env = Environment(loader=loader)
        with pytest.raises(TemplateNotFound):
            env.get_template("missing.j2")

    def test_multiple_templates(self):
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


# ---------------------------------------------------------------------------
# Edge cases and regression tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_multiple_severity_levels_all_rendered_in_markdown(self):
        """Report with groups across all four severity levels."""
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
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(tags=[])
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "# Bug Triage Report" in md

    def test_group_with_no_summary_renders_cleanly(self):
        reporter = Reporter(output_format=OutputFormat.MARKDOWN)
        group = _make_group(summary="")
        report = _make_report(groups=[group])
        md = reporter.render_markdown(report)
        assert "# Bug Triage Report" in md

    def test_large_number_of_groups_renders_without_error(self):
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
        reporter = Reporter(output_format=OutputFormat.JSON)
        group = _make_group(priority_score=87.35)
        report = _make_report(groups=[group])
        parsed = json.loads(reporter.render_json(report))
        assert parsed["groups"][0]["priority_score"] == 87.35

    def test_render_json_and_markdown_produce_different_output(self):
        report = _make_report()
        md = Reporter(output_format=OutputFormat.MARKDOWN).render(report)
        js = Reporter(output_format=OutputFormat.JSON).render(report)
        assert md != js
        assert "# Bug Triage Report" in md
        assert "{" in js
