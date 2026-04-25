"""Unit tests for bug_triage.triage — classification, deduplication, and priority scoring."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from bug_triage.models import (
    Complexity,
    ImpactCategory,
    Issue,
    IssueGroup,
    LLMProvider,
    OutputFormat,
    Severity,
    TriageResult,
)
from bug_triage.triage import (
    TriageEngine,
    TriageError,
    _UnionFind,
    _chunk,
    _default_triage_result,
    _dict_to_triage_result,
    _max_severity,
    _strip_markdown_fences,
    triage_issues,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_issue(
    id: int = 1,
    title: str = "Test issue",
    body: str = "Issue body text.",
    labels: list[str] | None = None,
    severity_hint: str = "medium",
) -> Issue:
    """Create a minimal Issue for testing."""
    return Issue(
        id=id,
        title=title,
        body=body,
        labels=labels or ["bug"],
        created_at=datetime(2024, 1, 10),
        updated_at=datetime(2024, 1, 11),
        url=f"https://github.com/example/repo/issues/{id}",
        author="alice",
        state="open",
        comments_count=0,
        repository="example/repo",
    )


def _make_triage_result(
    issue_id: int = 1,
    severity: Severity = Severity.MEDIUM,
    impact_category: ImpactCategory = ImpactCategory.OTHER,
    priority_score: float = 50.0,
    duplicate_of: int | None = None,
    related_issue_ids: list[int] | None = None,
    complexity: Complexity = Complexity.MEDIUM,
    tags: list[str] | None = None,
    summary: str = "A test summary.",
) -> TriageResult:
    """Create a minimal TriageResult for testing."""
    return TriageResult(
        issue_id=issue_id,
        severity=severity,
        impact_category=impact_category,
        priority_score=priority_score,
        summary=summary,
        reasoning="Test reasoning.",
        duplicate_of=duplicate_of,
        related_issue_ids=related_issue_ids or [],
        complexity=complexity,
        tags=tags or [],
    )


def _make_llm_client(response: str = "[]") -> MagicMock:
    """Return a mock LLMClient that returns `response` for any call."""
    client = MagicMock()
    client.provider = LLMProvider.OPENAI
    client.model = "gpt-4o"
    client.render_and_complete.return_value = response
    return client


# ---------------------------------------------------------------------------
# _strip_markdown_fences
# ---------------------------------------------------------------------------


class TestStripMarkdownFences:
    def test_strips_json_fences(self):
        text = "```json\n[{\"key\": 1}]\n```"
        assert _strip_markdown_fences(text) == '[{"key": 1}]'

    def test_strips_plain_fences(self):
        text = "```\n[1, 2, 3]\n```"
        assert _strip_markdown_fences(text) == "[1, 2, 3]"

    def test_passthrough_when_no_fences(self):
        text = '[{"key": 1}]'
        assert _strip_markdown_fences(text) == text

    def test_strips_whitespace(self):
        text = "  [1, 2]  "
        assert _strip_markdown_fences(text) == "[1, 2]"

    def test_strips_fences_with_language_tag(self):
        text = "```python\n[1, 2]\n```"
        assert _strip_markdown_fences(text) == "[1, 2]"

    def test_empty_string_passthrough(self):
        assert _strip_markdown_fences("") == ""

    def test_strips_nested_content_with_newlines(self):
        inner = '[\n  {"a": 1},\n  {"b": 2}\n]'
        text = f"```json\n{inner}\n```"
        result = _strip_markdown_fences(text)
        # Should be parseable as JSON after stripping
        parsed = json.loads(result)
        assert len(parsed) == 2


# ---------------------------------------------------------------------------
# _chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_even_split(self):
        result = _chunk([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        result = _chunk([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_empty_list(self):
        assert _chunk([], 5) == []

    def test_size_larger_than_list(self):
        assert _chunk([1, 2], 10) == [[1, 2]]

    def test_size_one(self):
        assert _chunk([1, 2, 3], 1) == [[1], [2], [3]]

    def test_preserves_order(self):
        data = list(range(100))
        chunks = _chunk(data, 10)
        flat = [x for c in chunks for x in c]
        assert flat == data

    def test_single_element_list(self):
        assert _chunk([42], 5) == [[42]]

    def test_size_equals_list_length(self):
        assert _chunk([1, 2, 3], 3) == [[1, 2, 3]]


# ---------------------------------------------------------------------------
# _max_severity
# ---------------------------------------------------------------------------


class TestMaxSeverity:
    def test_critical_wins(self):
        severities = [Severity.LOW, Severity.CRITICAL, Severity.HIGH]
        assert _max_severity(severities) == Severity.CRITICAL

    def test_high_when_no_critical(self):
        assert _max_severity([Severity.MEDIUM, Severity.HIGH]) == Severity.HIGH

    def test_single_low(self):
        assert _max_severity([Severity.LOW]) == Severity.LOW

    def test_empty_returns_low(self):
        assert _max_severity([]) == Severity.LOW

    def test_all_medium(self):
        assert _max_severity([Severity.MEDIUM, Severity.MEDIUM]) == Severity.MEDIUM

    def test_mixed_all_four_levels(self):
        all_severities = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        assert _max_severity(all_severities) == Severity.CRITICAL

    def test_single_critical(self):
        assert _max_severity([Severity.CRITICAL]) == Severity.CRITICAL

    def test_high_and_medium(self):
        assert _max_severity([Severity.HIGH, Severity.MEDIUM]) == Severity.HIGH

    def test_low_and_medium(self):
        assert _max_severity([Severity.LOW, Severity.MEDIUM]) == Severity.MEDIUM


# ---------------------------------------------------------------------------
# _default_triage_result
# ---------------------------------------------------------------------------


class TestDefaultTriageResult:
    def test_uses_issue_id(self):
        issue = _make_issue(id=99)
        result = _default_triage_result(issue)
        assert result.issue_id == 99

    def test_severity_is_medium(self):
        issue = _make_issue(id=1)
        result = _default_triage_result(issue)
        assert result.severity == Severity.MEDIUM

    def test_priority_score_is_50(self):
        issue = _make_issue(id=1)
        result = _default_triage_result(issue)
        assert result.priority_score == 50.0

    def test_complexity_is_unknown(self):
        issue = _make_issue(id=1)
        result = _default_triage_result(issue)
        assert result.complexity == Complexity.UNKNOWN

    def test_impact_category_is_other(self):
        issue = _make_issue(id=1)
        result = _default_triage_result(issue)
        assert result.impact_category == ImpactCategory.OTHER

    def test_summary_uses_issue_title(self):
        issue = _make_issue(id=1, title="Crash on startup")
        result = _default_triage_result(issue)
        assert result.summary == "Crash on startup"

    def test_duplicate_of_is_none(self):
        issue = _make_issue(id=1)
        result = _default_triage_result(issue)
        assert result.duplicate_of is None

    def test_related_issue_ids_is_empty(self):
        issue = _make_issue(id=1)
        result = _default_triage_result(issue)
        assert result.related_issue_ids == []

    def test_tags_is_empty(self):
        issue = _make_issue(id=1)
        result = _default_triage_result(issue)
        assert result.tags == []

    def test_reasoning_mentions_default(self):
        issue = _make_issue(id=1)
        result = _default_triage_result(issue)
        assert "Default" in result.reasoning or "default" in result.reasoning


# ---------------------------------------------------------------------------
# _dict_to_triage_result
# ---------------------------------------------------------------------------


class TestDictToTriageResult:
    def _valid_dict(self, issue_id: int = 1) -> dict[str, Any]:
        return {
            "issue_id": issue_id,
            "severity": "high",
            "impact_category": "crash",
            "priority_score": 75.5,
            "summary": "App crashes.",
            "reasoning": "It crashes.",
            "duplicate_of": None,
            "related_issue_ids": [2, 3],
            "complexity": "medium",
            "tags": ["crash", "login"],
        }

    def test_valid_dict(self):
        result = _dict_to_triage_result(self._valid_dict())
        assert result.issue_id == 1
        assert result.severity == Severity.HIGH
        assert result.impact_category == ImpactCategory.CRASH
        assert result.priority_score == 75.5
        assert result.related_issue_ids == [2, 3]
        assert result.tags == ["crash", "login"]

    def test_unknown_severity_defaults_to_medium(self):
        d = self._valid_dict()
        d["severity"] = "super-critical"
        result = _dict_to_triage_result(d)
        assert result.severity == Severity.MEDIUM

    def test_unknown_impact_defaults_to_other(self):
        d = self._valid_dict()
        d["impact_category"] = "explosion"
        result = _dict_to_triage_result(d)
        assert result.impact_category == ImpactCategory.OTHER

    def test_unknown_complexity_defaults_to_unknown(self):
        d = self._valid_dict()
        d["complexity"] = "very-hard"
        result = _dict_to_triage_result(d)
        assert result.complexity == Complexity.UNKNOWN

    def test_priority_score_clamped_to_100(self):
        d = self._valid_dict()
        d["priority_score"] = 150.0
        result = _dict_to_triage_result(d)
        assert result.priority_score == 100.0

    def test_priority_score_clamped_to_0(self):
        d = self._valid_dict()
        d["priority_score"] = -10.0
        result = _dict_to_triage_result(d)
        assert result.priority_score == 0.0

    def test_missing_issue_id_raises(self):
        d = self._valid_dict()
        del d["issue_id"]
        with pytest.raises((KeyError, ValueError)):
            _dict_to_triage_result(d)

    def test_duplicate_of_parsed_correctly(self):
        d = self._valid_dict()
        d["duplicate_of"] = 5
        result = _dict_to_triage_result(d)
        assert result.duplicate_of == 5

    def test_duplicate_of_none_stays_none(self):
        result = _dict_to_triage_result(self._valid_dict())
        assert result.duplicate_of is None

    def test_related_ids_non_list_becomes_empty(self):
        d = self._valid_dict()
        d["related_issue_ids"] = "not-a-list"
        result = _dict_to_triage_result(d)
        assert result.related_issue_ids == []

    def test_all_severity_levels_parsed(self):
        for sev in ["critical", "high", "medium", "low"]:
            d = self._valid_dict()
            d["severity"] = sev
            result = _dict_to_triage_result(d)
            assert result.severity == Severity(sev)

    def test_all_impact_categories_parsed(self):
        for cat in ["crash", "performance", "security", "ux", "data_loss", "regression", "other"]:
            d = self._valid_dict()
            d["impact_category"] = cat
            result = _dict_to_triage_result(d)
            assert result.impact_category == ImpactCategory(cat)

    def test_all_complexity_levels_parsed(self):
        for comp in ["low", "medium", "high", "unknown"]:
            d = self._valid_dict()
            d["complexity"] = comp
            result = _dict_to_triage_result(d)
            assert result.complexity == Complexity(comp)

    def test_priority_score_at_boundary_100(self):
        d = self._valid_dict()
        d["priority_score"] = 100.0
        result = _dict_to_triage_result(d)
        assert result.priority_score == 100.0

    def test_priority_score_at_boundary_0(self):
        d = self._valid_dict()
        d["priority_score"] = 0.0
        result = _dict_to_triage_result(d)
        assert result.priority_score == 0.0

    def test_summary_and_reasoning_preserved(self):
        d = self._valid_dict()
        d["summary"] = "Specific summary text."
        d["reasoning"] = "Specific reasoning text."
        result = _dict_to_triage_result(d)
        assert result.summary == "Specific summary text."
        assert result.reasoning == "Specific reasoning text."

    def test_empty_related_ids_list(self):
        d = self._valid_dict()
        d["related_issue_ids"] = []
        result = _dict_to_triage_result(d)
        assert result.related_issue_ids == []

    def test_empty_tags_list(self):
        d = self._valid_dict()
        d["tags"] = []
        result = _dict_to_triage_result(d)
        assert result.tags == []

    def test_tags_non_list_becomes_empty(self):
        d = self._valid_dict()
        d["tags"] = "not-a-list"
        result = _dict_to_triage_result(d)
        assert result.tags == []

    def test_issue_id_as_string_is_parsed(self):
        d = self._valid_dict()
        d["issue_id"] = "42"
        result = _dict_to_triage_result(d)
        assert result.issue_id == 42


# ---------------------------------------------------------------------------
# _UnionFind
# ---------------------------------------------------------------------------


class TestUnionFind:
    def test_each_element_own_root(self):
        uf = _UnionFind([1, 2, 3])
        assert uf.find(1) == 1
        assert uf.find(2) == 2
        assert uf.find(3) == 3

    def test_union_merges_sets(self):
        uf = _UnionFind([1, 2, 3])
        uf.union(1, 2)
        assert uf.find(1) == uf.find(2)

    def test_union_transitive(self):
        uf = _UnionFind([1, 2, 3])
        uf.union(1, 2)
        uf.union(2, 3)
        assert uf.find(1) == uf.find(3)

    def test_union_idempotent(self):
        uf = _UnionFind([1, 2])
        uf.union(1, 2)
        uf.union(1, 2)
        assert uf.find(1) == uf.find(2)

    def test_separate_sets_remain_distinct(self):
        uf = _UnionFind([1, 2, 3, 4])
        uf.union(1, 2)
        uf.union(3, 4)
        assert uf.find(1) != uf.find(3)
        assert uf.find(2) != uf.find(4)

    def test_single_element(self):
        uf = _UnionFind([42])
        assert uf.find(42) == 42

    def test_union_commutative(self):
        uf1 = _UnionFind([1, 2])
        uf1.union(1, 2)
        root_1 = uf1.find(1)

        uf2 = _UnionFind([1, 2])
        uf2.union(2, 1)
        root_2 = uf2.find(1)

        # Both should be in the same set (root may differ but find() is consistent)
        assert uf1.find(1) == uf1.find(2)
        assert uf2.find(1) == uf2.find(2)

    def test_path_compression_consistency(self):
        """After path compression, find() remains consistent."""
        uf = _UnionFind([1, 2, 3, 4, 5])
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(3, 4)
        uf.union(4, 5)
        root = uf.find(1)
        for i in range(1, 6):
            assert uf.find(i) == root

    def test_multiple_disjoint_sets(self):
        uf = _UnionFind([1, 2, 3, 4, 5, 6])
        uf.union(1, 2)
        uf.union(3, 4)
        uf.union(5, 6)
        assert uf.find(1) == uf.find(2)
        assert uf.find(3) == uf.find(4)
        assert uf.find(5) == uf.find(6)
        assert uf.find(1) != uf.find(3)
        assert uf.find(1) != uf.find(5)
        assert uf.find(3) != uf.find(5)


# ---------------------------------------------------------------------------
# TriageEngine._classify_issues / _parse_triage_response
# ---------------------------------------------------------------------------


class TestClassifyIssues:
    def _engine(self, llm_response: str) -> TriageEngine:
        return TriageEngine(
            llm_client=_make_llm_client(llm_response),
            repository="example/repo",
        )

    def test_returns_triage_results_for_each_issue(self):
        issues = [_make_issue(id=1), _make_issue(id=2)]
        response = json.dumps(
            [
                {
                    "issue_id": 1,
                    "severity": "high",
                    "impact_category": "crash",
                    "priority_score": 80.0,
                    "summary": "Crash.",
                    "reasoning": "Crashes.",
                    "duplicate_of": None,
                    "related_issue_ids": [],
                    "complexity": "medium",
                    "tags": [],
                },
                {
                    "issue_id": 2,
                    "severity": "low",
                    "impact_category": "ux",
                    "priority_score": 20.0,
                    "summary": "Minor UI glitch.",
                    "reasoning": "Cosmetic.",
                    "duplicate_of": None,
                    "related_issue_ids": [],
                    "complexity": "low",
                    "tags": [],
                },
            ]
        )
        engine = self._engine(response)
        results = engine._classify_issues(issues)
        assert len(results) == 2
        ids = {r.issue_id for r in results}
        assert ids == {1, 2}

    def test_falls_back_to_default_on_invalid_json(self):
        issues = [_make_issue(id=5)]
        engine = self._engine("not json at all")
        results = engine._classify_issues(issues)
        assert len(results) == 1
        assert results[0].issue_id == 5
        assert results[0].severity == Severity.MEDIUM  # default

    def test_falls_back_to_default_on_non_array_json(self):
        issues = [_make_issue(id=7)]
        engine = self._engine('{"key": "value"}')
        results = engine._classify_issues(issues)
        assert results[0].issue_id == 7
        assert results[0].severity == Severity.MEDIUM

    def test_missing_issue_id_gets_default_result(self):
        """If the LLM omits an issue, a default result is inserted."""
        issues = [_make_issue(id=10), _make_issue(id=11)]
        # LLM only returns result for issue 10
        response = json.dumps(
            [
                {
                    "issue_id": 10,
                    "severity": "critical",
                    "impact_category": "security",
                    "priority_score": 95.0,
                    "summary": "Security issue.",
                    "reasoning": "Critical.",
                    "duplicate_of": None,
                    "related_issue_ids": [],
                    "complexity": "high",
                    "tags": [],
                }
            ]
        )
        engine = self._engine(response)
        results = engine._classify_issues(issues)
        assert len(results) == 2
        result_ids = {r.issue_id for r in results}
        assert 11 in result_ids
        # The default result for 11 should be medium
        r11 = next(r for r in results if r.issue_id == 11)
        assert r11.severity == Severity.MEDIUM

    def test_strips_markdown_fences_from_response(self):
        issues = [_make_issue(id=3)]
        inner = json.dumps(
            [
                {
                    "issue_id": 3,
                    "severity": "low",
                    "impact_category": "ux",
                    "priority_score": 15.0,
                    "summary": "UX issue.",
                    "reasoning": "Minor.",
                    "duplicate_of": None,
                    "related_issue_ids": [],
                    "complexity": "low",
                    "tags": [],
                }
            ]
        )
        fenced = f"```json\n{inner}\n```"
        engine = self._engine(fenced)
        results = engine._classify_issues(issues)
        assert results[0].issue_id == 3
        assert results[0].severity == Severity.LOW

    def test_llm_error_raises_triage_error(self):
        from bug_triage.llm_client import LLMError

        client = _make_llm_client()
        client.render_and_complete.side_effect = LLMError("API failure")
        engine = TriageEngine(llm_client=client)
        issues = [_make_issue(id=1)]
        with pytest.raises(TriageError, match="LLM classification failed"):
            engine._classify_issues(issues)

    def test_batching_calls_llm_multiple_times(self):
        issues = [_make_issue(id=i) for i in range(1, 6)]
        # batch_size=2 → ceil(5/2)=3 batches
        client = _make_llm_client("[]")  # returns empty array each time
        client.render_and_complete.return_value = "[]"
        engine = TriageEngine(llm_client=client, batch_size=2)
        # Each empty batch means we fall back to defaults for the issues
        results = engine._classify_issues(issues)
        assert client.render_and_complete.call_count == 3
        assert len(results) == 5

    def test_correct_severity_from_llm_response(self):
        issues = [_make_issue(id=1)]
        response = json.dumps([
            {
                "issue_id": 1,
                "severity": "critical",
                "impact_category": "security",
                "priority_score": 99.0,
                "summary": "Critical security flaw.",
                "reasoning": "Auth bypass.",
                "duplicate_of": None,
                "related_issue_ids": [],
                "complexity": "high",
                "tags": ["security"],
            }
        ])
        engine = self._engine(response)
        results = engine._classify_issues(issues)
        assert results[0].severity == Severity.CRITICAL
        assert results[0].impact_category == ImpactCategory.SECURITY
        assert results[0].priority_score == 99.0

    def test_batch_size_one_processes_each_issue_separately(self):
        issues = [_make_issue(id=i) for i in range(1, 4)]
        client = _make_llm_client("[]")
        engine = TriageEngine(llm_client=client, batch_size=1)
        results = engine._classify_issues(issues)
        # 3 issues with batch_size=1 → 3 LLM calls
        assert client.render_and_complete.call_count == 3
        assert len(results) == 3

    def test_all_default_results_have_correct_issue_ids(self):
        """When LLM returns empty array, all issues should get default results."""
        issues = [_make_issue(id=i) for i in [10, 20, 30]]
        engine = self._engine("[]")
        results = engine._classify_issues(issues)
        result_ids = {r.issue_id for r in results}
        assert result_ids == {10, 20, 30}


# ---------------------------------------------------------------------------
# TriageEngine._build_groups
# ---------------------------------------------------------------------------


class TestBuildGroups:
    def _engine(self) -> TriageEngine:
        return TriageEngine(llm_client=_make_llm_client())

    def test_singleton_group_for_single_issue(self):
        engine = self._engine()
        issues = [_make_issue(id=1)]
        results = [_make_triage_result(issue_id=1)]
        groups = engine._build_groups(issues, results)
        assert len(groups) == 1
        assert groups[0].issue_ids == [1]

    def test_duplicate_merged_into_canonical(self):
        """Issue 2 is a duplicate of issue 1 — they should be in the same group."""
        engine = self._engine()
        issues = [_make_issue(id=1), _make_issue(id=2)]
        results = [
            _make_triage_result(issue_id=1, priority_score=80.0),
            _make_triage_result(issue_id=2, duplicate_of=1, priority_score=60.0),
        ]
        groups = engine._build_groups(issues, results)
        assert len(groups) == 1
        group = groups[0]
        assert set(group.issue_ids) == {1, 2}
        assert group.canonical_issue_id == 1  # highest priority_score

    def test_unrelated_issues_form_separate_groups(self):
        engine = self._engine()
        issues = [_make_issue(id=1), _make_issue(id=2), _make_issue(id=3)]
        results = [
            _make_triage_result(issue_id=1),
            _make_triage_result(issue_id=2),
            _make_triage_result(issue_id=3),
        ]
        groups = engine._build_groups(issues, results)
        assert len(groups) == 3

    def test_related_both_ways_merged(self):
        """Issues that mutually list each other as related are merged."""
        engine = self._engine()
        issues = [_make_issue(id=10), _make_issue(id=11)]
        results = [
            _make_triage_result(issue_id=10, related_issue_ids=[11]),
            _make_triage_result(issue_id=11, related_issue_ids=[10]),
        ]
        groups = engine._build_groups(issues, results)
        assert len(groups) == 1
        assert set(groups[0].issue_ids) == {10, 11}

    def test_one_way_related_not_merged(self):
        """One-sided 'related' reference does NOT merge the issues."""
        engine = self._engine()
        issues = [_make_issue(id=10), _make_issue(id=11)]
        results = [
            _make_triage_result(issue_id=10, related_issue_ids=[11]),
            _make_triage_result(issue_id=11, related_issue_ids=[]),
        ]
        groups = engine._build_groups(issues, results)
        assert len(groups) == 2

    def test_group_severity_is_max_of_members(self):
        engine = self._engine()
        issues = [_make_issue(id=1), _make_issue(id=2)]
        results = [
            _make_triage_result(issue_id=1, severity=Severity.LOW, priority_score=20.0),
            _make_triage_result(
                issue_id=2, duplicate_of=1, severity=Severity.CRITICAL, priority_score=90.0
            ),
        ]
        groups = engine._build_groups(issues, results)
        assert len(groups) == 1
        assert groups[0].severity == Severity.CRITICAL

    def test_group_priority_score_is_max_of_members(self):
        engine = self._engine()
        issues = [_make_issue(id=1), _make_issue(id=2)]
        results = [
            _make_triage_result(issue_id=1, priority_score=40.0),
            _make_triage_result(issue_id=2, duplicate_of=1, priority_score=75.0),
        ]
        groups = engine._build_groups(issues, results)
        assert groups[0].priority_score == 75.0

    def test_group_tags_are_union_of_member_tags(self):
        engine = self._engine()
        issues = [_make_issue(id=1), _make_issue(id=2)]
        results = [
            _make_triage_result(issue_id=1, tags=["auth", "jwt"]),
            _make_triage_result(issue_id=2, duplicate_of=1, tags=["jwt", "security"]),
        ]
        groups = engine._build_groups(issues, results)
        assert set(groups[0].tags) == {"auth", "jwt", "security"}

    def test_canonical_issue_is_highest_priority(self):
        engine = self._engine()
        issues = [_make_issue(id=1), _make_issue(id=2), _make_issue(id=3)]
        results = [
            _make_triage_result(issue_id=1, priority_score=50.0),
            _make_triage_result(issue_id=2, duplicate_of=1, priority_score=90.0),
            _make_triage_result(issue_id=3, duplicate_of=1, priority_score=70.0),
        ]
        groups = engine._build_groups(issues, results)
        assert len(groups) == 1
        # Issue 2 has highest priority_score so should be canonical
        assert groups[0].canonical_issue_id == 2

    def test_group_id_assigned(self):
        engine = self._engine()
        issues = [_make_issue(id=1)]
        results = [_make_triage_result(issue_id=1)]
        groups = engine._build_groups(issues, results)
        assert groups[0].id.startswith("group_")

    def test_group_issue_ids_are_sorted(self):
        engine = self._engine()
        issues = [_make_issue(id=3), _make_issue(id=1), _make_issue(id=2)]
        results = [
            _make_triage_result(issue_id=3, priority_score=80.0),
            _make_triage_result(issue_id=1, duplicate_of=3, priority_score=30.0),
            _make_triage_result(issue_id=2, duplicate_of=3, priority_score=50.0),
        ]
        groups = engine._build_groups(issues, results)
        assert len(groups) == 1
        assert groups[0].issue_ids == sorted(groups[0].issue_ids)

    def test_group_title_taken_from_canonical_issue(self):
        engine = self._engine()
        issues = [
            _make_issue(id=1, title="Primary issue title"),
            _make_issue(id=2, title="Duplicate issue title"),
        ]
        results = [
            _make_triage_result(issue_id=1, priority_score=80.0),
            _make_triage_result(issue_id=2, duplicate_of=1, priority_score=20.0),
        ]
        groups = engine._build_groups(issues, results)
        # Canonical is id=1 (highest priority_score)
        assert "Primary issue title" in groups[0].title

    def test_chain_of_duplicates_merged_into_single_group(self):
        """A chain 1 <- 2 <- 3 should form a single group."""
        engine = self._engine()
        issues = [_make_issue(id=i) for i in [1, 2, 3]]
        results = [
            _make_triage_result(issue_id=1, priority_score=90.0),
            _make_triage_result(issue_id=2, duplicate_of=1, priority_score=70.0),
            _make_triage_result(issue_id=3, duplicate_of=2, priority_score=50.0),
        ]
        groups = engine._build_groups(issues, results)
        assert len(groups) == 1
        assert set(groups[0].issue_ids) == {1, 2, 3}

    def test_impact_category_from_canonical_issue(self):
        engine = self._engine()
        issues = [_make_issue(id=1), _make_issue(id=2)]
        results = [
            _make_triage_result(
                issue_id=1,
                impact_category=ImpactCategory.SECURITY,
                priority_score=90.0,
            ),
            _make_triage_result(
                issue_id=2,
                duplicate_of=1,
                impact_category=ImpactCategory.CRASH,
                priority_score=50.0,
            ),
        ]
        groups = engine._build_groups(issues, results)
        # Canonical is id=1 (higher priority), so impact should be SECURITY
        assert groups[0].impact_category == ImpactCategory.SECURITY


# ---------------------------------------------------------------------------
# TriageEngine._estimate_complexity
# ---------------------------------------------------------------------------


class TestEstimateComplexity:
    def _engine_with_complexity_response(self, response: str) -> TriageEngine:
        client = _make_llm_client(response)
        return TriageEngine(llm_client=client)

    def _make_group(self, group_id: str, issue_ids: list[int]) -> IssueGroup:
        return IssueGroup(
            id=group_id,
            title="Test group",
            severity=Severity.MEDIUM,
            impact_category=ImpactCategory.OTHER,
            priority_score=50.0,
            complexity=Complexity.UNKNOWN,
            fix_order=0,
            issue_ids=issue_ids,
            canonical_issue_id=issue_ids[0],
            summary="Group summary.",
        )

    def test_sets_complexity_from_llm_response(self):
        group = self._make_group("group_001", [1, 2])
        response = json.dumps(
            [
                {
                    "group_id": "group_001",
                    "complexity": "high",
                    "reasoning": "Requires extensive refactoring.",
                }
            ]
        )
        issues = [_make_issue(id=1), _make_issue(id=2)]
        engine = self._engine_with_complexity_response(response)
        updated = engine._estimate_complexity([group], issues)
        assert len(updated) == 1
        assert updated[0].complexity == Complexity.HIGH

    def test_keeps_unknown_on_llm_failure(self):
        from bug_triage.llm_client import LLMError

        group = self._make_group("group_001", [1])
        client = _make_llm_client()
        client.render_and_complete.side_effect = LLMError("fail")
        engine = TriageEngine(llm_client=client)
        issues = [_make_issue(id=1)]
        updated = engine._estimate_complexity([group], issues)
        assert updated[0].complexity == Complexity.UNKNOWN

    def test_unknown_complexity_value_becomes_unknown_enum(self):
        group = self._make_group("group_001", [1])
        response = json.dumps(
            [
                {
                    "group_id": "group_001",
                    "complexity": "extremely-hard",
                    "reasoning": "Very hard.",
                }
            ]
        )
        issues = [_make_issue(id=1)]
        engine = self._engine_with_complexity_response(response)
        updated = engine._estimate_complexity([group], issues)
        assert updated[0].complexity == Complexity.UNKNOWN

    def test_empty_groups_returns_empty(self):
        engine = self._engine_with_complexity_response("[]")
        result = engine._estimate_complexity([], [])
        assert result == []

    def test_invalid_json_response_keeps_unknown(self):
        group = self._make_group("group_001", [1])
        engine = self._engine_with_complexity_response("not-json")
        issues = [_make_issue(id=1)]
        updated = engine._estimate_complexity([group], issues)
        assert updated[0].complexity == Complexity.UNKNOWN

    def test_sets_complexity_low(self):
        group = self._make_group("group_001", [1])
        response = json.dumps([
            {"group_id": "group_001", "complexity": "low", "reasoning": "Simple fix."}
        ])
        issues = [_make_issue(id=1)]
        engine = self._engine_with_complexity_response(response)
        updated = engine._estimate_complexity([group], issues)
        assert updated[0].complexity == Complexity.LOW

    def test_sets_complexity_medium(self):
        group = self._make_group("group_001", [1])
        response = json.dumps([
            {"group_id": "group_001", "complexity": "medium", "reasoning": "Moderate effort."}
        ])
        issues = [_make_issue(id=1)]
        engine = self._engine_with_complexity_response(response)
        updated = engine._estimate_complexity([group], issues)
        assert updated[0].complexity == Complexity.MEDIUM

    def test_multiple_groups_complexity_assigned_correctly(self):
        groups = [
            self._make_group("group_001", [1]),
            self._make_group("group_002", [2]),
            self._make_group("group_003", [3]),
        ]
        response = json.dumps([
            {"group_id": "group_001", "complexity": "low", "reasoning": "Easy."},
            {"group_id": "group_002", "complexity": "high", "reasoning": "Hard."},
            {"group_id": "group_003", "complexity": "medium", "reasoning": "Moderate."},
        ])
        issues = [_make_issue(id=i) for i in [1, 2, 3]]
        engine = self._engine_with_complexity_response(response)
        updated = engine._estimate_complexity(groups, issues)
        complexity_by_id = {g.id: g.complexity for g in updated}
        assert complexity_by_id["group_001"] == Complexity.LOW
        assert complexity_by_id["group_002"] == Complexity.HIGH
        assert complexity_by_id["group_003"] == Complexity.MEDIUM

    def test_non_array_response_keeps_unknown(self):
        group = self._make_group("group_001", [1])
        engine = self._engine_with_complexity_response('{"group_id": "group_001"}')
        issues = [_make_issue(id=1)]
        updated = engine._estimate_complexity([group], issues)
        assert updated[0].complexity == Complexity.UNKNOWN

    def test_missing_group_id_in_response_keeps_unknown(self):
        group = self._make_group("group_001", [1])
        response = json.dumps([
            {"complexity": "high", "reasoning": "Missing group_id field."},
        ])
        issues = [_make_issue(id=1)]
        engine = self._engine_with_complexity_response(response)
        updated = engine._estimate_complexity([group], issues)
        # group_001 not in response → stays UNKNOWN
        assert updated[0].complexity == Complexity.UNKNOWN


# ---------------------------------------------------------------------------
# TriageEngine._assign_fix_order
# ---------------------------------------------------------------------------


class TestAssignFixOrder:
    def _engine(self) -> TriageEngine:
        return TriageEngine(llm_client=_make_llm_client())

    def _make_group(
        self,
        group_id: str,
        severity: Severity,
        priority_score: float,
        issue_ids: list[int] | None = None,
    ) -> IssueGroup:
        ids = issue_ids or [int(group_id[-1])]
        return IssueGroup(
            id=group_id,
            title=f"Group {group_id}",
            severity=severity,
            impact_category=ImpactCategory.OTHER,
            priority_score=priority_score,
            complexity=Complexity.MEDIUM,
            fix_order=0,
            issue_ids=ids,
            canonical_issue_id=ids[0],
            summary="Summary.",
        )

    def test_critical_before_high(self):
        engine = self._engine()
        groups = [
            self._make_group("group_001", Severity.HIGH, 75.0, [1]),
            self._make_group("group_002", Severity.CRITICAL, 90.0, [2]),
        ]
        ranked = engine._assign_fix_order(groups)
        # group_002 (critical) should be fix_order=1
        ranked_by_id = {g.id: g for g in ranked}
        assert ranked_by_id["group_002"].fix_order == 1
        assert ranked_by_id["group_001"].fix_order == 2

    def test_higher_priority_score_wins_within_same_severity(self):
        engine = self._engine()
        groups = [
            self._make_group("group_001", Severity.HIGH, 60.0, [1]),
            self._make_group("group_002", Severity.HIGH, 85.0, [2]),
        ]
        ranked = engine._assign_fix_order(groups)
        ranked_by_id = {g.id: g for g in ranked}
        assert ranked_by_id["group_002"].fix_order == 1
        assert ranked_by_id["group_001"].fix_order == 2

    def test_all_groups_get_unique_fix_orders(self):
        engine = self._engine()
        groups = [
            self._make_group("group_001", Severity.LOW, 10.0, [1]),
            self._make_group("group_002", Severity.MEDIUM, 40.0, [2]),
            self._make_group("group_003", Severity.HIGH, 70.0, [3]),
            self._make_group("group_004", Severity.CRITICAL, 95.0, [4]),
        ]
        ranked = engine._assign_fix_order(groups)
        orders = [g.fix_order for g in ranked]
        assert sorted(orders) == [1, 2, 3, 4]

    def test_empty_groups_returns_empty(self):
        engine = self._engine()
        assert engine._assign_fix_order([]) == []

    def test_fix_orders_start_at_one(self):
        engine = self._engine()
        groups = [
            self._make_group("group_001", Severity.HIGH, 70.0, [1]),
            self._make_group("group_002", Severity.MEDIUM, 40.0, [2]),
        ]
        ranked = engine._assign_fix_order(groups)
        orders = sorted(g.fix_order for g in ranked)
        assert orders[0] == 1

    def test_single_group_gets_fix_order_one(self):
        engine = self._engine()
        groups = [self._make_group("group_001", Severity.CRITICAL, 99.0, [1])]
        ranked = engine._assign_fix_order(groups)
        assert ranked[0].fix_order == 1

    def test_severity_order_critical_high_medium_low(self):
        engine = self._engine()
        groups = [
            self._make_group("g_low", Severity.LOW, 25.0, [4]),
            self._make_group("g_med", Severity.MEDIUM, 50.0, [3]),
            self._make_group("g_high", Severity.HIGH, 75.0, [2]),
            self._make_group("g_crit", Severity.CRITICAL, 95.0, [1]),
        ]
        ranked = engine._assign_fix_order(groups)
        ranked_by_id = {g.id: g.fix_order for g in ranked}
        assert ranked_by_id["g_crit"] < ranked_by_id["g_high"]
        assert ranked_by_id["g_high"] < ranked_by_id["g_med"]
        assert ranked_by_id["g_med"] < ranked_by_id["g_low"]

    def test_original_groups_not_mutated(self):
        """_assign_fix_order should return new objects, not mutate originals."""
        engine = self._engine()
        original = self._make_group("group_001", Severity.HIGH, 70.0, [1])
        assert original.fix_order == 0
        engine._assign_fix_order([original])
        # Original should not be changed (model_copy creates new instance)
        assert original.fix_order == 0


# ---------------------------------------------------------------------------
# TriageEngine.run — integration-style tests
# ---------------------------------------------------------------------------


class TestTriageEngineRun:
    def _build_triage_response(self, issues: list[Issue]) -> str:
        """Build a valid LLM triage JSON response for the given issues."""
        items = [
            {
                "issue_id": i.id,
                "severity": "medium",
                "impact_category": "other",
                "priority_score": 50.0,
                "summary": f"Summary for #{i.id}.",
                "reasoning": "Test.",
                "duplicate_of": None,
                "related_issue_ids": [],
                "complexity": "medium",
                "tags": [],
            }
            for i in issues
        ]
        return json.dumps(items)

    def _build_complexity_response(self, groups_count: int) -> str:
        """Return an empty complexity response (all groups get UNKNOWN)."""
        return "[]"

    def test_empty_issues_returns_empty_report(self):
        client = _make_llm_client()
        engine = TriageEngine(llm_client=client, repository="owner/repo")
        report = engine.run([])
        assert report.metadata.total_issues == 0
        assert report.groups == []
        assert report.triage_results == []
        assert report.raw_issues == []
        client.render_and_complete.assert_not_called()

    def test_run_produces_correct_metadata(self):
        issues = [_make_issue(id=1), _make_issue(id=2)]
        triage_resp = self._build_triage_response(issues)
        complexity_resp = self._build_complexity_response(2)

        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        client.render_and_complete.side_effect = [triage_resp, complexity_resp]

        engine = TriageEngine(
            llm_client=client,
            repository="owner/repo",
            source_file="",
            output_format=OutputFormat.JSON,
        )
        report = engine.run(issues)

        assert report.metadata.total_issues == 2
        assert report.metadata.repository == "owner/repo"
        assert report.metadata.llm_provider == "openai"
        assert report.metadata.llm_model == "gpt-4o"
        assert report.metadata.output_format == OutputFormat.JSON

    def test_run_returns_one_group_per_independent_issue(self):
        issues = [_make_issue(id=i) for i in range(1, 4)]
        triage_resp = self._build_triage_response(issues)
        complexity_resp = self._build_complexity_response(3)

        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        client.render_and_complete.side_effect = [triage_resp, complexity_resp]

        engine = TriageEngine(llm_client=client)
        report = engine.run(issues)

        assert len(report.groups) == 3
        assert report.metadata.total_groups == 3

    def test_run_merges_duplicate_issues(self):
        issues = [_make_issue(id=1), _make_issue(id=2)]
        # Issue 2 is a duplicate of issue 1
        triage_items = [
            {
                "issue_id": 1,
                "severity": "high",
                "impact_category": "crash",
                "priority_score": 80.0,
                "summary": "App crashes.",
                "reasoning": "Crash.",
                "duplicate_of": None,
                "related_issue_ids": [],
                "complexity": "medium",
                "tags": [],
            },
            {
                "issue_id": 2,
                "severity": "high",
                "impact_category": "crash",
                "priority_score": 70.0,
                "summary": "Also crashes.",
                "reasoning": "Duplicate.",
                "duplicate_of": 1,
                "related_issue_ids": [],
                "complexity": "medium",
                "tags": [],
            },
        ]
        triage_resp = json.dumps(triage_items)
        complexity_resp = "[]"

        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        client.render_and_complete.side_effect = [triage_resp, complexity_resp]

        engine = TriageEngine(llm_client=client)
        report = engine.run(issues)

        assert len(report.groups) == 1
        assert set(report.groups[0].issue_ids) == {1, 2}

    def test_fix_order_set_on_all_groups(self):
        issues = [_make_issue(id=i) for i in range(1, 4)]
        triage_resp = self._build_triage_response(issues)
        complexity_resp = "[]"

        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        client.render_and_complete.side_effect = [triage_resp, complexity_resp]

        engine = TriageEngine(llm_client=client)
        report = engine.run(issues)

        fix_orders = [g.fix_order for g in report.groups]
        # All fix orders should be > 0
        assert all(fo > 0 for fo in fix_orders)
        # All should be unique
        assert len(set(fix_orders)) == len(fix_orders)

    def test_raw_issues_preserved_in_report(self):
        issues = [_make_issue(id=1), _make_issue(id=2)]
        triage_resp = self._build_triage_response(issues)
        complexity_resp = "[]"

        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        client.render_and_complete.side_effect = [triage_resp, complexity_resp]

        engine = TriageEngine(llm_client=client)
        report = engine.run(issues)

        assert len(report.raw_issues) == 2
        assert {i.id for i in report.raw_issues} == {1, 2}

    def test_triage_results_preserved_in_report(self):
        issues = [_make_issue(id=1)]
        triage_resp = self._build_triage_response(issues)
        complexity_resp = "[]"

        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        client.render_and_complete.side_effect = [triage_resp, complexity_resp]

        engine = TriageEngine(llm_client=client)
        report = engine.run(issues)

        assert len(report.triage_results) == 1
        assert report.triage_results[0].issue_id == 1

    def test_run_with_source_file_in_metadata(self):
        issues = [_make_issue(id=1)]
        triage_resp = self._build_triage_response(issues)

        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        client.render_and_complete.side_effect = [triage_resp, "[]"]

        engine = TriageEngine(
            llm_client=client,
            source_file="/path/to/issues.json",
        )
        report = engine.run(issues)
        assert report.metadata.source_file == "/path/to/issues.json"

    def test_ungrouped_is_empty_when_all_issues_in_groups(self):
        issues = [_make_issue(id=1), _make_issue(id=2)]
        triage_resp = self._build_triage_response(issues)

        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        client.render_and_complete.side_effect = [triage_resp, "[]"]

        engine = TriageEngine(llm_client=client)
        report = engine.run(issues)

        # All issues should be in groups
        grouped = {iid for g in report.groups for iid in g.issue_ids}
        assert {1, 2}.issubset(grouped)
        assert report.ungrouped_issue_ids == []

    def test_metadata_total_groups_matches_groups_list(self):
        issues = [_make_issue(id=i) for i in range(1, 5)]
        triage_resp = self._build_triage_response(issues)

        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        client.render_and_complete.side_effect = [triage_resp, "[]"]

        engine = TriageEngine(llm_client=client)
        report = engine.run(issues)

        assert report.metadata.total_groups == len(report.groups)

    def test_empty_report_has_generated_at_set(self):
        client = _make_llm_client()
        engine = TriageEngine(llm_client=client)
        report = engine.run([])
        assert report.metadata.generated_at is not None
        assert isinstance(report.metadata.generated_at, datetime)

    def test_run_with_triage_error_from_llm(self):
        from bug_triage.llm_client import LLMError

        issues = [_make_issue(id=1)]
        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        client.render_and_complete.side_effect = LLMError("API down")

        engine = TriageEngine(llm_client=client)
        with pytest.raises(TriageError):
            engine.run(issues)


# ---------------------------------------------------------------------------
# triage_issues convenience function
# ---------------------------------------------------------------------------


class TestTriageIssuesConvenienceFunction:
    def test_returns_report_output(self):
        issues = [_make_issue(id=1)]
        triage_resp = json.dumps(
            [
                {
                    "issue_id": 1,
                    "severity": "medium",
                    "impact_category": "other",
                    "priority_score": 50.0,
                    "summary": "Summary.",
                    "reasoning": "Reason.",
                    "duplicate_of": None,
                    "related_issue_ids": [],
                    "complexity": "low",
                    "tags": [],
                }
            ]
        )
        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        client.render_and_complete.side_effect = [triage_resp, "[]"]

        report = triage_issues(
            issues=issues,
            llm_client=client,
            repository="owner/repo",
            output_format=OutputFormat.MARKDOWN,
        )
        assert report.metadata.repository == "owner/repo"
        assert report.metadata.total_issues == 1

    def test_convenience_function_uses_batch_size(self):
        """batch_size parameter is forwarded to the TriageEngine."""
        issues = [_make_issue(id=i) for i in range(1, 7)]  # 6 issues
        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        # With batch_size=2 and 6 issues → 3 triage calls + 1 complexity call
        client.render_and_complete.return_value = "[]"

        triage_issues(
            issues=issues,
            llm_client=client,
            batch_size=2,
        )
        # 3 triage batches + 1 complexity call = 4 total calls
        assert client.render_and_complete.call_count == 4

    def test_convenience_function_empty_issues(self):
        client = _make_llm_client()
        report = triage_issues(
            issues=[],
            llm_client=client,
        )
        assert report.metadata.total_issues == 0
        assert report.groups == []
        client.render_and_complete.assert_not_called()

    def test_convenience_function_passes_source_file(self):
        issues = [_make_issue(id=1)]
        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        client.render_and_complete.side_effect = [
            json.dumps([{
                "issue_id": 1, "severity": "low", "impact_category": "other",
                "priority_score": 10.0, "summary": "S.", "reasoning": "R.",
                "duplicate_of": None, "related_issue_ids": [], "complexity": "low", "tags": [],
            }]),
            "[]",
        ]
        report = triage_issues(
            issues=issues,
            llm_client=client,
            source_file="/my/file.json",
        )
        assert report.metadata.source_file == "/my/file.json"

    def test_output_format_reflected_in_metadata(self):
        issues = [_make_issue(id=1)]
        client = MagicMock()
        client.provider = LLMProvider.OPENAI
        client.model = "gpt-4o"
        client.render_and_complete.side_effect = [
            json.dumps([{
                "issue_id": 1, "severity": "low", "impact_category": "other",
                "priority_score": 10.0, "summary": "S.", "reasoning": "R.",
                "duplicate_of": None, "related_issue_ids": [], "complexity": "low", "tags": [],
            }]),
            "[]",
        ]
        report = triage_issues(
            issues=issues,
            llm_client=client,
            output_format=OutputFormat.JSON,
        )
        assert report.metadata.output_format == OutputFormat.JSON
