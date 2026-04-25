"""Core triage pipeline for bug_triage.

This module orchestrates the full triage workflow:
1. Send issues to the LLM for severity/impact classification.
2. Parse and validate the structured LLM response into TriageResult objects.
3. Deduplicate and cluster related issues into IssueGroup objects.
4. Estimate fix complexity per group via a second LLM call.
5. Assign priority-ordered fix ranks and return a complete ReportOutput.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Optional

from bug_triage.llm_client import LLMClient, LLMError
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

logger = logging.getLogger(__name__)

# Maximum number of issues to send to the LLM in a single batch.
# Larger batches risk hitting context-window limits.
_DEFAULT_BATCH_SIZE = 20

# System prompt injected into every triage LLM call.
_TRIAGE_SYSTEM_PROMPT = (
    "You are a senior software engineering lead performing structured bug triage. "
    "Follow the instructions precisely and respond only with the requested JSON."
)

# System prompt for complexity estimation calls.
_COMPLEXITY_SYSTEM_PROMPT = (
    "You are a senior software engineer estimating bug-fix complexity. "
    "Follow the instructions precisely and respond only with the requested JSON."
)


class TriageError(Exception):
    """Raised when the triage pipeline encounters an unrecoverable error."""


class TriageEngine:
    """Orchestrates the full bug-triage pipeline.

    The engine accepts a list of :class:`~bug_triage.models.Issue` objects,
    calls the LLM to classify them, clusters duplicates and related issues
    into :class:`~bug_triage.models.IssueGroup` objects, estimates fix
    complexity per group, and returns a complete
    :class:`~bug_triage.models.ReportOutput`.

    Args:
        llm_client: A configured :class:`~bug_triage.llm_client.LLMClient`
            instance.
        batch_size: Number of issues to send to the LLM per API call.
            Smaller values reduce context-window pressure; larger values
            improve deduplication quality across issues.
        repository: Optional GitHub repository identifier (``owner/repo``).
        source_file: Optional path to the local source file.
        output_format: Desired output format for the final report metadata.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        repository: str = "",
        source_file: str = "",
        output_format: OutputFormat = OutputFormat.MARKDOWN,
    ) -> None:
        self.llm_client = llm_client
        self.batch_size = max(1, batch_size)
        self.repository = repository
        self.source_file = source_file
        self.output_format = output_format

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, issues: list[Issue]) -> ReportOutput:
        """Execute the full triage pipeline and return a structured report.

        Args:
            issues: List of raw issues to triage.

        Returns:
            A fully populated :class:`~bug_triage.models.ReportOutput`.

        Raises:
            TriageError: If a critical pipeline step fails.
        """
        if not issues:
            logger.info("No issues provided; returning empty report.")
            return self._build_empty_report()

        logger.info("Starting triage pipeline for %d issue(s).", len(issues))

        # Step 1 — Classify all issues via LLM (batched).
        triage_results = self._classify_issues(issues)
        logger.info("Received %d triage result(s) from LLM.", len(triage_results))

        # Step 2 — Build issue groups from triage results.
        groups = self._build_groups(issues, triage_results)
        logger.info("Built %d issue group(s).", len(groups))

        # Step 3 — Estimate complexity per group via LLM.
        groups = self._estimate_complexity(groups, issues)

        # Step 4 — Assign fix-order ranks.
        groups = self._assign_fix_order(groups)

        # Step 5 — Identify any ungrouped issues.
        grouped_ids: set[int] = set()
        for group in groups:
            grouped_ids.update(group.issue_ids)
        ungrouped = [i.id for i in issues if i.id not in grouped_ids]

        metadata = ReportMetadata(
            generated_at=datetime.utcnow(),
            repository=self.repository,
            source_file=self.source_file,
            llm_provider=self.llm_client.provider.value,
            llm_model=self.llm_client.model,
            total_issues=len(issues),
            total_groups=len(groups),
            output_format=self.output_format,
        )

        return ReportOutput(
            metadata=metadata,
            groups=groups,
            raw_issues=issues,
            triage_results=triage_results,
            ungrouped_issue_ids=ungrouped,
        )

    # ------------------------------------------------------------------
    # Step 1 — LLM classification
    # ------------------------------------------------------------------

    def _classify_issues(self, issues: list[Issue]) -> list[TriageResult]:
        """Send issues to the LLM in batches and collect TriageResult objects.

        Args:
            issues: All issues to classify.

        Returns:
            A list of :class:`TriageResult` objects (one per issue).

        Raises:
            TriageError: If the LLM call or response parsing fails.
        """
        all_results: list[TriageResult] = []

        batches = _chunk(issues, self.batch_size)
        for batch_idx, batch in enumerate(batches):
            logger.debug(
                "Classifying batch %d/%d (%d issues).",
                batch_idx + 1,
                len(batches),
                len(batch),
            )
            try:
                raw_response = self.llm_client.render_and_complete(
                    "triage_prompt.j2",
                    system_prompt=_TRIAGE_SYSTEM_PROMPT,
                    issues=batch,
                )
            except LLMError as exc:
                raise TriageError(
                    f"LLM classification failed for batch {batch_idx + 1}: {exc}"
                ) from exc

            batch_results = self._parse_triage_response(raw_response, batch)
            all_results.extend(batch_results)

        return all_results

    def _parse_triage_response(
        self, raw_response: str, batch: list[Issue]
    ) -> list[TriageResult]:
        """Parse a raw LLM JSON response into a list of TriageResult objects.

        Falls back to a default classification for any issue whose ID is
        missing from the LLM response.

        Args:
            raw_response: Raw text response from the LLM.
            batch: The issues that were sent in this batch (used for fallback).

        Returns:
            A list of :class:`TriageResult` objects.
        """
        cleaned = _strip_markdown_fences(raw_response)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse LLM triage response as JSON: %s.  "
                "Falling back to default classifications.",
                exc,
            )
            return [_default_triage_result(issue) for issue in batch]

        if not isinstance(data, list):
            logger.warning(
                "LLM triage response is not a JSON array; falling back to defaults."
            )
            return [_default_triage_result(issue) for issue in batch]

        # Build a map of issue_id -> parsed TriageResult.
        parsed: dict[int, TriageResult] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                result = _dict_to_triage_result(item)
                parsed[result.issue_id] = result
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed triage result item: %s", exc)

        # Ensure every issue in the batch has a result.
        results: list[TriageResult] = []
        for issue in batch:
            if issue.id in parsed:
                results.append(parsed[issue.id])
            else:
                logger.warning(
                    "LLM did not return a result for issue #%d; using default.",
                    issue.id,
                )
                results.append(_default_triage_result(issue))

        return results

    # ------------------------------------------------------------------
    # Step 2 — Build issue groups
    # ------------------------------------------------------------------

    def _build_groups(
        self, issues: list[Issue], triage_results: list[TriageResult]
    ) -> list[IssueGroup]:
        """Cluster triage results into IssueGroup objects.

        The clustering algorithm:
        1. Sort triage results by priority_score descending so that canonical
           issues are chosen as the highest-priority member of each cluster.
        2. Any issue whose ``duplicate_of`` points to another issue ID is
           merged into that issue's group.
        3. Issues sharing ``related_issue_ids`` references are linked into the
           same group when both agree on the relationship.
        4. Any remaining ungrouped issue becomes its own singleton group.

        Args:
            issues: Original raw issues.
            triage_results: Per-issue classification results.

        Returns:
            A list of :class:`IssueGroup` objects (unsorted).
        """
        result_map: dict[int, TriageResult] = {r.issue_id: r for r in triage_results}
        issue_map: dict[int, Issue] = {i.id: i for i in issues}

        # Union-Find to cluster issues.
        uf = _UnionFind(list(result_map.keys()))

        # Merge duplicates.
        for result in triage_results:
            if result.duplicate_of is not None and result.duplicate_of in result_map:
                uf.union(result.issue_id, result.duplicate_of)

        # Merge bidirectionally-agreed related issues (optional tighter clustering).
        # Only merge when both sides list each other as related.
        for result in triage_results:
            for related_id in result.related_issue_ids:
                if related_id not in result_map:
                    continue
                other = result_map[related_id]
                if result.issue_id in other.related_issue_ids:
                    uf.union(result.issue_id, related_id)

        # Collect clusters.
        clusters: dict[int, list[int]] = {}
        for issue_id in result_map:
            root = uf.find(issue_id)
            clusters.setdefault(root, []).append(issue_id)

        # Build IssueGroup objects.
        groups: list[IssueGroup] = []
        group_counter = 1
        for root, member_ids in clusters.items():
            # Choose the canonical issue as the one with the highest priority score.
            canonical_id = max(
                member_ids,
                key=lambda iid: result_map[iid].priority_score,
            )
            canonical_result = result_map[canonical_id]
            canonical_issue = issue_map.get(canonical_id)

            # Aggregate severity — take the maximum across all members.
            severity = _max_severity([result_map[iid].severity for iid in member_ids])

            # Impact category — use the canonical issue's category.
            impact_category = canonical_result.impact_category

            # Priority score — use the maximum across all members.
            priority_score = max(result_map[iid].priority_score for iid in member_ids)

            # Summary — use the canonical issue's LLM summary.
            summary = canonical_result.summary

            # Title — derive from canonical issue.
            title = (
                canonical_issue.title
                if canonical_issue
                else f"Issue group {group_counter}"
            )

            # Tags — union of all member tags.
            all_tags: list[str] = []
            for iid in member_ids:
                all_tags.extend(result_map[iid].tags)
            tags = list(dict.fromkeys(all_tags))  # deduplicate preserving order

            group_id = f"group_{group_counter:03d}"
            group_counter += 1

            group = IssueGroup(
                id=group_id,
                title=title,
                severity=severity,
                impact_category=impact_category,
                priority_score=round(priority_score, 2),
                complexity=Complexity.UNKNOWN,  # filled in step 3
                fix_order=0,  # filled in step 4
                issue_ids=sorted(member_ids),
                canonical_issue_id=canonical_id,
                summary=summary,
                similar_closed_issue_ids=[],
                tags=tags,
            )
            groups.append(group)

        return groups

    # ------------------------------------------------------------------
    # Step 3 — Complexity estimation
    # ------------------------------------------------------------------

    def _estimate_complexity(
        self, groups: list[IssueGroup], issues: list[Issue]
    ) -> list[IssueGroup]:
        """Ask the LLM to estimate fix complexity for each issue group.

        Attaches :class:`~bug_triage.models.Complexity` values to each group
        in-place (actually creates new IssueGroup instances since Pydantic
        models are immutable by default — we rebuild the list).

        Args:
            groups: Issue groups without complexity estimates.
            issues: Original raw issues for full context.

        Returns:
            A new list of :class:`IssueGroup` objects with complexity set.
        """
        if not groups:
            return groups

        issue_map: dict[int, Issue] = {i.id: i for i in issues}

        # Attach Issue objects to each group for the template.
        groups_with_issues = [
            {
                "group": group,
                "issues": [issue_map[iid] for iid in group.issue_ids if iid in issue_map],
            }
            for group in groups
        ]

        # Build template context — groups is a list of objects with `.issues` attr.
        template_groups = []
        for entry in groups_with_issues:
            g = entry["group"]
            # Build a simple namespace object the template can use.
            template_groups.append(
                _TemplateGroup(
                    id=g.id,
                    title=g.title,
                    severity=g.severity.value,
                    impact_category=g.impact_category.value,
                    issue_ids=g.issue_ids,
                    summary=g.summary,
                    issues=entry["issues"],
                )
            )

        try:
            raw_response = self.llm_client.render_and_complete(
                "complexity_prompt.j2",
                system_prompt=_COMPLEXITY_SYSTEM_PROMPT,
                groups=template_groups,
            )
        except LLMError as exc:
            logger.warning(
                "Complexity estimation LLM call failed: %s.  "
                "Keeping complexity as UNKNOWN for all groups.",
                exc,
            )
            return groups

        complexity_map = self._parse_complexity_response(raw_response)

        updated_groups: list[IssueGroup] = []
        for group in groups:
            complexity = complexity_map.get(group.id, Complexity.UNKNOWN)
            updated = group.model_copy(update={"complexity": complexity})
            updated_groups.append(updated)

        return updated_groups

    def _parse_complexity_response(
        self, raw_response: str
    ) -> dict[str, Complexity]:
        """Parse the LLM complexity estimation response.

        Args:
            raw_response: Raw text from the LLM.

        Returns:
            A mapping from group_id to :class:`Complexity`.
        """
        cleaned = _strip_markdown_fences(raw_response)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse complexity response as JSON: %s.", exc
            )
            return {}

        if not isinstance(data, list):
            logger.warning("Complexity response is not a JSON array; ignoring.")
            return {}

        result: dict[str, Complexity] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            group_id = item.get("group_id", "")
            if not group_id:
                continue
            raw_complexity = str(item.get("complexity", "unknown")).lower().strip()
            try:
                complexity = Complexity(raw_complexity)
            except ValueError:
                complexity = Complexity.UNKNOWN
            result[group_id] = complexity

        return result

    # ------------------------------------------------------------------
    # Step 4 — Assign fix order
    # ------------------------------------------------------------------

    def _assign_fix_order(self, groups: list[IssueGroup]) -> list[IssueGroup]:
        """Sort groups by priority and assign ascending fix_order ranks.

        Sorting criteria (descending priority):
        1. Severity (critical > high > medium > low)
        2. Priority score (higher is more urgent)
        3. Number of issues in the group (more issues = more impact)

        Args:
            groups: Unranked issue groups.

        Returns:
            A new list of :class:`IssueGroup` objects with ``fix_order`` set.
        """
        _severity_rank = {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1,
        }

        sorted_groups = sorted(
            groups,
            key=lambda g: (
                _severity_rank.get(g.severity, 0),
                g.priority_score,
                len(g.issue_ids),
            ),
            reverse=True,
        )

        ranked: list[IssueGroup] = []
        for rank, group in enumerate(sorted_groups, start=1):
            ranked.append(group.model_copy(update={"fix_order": rank}))

        return ranked

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_empty_report(self) -> ReportOutput:
        """Return an empty ReportOutput with correct metadata."""
        metadata = ReportMetadata(
            generated_at=datetime.utcnow(),
            repository=self.repository,
            source_file=self.source_file,
            llm_provider=self.llm_client.provider.value,
            llm_model=self.llm_client.model,
            total_issues=0,
            total_groups=0,
            output_format=self.output_format,
        )
        return ReportOutput(metadata=metadata)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def triage_issues(
    issues: list[Issue],
    llm_client: LLMClient,
    repository: str = "",
    source_file: str = "",
    output_format: OutputFormat = OutputFormat.MARKDOWN,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> ReportOutput:
    """Convenience wrapper that creates a :class:`TriageEngine` and runs it.

    Args:
        issues: Raw issues to triage.
        llm_client: Configured LLM client.
        repository: GitHub repository identifier, if applicable.
        source_file: Local file path, if applicable.
        output_format: Desired report output format.
        batch_size: Number of issues per LLM batch.

    Returns:
        A fully populated :class:`~bug_triage.models.ReportOutput`.
    """
    engine = TriageEngine(
        llm_client=llm_client,
        batch_size=batch_size,
        repository=repository,
        source_file=source_file,
        output_format=output_format,
    )
    return engine.run(issues)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


class _TemplateGroup:
    """Lightweight data-holder used inside Jinja2 templates for complexity prompts.

    Jinja2 templates access attributes directly, so we use a simple class
    rather than a dict to allow dot-notation access.
    """

    def __init__(
        self,
        id: str,
        title: str,
        severity: str,
        impact_category: str,
        issue_ids: list[int],
        summary: str,
        issues: list[Issue],
    ) -> None:
        self.id = id
        self.title = title
        self.severity = severity
        self.impact_category = impact_category
        self.issue_ids = issue_ids
        self.summary = summary
        self.issues = issues


class _UnionFind:
    """Simple Union-Find (Disjoint Set Union) data structure.

    Used to cluster issue IDs into groups based on duplicate/related links.

    Args:
        elements: Initial set of integer IDs.
    """

    def __init__(self, elements: list[int]) -> None:
        self._parent: dict[int, int] = {e: e for e in elements}
        self._rank: dict[int, int] = {e: 0 for e in elements}

    def find(self, x: int) -> int:
        """Return the root representative of the set containing ``x``."""
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, x: int, y: int) -> None:
        """Merge the sets containing ``x`` and ``y``."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank.
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


def _chunk(lst: list, size: int) -> list[list]:
    """Split ``lst`` into sublists of at most ``size`` elements."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _strip_markdown_fences(text: str) -> str:
    """Remove Markdown code fences from an LLM response if present.

    Some LLMs wrap their JSON output in triple-backtick fences even when
    instructed not to.  This function strips the outer fence and any language
    tag so the result can be passed directly to :func:`json.loads`.

    Args:
        text: Raw LLM response text.

    Returns:
        Cleaned text with fences removed.
    """
    # Match ```json ... ``` or ``` ... ``` blocks.
    fence_pattern = re.compile(
        r"^```(?:[a-zA-Z]*)\s*\n?(.+?)\n?```\s*$", re.DOTALL
    )
    match = fence_pattern.search(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


_SEVERITY_ORDER = [
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
]


def _max_severity(severities: list[Severity]) -> Severity:
    """Return the highest (most critical) severity from a list.

    Args:
        severities: A non-empty list of :class:`Severity` values.

    Returns:
        The most critical severity present.
    """
    if not severities:
        return Severity.LOW
    for sev in _SEVERITY_ORDER:
        if sev in severities:
            return sev
    return severities[0]


def _default_triage_result(issue: Issue) -> TriageResult:
    """Create a conservative default :class:`TriageResult` for an issue.

    Used as a fallback when the LLM fails to return a result for an issue.

    Args:
        issue: The issue to create a default result for.

    Returns:
        A :class:`TriageResult` with medium severity and 50/100 priority.
    """
    return TriageResult(
        issue_id=issue.id,
        severity=Severity.MEDIUM,
        impact_category=ImpactCategory.OTHER,
        priority_score=50.0,
        summary=issue.title,
        reasoning="Default classification (LLM response unavailable).",
        duplicate_of=None,
        related_issue_ids=[],
        complexity=Complexity.UNKNOWN,
        tags=[],
    )


def _dict_to_triage_result(data: dict[str, Any]) -> TriageResult:
    """Convert a raw dict from the LLM JSON response to a :class:`TriageResult`.

    Args:
        data: A dict parsed from the LLM's JSON array response.

    Returns:
        A validated :class:`TriageResult` instance.

    Raises:
        ValueError: If required fields are missing or invalid.
        KeyError: If ``issue_id`` is absent from the dict.
    """
    issue_id = int(data["issue_id"])

    raw_severity = str(data.get("severity", "medium")).lower().strip()
    try:
        severity = Severity(raw_severity)
    except ValueError:
        logger.warning("Unknown severity '%s'; defaulting to medium.", raw_severity)
        severity = Severity.MEDIUM

    raw_impact = str(data.get("impact_category", "other")).lower().strip()
    try:
        impact_category = ImpactCategory(raw_impact)
    except ValueError:
        logger.warning(
            "Unknown impact_category '%s'; defaulting to other.", raw_impact
        )
        impact_category = ImpactCategory.OTHER

    raw_complexity = str(data.get("complexity", "unknown")).lower().strip()
    try:
        complexity = Complexity(raw_complexity)
    except ValueError:
        complexity = Complexity.UNKNOWN

    priority_score = float(data.get("priority_score", 50.0))
    priority_score = max(0.0, min(100.0, priority_score))

    duplicate_of_raw = data.get("duplicate_of")
    duplicate_of: Optional[int] = (
        int(duplicate_of_raw) if duplicate_of_raw is not None else None
    )

    related_raw = data.get("related_issue_ids", [])
    if not isinstance(related_raw, list):
        related_raw = []
    related_issue_ids = [int(r) for r in related_raw if r is not None]

    tags_raw = data.get("tags", [])
    if not isinstance(tags_raw, list):
        tags_raw = []
    tags = [str(t) for t in tags_raw]

    return TriageResult(
        issue_id=issue_id,
        severity=severity,
        impact_category=impact_category,
        priority_score=round(priority_score, 2),
        summary=str(data.get("summary", "")),
        reasoning=str(data.get("reasoning", "")),
        duplicate_of=duplicate_of,
        related_issue_ids=related_issue_ids,
        complexity=complexity,
        tags=tags,
    )
