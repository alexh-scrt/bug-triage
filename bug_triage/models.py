"""Pydantic data models for the bug_triage pipeline.

This module defines the core data contracts used throughout the entire
bug_triage pipeline: Issue, TriageResult, IssueGroup, and ReportOutput.
All models use Pydantic v2 for validation and serialization.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Severity(str, Enum):
    """Severity level for a triaged issue or issue group."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImpactCategory(str, Enum):
    """Impact category describing the nature of the bug."""

    CRASH = "crash"
    PERFORMANCE = "performance"
    SECURITY = "security"
    UX = "ux"
    DATA_LOSS = "data_loss"
    REGRESSION = "regression"
    OTHER = "other"


class Complexity(str, Enum):
    """Estimated fix complexity for an issue or issue group."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class OutputFormat(str, Enum):
    """Supported report output formats."""

    MARKDOWN = "markdown"
    JSON = "json"


class LLMProvider(str, Enum):
    """Supported LLM backend providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Issue(BaseModel):
    """Represents a single bug report or GitHub issue ingested into the pipeline.

    Attributes:
        id: Unique identifier for the issue (GitHub issue number or local ID).
        title: Short summary title of the issue.
        body: Full description/body text of the issue.
        labels: List of label strings attached to the issue.
        created_at: Timestamp when the issue was created.
        updated_at: Timestamp when the issue was last updated.
        url: URL linking to the original issue.
        author: Username or identifier of the issue author.
        state: Current state of the issue (open/closed).
        comments_count: Number of comments on the issue.
        repository: Repository identifier in "owner/repo" format.
    """

    id: int = Field(..., description="Unique issue identifier")
    title: str = Field(..., min_length=1, description="Issue title/summary")
    body: str = Field(default="", description="Full issue description")
    labels: list[str] = Field(default_factory=list, description="Issue labels")
    created_at: Optional[datetime] = Field(default=None, description="Issue creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Issue last-updated timestamp")
    url: str = Field(default="", description="URL to the original issue")
    author: str = Field(default="", description="Issue author username")
    state: str = Field(default="open", description="Issue state (open/closed)")
    comments_count: int = Field(default=0, ge=0, description="Number of comments")
    repository: str = Field(default="", description="Repository in owner/repo format")

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, v: str) -> str:
        """Ensure title is not just whitespace."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Issue title must not be blank or whitespace only")
        return stripped

    @field_validator("body", mode="before")
    @classmethod
    def coerce_none_body_to_empty(cls, v: object) -> str:
        """Convert None body to empty string."""
        if v is None:
            return ""
        return str(v)

    @field_validator("labels", mode="before")
    @classmethod
    def coerce_labels(cls, v: object) -> list[str]:
        """Accept None or a comma-separated string in addition to a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [label.strip() for label in v.split(",") if label.strip()]
        return list(v)

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": 42,
                "title": "App crashes on login with malformed token",
                "body": "Steps to reproduce: 1. Send a malformed JWT...",
                "labels": ["bug", "security"],
                "created_at": "2024-01-10T09:00:00Z",
                "url": "https://github.com/example/repo/issues/42",
                "author": "alice",
                "state": "open",
                "comments_count": 3,
                "repository": "example/repo",
            }
        }
    }


class TriageResult(BaseModel):
    """Triage classification result for a single issue produced by the LLM.

    Attributes:
        issue_id: ID of the issue this result corresponds to.
        severity: Classified severity level.
        impact_category: Classified impact category.
        priority_score: Numeric priority score from 0–100 (higher = more urgent).
        summary: LLM-generated one-sentence summary of the issue.
        reasoning: LLM explanation for the severity/impact classification.
        duplicate_of: ID of the canonical issue this is a duplicate of, if any.
        related_issue_ids: IDs of related (non-duplicate) issues.
        complexity: Estimated fix complexity.
        tags: Additional classification tags.
    """

    issue_id: int = Field(..., description="ID of the triaged issue")
    severity: Severity = Field(..., description="Classified severity level")
    impact_category: ImpactCategory = Field(
        default=ImpactCategory.OTHER, description="Impact category"
    )
    priority_score: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Priority score from 0 (lowest) to 100 (highest)",
    )
    summary: str = Field(default="", description="One-sentence LLM summary of the issue")
    reasoning: str = Field(
        default="", description="LLM reasoning for the classification"
    )
    duplicate_of: Optional[int] = Field(
        default=None,
        description="ID of canonical issue if this is a duplicate",
    )
    related_issue_ids: list[int] = Field(
        default_factory=list,
        description="IDs of related (non-duplicate) issues",
    )
    complexity: Complexity = Field(
        default=Complexity.UNKNOWN, description="Estimated fix complexity"
    )
    tags: list[str] = Field(
        default_factory=list, description="Additional classification tags"
    )

    @field_validator("priority_score")
    @classmethod
    def round_priority_score(cls, v: float) -> float:
        """Round priority score to two decimal places."""
        return round(v, 2)

    model_config = {
        "json_schema_extra": {
            "example": {
                "issue_id": 42,
                "severity": "critical",
                "impact_category": "security",
                "priority_score": 97.5,
                "summary": "JWT validation can be bypassed with a malformed token.",
                "reasoning": "Authentication bypass is a critical security vulnerability.",
                "duplicate_of": None,
                "related_issue_ids": [35, 38],
                "complexity": "high",
                "tags": ["auth", "jwt"],
            }
        }
    }


class IssueGroup(BaseModel):
    """A cluster of related or duplicate issues grouped together after triage.

    Attributes:
        id: Unique group identifier (e.g., "group_001").
        title: Descriptive title for the group.
        severity: Highest severity among issues in the group.
        impact_category: Dominant impact category for the group.
        priority_score: Aggregate priority score for the group.
        complexity: Estimated fix complexity for the group.
        fix_order: Suggested fix order rank (1 = fix first).
        issue_ids: List of issue IDs belonging to this group.
        canonical_issue_id: The primary/canonical issue representing the group.
        summary: LLM-generated summary describing the group.
        similar_closed_issue_ids: IDs of similar closed/resolved issues for reference.
        tags: Shared classification tags for the group.
    """

    id: str = Field(..., description="Unique group identifier")
    title: str = Field(..., min_length=1, description="Descriptive group title")
    severity: Severity = Field(..., description="Highest severity in the group")
    impact_category: ImpactCategory = Field(
        default=ImpactCategory.OTHER, description="Dominant impact category"
    )
    priority_score: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Aggregate priority score 0–100",
    )
    complexity: Complexity = Field(
        default=Complexity.UNKNOWN, description="Estimated fix complexity"
    )
    fix_order: int = Field(
        default=0, ge=0, description="Suggested fix order (1 = highest priority)"
    )
    issue_ids: list[int] = Field(
        default_factory=list, description="All issue IDs in this group"
    )
    canonical_issue_id: Optional[int] = Field(
        default=None,
        description="Primary issue ID representing this group",
    )
    summary: str = Field(default="", description="Group summary from the LLM")
    similar_closed_issue_ids: list[int] = Field(
        default_factory=list,
        description="Similar closed issue IDs for reference",
    )
    tags: list[str] = Field(default_factory=list, description="Shared group tags")

    @model_validator(mode="after")
    def canonical_in_group(self) -> "IssueGroup":
        """If canonical_issue_id is set, verify it is a member of issue_ids."""
        if self.canonical_issue_id is not None and self.issue_ids:
            if self.canonical_issue_id not in self.issue_ids:
                raise ValueError(
                    f"canonical_issue_id {self.canonical_issue_id} must be in issue_ids {self.issue_ids}"
                )
        return self

    @field_validator("priority_score")
    @classmethod
    def round_score(cls, v: float) -> float:
        """Round priority score to two decimal places."""
        return round(v, 2)

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "group_001",
                "title": "Authentication bypass vulnerability",
                "severity": "critical",
                "impact_category": "security",
                "priority_score": 98.0,
                "complexity": "high",
                "fix_order": 1,
                "issue_ids": [234, 241, 255],
                "canonical_issue_id": 234,
                "summary": "Users can bypass JWT validation by sending a malformed token.",
                "similar_closed_issue_ids": [189],
                "tags": ["auth", "jwt", "security"],
            }
        }
    }


class ReportMetadata(BaseModel):
    """Metadata section of a triage report.

    Attributes:
        generated_at: UTC timestamp when the report was generated.
        repository: Repository identifier in "owner/repo" format, or empty for local files.
        source_file: Path to the local source file, if applicable.
        llm_provider: LLM provider used for triage.
        llm_model: Specific model name used.
        total_issues: Total number of raw issues analyzed.
        total_groups: Number of groups after deduplication/clustering.
        output_format: Format of the rendered report.
    """

    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of report generation",
    )
    repository: str = Field(
        default="", description="GitHub repository in owner/repo format"
    )
    source_file: str = Field(
        default="", description="Local source file path, if applicable"
    )
    llm_provider: str = Field(default="", description="LLM provider used")
    llm_model: str = Field(default="", description="LLM model name used")
    total_issues: int = Field(
        default=0, ge=0, description="Total number of raw issues analyzed"
    )
    total_groups: int = Field(
        default=0, ge=0, description="Number of issue groups after deduplication"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MARKDOWN, description="Report output format"
    )


class ReportOutput(BaseModel):
    """The complete structured output of the triage pipeline, ready for rendering.

    Attributes:
        metadata: Report metadata including timestamps and settings.
        groups: Ordered list of issue groups, sorted by fix_order ascending.
        raw_issues: The original issues that were analyzed.
        triage_results: Per-issue triage classification results.
        ungrouped_issue_ids: Issue IDs that were not assigned to any group.
    """

    metadata: ReportMetadata = Field(
        default_factory=ReportMetadata, description="Report metadata"
    )
    groups: list[IssueGroup] = Field(
        default_factory=list, description="Issue groups ordered by priority"
    )
    raw_issues: list[Issue] = Field(
        default_factory=list, description="Original raw issues"
    )
    triage_results: list[TriageResult] = Field(
        default_factory=list, description="Per-issue triage results"
    )
    ungrouped_issue_ids: list[int] = Field(
        default_factory=list,
        description="Issue IDs not assigned to any group",
    )

    def get_groups_by_severity(self, severity: Severity) -> list[IssueGroup]:
        """Return all issue groups matching the given severity level.

        Args:
            severity: The severity level to filter by.

        Returns:
            A list of IssueGroup objects with the specified severity.
        """
        return [g for g in self.groups if g.severity == severity]

    def get_issue_by_id(self, issue_id: int) -> Optional[Issue]:
        """Look up a raw issue by its ID.

        Args:
            issue_id: The issue ID to search for.

        Returns:
            The matching Issue object, or None if not found.
        """
        for issue in self.raw_issues:
            if issue.id == issue_id:
                return issue
        return None

    def get_triage_result_by_id(self, issue_id: int) -> Optional[TriageResult]:
        """Look up a triage result by issue ID.

        Args:
            issue_id: The issue ID to search for.

        Returns:
            The matching TriageResult, or None if not found.
        """
        for result in self.triage_results:
            if result.issue_id == issue_id:
                return result
        return None

    def sorted_groups(self) -> list[IssueGroup]:
        """Return groups sorted by fix_order ascending, then priority_score descending.

        Returns:
            A sorted list of IssueGroup objects.
        """
        return sorted(
            self.groups,
            key=lambda g: (g.fix_order if g.fix_order > 0 else 9999, -g.priority_score),
        )

    model_config = {
        "json_schema_extra": {
            "example": {
                "metadata": {
                    "generated_at": "2024-01-15T14:32:00Z",
                    "repository": "acme-corp/backend-api",
                    "llm_provider": "openai",
                    "llm_model": "gpt-4o",
                    "total_issues": 47,
                    "total_groups": 31,
                    "output_format": "markdown",
                },
                "groups": [],
                "raw_issues": [],
                "triage_results": [],
                "ungrouped_issue_ids": [],
            }
        }
    }
