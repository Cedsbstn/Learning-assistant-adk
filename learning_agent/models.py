# Copyright 2026 Cedric Sebastian
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Data models and typed definitions for the Kythe Agentic Deep Research Agent.
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class RunStatus(str, enum.Enum):
    """Lifecycle status of a research run."""
    INITIALIZING = "INITIALIZING"
    OUTLINE_REVIEW = "OUTLINE_REVIEW"
    RESEARCHING = "RESEARCHING"
    SYNTHESIZING = "SYNTHESIZING"
    EXPORTING = "EXPORTING"
    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"


class SectionStatus(str, enum.Enum):
    """Lifecycle status of a curriculum section."""
    PENDING = "PENDING"
    RESEARCHING = "RESEARCHING"
    EVALUATING = "EVALUATING"
    NEEDS_MORE_RESEARCH = "NEEDS_MORE_RESEARCH"
    COMPLETE = "COMPLETE"
    EXHAUSTED = "EXHAUSTED"
    FAILED = "FAILED"


class PassStatus(str, enum.Enum):
    """Lifecycle status of a section research pass."""
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"


class GapStatus(str, enum.Enum):
    """Lifecycle status of a knowledge gap."""
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    WAIVED = "WAIVED"
    SUPERSEDED = "SUPERSEDED"


class GapCategory(str, enum.Enum):
    """Category of knowledge gap."""
    DETAIL = "detail"
    SOURCES = "sources"
    EXAMPLES = "examples"
    TECHNICAL = "technical"
    CITATION = "citation"
    STYLE = "style"
    CONFLICT = "conflict"
    FRESHNESS = "freshness"
    OTHER = "other"


@dataclass
class Run:
    """A top-level research execution."""
    run_id: str
    topic: str
    status: RunStatus = RunStatus.INITIALIZING
    config_json: str = "{}"
    outline_version: int = 1
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Run":
        return cls(
            run_id=row["run_id"],
            topic=row["topic"],
            status=RunStatus(row["status"]),
            config_json=row.get("config_json", "{}"),
            outline_version=row.get("outline_version", 1),
            created_at=row.get("created_at", utc_now_iso()),
            updated_at=row.get("updated_at", utc_now_iso()),
            completed_at=row.get("completed_at"),
        )


@dataclass
class Section:
    """A discrete curriculum module work item."""
    section_id: str
    run_id: str
    ordinal: int
    title: str
    objectives_json: str = "[]"
    depth_target: str = "intermediate"  # basic, intermediate, advanced
    status: SectionStatus = SectionStatus.PENDING
    attempt_count: int = 0
    latest_score: Optional[float] = None
    updated_at: str = field(default_factory=utc_now_iso)

    @property
    def objectives(self) -> List[str]:
        try:
            return json.loads(self.objectives_json)
        except Exception:
            return []

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["objectives"] = self.objectives
        return data

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Section":
        return cls(
            section_id=row["section_id"],
            run_id=row["run_id"],
            ordinal=row["ordinal"],
            title=row["title"],
            objectives_json=row.get("objectives_json", "[]"),
            depth_target=row.get("depth_target", "intermediate"),
            status=SectionStatus(row["status"]),
            attempt_count=row.get("attempt_count", 0),
            latest_score=row.get("latest_score"),
            updated_at=row.get("updated_at", utc_now_iso()),
        )


@dataclass
class Pass:
    """A single research iteration pass on a section."""
    pass_id: str
    section_id: str
    ordinal: int
    status: PassStatus = PassStatus.STARTED
    started_at: str = field(default_factory=utc_now_iso)
    completed_at: Optional[str] = None
    query_plan_json: Optional[str] = None
    draft_text: Optional[str] = None
    evaluation_json: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Pass":
        return cls(
            pass_id=row["pass_id"],
            section_id=row["section_id"],
            ordinal=row["ordinal"],
            status=PassStatus(row["status"]),
            started_at=row.get("started_at", utc_now_iso()),
            completed_at=row.get("completed_at"),
            query_plan_json=row.get("query_plan_json"),
            draft_text=row.get("draft_text"),
            evaluation_json=row.get("evaluation_json"),
        )


@dataclass
class Source:
    """A fetched and extracted external source."""
    source_id: str
    canonical_url: str
    title: str
    domain: str
    fetched_at: str = field(default_factory=utc_now_iso)
    content_hash: str = ""
    content_type: str = "text/html"
    extraction_status: str = "success"  # success, failed, cached, skipped
    extracted_text: Optional[str] = None
    snippet: Optional[str] = None
    cache_pointer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Source":
        return cls(
            source_id=row["source_id"],
            canonical_url=row["canonical_url"],
            title=row["title"],
            domain=row["domain"],
            fetched_at=row.get("fetched_at", utc_now_iso()),
            content_hash=row.get("content_hash", ""),
            content_type=row.get("content_type", "text/html"),
            extraction_status=row.get("extraction_status", "success"),
            extracted_text=row.get("extracted_text"),
            snippet=row.get("snippet"),
            cache_pointer=row.get("cache_pointer"),
        )


@dataclass
class SectionSource:
    """Association between a section, pass, and source."""
    section_id: str
    source_id: str
    pass_id: str
    relevance_score: float = 1.0
    selected: bool = True
    citation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "SectionSource":
        return cls(
            section_id=row["section_id"],
            source_id=row["source_id"],
            pass_id=row["pass_id"],
            relevance_score=row.get("relevance_score", 1.0),
            selected=bool(row.get("selected", 1)),
            citation_count=row.get("citation_count", 0),
        )


@dataclass
class Gap:
    """An actionable missing knowledge gap."""
    gap_id: str
    section_id: str
    category: GapCategory
    description: str
    severity: str = "normal"  # critical, normal, minor
    status: GapStatus = GapStatus.OPEN
    created_pass_id: Optional[str] = None
    resolved_pass_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Gap":
        return cls(
            gap_id=row["gap_id"],
            section_id=row["section_id"],
            category=GapCategory(row["category"]),
            description=row["description"],
            severity=row.get("severity", "normal"),
            status=GapStatus(row["status"]),
            created_pass_id=row.get("created_pass_id"),
            resolved_pass_id=row.get("resolved_pass_id"),
        )


@dataclass
class Claim:
    """A factual claim unit extracted from a draft."""
    claim_id: str
    section_id: str
    pass_id: str
    claim_text: str
    claim_hash: str
    source_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Artifact:
    """A generated export artifact file."""
    artifact_id: str
    run_id: str
    type: str  # markdown, html, pdf, quiz, flashcards_json, flashcards_csv, metadata
    path: str
    sha256: str
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Artifact":
        return cls(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            type=row["type"],
            path=row["path"],
            sha256=row["sha256"],
            created_at=row.get("created_at", utc_now_iso()),
        )


@dataclass
class GapSpec:
    """Specification for generating or resolving a gap."""
    category: str
    description: str
    severity: str = "normal"


@dataclass
class SectionEvaluation:
    """Outcome of evaluating a synthesized section draft."""
    passed: bool
    score: float
    word_count: int
    source_count: int
    unique_domains: int
    has_examples: bool
    has_technical_details: bool
    citation_coverage: float
    open_gaps: List[GapSpec] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "word_count": self.word_count,
            "source_count": self.source_count,
            "unique_domains": self.unique_domains,
            "has_examples": self.has_examples,
            "has_technical_details": self.has_technical_details,
            "citation_coverage": self.citation_coverage,
            "open_gaps": [asdict(g) if isinstance(g, GapSpec) else g for g in self.open_gaps],
            "reasons": self.reasons,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SectionEvaluation":
        open_gaps = [
            GapSpec(**g) if isinstance(g, dict) else g
            for g in data.get("open_gaps", [])
        ]
        return cls(
            passed=bool(data.get("passed", False)),
            score=float(data.get("score", 0.0)),
            word_count=int(data.get("word_count", 0)),
            source_count=int(data.get("source_count", 0)),
            unique_domains=int(data.get("unique_domains", 0)),
            has_examples=bool(data.get("has_examples", False)),
            has_technical_details=bool(data.get("has_technical_details", False)),
            citation_coverage=float(data.get("citation_coverage", 0.0)),
            open_gaps=open_gaps,
            reasons=list(data.get("reasons", [])),
        )


@dataclass
class OutlineSection:
    """A section proposal during curriculum outline generation."""
    ordinal: int
    title: str
    objectives: List[str] = field(default_factory=list)
    depth_target: str = "intermediate"


@dataclass
class CurriculumOutline:
    """Generated outline before review."""
    topic: str
    prerequisites: List[str] = field(default_factory=list)
    sections: List[OutlineSection] = field(default_factory=list)
    raw_markdown: str = ""


@dataclass
class QuizQuestion:
    """A structured assessment question."""
    question_id: str
    type: str  # multiple_choice, true_false, open_ended
    prompt: str
    choices: Optional[List[str]] = None
    answer: str = ""
    explanation: str = ""
    section_id: str = ""
    source_ids: List[str] = field(default_factory=list)


@dataclass
class Flashcard:
    """A learning flashcard."""
    card_id: str
    front: str
    back: str
    section_id: str = ""
    tags: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    difficulty: str = "intermediate"  # basic, intermediate, advanced


@dataclass
class ExtractedDocument:
    """Clean extracted content from a fetched web page."""
    canonical_url: str
    title: str
    author: Optional[str] = None
    published_at: Optional[str] = None
    text: str = ""
    headings: List[str] = field(default_factory=list)
    code_blocks: List[str] = field(default_factory=list)
    content_hash: str = ""
    word_count: int = 0
    extraction_method: str = "bs4_html"
    warnings: List[str] = field(default_factory=list)


@dataclass
class FetchResult:
    """Result of an HTTP fetch operation."""
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    body_bytes: bytes
    fetched_at: str = field(default_factory=utc_now_iso)
    error_reason: Optional[str] = None


@dataclass
class EvidenceChunk:
    """Bounded chunk of source text for prompt grounding."""
    chunk_id: str
    source_id: str
    source_title: str
    source_url: str
    text: str
    token_estimate: int = 0


@dataclass
class CurriculumModel:
    """Canonical structured representation of the entire research output."""
    topic: str
    generated_at: str
    run_id: str
    sections_content: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    quizzes: List[Dict[str, Any]] = field(default_factory=list)
    flashcards: List[Dict[str, Any]] = field(default_factory=list)
    open_gaps: List[Dict[str, Any]] = field(default_factory=list)
    exhausted_sections: List[str] = field(default_factory=list)
