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
Deterministic Quality Evaluator for Kythe Agentic Deep Research Agent.

Evaluates section depth, per-section evidence, source domain diversity,
practical examples, technical depth, citation coverage, and antislop compliance.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from config import DEFAULT_BANNED_SLOP_WORDS, ResearchConfig
from models import GapCategory, GapSpec, Section, SectionEvaluation, Source

logger = logging.getLogger(__name__)

# Regular expressions for citations, markdown, and antislop detection
CITE_TAG_PATTERN = re.compile(r'<cite\s+source\s*=\s*["\']?\s*(src-\d+)\s*["\']?\s*/>', re.IGNORECASE)
MARKDOWN_CITE_PATTERN = re.compile(r'\[(?:[^\]]+)\]\((?:https?://[^\)]+|#(src-\d+))\)', re.IGNORECASE)
RAW_SRC_PATTERN = re.compile(r'\b(src-\d+)\b', re.IGNORECASE)
CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```')
HEADING_PATTERN = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
DASH_PATTERN = re.compile(r'[—–]')

# Heuristic terms for examples and technical substance
EXAMPLE_INDICATORS: Set[str] = {
    "example",
    "case study",
    "for instance",
    "such as",
    "e.g.",
    "walkthrough",
    "demonstration",
    "sample code",
    "illustration",
    "practical application",
}

TECHNICAL_INDICATORS: Set[str] = {
    "algorithm",
    "implementation",
    "architecture",
    "methodology",
    "technique",
    "approach",
    "framework",
    "mechanism",
    "protocol",
    "data structure",
    "complexity",
    "tradeoff",
    "concurrency",
    "memory layout",
    "lifecycle",
}


def extract_cited_source_ids(text: str) -> Set[str]:
    """Extract all cited source IDs (e.g. src-1, src-2) from draft text."""
    cited = set()
    for match in CITE_TAG_PATTERN.finditer(text):
        cited.add(match.group(1).lower())
    for match in RAW_SRC_PATTERN.finditer(text):
        cited.add(match.group(1).lower())
    return cited


def calculate_citation_coverage(text: str, valid_source_ids: Set[str]) -> Tuple[float, int, int]:
    """
    Calculate citation coverage ratio across substantive factual units.
    Returns (coverage_ratio, cited_units, total_factual_units).
    """
    if not text:
        return 0.0, 0, 0

    text_without_code = CODE_BLOCK_PATTERN.sub("", text)
    raw_paragraphs = [p.strip() for p in text_without_code.split("\n\n") if p.strip()]
    paragraphs = []
    for p in raw_paragraphs:
        if p.startswith("#") and "\n" not in p:
            continue
        words_in_p = p.split()
        if len(words_in_p) >= 4:
            paragraphs.append(p)

    if not paragraphs:
        paragraphs = [text_without_code.strip()] if text_without_code.strip() else []

    total_units = len(paragraphs)
    if total_units == 0:
        return 0.0, 0, 0

    cited_units = 0
    for p in paragraphs:
        cited_in_p = extract_cited_source_ids(p)
        if any(sid in valid_source_ids for sid in cited_in_p):
            cited_units += 1

    coverage = cited_units / total_units
    return round(coverage, 3), cited_units, total_units


def detect_slop_patterns(text: str, banned_words: Optional[List[str]] = None) -> List[str]:
    """Identify AI slop words and forbidden punctuation tells in text."""
    detected = []
    text_lower = text.lower()
    words_to_check = banned_words if banned_words is not None else DEFAULT_BANNED_SLOP_WORDS

    for word in words_to_check:
        pattern = r'\b' + re.escape(word.lower()) + r'\b'
        if re.search(pattern, text_lower):
            detected.append(word)

    if DASH_PATTERN.search(text):
        detected.append("em/en dash punctuation")

    return detected


class QualityEvaluator:
    """Evaluates section draft against deterministic quality and antislop gates."""

    def __init__(self, config: ResearchConfig):
        self.config = config

    def evaluate_section(
        self,
        section: Section,
        draft_text: str,
        section_sources: List[Source],
    ) -> SectionEvaluation:
        """
        Evaluate a synthesized section draft against configured quality gates:
        1. Minimum detail (words adjusted for depth target).
        2. Per-section evidence (unique sources linked to this section).
        3. Source diversity (unique domains).
        4. Practical examples (indicators or code blocks).
        5. Technical depth (indicators, mechanics, architectural concepts).
        6. Citation coverage (factual paragraphs containing valid source links).
        7. Antislop compliance (free from banned buzzwords and em dashes).
        """
        reasons: List[str] = []
        gaps: List[GapSpec] = []
        passed = True

        words = draft_text.split()
        word_count = len(words)

        depth_target = (section.depth_target or "intermediate").lower()
        if depth_target == "basic":
            target_words = max(200, int(self.config.min_word_count * 0.7))
        elif depth_target == "advanced":
            target_words = int(self.config.min_word_count * 1.3)
        else:
            target_words = self.config.min_word_count

        # Minimum detail check
        if word_count < target_words:
            passed = False
            desc = f"Insufficient word count ({word_count}/{target_words} words for {depth_target} depth)"
            reasons.append(desc)
            gaps.append(GapSpec(category=GapCategory.DETAIL.value, description=desc, severity="critical"))

        # Section-scoped source count
        source_count = len(section_sources)
        if source_count < self.config.min_sources:
            passed = False
            desc = f"Insufficient section sources ({source_count}/{self.config.min_sources} required)"
            reasons.append(desc)
            gaps.append(GapSpec(category=GapCategory.SOURCES.value, description=desc, severity="critical"))

        # Source diversity across unique domains
        unique_domains = {s.domain for s in section_sources if s.domain}
        domain_count = len(unique_domains)
        if domain_count < self.config.min_unique_domains and source_count >= self.config.min_unique_domains:
            passed = False
            desc = f"Low source diversity ({domain_count}/{self.config.min_unique_domains} unique domains)"
            reasons.append(desc)
            gaps.append(GapSpec(category=GapCategory.SOURCES.value, description=desc, severity="normal"))

        # Practical examples or code blocks
        draft_lower = draft_text.lower()
        has_code_blocks = len(CODE_BLOCK_PATTERN.findall(draft_text)) > 0
        has_example_terms = any(term in draft_lower for term in EXAMPLE_INDICATORS)
        has_examples = has_code_blocks or has_example_terms

        if not has_examples and self.config.examples_weight > 0:
            passed = False
            desc = "Missing practical examples, case studies, or code implementations"
            reasons.append(desc)
            gaps.append(GapSpec(category=GapCategory.EXAMPLES.value, description=desc, severity="normal"))

        # Technical depth and architectural detail
        has_technical_terms = any(term in draft_lower for term in TECHNICAL_INDICATORS)
        has_headings = len(HEADING_PATTERN.findall(draft_text)) >= 2
        has_technical_details = has_technical_terms or (has_code_blocks and has_headings)

        if not has_technical_details and self.config.technical_weight > 0:
            passed = False
            desc = "Lacks technical depth, architectural details, or implementation tradeoffs"
            reasons.append(desc)
            gaps.append(GapSpec(category=GapCategory.TECHNICAL.value, description=desc, severity="normal"))

        # Sentence-level citation coverage
        valid_source_ids = {s.source_id.lower() for s in section_sources if s.source_id}
        coverage, cited_units, total_units = calculate_citation_coverage(draft_text, valid_source_ids)

        if coverage < self.config.min_citation_coverage and source_count > 0:
            passed = False
            desc = f"Citation coverage low ({coverage * 100:.1f}% < {self.config.min_citation_coverage * 100:.0f}% required, {cited_units}/{total_units} units cited)"
            reasons.append(desc)
            gaps.append(GapSpec(category=GapCategory.CITATION.value, description=desc, severity="critical"))

        # Style and antislop filter
        slop_penalty = 0.0
        if getattr(self.config, "enable_antislop", True):
            detected_slop = detect_slop_patterns(draft_text, getattr(self.config, "banned_words", []))
            if detected_slop:
                slop_penalty = min(20.0, len(detected_slop) * 5.0)
                desc = f"Detected AI slop tokens: {', '.join(detected_slop[:4])}"
                reasons.append(desc)
                gaps.append(GapSpec(category=GapCategory.STYLE.value, description=f"Rewrite without AI slop phrases: {', '.join(detected_slop[:4])}", severity="normal"))

        # Composite quality score calculation (0.0 - 100.0)
        score_components = [
            min(1.0, word_count / target_words) * 30.0,
            min(1.0, source_count / max(1, self.config.min_sources)) * 25.0,
            (1.0 if has_examples else 0.0) * 15.0,
            (1.0 if has_technical_details else 0.0) * 15.0,
            min(1.0, coverage / max(0.1, self.config.min_citation_coverage)) * 15.0,
        ]
        raw_score = sum(score_components) - slop_penalty
        score = round(max(0.0, min(100.0, raw_score)), 1)

        if passed and not gaps:
            reasons.append("All quality gates passed successfully")

        return SectionEvaluation(
            passed=passed,
            score=score,
            word_count=word_count,
            source_count=source_count,
            unique_domains=domain_count,
            has_examples=has_examples,
            has_technical_details=has_technical_details,
            citation_coverage=coverage,
            open_gaps=gaps,
            reasons=reasons,
        )


DeterministicQualityEvaluator = QualityEvaluator
