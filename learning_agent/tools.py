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
Tools and Output Utilities for Kythe Agentic Deep Research Agent.

Provides output writers, summary formatters, and export integrations.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from exporters import ExportManager, atomic_write, compute_sha256, sanitize_filename
from models import Artifact, CurriculumModel, utc_now_iso

logger = logging.getLogger(__name__)


def save_markdown_curriculum(
    markdown_content: str,
    topic: str,
    output_dir: str = "output",
    filename: Optional[str] = None,
) -> str:
    """
    Save the generated markdown curriculum to a file atomically.

    Args:
        markdown_content: The markdown content to save
        topic: The research topic
        output_dir: Directory to save the file in
        filename: Optional custom filename

    Returns:
        Absolute path to the saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not filename:
        safe_topic = sanitize_filename(topic)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_topic}_{timestamp}.md"

    if not filename.endswith(".md"):
        filename += ".md"

    file_path = output_path / filename
    saved_path = atomic_write(str(file_path), markdown_content)
    print(f"[SAVED] Curriculum written to: {saved_path}")
    return saved_path


def save_research_metadata(
    state: Dict[str, Any],
    output_dir: str = "output",
    topic: str = "research",
) -> str:
    """
    Save research metadata and metrics to a JSON file atomically.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "topic": topic,
        "timestamp": utc_now_iso(),
        "iteration_count": state.get("iteration_count", 0),
        "total_sources": len(state.get("sources", [])),
        "explored_topics": state.get("explored_topics", []),
        "knowledge_gaps": state.get("knowledge_gaps", []),
        "depth_scores": state.get("depth_scores", {}),
        "sources": state.get("sources", []),
    }

    safe_topic = sanitize_filename(topic)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_topic}_metadata_{timestamp}.json"
    file_path = output_path / filename

    saved_path = atomic_write(str(file_path), json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"[SAVED] Research metadata written to: {saved_path}")
    return saved_path


def print_research_summary(state: Dict[str, Any]) -> None:
    """
    Print a clean terminal summary of the research process.
    """
    print("\n" + "=" * 80)
    print("KYTHE RESEARCH SUMMARY")
    print("=" * 80)

    iterations = state.get("iteration_count", 0)
    sources = state.get("sources", [])
    explored = state.get("explored_topics", [])
    gaps = state.get("knowledge_gaps", [])
    depth_scores = state.get("depth_scores", {})

    print(f"Iterations Completed: {iterations}")
    print(f"Total Sources Referenced: {len(sources)}")
    print(f"Topics Explored: {len(explored)}")
    print(f"Knowledge Gaps: {len(gaps)}")

    if depth_scores:
        print("\nSection Depth Metrics:")
        for section, scores in depth_scores.items():
            completeness = scores.get("completeness", 0)
            status = "[PASS]" if completeness >= 70 else "[WARN]" if completeness >= 50 else "[FAIL]"
            print(f"  {status} {section[:50]}: {completeness:.1f}% complete")

    if sources:
        print("\nKey Sources:")
        for i, source in enumerate(sources[:5], 1):
            title = source.get("title") or "Untitled Source"
            url = source.get("url") or source.get("canonical_url") or "N/A"
            print(f"  {i}. {title}")
            print(f"     {url}")

    print("=" * 80 + "\n")


def create_quick_reference(state: Dict[str, Any], output_dir: str = "output") -> str:
    """
    Create a quick reference markdown file with key insights.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    topic = state.get("research_topic", "Topic")
    curriculum = state.get("curriculum_outline", "")
    sources = state.get("sources", [])

    content = [
        f"# {topic} - Quick Reference\n\n",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n",
        "## Core Curriculum\n\n",
        f"{curriculum}\n\n",
        "## Key Sources\n\n",
    ]

    for source in sources[:10]:
        title = source.get("title", "Untitled")
        url = source.get("url") or source.get("canonical_url") or ""
        snippet = source.get("snippet", "")
        content.append(f"- **[{title}]({url})**\n")
        if snippet:
            content.append(f"  > {snippet}\n")

    safe_topic = sanitize_filename(topic)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_path = output_path / f"{safe_topic}_quickref_{timestamp}.md"

    saved_path = atomic_write(str(file_path), "".join(content))
    return saved_path
