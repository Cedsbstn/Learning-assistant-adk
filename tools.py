# Copyright 2025 Cedric Sebastian
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

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def save_markdown_curriculum(
    markdown_content: str,
    topic: str,
    output_dir: str = "output",
    filename: Optional[str] = None
) -> str:
    """
    Save the generated markdown curriculum to a file.

    Args:
        markdown_content: The markdown content to save
        topic: The research topic (used for filename if not provided)
        output_dir: Directory to save the file in
        filename: Optional custom filename

    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    if not filename:
        # Sanitize topic for filename
        safe_topic = "".join(c if c.isalnum() or c in (
            ' ', '-', '_') else '_' for c in topic)
        safe_topic = safe_topic.replace(' ', '_')[:50]  # Limit length
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_topic}_{timestamp}.md"

    # Ensure .md extension
    if not filename.endswith('.md'):
        filename += '.md'

    file_path = output_path / filename

    # Save the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"✅ Curriculum saved to: {file_path.absolute()}")
    return str(file_path.absolute())


def save_research_metadata(
    state: Dict[str, Any],
    output_dir: str = "output",
    topic: str = "research"
) -> str:
    """
    Save research metadata and metrics to a JSON file.

    Args:
        state: The research state containing metrics
        output_dir: Directory to save the file in
        topic: The research topic

    Returns:
        Path to the saved metadata file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract relevant metadata
    metadata = {
        "topic": topic,
        "timestamp": datetime.now().isoformat(),
        "iteration_count": state.get("iteration_count", 0),
        "total_sources": len(state.get("sources", [])),
        "explored_topics": state.get("explored_topics", []),
        "knowledge_gaps": state.get("knowledge_gaps", []),
        "depth_scores": state.get("depth_scores", {}),
        "sources": state.get("sources", []),
    }

    # Generate filename
    safe_topic = "".join(c if c.isalnum() or c in (
        ' ', '-', '_') else '_' for c in topic)
    safe_topic = safe_topic.replace(' ', '_')[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_topic}_metadata_{timestamp}.json"

    file_path = output_path / filename

    # Save the file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"📊 Research metadata saved to: {file_path.absolute()}")
    return str(file_path.absolute())


def print_research_summary(state: Dict[str, Any]) -> None:
    """
    Print a summary of the research process.

    Args:
        state: The research state containing metrics
    """
    print("\n" + "="*80)
    print("📚 RESEARCH SUMMARY")
    print("="*80)

    iterations = state.get("iteration_count", 0)
    sources = state.get("sources", [])
    explored = state.get("explored_topics", [])
    gaps = state.get("knowledge_gaps", [])
    depth_scores = state.get("depth_scores", {})

    print(f"\n🔄 Iterations Completed: {iterations}")
    print(f"📖 Total Sources Referenced: {len(sources)}")
    print(f"🎯 Topics Explored: {len(explored)}")
    print(f"⚠️  Knowledge Gaps Identified: {len(gaps)}")

    if depth_scores:
        print("\n📊 Section Depth Metrics:")
        for section, scores in depth_scores.items():
            completeness = scores.get('completeness', 0)
            status = "✅" if completeness >= 70 else "⚠️" if completeness >= 50 else "❌"
            print(f"  {status} {section[:50]}: {completeness:.1f}% complete")

    if sources:
        print(f"\n🔗 Top Sources:")
        for i, source in enumerate(sources[:5], 1):
            print(f"  {i}. {source.get('title', 'Untitled')}")
            print(f"     {source.get('url', 'N/A')}")

    print("\n" + "="*80)


def create_quick_reference(state: Dict[str, Any], output_dir: str = "output") -> str:
    """
    Create a quick reference markdown file with key insights.

    Args:
        state: The research state
        output_dir: Directory to save the file in

    Returns:
        Path to the saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    topic = state.get("research_topic", "Topic")
    curriculum = state.get("curriculum_outline", "")
    sources = state.get("sources", [])

    # Build quick reference
    content = []
    content.append(f"# {topic} - Quick Reference\n")
    content.append(
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

    content.append("## 📋 Curriculum Structure\n\n")
    if curriculum:
        # Extract just the headers
        for line in curriculum.split('\n'):
            if line.strip().startswith(('#', '-', '*')):
                content.append(f"{line}\n")

    content.append("\n## 🔗 Key Resources\n\n")
    for i, source in enumerate(sources[:10], 1):
        content.append(
            f"{i}. [{source.get('title', 'Source')}]({source.get('url', '#')})\n")

    content.append("\n## 💡 Study Tips\n\n")
    content.append("- Follow the curriculum structure sequentially\n")
    content.append("- Cross-reference multiple sources for each concept\n")
    content.append("- Practice with hands-on examples\n")
    content.append("- Review and revise regularly\n")

    markdown = "".join(content)

    # Save file
    safe_topic = "".join(c if c.isalnum() or c in (
        ' ', '-', '_') else '_' for c in topic)
    safe_topic = safe_topic.replace(' ', '_')[:50]
    filename = f"{safe_topic}_quick_reference.md"
    file_path = output_path / filename

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    print(f"⚡ Quick reference saved to: {file_path.absolute()}")
    return str(file_path.absolute())
