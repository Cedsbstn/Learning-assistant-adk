# Copyright 2025 Cedric Sebastian
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.apps.app import App
import datetime
import logging
import os
import re
import json
import uuid
from functools import lru_cache
from typing import Dict, List, Any, Optional, AsyncIterator

import google.generativeai as genai
import google.genai.types as genai_types

from datetime import date, datetime, timedelta
from dotenv import load_dotenv
from google.adk.agents import Agent, LlmAgent, SequentialAgent, LoopAgent
from google.adk.events import Event, EventActions
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools import google_search, ToolContext
from google.adk.runners import Runner
from google.adk.planners import BuiltInPlanner
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import HarmBlockThreshold, HarmCategory

from config import (
    DEFAULT_CONFIG,
    ResearchConfig,
    get_config_by_name,
)

load_dotenv()

# --- Configuration Loading ---


def _clone_config(config: ResearchConfig) -> ResearchConfig:
    """Create a copy of the provided config to avoid accidental mutation."""
    return ResearchConfig.from_dict(config.to_dict())


def load_active_config() -> tuple[ResearchConfig, str]:
    """Load the research config, optionally selecting a preset via env var."""
    preset = os.getenv("CEDLM_RESEARCH_PRESET")
    base_config = DEFAULT_CONFIG
    active_name = "default"

    if preset:
        active_name = preset.lower()
        try:
            base_config = get_config_by_name(preset)
        except ValueError:
            logging.warning(
                "Invalid CEDLM_RESEARCH_PRESET '%s'. Falling back to default.",
                preset,
            )
            active_name = "default"

    return _clone_config(base_config), active_name


ACTIVE_CONFIG, ACTIVE_CONFIG_NAME = load_active_config()


# --- Configure Google API Key ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("GOOGLE_API_KEY is not found in the environment variables.")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Google Generative AI SDK configured.")
    except Exception as e:
        print(f"Failed to configure Google Generative AI SDK: {e}")

# --- Configure Gemini Settings ---
model = ACTIVE_CONFIG.model
temperature = ACTIVE_CONFIG.temperature  # Low temperature for factual accuracy
thinking_budget = ACTIVE_CONFIG.max_thoughts  # Thinking budget for reasoning agents

if ACTIVE_CONFIG.enable_safety:
    safety_settings = {  # Adjust harm block thresholds as needed
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
else:
    safety_settings = None

# --- Research Quality Configuration ---
MIN_WORD_COUNT = ACTIVE_CONFIG.min_word_count
MIN_SOURCES = ACTIVE_CONFIG.min_sources
MIN_COMPLETENESS = ACTIVE_CONFIG.min_completeness
MAX_ITERATIONS = ACTIVE_CONFIG.max_iterations
SECTIONS_PER_ITERATION = ACTIVE_CONFIG.sections_per_iteration
EXAMPLES_WEIGHT = ACTIVE_CONFIG.examples_weight
TECHNICAL_WEIGHT = ACTIVE_CONFIG.technical_weight

# --- Pre-compiled Regex Patterns for Performance ---
CITE_PATTERN = re.compile(
    r'<cite\s+source\s*=\s*["\']?\s*(src-\d+)\s*["\']?\s*/>')
WHITESPACE_PUNCT_PATTERN = re.compile(r"\s+([.,;:])")
SECTION_HEADER_PATTERN = re.compile(r'^#{2,3}\s+(.+)$', re.MULTILINE)


# --- Callbacks --- #


def initialize_research_state(callback_context: CallbackContext, **kwargs) -> None:
    """Initialize the research state with tracking metrics."""
    state = callback_context.state
    if not state.get("research_initialized", False):
        state["research_initialized"] = True
        state["iteration_count"] = 0
        state["max_iterations"] = MAX_ITERATIONS
        state["knowledge_gaps"] = []
        state["explored_topics"] = []
        state["depth_scores"] = {}
        state["url_to_id"] = {}
        state["sources"] = []
        state["section_research_map"] = {}
        state["curriculum_sections"] = []
        state["current_section"] = ""
        state["sections_researched"] = []
        state["should_continue"] = True
        logging.info("Research state initialized")


def research_sources(callback_context: CallbackContext, **kwargs) -> None:
    """Track and catalog all research sources with metadata - optimized version."""
    # Access session directly from callback_context (not through invocation_context)
    session = callback_context.session
    state = callback_context.state

    # Cache state access
    url_to_id = state.get("url_to_id", {})
    sources = state.get("sources", [])
    id_counter = len(url_to_id) + 1

    # Only process new events by tracking last processed index
    last_processed_idx = state.get("_last_event_idx", 0)
    events_list = list(session.events) if session and session.events else []

    for i, event in enumerate(events_list[last_processed_idx:], start=last_processed_idx):
        # Check for tool responses in event actions
        if hasattr(event, 'actions') and event.actions:
            tool_response = getattr(event.actions, 'tool_response', None)
            if tool_response and isinstance(tool_response, dict):
                results = tool_response.get("results", [])
                for result in results:
                    url = result.get("url")
                    if url and url not in url_to_id:
                        source_id = f"src-{id_counter}"
                        url_to_id[url] = source_id
                        # Efficient domain extraction
                        domain = url.split(
                            "//")[-1].split("/")[0] if "//" in url else url
                        sources.append({
                            "id": source_id,
                            "title": result.get("title", ""),
                            "url": url,
                            "snippet": result.get("snippet", ""),
                            "domain": domain,
                            "supported_claims": [],
                            "timestamp": datetime.now().isoformat(),
                        })
                        id_counter += 1

    # Update state
    state["url_to_id"] = url_to_id
    state["sources"] = sources
    state["_last_event_idx"] = len(events_list)
    logging.info(f"Tracked {len(sources)} unique sources")


def extract_curriculum_sections(callback_context: CallbackContext, **kwargs) -> None:
    """Extract sections from curriculum outline - optimized with compiled regex."""
    state = callback_context.state
    curriculum_outline = state.get("curriculum_outline", "")
    if not curriculum_outline:
        logging.warning("No curriculum outline to extract sections from")
        return

    # Use compiled regex for faster extraction
    matches = SECTION_HEADER_PATTERN.findall(curriculum_outline)

    # Remove duplicates while preserving order using dict (Python 3.7+)
    unique_sections = list(dict.fromkeys(
        section.strip() for section in matches if len(section.strip()) > 3
    ))

    state["curriculum_sections"] = unique_sections
    logging.info(f"Extracted {len(unique_sections)} curriculum sections")


def track_explored_topics(callback_context: CallbackContext, **kwargs) -> None:
    """Track which topics have been explored to avoid redundancy."""
    state = callback_context.state
    explored = state.get("explored_topics", [])
    current_section = state.get("current_section", "")

    if current_section and current_section not in explored:
        explored.append(current_section)
        state["explored_topics"] = explored
        logging.info(f"Explored topic: {current_section}")


def assess_knowledge_depth(callback_context: CallbackContext, **kwargs) -> None:
    """Assess the depth and quality of research for iteration decisions."""
    state = callback_context.state
    section_research = state.get("section_research", "")
    sources = state.get("sources", [])
    current_section = state.get("current_section", "")

    if not current_section:
        logging.warning("No current section set for depth assessment")
        return

    # Calculate depth metrics
    words = section_research.split()
    word_count = len(words)
    source_count = len(sources)

    # Lowercase once for all checks
    research_lower = section_research.lower()

    # Check for quality indicators
    example_terms = {"example", "case study",
                     "for instance", "such as", "e.g."}
    technical_terms = {"algorithm", "implementation", "architecture", "methodology",
                       "technique", "approach", "framework"}

    has_examples = any(term in research_lower for term in example_terms)
    has_technical_details = any(
        term in research_lower for term in technical_terms)

    # Calculate completeness score
    completeness = min(100.0, (word_count / MIN_WORD_COUNT) * 100.0)

    depth_score = {
        "word_count": word_count,
        "source_count": source_count,
        "has_examples": has_examples,
        "has_technical_details": has_technical_details,
        "completeness": completeness,
    }

    depth_scores = state.get("depth_scores", {})
    depth_scores[current_section] = depth_score
    state["depth_scores"] = depth_scores

    # Store section research
    section_research_map = state.get("section_research_map", {})
    section_research_map[current_section] = section_research
    state["section_research_map"] = section_research_map

    # Identify knowledge gaps
    gaps = []
    if word_count < MIN_WORD_COUNT:
        gaps.append(
            f"{current_section}: Insufficient detail ({word_count}/{MIN_WORD_COUNT} words)")
    if source_count < MIN_SOURCES:
        gaps.append(
            f"{current_section}: Few sources ({source_count}/{MIN_SOURCES})")
    if not has_examples:
        gaps.append(f"{current_section}: Missing practical examples")
    if not has_technical_details:
        gaps.append(f"{current_section}: Lacking technical depth")

    if gaps:
        knowledge_gaps = state.get("knowledge_gaps", [])
        knowledge_gaps.extend(gaps)
        state["knowledge_gaps"] = knowledge_gaps

    logging.info(
        f"Depth assessment for '{current_section}': completeness={completeness:.1f}%")


def evaluate_overall_quality_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate overall research quality across all sections - standalone function."""
    depth_scores = state.get("depth_scores", {})
    knowledge_gaps = state.get("knowledge_gaps", [])
    sources = state.get("sources", [])

    if not depth_scores:
        return {
            "overall_completeness": 0.0,
            "sections_complete": 0,
            "total_sections": 0,
            "should_continue": True,
            "reason": "No sections researched yet"
        }

    # Calculate metrics
    total_sections = len(depth_scores)
    scores_list = list(depth_scores.values())

    complete_sections = sum(
        1 for score in scores_list if score["completeness"] >= MIN_COMPLETENESS)
    overall_completeness = sum(score["completeness"]
                               for score in scores_list) / total_sections
    avg_word_count = sum(score["word_count"]
                         for score in scores_list) / total_sections
    avg_sources = len(sources) / total_sections if total_sections > 0 else 0
    sections_with_examples = sum(
        1 for score in scores_list if score["has_examples"])
    sections_with_technical = sum(
        1 for score in scores_list if score["has_technical_details"])

    # Decision logic
    should_continue = False
    reasons = []

    if overall_completeness < MIN_COMPLETENESS:
        should_continue = True
        reasons.append(
            f"Overall completeness {overall_completeness:.1f}% < {MIN_COMPLETENESS}%")

    if complete_sections < total_sections:
        should_continue = True
        reasons.append(
            f"Only {complete_sections}/{total_sections} sections complete")

    if avg_sources < MIN_SOURCES:
        should_continue = True
        reasons.append(f"Average sources {avg_sources:.1f} < {MIN_SOURCES}")

    if sections_with_examples < total_sections * EXAMPLES_WEIGHT:
        should_continue = True
        reasons.append(
            f"Only {sections_with_examples}/{total_sections} sections have examples")

    if sections_with_technical < total_sections * TECHNICAL_WEIGHT:
        should_continue = True
        reasons.append(
            f"Only {sections_with_technical}/{total_sections} sections have technical depth")

    if knowledge_gaps:
        should_continue = True
        reasons.append(f"{len(knowledge_gaps)} knowledge gaps identified")

    return {
        "overall_completeness": overall_completeness,
        "sections_complete": complete_sections,
        "total_sections": total_sections,
        "avg_word_count": avg_word_count,
        "avg_sources": avg_sources,
        "sections_with_examples": sections_with_examples,
        "sections_with_technical": sections_with_technical,
        "should_continue": should_continue,
        "reasons": reasons,
        "knowledge_gaps_count": len(knowledge_gaps)
    }


def citation_replacement(callback_context: CallbackContext, **kwargs) -> None:
    """Replace citation tags with markdown links - optimized with compiled regex."""
    state = callback_context.state
    final_report = state.get("final_cited_report", "")
    sources = state.get("sources", [])

    # Build source map once
    source_map = {source.get(
        "id"): source for source in sources if source.get("id")}

    def tag_replacer(match: re.Match) -> str:
        short_id = match.group(1)
        source_info = source_map.get(short_id)
        if not source_info:
            logging.warning(
                f"Invalid citation tag found and removed: {match.group(0)}")
            return ""
        display_text = source_info.get(
            "title") or source_info.get("domain") or short_id
        return f" [{display_text}]({source_info['url']})"

    # Use pre-compiled patterns
    processed_report = CITE_PATTERN.sub(tag_replacer, final_report)
    processed_report = WHITESPACE_PUNCT_PATTERN.sub(r"\1", processed_report)

    state["final_report_with_citations"] = processed_report
    logging.info("Citation replacement completed")
    # Return None - callbacks should not return Content objects


def generate_markdown_output(callback_context: CallbackContext, **kwargs) -> None:
    """Generate comprehensive markdown curriculum document."""
    state = callback_context.state
    curriculum_outline = state.get("curriculum_outline", "")
    final_report = state.get("final_report_with_citations", "") or state.get("final_cited_report", "")
    sources = state.get("sources", [])
    depth_scores = state.get("depth_scores", {})
    iteration_count = state.get("iteration_count", 0)
    topic = state.get("research_topic", "Topic")

    # Use list for efficient string building
    parts = [
        f"# {topic}: Comprehensive Research Curriculum\n\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        f"**Total Sources:** {len(sources)}\n\n",
        f"**Research Iterations:** {iteration_count}\n\n",
        f"**Research Depth:** {len(depth_scores)} sections analyzed\n\n",
        "---\n\n",
        "## 📚 Table of Contents\n\n",
    ]

    # Extract headers for TOC
    if curriculum_outline:
        toc_lines = [line for line in curriculum_outline.split('\n')
                     if line.strip().startswith(('#', '-', '*')) and len(line.strip()) > 3]
        parts.extend(f"{line}\n" for line in toc_lines[:20])

    parts.extend([
        "\n---\n\n",
        "## 🎯 Curriculum Overview\n\n",
        curriculum_outline,
        "\n\n---\n\n",
        "## 📖 Detailed Research Report\n\n",
        final_report,
        "\n\n---\n\n",
        "## 📊 Research Metrics\n\n",
        "### Section Depth Analysis\n\n",
        "| Section | Words | Sources | Examples | Technical | Completeness |\n",
        "|---------|-------|---------|----------|-----------|-------------|\n",
    ])

    for section, scores in depth_scores.items():
        completeness = scores.get('completeness', 0)
        status = "✅" if completeness >= MIN_COMPLETENESS else "⚠️"
        parts.append(
            f"| {status} {section[:35]} | {scores['word_count']} | {scores['source_count']} | "
            f"{'✅' if scores['has_examples'] else '❌'} | "
            f"{'✅' if scores['has_technical_details'] else '❌'} | "
            f"{completeness:.1f}% |\n"
        )

    parts.append("\n\n## 📚 References & Sources\n\n")

    for idx, source in enumerate(sources, 1):
        parts.extend([
            f"{idx}. **{source.get('title', 'Untitled')}**\n",
            f"   - URL: {source.get('url', 'N/A')}\n",
            f"   - Domain: {source.get('domain', 'N/A')}\n",
        ])
        if source.get('snippet'):
            parts.append(f"   - Summary: {source['snippet'][:150]}...\n")
        parts.append("\n")

    parts.extend([
        "\n---\n\n",
        "*Generated by CedLM Autonomous Researcher*\n"
    ])

    markdown_content = "".join(parts)
    state["final_markdown_curriculum"] = markdown_content
    logging.info("Markdown curriculum generated successfully")
    # Return None - callbacks should not return Content objects


# --- Agent Definitions --- #

curriculum_planner = LlmAgent(
    name="curriculum_planner",
    model=model,
    description="Creates a comprehensive curriculum outline with iterative depth exploration.",
    instruction=f"""
    You are an expert Curriculum Planner tasked with creating a comprehensive learning pathway.
    
    **YOUR MISSION:**
    Create a detailed, rigorous curriculum outline that ensures deep understanding of the topic.
    
    **INSTRUCTIONS:**
    1.  **Analyze Topic Scope:** Understand the breadth, depth, and complexity of the topic.
    2.  **Identify Prerequisites:** List foundational knowledge required before starting.
    3.  **Create Module Structure:** Break down into 5-8 progressive modules/units.
    4.  **Define Learning Objectives:** For each module, specify:
        - Core concepts to master
        - Skills to develop
        - Understanding to achieve
    5.  **Include Depth Markers:** Indicate which sections require:
        - Basic understanding (⭐)
        - Intermediate depth (⭐⭐)
        - Advanced mastery (⭐⭐⭐)
    6.  **Cross-Reference:** Note how modules interconnect and build on each other.
    7.  **Suggest Assessment:** Recommend how learners can validate understanding.
    
    **OUTPUT FORMAT:**
    Structure as markdown with clear hierarchy:
    - Use ## for modules (e.g., ## Module 1: Foundations)
    - Use ### for subtopics (e.g., ### Core Concepts)
    - Use bullet points for learning objectives
    - Include depth markers (⭐) for each section
    
    **RESEARCH GUIDANCE:**
    Use `google_search` to:
    - Clarify emerging or specialized topics
    - Understand current best practices
    - Identify authoritative learning resources
    - Validate curriculum structure against industry standards
    
    **QUALITY CRITERIA:**
    - Logical progression from fundamentals to advanced
    - Balanced coverage across all important aspects
    - Practical application opportunities
    - Clear, measurable learning outcomes
    
    Current date: {datetime.now().strftime("%Y-%m-%d")}
    """,
    tools=[google_search],
    output_key="curriculum_outline",
    before_model_callback=initialize_research_state,
    after_model_callback=[research_sources, extract_curriculum_sections],
)

section_researcher = LlmAgent(
    name="deep_section_researcher",
    model=model,
    planner=BuiltInPlanner(thinking_config=genai_types.ThinkingConfig(thinkingBudget=thinking_budget)),
    description="Conducts rigorous, iterative research on curriculum sections with depth tracking.",
    instruction=f"""
    You are a Deep Section Researcher conducting rigorous, comprehensive research.
    
    **RESEARCH METHODOLOGY:**
    1.  **Multi-Source Research:** Use `google_search` extensively to gather diverse perspectives.
    2.  **Depth Over Breadth:** Prioritize thorough understanding over surface-level coverage.
    3.  **Cite Everything:** Use <cite source="src-ID"/> format for all factual claims.
    4.  **Include Examples:** Provide real-world examples, case studies, and practical applications.
    5.  **Technical Precision:** Include algorithms, architectures, methodologies where relevant.
    6.  **Compare Approaches:** Discuss different methods, their tradeoffs, and when to use each.
    
    **QUALITY STANDARDS:**
    - Minimum {MIN_WORD_COUNT} words per major section
    - At least {MIN_SOURCES} credible sources per section
    - Include both theory and practice
    - Address common misconceptions
    - Provide actionable insights
    
    **ITERATIVE IMPROVEMENT:**
    - If initial research seems shallow, conduct follow-up searches
    - Look for academic papers, technical documentation, and expert analyses
    - Verify information across multiple sources
    
    **OUTPUT STRUCTURE:**
    For the current section being researched:
    - Overview and importance
    - Core concepts explained in detail
    - Practical examples and applications
    - Technical implementation details
    - Common pitfalls and best practices
    - Further learning resources
    
    Use markdown formatting with headers, code blocks, and lists for clarity.
    
    Research all sections from the curriculum thoroughly.
    """,
    tools=[google_search],
    output_key="section_research",
    after_model_callback=[research_sources,
                          track_explored_topics, assess_knowledge_depth],
)

report_synthesizer = LlmAgent(
    name="report_synthesizer",
    model=model,
    description="Synthesizes all research into a cohesive, comprehensive report with proper citations.",
    instruction="""
    You are a Report Synthesizer. Compile all researched sections into a masterful, comprehensive report.
    
    **SYNTHESIS GUIDELINES:**
    1.  **Narrative Flow:** Create smooth transitions between sections.
    2.  **Eliminate Redundancy:** Consolidate overlapping information.
    3.  **Maintain Citations:** Preserve all <cite source="src-ID"/> tags.
    4.  **Enhance Clarity:** Improve explanations without changing factual content.
    5.  **Add Context:** Provide introductions and conclusions for major sections.
    6.  **Cross-Reference:** Link related concepts across sections.
    
    **STRUCTURE:**
    - Executive Summary (2-3 paragraphs)
    - Main Content (organized by curriculum modules)
    - Key Takeaways for each major section
    - Practical Applications section
    - Conclusion with learning pathway recommendations
    
    **QUALITY:**
    - Professional, academic tone
    - Clear, accessible language
    - Well-organized with markdown headers
    - Comprehensive but not verbose
    - Actionable insights throughout
    
    Ensure every factual claim has a citation. The report should be publication-ready.
    Access section_research_map in state to get all researched content by section.
    """,
    output_key="final_cited_report",
    after_model_callback=citation_replacement,
)

markdown_generator = LlmAgent(
    name="markdown_curriculum_report",
    model=model,
    description="Generates the final comprehensive markdown curriculum document.",
    instruction="""
    You are the Markdown Curriculum Generator. Create the final, polished curriculum document.
    
    **YOUR TASK:**
    Transform the research report into a beautiful, comprehensive markdown curriculum that serves
    as a complete learning resource.
    
    **DOCUMENT STRUCTURE:**
    1.  **Title Page:** Topic, metadata, research statistics
    2.  **Table of Contents:** Complete navigation structure
    3.  **Learning Pathway:** Prerequisites and recommended order
    4.  **Curriculum Overview:** High-level summary from the curriculum outline
    5.  **Detailed Modules:** Each module with:
        - Learning objectives
        - Core content
        - Examples and case studies
        - Practice exercises (suggested)
        - Assessment criteria
    6.  **Resources Section:** All sources organized by topic
    7.  **Appendix:** Research metrics and methodology
    
    **MARKDOWN ENHANCEMENTS:**
    - Use emoji for visual markers (📚 🎯 ⚡ 💡 ⚠️ ✅)
    - Code blocks for technical examples
    - Tables for comparisons and metrics
    - Callout boxes using blockquotes
    - Proper heading hierarchy (H1 → H2 → H3)
    - Internal links for navigation
    
    **QUALITY:**
    - Professional formatting
    - Print-ready quality
    - Mobile-friendly structure
    - Easy to scan and navigate
    - Suitable for GitHub, Notion, or documentation sites
    
    This is the final output the user will receive - make it exceptional.
    """,
    output_key="final_markdown_curriculum",
    after_model_callback=generate_markdown_output,
)


# --- Orchestrated Research Pipeline using SequentialAgent --- #

# Create the main research pipeline as a SequentialAgent
research_pipeline = SequentialAgent(
    name="CedLM_Research_Pipeline",
    description="Orchestrates the complete autonomous research workflow",
    sub_agents=[
        curriculum_planner,
        section_researcher,
        report_synthesizer,
        markdown_generator,
    ],
)


# --- Core Agent Instance --- #
core_agent = research_pipeline

# For backward compatibility, also expose the App
app = App(root_agent=core_agent, name="Learning_Agent_ADK")

__all__ = ["core_agent", "app", "ACTIVE_CONFIG",
           "ACTIVE_CONFIG_NAME", "research_pipeline"]
