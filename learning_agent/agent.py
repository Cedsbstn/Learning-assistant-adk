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

import datetime
import logging
import os
import re
import json
import google.generativeai as genai
import google.genai.types as genai_types

from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from google.adk.apps.app import App
from google.adk.agents import Agent as LLmAgent, Agent
from google.adk.events import Event, EventActions
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools import google_search, ToolContext
from google.adk.runners import Runner
from google.adk.planners import BuiltInPlanner as BuiltinReasoner
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
max_thoughts = ACTIVE_CONFIG.max_thoughts  # Max thoughts for reasoning agents

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

# --- Callbacks --- #


def initialize_research_state(callback_context: CallbackContext) -> None:
    """Initialize the research state with tracking metrics."""
    if not callback_context.state.get("research_initialized", default_value=False):
        callback_context.state["research_initialized"] = True
        callback_context.state["iteration_count"] = 0
        callback_context.state["max_iterations"] = MAX_ITERATIONS
        callback_context.state["knowledge_gaps"] = []
        callback_context.state["explored_topics"] = []
        callback_context.state["depth_scores"] = {}
        callback_context.state["url_to_id"] = {}
        callback_context.state["sources"] = []
        # Map sections to research content
        callback_context.state["section_research_map"] = {}
        # List of curriculum sections
        callback_context.state["curriculum_sections"] = []
        callback_context.state["current_section"] = ""
        callback_context.state["sections_researched"] = []
        callback_context.state["should_continue"] = True
        logging.info("Research state initialized")


def research_sources(callback_context: CallbackContext) -> None:
    """Track and catalog all research sources with metadata."""
    session = callback_context.invocation_context.session
    url_to_id = callback_context.state.get("url_to_id", default_value={})
    sources = callback_context.state.get("sources", default_value=[])
    id_counter = len(url_to_id) + 1

    for event in session.events:
        if event.action == EventActions.TOOL_INVOCATION and event.tool_name == "google_search":
            tool_output = event.tool_output
            if tool_output and "results" in tool_output:
                for result in tool_output["results"]:
                    url = result.get("url")
                    title = result.get("title")
                    snippet = result.get("snippet")
                    if url and url not in url_to_id:
                        source_id = f"src-{id_counter}"
                        url_to_id[url] = source_id
                        domain = url.split(
                            "//")[-1].split("/")[0] if "//" in url else url
                        sources.append({
                            "id": source_id,
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "domain": domain,
                            "supported_claims": [],
                            "timestamp": datetime.now().isoformat(),
                        })
                        id_counter += 1

    callback_context.state["url_to_id"] = url_to_id
    callback_context.state["sources"] = sources
    logging.info(f"Tracked {len(sources)} unique sources")


def extract_curriculum_sections(callback_context: CallbackContext) -> None:
    """Extract sections from curriculum outline for targeted research."""
    curriculum_outline = callback_context.state.get("curriculum_outline", "")
    if not curriculum_outline:
        logging.warning("No curriculum outline to extract sections from")
        return

    sections = []
    lines = curriculum_outline.split('\n')

    for line in lines:
        stripped = line.strip()
        # Extract section headers (## or ###)
        if stripped.startswith('##') and not stripped.startswith('###'):
            # Main module header
            section_name = stripped.lstrip('#').strip()
            # Filter out empty or too short
            if section_name and len(section_name) > 3:
                sections.append(section_name)
        elif stripped.startswith('###'):
            # Sub-section header
            section_name = stripped.lstrip('#').strip()
            if section_name and len(section_name) > 3:
                sections.append(section_name)

    # Remove duplicates while preserving order
    seen = set()
    unique_sections = []
    for section in sections:
        if section not in seen:
            seen.add(section)
            unique_sections.append(section)

    callback_context.state["curriculum_sections"] = unique_sections
    logging.info(
        f"Extracted {len(unique_sections)} curriculum sections: {unique_sections}")


def track_explored_topics(callback_context: CallbackContext) -> None:
    """Track which topics have been explored to avoid redundancy."""
    explored = callback_context.state.get("explored_topics", default_value=[])
    current_section = callback_context.state.get(
        "current_section", default_value="")

    if current_section and current_section not in explored:
        explored.append(current_section)
        callback_context.state["explored_topics"] = explored
        logging.info(f"Explored topic: {current_section}")


def assess_knowledge_depth(callback_context: CallbackContext) -> None:
    """Assess the depth and quality of research for iteration decisions."""
    section_research = callback_context.state.get(
        "section_research", default_value="")
    sources = callback_context.state.get("sources", default_value=[])
    current_section = callback_context.state.get(
        "current_section", default_value="")

    if not current_section:
        logging.warning("No current section set for depth assessment")
        return

    # Calculate depth metrics
    word_count = len(section_research.split())
    source_count = len(sources)

    # Check for quality indicators
    has_examples = any(term in section_research.lower() for term in
                       ["example", "case study", "for instance", "such as", "e.g."])
    has_technical_details = any(term in section_research.lower() for term in
                                ["algorithm", "implementation", "architecture", "methodology",
                                 "technique", "approach", "framework"])

    # Calculate completeness score
    completeness = min(100.0, (word_count / MIN_WORD_COUNT) * 100.0)

    depth_score = {
        "word_count": word_count,
        "source_count": source_count,
        "has_examples": has_examples,
        "has_technical_details": has_technical_details,
        "completeness": completeness,
    }

    depth_scores = callback_context.state.get("depth_scores", default_value={})
    depth_scores[current_section] = depth_score
    callback_context.state["depth_scores"] = depth_scores

    # Store section research
    section_research_map = callback_context.state.get(
        "section_research_map", default_value={})
    section_research_map[current_section] = section_research
    callback_context.state["section_research_map"] = section_research_map

    # Identify knowledge gaps
    gaps = []
    if depth_score["word_count"] < MIN_WORD_COUNT:
        gaps.append(
            f"{current_section}: Insufficient detail ({word_count}/{MIN_WORD_COUNT} words)")
    if depth_score["source_count"] < MIN_SOURCES:
        gaps.append(
            f"{current_section}: Few sources ({source_count}/{MIN_SOURCES})")
    if not depth_score["has_examples"]:
        gaps.append(f"{current_section}: Missing practical examples")
    if not depth_score["has_technical_details"]:
        gaps.append(f"{current_section}: Lacking technical depth")

    if gaps:
        knowledge_gaps = callback_context.state.get(
            "knowledge_gaps", default_value=[])
        knowledge_gaps.extend(gaps)
        callback_context.state["knowledge_gaps"] = knowledge_gaps

    logging.info(f"Depth assessment for '{current_section}': {depth_score}")
    logging.info(f"  Completeness: {completeness:.1f}%")
    logging.info(f"  Identified {len(gaps)} gaps for this section")


def evaluate_overall_quality(callback_context: CallbackContext) -> Dict[str, Any]:
    """Evaluate overall research quality across all sections."""
    depth_scores = callback_context.state.get("depth_scores", default_value={})
    knowledge_gaps = callback_context.state.get(
        "knowledge_gaps", default_value=[])
    sources = callback_context.state.get("sources", default_value=[])

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
    complete_sections = sum(1 for score in depth_scores.values()
                            if score["completeness"] >= MIN_COMPLETENESS)

    overall_completeness = sum(score["completeness"]
                               for score in depth_scores.values()) / total_sections

    avg_word_count = sum(score["word_count"]
                         for score in depth_scores.values()) / total_sections
    avg_sources = len(sources) / total_sections if total_sections > 0 else 0

    sections_with_examples = sum(
        1 for score in depth_scores.values() if score["has_examples"])
    sections_with_technical = sum(
        1 for score in depth_scores.values() if score["has_technical_details"])

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

    if len(knowledge_gaps) > 0:
        should_continue = True
        reasons.append(f"{len(knowledge_gaps)} knowledge gaps identified")

    evaluation = {
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

    logging.info(f"Overall quality evaluation: {evaluation}")
    return evaluation


def citation_replacement(callback_context: CallbackContext) -> genai_types.Content:
    """Replace citation tags with markdown links."""
    final_report = callback_context.state.get("final_cited_report", "")
    sources = callback_context.state.get("sources", [])
    source_map = {}

    for source in sources:
        s_id = source.get("id")
        if s_id:
            source_map[s_id] = source

    def tag_replacer(match: re.Match) -> str:
        short_id = match.group(1)
        if not (source_info := source_map.get(short_id)):
            logging.warning(
                f"Invalid citation tag found and removed: {match.group(0)}")
            return ""
        display_text = source_info.get(
            "title", source_info.get("domain", short_id))
        return f" [{display_text}]({source_info['url']})"

    processed_report = re.sub(
        r'<cite\s+source\s*=\s*["\']?\s*(src-\d+)\s*["\']?\s*/>',
        tag_replacer,
        final_report,
    )
    processed_report = re.sub(r"\s+([.,;:])", r"\1", processed_report)
    callback_context.state["final_report_with_citations"] = processed_report
    return genai_types.Content(parts=[genai_types.Part(text=processed_report)])


def generate_markdown_output(callback_context: CallbackContext) -> genai_types.Content:
    """Generate comprehensive markdown curriculum document."""
    curriculum_outline = callback_context.state.get("curriculum_outline", "")
    final_report = callback_context.state.get(
        "final_report_with_citations", "")
    sources = callback_context.state.get("sources", [])
    depth_scores = callback_context.state.get("depth_scores", default_value={})
    iteration_count = callback_context.state.get("iteration_count", 0)

    # Build markdown document
    markdown_parts = []

    # Title and metadata
    topic = callback_context.state.get("research_topic", "Topic")
    markdown_parts.append(f"# {topic}: Comprehensive Research Curriculum\n\n")
    markdown_parts.append(
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    markdown_parts.append(f"**Total Sources:** {len(sources)}\n\n")
    markdown_parts.append(f"**Research Iterations:** {iteration_count}\n\n")
    markdown_parts.append(
        f"**Research Depth:** {len(depth_scores)} sections analyzed\n\n")
    markdown_parts.append("---\n\n")

    # Table of Contents
    markdown_parts.append("## 📚 Table of Contents\n\n")
    if curriculum_outline:
        # Extract headers from curriculum outline for TOC
        toc_lines = [line for line in curriculum_outline.split('\n')
                     if line.strip().startswith(('#', '-', '*')) and len(line.strip()) > 3]
        for line in toc_lines[:20]:  # Limit TOC entries
            markdown_parts.append(f"{line}\n")
    markdown_parts.append("\n---\n\n")

    # Curriculum Overview
    markdown_parts.append("## 🎯 Curriculum Overview\n\n")
    markdown_parts.append(curriculum_outline)
    markdown_parts.append("\n\n---\n\n")

    # Main Content
    markdown_parts.append("## 📖 Detailed Research Report\n\n")
    markdown_parts.append(final_report)
    markdown_parts.append("\n\n---\n\n")

    # Research Metrics
    markdown_parts.append("## 📊 Research Metrics\n\n")
    markdown_parts.append("### Section Depth Analysis\n\n")
    markdown_parts.append(
        "| Section | Words | Sources | Examples | Technical | Completeness |\n")
    markdown_parts.append(
        "|---------|-------|---------|----------|-----------|-------------|\n")

    for section, scores in depth_scores.items():
        completeness = scores.get('completeness', 0)
        status = "✅" if completeness >= MIN_COMPLETENESS else "⚠️"
        markdown_parts.append(
            f"| {status} {section[:35]} | {scores['word_count']} | {scores['source_count']} | "
            f"{'✅' if scores['has_examples'] else '❌'} | "
            f"{'✅' if scores['has_technical_details'] else '❌'} | "
            f"{completeness:.1f}% |\n"
        )

    markdown_parts.append("\n\n")

    # Sources Bibliography
    markdown_parts.append("## 📚 References & Sources\n\n")
    for idx, source in enumerate(sources, 1):
        markdown_parts.append(
            f"{idx}. **{source.get('title', 'Untitled')}**\n")
        markdown_parts.append(f"   - URL: {source.get('url', 'N/A')}\n")
        markdown_parts.append(f"   - Domain: {source.get('domain', 'N/A')}\n")
        if source.get('snippet'):
            markdown_parts.append(
                f"   - Summary: {source['snippet'][:150]}...\n")
        markdown_parts.append("\n")

    markdown_parts.append("\n---\n\n")
    markdown_parts.append("*Generated by CedLM Autonomous Researcher*\n")

    markdown_content = "".join(markdown_parts)
    callback_context.state["final_markdown_curriculum"] = markdown_content

    return genai_types.Content(parts=[genai_types.Part(text=markdown_content)])


# --- Agent Definitions --- #

curriculum_planner = LLmAgent(
    name="curriculum_planner",
    model=model,
    description="Creates a comprehensive curriculum outline with iterative depth exploration.",
    instruction="""
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
    
    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    tools=[google_search],
    output_key="curriculum_outline",
    before_model_callback=[initialize_research_state],
    after_model_callback=[research_sources, extract_curriculum_sections],
)

knowledge_gap_analyzer = LLmAgent(
    name="knowledge_gap_analyzer",
    model=model,
    planner=BuiltinReasoner(thinking_config={"max_thoughts": max_thoughts}),
    description="Identifies gaps in the curriculum and prioritizes areas needing deeper research.",
    instruction="""
    You are a Knowledge Gap Analyzer. Your role is to critically evaluate the curriculum outline
    and identify areas that require deeper exploration.
    
    **ANALYSIS FRAMEWORK:**
    1.  **Completeness Check:** Are all major aspects of the topic covered?
    2.  **Depth Assessment:** Which sections need more detailed exploration?
    3.  **Practical Application:** Are there sufficient real-world examples and use cases?
    4.  **Technical Rigor:** Are technical concepts explained with appropriate depth?
    5.  **Current Relevance:** Does the curriculum reflect latest developments?
    
    **OUTPUT:**
    Provide a prioritized list of:
    - Topics requiring additional research (with reasoning)
    - Specific questions that need answers
    - Areas lacking practical examples
    - Technical details that need clarification
    
    **FORMAT:**
    Return as a structured list with priority levels (High/Medium/Low) for each gap.
    Group gaps by curriculum section for targeted research.
    """,
    output_key="knowledge_gaps_analysis",
)

section_researcher = LLmAgent(
    name="deep_section_researcher",
    model=model,
    planner=BuiltinReasoner(thinking_config={"max_thoughts": max_thoughts}),
    description="Conducts rigorous, iterative research on curriculum sections with depth tracking.",
    instruction="""
    You are a Deep Section Researcher conducting rigorous, comprehensive research.
    
    **RESEARCH METHODOLOGY:**
    1.  **Multi-Source Research:** Use `google_search` extensively to gather diverse perspectives.
    2.  **Depth Over Breadth:** Prioritize thorough understanding over surface-level coverage.
    3.  **Cite Everything:** Use <cite source="src-ID"/> format for all factual claims.
    4.  **Include Examples:** Provide real-world examples, case studies, and practical applications.
    5.  **Technical Precision:** Include algorithms, architectures, methodologies where relevant.
    6.  **Compare Approaches:** Discuss different methods, their tradeoffs, and when to use each.
    
    **QUALITY STANDARDS:**
    - Minimum {min_words} words per major section
    - At least {min_sources} credible sources per section
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
    
    **CURRENT SECTION FOCUS:** Pay special attention to the current_section in state.
    """.format(min_words=MIN_WORD_COUNT, min_sources=MIN_SOURCES),
    tools=[google_search],
    output_key="section_research",
    after_model_callback=[research_sources,
                          track_explored_topics, assess_knowledge_depth],
)

iterative_refinement_agent = LLmAgent(
    name="iterative_refinement_agent",
    model=model,
    planner=BuiltinReasoner(thinking_config={"max_thoughts": max_thoughts}),
    description="Evaluates research quality and determines if additional iteration is needed.",
    instruction="""
    You are an Iterative Refinement Agent. Evaluate the research quality and decide if more work is needed.
    
    **EVALUATION CRITERIA:**
    1.  **Depth Score:** Is each section substantive ({min_words}+ words with multiple sources)?
    2.  **Coverage:** Are all curriculum sections researched thoroughly?
    3.  **Quality:** Are explanations clear, accurate, and well-cited?
    4.  **Practical Value:** Are there sufficient examples and applications?
    5.  **Coherence:** Does the research flow logically?
    
    **DECISION LOGIC:**
    Analyze the depth_scores and knowledge_gaps in state:
    - If any section completeness < {min_completeness}%: Recommend CONTINUE with specific improvements
    - If knowledge gaps remain unfilled: Recommend CONTINUE with gap-filling queries
    - If fewer than {min_sources} sources per section: Recommend CONTINUE with more research
    - If iteration count < max_iterations and quality can improve: Recommend CONTINUE
    - Otherwise: Recommend PROCEED to final compilation
    
    **OUTPUT:**
    Provide a structured evaluation:
    1. **Overall Quality Score:** Percentage (0-100%)
    2. **Sections Analysis:** List each section with its completeness
    3. **Specific Improvements Needed:** Detailed list of what's missing
    4. **Decision:** Either "CONTINUE" or "PROCEED"
    5. **Next Actions:** If CONTINUE, provide specific research queries for each weak section
    
    Be thorough and specific in your recommendations.
    """.format(min_words=MIN_WORD_COUNT, min_sources=MIN_SOURCES, min_completeness=MIN_COMPLETENESS),
    output_key="refinement_decision",
)

report_synthesizer = LLmAgent(
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
    after_model_callback=[citation_replacement],
)

markdown_generator = LLmAgent(
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
    after_model_callback=[generate_markdown_output],
)


# --- Custom Iterative Research Agent --- #

class IterativeResearchAgent(Agent):
    """
    Custom agent that manages autonomous iterative research with dynamic looping.

    This agent orchestrates the research process by:
    1. Creating curriculum structure
    2. Identifying knowledge gaps
    3. Researching each section iteratively
    4. Evaluating quality after each iteration
    5. Continuing until quality standards are met or max iterations reached
    6. Synthesizing and generating final output
    """

    curriculum_planner: Optional[Agent] = None
    gap_analyzer: Optional[Agent] = None
    section_researcher: Optional[Agent] = None
    refinement_agent: Optional[Agent] = None
    synthesizer: Optional[Agent] = None
    markdown_gen: Optional[Agent] = None

    def __init__(self, name: str = "CedLM Autonomous Researcher"):
        super().__init__(name=name)
        self.curriculum_planner = curriculum_planner
        self.gap_analyzer = knowledge_gap_analyzer
        self.section_researcher = section_researcher
        self.refinement_agent = iterative_refinement_agent
        self.synthesizer = report_synthesizer
        self.markdown_gen = markdown_generator
        self.synthesizer = report_synthesizer
        self.markdown_gen = markdown_generator

    async def run(self, context: InvocationContext) -> genai_types.Content:
        """
        Execute the autonomous iterative research process.

        Args:
            context: The invocation context with session and state

        Returns:
            The final markdown curriculum content
        """
        session = context.session
        state = session.state
        user_query = context.input

        logging.info(f"=== Starting Autonomous Research for: {user_query} ===")
        print(f"\n🔬 Autonomous Research Agent Initialized")
        print(f"📋 Topic: {user_query}\n")

        # Store the query
        state["research_topic"] = user_query

        # --- Phase 1: Curriculum Planning ---
        print("📚 Phase 1: Creating Curriculum Structure...")
        curriculum_response = await self.curriculum_planner.run(context)
        curriculum_outline = state.get("curriculum_outline", "")
        sections = state.get("curriculum_sections", [])
        print(f"✅ Curriculum created with {len(sections)} sections\n")
        logging.info(f"Curriculum sections: {sections}")

        # --- Phase 2: Gap Analysis ---
        print("🔍 Phase 2: Analyzing Knowledge Gaps...")
        gap_response = await self.gap_analyzer.run(context)
        gaps_analysis = state.get("knowledge_gaps_analysis", "")
        print(f"✅ Knowledge gaps identified\n")

        # --- Phase 3: Iterative Research ---
        print("🔄 Phase 3: Iterative Deep Research")
        print(
            f"Target: {MIN_WORD_COUNT}+ words, {MIN_SOURCES}+ sources, {MIN_COMPLETENESS}%+ completeness per section")
        print(f"Max iterations: {MAX_ITERATIONS}\n")

        iteration = 0
        should_continue = True

        while should_continue and iteration < MAX_ITERATIONS:
            iteration += 1
            state["iteration_count"] = iteration

            print(f"🔄 Iteration {iteration}/{MAX_ITERATIONS}")
            logging.info(f"=== Starting Iteration {iteration} ===")

            # Research each section that needs improvement
            sections_to_research = self._identify_sections_to_research(
                state, sections)

            if not sections_to_research:
                print("   All sections meet quality standards!")
                should_continue = False
                break

            print(f"   Researching {len(sections_to_research)} sections...")

            for idx, section in enumerate(sections_to_research, 1):
                print(
                    f"   [{idx}/{len(sections_to_research)}] Researching: {section[:50]}...")
                state["current_section"] = section

                # Create focused research prompt
                research_prompt = self._create_research_prompt(
                    section, state, user_query)
                context.input = research_prompt

                # Conduct research
                research_response = await self.section_researcher.run(context)

                # Log progress
                depth_score = state.get("depth_scores", {}).get(section, {})
                completeness = depth_score.get("completeness", 0)
                word_count = depth_score.get("word_count", 0)

                status = "✅" if completeness >= MIN_COMPLETENESS else "⚠️"
                print(
                    f"      {status} {completeness:.1f}% complete ({word_count} words)")

            # --- Evaluate Quality ---
            print(f"\n   📊 Evaluating research quality...")
            evaluation = evaluate_overall_quality(CallbackContext(
                invocation_context=context,
                state=state
            ))

            overall_completeness = evaluation["overall_completeness"]
            complete_sections = evaluation["sections_complete"]
            total_sections = evaluation["total_sections"]

            print(f"   Overall Completeness: {overall_completeness:.1f}%")
            print(
                f"   Complete Sections: {complete_sections}/{total_sections}")

            # --- Refinement Decision ---
            print(f"\n   🤔 Assessing if more research is needed...")
            context.input = self._create_refinement_prompt(state, evaluation)
            refinement_response = await self.refinement_agent.run(context)
            refinement_decision = state.get("refinement_decision", "")

            # Parse decision
            should_continue = self._parse_refinement_decision(
                refinement_decision, evaluation, iteration)

            if should_continue:
                print(
                    f"   ➡️  Continuing research (Quality: {overall_completeness:.1f}%)")
                if evaluation.get("reasons"):
                    print(
                        f"   Reasons: {', '.join(evaluation['reasons'][:2])}")
            else:
                print(f"   ✅ Research quality standards met! Proceeding to synthesis.")

            print()  # Blank line for readability

        # --- Phase 4: Synthesis ---
        print("✨ Phase 4: Synthesizing Research Report...")
        context.input = user_query  # Reset to original query
        synthesis_response = await self.synthesizer.run(context)
        print("✅ Report synthesized with citations\n")

        # --- Phase 5: Markdown Generation ---
        print("📝 Phase 5: Generating Markdown Curriculum...")
        markdown_response = await self.markdown_gen.run(context)
        print("✅ Final curriculum generated\n")

        # Final summary
        final_sources = len(state.get("sources", []))
        final_iterations = state.get("iteration_count", 0)
        final_completeness = evaluate_overall_quality(CallbackContext(
            invocation_context=context,
            state=state
        ))["overall_completeness"]

        print("="*80)
        print("🎉 AUTONOMOUS RESEARCH COMPLETE!")
        print("="*80)
        print(f"✅ Iterations: {final_iterations}")
        print(f"✅ Sources: {final_sources}")
        print(f"✅ Overall Quality: {final_completeness:.1f}%")
        print(f"✅ Sections: {len(state.get('depth_scores', {}))}")
        print("="*80 + "\n")

        logging.info(
            f"=== Research Complete: {final_iterations} iterations, {final_sources} sources ===")

        return markdown_response

    def _identify_sections_to_research(self, state: Dict, sections: List[str]) -> List[str]:
        """Identify which sections need (more) research."""
        depth_scores = state.get("depth_scores", {})
        sections_researched = state.get("sections_researched", [])

        sections_to_research = []

        # First, research sections that haven't been researched yet
        for section in sections:
            if section not in sections_researched:
                sections_to_research.append(section)

        # Then, research sections that don't meet quality standards
        for section, score in depth_scores.items():
            if score["completeness"] < MIN_COMPLETENESS and section not in sections_to_research:
                sections_to_research.append(section)

        # Mark sections as researched
        for section in sections_to_research:
            if section not in sections_researched:
                sections_researched.append(section)

        state["sections_researched"] = sections_researched

        return sections_to_research[:SECTIONS_PER_ITERATION]

    def _create_research_prompt(self, section: str, state: Dict, original_query: str) -> str:
        """Create a focused research prompt for a specific section."""
        knowledge_gaps = state.get("knowledge_gaps", [])
        section_gaps = [gap for gap in knowledge_gaps if section in gap]

        prompt = f"""Research the following section of the curriculum in depth:

**Topic:** {original_query}
**Section:** {section}

**Research Requirements:**
- Minimum {MIN_WORD_COUNT} words
- At least {MIN_SOURCES} credible sources
- Include practical examples and case studies
- Include technical details and implementations
- Cite all sources using <cite source="src-ID"/> format

"""

        if section_gaps:
            prompt += f"""**Address These Knowledge Gaps:**
{chr(10).join(f'- {gap}' for gap in section_gaps[:3])}

"""

        prompt += """Conduct comprehensive research and provide detailed, well-cited content for this section."""

        return prompt

    def _create_refinement_prompt(self, state: Dict, evaluation: Dict) -> str:
        """Create a prompt for the refinement agent to evaluate quality."""
        depth_scores = state.get("depth_scores", {})
        knowledge_gaps = state.get("knowledge_gaps", [])
        iteration = state.get("iteration_count", 0)

        prompt = f"""Evaluate the current research quality and decide if more iteration is needed.

**Current Iteration:** {iteration}/{MAX_ITERATIONS}

**Overall Metrics:**
- Overall Completeness: {evaluation['overall_completeness']:.1f}%
- Complete Sections: {evaluation['sections_complete']}/{evaluation['total_sections']}
- Average Word Count: {evaluation.get('avg_word_count', 0):.0f}
- Average Sources: {evaluation.get('avg_sources', 0):.1f}
- Knowledge Gaps: {evaluation.get('knowledge_gaps_count', 0)}

**Section Details:**
"""

        for section, scores in depth_scores.items():
            status = "✅" if scores["completeness"] >= MIN_COMPLETENESS else "❌"
            prompt += f"\n{status} {section}: {scores['completeness']:.1f}% ({scores['word_count']} words, {scores['source_count']} sources)"

        prompt += f"""

**Decision Required:**
Analyze the metrics and provide:
1. Overall quality assessment (0-100%)
2. Specific sections that need improvement
3. Decision: "CONTINUE" or "PROCEED"
4. If CONTINUE: specific improvements needed

Be rigorous in your evaluation.
"""

        return prompt

    def _parse_refinement_decision(self, decision_text: str, evaluation: Dict, iteration: int) -> bool:
        """Parse the refinement decision to determine if research should continue."""
        # Check for explicit decision
        decision_upper = decision_text.upper()

        # If explicitly says PROCEED, stop
        if "PROCEED" in decision_upper and "CONTINUE" not in decision_upper:
            return False

        # If explicitly says CONTINUE, continue
        if "CONTINUE" in decision_upper:
            return True

        # Fallback to evaluation metrics
        should_continue = evaluation.get("should_continue", False)

        # Override if we've hit max iterations
        if iteration >= MAX_ITERATIONS:
            logging.info("Max iterations reached, proceeding to synthesis")
            return False

        return should_continue


# --- Core Agent Instance --- #
core_agent = IterativeResearchAgent(name="CedLM_Autonomous_Researcher")

app = App(root_agent=core_agent, name="Learning_Agent_ADK")

__all__ = ["core_agent", "ACTIVE_CONFIG",
           "ACTIVE_CONFIG_NAME", "IterativeResearchAgent"]
