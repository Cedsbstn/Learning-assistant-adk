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
Unit tests verifying Google ADK pipeline integration and agent structure.
"""

import pytest
from google.adk.agents import Agent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.apps.app import App

import agent
from agent import (
    app,
    core_agent,
    curriculum_planner,
    markdown_generator,
    report_synthesizer,
    research_pipeline,
    section_researcher,
)


def test_adk_app_structure():
    """Verify that the Google ADK App instance is configured with Kythe root agent."""
    assert isinstance(app, App)
    assert app.name == "Kythe_ADK_Engine"
    assert app.root_agent == core_agent
    assert app.root_agent == research_pipeline


def test_adk_sequential_pipeline_subagents():
    """Verify that research_pipeline contains all four required sequential stages."""
    assert isinstance(research_pipeline, SequentialAgent)
    assert len(research_pipeline.sub_agents) == 4

    stage_names = [sa.name for sa in research_pipeline.sub_agents]
    assert stage_names == [
        "curriculum_planner",
        "deep_section_researcher",
        "report_synthesizer",
        "markdown_curriculum_report",
    ]


def test_adk_subagent_types_and_instructions():
    """Verify individual subagent types and configurations."""
    assert isinstance(curriculum_planner, LlmAgent)
    assert curriculum_planner.output_key == "curriculum_outline"

    assert isinstance(section_researcher, LlmAgent)
    assert section_researcher.name == "deep_section_researcher"
    assert section_researcher.output_key == "section_research"

    assert isinstance(report_synthesizer, LlmAgent)
    assert report_synthesizer.output_key == "final_cited_report"

    assert isinstance(markdown_generator, LlmAgent)
    assert markdown_generator.output_key == "final_markdown_curriculum"


def test_adk_exports():
    """Verify public symbols exported by agent module."""
    expected_exports = [
        "core_agent",
        "app",
        "ACTIVE_CONFIG",
        "ACTIVE_CONFIG_NAME",
        "research_pipeline",
        "curriculum_planner",
        "section_researcher",
        "report_synthesizer",
        "markdown_generator",
        "generate_curriculum_outline",
        "plan_section_queries",
        "perform_search",
        "research_section_content",
        "synthesize_final_report",
        "generate_learning_assessments",
    ]
    for sym in expected_exports:
        assert hasattr(agent, sym), f"Expected symbol '{sym}' missing from agent module"
