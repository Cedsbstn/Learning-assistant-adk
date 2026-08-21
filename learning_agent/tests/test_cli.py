import io
import json
import pytest
from unittest.mock import patch, MagicMock

from cli import format_depth_label, interactive_outline_review
from main import build_parser
from config import ResearchConfig
from models import OutlineSection, RunStatus
from orchestrator import RunOrchestrator
from tests.test_orchestrator import MockLLMProvider, mock_search


def test_build_parser_subcommands():
    parser = build_parser()

    args = parser.parse_args(["research", "Python Concurrency", "--preset", "deep", "--auto-approve", "--formats", "markdown,pdf"])
    assert args.command == "research"
    assert args.topic == "Python Concurrency"
    assert args.preset == "deep"
    assert args.auto_approve is True
    assert args.formats == "markdown,pdf"

    args = parser.parse_args(["resume", "run_12345", "--db", "custom.db"])
    assert args.command == "resume"
    assert args.run_id == "run_12345"
    assert args.db == "custom.db"

    args = parser.parse_args(["status", "run_12345"])
    assert args.command == "status"
    assert args.run_id == "run_12345"

    args = parser.parse_args(["list-runs", "--status", "completed"])
    assert args.command == "list-runs"
    assert args.status == "completed"

    args = parser.parse_args(["export", "run_12345", "--format", "html,pdf"])
    assert args.command == "export"
    assert args.run_id == "run_12345"
    assert args.format == "html,pdf"


def test_format_depth_label():
    assert "Basic" in format_depth_label("basic")
    assert "Advanced" in format_depth_label("advanced")
    assert "Intermediate" in format_depth_label("intermediate")


def test_interactive_outline_review_actions(tmp_path):
    db_file = str(tmp_path / "test_cli_rev.db")
    orch = RunOrchestrator(
        db_path=db_file,
        search_provider=mock_search,
        llm_provider=MockLLMProvider(),
    )
    cfg = ResearchConfig(min_word_count=100)
    run_id = orch.create_run(topic="Operating Systems", config=cfg)
    orch.plan_curriculum(run_id)

    # Edit section 1 title, change section 2 depth to advanced, add section 3, then approve
    inputs = [
        "2", "1", "OS Kernel Architecture", "",
        "6", "2", "3",
        "3", "Virtual Memory", "Paging, Segmentation", "advanced",
        "1",
    ]

    with patch("builtins.input", side_effect=inputs):
        approved = interactive_outline_review(orch, run_id, "Operating Systems", cfg)

    assert approved is True
    status = orch.get_status(run_id)
    assert status["total_sections"] == 3
    assert status["sections"][0]["title"] == "OS Kernel Architecture"
    assert status["sections"][1]["depth"] == "advanced"
    assert status["sections"][2]["title"] == "Virtual Memory"
