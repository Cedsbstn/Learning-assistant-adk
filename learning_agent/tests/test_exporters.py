import csv
import json
import os
import pytest
from pathlib import Path

from models import CurriculumModel, Artifact
from exporters import (
    ExportManager,
    MarkdownExporter,
    HTMLExporter,
    PDFExporter,
    QuizExporter,
    FlashcardsExporter,
    MetadataExporter,
    atomic_write,
    compute_sha256,
)


@pytest.fixture
def sample_curriculum():
    return CurriculumModel(
        topic="Distributed Consensus and Raft",
        generated_at="2026-08-21T12:00:00Z",
        run_id="run_test_exp_123",
        sections_content=[
            {
                "section_id": "sec_1",
                "ordinal": 1,
                "title": "Leader Election Mechanics",
                "objectives": ["Understand randomized election timeouts", "Analyze split votes"],
                "depth_target": "intermediate",
                "status": "COMPLETE",
                "score": 85.0,
                "content": "# Leader Election Mechanics\n\nRaft guarantees safety via leader election <cite source=\"src-1\"/>.\n\n```python\ndef start_election():\n    pass\n```",
                "sources": [{"source_id": "src-1", "title": "Raft Paper", "url": "https://raft.github.io"}],
            },
            {
                "section_id": "sec_2",
                "ordinal": 2,
                "title": "Log Replication and Commit Invariants",
                "objectives": ["Understand log matching property"],
                "depth_target": "advanced",
                "status": "EXHAUSTED",
                "score": 62.0,
                "content": "# Log Replication\n\nLeader replicates entries to followers <cite source=\"src-2\"/>.",
                "sources": [{"source_id": "src-2", "title": "Ongaro Thesis", "url": "https://web.stanford.edu/ongaro"}],
            },
        ],
        sources=[
            {"source_id": "src-1", "title": "Raft Paper", "canonical_url": "https://raft.github.io", "domain": "raft.github.io"},
            {"source_id": "src-2", "title": "Ongaro Thesis", "canonical_url": "https://web.stanford.edu/ongaro", "domain": "stanford.edu"},
        ],
        metrics={
            "total_sections": 2,
            "completed_sections": 1,
            "exhausted_sections_count": 1,
            "total_sources": 2,
        },
        quizzes=[
            {
                "question_id": "q1",
                "type": "multiple_choice",
                "prompt": "How does Raft prevent split vote cycles?",
                "choices": ["Randomized election timeouts", "Global lock", "Round robin", "Manual intervention"],
                "answer": "Randomized election timeouts",
                "explanation": "Randomized timeouts spread candidate declarations apart.",
                "section_id": "sec_1",
                "source_ids": ["src-1"],
            }
        ],
        flashcards=[
            {
                "card_id": "c1",
                "front": "What is the Log Matching Invariant in Raft?",
                "back": "If two logs contain an entry with the same index and term, they are identical up to that point.",
                "section_id": "sec_2",
                "tags": ["raft", "invariants"],
                "source_ids": ["src-2"],
                "difficulty": "advanced",
            }
        ],
        open_gaps=[{"category": "technical", "description": "Need formal TLA+ spec verification details"}],
        exhausted_sections=["Log Replication and Commit Invariants"],
    )


def test_atomic_write(tmp_path):
    dest = tmp_path / "test_atomic.txt"
    saved = atomic_write(str(dest), "Atomic file content")
    assert Path(saved).exists()
    assert Path(saved).read_text(encoding="utf-8") == "Atomic file content"


def test_markdown_exporter(sample_curriculum, tmp_path):
    md_path = str(tmp_path / "test_out.md")
    report = "## Detailed Report\n\nRaft ensures strong consistency [Raft Paper](https://raft.github.io)."
    art = MarkdownExporter.export(sample_curriculum, report, md_path)

    assert art.type == "markdown"
    assert Path(art.path).exists()
    content = Path(art.path).read_text(encoding="utf-8")
    assert "Leader Election Mechanics" in content
    assert "Section Quality & Research Depth" in content or "Research Depth and Quality Metrics" in content
    assert "Quality Note on Exhausted Sections" in content
    assert art.sha256 == compute_sha256(art.path)


def test_html_exporter(sample_curriculum, tmp_path):
    html_path = str(tmp_path / "test_out.html")
    report = "# Title\n\n## Subhead\n\nParagraph text with [Link](https://example.com)."
    art = HTMLExporter.export(sample_curriculum, report, html_path)

    assert art.type == "html"
    assert Path(art.path).exists()
    content = Path(art.path).read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "@media print" in content
    assert "Curriculum Modules" in content
    assert "Leader Election Mechanics" in content


def test_pdf_exporter(sample_curriculum, tmp_path):
    pdf_path = str(tmp_path / "test_out.pdf")
    report = "# Raft Distributed Consensus\n\n## Core Invariants\n\nDetailed technical prose describing leader election and log safety."
    art = PDFExporter.export(sample_curriculum, report, pdf_path)

    assert art.type == "pdf"
    assert Path(art.path).exists()
    # Check PDF header
    with open(art.path, "rb") as f:
        header = f.read(4)
        assert header == b"%PDF"


def test_quiz_and_flashcards_exporters(sample_curriculum, tmp_path):
    quiz_path = str(tmp_path / "test_out.quiz.json")
    art_quiz = QuizExporter.export(sample_curriculum, quiz_path)
    assert art_quiz.type == "quiz"
    quiz_data = json.loads(Path(art_quiz.path).read_text(encoding="utf-8"))
    assert len(quiz_data["quiz_questions"]) == 1
    assert quiz_data["quiz_questions"][0]["answer"] == "Randomized election timeouts"

    fc_json_path = str(tmp_path / "test_out.flashcards.json")
    art_fc_json = FlashcardsExporter.export_json(sample_curriculum, fc_json_path)
    assert art_fc_json.type == "flashcards_json"
    fc_data = json.loads(Path(art_fc_json.path).read_text(encoding="utf-8"))
    assert len(fc_data["flashcards"]) == 1

    fc_csv_path = str(tmp_path / "test_out.flashcards.csv")
    art_fc_csv = FlashcardsExporter.export_csv(sample_curriculum, fc_csv_path)
    assert art_fc_csv.type == "flashcards_csv"
    with open(art_fc_csv.path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 2  # Header + 1 row
        assert rows[0][0] == "card_id"
        assert rows[1][0] == "c1"


def test_export_manager_all(sample_curriculum, tmp_path):
    manager = ExportManager(output_dir=str(tmp_path))
    formats = ["markdown", "html", "pdf", "quiz", "flashcards", "metadata"]
    artifacts = manager.export_all(
        curriculum_model=sample_curriculum,
        final_report="## Full Final Report\n\nContent here.",
        formats=formats,
        run_id=sample_curriculum.run_id,
    )

    types = {a.type for a in artifacts}
    assert "markdown" in types
    assert "html" in types
    assert "pdf" in types
    assert "quiz" in types
    assert "flashcards_json" in types
    assert "flashcards_csv" in types
    assert "metadata" in types
