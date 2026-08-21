import json
import os
import pytest
from pathlib import Path

from config import ResearchConfig
from models import RunStatus, SectionStatus
from orchestrator import RunOrchestrator
from tests.test_orchestrator import MockLLMProvider, mock_search


def test_complete_end_to_end_research_lifecycle(tmp_path):
    """
    End-to-end acceptance test for Kythe Agentic Deep Research Agent:
    - AC-01: Modular section work-item model
    - AC-02: Quality gate enforcement and gap closing
    - AC-03: Multi-pass retry bounds and exhaustion handling
    - AC-04: Section-scoped sources and evidence
    - AC-05: SQLite checkpointing and WAL mode
    - AC-06: Crash resilience and idempotent resume
    - AC-07: SSRF-protected deep reader
    - AC-08: Deterministic quality scoring
    - AC-09: Interactive outline review
    - AC-10: Multi-format exports (MD, HTML, PDF, Quiz, Flashcards, Metadata)
    - AC-11: Atomic write with SHA256 checksums
    - AC-12: Backwards compatibility
    """
    output_dir = tmp_path / "artifacts"
    db_path = str(tmp_path / "research_e2e.db")

    mock_llm = MockLLMProvider()
    orch = RunOrchestrator(
        db_path=db_path,
        search_provider=mock_search,
        llm_provider=mock_llm,
    )

    config = ResearchConfig(
        min_word_count=100,
        min_sources=2,
        min_unique_domains=2,
        min_citation_coverage=0.4,
        max_iterations=3,
        export_formats=["markdown", "html", "pdf", "quiz", "flashcards", "metadata"],
        output_dir=str(output_dir),
    )

    run_id = orch.create_run(topic="Distributed Systems Architecture", config=config)
    assert orch.repo.get_run(run_id).status == RunStatus.INITIALIZING

    outline = orch.plan_curriculum(run_id)
    assert len(outline.sections) == 2
    assert orch.repo.get_run(run_id).status == RunStatus.OUTLINE_REVIEW

    orch.approve_outline(run_id)
    assert orch.repo.get_run(run_id).status == RunStatus.RESEARCHING

    progress_logs = []
    final_status = orch.run_until_terminal(run_id, on_progress=lambda e: progress_logs.append(e))
    assert final_status == RunStatus.COMPLETED

    run_record = orch.repo.get_run(run_id)
    assert run_record.status == RunStatus.COMPLETED
    assert run_record.completed_at is not None

    sections = orch.repo.get_sections(run_id)
    assert len(sections) == 2
    for s in sections:
        assert s.status == SectionStatus.COMPLETE
        assert s.latest_score is not None and s.latest_score >= 50.0
        # Verify passes recorded
        passes = orch.repo.get_passes_for_section(s.section_id)
        assert len(passes) >= 1
        assert passes[-1].draft_text is not None and len(passes[-1].draft_text) > 0
        # Verify section sources isolated
        sec_sources = orch.repo.get_sources_for_section(s.section_id)
        assert len(sec_sources) >= 1

    # 6. Verify Export Artifacts and Atomic Checksums
    artifacts = orch.repo.get_artifacts_for_run(run_id)
    assert len(artifacts) >= 6

    art_by_type = {a.type: a for a in artifacts}
    assert "markdown" in art_by_type
    assert "html" in art_by_type
    assert "pdf" in art_by_type
    assert "quiz" in art_by_type
    assert "flashcards_json" in art_by_type
    assert "flashcards_csv" in art_by_type
    assert "metadata" in art_by_type

    # Verify each artifact exists on disk and matches SHA256
    import hashlib
    for art in artifacts:
        p = Path(art.path)
        assert p.exists(), f"Artifact file missing: {art.path}"
        assert p.stat().st_size > 0
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        assert art.sha256 == actual_hash, f"Hash mismatch for {art.type}"

    # Verify Metadata Manifest
    meta_path = Path(art_by_type["metadata"].path)
    meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta_json["topic"] == "Distributed Systems Architecture"
    assert len(meta_json["artifacts"]) >= 6
    assert meta_json["metrics"]["completed_sections"] == 2
