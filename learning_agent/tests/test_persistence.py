import json
import os
import tempfile
import pytest
from models import (
    Run,
    RunStatus,
    Section,
    SectionStatus,
    Pass,
    PassStatus,
    Source,
    Gap,
    GapStatus,
    GapCategory,
    Artifact,
)
from persistence import DatabaseManager, PersistenceRepository


@pytest.fixture
def repo(tmp_path):
    db_file = tmp_path / "test_research.db"
    mgr = DatabaseManager(str(db_file))
    return PersistenceRepository(mgr)


def test_run_crud(repo):
    run = Run(
        run_id="run-test-1",
        topic="Machine Learning Transformers",
        status=RunStatus.INITIALIZING,
        config_json=json.dumps({"min_word_count": 500}),
    )
    repo.create_run(run)

    fetched = repo.get_run("run-test-1")
    assert fetched is not None
    assert fetched.topic == "Machine Learning Transformers"
    assert fetched.status == RunStatus.INITIALIZING

    repo.update_run_status("run-test-1", RunStatus.RESEARCHING)
    updated = repo.get_run("run-test-1")
    assert updated.status == RunStatus.RESEARCHING

    runs = repo.list_runs(RunStatus.RESEARCHING)
    assert len(runs) == 1
    assert runs[0].run_id == "run-test-1"


def test_section_batch_and_cas(repo):
    run = Run(run_id="run-test-2", topic="Quantum Computing")
    repo.create_run(run)

    sec1 = Section(
        section_id="sec-1",
        run_id="run-test-2",
        ordinal=1,
        title="Qubits and Superposition",
        objectives_json=json.dumps(["Understand Bloch sphere"]),
    )
    sec2 = Section(
        section_id="sec-2",
        run_id="run-test-2",
        ordinal=2,
        title="Quantum Gates",
        objectives_json=json.dumps(["Hadamard, CNOT"]),
    )
    repo.save_sections([sec1, sec2])

    sections = repo.get_sections("run-test-2")
    assert len(sections) == 2
    assert sections[0].title == "Qubits and Superposition"
    assert sections[1].title == "Quantum Gates"

    # Test Compare-and-Swap (CAS)
    success = repo.compare_and_swap_section_status("sec-1", SectionStatus.PENDING, SectionStatus.RESEARCHING)
    assert success is True
    assert repo.get_section("sec-1").status == SectionStatus.RESEARCHING

    # Attempt CAS with wrong expected status
    fail_cas = repo.compare_and_swap_section_status("sec-1", SectionStatus.PENDING, SectionStatus.COMPLETE)
    assert fail_cas is False
    assert repo.get_section("sec-1").status == SectionStatus.RESEARCHING

    # Increment attempt count
    cnt = repo.increment_section_attempt("sec-1")
    assert cnt == 1
    cnt2 = repo.increment_section_attempt("sec-1")
    assert cnt2 == 2


def test_pass_and_interruption(repo):
    run = Run(run_id="run-test-3", topic="Operating Systems")
    repo.create_run(run)
    sec = Section(section_id="sec-os-1", run_id="run-test-3", ordinal=1, title="Virtual Memory")
    repo.save_sections([sec])

    p1 = Pass(pass_id="pass-1", section_id="sec-os-1", ordinal=1, status=PassStatus.STARTED)
    repo.create_pass(p1)

    passes = repo.get_passes_for_section("sec-os-1")
    assert len(passes) == 1
    assert passes[0].status == PassStatus.STARTED

    # Mark interrupted
    repo.mark_pass_interrupted("pass-1")
    latest = repo.get_latest_pass("sec-os-1")
    assert latest.status == PassStatus.INTERRUPTED

    # Complete a second pass
    p2 = Pass(pass_id="pass-2", section_id="sec-os-1", ordinal=2, status=PassStatus.STARTED)
    repo.create_pass(p2)
    repo.complete_pass("pass-2", draft_text="Draft content", evaluation_json="{}")

    p2_res = repo.get_latest_pass("sec-os-1")
    assert p2_res.status == PassStatus.COMPLETED
    assert p2_res.draft_text == "Draft content"


def test_sources_and_section_sources(repo):
    run = Run(run_id="run-test-4", topic="Compiler Design")
    repo.create_run(run)
    sec1 = Section(section_id="sec-c-1", run_id="run-test-4", ordinal=1, title="Lexing")
    sec2 = Section(section_id="sec-c-2", run_id="run-test-4", ordinal=2, title="Parsing")
    repo.save_sections([sec1, sec2])

    p1 = Pass(pass_id="pass-c-1", section_id="sec-c-1", ordinal=1, status=PassStatus.STARTED)
    p2 = Pass(pass_id="pass-c-2", section_id="sec-c-2", ordinal=1, status=PassStatus.STARTED)
    repo.create_pass(p1)
    repo.create_pass(p2)

    src1 = Source(
        source_id="src-1",
        canonical_url="https://example.com/dragons-book",
        title="Dragon Book",
        domain="example.com",
    )
    repo.upsert_source(src1)

    # Link to section 1
    repo.link_section_source(section_id="sec-c-1", source_id="src-1", pass_id="pass-c-1", citation_count=3)
    # Link same source to section 2 (independent association)
    repo.link_section_source(section_id="sec-c-2", source_id="src-1", pass_id="pass-c-2", citation_count=1)

    sec1_sources = repo.get_sources_for_section("sec-c-1")
    assert len(sec1_sources) == 1
    assert sec1_sources[0].source_id == "src-1"

    sec2_sources = repo.get_sources_for_section("sec-c-2")
    assert len(sec2_sources) == 1
    assert sec2_sources[0].source_id == "src-1"

    # Check run sources
    all_run_sources = repo.get_all_sources_for_run("run-test-4")
    assert len(all_run_sources) == 1


def test_gaps_lifecycle(repo):
    run = Run(run_id="run-test-5", topic="Distributed Systems")
    repo.create_run(run)
    sec = Section(section_id="sec-dist-1", run_id="run-test-5", ordinal=1, title="Raft Consensus")
    repo.save_sections([sec])

    g1 = Gap(
        gap_id="gap-1",
        section_id="sec-dist-1",
        category=GapCategory.EXAMPLES,
        description="Missing log compaction example",
    )
    g2 = Gap(
        gap_id="gap-2",
        section_id="sec-dist-1",
        category=GapCategory.TECHNICAL,
        description="Need leader election state machine details",
    )
    repo.save_gaps([g1, g2])

    open_gaps = repo.get_open_gaps("sec-dist-1")
    assert len(open_gaps) == 2

    # Resolve gaps
    repo.resolve_gaps_for_section("sec-dist-1", pass_id="pass-dist-2")
    remaining_open = repo.get_open_gaps("sec-dist-1")
    assert len(remaining_open) == 0


def test_cache_and_artifacts(repo):
    # Cache
    repo.set_cached_page(
        url="https://example.com/cached?utm_source=test",
        canonical_url="https://example.com/cached",
        content_hash="abc123hash",
        status_code=200,
        content_type="text/html",
        extracted_text="Cached extracted content",
        metadata={"title": "Test Page"},
        ttl_hours=24,
    )
    cached = repo.get_cached_page("https://example.com/cached")
    assert cached is not None
    assert cached["extracted_text"] == "Cached extracted content"

    # Artifact
    run = Run(run_id="run-test-6", topic="Algorithms")
    repo.create_run(run)
    art = Artifact(
        artifact_id="art-1",
        run_id="run-test-6",
        type="markdown",
        path="output/algo.md",
        sha256="fakehash256",
    )
    repo.record_artifact(art)
    artifacts = repo.get_artifacts("run-test-6")
    assert len(artifacts) == 1
    assert artifacts[0].type == "markdown"
