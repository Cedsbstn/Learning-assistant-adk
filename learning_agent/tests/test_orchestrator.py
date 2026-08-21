import json
import pytest
from unittest.mock import MagicMock

from config import ResearchConfig
from models import (
    Gap,
    GapCategory,
    GapStatus,
    OutlineSection,
    Pass,
    PassStatus,
    RunStatus,
    Section,
    SectionStatus,
    Source,
)
from orchestrator import RunOrchestrator


class MockLLMProvider:
    def __init__(self):
        self.call_counts = {}

    def generate_curriculum_outline(self, topic, config):
        self.call_counts["outline"] = self.call_counts.get("outline", 0) + 1
        sections = [
            OutlineSection(ordinal=1, title="Module 1: Foundations", objectives=["Learn basics"], depth_target="basic"),
            OutlineSection(ordinal=2, title="Module 2: Advanced Design", objectives=["Master internals"], depth_target="advanced"),
        ]
        from models import CurriculumOutline
        return CurriculumOutline(topic=topic, sections=sections), sections

    def plan_section_queries(self, topic, section, gaps, config):
        return [f"{topic} {section.title} guide", f"{topic} {section.title} architecture"]

    def research_section_content(self, topic, section, evidence_chunks, section_sources, gaps, prior_draft, config):
        self.call_counts[f"research_{section.section_id}"] = self.call_counts.get(f"research_{section.section_id}", 0) + 1
        attempt = self.call_counts[f"research_{section.section_id}"]

        s1 = section_sources[0].source_id if section_sources else "src-1"
        s2 = section_sources[1].source_id if len(section_sources) > 1 else s1

        # Simulate progressive improvement if needed
        return f"""# {section.title}

## Technical Overview and Mental Models
The internal architecture of {section.title} in {topic} provides strong guarantees <cite source="{s1}"/>.
To achieve high throughput, algorithmic complexity is minimized while verifying data structure invariants <cite source="{s2}"/>.

## Practical Code Implementation
```python
def execute_system():
    # Production example implementation
    return True
```

## Production Tradeoffs and Best Practices
We analyze memory layout and concurrency protocols <cite source="{s1}"/>.
""" + " ".join(["detailed analysis"] * 100)

    def synthesize_final_report(self, topic, sections, section_drafts, sources, config):
        return f"# {topic} Final Report\n\nExecutive summary and content across {len(sections)} sections."

    def generate_learning_assessments(self, topic, sections_content, sources, config):
        from agent import generate_learning_assessments
        return generate_learning_assessments(topic, sections_content, sources, config)


def mock_search(queries, max_candidates):
    return [
        {"url": "https://docs.example.org/guide", "title": "Official Docs", "snippet": "Technical guide"},
        {"url": "https://blog.example.org/arch", "title": "Architecture Deep Dive", "snippet": "Deep dive"},
    ]


@pytest.fixture
def orchestrator(tmp_path):
    db_file = tmp_path / "test_orch.db"
    mock_llm = MockLLMProvider()
    orch = RunOrchestrator(
        db_path=str(db_file),
        search_provider=mock_search,
        llm_provider=mock_llm,
    )
    return orch, mock_llm


def test_full_orchestration_workflow(orchestrator, tmp_path):
    orch, mock_llm = orchestrator
    cfg = ResearchConfig(
        min_word_count=100,
        min_sources=2,
        min_unique_domains=2,
        min_citation_coverage=0.4,
        max_iterations=3,
        output_dir=str(tmp_path / "output"),
    )

    run_id = orch.create_run(topic="Rust Concurrency", config=cfg)
    assert run_id.startswith("run_")

    outline = orch.plan_curriculum(run_id)
    assert len(outline.sections) == 2
    status_info = orch.get_status(run_id)
    assert status_info["status"] == "OUTLINE_REVIEW"
    assert status_info["total_sections"] == 2

    orch.approve_outline(run_id)
    assert orch.get_status(run_id)["status"] == "RESEARCHING"

    progress_events = []
    final_status = orch.run_until_terminal(run_id, on_progress=lambda e: progress_events.append(e))
    assert final_status == RunStatus.COMPLETED

    status_final = orch.get_status(run_id)
    assert status_final["status"] == "COMPLETED"
    assert status_final["complete_sections"] == 2
    assert len(status_final["artifacts"]) >= 5


def test_outline_approval_with_edits(orchestrator):
    orch, _ = orchestrator
    run_id = orch.create_run(topic="Modern Cryptography")
    orch.plan_curriculum(run_id)

    # Apply custom edits: change titles, add a section
    edits = [
        {"title": "Custom Module 1: Zero Knowledge Proofs", "objectives": ["zk-SNARKs"], "depth_target": "advanced"},
        {"title": "Custom Module 2: Elliptic Curves", "objectives": ["secp256k1"], "depth_target": "intermediate"},
        {"title": "Custom Module 3: Post-Quantum Crypto", "objectives": ["Lattice-based cryptography"], "depth_target": "advanced"},
    ]
    orch.approve_outline(run_id, edits=edits)

    status = orch.get_status(run_id)
    assert status["total_sections"] == 3
    assert status["sections"][0]["title"] == "Custom Module 1: Zero Knowledge Proofs"
    assert status["sections"][2]["title"] == "Custom Module 3: Post-Quantum Crypto"


def test_interruption_and_resumption(orchestrator, tmp_path):
    orch, mock_llm = orchestrator
    cfg = ResearchConfig(
        min_word_count=100,
        min_sources=1,
        min_citation_coverage=0.0,
        output_dir=str(tmp_path / "output"),
    )
    run_id = orch.create_run(topic="Database Internals", config=cfg)
    orch.plan_curriculum(run_id)
    orch.approve_outline(run_id)

    # Execute pass for section 1 manually
    sections = orch.repo.get_sections(run_id)
    sec1 = sections[0]
    p1, eval1 = orch.execute_section_pass(sec1, cfg, run_id, "Database Internals")
    assert eval1.passed is True
    assert orch.repo.get_section(sec1.section_id).status == SectionStatus.COMPLETE

    # Simulate process crash while section 2 was in STARTED pass
    sec2 = sections[1]
    orch.repo.update_section_status(sec2.section_id, SectionStatus.RESEARCHING)
    hanging_pass = Pass(
        pass_id=f"pass_{sec2.section_id}_1",
        section_id=sec2.section_id,
        ordinal=1,
        status=PassStatus.STARTED,
    )
    orch.repo.create_pass(hanging_pass)

    # Resume run
    resumed_status = orch.resume_run(run_id)
    assert resumed_status == RunStatus.COMPLETED

    # Verify hanging pass was marked INTERRUPTED and section completed
    passes = orch.repo.get_passes_for_section(sec2.section_id)
    assert any(p.status == PassStatus.INTERRUPTED for p in passes)
    assert orch.repo.get_section(sec2.section_id).status == SectionStatus.COMPLETE
    # Verify section 1 was not re-researched
    assert orch.repo.get_section(sec1.section_id).attempt_count == 1


def test_section_exhaustion_on_max_iterations(orchestrator, tmp_path):
    orch, mock_llm = orchestrator
    # Set high min_sources that mock won't satisfy
    cfg = ResearchConfig(
        min_word_count=100,
        min_sources=10,  # Impossible threshold with 2 search results
        max_iterations=2,
        output_dir=str(tmp_path / "output"),
    )
    run_id = orch.create_run(topic="Impossible Standards", config=cfg)
    orch.plan_curriculum(run_id)
    orch.approve_outline(run_id)

    orch.run_until_terminal(run_id)
    status = orch.get_status(run_id)
    assert status["status"] == "COMPLETED"
    assert status["exhausted_sections"] == 2
    assert all(s["status"] == "EXHAUSTED" for s in status["sections"])
