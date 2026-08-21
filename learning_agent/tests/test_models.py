import json
import pytest
from models import (
    Run,
    RunStatus,
    Section,
    SectionStatus,
    Pass,
    PassStatus,
    Source,
    SectionSource,
    Gap,
    GapStatus,
    GapCategory,
    SectionEvaluation,
    GapSpec,
    CurriculumOutline,
    OutlineSection,
    QuizQuestion,
    Flashcard,
    Artifact,
)
from config import ResearchConfig, get_config_by_name


def test_run_model():
    run = Run(
        run_id="run-123",
        topic="Advanced Rust",
        status=RunStatus.INITIALIZING,
        config_json=json.dumps({"min_sources": 3}),
    )
    d = run.to_dict()
    assert d["run_id"] == "run-123"
    assert d["status"] == "INITIALIZING"

    restored = Run.from_row(d)
    assert restored.run_id == "run-123"
    assert restored.status == RunStatus.INITIALIZING


def test_section_model():
    sec = Section(
        section_id="sec-1",
        run_id="run-123",
        ordinal=1,
        title="Ownership and Borrowing",
        objectives_json=json.dumps(["Learn lifetimes", "Understand move semantics"]),
        depth_target="advanced",
        status=SectionStatus.PENDING,
    )
    assert sec.objectives == ["Learn lifetimes", "Understand move semantics"]
    d = sec.to_dict()
    assert d["ordinal"] == 1
    assert d["status"] == "PENDING"

    restored = Section.from_row(d)
    assert restored.section_id == "sec-1"
    assert restored.objectives == ["Learn lifetimes", "Understand move semantics"]


def test_section_evaluation_serialization():
    eval_res = SectionEvaluation(
        passed=True,
        score=88.5,
        word_count=650,
        source_count=4,
        unique_domains=3,
        has_examples=True,
        has_technical_details=True,
        citation_coverage=0.75,
        open_gaps=[GapSpec(category="examples", description="Add real world case study")],
        reasons=["All gates passed"],
    )
    d = eval_res.to_dict()
    assert d["passed"] is True
    assert d["score"] == 88.5

    restored = SectionEvaluation.from_dict(d)
    assert restored.passed is True
    assert len(restored.open_gaps) == 1
    assert restored.open_gaps[0].category == "examples"


def test_config_validation():
    cfg = ResearchConfig()
    assert cfg.validate() is True

    with pytest.raises(ValueError):
        invalid_cfg = ResearchConfig(min_word_count=50)
        invalid_cfg.validate()

    with pytest.raises(ValueError):
        invalid_cfg = ResearchConfig(min_sources=0)
        invalid_cfg.validate()

    with pytest.raises(ValueError):
        invalid_cfg = ResearchConfig(min_completeness=150.0)
        invalid_cfg.validate()

    with pytest.raises(ValueError):
        invalid_cfg = ResearchConfig(fetch_timeout_s=-5.0)
        invalid_cfg.validate()


def test_config_presets():
    for name in ["quick", "standard", "deep", "comprehensive"]:
        cfg = get_config_by_name(name)
        assert cfg.validate() is True
        assert cfg.max_iterations >= 1
