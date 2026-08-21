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
Run Orchestrator and Control Plane for Kythe Autonomous Deep Research Agent.

Implements the section-by-section state machine, pass scheduling, closed-loop
quality gating, crash recovery, and rich export orchestration.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import DEFAULT_CONFIG, ResearchConfig
from models import (
    Artifact,
    CurriculumModel,
    CurriculumOutline,
    EvidenceChunk,
    Flashcard,
    Gap,
    GapCategory,
    GapStatus,
    OutlineSection,
    Pass,
    PassStatus,
    QuizQuestion,
    Run,
    RunStatus,
    Section,
    SectionEvaluation,
    SectionStatus,
    Source,
    utc_now_iso,
)
from persistence import DatabaseManager, PersistenceRepository
from quality import QualityEvaluator
from reader import DeepReader

logger = logging.getLogger(__name__)


class RunOrchestrator:
    """Deterministic orchestrator managing research runs, passes, transitions, and resumption."""

    def __init__(
        self,
        db_path: str = "research.db",
        search_provider: Optional[Callable[[List[str], int], List[Dict[str, Any]]]] = None,
        llm_provider: Optional[Any] = None,
    ):
        self.db_manager = DatabaseManager(db_path)
        self.repo = PersistenceRepository(self.db_manager)
        self.search_provider = search_provider
        self.llm_provider = llm_provider

    # --- Run Creation and Planning --- #

    def create_run(self, topic: str, config: Optional[ResearchConfig] = None) -> str:
        """
        Create a new research run in SQLite.
        """
        cfg = config or DEFAULT_CONFIG
        cfg.validate()
        run_id = f"run_{uuid.uuid4().hex[:12]}"

        run = Run(
            run_id=run_id,
            topic=topic,
            status=RunStatus.INITIALIZING,
            config_json=json.dumps(cfg.to_dict()),
            outline_version=1,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        self.repo.create_run(run)
        logger.info("Created research run %s for topic '%s'", run_id, topic)
        return run_id

    def plan_curriculum(self, run_id: str) -> CurriculumOutline:
        """
        Generate curriculum outline and initial section records.
        Transitions run to OUTLINE_REVIEW.
        """
        run = self.repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        config = ResearchConfig.from_dict(json.loads(run.config_json))

        # Import agent planner dynamically to allow override or mock
        from agent import generate_curriculum_outline

        if self.llm_provider and hasattr(self.llm_provider, "generate_curriculum_outline"):
            outline, outline_sections = self.llm_provider.generate_curriculum_outline(run.topic, config)
        else:
            outline, outline_sections = generate_curriculum_outline(run.topic, config)

        # Convert outline sections into Section records
        sections: List[Section] = []
        for idx, osec in enumerate(outline_sections, start=1):
            sec_id = f"sec_{run_id}_{idx}"
            sections.append(
                Section(
                    section_id=sec_id,
                    run_id=run_id,
                    ordinal=idx,
                    title=osec.title,
                    objectives_json=json.dumps(osec.objectives),
                    depth_target=osec.depth_target or "intermediate",
                    status=SectionStatus.PENDING,
                    attempt_count=0,
                    updated_at=utc_now_iso(),
                )
            )

        # Save sections to DB
        self.repo.save_sections(sections)
        self.repo.update_run_status(run_id, RunStatus.OUTLINE_REVIEW)
        logger.info("Planned %d sections for run %s; transitioned to OUTLINE_REVIEW", len(sections), run_id)
        return outline

    def approve_outline(
        self,
        run_id: str,
        edits: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Approve the curriculum outline, optionally applying user edits.
        Transitions run to RESEARCHING.
        """
        run = self.repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        if edits:
            sections: List[Section] = []
            for idx, item in enumerate(edits, start=1):
                sec_id = item.get("section_id") or f"sec_{run_id}_{idx}"
                objectives = item.get("objectives", [])
                sections.append(
                    Section(
                        section_id=sec_id,
                        run_id=run_id,
                        ordinal=idx,
                        title=item.get("title", f"Section {idx}"),
                        objectives_json=json.dumps(objectives),
                        depth_target=item.get("depth_target", "intermediate"),
                        status=SectionStatus.PENDING,
                        attempt_count=0,
                        updated_at=utc_now_iso(),
                    )
                )
            self.repo.save_sections(sections)

        self.repo.update_run_status(run_id, RunStatus.RESEARCHING)
        logger.info("Outline approved for run %s; transitioned to RESEARCHING", run_id)

    # --- Section Pass Execution --- #

    def execute_section_pass(
        self,
        section: Section,
        config: ResearchConfig,
        run_id: str,
        topic: str,
    ) -> Tuple[Pass, SectionEvaluation]:
        """
        Execute a single research pass on a section:
        1. Create Pass record in STARTED state.
        2. Plan search queries from objectives and open gaps.
        3. Search for candidate URLs.
        4. Deep read (fetch, validate, extract, chunk) candidates safely.
        5. Synthesize section draft via Section Researcher.
        6. Evaluate section draft with deterministic quality gates.
        7. Atomically commit pass completion, draft, and evaluation.
        """
        # Increment section attempt count
        attempt_ordinal = self.repo.increment_section_attempt(section.section_id)
        pass_id = f"pass_{section.section_id}_{attempt_ordinal}_{uuid.uuid4().hex[:6]}"

        p = Pass(
            pass_id=pass_id,
            section_id=section.section_id,
            ordinal=attempt_ordinal,
            status=PassStatus.STARTED,
            started_at=utc_now_iso(),
        )
        self.repo.create_pass(p)
        self.repo.update_section_status(section.section_id, SectionStatus.RESEARCHING)

        # 1. Open Gaps
        open_gaps = self.repo.get_open_gaps(section.section_id)

        # 2. Query Planning
        from agent import plan_section_queries, research_section_content

        if self.llm_provider and hasattr(self.llm_provider, "plan_section_queries"):
            queries = self.llm_provider.plan_section_queries(topic, section, open_gaps, config)
        else:
            queries = plan_section_queries(topic, section, open_gaps, config)

        query_plan_json = json.dumps(queries)

        # 3. Search
        candidate_results: List[Dict[str, Any]] = []
        if self.search_provider:
            candidate_results = self.search_provider(queries, config.max_candidates_per_query)
        else:
            from agent import perform_search
            candidate_results = perform_search(queries, config.max_candidates_per_query)

        # 4. Deep Reading & Evidence Gathering
        deep_reader = DeepReader(
            repo=self.repo,
            fetch_timeout_s=config.fetch_timeout_s,
            max_page_bytes=config.max_page_bytes,
            cache_ttl_hours=config.cache_ttl_hours,
        )

        all_evidence_chunks: List[EvidenceChunk] = []
        # Get existing sources linked to this section
        existing_sources = self.repo.get_sources_for_section(section.section_id)
        existing_urls = {s.canonical_url for s in existing_sources}

        # Process candidates up to max_pages_per_pass
        fetched_count = 0
        src_counter = len(self.repo.get_all_sources_for_run(run_id)) + 1

        for cand in candidate_results:
            if fetched_count >= config.max_pages_per_pass:
                break

            raw_url = cand.get("url") or cand.get("link")
            if not raw_url:
                continue

            source_id = f"src-{src_counter}"
            title_hint = cand.get("title", "")
            snippet_hint = cand.get("snippet", "")

            source, chunks = deep_reader.process_url(
                raw_url=raw_url,
                source_id=source_id,
                title_hint=title_hint,
                snippet_hint=snippet_hint,
            )

            # Link source to section
            self.repo.link_section_source(
                section_id=section.section_id,
                source_id=source.source_id,
                pass_id=pass_id,
                relevance_score=1.0,
                selected=True,
                citation_count=0,
            )

            if chunks:
                all_evidence_chunks.extend(chunks)

            src_counter += 1
            fetched_count += 1

        # 5. Section Synthesis
        all_section_sources = self.repo.get_sources_for_section(section.section_id)
        latest_pass = self.repo.get_latest_pass(section.section_id)
        prior_draft = latest_pass.draft_text if latest_pass and latest_pass.pass_id != pass_id else None

        if self.llm_provider and hasattr(self.llm_provider, "research_section_content"):
            draft = self.llm_provider.research_section_content(
                topic=topic,
                section=section,
                evidence_chunks=all_evidence_chunks,
                section_sources=all_section_sources,
                gaps=open_gaps,
                prior_draft=prior_draft,
                config=config,
            )
        else:
            draft = research_section_content(
                topic=topic,
                section=section,
                evidence_chunks=all_evidence_chunks,
                section_sources=all_section_sources,
                gaps=open_gaps,
                prior_draft=prior_draft,
                config=config,
            )

        # 6. Quality Evaluation
        evaluator = QualityEvaluator(config)
        evaluation = evaluator.evaluate_section(section, draft, all_section_sources)
        evaluation_json = json.dumps(evaluation.to_dict())

        # 7. Checkpoint Pass Atomically
        self.repo.complete_pass(
            pass_id=pass_id,
            draft_text=draft,
            evaluation_json=evaluation_json,
            query_plan_json=query_plan_json,
        )

        # Update Section & Gap State
        if evaluation.passed:
            self.repo.update_section_status(section.section_id, SectionStatus.COMPLETE, evaluation.score)
            self.repo.resolve_gaps_for_section(section.section_id, pass_id)
            logger.info("Section %s PASSED quality gate on pass %d (score: %.1f)", section.section_id, attempt_ordinal, evaluation.score)
        elif attempt_ordinal >= config.max_iterations:
            self.repo.update_section_status(section.section_id, SectionStatus.EXHAUSTED, evaluation.score)
            # Supersede old gaps with latest open gaps
            self.repo.supersede_gaps_for_section(section.section_id, pass_id)
            new_gaps = [
                Gap(
                    gap_id=f"gap_{uuid.uuid4().hex[:8]}",
                    section_id=section.section_id,
                    category=GapCategory(g.category) if isinstance(g.category, str) else g.category,
                    description=g.description,
                    severity=g.severity,
                    status=GapStatus.OPEN,
                    created_pass_id=pass_id,
                )
                for g in evaluation.open_gaps
            ]
            self.repo.save_gaps(new_gaps)
            logger.warning("Section %s EXHAUSTED max iterations (%d) (score: %.1f)", section.section_id, config.max_iterations, evaluation.score)
        else:
            self.repo.update_section_status(section.section_id, SectionStatus.NEEDS_MORE_RESEARCH, evaluation.score)
            self.repo.supersede_gaps_for_section(section.section_id, pass_id)
            new_gaps = [
                Gap(
                    gap_id=f"gap_{uuid.uuid4().hex[:8]}",
                    section_id=section.section_id,
                    category=GapCategory(g.category) if isinstance(g.category, str) else g.category,
                    description=g.description,
                    severity=g.severity,
                    status=GapStatus.OPEN,
                    created_pass_id=pass_id,
                )
                for g in evaluation.open_gaps
            ]
            self.repo.save_gaps(new_gaps)
            logger.info("Section %s requires more research (score: %.1f, %d gaps)", section.section_id, evaluation.score, len(new_gaps))

        p.status = PassStatus.COMPLETED
        p.draft_text = draft
        p.evaluation_json = evaluation_json
        return p, evaluation

    # --- Run Loop Execution --- #

    def run_until_terminal(
        self,
        run_id: str,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> RunStatus:
        """
        Execute the iterative research control loop across all sections until completion.
        """
        run = self.repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        config = ResearchConfig.from_dict(json.loads(run.config_json))
        self.repo.update_run_status(run_id, RunStatus.RESEARCHING)

        sections = self.repo.get_sections(run_id)
        if not sections:
            logger.warning("No sections found for run %s. Running plan_curriculum first.", run_id)
            self.plan_curriculum(run_id)
            if not config.require_outline_approval:
                self.approve_outline(run_id)
            sections = self.repo.get_sections(run_id)

        # Section-by-section pass loop
        for section in sections:
            # Re-fetch section to get latest state
            sec = self.repo.get_section(section.section_id) or section
            while sec.status not in (SectionStatus.COMPLETE, SectionStatus.EXHAUSTED, SectionStatus.FAILED):
                if sec.attempt_count >= config.max_iterations:
                    self.repo.update_section_status(sec.section_id, SectionStatus.EXHAUSTED)
                    break

                if on_progress:
                    on_progress({
                        "event": "section_pass_start",
                        "run_id": run_id,
                        "section_id": sec.section_id,
                        "section_title": sec.title,
                        "ordinal": sec.ordinal,
                        "attempt": sec.attempt_count + 1,
                        "max_iterations": config.max_iterations,
                    })

                _, evaluation = self.execute_section_pass(sec, config, run_id, run.topic)

                if on_progress:
                    on_progress({
                        "event": "section_pass_end",
                        "run_id": run_id,
                        "section_id": sec.section_id,
                        "section_title": sec.title,
                        "passed": evaluation.passed,
                        "score": evaluation.score,
                        "word_count": evaluation.word_count,
                        "source_count": evaluation.source_count,
                        "open_gaps": len(evaluation.open_gaps),
                    })

                sec = self.repo.get_section(sec.section_id)

        # Final Synthesis and Export
        all_sections = self.repo.get_sections(run_id)
        has_failed = any(s.status == SectionStatus.FAILED for s in all_sections)
        all_terminal = all(s.status in (SectionStatus.COMPLETE, SectionStatus.EXHAUSTED) for s in all_sections)

        if all_terminal and (not has_failed or config.allow_exhausted_sections):
            self.repo.update_run_status(run_id, RunStatus.SYNTHESIZING)
            if on_progress:
                on_progress({"event": "synthesis_start", "run_id": run_id})

            self.export_run(run_id, config.export_formats)
            self.repo.update_run_status(run_id, RunStatus.COMPLETED, completed=True)

            if on_progress:
                on_progress({"event": "run_completed", "run_id": run_id})
            return RunStatus.COMPLETED

        return RunStatus.RESEARCHING

    def resume_run(
        self,
        run_id: str,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> RunStatus:
        """
        Safely and idempotently resume an interrupted research run from SQLite.
        """
        run = self.repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        logger.info("Resuming research run %s (current status: %s)", run_id, run.status.value)

        # Clean up any in-flight pass left in STARTED status
        sections = self.repo.get_sections(run_id)
        for s in sections:
            passes = self.repo.get_passes_for_section(s.section_id)
            for p in passes:
                if p.status == PassStatus.STARTED:
                    self.repo.mark_pass_interrupted(p.pass_id)

            if s.status in (SectionStatus.RESEARCHING, SectionStatus.EVALUATING):
                self.repo.update_section_status(s.section_id, SectionStatus.NEEDS_MORE_RESEARCH)

        return self.run_until_terminal(run_id, on_progress=on_progress)

    def pause_run(self, run_id: str) -> None:
        """Pause a run."""
        self.repo.update_run_status(run_id, RunStatus.PAUSED)
        logger.info("Paused run %s", run_id)

    # --- Rich Exports --- #

    def export_run(
        self,
        run_id: str,
        formats: Optional[List[str]] = None,
    ) -> List[Artifact]:
        """
        Generate rich learning exports (Markdown, HTML, PDF, Quiz, Flashcards, Metadata)
        and record them in SQLite artifacts table.
        """
        run = self.repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        config = ResearchConfig.from_dict(json.loads(run.config_json))
        requested_formats = formats or config.export_formats

        sections = self.repo.get_sections(run_id)
        all_sources = self.repo.get_all_sources_for_run(run_id)

        # Build Section Content map and drafts
        sections_content: List[Dict[str, Any]] = []
        section_drafts: Dict[str, str] = {}
        exhausted_sections: List[str] = []

        for s in sections:
            latest_pass = self.repo.get_latest_pass(s.section_id)
            draft = latest_pass.draft_text if latest_pass and latest_pass.draft_text else ""
            section_drafts[s.title] = draft
            if s.status == SectionStatus.EXHAUSTED:
                exhausted_sections.append(s.title)

            sec_sources = self.repo.get_sources_for_section(s.section_id)
            sections_content.append({
                "section_id": s.section_id,
                "ordinal": s.ordinal,
                "title": s.title,
                "objectives": s.objectives,
                "depth_target": s.depth_target,
                "status": s.status.value,
                "score": s.latest_score,
                "content": draft,
                "sources": [src.to_dict() for src in sec_sources],
            })

        # Assemble full curriculum report
        from agent import synthesize_final_report, generate_learning_assessments

        if self.llm_provider and hasattr(self.llm_provider, "synthesize_final_report"):
            final_report = self.llm_provider.synthesize_final_report(
                run.topic, sections, section_drafts, all_sources, config
            )
        else:
            final_report = synthesize_final_report(
                run.topic, sections, section_drafts, all_sources, config
            )

        # Generate assessments
        if self.llm_provider and hasattr(self.llm_provider, "generate_learning_assessments"):
            quizzes, flashcards = self.llm_provider.generate_learning_assessments(
                run.topic, sections_content, all_sources, config
            )
        else:
            quizzes, flashcards = generate_learning_assessments(
                run.topic, sections_content, all_sources, config
            )

        # Build Canonical Curriculum Model
        curriculum_model = CurriculumModel(
            topic=run.topic,
            generated_at=utc_now_iso(),
            run_id=run_id,
            sections_content=sections_content,
            sources=[s.to_dict() for s in all_sources],
            metrics={
                "total_sections": len(sections),
                "completed_sections": sum(1 for s in sections if s.status == SectionStatus.COMPLETE),
                "exhausted_sections_count": len(exhausted_sections),
                "total_sources": len(all_sources),
                "average_score": sum((s.latest_score or 0) for s in sections) / max(1, len(sections)),
            },
            quizzes=[asdict(q) if hasattr(q, "question_id") else q for q in quizzes],
            flashcards=[asdict(f) if hasattr(f, "card_id") else f for f in flashcards],
            open_gaps=[g.to_dict() for s in sections for g in self.repo.get_open_gaps(s.section_id)],
            exhausted_sections=exhausted_sections,
        )

        # Trigger Exporters
        from exporters import ExportManager
        exporter = ExportManager(output_dir=config.output_dir)
        artifacts = exporter.export_all(
            curriculum_model=curriculum_model,
            final_report=final_report,
            formats=requested_formats,
            run_id=run_id,
        )

        # Save artifacts in SQLite
        for art in artifacts:
            self.repo.record_artifact(art)

        logger.info("Generated %d artifacts for run %s", len(artifacts), run_id)
        return artifacts

    def get_status(self, run_id: str) -> Dict[str, Any]:
        """Get comprehensive run metrics and section details."""
        run = self.repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        sections = self.repo.get_sections(run_id)
        sources = self.repo.get_all_sources_for_run(run_id)
        artifacts = self.repo.get_artifacts(run_id)

        complete_count = sum(1 for s in sections if s.status == SectionStatus.COMPLETE)
        exhausted_count = sum(1 for s in sections if s.status == SectionStatus.EXHAUSTED)
        pending_count = sum(1 for s in sections if s.status == SectionStatus.PENDING)
        active_count = sum(1 for s in sections if s.status == SectionStatus.RESEARCHING)

        return {
            "run_id": run.run_id,
            "topic": run.topic,
            "status": run.status.value,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
            "completed_at": run.completed_at,
            "total_sections": len(sections),
            "complete_sections": complete_count,
            "exhausted_sections": exhausted_count,
            "pending_sections": pending_count,
            "active_sections": active_count,
            "total_sources": len(sources),
            "artifacts": [a.to_dict() for a in artifacts],
            "sections": [
                {
                    "ordinal": s.ordinal,
                    "title": s.title,
                    "depth": s.depth_target,
                    "status": s.status.value,
                    "attempt_count": s.attempt_count,
                    "score": s.latest_score,
                    "open_gaps": len(self.repo.get_open_gaps(s.section_id)),
                }
                for s in sections
            ],
        }
