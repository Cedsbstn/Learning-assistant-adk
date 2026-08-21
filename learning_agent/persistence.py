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
SQLite Persistence Layer for Kythe Autonomous Deep Research Agent.

Implements durable state, explicit schema migrations, transaction safety,
and granular repositories for all research entities.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from models import (
    Artifact,
    Claim,
    Gap,
    GapCategory,
    GapStatus,
    Pass,
    PassStatus,
    Run,
    RunStatus,
    Section,
    SectionSource,
    SectionStatus,
    Source,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

# Schema Version 1
CURRENT_SCHEMA_VERSION = 1

SCHEMA_V1_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        topic TEXT NOT NULL,
        status TEXT NOT NULL,
        config_json TEXT NOT NULL,
        outline_version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sections (
        section_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        title TEXT NOT NULL,
        objectives_json TEXT NOT NULL,
        depth_target TEXT NOT NULL DEFAULT 'intermediate',
        status TEXT NOT NULL,
        attempt_count INTEGER NOT NULL DEFAULT 0,
        latest_score REAL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS passes (
        pass_id TEXT PRIMARY KEY,
        section_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        query_plan_json TEXT,
        draft_text TEXT,
        evaluation_json TEXT,
        FOREIGN KEY (section_id) REFERENCES sections(section_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sources (
        source_id TEXT PRIMARY KEY,
        canonical_url TEXT UNIQUE NOT NULL,
        title TEXT NOT NULL,
        domain TEXT NOT NULL,
        fetched_at TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        content_type TEXT NOT NULL,
        extraction_status TEXT NOT NULL,
        extracted_text TEXT,
        snippet TEXT,
        cache_pointer TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS section_sources (
        section_id TEXT NOT NULL,
        source_id TEXT NOT NULL,
        pass_id TEXT NOT NULL,
        relevance_score REAL NOT NULL DEFAULT 1.0,
        selected INTEGER NOT NULL DEFAULT 1,
        citation_count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (section_id, source_id, pass_id),
        FOREIGN KEY (section_id) REFERENCES sections(section_id) ON DELETE CASCADE,
        FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE,
        FOREIGN KEY (pass_id) REFERENCES passes(pass_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS gaps (
        gap_id TEXT PRIMARY KEY,
        section_id TEXT NOT NULL,
        category TEXT NOT NULL,
        description TEXT NOT NULL,
        severity TEXT NOT NULL DEFAULT 'normal',
        status TEXT NOT NULL,
        created_pass_id TEXT,
        resolved_pass_id TEXT,
        FOREIGN KEY (section_id) REFERENCES sections(section_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS claims (
        claim_id TEXT PRIMARY KEY,
        section_id TEXT NOT NULL,
        pass_id TEXT NOT NULL,
        claim_text TEXT NOT NULL,
        claim_hash TEXT NOT NULL,
        FOREIGN KEY (section_id) REFERENCES sections(section_id) ON DELETE CASCADE,
        FOREIGN KEY (pass_id) REFERENCES passes(pass_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS claim_sources (
        claim_id TEXT NOT NULL,
        source_id TEXT NOT NULL,
        PRIMARY KEY (claim_id, source_id),
        FOREIGN KEY (claim_id) REFERENCES claims(claim_id) ON DELETE CASCADE,
        FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        type TEXT NOT NULL,
        path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS fetch_cache (
        url TEXT PRIMARY KEY,
        canonical_url TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        status_code INTEGER NOT NULL,
        content_type TEXT NOT NULL,
        extracted_text TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        fetched_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    );
    """,
    # Indexes
    "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);",
    "CREATE INDEX IF NOT EXISTS idx_sections_run_ordinal ON sections(run_id, ordinal);",
    "CREATE INDEX IF NOT EXISTS idx_sections_status ON sections(status);",
    "CREATE INDEX IF NOT EXISTS idx_passes_section_ordinal ON passes(section_id, ordinal);",
    "CREATE INDEX IF NOT EXISTS idx_sources_canonical_url ON sources(canonical_url);",
    "CREATE INDEX IF NOT EXISTS idx_gaps_section_status ON gaps(section_id, status);",
    "CREATE INDEX IF NOT EXISTS idx_section_sources_sec ON section_sources(section_id);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_fetch_cache_canonical ON fetch_cache(canonical_url);",
]


class DatabaseManager:
    """Manages SQLite database connections, schema migrations, and transactions."""

    def __init__(self, db_path: str = "research.db"):
        self.db_path = db_path
        self._ensure_parent_directory()
        self._init_database()

    def _ensure_parent_directory(self) -> None:
        parent = Path(self.db_path).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Create a configured SQLite connection with foreign keys and WAL mode."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            detect_types=sqlite3.PARSE_DECLTYPES,
            autocommit=True,
        )
        conn.row_factory = sqlite3.Row
        # Enable Foreign Keys & WAL in autocommit mode
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 10000;")
        return conn

    @contextlib.contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an atomic database transaction."""
        conn = self.get_connection()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            yield conn
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Apply migrations up to the current schema version."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            # Check if schema_version exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version';"
            )
            has_version_table = cursor.fetchone() is not None

            current_v = 0
            if has_version_table:
                cursor.execute("SELECT MAX(version) FROM schema_version;")
                row = cursor.fetchone()
                if row and row[0] is not None:
                    current_v = row[0]

            if current_v < 1:
                logger.info("Applying SQLite schema migration v1")
                for stmt in SCHEMA_V1_STATEMENTS:
                    cursor.execute(stmt)
                cursor.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?);",
                    (1, utc_now_iso()),
                )
                logger.info("Schema migration v1 applied successfully")

    def backup_database(self, destination_path: str) -> str:
        """Safely copy the SQLite database to a backup location using SQLite backup API."""
        dest = Path(destination_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self.get_connection() as src_conn:
            dest_conn = sqlite3.connect(str(dest))
            try:
                src_conn.backup(dest_conn)
            finally:
                dest_conn.close()
        return str(dest.absolute())


class PersistenceRepository:
    """Unified repository for domain entities in SQLite."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_run(self, run: Run) -> Run:
        """Insert a new run."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, topic, status, config_json, outline_version, created_at, updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    run.run_id,
                    run.topic,
                    run.status.value,
                    run.config_json,
                    run.outline_version,
                    run.created_at,
                    run.updated_at,
                    run.completed_at,
                ),
            )
        return run

    def get_run(self, run_id: str) -> Optional[Run]:
        """Fetch a run by run_id."""
        with self.db.transaction() as conn:
            cursor = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return Run.from_row(dict(row))

    def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        completed: bool = False,
    ) -> None:
        """Update run status and timestamp."""
        now = utc_now_iso()
        completed_at = now if completed else None
        with self.db.transaction() as conn:
            if completed:
                conn.execute(
                    """
                    UPDATE runs
                    SET status = ?, updated_at = ?, completed_at = ?
                    WHERE run_id = ?;
                    """,
                    (status.value, now, completed_at, run_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE runs
                    SET status = ?, updated_at = ?
                    WHERE run_id = ?;
                    """,
                    (status.value, now, run_id),
                )

    def list_runs(self, status: Optional[RunStatus] = None) -> List[Run]:
        """List runs, optionally filtered by status, newest first."""
        with self.db.transaction() as conn:
            if status:
                cursor = conn.execute(
                    "SELECT * FROM runs WHERE status = ? ORDER BY created_at DESC;",
                    (status.value,),
                )
            else:
                cursor = conn.execute("SELECT * FROM runs ORDER BY created_at DESC;")
            return [Run.from_row(dict(row)) for row in cursor.fetchall()]

    def save_sections(self, sections: List[Section]) -> None:
        """Insert or replace a batch of sections for a run."""
        with self.db.transaction() as conn:
            for s in sections:
                conn.execute(
                    """
                    INSERT INTO sections (
                        section_id, run_id, ordinal, title, objectives_json,
                        depth_target, status, attempt_count, latest_score, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(section_id) DO UPDATE SET
                        ordinal = excluded.ordinal,
                        title = excluded.title,
                        objectives_json = excluded.objectives_json,
                        depth_target = excluded.depth_target,
                        status = excluded.status,
                        attempt_count = excluded.attempt_count,
                        latest_score = excluded.latest_score,
                        updated_at = excluded.updated_at;
                    """,
                    (
                        s.section_id,
                        s.run_id,
                        s.ordinal,
                        s.title,
                        s.objectives_json,
                        s.depth_target,
                        s.status.value,
                        s.attempt_count,
                        s.latest_score,
                        s.updated_at,
                    ),
                )

    def get_sections(self, run_id: str) -> List[Section]:
        """Get all sections for a run, ordered by ordinal."""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                "SELECT * FROM sections WHERE run_id = ? ORDER BY ordinal ASC;",
                (run_id,),
            )
            return [Section.from_row(dict(row)) for row in cursor.fetchall()]

    def get_section(self, section_id: str) -> Optional[Section]:
        """Fetch a section by ID."""
        with self.db.transaction() as conn:
            cursor = conn.execute("SELECT * FROM sections WHERE section_id = ?;", (section_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return Section.from_row(dict(row))

    def update_section_status(
        self,
        section_id: str,
        status: SectionStatus,
        latest_score: Optional[float] = None,
    ) -> None:
        """Update section status and optionally latest quality score."""
        now = utc_now_iso()
        with self.db.transaction() as conn:
            if latest_score is not None:
                conn.execute(
                    """
                    UPDATE sections
                    SET status = ?, latest_score = ?, updated_at = ?
                    WHERE section_id = ?;
                    """,
                    (status.value, latest_score, now, section_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE sections
                    SET status = ?, updated_at = ?
                    WHERE section_id = ?;
                    """,
                    (status.value, now, section_id),
                )

    def compare_and_swap_section_status(
        self,
        section_id: str,
        expected_status: SectionStatus,
        new_status: SectionStatus,
    ) -> bool:
        """
        Compare-and-set status transition.
        Returns True if transition occurred, False if expected status didn't match.
        """
        now = utc_now_iso()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE sections
                SET status = ?, updated_at = ?
                WHERE section_id = ? AND status = ?;
                """,
                (new_status.value, now, section_id, expected_status.value),
            )
            return cursor.rowcount > 0

    def increment_section_attempt(self, section_id: str) -> int:
        """Increment attempt count for a section and return new count."""
        now = utc_now_iso()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE sections
                SET attempt_count = attempt_count + 1, updated_at = ?
                WHERE section_id = ?;
                """,
                (now, section_id),
            )
            cursor = conn.execute(
                "SELECT attempt_count FROM sections WHERE section_id = ?;",
                (section_id,),
            )
            row = cursor.fetchone()
            return row["attempt_count"] if row else 1

    def create_pass(self, p: Pass) -> Pass:
        """Record the start of a research pass."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO passes (
                    pass_id, section_id, ordinal, status, started_at,
                    completed_at, query_plan_json, draft_text, evaluation_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pass_id) DO UPDATE SET
                    status = excluded.status,
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at,
                    query_plan_json = excluded.query_plan_json,
                    draft_text = excluded.draft_text,
                    evaluation_json = excluded.evaluation_json;
                """,
                (
                    p.pass_id,
                    p.section_id,
                    p.ordinal,
                    p.status.value,
                    p.started_at,
                    p.completed_at,
                    p.query_plan_json,
                    p.draft_text,
                    p.evaluation_json,
                ),
            )
        return p

    def complete_pass(
        self,
        pass_id: str,
        draft_text: str,
        evaluation_json: str,
        query_plan_json: Optional[str] = None,
    ) -> None:
        """Atomically commit synthesized draft and quality evaluation for a pass."""
        now = utc_now_iso()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE passes
                SET status = ?, completed_at = ?, draft_text = ?, evaluation_json = ?,
                    query_plan_json = COALESCE(?, query_plan_json)
                WHERE pass_id = ?;
                """,
                (
                    PassStatus.COMPLETED.value,
                    now,
                    draft_text,
                    evaluation_json,
                    query_plan_json,
                    pass_id,
                ),
            )

    def mark_pass_interrupted(self, pass_id: str) -> None:
        """Mark an in-flight pass as INTERRUPTED."""
        now = utc_now_iso()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE passes
                SET status = ?, completed_at = ?
                WHERE pass_id = ? AND status = ?;
                """,
                (PassStatus.INTERRUPTED.value, now, pass_id, PassStatus.STARTED.value),
            )

    def get_passes_for_section(self, section_id: str) -> List[Pass]:
        """Get all passes for a section, ordered by ordinal."""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                "SELECT * FROM passes WHERE section_id = ? ORDER BY ordinal ASC;",
                (section_id,),
            )
            return [Pass.from_row(dict(row)) for row in cursor.fetchall()]

    def get_latest_pass(self, section_id: str) -> Optional[Pass]:
        """Get the most recent pass for a section."""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                "SELECT * FROM passes WHERE section_id = ? ORDER BY ordinal DESC LIMIT 1;",
                (section_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return Pass.from_row(dict(row))

    def upsert_source(self, source: Source) -> Source:
        """Insert or update a source record by canonical_url."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO sources (
                    source_id, canonical_url, title, domain, fetched_at,
                    content_hash, content_type, extraction_status, extracted_text, snippet, cache_pointer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_url) DO UPDATE SET
                    title = excluded.title,
                    domain = excluded.domain,
                    fetched_at = excluded.fetched_at,
                    content_hash = excluded.content_hash,
                    content_type = excluded.content_type,
                    extraction_status = excluded.extraction_status,
                    extracted_text = COALESCE(excluded.extracted_text, sources.extracted_text),
                    snippet = COALESCE(excluded.snippet, sources.snippet),
                    cache_pointer = COALESCE(excluded.cache_pointer, sources.cache_pointer);
                """,
                (
                    source.source_id,
                    source.canonical_url,
                    source.title,
                    source.domain,
                    source.fetched_at,
                    source.content_hash,
                    source.content_type,
                    source.extraction_status,
                    source.extracted_text,
                    source.snippet,
                    source.cache_pointer,
                ),
            )
            # Retrieve generated/stored source_id
            cursor = conn.execute(
                "SELECT source_id FROM sources WHERE canonical_url = ?;",
                (source.canonical_url,),
            )
            row = cursor.fetchone()
            if row:
                source.source_id = row["source_id"]
        return source

    def get_source_by_url(self, canonical_url: str) -> Optional[Source]:
        """Get source by its canonical URL."""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                "SELECT * FROM sources WHERE canonical_url = ?;",
                (canonical_url,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return Source.from_row(dict(row))

    def get_source_by_id(self, source_id: str) -> Optional[Source]:
        """Get source by source_id."""
        with self.db.transaction() as conn:
            cursor = conn.execute("SELECT * FROM sources WHERE source_id = ?;", (source_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return Source.from_row(dict(row))

    def link_section_source(
        self,
        section_id: str,
        source_id: str,
        pass_id: str,
        relevance_score: float = 1.0,
        selected: bool = True,
        citation_count: int = 0,
    ) -> None:
        """Associate a source with a specific section and pass."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO section_sources (
                    section_id, source_id, pass_id, relevance_score, selected, citation_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(section_id, source_id, pass_id) DO UPDATE SET
                    relevance_score = excluded.relevance_score,
                    selected = excluded.selected,
                    citation_count = excluded.citation_count;
                """,
                (
                    section_id,
                    source_id,
                    pass_id,
                    relevance_score,
                    1 if selected else 0,
                    citation_count,
                ),
            )

    def get_sources_for_section(self, section_id: str) -> List[Source]:
        """Get all unique sources linked to a specific section."""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT s.*
                FROM sources s
                JOIN section_sources ss ON s.source_id = ss.source_id
                WHERE ss.section_id = ?;
                """,
                (section_id,),
            )
            return [Source.from_row(dict(row)) for row in cursor.fetchall()]

    def get_all_sources_for_run(self, run_id: str) -> List[Source]:
        """Get all sources associated with any section in the given run."""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT s.*
                FROM sources s
                JOIN section_sources ss ON s.source_id = ss.source_id
                JOIN sections sec ON ss.section_id = sec.section_id
                WHERE sec.run_id = ?
                ORDER BY s.source_id ASC;
                """,
                (run_id,),
            )
            return [Source.from_row(dict(row)) for row in cursor.fetchall()]

    def update_citation_count(
        self,
        section_id: str,
        source_id: str,
        pass_id: str,
        citation_count: int,
    ) -> None:
        """Update citation count for section-source association."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE section_sources
                SET citation_count = ?
                WHERE section_id = ? AND source_id = ? AND pass_id = ?;
                """,
                (citation_count, section_id, source_id, pass_id),
            )

    def save_gaps(self, gaps: List[Gap]) -> None:
        """Batch insert gaps."""
        with self.db.transaction() as conn:
            for g in gaps:
                conn.execute(
                    """
                    INSERT INTO gaps (
                        gap_id, section_id, category, description, severity,
                        status, created_pass_id, resolved_pass_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(gap_id) DO UPDATE SET
                        category = excluded.category,
                        description = excluded.description,
                        severity = excluded.severity,
                        status = excluded.status,
                        resolved_pass_id = excluded.resolved_pass_id;
                    """,
                    (
                        g.gap_id,
                        g.section_id,
                        g.category.value if isinstance(g.category, GapCategory) else str(g.category),
                        g.description,
                        g.severity,
                        g.status.value if isinstance(g.status, GapStatus) else str(g.status),
                        g.created_pass_id,
                        g.resolved_pass_id,
                    ),
                )

    def get_open_gaps(self, section_id: str) -> List[Gap]:
        """Get all currently OPEN gaps for a section."""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                "SELECT * FROM gaps WHERE section_id = ? AND status = ?;",
                (section_id, GapStatus.OPEN.value),
            )
            return [Gap.from_row(dict(row)) for row in cursor.fetchall()]

    def resolve_gaps_for_section(self, section_id: str, pass_id: str) -> None:
        """Mark all open gaps for a section as RESOLVED by the given pass."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE gaps
                SET status = ?, resolved_pass_id = ?
                WHERE section_id = ? AND status = ?;
                """,
                (GapStatus.RESOLVED.value, pass_id, section_id, GapStatus.OPEN.value),
            )

    def supersede_gaps_for_section(self, section_id: str, pass_id: str) -> None:
        """Mark all open gaps for a section as SUPERSEDED."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE gaps
                SET status = ?, resolved_pass_id = ?
                WHERE section_id = ? AND status = ?;
                """,
                (GapStatus.SUPERSEDED.value, pass_id, section_id, GapStatus.OPEN.value),
            )

    def save_claims(self, claims: List[Claim]) -> None:
        """Save claims and link them to sources."""
        with self.db.transaction() as conn:
            for c in claims:
                conn.execute(
                    """
                    INSERT INTO claims (claim_id, section_id, pass_id, claim_text, claim_hash)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(claim_id) DO UPDATE SET
                        claim_text = excluded.claim_text,
                        claim_hash = excluded.claim_hash;
                    """,
                    (c.claim_id, c.section_id, c.pass_id, c.claim_text, c.claim_hash),
                )
                for src_id in c.source_ids:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO claim_sources (claim_id, source_id)
                        VALUES (?, ?);
                        """,
                        (c.claim_id, src_id),
                    )

    def record_artifact(self, artifact: Artifact) -> Artifact:
        """Record a generated export artifact."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (artifact_id, run_id, type, path, sha256, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    artifact.artifact_id,
                    artifact.run_id,
                    artifact.type,
                    artifact.path,
                    artifact.sha256,
                    artifact.created_at,
                ),
            )
        return artifact

    def get_artifacts(self, run_id: str) -> List[Artifact]:
        """Fetch all artifacts created for a run."""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at ASC;",
                (run_id,),
            )
            return [Artifact.from_row(dict(row)) for row in cursor.fetchall()]

    get_artifacts_for_run = get_artifacts

    def get_cached_page(self, canonical_url: str) -> Optional[Dict[str, Any]]:
        """Get cached extracted page if not expired."""
        now = utc_now_iso()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM fetch_cache
                WHERE canonical_url = ? AND expires_at > ?;
                """,
                (canonical_url, now),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)

    def set_cached_page(
        self,
        url: str,
        canonical_url: str,
        content_hash: str,
        status_code: int,
        content_type: str,
        extracted_text: str,
        metadata: Dict[str, Any],
        ttl_hours: int = 72,
    ) -> None:
        """Store extracted page in cache with expiration."""
        now_dt = datetime.now(timezone.utc)
        expires_dt = now_dt + timedelta(hours=ttl_hours)
        fetched_at = now_dt.isoformat()
        expires_at = expires_dt.isoformat()

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO fetch_cache (
                    url, canonical_url, content_hash, status_code, content_type,
                    extracted_text, metadata_json, fetched_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    canonical_url = excluded.canonical_url,
                    content_hash = excluded.content_hash,
                    status_code = excluded.status_code,
                    content_type = excluded.content_type,
                    extracted_text = excluded.extracted_text,
                    metadata_json = excluded.metadata_json,
                    fetched_at = excluded.fetched_at,
                    expires_at = excluded.expires_at;
                """,
                (
                    url,
                    canonical_url,
                    content_hash,
                    status_code,
                    content_type,
                    extracted_text,
                    json.dumps(metadata),
                    fetched_at,
                    expires_at,
                ),
            )
