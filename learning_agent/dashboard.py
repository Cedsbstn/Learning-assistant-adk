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
Live Terminal Dashboard for Kythe Agentic Deep Research Agent.

Renders real-time multi-panel research state, section progress, quality gate metrics,
and live activity streams using the Rich library.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class LiveTerminalDashboard:
    """
    Real-time interactive terminal dashboard for research runs.
    """

    def __init__(
        self,
        run_id: str,
        topic: str,
        preset: str,
        enabled: bool = True,
        console: Optional[Console] = None,
    ) -> None:
        self.run_id = run_id
        self.topic = topic
        self.preset = preset
        self.enabled = enabled and (console is not None or sys.stdout.isatty())
        self.console = console or Console()
        self.start_time = time.time()
        self.status = "INITIALIZING"
        self.sections: Dict[str, Dict[str, Any]] = {}
        self.activity_log: List[str] = []
        self.max_log_entries = 6
        self._live: Optional[Live] = None

    def start(self) -> None:
        """Start the live rendering context."""
        if not self.enabled:
            return
        self.status = "RUNNING"
        self._live = Live(
            self._render_dashboard(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live rendering context and render the final state."""
        if not self.enabled or not self._live:
            return
        self._live.update(self._render_dashboard())
        self._live.stop()
        self._live = None

    def log_activity(self, message: str) -> None:
        """Add a timestamped event entry to the scrolling activity log."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.activity_log.append(entry)
        if len(self.activity_log) > self.max_log_entries:
            self.activity_log.pop(0)
        self._refresh()

    def init_sections(self, sections: List[Dict[str, Any]]) -> None:
        """Initialize the sections table from planned curriculum outline."""
        for sec in sections:
            sec_id = sec.get("id") or sec.get("section_id") or str(sec.get("order_index", 0))
            self.sections[sec_id] = {
                "order_index": sec.get("order_index", 0),
                "title": sec.get("title", "Untitled Section"),
                "depth": sec.get("target_depth", "standard"),
                "status": "QUEUED",
                "pass_num": 0,
                "max_passes": sec.get("max_passes", 3),
                "words": 0,
                "sources": 0,
                "score": 0.0,
                "gate_status": "PENDING",
            }
        self._refresh()

    def on_progress_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle structured progress events emitted by the orchestrator."""
        sec_id = data.get("section_id")

        if event_type == "run_start":
            self.status = "RESEARCHING"
            self.log_activity(f"Run {self.run_id} started for topic: '{self.topic}'")

        elif event_type == "section_start":
            if sec_id and sec_id in self.sections:
                self.sections[sec_id]["status"] = "IN_PROGRESS"
                self.sections[sec_id]["pass_num"] = data.get("pass_num", 1)
            title = data.get("title", sec_id)
            self.log_activity(f"Starting research on section: '{title}' (Pass {data.get('pass_num', 1)})")

        elif event_type == "search_executed":
            query = data.get("query", "")
            found = data.get("sources_found", 0)
            self.log_activity(f"Search executed: '{query}' -> {found} sources acquired")

        elif event_type == "evidence_chunked":
            chunks = data.get("chunks_count", 0)
            cached = data.get("cached_count", 0)
            self.log_activity(f"Evidence processed: {chunks} chunks ({cached} from cache)")

        elif event_type == "draft_generated":
            words = data.get("word_count", 0)
            if sec_id and sec_id in self.sections:
                self.sections[sec_id]["words"] = words
            self.log_activity(f"Draft generated: {words} words for section {sec_id}")

        elif event_type == "gate_evaluated":
            passed = data.get("passed", False)
            score = data.get("quality_score", 0.0)
            words = data.get("word_count", 0)
            sources = data.get("source_count", 0)
            if sec_id and sec_id in self.sections:
                self.sections[sec_id]["score"] = score
                self.sections[sec_id]["words"] = words
                self.sections[sec_id]["sources"] = sources
                self.sections[sec_id]["gate_status"] = "PASSED" if passed else "FAILED"
            status_text = "PASSED" if passed else "FAILED"
            self.log_activity(f"Gate evaluation: {status_text} (Score: {score:.2f})")

        elif event_type == "gap_retry":
            gap_count = data.get("gap_count", 0)
            pass_num = data.get("pass_num", 1)
            if sec_id and sec_id in self.sections:
                self.sections[sec_id]["pass_num"] = pass_num
                self.sections[sec_id]["status"] = "RETRYING"
            self.log_activity(f"Gap identified: {gap_count} gaps found. Launching retry pass {pass_num}")

        elif event_type == "section_complete":
            passed = data.get("passed", True)
            if sec_id and sec_id in self.sections:
                self.sections[sec_id]["status"] = "COMPLETED" if passed else "EXHAUSTED"
                self.sections[sec_id]["gate_status"] = "PASSED" if passed else "FAILED"
            status_label = "completed" if passed else "exhausted"
            self.log_activity(f"Section {sec_id} {status_label}")

        elif event_type == "run_complete":
            self.status = "COMPLETED"
            self.log_activity("All curriculum sections finished. Generating master exports...")

        elif event_type == "run_interrupted":
            self.status = "PAUSED"
            self.log_activity("Run interrupted. Checkpointed state safely in SQLite.")

        self._refresh()

    def _refresh(self) -> None:
        """Update the live console view."""
        if self.enabled and self._live:
            self._live.update(self._render_dashboard())

    def _render_dashboard(self) -> Panel:
        """Assemble the complete multi-panel dashboard display."""
        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins:02d}:{secs:02d}"

        # Header info
        header_text = Text()
        header_text.append("KYTHE AGENTIC DEEP RESEARCH ENGINE\n", style="bold cyan")
        header_text.append(f"Run ID: ", style="bold white")
        header_text.append(f"{self.run_id}  ", style="yellow")
        header_text.append(f"Preset: ", style="bold white")
        header_text.append(f"{self.preset}  ", style="green")
        header_text.append(f"Elapsed: ", style="bold white")
        header_text.append(f"{time_str}  ", style="magenta")
        header_text.append(f"Status: ", style="bold white")

        status_style = "bold green" if self.status == "COMPLETED" else "bold yellow"
        if self.status == "PAUSED":
            status_style = "bold red"
        header_text.append(f"{self.status}\n", style=status_style)
        header_text.append(f"Topic: '{self.topic}'", style="italic white")

        header_panel = Panel(header_text, border_style="cyan", padding=(0, 1))

        # Section progress table
        table = Table(
            title="Curriculum Section Pipeline",
            expand=True,
            header_style="bold cyan",
            border_style="dim",
            row_styles=["none", "dim"],
        )
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Section Title", style="white", ratio=3)
        table.add_column("Depth", justify="center", style="cyan", width=12)
        table.add_column("Status", justify="center", width=12)
        table.add_column("Pass", justify="center", width=6)
        table.add_column("Words", justify="right", width=7)
        table.add_column("Sources", justify="right", width=7)
        table.add_column("Score", justify="right", width=6)

        sorted_sections = sorted(self.sections.values(), key=lambda s: s.get("order_index", 0))

        for sec in sorted_sections:
            sec_status = sec.get("status", "QUEUED")
            if sec_status == "COMPLETED":
                status_styled = Text("COMPLETED", style="bold green")
            elif sec_status in ("IN_PROGRESS", "RESEARCHING"):
                status_styled = Text("RESEARCHING", style="bold yellow")
            elif sec_status == "RETRYING":
                status_styled = Text("RETRYING", style="bold magenta")
            elif sec_status == "EXHAUSTED":
                status_styled = Text("EXHAUSTED", style="bold red")
            else:
                status_styled = Text("QUEUED", style="dim")

            score_val = sec.get("score", 0.0)
            score_str = f"{score_val:.2f}" if score_val > 0 else "-"

            table.add_row(
                str(sec.get("order_index", 1)),
                sec.get("title", "Untitled"),
                sec.get("depth", "standard"),
                status_styled,
                f"{sec.get('pass_num', 0)}/{sec.get('max_passes', 3)}",
                str(sec.get("words", 0)),
                str(sec.get("sources", 0)),
                score_str,
            )

        if not sorted_sections:
            table.add_row("-", "Planning curriculum structure...", "-", "PLANNING", "-", "-", "-", "-")

        # Activity log panel
        log_text = Text()
        if self.activity_log:
            for line in self.activity_log:
                log_text.append(f"{line}\n", style="dim white")
        else:
            log_text.append("Initializing orchestrator control plane...\n", style="dim italic")

        log_panel = Panel(
            log_text,
            title="Live Activity Stream",
            border_style="blue",
            padding=(0, 1),
        )

        return Panel(
            Group(header_panel, table, log_panel),
            border_style="cyan",
            title="[bold white]Kythe Control Center[/bold white]",
        )
