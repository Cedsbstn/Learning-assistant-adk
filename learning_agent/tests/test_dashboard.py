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
Unit tests for the Live Terminal Dashboard subsystem.
"""

import io
import pytest
from rich.console import Console

from dashboard import LiveTerminalDashboard


def test_dashboard_initialization():
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=120)

    dash = LiveTerminalDashboard(
        run_id="run_test_123",
        topic="Distributed Consensus and Paxos",
        preset="standard",
        enabled=True,
        console=console,
    )

    assert dash.run_id == "run_test_123"
    assert dash.topic == "Distributed Consensus and Paxos"
    assert dash.status == "INITIALIZING"
    assert len(dash.sections) == 0


def test_dashboard_section_lifecycle_and_events():
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=120)

    dash = LiveTerminalDashboard(
        run_id="run_test_456",
        topic="Raft Protocol",
        preset="deep",
        enabled=True,
        console=console,
    )

    dash.init_sections([
        {
            "id": "sec_01",
            "order_index": 1,
            "title": "Leader Election",
            "target_depth": "intermediate",
            "max_passes": 3,
        },
        {
            "id": "sec_02",
            "order_index": 2,
            "title": "Log Replication",
            "target_depth": "advanced",
            "max_passes": 4,
        },
    ])

    assert len(dash.sections) == 2
    assert dash.sections["sec_01"]["status"] == "QUEUED"

    # Start section
    dash.on_progress_event("section_start", {
        "section_id": "sec_01",
        "title": "Leader Election",
        "pass_num": 1,
    })
    assert dash.sections["sec_01"]["status"] == "IN_PROGRESS"
    assert dash.sections["sec_01"]["pass_num"] == 1

    # Search and chunking
    dash.on_progress_event("search_executed", {
        "query": "Raft leader election split votes",
        "sources_found": 4,
    })
    dash.on_progress_event("evidence_chunked", {
        "chunks_count": 12,
        "cached_count": 2,
    })

    # Draft generated
    dash.on_progress_event("draft_generated", {
        "section_id": "sec_01",
        "word_count": 420,
    })
    assert dash.sections["sec_01"]["words"] == 420

    # Gate evaluated (failed -> gap retry)
    dash.on_progress_event("gate_evaluated", {
        "section_id": "sec_01",
        "passed": False,
        "quality_score": 55.0,
        "word_count": 420,
        "source_count": 2,
    })
    dash.on_progress_event("gap_retry", {
        "section_id": "sec_01",
        "gap_count": 2,
        "pass_num": 2,
    })
    assert dash.sections["sec_01"]["status"] == "RETRYING"
    assert dash.sections["sec_01"]["pass_num"] == 2

    # Pass 2 -> Section Complete
    dash.on_progress_event("gate_evaluated", {
        "section_id": "sec_01",
        "passed": True,
        "quality_score": 92.0,
        "word_count": 650,
        "source_count": 4,
    })
    dash.on_progress_event("section_complete", {
        "section_id": "sec_01",
        "passed": True,
    })
    assert dash.sections["sec_01"]["status"] == "COMPLETED"
    assert dash.sections["sec_01"]["score"] == 92.0

    # Render dashboard
    panel = dash._render_dashboard()
    console.print(panel)
    rendered_text = output.getvalue()

    assert "KYTHE AGENTIC DEEP RESEARCH ENGINE" in rendered_text
    assert "Leader Election" in rendered_text
    assert "Log Replication" in rendered_text
    assert "run_test_456" in rendered_text
    assert "COMPLETED" in rendered_text


def test_dashboard_start_stop_lifecycle():
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=120)

    dash = LiveTerminalDashboard(
        run_id="run_lifecycle",
        topic="Zero Knowledge Proofs",
        preset="quick",
        enabled=True,
        console=console,
    )

    dash.start()
    assert dash.status == "RUNNING"
    assert dash._live is not None

    dash.log_activity("Executing test query...")
    dash.stop()
    assert dash._live is None


def test_dashboard_disabled_fallback():
    dash = LiveTerminalDashboard(
        run_id="run_disabled",
        topic="Fallback Test",
        preset="quick",
        enabled=False,
    )

    assert not dash.enabled
    dash.start()
    assert dash._live is None
    dash.stop()
