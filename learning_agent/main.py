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
Main entry point and CLI command router for Kythe Agentic Deep Research Agent.

Supports subcommands:
- research: Start a new agentic research run with outline review
- resume: Resume an interrupted or paused run by run_id
- status: Inspect run progress, section scores, and open gaps
- list-runs: List all research runs and their statuses
- export: Re-generate or render specific learning export formats
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent import ACTIVE_CONFIG, ACTIVE_CONFIG_NAME, core_agent
from cli import cli_progress_handler, interactive_outline_review
from config import DEFAULT_CONFIG, ResearchConfig, get_config_by_name, list_preset_names, print_config
from models import RunStatus
from orchestrator import RunOrchestrator
from tools import create_quick_reference, print_research_summary, save_markdown_curriculum, save_research_metadata

# Ensure current directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configure logging
log_file_path = ACTIVE_CONFIG.log_file
log_dir = os.path.dirname(log_file_path)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Global active run tracker for clean signal handling
_ACTIVE_ORCHESTRATOR: Optional[RunOrchestrator] = None
_ACTIVE_RUN_ID: Optional[str] = None


def handle_interrupt(signum, frame):
    """Graceful interrupt handler to checkpoint run as PAUSED on SIGINT."""
    global _ACTIVE_ORCHESTRATOR, _ACTIVE_RUN_ID
    print("\n\n[WARN] Research process interrupted by user.")
    if _ACTIVE_ORCHESTRATOR and _ACTIVE_RUN_ID:
        try:
            _ACTIVE_ORCHESTRATOR.pause_run(_ACTIVE_RUN_ID)
            print(f"[CHECKPOINT] Saved run '{_ACTIVE_RUN_ID}' to SQLite database.")
            print(f"[INFO] To resume this run later, execute:")
            print(f"   python main.py resume {_ACTIVE_RUN_ID}\n")
        except Exception as e:
            logger.error(f"Error checkpointing paused run: {e}")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_interrupt)


def cmd_research(args: argparse.Namespace) -> None:
    """Handle 'research' subcommand."""
    global _ACTIVE_ORCHESTRATOR, _ACTIVE_RUN_ID

    topic = args.topic
    if not topic:
        print("\nEnter research topic (or 'quit' to exit):")
        topic = input("Topic: ").strip()
        if not topic or topic.lower() in ("quit", "exit", "q"):
            print("[INFO] Exited.")
            return

    # Determine configuration preset
    config = _clone_or_load_config(args.preset)
    if hasattr(args, "auto_approve") and args.auto_approve:
        config.require_outline_approval = False
    if hasattr(args, "formats") and args.formats:
        config.export_formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    print("\n" + "=" * 80)
    print("KYTHE AGENTIC DEEP RESEARCH")
    print("=" * 80)
    print(f"Topic: '{topic}'")
    print(f"Active Preset: {args.preset or ACTIVE_CONFIG_NAME}")
    print(f"Database: {config.database_path}")
    print("=" * 80 + "\n")

    orch = RunOrchestrator(db_path=config.database_path)
    _ACTIVE_ORCHESTRATOR = orch

    run_id = orch.create_run(topic=topic, config=config)
    _ACTIVE_RUN_ID = run_id
    print(f"[RUN] Initialized Run ID: {run_id}")

    print("[PLAN] Generating curriculum outline and modular work items...")
    outline = orch.plan_curriculum(run_id)

    if config.require_outline_approval:
        approved = interactive_outline_review(orch, run_id, topic, config)
        if not approved:
            return
    else:
        print("[INFO] Auto-approving generated outline per configuration.")
        orch.approve_outline(run_id)

    print("\n[LOOP] Starting agentic iterative research control loop...")
    use_dashboard = getattr(args, "dashboard", True) and sys.stdout.isatty()
    progress_handler, dash = _create_dashboard_handler(
        orch=orch,
        run_id=run_id,
        topic=topic,
        preset=args.preset or ACTIVE_CONFIG_NAME,
        use_dashboard=use_dashboard,
    )

    if dash:
        dash.start()

    try:
        final_status = orch.run_until_terminal(run_id, on_progress=progress_handler)
    finally:
        if dash:
            dash.stop()

    _print_completion_summary(orch, run_id)


def _create_dashboard_handler(
    orch: RunOrchestrator,
    run_id: str,
    topic: str,
    preset: str,
    use_dashboard: bool,
):
    """Create a dashboard progress handler or fallback to CLI handler."""
    if not use_dashboard:
        return cli_progress_handler, None

    try:
        from dashboard import LiveTerminalDashboard
        sections = orch.repo.get_sections(run_id)
        run = orch.repo.get_run(run_id)
        max_passes = 3
        if run:
            try:
                cfg_data = json.loads(run.config_json)
                max_passes = cfg_data.get("max_iterations", 3)
            except Exception:
                pass

        dash = LiveTerminalDashboard(
            run_id=run_id,
            topic=topic,
            preset=preset,
            enabled=True,
        )
        dash.init_sections([
            {
                "id": s.section_id,
                "order_index": s.ordinal,
                "title": s.title,
                "target_depth": s.depth_target,
                "max_passes": max_passes,
            }
            for s in sections
        ])

        def handler(event: Dict[str, Any]) -> None:
            ev_type = event.get("event")
            if ev_type == "section_pass_start":
                dash.on_progress_event("section_start", {
                    "section_id": event.get("section_id"),
                    "title": event.get("section_title"),
                    "pass_num": event.get("attempt", 1),
                })
            elif ev_type == "section_pass_end":
                dash.on_progress_event("gate_evaluated", {
                    "section_id": event.get("section_id"),
                    "passed": event.get("passed", False),
                    "quality_score": event.get("score", 0.0),
                    "word_count": event.get("word_count", 0),
                    "source_count": event.get("source_count", 0),
                })
                if not event.get("passed", False):
                    dash.on_progress_event("gap_retry", {
                        "section_id": event.get("section_id"),
                        "gap_count": event.get("open_gaps", 0),
                        "pass_num": event.get("attempt", 1) + 1,
                    })
                else:
                    dash.on_progress_event("section_complete", {
                        "section_id": event.get("section_id"),
                        "passed": True,
                    })
            elif ev_type == "synthesis_start":
                dash.log_activity("Compiling master research dossier...")
            elif ev_type == "run_completed":
                dash.on_progress_event("run_complete", {})

        return handler, dash
    except Exception as e:
        logger.warning("Could not initialize live dashboard: %s. Using standard CLI.", e)
        return cli_progress_handler, None


def cmd_resume(args: argparse.Namespace) -> None:
    """Handle 'resume' subcommand."""
    global _ACTIVE_ORCHESTRATOR, _ACTIVE_RUN_ID

    run_id = args.run_id
    db_path = getattr(args, "db", DEFAULT_CONFIG.database_path)
    orch = RunOrchestrator(db_path=db_path)
    _ACTIVE_ORCHESTRATOR = orch
    _ACTIVE_RUN_ID = run_id

    run = orch.repo.get_run(run_id)
    if not run:
        print(f"[ERROR] Run '{run_id}' not found in database ({db_path}).")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("KYTHE RESUMING RESEARCH RUN")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Topic: '{run.topic}'")
    print(f"Previous Status: {run.status.value}")
    print("=" * 80 + "\n")

    if run.status == RunStatus.OUTLINE_REVIEW:
        config = ResearchConfig.from_dict(json.loads(run.config_json))
        approved = interactive_outline_review(orch, run_id, run.topic, config)
        if not approved:
            return

    use_dashboard = getattr(args, "dashboard", True) and sys.stdout.isatty()
    progress_handler, dash = _create_dashboard_handler(
        orch=orch,
        run_id=run_id,
        topic=run.topic,
        preset="standard",
        use_dashboard=use_dashboard,
    )

    if dash:
        dash.start()

    try:
        final_status = orch.resume_run(run_id, on_progress=progress_handler)
    finally:
        if dash:
            dash.stop()

    _print_completion_summary(orch, run_id)


def cmd_status(args: argparse.Namespace) -> None:
    """Handle 'status' subcommand."""
    run_id = args.run_id
    db_path = getattr(args, "db", DEFAULT_CONFIG.database_path)
    orch = RunOrchestrator(db_path=db_path)

    try:
        info = orch.get_status(run_id)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("KYTHE RESEARCH RUN STATUS")
    print("=" * 80)
    print(f"Run ID: {info['run_id']}")
    print(f"Topic: {info['topic']}")
    print(f"Status: {info['status']}")
    print(f"Created: {info['created_at']}")
    print(f"Updated: {info['updated_at']}")
    if info.get("completed_at"):
        print(f"Completed: {info['completed_at']}")

    print(f"\nModules Summary:")
    print(f"  Total: {info['total_sections']} | Completed: {info['complete_sections']} | Exhausted: {info['exhausted_sections']} | Active: {info['active_sections']}")
    print(f"  Total Sources Referenced: {info['total_sources']}")

    print("\nSection Details:")
    for s in info["sections"]:
        st = s["status"]
        status_tag = "[PASS]" if st == "COMPLETE" else "[EXHAUST]" if st == "EXHAUSTED" else "[IN_PROG]"
        score_str = f"{s['score']:.1f}%" if s["score"] is not None else "N/A"
        print(f"  {status_tag} [{s['ordinal']}] {s['title'][:40]} | Depth: {s['depth']} | Status: {st} | Passes: {s['attempt_count']} | Score: {score_str} | Open Gaps: {s['open_gaps']}")

    if info.get("artifacts"):
        print("\nGenerated Artifacts:")
        for a in info["artifacts"]:
            print(f"  [{a['type'].upper()}] {a['path']}")

    print("=" * 80 + "\n")


def cmd_list_runs(args: argparse.Namespace) -> None:
    """Handle 'list-runs' subcommand."""
    db_path = getattr(args, "db", DEFAULT_CONFIG.database_path)
    orch = RunOrchestrator(db_path=db_path)

    filter_status = None
    if hasattr(args, "status") and args.status:
        try:
            filter_status = RunStatus(args.status.upper())
        except ValueError:
            print(f"[WARN] Unknown status '{args.status}'. Listing all runs.")

    runs = orch.repo.list_runs(status=filter_status)

    print("\n" + "=" * 80)
    print(f"KYTHE RESEARCH RUNS ({len(runs)} found)")
    print("=" * 80)
    if not runs:
        print("No research runs found in database.")
    else:
        for r in runs:
            sections = orch.repo.get_sections(r.run_id)
            comp = sum(1 for s in sections if s.status == "COMPLETE")
            print(f"- ID: {r.run_id} | Status: {r.status.value:<12} | Modules: {comp}/{len(sections)} | Topic: '{r.topic}'")
    print("=" * 80 + "\n")


def cmd_export(args: argparse.Namespace) -> None:
    """Handle 'export' subcommand."""
    run_id = args.run_id
    db_path = getattr(args, "db", DEFAULT_CONFIG.database_path)
    orch = RunOrchestrator(db_path=db_path)

    formats = [f.strip() for f in args.format.split(",") if f.strip()] if args.format else None
    print(f"\n[EXPORT] Exporting run '{run_id}' (formats: {formats or 'default'})...")

    try:
        artifacts = orch.export_run(run_id, formats=formats)
        print("\n[OK] Export artifacts generated:")
        for a in artifacts:
            print(f"  [{a.type.upper()}] {a.path} (SHA256: {a.sha256[:12]}...)")
        print()
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        logger.error(f"Export error for run {run_id}: {e}", exc_info=True)


def _clone_or_load_config(preset_name: Optional[str] = None) -> ResearchConfig:
    """Load configuration from preset or defaults."""
    if preset_name:
        return ResearchConfig.from_dict(get_config_by_name(preset_name).to_dict())
    return ResearchConfig.from_dict(ACTIVE_CONFIG.to_dict())


def _print_completion_summary(orch: RunOrchestrator, run_id: str) -> None:
    """Display final summary and artifact links after run completes."""
    status = orch.get_status(run_id)
    print(f"\nResearch Run '{run_id}' complete.")
    print(f"Completed Modules: {status['complete_sections']}/{status['total_sections']}")

    if status.get("artifacts"):
        print("\nExported Artifacts:")
        for a in status["artifacts"]:
            print(f"  - [{a['type'].upper()}]: {a['path']}")


def build_cli_parser() -> argparse.ArgumentParser:
    """Construct argument parser with subcommands and default aliases."""
    parser = argparse.ArgumentParser(
        prog="kythe",
        description="Kythe: Agentic Section-by-Section Deep Research Agent",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    p_research = subparsers.add_parser("research", help="Start a new deep research run")
    p_research.add_argument("topic", nargs="?", help="Research topic or question")
    p_research.add_argument("--preset", choices=list_preset_names(), default=None, help="Configuration preset")
    p_research.add_argument("--auto-approve", action="store_true", help="Auto-approve outline without interactive prompt")
    p_research.add_argument("--formats", default="markdown,html,pdf,quiz,flashcards,metadata", help="Comma-separated export formats")
    p_research.add_argument("--dashboard", action=argparse.BooleanOptionalAction, default=True, help="Enable live interactive terminal dashboard")
    p_research.set_defaults(func=cmd_research)

    p_resume = subparsers.add_parser("resume", help="Resume an interrupted research run")
    p_resume.add_argument("run_id", help="Run ID to resume")
    p_resume.add_argument("--db", default=DEFAULT_CONFIG.database_path, help="SQLite database path")
    p_resume.add_argument("--dashboard", action=argparse.BooleanOptionalAction, default=True, help="Enable live interactive terminal dashboard")
    p_resume.set_defaults(func=cmd_resume)

    p_status = subparsers.add_parser("status", help="Inspect status and progress of a run")
    p_status.add_argument("run_id", help="Run ID to inspect")
    p_status.add_argument("--db", default=DEFAULT_CONFIG.database_path, help="SQLite database path")
    p_status.set_defaults(func=cmd_status)

    p_list = subparsers.add_parser("list-runs", help="List recent research runs")
    p_list.add_argument("--status", choices=[s.value for s in RunStatus] + [s.value.lower() for s in RunStatus], default=None, help="Filter by run status")
    p_list.add_argument("--limit", type=int, default=10, help="Maximum runs to display")
    p_list.add_argument("--db", default=DEFAULT_CONFIG.database_path, help="SQLite database path")
    p_list.set_defaults(func=cmd_list_runs)

    p_export = subparsers.add_parser("export", help="Re-export artifacts for a completed run")
    p_export.add_argument("run_id", help="Run ID to export")
    p_export.add_argument("--output-dir", default="output", help="Directory for exported artifacts")
    p_export.add_argument("--formats", "--format", dest="format", default="markdown,html,pdf,quiz,flashcards,metadata", help="Comma-separated export formats")
    p_export.add_argument("--db", default=DEFAULT_CONFIG.database_path, help="SQLite database path")
    p_export.set_defaults(func=cmd_export)
    return parser


build_parser = build_cli_parser


def main() -> None:
    """Main CLI entry point with backwards compatibility support."""
    parser = build_cli_parser()

    known_commands = {"research", "resume", "status", "list-runs", "export", "-h", "--help"}
    raw_args = sys.argv[1:]

    if raw_args and raw_args[0] not in known_commands:
        topic_arg = " ".join(raw_args)
        args = parser.parse_args(["research", topic_arg])
    elif not raw_args:
        args = parser.parse_args(["research", ""])
    else:
        args = parser.parse_args(raw_args)

    if args.command == "research":
        cmd_research(args)
    elif args.command == "resume":
        cmd_resume(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "list-runs":
        cmd_list_runs(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
