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
Interactive CLI & Outline Review UI for Kythe Autonomous Deep Research Agent.

Provides argument parsing, interactive outline customization, progress reporting,
and subcommand handlers (research, resume, status, list-runs, export).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from config import ResearchConfig, get_config_by_name, print_config
from models import CurriculumOutline, OutlineSection, RunStatus, Section, SectionStatus
from orchestrator import RunOrchestrator

logger = logging.getLogger(__name__)


def format_depth_label(depth: str) -> str:
    """Format depth target without emojis."""
    d = (depth or "intermediate").lower()
    if d == "basic":
        return "Basic"
    elif d == "advanced":
        return "Advanced"
    return "Intermediate"


def print_outline_view(topic: str, sections: List[Dict[str, Any]], config: ResearchConfig) -> None:
    """Render interactive outline review screen."""
    print("\n" + "=" * 80)
    print("KYTHE CURRICULUM OUTLINE REVIEW")
    print("=" * 80)
    print(f"Topic: {topic}")
    print(f"Proposed Modules: {len(sections)}")
    print(f"Max Passes per Section: {config.max_iterations} | Min Words: {config.min_word_count}")
    print("-" * 80)

    for idx, s in enumerate(sections, 1):
        depth_label = format_depth_label(s.get("depth_target", "intermediate"))
        print(f"\n[{idx}] {s.get('title', 'Untitled')} ({depth_label})")
        objectives = s.get("objectives", [])
        if isinstance(objectives, str):
            try:
                objectives = json.loads(objectives)
            except Exception:
                objectives = [objectives]
        for obj in objectives:
            print(f"    - {obj}")

    print("\n" + "=" * 80)


def interactive_outline_review(
    orchestrator: RunOrchestrator,
    run_id: str,
    topic: str,
    config: ResearchConfig,
) -> bool:
    """
    Interactive terminal menu for reviewing and editing the curriculum outline.
    Returns True if approved, False if cancelled.
    """
    sections = orchestrator.repo.get_sections(run_id)
    section_data: List[Dict[str, Any]] = [
        {
            "section_id": s.section_id,
            "ordinal": s.ordinal,
            "title": s.title,
            "objectives": s.objectives,
            "depth_target": s.depth_target,
        }
        for s in sections
    ]

    while True:
        print_outline_view(topic, section_data, config)
        print("\nReview Actions:")
        print("  [1] Approve outline and start research")
        print("  [2] Edit section title / objectives")
        print("  [3] Add a new section")
        print("  [4] Remove a section")
        print("  [5] Reorder sections")
        print("  [6] Change section depth target (basic, intermediate, advanced)")
        print("  [7] Regenerate entire outline")
        print("  [q] Cancel and Exit")

        choice = input("\nChoose an action [1-7 or q]: ").strip().lower()

        if choice in ("1", "a", "approve", "y", "yes"):
            orchestrator.approve_outline(run_id, edits=section_data)
            print("\n[OK] Outline approved. Starting autonomous research.\n")
            return True

        elif choice in ("q", "quit", "exit"):
            print("\n[INFO] Outline review cancelled.")
            return False

        elif choice == "2":
            # Edit section
            sec_num_str = input(f"Enter section number to edit (1-{len(section_data)}): ").strip()
            if not sec_num_str.isdigit() or not (1 <= int(sec_num_str) <= len(section_data)):
                print("[ERROR] Invalid section number.")
                continue
            sec_idx = int(sec_num_str) - 1
            sec = section_data[sec_idx]

            new_title = input(f"New title (leave blank to keep '{sec['title']}'): ").strip()
            if new_title:
                sec["title"] = new_title

            print(f"Current objectives: {', '.join(sec['objectives'])}")
            new_objs = input("Enter new objectives separated by commas (or blank to keep): ").strip()
            if new_objs:
                sec["objectives"] = [o.strip() for o in new_objs.split(",") if o.strip()]

        elif choice == "3":
            # Add section
            title = input("New section title: ").strip()
            if not title:
                print("[ERROR] Title cannot be empty.")
                continue
            objs_input = input("Learning objectives separated by commas: ").strip()
            objs = [o.strip() for o in objs_input.split(",") if o.strip()] or ["Explore core concepts"]
            depth = input("Depth target (basic, intermediate, advanced) [intermediate]: ").strip().lower() or "intermediate"
            if depth not in ("basic", "intermediate", "advanced"):
                depth = "intermediate"

            new_sec = {
                "section_id": f"sec_{run_id}_{len(section_data) + 1}",
                "ordinal": len(section_data) + 1,
                "title": title,
                "objectives": objs,
                "depth_target": depth,
            }
            section_data.append(new_sec)
            print(f"[OK] Added section: {title}")

        elif choice == "4":
            # Remove section
            sec_num_str = input(f"Enter section number to remove (1-{len(section_data)}): ").strip()
            if not sec_num_str.isdigit() or not (1 <= int(sec_num_str) <= len(section_data)):
                print("[ERROR] Invalid section number.")
                continue
            if len(section_data) <= 1:
                print("[ERROR] Cannot remove the only remaining section.")
                continue
            removed = section_data.pop(int(sec_num_str) - 1)
            # Re-index ordinals
            for i, s in enumerate(section_data, 1):
                s["ordinal"] = i
            print(f"[OK] Removed section: {removed['title']}")

        elif choice == "5":
            # Reorder sections
            order_str = input(f"Enter new order of section numbers separated by commas (e.g. 2,1,3): ").strip()
            try:
                indexes = [int(x.strip()) - 1 for x in order_str.split(",") if x.strip()]
                if len(indexes) == len(section_data) and set(indexes) == set(range(len(section_data))):
                    section_data = [section_data[i] for i in indexes]
                    for i, s in enumerate(section_data, 1):
                        s["ordinal"] = i
                    print("[OK] Sections reordered successfully.")
                else:
                    print("[ERROR] Invalid permutation of section numbers.")
            except Exception:
                print("[ERROR] Failed to parse order.")

        elif choice == "6":
            # Change depth target
            sec_num_str = input(f"Enter section number (1-{len(section_data)}): ").strip()
            if not sec_num_str.isdigit() or not (1 <= int(sec_num_str) <= len(section_data)):
                print("[ERROR] Invalid section number.")
                continue
            sec_idx = int(sec_num_str) - 1
            new_depth = input("Choose depth (1: Basic, 2: Intermediate, 3: Advanced): ").strip()
            if new_depth in ("1", "basic"):
                section_data[sec_idx]["depth_target"] = "basic"
            elif new_depth in ("2", "intermediate"):
                section_data[sec_idx]["depth_target"] = "intermediate"
            elif new_depth in ("3", "advanced"):
                section_data[sec_idx]["depth_target"] = "advanced"

        elif choice == "7":
            # Regenerate outline
            print("\n[INFO] Regenerating curriculum outline from planner...")
            outline = orchestrator.plan_curriculum(run_id)
            sections = orchestrator.repo.get_sections(run_id)
            section_data = [
                {
                    "section_id": s.section_id,
                    "ordinal": s.ordinal,
                    "title": s.title,
                    "objectives": s.objectives,
                    "depth_target": s.depth_target,
                }
                for s in sections
            ]


def cli_progress_handler(event: Dict[str, Any]) -> None:
    """Format and print real-time research progress events to terminal."""
    ev_type = event.get("event")

    if ev_type == "section_pass_start":
        title = event.get("section_title", "Section")
        att = event.get("attempt", 1)
        max_att = event.get("max_iterations", 5)
        print(f"\n[RESEARCH] [{event.get('ordinal', 1)}] '{title}' (Pass {att}/{max_att})...")

    elif ev_type == "section_pass_end":
        passed = event.get("passed", False)
        score = event.get("score", 0.0)
        wc = event.get("word_count", 0)
        sc = event.get("source_count", 0)
        gaps = event.get("open_gaps", 0)

        if passed:
            print(f"   [PASS] Section Passed. Score: {score:.1f}% | Words: {wc} | Sources: {sc}")
        else:
            print(f"   [GATE-FAIL] Quality Gate: Score: {score:.1f}% | Words: {wc} | Sources: {sc} | Open Gaps: {gaps}")

    elif ev_type == "synthesis_start":
        print("\n[SYNTHESIS] Compiling completed sections into master research dossier...")

    elif ev_type == "run_completed":
        print("\n[COMPLETE] Research run successfully completed and exported.")
