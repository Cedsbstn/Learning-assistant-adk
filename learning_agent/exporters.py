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
Rich Exporters for Kythe Autonomous Deep Research Agent.

Implements generation for:
- Canonical Markdown (.md)
- Standalone Responsive HTML with Print CSS (.html)
- Vector PDF via ReportLab (.pdf)
- Structured Quiz JSON (.quiz.json)
- Flashcards JSON (.flashcards.json) & CSV (.flashcards.csv)
- Run Metadata & Artifact Manifest (.metadata.json)
- Atomic file writing with SHA256 hashing.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from models import Artifact, CurriculumModel, Flashcard, QuizQuestion, utc_now_iso

logger = logging.getLogger(__name__)


def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(filepath: str, content: str | bytes, is_binary: bool = False) -> str:
    """
    Safely write content to a temporary file and atomically rename into target path.
    """
    target = Path(filepath).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    mode = "wb" if is_binary else "w"
    encoding = None if is_binary else "utf-8"

    temp_fd, temp_path = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".tmp_{target.name}_",
    )
    try:
        with os.fdopen(temp_fd, mode, encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, str(target))
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return str(target)


def sanitize_filename(topic: str, max_len: int = 50) -> str:
    """Sanitize topic name for filesystem paths."""
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in topic)
    safe = safe.replace(" ", "_")[:max_len].strip("_")
    return safe or "research"


class MarkdownExporter:
    """Generates canonical human-readable markdown curriculum."""

    @staticmethod
    def export(curriculum: CurriculumModel, final_report: str, output_path: str) -> Artifact:
        topic = curriculum.topic
        metrics = curriculum.metrics
        sources = curriculum.sources

        content_parts = [
            f"# {topic}: Master Research Curriculum\n\n",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  \n",
            f"**Run ID:** `{curriculum.run_id}`  \n",
            f"**Total Modules:** {len(curriculum.sections_content)} | **Sources Cited:** {len(sources)}  \n\n",
            "---\n\n",
            "## Table of Contents\n\n",
        ]

        for s in curriculum.sections_content:
            status_icon = "[EXHAUSTED]" if s.get("status") == "EXHAUSTED" else "[PASS]"
            content_parts.append(
                f"- [Module {s.get('ordinal', 1)}: {s.get('title', 'Section')}](#module-{s.get('ordinal', 1)}-{s.get('title', '').lower().replace(' ', '-')}) {status_icon}\n"
            )

        content_parts.extend([
            "\n---\n\n",
            final_report,
            "\n\n---\n\n",
            "## Research Depth and Quality Metrics\n\n",
            "| Module | Depth Target | Status | Quality Score | Sources |\n",
            "|---|---|---|---|---|\n",
        ])

        for s in curriculum.sections_content:
            st = s.get("status", "COMPLETE")
            score = s.get("score")
            score_str = f"{score:.1f}%" if score is not None else "N/A"
            icon = "[PASS]" if st == "COMPLETE" else "[WARN]"
            content_parts.append(
                f"| {icon} {s.get('title', '')[:40]} | {s.get('depth_target', 'intermediate').capitalize()} | {st} | {score_str} | {len(s.get('sources', []))} |\n"
            )

        if curriculum.exhausted_sections:
            content_parts.extend([
                "\n\n> [!WARNING]\n",
                "> **Quality Note on Exhausted Sections**:\n",
                "> The following sections reached iteration limits with open research gaps:\n",
            ])
            for exh in curriculum.exhausted_sections:
                content_parts.append(f"> - **{exh}**\n")

        full_md = "".join(content_parts)
        saved_path = atomic_write(output_path, full_md)
        sha256_hash = compute_sha256(saved_path)

        return Artifact(
            artifact_id=f"art_md_{curriculum.run_id}",
            run_id=curriculum.run_id,
            type="markdown",
            path=saved_path,
            sha256=sha256_hash,
            created_at=utc_now_iso(),
        )


class HTMLExporter:
    """Generates standalone responsive HTML document with print CSS."""

    @staticmethod
    def export(curriculum: CurriculumModel, final_report: str, output_path: str) -> Artifact:
        topic_escaped = html.escape(curriculum.topic)
        run_id = curriculum.run_id

        # Convert simple markdown headers and code blocks to HTML for presentation
        body_html = ""
        in_code_block = False
        code_lang = ""

        for line in final_report.split("\n"):
            if line.startswith("```"):
                if in_code_block:
                    body_html += "</code></pre>\n"
                    in_code_block = False
                else:
                    code_lang = line[3:].strip()
                    body_html += f'<pre><code class="language-{code_lang}">'
                    in_code_block = True
                continue

            if in_code_block:
                body_html += html.escape(line) + "\n"
                continue

            if line.startswith("# "):
                body_html += f"<h1>{html.escape(line[2:])}</h1>\n"
            elif line.startswith("## "):
                body_html += f"<h2>{html.escape(line[3:])}</h2>\n"
            elif line.startswith("### "):
                body_html += f"<h3>{html.escape(line[4:])}</h3>\n"
            elif line.startswith("- "):
                body_html += f"<li>{html.escape(line[2:])}</li>\n"
            elif line.strip() == "---":
                body_html += "<hr/>\n"
            elif line.strip():
                # Convert markdown links [text](url)
                formatted_line = re.sub(
                    r'\[([^\]]+)\]\(([^)]+)\)',
                    r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
                    html.escape(line),
                )
                body_html += f"<p>{formatted_line}</p>\n"

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{topic_escaped} - Kythe Research Dossier</title>
    <style>
        :root {{
            --bg: #ffffff;
            --text: #1a202c;
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --sidebar-bg: #f8fafc;
            --border: #e2e8f0;
            --code-bg: #f1f5f9;
            --badge-bg: #e0f2fe;
            --badge-text: #0369a1;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg: #0f172a;
                --text: #f1f5f9;
                --primary: #3b82f6;
                --primary-dark: #60a5fa;
                --sidebar-bg: #1e293b;
                --border: #334155;
                --code-bg: #1e293b;
                --badge-bg: #1e3a8a;
                --badge-text: #93c5fd;
            }}
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        :focus-visible {{
            outline: 2px solid var(--primary);
            outline-offset: 2px;
        }}
        .skip-link {{
            position: absolute;
            top: -50px;
            left: 0;
            background: var(--primary);
            color: #ffffff;
            padding: 8px 16px;
            z-index: 100;
            text-decoration: none;
            font-weight: 600;
            border-radius: 0 0 4px 0;
        }}
        .skip-link:focus {{
            top: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.65;
            display: flex;
            min-height: 100vh;
        }}
        #sidebar {{
            width: 320px;
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border);
            padding: 2rem 1.5rem;
            position: sticky;
            top: 0;
            height: 100vh;
            overflow-y: auto;
            flex-shrink: 0;
        }}
        #sidebar h2 {{ font-size: 1.1rem; margin-bottom: 1rem; color: var(--primary); text-transform: uppercase; letter-spacing: 0.05em; }}
        #sidebar ul {{ list-style: none; }}
        #sidebar li {{ margin-bottom: 0.75rem; font-size: 0.95rem; }}
        #sidebar a {{ color: var(--text); text-decoration: none; transition: color 0.2s; }}
        #sidebar a:hover {{ color: var(--primary); }}
        #main {{
            flex: 1;
            padding: 3rem 4rem;
            max-width: 960px;
            margin: 0 auto;
        }}
        h1 {{ font-size: 2.25rem; margin-bottom: 1rem; line-height: 1.25; color: var(--primary); }}
        h2 {{ font-size: 1.6rem; margin-top: 2.5rem; margin-bottom: 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
        h3 {{ font-size: 1.25rem; margin-top: 1.75rem; margin-bottom: 0.75rem; }}
        p {{ margin-bottom: 1.25rem; }}
        ul, ol {{ margin-left: 1.5rem; margin-bottom: 1.25rem; }}
        li {{ margin-bottom: 0.5rem; }}
        a {{ color: var(--primary); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        hr {{ border: 0; height: 1px; background: var(--border); margin: 3rem 0; }}
        pre {{
            background-color: var(--code-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            overflow-x: auto;
            margin: 1.5rem 0;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.9rem;
        }}
        .badge {{
            display: inline-block;
            background-color: var(--badge-bg);
            color: var(--badge-text);
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
        }}
        @media (max-width: 768px) {{
            body {{
                flex-direction: column;
            }}
            #sidebar {{
                width: 100%;
                height: auto;
                position: static;
                border-right: none;
                border-bottom: 1px solid var(--border);
                padding: 1.5rem 1rem;
            }}
            #main {{
                padding: 1.5rem 1rem;
                max-width: 100%;
            }}
            #sidebar a {{
                display: inline-block;
                min-height: 44px;
                line-height: 44px;
            }}
        }}
        @media print {{
            #sidebar, .skip-link {{ display: none; }}
            body {{ display: block; background: #fff; color: #000; }}
            #main {{ max-width: 100%; padding: 0; }}
            pre, blockquote, table {{ page-break-inside: avoid; }}
            h2, h3 {{ page-break-after: avoid; }}
        }}
    </style>
</head>
<body>
    <a href="#main" class="skip-link">Skip to main content</a>
    <nav id="sidebar" aria-label="Curriculum Navigation">
        <h2>Curriculum Modules</h2>
        <ul>
            {"".join(f'<li><a href="#module-{s.get("ordinal", 1)}">Module {s.get("ordinal", 1)}: {html.escape(s.get("title", ""))[:30]}</a></li>' for s in curriculum.sections_content)}
        </ul>
        <hr style="margin: 1.5rem 0;"/>
        <p style="font-size: 0.8rem; color: #64748b;">
            Run ID: <code>{run_id}</code><br/>
            Sources: {len(curriculum.sources)}<br/>
            Generated by Kythe
        </p>
    </nav>
    <main id="main">
        <div class="badge">Kythe Research Output</div>
        {body_html}
    </main>
</body>
</html>"""

        saved_path = atomic_write(output_path, html_template)
        sha256_hash = compute_sha256(saved_path)

        return Artifact(
            artifact_id=f"art_html_{curriculum.run_id}",
            run_id=curriculum.run_id,
            type="html",
            path=saved_path,
            sha256=sha256_hash,
            created_at=utc_now_iso(),
        )


class PDFExporter:
    """Generates professional PDF documents via ReportLab."""

    @staticmethod
    def export(curriculum: CurriculumModel, final_report: str, output_path: str) -> Artifact:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            HRFlowable,
            KeepTogether,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=54,
            leftMargin=54,
            topMargin=54,
            bottomMargin=54,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1e3a8a"),
            alignment=0,
            spaceAfter=12,
        )
        h1_style = ParagraphStyle(
            "Heading1_Custom",
            parent=styles["Heading1"],
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#1d4ed8"),
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True,
        )
        h2_style = ParagraphStyle(
            "Heading2_Custom",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        )
        body_style = ParagraphStyle(
            "Body_Custom",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#334155"),
            spaceAfter=8,
        )
        code_style = ParagraphStyle(
            "Code_Custom",
            parent=styles["Code"],
            fontSize=8,
            leading=10.5,
            fontName="Courier",
            textColor=colors.HexColor("#0f172a"),
            backColor=colors.HexColor("#f1f5f9"),
            spaceAfter=8,
            leftIndent=12,
            rightIndent=12,
        )

        elements = []

        # Title Block
        elements.append(Paragraph(f"{curriculum.topic}", title_style))
        elements.append(Paragraph(f"<b>Comprehensive Research Curriculum & Technical Specification</b>", h2_style))
        elements.append(Paragraph(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
            f"Run ID: {curriculum.run_id} | Modules: {len(curriculum.sections_content)} | Sources: {len(curriculum.sources)}",
            body_style,
        ))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=14))

        # Render sections from final report
        in_code = False
        code_lines: List[str] = []

        for line in final_report.split("\n"):
            line_str = line.strip()
            if line_str.startswith("```"):
                if in_code:
                    code_block_text = "<br/>".join(html.escape(cl) for cl in code_lines)
                    elements.append(Paragraph(code_block_text, code_style))
                    code_lines = []
                    in_code = False
                else:
                    in_code = True
                continue

            if in_code:
                code_lines.append(line)
                continue

            if not line_str:
                continue

            if line_str.startswith("# "):
                elements.append(Paragraph(html.escape(line_str[2:]), h1_style))
            elif line_str.startswith("## "):
                elements.append(Paragraph(html.escape(line_str[3:]), h2_style))
            elif line_str.startswith("### "):
                elements.append(Paragraph(f"<b>{html.escape(line_str[4:])}</b>", body_style))
            elif line_str.startswith("- "):
                elements.append(Paragraph(f"• {html.escape(line_str[2:])}", body_style))
            elif line_str == "---":
                elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceBefore=8, spaceAfter=8))
            else:
                formatted = re.sub(r'<cite\s+source="([^"]+)"/>', r'<b>[\1]</b>', line_str)
                elements.append(Paragraph(html.escape(formatted), body_style))

        # Metrics Table
        elements.append(Spacer(1, 14))
        elements.append(Paragraph("<b>Section Quality & Research Depth Summary</b>", h2_style))

        table_data = [["#", "Module Title", "Depth", "Status", "Score", "Sources"]]
        for s in curriculum.sections_content:
            score_val = f"{s.get('score'):.1f}%" if s.get("score") is not None else "N/A"
            table_data.append([
                str(s.get("ordinal", 1)),
                Paragraph(s.get("title", "")[:35], body_style),
                s.get("depth_target", "intermediate").capitalize(),
                s.get("status", "COMPLETE"),
                score_val,
                str(len(s.get("sources", []))),
            ])

        t = Table(table_data, colWidths=[24, 210, 70, 75, 55, 50])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ]))
        elements.append(t)

        doc.build(elements)
        sha256_hash = compute_sha256(output_path)

        return Artifact(
            artifact_id=f"art_pdf_{curriculum.run_id}",
            run_id=curriculum.run_id,
            type="pdf",
            path=str(Path(output_path).resolve()),
            sha256=sha256_hash,
            created_at=utc_now_iso(),
        )


class QuizExporter:
    """Generates structured Quiz assessment JSON."""

    @staticmethod
    def export(curriculum: CurriculumModel, output_path: str) -> Artifact:
        data = {
            "topic": curriculum.topic,
            "run_id": curriculum.run_id,
            "generated_at": curriculum.generated_at,
            "quiz_questions": curriculum.quizzes,
        }
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        saved_path = atomic_write(output_path, json_str)
        sha256_hash = compute_sha256(saved_path)

        return Artifact(
            artifact_id=f"art_quiz_{curriculum.run_id}",
            run_id=curriculum.run_id,
            type="quiz",
            path=saved_path,
            sha256=sha256_hash,
            created_at=utc_now_iso(),
        )


class FlashcardsExporter:
    """Generates Flashcards in JSON and CSV formats."""

    @staticmethod
    def export_json(curriculum: CurriculumModel, output_path: str) -> Artifact:
        data = {
            "topic": curriculum.topic,
            "run_id": curriculum.run_id,
            "generated_at": curriculum.generated_at,
            "flashcards": curriculum.flashcards,
        }
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        saved_path = atomic_write(output_path, json_str)
        sha256_hash = compute_sha256(saved_path)

        return Artifact(
            artifact_id=f"art_fc_json_{curriculum.run_id}",
            run_id=curriculum.run_id,
            type="flashcards_json",
            path=saved_path,
            sha256=sha256_hash,
            created_at=utc_now_iso(),
        )

    @staticmethod
    def export_csv(curriculum: CurriculumModel, output_path: str) -> Artifact:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(["card_id", "front", "back", "section_id", "difficulty", "tags", "source_ids"])

        for f in curriculum.flashcards:
            tags_str = ";".join(f.get("tags", [])) if isinstance(f.get("tags"), list) else str(f.get("tags", ""))
            src_str = ";".join(f.get("source_ids", [])) if isinstance(f.get("source_ids"), list) else str(f.get("source_ids", ""))
            writer.writerow([
                f.get("card_id", ""),
                f.get("front", ""),
                f.get("back", ""),
                f.get("section_id", ""),
                f.get("difficulty", "intermediate"),
                tags_str,
                src_str,
            ])

        saved_path = atomic_write(output_path, buf.getvalue())
        sha256_hash = compute_sha256(saved_path)

        return Artifact(
            artifact_id=f"art_fc_csv_{curriculum.run_id}",
            run_id=curriculum.run_id,
            type="flashcards_csv",
            path=saved_path,
            sha256=sha256_hash,
            created_at=utc_now_iso(),
        )


class MetadataExporter:
    """Generates complete research run metadata and artifact manifest."""

    @staticmethod
    def export(
        curriculum: CurriculumModel,
        artifacts: List[Artifact],
        output_path: str,
    ) -> Artifact:
        data = {
            "topic": curriculum.topic,
            "run_id": curriculum.run_id,
            "generated_at": curriculum.generated_at,
            "metrics": curriculum.metrics,
            "sections": [
                {
                    "ordinal": s.get("ordinal"),
                    "title": s.get("title"),
                    "depth_target": s.get("depth_target"),
                    "status": s.get("status"),
                    "score": s.get("score"),
                    "source_count": len(s.get("sources", [])),
                }
                for s in curriculum.sections_content
            ],
            "sources": curriculum.sources,
            "open_gaps": curriculum.open_gaps,
            "exhausted_sections": curriculum.exhausted_sections,
            "artifact_manifest": [a.to_dict() for a in artifacts],
            "artifacts": [a.to_dict() for a in artifacts],
        }
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        saved_path = atomic_write(output_path, json_str)
        sha256_hash = compute_sha256(saved_path)

        return Artifact(
            artifact_id=f"art_meta_{curriculum.run_id}",
            run_id=curriculum.run_id,
            type="metadata",
            path=saved_path,
            sha256=sha256_hash,
            created_at=utc_now_iso(),
        )


class ExportManager:
    """Coordinates multi-format exporting and artifact recording."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir

    def export_all(
        self,
        curriculum_model: CurriculumModel,
        final_report: str,
        formats: List[str],
        run_id: str,
    ) -> List[Artifact]:
        """Generate requested export targets."""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{sanitize_filename(curriculum_model.topic)}_{run_id[:8]}"
        generated_artifacts: List[Artifact] = []

        if "markdown" in formats or "md" in formats:
            md_path = str(out_dir / f"{base_name}.md")
            generated_artifacts.append(
                MarkdownExporter.export(curriculum_model, final_report, md_path)
            )

        if "html" in formats:
            html_path = str(out_dir / f"{base_name}.html")
            generated_artifacts.append(
                HTMLExporter.export(curriculum_model, final_report, html_path)
            )

        if "pdf" in formats:
            pdf_path = str(out_dir / f"{base_name}.pdf")
            generated_artifacts.append(
                PDFExporter.export(curriculum_model, final_report, pdf_path)
            )

        if "quiz" in formats:
            quiz_path = str(out_dir / f"{base_name}.quiz.json")
            generated_artifacts.append(
                QuizExporter.export(curriculum_model, quiz_path)
            )

        if "flashcards" in formats or "flashcard" in formats:
            fc_json_path = str(out_dir / f"{base_name}.flashcards.json")
            generated_artifacts.append(
                FlashcardsExporter.export_json(curriculum_model, fc_json_path)
            )
            fc_csv_path = str(out_dir / f"{base_name}.flashcards.csv")
            generated_artifacts.append(
                FlashcardsExporter.export_csv(curriculum_model, fc_csv_path)
            )

        if "metadata" in formats or "meta" in formats or True:
            meta_path = str(out_dir / f"{base_name}.metadata.json")
            meta_artifact = MetadataExporter.export(
                curriculum_model, generated_artifacts, meta_path
            )
            generated_artifacts.append(meta_artifact)

        return generated_artifacts
