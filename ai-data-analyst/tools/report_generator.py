"""
Report generation tool.

Generates downloadable reports in two formats:
1. Markdown — lightweight, human-readable, version-control friendly
2. PDF — polished business report using ReportLab

Reports include:
- Executive summary
- Dataset overview and quality metrics
- Key insights
- Anomaly report
- Charts (as text descriptions in MD; as images in PDF)
- Full conversation history
- Generated SQL / Pandas code
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import settings
from models.analysis_models import AnomalyReport, BusinessInsight, DataProfile
from utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generates analytical reports in Markdown and PDF formats.

    Designed to be called at the end of an analysis session
    or on demand from the UI download buttons.
    """

    def generate_markdown_report(
        self,
        dataset_name: str,
        profile: Optional[DataProfile] = None,
        insights: Optional[list[BusinessInsight]] = None,
        anomaly_report: Optional[AnomalyReport] = None,
        conversation_history: Optional[list[dict]] = None,
        additional_sections: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Generate a complete Markdown report.

        Args:
            dataset_name: Name of the analyzed dataset.
            profile: DataProfile from the profiler.
            insights: List of BusinessInsight objects.
            anomaly_report: AnomalyReport if detection was run.
            conversation_history: List of {"role", "content"} dicts.
            additional_sections: Dict of {section_title: content} for custom sections.

        Returns:
            Complete Markdown string.
        """
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        sections: list[str] = []

        # ── Header ─────────────────────────────────────────────────────────────
        sections.append(f"# AI Data Analysis Report: {dataset_name}")
        sections.append(f"*Generated: {now}*")
        sections.append(f"*Powered by AI Data Analyst v{settings.app_version}*")
        sections.append("")

        # ── Dataset Overview ───────────────────────────────────────────────────
        if profile:
            sections.append("## Dataset Overview")
            sections.append(f"| Metric | Value |")
            sections.append(f"|---|---|")
            sections.append(f"| Dataset | {profile.dataset_name} |")
            sections.append(f"| Rows | {profile.row_count:,} |")
            sections.append(f"| Columns | {profile.column_count} |")
            sections.append(
                f"| Memory | {profile.memory_usage_bytes / 1024 / 1024:.2f} MB |"
            )
            sections.append(
                f"| Quality Score | {profile.quality_report.overall_quality_score}/100 |"
            )
            sections.append("")

            # Column summary
            sections.append("### Columns")
            sections.append("| Column | Type | Nulls | Unique |")
            sections.append("|---|---|---|---|")
            for col in profile.column_profiles[:30]:
                sections.append(
                    f"| {col.name} | {col.dtype} | "
                    f"{col.null_percentage:.1f}% | {col.unique_count:,} |"
                )
            if len(profile.column_profiles) > 30:
                sections.append(f"| ... and {len(profile.column_profiles) - 30} more | | | |")
            sections.append("")

            # Quality issues
            if profile.quality_report.quality_issues:
                sections.append("### Data Quality Issues")
                for issue in profile.quality_report.quality_issues:
                    sections.append(f"- ⚠️ {issue}")
                sections.append("")

        # ── Key Insights ───────────────────────────────────────────────────────
        if insights:
            sections.append("## Business Insights")
            for i, insight in enumerate(insights[:10], 1):
                sections.append(f"### {i}. {insight.title}")
                sections.append(insight.description)
                if insight.metric_name and insight.metric_value is not None:
                    sections.append(
                        f"*Metric: {insight.metric_name} = {insight.metric_value}*"
                    )
                sections.append("")

        # ── Anomaly Report ─────────────────────────────────────────────────────
        if anomaly_report and anomaly_report.anomalies_detected > 0:
            sections.append("## Anomaly Detection Report")
            sections.append(f"**Method:** {anomaly_report.detection_method.replace('_', ' ').title()}")
            sections.append(
                f"**Records Analyzed:** {anomaly_report.total_records_analyzed:,}"
            )
            sections.append(
                f"**Anomalies Found:** {anomaly_report.anomalies_detected} "
                f"({anomaly_report.anomaly_percentage:.1f}%)"
            )
            sections.append("")
            sections.append(anomaly_report.summary)
            sections.append("")

            if anomaly_report.anomaly_records:
                sections.append("### Sample Anomalous Records")
                sections.append("| Row | Score | Flagged Columns | Explanation |")
                sections.append("|---|---|---|---|")
                for rec in anomaly_report.anomaly_records[:10]:
                    flagged = ", ".join(rec.flagged_columns[:3])
                    explanation = rec.explanation[:100] + "..." if len(rec.explanation) > 100 else rec.explanation
                    sections.append(
                        f"| {rec.row_index} | {rec.anomaly_score:.4f} | {flagged} | {explanation} |"
                    )
                sections.append("")

        # ── Conversation History ───────────────────────────────────────────────
        if conversation_history:
            sections.append("## Conversation History")
            for msg in conversation_history:
                role = msg.get("role", "unknown").title()
                content = msg.get("content", "")
                if len(content) > 500:
                    content = content[:500] + "..."
                sections.append(f"**{role}:** {content}")
                sections.append("")

        # ── Custom Sections ────────────────────────────────────────────────────
        if additional_sections:
            for title, content in additional_sections.items():
                sections.append(f"## {title}")
                sections.append(content)
                sections.append("")

        # ── Footer ─────────────────────────────────────────────────────────────
        sections.append("---")
        sections.append(
            f"*This report was automatically generated by {settings.app_title}. "
            f"All insights are based on the provided dataset and should be validated "
            f"before making business decisions.*"
        )

        return "\n".join(sections)

    def generate_pdf_report(
        self,
        dataset_name: str,
        profile: Optional[DataProfile] = None,
        insights: Optional[list[BusinessInsight]] = None,
        anomaly_report: Optional[AnomalyReport] = None,
    ) -> bytes:
        """
        Generate a PDF report using ReportLab.

        Args:
            dataset_name: Name of the analyzed dataset.
            profile: DataProfile if available.
            insights: Business insights list.
            anomaly_report: Anomaly report if available.

        Returns:
            PDF content as bytes.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
                HRFlowable,
            )

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=inch,
                leftMargin=inch,
                topMargin=inch,
                bottomMargin=inch,
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Title"],
                fontSize=20,
                spaceAfter=12,
                textColor=colors.HexColor("#1a1a2e"),
            )
            heading_style = ParagraphStyle(
                "CustomHeading",
                parent=styles["Heading2"],
                fontSize=14,
                spaceAfter=8,
                textColor=colors.HexColor("#16213e"),
            )
            body_style = ParagraphStyle(
                "CustomBody",
                parent=styles["Normal"],
                fontSize=10,
                spaceAfter=6,
            )

            story = []
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

            # Title
            story.append(Paragraph(f"AI Data Analysis Report", title_style))
            story.append(Paragraph(f"Dataset: {dataset_name}", heading_style))
            story.append(Paragraph(f"Generated: {now}", body_style))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0")))
            story.append(Spacer(1, 0.2 * inch))

            # Dataset Overview
            if profile:
                story.append(Paragraph("Dataset Overview", heading_style))
                overview_data = [
                    ["Metric", "Value"],
                    ["Rows", f"{profile.row_count:,}"],
                    ["Columns", str(profile.column_count)],
                    [
                        "Memory",
                        f"{profile.memory_usage_bytes / 1024 / 1024:.2f} MB",
                    ],
                    [
                        "Quality Score",
                        f"{profile.quality_report.overall_quality_score}/100",
                    ],
                ]

                t = Table(overview_data, colWidths=[2.5 * inch, 4 * inch])
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )
                story.append(t)
                story.append(Spacer(1, 0.2 * inch))

            # Business Insights
            if insights:
                story.append(Paragraph("Business Insights", heading_style))
                for i, insight in enumerate(insights[:8], 1):
                    story.append(
                        Paragraph(f"{i}. {insight.title}", body_style)
                    )
                    story.append(
                        Paragraph(insight.description, body_style)
                    )
                    story.append(Spacer(1, 0.1 * inch))

            # Anomaly Summary
            if anomaly_report and anomaly_report.anomalies_detected > 0:
                story.append(Paragraph("Anomaly Detection Summary", heading_style))
                story.append(
                    Paragraph(
                        f"Detected {anomaly_report.anomalies_detected} anomalies "
                        f"({anomaly_report.anomaly_percentage:.1f}%) using "
                        f"{anomaly_report.detection_method.replace('_', ' ').title()}.",
                        body_style,
                    )
                )
                story.append(Paragraph(anomaly_report.summary, body_style))
                story.append(Spacer(1, 0.1 * inch))

            # Footer
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0")))
            story.append(
                Paragraph(
                    f"Generated by {settings.app_title} v{settings.app_version}",
                    body_style,
                )
            )

            doc.build(story)
            return buffer.getvalue()

        except ImportError as exc:
            logger.error("ReportLab not available for PDF generation", error=str(exc))
            # Fallback: return the markdown report encoded as bytes
            md = self.generate_markdown_report(
                dataset_name=dataset_name,
                profile=profile,
                insights=insights,
                anomaly_report=anomaly_report,
            )
            return md.encode("utf-8")

    def save_markdown_report(
        self,
        content: str,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save a Markdown report to the reports directory.

        Args:
            content: Markdown string to save.
            filename: Optional filename override.

        Returns:
            Path to the saved file.
        """
        output_dir = settings.report_output_path
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.md"

        filepath = output_dir / filename
        filepath.write_text(content, encoding="utf-8")
        logger.info("Markdown report saved", path=str(filepath))
        return filepath


# Module-level singleton
report_generator = ReportGenerator()
