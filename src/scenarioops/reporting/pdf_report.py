from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from scenarioops.graph.tools.storage import ensure_run_dirs, default_runs_dir


def _load_artifact(run_id: str, name: str, base_dir: Path) -> dict[str, Any] | None:
    path = base_dir / run_id / "artifacts" / f"{name}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def _bullet_lines(items: list[str]) -> str:
    return "<br/>".join(f"- {item}" for item in items if item)


def build_management_report(run_id: str, base_dir: Path | None = None) -> Path:
    base_dir = base_dir or default_runs_dir()
    dirs = ensure_run_dirs(run_id, base_dir=base_dir)
    reports_dir = dirs["reports_dir"]
    output_path = reports_dir / f"management_report_{run_id}.pdf"

    charter = _load_artifact(run_id, "scenario_charter", base_dir)
    focal = _load_artifact(run_id, "focal_issue", base_dir)
    profile = _load_artifact(run_id, "company_profile", base_dir)
    retrieval = _load_artifact(run_id, "retrieval_report", base_dir)
    forces = _load_artifact(run_id, "forces", base_dir)
    forces_ranked = _load_artifact(run_id, "forces_ranked", base_dir)
    clusters = _load_artifact(run_id, "clusters", base_dir)
    axes = _load_artifact(run_id, "uncertainty_axes", base_dir)
    scenarios = _load_artifact(run_id, "scenarios", base_dir)
    strategies = _load_artifact(run_id, "strategies", base_dir)
    wind = _load_artifact(run_id, "wind_tunnel", base_dir)
    audit = _load_artifact(run_id, "audit_report", base_dir)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleLarge", parent=styles["Title"], fontSize=20, spaceAfter=12))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=14, spaceBefore=12))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], leading=14))

    story: list[Any] = []
    title = "ScenarioOps Management Report"
    story.append(_paragraph(title, styles["TitleLarge"]))
    story.append(Spacer(1, 0.2 * inch))

    company_name = None
    if profile:
        company_name = profile.get("company_name")
    if not company_name and charter:
        company_name = charter.get("title")
    company_name = company_name or "Company"
    report_date = datetime.now(timezone.utc).date().isoformat()
    story.append(_paragraph(f"{company_name} | Run {run_id} | {report_date}", styles["Body"]))
    story.append(Spacer(1, 0.2 * inch))

    # Executive summary
    summary_lines = []
    if charter:
        summary_lines.append(charter.get("purpose", ""))
    if focal:
        summary_lines.append(focal.get("focal_issue", ""))
    summary_text = " ".join([line for line in summary_lines if line]) or "Executive summary pending."
    story.append(_paragraph("Executive Summary", styles["Section"]))
    story.append(_paragraph(summary_text, styles["Body"]))

    # Charter
    story.append(_paragraph("Charter", styles["Section"]))
    if charter:
        body = [
            f"Decision context: {charter.get('decision_context', 'Pending')}.",
            f"Scope: {charter.get('scope', 'Pending')}.",
            f"Time horizon: {charter.get('time_horizon', 'Pending')}.",
        ]
        story.append(_paragraph(" ".join(body), styles["Body"]))
    else:
        story.append(_paragraph("Charter not available.", styles["Body"]))

    # Focal issue
    story.append(_paragraph("Focal Issue", styles["Section"]))
    if focal:
        scope = focal.get("scope", {}) if isinstance(focal.get("scope"), dict) else {}
        body = [
            focal.get("focal_issue", "Pending"),
            f"Geography: {scope.get('geography', 'Pending')}.",
            f"Time horizon (years): {scope.get('time_horizon_years', 'Pending')}.",
        ]
        story.append(_paragraph(" ".join(body), styles["Body"]))
    else:
        story.append(_paragraph("Focal issue not available.", styles["Body"]))

    # Retrieval summary
    story.append(_paragraph("Evidence & Retrieval", styles["Section"]))
    if retrieval:
        counts = retrieval.get("counts", {}) if isinstance(retrieval.get("counts"), dict) else {}
        body = [
            f"Evidence OK: {counts.get('ok', 0)}.",
            f"Evidence total: {counts.get('total', 0)}.",
            f"Evidence failed: {counts.get('failed', 0)}.",
        ]
        story.append(_paragraph(" ".join(body), styles["Body"]))
    else:
        story.append(_paragraph("Retrieval summary not available.", styles["Body"]))

    # Forces
    story.append(_paragraph("Forces", styles["Section"]))
    if forces and isinstance(forces.get("forces"), list):
        top_forces = [f.get("label", "") for f in forces.get("forces", [])[:8]]
        story.append(_paragraph(_bullet_lines(top_forces) or "Forces pending.", styles["Body"]))
    else:
        story.append(_paragraph("Forces not available.", styles["Body"]))

    # EBE ranking
    story.append(_paragraph("EBE Ranking Highlights", styles["Section"]))
    if forces_ranked and isinstance(forces_ranked.get("forces"), list):
        ranked = sorted(forces_ranked.get("forces", []), key=lambda f: f.get("ebe_score", 0), reverse=True)
        top = [f.get("rationale", "") for f in ranked[:5]]
        story.append(_paragraph(_bullet_lines(top) or "EBE ranking pending.", styles["Body"]))
    else:
        story.append(_paragraph("EBE ranking not available.", styles["Body"]))

    # Clusters and uncertainty axes
    story.append(_paragraph("Clusters & Uncertainty Axes", styles["Section"]))
    cluster_lines = []
    if clusters and isinstance(clusters.get("clusters"), list):
        cluster_lines = [c.get("cluster_label", "") for c in clusters.get("clusters", [])[:5]]
    axis_lines = []
    if axes and isinstance(axes.get("axes"), list):
        axis_lines = [a.get("axis_name", "") for a in axes.get("axes", [])]
    combined = _bullet_lines(cluster_lines + axis_lines)
    story.append(_paragraph(combined or "Cluster and axis outputs pending.", styles["Body"]))

    # Scenarios
    story.append(_paragraph("Scenario Narratives", styles["Section"]))
    if scenarios and isinstance(scenarios.get("scenarios"), list):
        scenario_blocks = []
        for scenario in scenarios.get("scenarios", []):
            name = scenario.get("name", "Scenario")
            summary = scenario.get("summary") or scenario.get("narrative", "")
            if summary:
                summary = summary[:420] + ("..." if len(summary) > 420 else "")
            scenario_blocks.append(f"{name}: {summary}")
        story.append(_paragraph(_bullet_lines(scenario_blocks), styles["Body"]))
    else:
        story.append(_paragraph("Scenario narratives not available.", styles["Body"]))

    # Strategies
    story.append(_paragraph("Strategy Options", styles["Section"]))
    if strategies and isinstance(strategies.get("strategies"), list):
        strategy_blocks = []
        for strategy in strategies.get("strategies", []):
            name = strategy.get("name", "Strategy")
            objective = strategy.get("objective", "")
            strategy_blocks.append(f"{name}: {objective}")
        story.append(_paragraph(_bullet_lines(strategy_blocks), styles["Body"]))
    else:
        story.append(_paragraph("Strategies not available.", styles["Body"]))

    # Wind tunnel
    story.append(_paragraph("Wind Tunnel Outcomes", styles["Section"]))
    if wind and isinstance(wind.get("matrix"), list):
        outcomes = {}
        for row in wind.get("matrix", []):
            outcome = row.get("outcome", "UNKNOWN")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        table_data = [["Outcome", "Count"]] + [[k, v] for k, v in outcomes.items()]
        table = Table(table_data, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e7c86")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7f4ef")),
                ]
            )
        )
        story.append(table)
    else:
        story.append(_paragraph("Wind tunnel outcomes not available.", styles["Body"]))

    # Audit
    story.append(_paragraph("Audit Findings", styles["Section"]))
    if audit:
        summary = audit.get("summary", "")
        findings = audit.get("findings", [])
        story.append(_paragraph(summary or "Audit summary pending.", styles["Body"]))
        if findings:
            story.append(_paragraph(_bullet_lines(findings), styles["Body"]))
    else:
        story.append(_paragraph("Audit report not available.", styles["Body"]))

    doc = SimpleDocTemplate(str(output_path), pagesize=LETTER, title=title)
    doc.build(story)
    return output_path
