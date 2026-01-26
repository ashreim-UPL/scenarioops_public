from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart

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


def _section_heading(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return _paragraph(text, styles["Section"])


def _simple_table(rows: list[list[Any]]) -> Table:
    table = Table(rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e7c86")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#fbf7f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    return table


def _bar_chart(title: str, labels: list[str], values: list[float]) -> Drawing:
    drawing = Drawing(400, 180)
    drawing.add(String(0, 165, title, fontSize=10))
    chart = VerticalBarChart()
    chart.x = 40
    chart.y = 25
    chart.height = 120
    chart.width = 340
    chart.data = [values]
    chart.strokeColor = colors.black
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(values) if values else 1
    chart.valueAxis.valueStep = max(1, int(chart.valueAxis.valueMax / 4))
    chart.categoryAxis.categoryNames = labels
    chart.bars[0].fillColor = colors.HexColor("#0e7c86")
    drawing.add(chart)
    return drawing


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
    wind_eval = _load_artifact(run_id, "wind_tunnel_evaluations_v2", base_dir)
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
    story.append(_section_heading("Executive Summary", styles))
    story.append(_paragraph(summary_text, styles["Body"]))

    # Charter
    story.append(_section_heading("Charter", styles))
    if charter:
        table_rows = [
            ["Field", "Detail"],
            ["Decision context", charter.get("decision_context", "Pending")],
            ["Scope", charter.get("scope", "Pending")],
            ["Time horizon", charter.get("time_horizon", "Pending")],
        ]
        story.append(_simple_table(table_rows))
    else:
        story.append(_paragraph("Charter not available.", styles["Body"]))

    # Focal issue
    story.append(_section_heading("Focal Issue", styles))
    if focal:
        scope = focal.get("scope", {}) if isinstance(focal.get("scope"), dict) else {}
        story.append(_paragraph(focal.get("focal_issue", "Pending"), styles["Body"]))
        table_rows = [
            ["Field", "Detail"],
            ["Geography", scope.get("geography", "Pending")],
            ["Sectors", ", ".join(scope.get("sectors", []))],
            ["Time horizon (months)", scope.get("time_horizon_months", "Pending")],
            ["Decision type", focal.get("decision_type", "Pending")],
        ]
        story.append(_simple_table(table_rows))
    else:
        story.append(_paragraph("Focal issue not available.", styles["Body"]))

    # Retrieval summary
    story.append(_section_heading("Evidence & Retrieval", styles))
    if retrieval:
        counts = retrieval.get("counts", {}) if isinstance(retrieval.get("counts"), dict) else {}
        table_rows = [
            ["Metric", "Count"],
            ["Evidence OK", counts.get("ok", 0)],
            ["Evidence total", counts.get("total", 0)],
            ["Evidence failed", counts.get("failed", 0)],
        ]
        story.append(_simple_table(table_rows))
        story.append(Spacer(1, 0.1 * inch))
        story.append(
            _bar_chart(
                "Evidence health",
                ["OK", "Total", "Failed"],
                [counts.get("ok", 0), counts.get("total", 0), counts.get("failed", 0)],
            )
        )
    else:
        story.append(_paragraph("Retrieval summary not available.", styles["Body"]))

    # Forces
    story.append(_section_heading("Forces", styles))
    if forces and isinstance(forces.get("forces"), list):
        top_forces = [f.get("label", "") for f in forces.get("forces", [])[:8]]
        table_rows = [["Top forces"]] + [[name] for name in top_forces if name]
        story.append(_simple_table(table_rows))
    else:
        story.append(_paragraph("Forces not available.", styles["Body"]))

    # EBE ranking
    story.append(_section_heading("EBE Ranking Highlights", styles))
    if forces_ranked and isinstance(forces_ranked.get("forces"), list):
        ranked = sorted(forces_ranked.get("forces", []), key=lambda f: f.get("ebe_score", 0), reverse=True)
        top = ranked[:5]
        table_rows = [["Force ID", "EBE Score", "Rationale"]]
        for item in top:
            table_rows.append([
                item.get("force_id", ""),
                item.get("ebe_score", ""),
                item.get("rationale", ""),
            ])
        story.append(_simple_table(table_rows))
        story.append(Spacer(1, 0.1 * inch))
        story.append(
            _bar_chart(
                "Top EBE scores",
                [item.get("force_id", "") for item in top],
                [float(item.get("ebe_score", 0)) for item in top],
            )
        )
    else:
        story.append(_paragraph("EBE ranking not available.", styles["Body"]))

    # Clusters and uncertainty axes
    story.append(_section_heading("Clusters & Uncertainty Axes", styles))
    cluster_lines = []
    if clusters and isinstance(clusters.get("clusters"), list):
        cluster_lines = [c.get("cluster_label", "") for c in clusters.get("clusters", [])[:5]]
    axis_lines = []
    if axes and isinstance(axes.get("axes"), list):
        axis_lines = [a.get("axis_name", "") for a in axes.get("axes", [])]
    combined = _bullet_lines(cluster_lines + axis_lines)
    story.append(_paragraph(combined or "Cluster and axis outputs pending.", styles["Body"]))

    # Scenarios
    story.append(PageBreak())
    story.append(_section_heading("Scenario Narratives", styles))
    if scenarios and isinstance(scenarios.get("scenarios"), list):
        table_rows = [["Scenario", "Axis states", "Key implications"]]
        for scenario in scenarios.get("scenarios", []):
            axis_states = scenario.get("axis_states", {})
            axis_text = ", ".join(f"{k}: {v}" for k, v in axis_states.items()) if isinstance(axis_states, dict) else ""
            implications = scenario.get("implications", [])
            imp_text = "; ".join(implications[:3]) if isinstance(implications, list) else ""
            table_rows.append([scenario.get("name", "Scenario"), axis_text, imp_text])
        story.append(_simple_table(table_rows))
    else:
        story.append(_paragraph("Scenario narratives not available.", styles["Body"]))

    # Strategies
    story.append(_section_heading("Strategy Options", styles))
    if strategies and isinstance(strategies.get("strategies"), list):
        table_rows = [["Strategy", "Objective", "KPIs"]]
        for strategy in strategies.get("strategies", []):
            kpis = "; ".join(strategy.get("kpis", [])) if isinstance(strategy.get("kpis"), list) else ""
            table_rows.append([strategy.get("name", "Strategy"), strategy.get("objective", ""), kpis])
        story.append(_simple_table(table_rows))
    else:
        story.append(_paragraph("Strategies not available.", styles["Body"]))

    # Wind tunnel
    story.append(_section_heading("Wind Tunnel Outcomes", styles))
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

    # Wind tunnel scorecard & rankings
    if wind_eval and isinstance(wind_eval.get("evaluations"), list):
        evaluations = wind_eval.get("evaluations", [])
        rankings = wind_eval.get("rankings", {}) if isinstance(wind_eval.get("rankings"), dict) else {}
        recommendations = (
            wind_eval.get("recommendations", {})
            if isinstance(wind_eval.get("recommendations"), dict)
            else {}
        )

        story.append(_paragraph("Wind Tunnel Scorecard", styles["Section"]))
        scenario_ids = []
        strategy_ids = []
        for item in evaluations:
            scenario_id = item.get("scenario_id")
            strategy_id = item.get("strategy_id")
            if scenario_id and scenario_id not in scenario_ids:
                scenario_ids.append(scenario_id)
            if strategy_id and strategy_id not in strategy_ids:
                strategy_ids.append(strategy_id)
        header = ["Strategy"] + scenario_ids
        table_rows = [header]
        for strategy_id in strategy_ids:
            strategy_name = next(
                (
                    item.get("strategy_name")
                    for item in evaluations
                    if item.get("strategy_id") == strategy_id
                ),
                strategy_id,
            )
            row = [strategy_name]
            for scenario_id in scenario_ids:
                cell = next(
                    (
                        item
                        for item in evaluations
                        if item.get("strategy_id") == strategy_id
                        and item.get("scenario_id") == scenario_id
                    ),
                    None,
                )
                if not cell:
                    row.append("-")
                else:
                    row.append(
                        f"{cell.get('grade_letter')} ({cell.get('score_0_100')})"
                    )
            table_rows.append(row)
        score_table = Table(table_rows, hAlign="LEFT")
        score_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e7c86")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7f4ef")),
                ]
            )
        )
        story.append(score_table)

        story.append(_paragraph("Strategy Ranking Summary", styles["Section"]))
        overall = rankings.get("overall", []) if isinstance(rankings, dict) else []
        if overall:
            ranking_rows = [["Strategy", "Overall", "Min", "Variance", "Robustness"]]
            for entry in overall:
                ranking_rows.append(
                    [
                        entry.get("strategy_name", entry.get("strategy_id")),
                        entry.get("overall_score"),
                        entry.get("min_score"),
                        entry.get("variance"),
                        entry.get("robustness_index"),
                    ]
                )
            ranking_table = Table(ranking_rows, hAlign="LEFT")
            ranking_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e7c86")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7f4ef")),
                    ]
                )
            )
            story.append(ranking_table)
        else:
            story.append(_paragraph("Strategy ranking pending.", styles["Body"]))

        story.append(_paragraph("Recommendation", styles["Section"]))
        primary = recommendations.get("primary_recommended_strategy", {}) if isinstance(recommendations, dict) else {}
        if primary:
            story.append(
                _paragraph(
                    f"Recommended strategy now: {primary.get('strategy_name', primary.get('strategy_id'))}.",
                    styles["Body"],
                )
            )
            story.append(_paragraph(primary.get("rationale", ""), styles["Body"]))
        hardening = recommendations.get("hardening_actions", []) if isinstance(recommendations, dict) else []
        if hardening:
            story.append(_paragraph("Hardening actions:", styles["Body"]))
            story.append(_paragraph(_bullet_lines(hardening[:5]), styles["Body"]))
        triggers = recommendations.get("triggers_to_watch", []) if isinstance(recommendations, dict) else []
        if triggers:
            trigger_lines = [t.get("description", "") for t in triggers if isinstance(t, dict)]
            story.append(_paragraph("Triggers and pivot plan:", styles["Body"]))
            story.append(_paragraph(_bullet_lines(trigger_lines[:5]), styles["Body"]))

        story.append(_paragraph("Appendix: Cell Evidence", styles["Section"]))
        by_strategy: dict[str, list[dict[str, Any]]] = {}
        for item in evaluations:
            by_strategy.setdefault(item.get("strategy_id", "unknown"), []).append(item)
        for strategy_id, items in by_strategy.items():
            story.append(_paragraph(f"Strategy: {strategy_id}", styles["Body"]))
            for item in items:
                headline = (
                    f"{item.get('scenario_id')}: {item.get('outcome_label')} "
                    f"{item.get('grade_letter')} ({item.get('score_0_100')}) "
                    f"confidence {item.get('confidence_0_1')}"
                )
                story.append(_paragraph(headline, styles["Body"]))
                rationale = item.get("rationale_bullets", [])
                if rationale:
                    story.append(_paragraph(_bullet_lines(rationale[:3]), styles["Body"]))
                triggers = item.get("trigger_points", [])
                if triggers:
                    trigger_lines = [t.get("description", "") for t in triggers if isinstance(t, dict)]
                    story.append(_paragraph(_bullet_lines(trigger_lines[:2]), styles["Body"]))

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
