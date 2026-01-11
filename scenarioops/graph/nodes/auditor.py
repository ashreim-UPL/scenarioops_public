from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import LLMConfig
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.nodes.narratives import extract_numeric_claims_without_citations
from scenarioops.graph.state import AuditFinding, AuditReport, ScenarioOpsState
from scenarioops.graph.tools.schema_validate import validate_artifact, validate_jsonl
from scenarioops.graph.tools.storage import default_runs_dir, write_artifact


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _artifact_schema(name: str, suffix: str) -> str | None:
    if suffix == ".md":
        return "markdown"
    if name == "drivers" and suffix == ".jsonl":
        return "driver_entry"
    mapping = {
        "scenario_charter": "charter",
        "uncertainties": "uncertainties",
        "logic": "logic",
        "skeletons": "skeleton",
        "ewi": "ewi",
        "strategies": "strategies",
        "wind_tunnel": "wind_tunnel",
        "daily_brief": "daily_brief",
        "audit_report": "audit_report",
    }
    return mapping.get(name)


def run_auditor_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    base_dir: Path | None = None,
    llm_client=None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    if base_dir is None:
        base_dir = default_runs_dir()
    artifacts_dir = base_dir / run_id / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts not found for run {run_id}.")

    findings: list[AuditFinding] = []

    artifacts = [path for path in artifacts_dir.iterdir() if path.is_file()]
    for path in artifacts:
        if path.name.endswith(".meta.json"):
            continue
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            findings.append(
                AuditFinding(
                    id=f"meta-missing-{path.stem}",
                    finding=f"Missing provenance for {path.name}",
                )
            )

        schema_name = _artifact_schema(path.stem, path.suffix)
        if not schema_name:
            continue
        try:
            if path.suffix == ".json":
                validate_artifact(schema_name, _load_json(path))
            elif path.suffix == ".jsonl":
                validate_jsonl(schema_name, _load_jsonl(path))
            elif path.suffix == ".md":
                validate_artifact(schema_name, path.read_text(encoding="utf-8"))
        except Exception as exc:
            findings.append(
                AuditFinding(
                    id=f"schema-failure-{path.stem}",
                    finding=str(exc),
                )
            )

        if path.stem == "drivers" and path.suffix == ".jsonl":
            for entry in _load_jsonl(path):
                if not entry.get("citations"):
                    findings.append(
                        AuditFinding(
                            id=f"citations-missing-{entry.get('id')}",
                            finding="Driver missing citations.",
                        )
                    )

        if path.suffix == ".md" and path.stem.startswith("narrative"):
            markdown = path.read_text(encoding="utf-8")
            claims = extract_numeric_claims_without_citations(markdown)
            if claims:
                findings.append(
                    AuditFinding(
                        id=f"numeric-claims-{path.stem}",
                        finding="Numeric claims missing citations.",
                        evidence=claims[:3],
                    )
                )

        if path.stem == "daily_brief" and path.suffix == ".md":
            markdown = path.read_text(encoding="utf-8")
            claims = extract_numeric_claims_without_citations(markdown)
            if claims:
                findings.append(
                    AuditFinding(
                        id="daily-brief-numeric-claims",
                        finding="Daily brief has numeric claims without citations.",
                        evidence=claims[:3],
                    )
                )

    summary = "audit passed" if not findings else "audit failed"
    remediation_actions: list[str] = []
    if findings:
        prompt_template = load_prompt("auditor")
        prompt = render_prompt(
            prompt_template,
            {"findings": [finding.__dict__ for finding in findings]},
        )
        client = get_client(llm_client, config)
        suggestions = client.generate_markdown(prompt)
        remediation_actions = [line.strip("- ").strip() for line in suggestions.splitlines() if line]

    report = AuditReport(
        id=f"audit-{run_id}",
        period_start=run_id,
        period_end=run_id,
        summary=summary,
        findings=findings,
        lessons=[],
        actions=remediation_actions,
    )

    write_artifact(
        run_id=run_id,
        artifact_name="audit_report",
        payload={
            "id": report.id,
            "period_start": report.period_start,
            "period_end": report.period_end,
            "summary": report.summary,
            "findings": [finding.__dict__ for finding in findings],
            "lessons": report.lessons,
            "actions": report.actions,
        },
        ext="json",
        input_values={"artifact_count": len(artifacts)},
        prompt_values={"prompt": "audit"},
        tool_versions={"auditor_node": "0.1.0"},
        base_dir=base_dir,
    )

    state.audit_report = report
    if findings:
        raise RuntimeError("Audit failed with findings.")
    return state
