from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scenarioops.app.config import LLMConfig
from scenarioops.app.config import ScenarioOpsSettings
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.nodes.narratives import extract_numeric_claims_without_citations
from scenarioops.graph.state import AuditFinding, AuditReport, ScenarioOpsState
from scenarioops.graph.tools.artifact_contracts import schema_for_artifact
from scenarioops.graph.tools.schema_validate import validate_artifact, validate_jsonl
from scenarioops.graph.tools.storage import default_runs_dir, write_artifact


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[Any]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _extract_citations(payload: Any) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "citations" and isinstance(value, list):
                citations.extend(
                    [entry for entry in value if isinstance(entry, dict)]
                )
            else:
                citations.extend(_extract_citations(value))
    elif isinstance(payload, list):
        for item in payload:
            citations.extend(_extract_citations(item))
    return citations


def _fixture_findings(citations: list[dict[str, Any]]) -> list[AuditFinding]:
    fixture_urls: list[str] = []
    fixture_hashes: list[str] = []
    for citation in citations:
        url = str(citation.get("url", "")).lower()
        if "example.com" in url:
            fixture_urls.append(url)
        excerpt_hash = str(citation.get("excerpt_hash", ""))
        if excerpt_hash.startswith("hash-"):
            fixture_hashes.append(excerpt_hash)

    findings: list[AuditFinding] = []
    if fixture_urls:
        unique_urls = list(dict.fromkeys(fixture_urls))
        findings.append(
            AuditFinding(
                id="fixture-citation-url",
                finding="Fixture citation url detected.",
                evidence=unique_urls[:3],
            )
        )
    if fixture_hashes:
        unique_hashes = list(dict.fromkeys(fixture_hashes))
        findings.append(
            AuditFinding(
                id="fixture-citation-hash",
                finding="Fixture citation excerpt hash detected.",
                evidence=unique_hashes[:3],
            )
        )
    return findings


def _publisher_findings(citations: list[dict[str, Any]]) -> list[AuditFinding]:
    missing: list[str] = []
    for citation in citations:
        publisher = str(citation.get("publisher", "")).strip()
        source_type = str(citation.get("source_type", "")).strip()
        if not publisher and not source_type:
            url = str(citation.get("url", "")).strip()
            missing.append(url or "unknown")
    if not missing:
        return []
    unique_missing = list(dict.fromkeys(missing))
    return [
        AuditFinding(
            id="citation-publisher-missing",
            finding="Citation missing publisher or source_type.",
            evidence=unique_missing[:3],
        )
    ]


def _serialize_finding(finding: AuditFinding) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": finding.id,
        "finding": finding.finding,
        "evidence": finding.evidence,
        "recommendations": finding.recommendations,
    }
    if finding.impact is not None:
        payload["impact"] = finding.impact
    return payload


def run_auditor_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    base_dir: Path | None = None,
    llm_client=None,
    config: LLMConfig | None = None,
    settings: ScenarioOpsSettings | None = None,
) -> ScenarioOpsState:
    if base_dir is None:
        base_dir = default_runs_dir()
    artifacts_dir = base_dir / run_id / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts not found for run {run_id}.")

    findings: list[AuditFinding] = []
    citations: list[dict[str, Any]] = []

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

        schema_name = schema_for_artifact(path.stem, path.suffix)
        if not schema_name:
            continue
        payload: Any = None
        try:
            if path.suffix == ".json":
                payload = _load_json(path)
                validate_artifact(schema_name, payload)
            elif path.suffix == ".jsonl":
                payload = _load_jsonl(path)
                validate_jsonl(schema_name, payload)
            elif path.suffix == ".md":
                validate_artifact(schema_name, path.read_text(encoding="utf-8"))
        except Exception as exc:
            findings.append(
                AuditFinding(
                    id=f"schema-failure-{path.stem}",
                    finding=str(exc),
                )
            )
        if payload is not None:
            citations.extend(_extract_citations(payload))

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

    findings.extend(_fixture_findings(citations))
    findings.extend(_publisher_findings(citations))

    resolved_mode = (settings.mode if settings else "demo").lower()
    fixture_only = all(finding.id.startswith("fixture-") for finding in findings)
    hard_fail = bool(findings) and (resolved_mode == "live" or not fixture_only)
    summary = "audit passed" if not hard_fail else "audit failed"
    remediation_actions: list[str] = []
    if findings:
        prompt_template = load_prompt("auditor")
        prompt = render_prompt(
            prompt_template,
            {"findings": [_serialize_finding(finding) for finding in findings]},
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
            "findings": [_serialize_finding(finding) for finding in findings],
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
    if hard_fail:
        raise RuntimeError("Audit failed with findings.")
    return state
