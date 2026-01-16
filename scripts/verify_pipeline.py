from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenarioops.app.config import load_settings
from scenarioops.graph.build_graph import GraphInputs, run_graph
from scenarioops.graph.tools.run_manifest import build_artifact_index
from scenarioops.graph.tools.storage import default_runs_dir


@dataclass(frozen=True)
class Fixture:
    name: str
    path: Path
    inputs: dict[str, Any]
    expected_root: Path


@dataclass
class Issue:
    criterion: str
    location: str
    message: str
    fix: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fixture(path: Path) -> Fixture:
    inputs_path = path / "inputs.json"
    if not inputs_path.exists():
        raise FileNotFoundError(f"Fixture missing inputs.json: {inputs_path}")
    inputs = json.loads(inputs_path.read_text(encoding="utf-8-sig"))
    expected_root = path / "expected"
    return Fixture(name=path.name, path=path, inputs=inputs, expected_root=expected_root)


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    artifacts_dir = run_dir / "artifacts"
    trace_dir = run_dir / "trace"
    files: list[Path] = []
    if artifacts_dir.exists():
        files.extend([path for path in artifacts_dir.iterdir() if path.is_file()])
    if trace_dir.exists():
        files.extend([path for path in trace_dir.iterdir() if path.is_file()])

    hashes: dict[str, str] = {}
    for path in sorted(files):
        if path.name.endswith(".meta.json"):
            continue
        payload = path.read_bytes()
        relative = str(path.relative_to(run_dir))
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    return hashes


def _compare_expected(run_dir: Path, expected_root: Path) -> list[Issue]:
    issues: list[Issue] = []
    if not expected_root.exists():
        return issues
    for expected_path in sorted(expected_root.rglob("*")):
        if expected_path.is_dir():
            continue
        relative = expected_path.relative_to(expected_root)
        actual_path = run_dir / relative
        if not actual_path.exists():
            issues.append(
                Issue(
                    criterion="Golden",
                    location=str(actual_path),
                    message="Expected artifact missing.",
                    fix="Regenerate fixture artifacts from the deterministic run outputs.",
                )
            )
            continue
        if expected_path.suffix == ".json":
            expected = _load_json(expected_path)
            actual = _load_json(actual_path)
            if expected != actual:
                issues.append(
                    Issue(
                        criterion="Golden",
                        location=str(actual_path),
                        message="Artifact JSON mismatch vs canonical.",
                        fix="Update fixture artifacts after validating the change.",
                    )
                )
        elif expected_path.suffix == ".jsonl":
            expected_lines = expected_path.read_text(encoding="utf-8").splitlines()
            actual_lines = actual_path.read_text(encoding="utf-8").splitlines()
            if expected_lines != actual_lines:
                issues.append(
                    Issue(
                        criterion="Golden",
                        location=str(actual_path),
                        message="Artifact JSONL mismatch vs canonical.",
                        fix="Update fixture artifacts after validating the change.",
                    )
                )
        else:
            expected_text = expected_path.read_text(encoding="utf-8")
            actual_text = actual_path.read_text(encoding="utf-8")
            if expected_text != actual_text:
                issues.append(
                    Issue(
                        criterion="Golden",
                        location=str(actual_path),
                        message="Artifact text mismatch vs canonical.",
                        fix="Update fixture artifacts after validating the change.",
                    )
                )
    return issues


def _check_coverage(run_dir: Path) -> list[Issue]:
    issues: list[Issue] = []
    path = run_dir / "artifacts" / "coverage_report.json"
    if not path.exists():
        issues.append(
            Issue(
                criterion="Coverage",
                location=str(path),
                message="Coverage report missing.",
                fix="Ensure run_coverage_node executes after scan/washout.",
            )
        )
        return issues
    payload = _load_json(path)
    for entry in payload.get("steep", []):
        if entry.get("status") != "covered":
            issues.append(
                Issue(
                    criterion="Coverage",
                    location=str(path),
                    message=f"STEEP dimension missing: {entry.get('dimension')}.",
                    fix="Require at least one driving force per STEEP dimension.",
                )
            )
    for entry in payload.get("lenses", []):
        if entry.get("status") != "covered":
            issues.append(
                Issue(
                    criterion="Coverage",
                    location=str(path),
                    message=f"Study lens missing: {entry.get('lens')}.",
                    fix="Require at least one driving force tagged with each lens.",
                )
            )
    return issues


def _check_epistemics(run_dir: Path) -> list[Issue]:
    issues: list[Issue] = []
    path = run_dir / "artifacts" / "epistemic_summary.json"
    if not path.exists():
        issues.append(
            Issue(
                criterion="Epistemics",
                location=str(path),
                message="Epistemic summary missing.",
                fix="Ensure run_epistemic_summary_node executes after effects.",
            )
        )
        return issues
    payload = _load_json(path)
    for field in ("facts", "assumptions", "interpretations", "unknowns"):
        if not payload.get(field):
            issues.append(
                Issue(
                    criterion="Epistemics",
                    location=str(path),
                    message=f"Epistemic field empty: {field}.",
                    fix="Ensure upstream nodes populate the epistemic summary.",
                )
            )
    return issues


def _check_outcomes(run_dir: Path) -> list[Issue]:
    issues: list[Issue] = []
    path = run_dir / "artifacts" / "scenario_profiles.json"
    if not path.exists():
        issues.append(
            Issue(
                criterion="Outcomes",
                location=str(path),
                message="Scenario profiles missing.",
                fix="Ensure run_scenario_profiles_node executes before trace map.",
            )
        )
        return issues
    payload = _load_json(path)
    scenarios = payload.get("scenarios", [])
    narratives = {item.get("narrative") for item in scenarios if item.get("narrative")}
    if len(narratives) < 2:
        issues.append(
            Issue(
                criterion="Outcomes",
                location=str(path),
                message="Scenario narratives are not differentiated.",
                fix="Ensure scenarios produce distinct narrative text.",
            )
        )
    for scenario in scenarios:
        scenario_id = scenario.get("id", "unknown")
        if not scenario.get("options"):
            issues.append(
                Issue(
                    criterion="Outcomes",
                    location=str(path),
                    message=f"Scenario {scenario_id} missing options.",
                    fix="Generate strategies and wind-tunnel tests for each scenario.",
                )
            )
        if not scenario.get("what_would_change_my_mind"):
            issues.append(
                Issue(
                    criterion="Outcomes",
                    location=str(path),
                    message=f"Scenario {scenario_id} missing trigger signals.",
                    fix="Ensure EWIs are linked to each scenario.",
                )
            )
    return issues


def _check_governance(run_dir: Path) -> list[Issue]:
    issues: list[Issue] = []
    manifest_path = run_dir / "manifest.json"
    index_path = run_dir / "artifacts" / "index.json"
    trace_path = run_dir / "trace" / "trace_map.json"
    if not manifest_path.exists():
        issues.append(
            Issue(
                criterion="Governance",
                location=str(manifest_path),
                message="Run manifest missing.",
                fix="Ensure write_run_manifest executes on success.",
            )
        )
    if not index_path.exists():
        issues.append(
            Issue(
                criterion="Governance",
                location=str(index_path),
                message="Artifact index missing.",
                fix="Ensure write_artifact_index executes on success.",
            )
        )
    if not trace_path.exists():
        issues.append(
            Issue(
                criterion="Governance",
                location=str(trace_path),
                message="Trace map missing.",
                fix="Ensure run_trace_map_node executes after scenario profiles.",
            )
        )
    return issues


def _check_trace_map(run_dir: Path) -> list[Issue]:
    issues: list[Issue] = []
    trace_path = run_dir / "trace" / "trace_map.json"
    if not trace_path.exists():
        return issues
    payload = _load_json(trace_path)
    for scenario in payload.get("scenarios", []):
        scenario_id = scenario.get("scenario_id", "unknown")
        links = scenario.get("links", [])
        for link in links:
            if link.get("ids") == []:
                issues.append(
                    Issue(
                        criterion="Governance",
                        location=str(trace_path),
                        message=f"Scenario {scenario_id} has empty trace ids for {link.get('claim_type')}.",
                        fix="Link scenario claims to upstream artifacts or mark as assumptions.",
                    )
                )
    return issues


def _write_summary(run_dir: Path, fixture: Fixture, issues: list[Issue]) -> None:
    summary_path = run_dir / "reports" / "verification_summary.md"
    criteria = ["Process", "Epistemics", "Coverage", "Outcomes", "Governance", "Golden"]
    lines = [f"# Verification Summary ({fixture.name})", ""]
    lines.append("## Criteria")
    for criterion in criteria:
        status = "PASS"
        if any(issue.criterion == criterion for issue in issues):
            status = "FAIL"
        lines.append(f"- {criterion}: {status}")
    if issues:
        lines.append("")
        lines.append("## Issues")
        for issue in issues:
            lines.append(f"- Criterion: {issue.criterion}")
            lines.append(f"  Location: {issue.location}")
            lines.append(f"  Issue: {issue.message}")
            lines.append(f"  Fix: {issue.fix}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_fixture(fixture: Fixture, base_dir: Path, runs: int = 3) -> list[Issue]:
    overrides = fixture.inputs.get("settings_overrides") or {}
    overrides.setdefault("mode", "demo")
    overrides.setdefault("sources_policy", "fixtures")
    overrides.setdefault("llm_provider", "mock")
    settings = load_settings(overrides)
    run_timestamp = fixture.inputs.get("run_timestamp") or "2026-01-01T00:00:00+00:00"
    report_date = fixture.inputs.get("report_date") or "2026-01-01"

    hashes: list[dict[str, str]] = []
    issues: list[Issue] = []
    last_run_dir: Path | None = None

    run_id = fixture.name
    for _ in range(runs):
        inputs = GraphInputs(
            user_params=fixture.inputs.get("user_params", {}),
            sources=fixture.inputs.get("sources", []),
            signals=fixture.inputs.get("signals", []),
        )
        run_graph(
            inputs,
            run_id=run_id,
            run_timestamp=run_timestamp,
            base_dir=base_dir,
            mock_mode=True,
            settings=settings,
            generate_strategies=True,
            report_date=report_date,
            command="verify-pipeline",
        )
        run_dir = base_dir / run_id
        last_run_dir = run_dir
        hashes.append(_artifact_hashes(run_dir))

    if hashes[0] != hashes[1] or hashes[1] != hashes[2]:
        issues.append(
            Issue(
                criterion="Process",
                location=str(base_dir / fixture.name),
                message="Artifact hashes differ across runs.",
                fix="Ensure deterministic IDs and stable normalization.",
            )
        )

    if last_run_dir is None:
        return issues

    issues.extend(_check_coverage(last_run_dir))
    issues.extend(_check_epistemics(last_run_dir))
    issues.extend(_check_outcomes(last_run_dir))
    issues.extend(_check_governance(last_run_dir))
    issues.extend(_check_trace_map(last_run_dir))

    try:
        build_artifact_index(last_run_dir.name, base_dir=base_dir, strict=True)
    except Exception as exc:
        issues.append(
            Issue(
                criterion="Process",
                location=str(last_run_dir / "artifacts"),
                message=f"Artifact schema validation failed: {exc}",
                fix="Ensure all artifacts validate against their schemas.",
            )
        )

    issues.extend(_compare_expected(last_run_dir, fixture.expected_root))
    _write_summary(last_run_dir, fixture, issues)
    return issues


def run_verification(fixtures_root: Path, base_dir: Path) -> int:
    fixtures = [
        _load_fixture(path)
        for path in sorted(fixtures_root.iterdir())
        if path.is_dir()
    ]
    if not fixtures:
        print(f"No fixtures found in {fixtures_root}")
        return 1
    all_issues: list[Issue] = []
    for fixture in fixtures:
        all_issues.extend(verify_fixture(fixture, base_dir))
    if all_issues:
        print("verification failed")
        for issue in all_issues[:5]:
            print(f"- {issue.criterion}: {issue.message} ({issue.location})")
        return 1
    print("verification ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        default=str(Path("tests") / "fixtures"),
        help="Path to fixtures root.",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for runs (defaults to storage/runs).",
    )
    args = parser.parse_args(argv)
    fixtures_root = Path(args.fixtures)
    base_dir = Path(args.base_dir) if args.base_dir else default_runs_dir()
    return run_verification(fixtures_root, base_dir)


if __name__ == "__main__":
    sys.exit(main())
