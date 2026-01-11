from __future__ import annotations

import json
import random
import re
from pathlib import Path
import subprocess
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scenarioops.graph.tools.view_model import build_view_model


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "storage" / "runs"
LATEST_POINTER = RUNS_DIR / "latest.json"
RUN_LIMIT = 12
RANDOM_SEED = 42

STOPWORDS = {
    "a",
    "and",
    "the",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "be",
    "or",
    "that",
    "this",
    "an",
    "at",
    "into",
    "its",
    "it",
    "their",
    "will",
    "new",
    "more",
    "less",
    "over",
    "across",
    "within",
    "between",
    "about",
    "after",
    "before",
    "than",
    "via",
    "per",
}

PESTEL_DOMAINS = (
    "political",
    "economic",
    "social",
    "technological",
    "environmental",
    "legal",
)

DOMAIN_ALIASES = {
    "politics": "political",
    "policy": "legal",
    "regulatory": "legal",
    "regulation": "legal",
    "economy": "economic",
    "financial": "economic",
    "finance": "economic",
    "societal": "social",
    "demographic": "social",
    "technology": "technological",
    "tech": "technological",
    "environment": "environmental",
    "climate": "environmental",
    "law": "legal",
}


def run_cli(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        ["python", "-m", "scenarioops.app.main", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output.strip()


def load_latest_status() -> dict[str, Any] | None:
    if not LATEST_POINTER.exists():
        return None
    try:
        return json.loads(LATEST_POINTER.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def list_runs(limit: int = RUN_LIMIT) -> list[str]:
    if not RUNS_DIR.exists():
        return []
    runs = [path for path in RUNS_DIR.iterdir() if path.is_dir()]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.name for path in runs[:limit]]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            items.append(parsed)
    return items


def _load_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _load_view_model(run_dir: Path) -> dict[str, Any]:
    view_path = run_dir / "artifacts" / "view_model.json"
    if view_path.exists():
        payload = _load_json(view_path)
        if payload:
            return payload
    return build_view_model(run_dir)


def _load_charter_artifacts(
    run_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    artifacts_dir = run_dir / "artifacts"
    charter = _load_json(artifacts_dir / "scenario_charter.json")
    if not charter:
        charter = _load_json(artifacts_dir / "charter.json")
    focal_issue = _load_json(artifacts_dir / "focal_issue.json")
    return charter, focal_issue


def _artifact_registry(run_dir: Path | None) -> dict[str, Any] | None:
    if not run_dir or not run_dir.exists():
        return None
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.exists():
        return {"run_id": run_dir.name, "artifacts": []}
    artifacts = sorted(
        [path.name for path in artifacts_dir.iterdir() if path.is_file()]
    )
    return {"run_id": run_dir.name, "artifacts": artifacts}


def _group_drivers_by_domain(drivers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in drivers:
        domain = str(entry.get("category") or entry.get("domain") or "Other")
        grouped.setdefault(domain, []).append(entry)
    return {key: grouped[key] for key in sorted(grouped.keys())}


def _normalize_domain(domain: str | None) -> str | None:
    if not domain:
        return None
    value = str(domain).strip().lower()
    if value in PESTEL_DOMAINS:
        return value
    return DOMAIN_ALIASES.get(value)


def _normalize_text(value: Any | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _parse_horizon_months(value: Any | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    return int(match.group())


def _domain_counts(forces: list[dict[str, Any]]) -> dict[str, int]:
    counts = {domain: 0 for domain in PESTEL_DOMAINS}
    for force in forces:
        raw_domain = force.get("domain") or force.get("category")
        domain = _normalize_domain(raw_domain)
        if domain:
            counts[domain] += 1
    return counts


def _washout_pass(
    forces: list[dict[str, Any]],
    washout_report: dict[str, Any] | None,
    *,
    min_total: int = 30,
    min_per_domain: int = 5,
    max_duplicate_ratio: float = 0.2,
) -> bool:
    if not forces or len(forces) < min_total:
        return False
    counts = _domain_counts(forces)
    if any(count < min_per_domain for count in counts.values()):
        return False
    for force in forces:
        citations = force.get("citations") if isinstance(force, dict) else None
        if not isinstance(citations, list) or not citations:
            return False
    duplicate_ratio = None
    if isinstance(washout_report, dict):
        duplicate_ratio = washout_report.get("duplicate_ratio")
        missing_categories = washout_report.get("missing_categories", [])
        if isinstance(missing_categories, list) and missing_categories:
            return False
    if not isinstance(duplicate_ratio, (int, float)):
        return False
    return float(duplicate_ratio) <= max_duplicate_ratio


def _keyword_weights(drivers: list[dict[str, Any]]) -> list[tuple[str, float]]:
    weights: dict[str, float] = {}
    for entry in drivers:
        confidence = entry.get("confidence")
        weight = float(confidence) if isinstance(confidence, (int, float)) else 1.0
        text = " ".join(
            str(entry.get(field, ""))
            for field in ("name", "description", "domain", "category", "trend", "why_it_matters")
        )
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text.lower())
        for token in tokens:
            if token in STOPWORDS:
                continue
            weights[token] = weights.get(token, 0.0) + weight
    ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    return ranked[:40]


def _plot_word_map(drivers: list[dict[str, Any]]) -> None:
    keywords = _keyword_weights(drivers)
    if not keywords:
        st.info("No driver keywords available yet.")
        return
    rng = random.Random(RANDOM_SEED)
    rows = []
    for word, weight in keywords:
        rows.append(
            {
                "word": word,
                "weight": weight,
                "x": rng.random(),
                "y": rng.random(),
                "size": 12 + weight * 8,
            }
        )
    df = pd.DataFrame(rows)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["x"],
                y=df["y"],
                mode="markers+text",
                text=df["word"],
                textposition="top center",
                marker={"size": df["size"], "color": df["weight"], "colorscale": "Blues"},
            )
        ]
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=420,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    st.plotly_chart(fig, width="stretch")


def _plot_uncertainties(
    uncertainties: list[dict[str, Any]], axis_ids: set[str]
) -> None:
    if not uncertainties:
        st.info("No uncertainties available yet.")
        return
    rows = []
    for entry in uncertainties:
        uncertainty_score = entry.get("volatility") or entry.get("uncertainty_score") or 0
        impact = entry.get("criticality") or entry.get("impact") or 0
        rows.append(
            {
                "id": entry.get("id", ""),
                "label": entry.get("name") or entry.get("title") or entry.get("id"),
                "uncertainty_score": float(uncertainty_score),
                "impact": float(impact),
                "axis": "Axis" if entry.get("id") in axis_ids else "Other",
            }
        )
    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="uncertainty_score",
        y="impact",
        size="impact",
        color="axis",
        text="label",
        color_discrete_map={"Axis": "#d62728", "Other": "#1f77b4"},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="Uncertainty score",
        yaxis_title="Impact",
        height=420,
        legend_title=None,
    )
    st.plotly_chart(fig, width="stretch")


def _narrative_map(narratives: list[dict[str, str]]) -> dict[str, str]:
    return {entry.get("scenario_id", ""): entry.get("markdown", "") for entry in narratives}


def _premise_bullets(scenario: dict[str, Any], narrative: str | None) -> list[str]:
    text = ""
    if narrative:
        text = narrative
    else:
        text = str(scenario.get("logic") or scenario.get("summary") or scenario.get("narrative") or "")
    sentences = [part.strip() for part in re.split(r"[.!?]\s+", text) if part.strip()]
    return sentences[:2] if sentences else ["No premise available yet."]


def _top_ewis(ewis: list[dict[str, Any]], scenario_id: str) -> list[str]:
    matched = [
        entry.get("name", "")
        for entry in ewis
        if scenario_id in (entry.get("linked_scenarios") or [])
    ]
    return [item for item in matched if item][:3]


def _axis_uncertainty_ids(scenario_logic: dict[str, Any]) -> set[str]:
    axes = scenario_logic.get("axes", []) if isinstance(scenario_logic, dict) else []
    return {axis.get("uncertainty_id") for axis in axes if axis.get("uncertainty_id")}


def _top_uncertainties(uncertainties: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    def score(entry: dict[str, Any]) -> float:
        return float(entry.get("criticality") or entry.get("impact") or 0)

    return sorted(uncertainties, key=score, reverse=True)[:limit]


def _citation_domains(citations: list[dict[str, Any]]) -> set[str]:
    domains = set()
    for citation in citations:
        url = citation.get("url", "")
        parsed = urlparse(url)
        if parsed.hostname:
            domains.add(parsed.hostname)
    return domains


def _render_network_graph(drivers: list[dict[str, Any]]) -> None:
    try:
        from pyvis.network import Network  # type: ignore
    except Exception:
        return

    if not drivers:
        return
    if not st.checkbox("Show driver network graph", value=False):
        return

    net = Network(height="480px", width="100%", bgcolor="#ffffff", font_color="#222222")
    for driver in drivers:
        node_id = driver.get("id") or driver.get("name")
        label = driver.get("name") or driver.get("id") or "driver"
        group = driver.get("category") or driver.get("domain") or "Other"
        net.add_node(node_id, label=label, group=group)

    for idx, left in enumerate(drivers):
        for right in drivers[idx + 1 :]:
            left_id = left.get("id") or left.get("name")
            right_id = right.get("id") or right.get("name")
            if not left_id or not right_id:
                continue
            shared_domain = (left.get("category") or left.get("domain") or "") == (
                right.get("category") or right.get("domain") or ""
            )
            left_citations = _citation_domains(left.get("citations") or [])
            right_citations = _citation_domains(right.get("citations") or [])
            shared_citation = bool(left_citations & right_citations)
            if shared_domain or shared_citation:
                net.add_edge(left_id, right_id)

    html = net.generate_html()
    st.components.v1.html(html, height=520, scrolling=False)


st.set_page_config(page_title="ScenarioOps", layout="wide")
st.title("ScenarioOps Live")

scope = st.selectbox("Scope", ["world", "region", "country"], index=0)
value = st.text_input("Value", "UAE")
horizon = st.slider("Horizon (months)", 6, 60, 24)

if "action_logs" not in st.session_state:
    st.session_state["action_logs"] = {}

action_cols = st.columns(3)
with action_cols[0]:
    if st.button("Build Scenarios"):
        rc, out = run_cli(
            ["build-scenarios", "--scope", scope, "--value", value, "--horizon", str(horizon)]
        )
        st.session_state["action_logs"]["Build Scenarios"] = {"rc": rc, "out": out}
with action_cols[1]:
    if st.button("Export View Model"):
        rc, out = run_cli(["export-view"])
        st.session_state["action_logs"]["Export View Model"] = {"rc": rc, "out": out}
with action_cols[2]:
    if st.button("Run Daily Update"):
        rc, out = run_cli(["run-daily"])
        st.session_state["action_logs"]["Run Daily Update"] = {"rc": rc, "out": out}

for label, payload in st.session_state["action_logs"].items():
    with st.expander(f"{label} logs", expanded=False):
        st.code(payload.get("out") or "(no output)")
        if payload.get("rc") == 0:
            st.success("Command completed successfully.")
        else:
            st.error("Command failed.")

latest_status = load_latest_status() or {}
run_ids = list_runs()
default_run = latest_status.get("run_id")
selected_run = None
if run_ids:
    default_index = run_ids.index(default_run) if default_run in run_ids else 0
    selected_run = st.selectbox("Run", run_ids, index=default_index)

if latest_status.get("status") == "OK":
    st.success("Latest run status: OK")
elif latest_status.get("status") == "FAIL":
    st.error("Latest run status: FAIL")
else:
    st.info("Latest run status: unknown")

if latest_status.get("error_summary"):
    st.warning(f"Error summary: {latest_status['error_summary']}")

view_model: dict[str, Any] = {}
run_dir: Path | None = None
charter: dict[str, Any] | None = None
focal_issue: dict[str, Any] | None = None
if selected_run:
    run_dir = RUNS_DIR / selected_run
    if run_dir.exists():
        view_model = _load_view_model(run_dir)
        charter, focal_issue = _load_charter_artifacts(run_dir)

charter_payload = charter or {}
focal_issue_payload = focal_issue or {}
driving_forces = view_model.get("driving_forces") or []
washout_report = view_model.get("washout_report") or {}
evidence_units = view_model.get("evidence_units") or []
belief_sets = view_model.get("belief_sets") or []
effects = view_model.get("effects") or []
drivers = view_model.get("drivers") or []
drivers_by_domain = view_model.get("drivers_by_domain") or _group_drivers_by_domain(drivers)
uncertainties = view_model.get("uncertainties") or []
scenario_logic = view_model.get("scenario_logic") or {}
scenarios = view_model.get("scenarios") or []
narratives = view_model.get("narratives") or []
ewis = view_model.get("ewis") or []
daily_brief_md = view_model.get("daily_brief_md")
force_entries = driving_forces if driving_forces else drivers
force_groups = _group_drivers_by_domain(force_entries)
artifact_registry = _artifact_registry(run_dir)
if artifact_registry:
    exp = st.expander("Debug: Artifacts", expanded=False)
    exp.json(artifact_registry)

tabs = st.tabs(
    [
        "Overview",
        "Driving Forces",
        "Critical Uncertainties",
        "Brainstorm Map (EBE)",
        "Scenario Logic (2x2)",
        "Scenarios",
        "Daily Brief",
    ]
)

with tabs[0]:
    st.write(f"Viewing run: {selected_run or 'None'}")
    run_scope = charter_payload.get("scope")
    run_value = charter_payload.get("title") or charter_payload.get("value")
    run_horizon = charter_payload.get("time_horizon")
    st.write(
        "Run charter: "
        f"{run_scope or 'N/A'}/{run_value or 'N/A'}/{run_horizon or 'N/A'}"
    )
    build_horizon_months = horizon
    run_horizon_months = _parse_horizon_months(run_horizon)
    mismatch = False
    if run_scope and _normalize_text(run_scope) != _normalize_text(scope):
        mismatch = True
    if run_value and _normalize_text(run_value) != _normalize_text(value):
        mismatch = True
    if run_horizon_months is not None:
        mismatch = mismatch or run_horizon_months != build_horizon_months
    elif run_horizon and _normalize_text(run_horizon) != _normalize_text(f"{horizon} months"):
        mismatch = True
    if mismatch:
        st.warning(
            "Build controls affect the next run only and do not change the selected run."
        )

    st.subheader("Charter Summary")
    domains = ", ".join(force_groups.keys()) if force_groups else "None"
    if charter_payload:
        st.write(f"Scope: {run_scope or 'N/A'}")
        st.write(f"Value: {run_value or 'N/A'}")
        st.write(f"Horizon: {run_horizon or 'N/A'}")
        st.write(f"Domains: {domains}")
    else:
        st.write("No charter found for this run.")

    if focal_issue_payload:
        st.subheader("Focal Issue")
        focal_text = focal_issue_payload.get("focal_issue")
        if focal_text:
            st.write(focal_text)
        decision_type = focal_issue_payload.get("decision_type")
        if decision_type:
            st.write(f"Decision type: {decision_type}")
        success_criteria = focal_issue_payload.get("success_criteria")
        if success_criteria:
            st.write(f"Success criteria: {success_criteria}")
        exclusions = focal_issue_payload.get("exclusions") or []
        if isinstance(exclusions, list) and exclusions:
            st.write(f"Exclusions: {', '.join(exclusions)}")

    st.subheader("Counts")
    st.write(f"Driving forces: {len(force_entries)}")
    st.write(f"Uncertainties: {len(uncertainties)}")
    st.write(f"Scenarios: {len(scenarios)}")
    st.write(f"EWIs: {len(ewis)}")

    st.subheader("Exploration")
    domain_counts = _domain_counts(force_entries)
    coverage = ", ".join([f"{domain}:{domain_counts[domain]}" for domain in PESTEL_DOMAINS])
    duplicate_ratio = washout_report.get("duplicate_ratio")
    washout_status = "PASS" if _washout_pass(force_entries, washout_report) else "FAIL"
    st.write(f"Forces: {len(force_entries)}")
    st.write(f"Per-domain coverage: {coverage if force_entries else 'N/A'}")
    if isinstance(duplicate_ratio, (int, float)):
        st.write(f"Duplicates ratio: {duplicate_ratio:.2f}")
    else:
        st.write("Duplicates ratio: N/A")
    st.write(f"Washout: {washout_status}")

    updated_at = latest_status.get("updated_at")
    st.write(f"Last updated: {updated_at or 'N/A'}")

with tabs[1]:
    st.subheader("Clustered Driving Forces")
    if not force_groups:
        st.info("No driving forces available yet.")
    for domain, entries in force_groups.items():
        with st.expander(domain, expanded=False):
            for entry in entries:
                st.markdown(f"**{entry.get('name', 'Force')}**")
                st.write(entry.get("description", ""))
                why = entry.get("why_it_matters")
                if why:
                    st.write(f"Why it matters: {why}")
                citations = entry.get("citations") or []
                if citations:
                    st.caption(", ".join([c.get("url", "") for c in citations if c.get("url")]))

    st.subheader("Force Word Map")
    _plot_word_map(force_entries)
    _render_network_graph(force_entries)

with tabs[2]:
    st.subheader("Impact vs Uncertainty")
    axis_ids = _axis_uncertainty_ids(scenario_logic)
    _plot_uncertainties(uncertainties, axis_ids)

with tabs[3]:
    st.subheader("Evidence -> Beliefs -> Effects")
    evidence_by_id = {
        entry.get("id"): entry for entry in evidence_units if entry.get("id")
    }
    effects_by_belief: dict[str, list[dict[str, Any]]] = {}
    for effect in effects:
        belief_id = effect.get("belief_id")
        if belief_id:
            effects_by_belief.setdefault(belief_id, []).append(effect)

    if not belief_sets:
        st.info("No belief sets available yet.")
    for belief_set in belief_sets:
        uncertainty_label = belief_set.get("uncertainty_id", "Uncertainty")
        st.markdown(f"### {uncertainty_label}")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Evidence**")
            evidence_ids: list[str] = []
            dominant = belief_set.get("dominant_belief") or {}
            counter = belief_set.get("counter_belief") or {}
            for belief in (dominant, counter):
                for evidence_id in belief.get("evidence_ids", []) or []:
                    if evidence_id not in evidence_ids:
                        evidence_ids.append(evidence_id)
            if not evidence_ids:
                st.write("No evidence linked.")
            for evidence_id in evidence_ids:
                evidence = evidence_by_id.get(evidence_id)
                if not evidence:
                    continue
                statement = evidence.get("statement") or evidence_id
                detail_parts = [
                    evidence.get("actor"),
                    evidence.get("variable"),
                    evidence.get("direction"),
                    evidence.get("time_signal"),
                ]
                details = ", ".join([part for part in detail_parts if part])
                if details:
                    st.write(f"{statement} ({details})")
                else:
                    st.write(statement)
                citations = evidence.get("citations") or []
                if citations:
                    st.caption(", ".join([c.get("url", "") for c in citations if c.get("url")]))
        with cols[1]:
            st.markdown("**Beliefs**")
            dominant = belief_set.get("dominant_belief") or {}
            counter = belief_set.get("counter_belief") or {}
            if dominant:
                st.write(f"Dominant: {dominant.get('statement', '')}")
                assumptions = dominant.get("assumptions") or []
                if assumptions:
                    st.caption(f"Assumptions: {', '.join(assumptions)}")
            if counter:
                st.write(f"Counter: {counter.get('statement', '')}")
                assumptions = counter.get("assumptions") or []
                if assumptions:
                    st.caption(f"Assumptions: {', '.join(assumptions)}")
        with cols[2]:
            st.markdown("**Effects**")
            for label, belief in (("Dominant", dominant), ("Counter", counter)):
                belief_id = belief.get("id")
                belief_effects = effects_by_belief.get(belief_id, [])
                if belief_effects:
                    st.write(f"{label}:")
                    for effect in belief_effects[:3]:
                        order = effect.get("order")
                        description = effect.get("description") or ""
                        suffix = f" (order {order})" if order else ""
                        st.write(f"- {description}{suffix}")
                else:
                    st.write(f"{label}: No effects.")
        with st.expander("Raw JSON", expanded=False):
            st.json(belief_set)

with tabs[4]:
    st.subheader("Axes")
    axes = scenario_logic.get("axes", []) if isinstance(scenario_logic, dict) else []
    if len(axes) >= 2:
        axis_a = axes[0]
        axis_b = axes[1]
        st.write(f"Axis A: {axis_a.get('low', 'low')} <-> {axis_a.get('high', 'high')}")
        st.write(f"Axis B: {axis_b.get('low', 'low')} <-> {axis_b.get('high', 'high')}")
    else:
        st.info("Scenario logic axes are not available yet.")

    st.subheader("2x2 Matrix")
    grid = [None, None, None, None]
    for idx, scenario in enumerate(scenarios[:4]):
        grid[idx] = scenario

    row1 = st.columns(2)
    row2 = st.columns(2)
    rows = [row1, row2]
    for idx, cell in enumerate(grid):
        col = rows[0][idx] if idx < 2 else rows[1][idx - 2]
        with col:
            if cell:
                st.markdown(f"**{cell.get('name', 'Scenario')}**")
                premise = cell.get("logic") or cell.get("summary") or cell.get("narrative") or ""
                st.write(premise if premise else "No premise available.")
            else:
                st.write("No scenario.")

with tabs[5]:
    narrative_by_id = _narrative_map(narratives)
    if not scenarios:
        st.info("No scenarios available yet.")
    for scenario in scenarios:
        scenario_id = scenario.get("id", "")
        name = scenario.get("name", scenario_id or "Scenario")
        with st.container():
            st.subheader(name)
            bullets = _premise_bullets(scenario, narrative_by_id.get(scenario_id))
            st.markdown("**Premise**")
            for bullet in bullets:
                st.markdown(f"- {bullet}")

            st.markdown("**Operating Rules**")
            operating_rules = scenario.get("operating_rules") or {}
            if operating_rules:
                for key, value in operating_rules.items():
                    st.write(f"{key}: {value}")
            else:
                st.write("No operating rules available.")

            winners = scenario.get("winners") or []
            losers = scenario.get("losers") or []
            st.markdown("**Winners / Losers**")
            st.write(f"Winners: {', '.join(winners) if winners else 'N/A'}")
            st.write(f"Losers: {', '.join(losers) if losers else 'N/A'}")

            st.markdown("**Top EWIs**")
            top_ewis = _top_ewis(ewis, scenario_id)
            if top_ewis:
                for ewi in top_ewis:
                    st.write(f"- {ewi}")
            else:
                st.write("No linked EWIs.")

            with st.expander("Narrative", expanded=False):
                narrative = narrative_by_id.get(scenario_id) or scenario.get("narrative")
                st.markdown(narrative or "No narrative available yet.")

with tabs[6]:
    st.subheader("Daily Brief")
    if daily_brief_md:
        st.markdown(daily_brief_md)
    else:
        st.info("No daily brief yet.")
    if st.button("Run Daily Update", key="run_daily_in_tab"):
        rc, out = run_cli(["run-daily"])
        st.session_state["action_logs"]["Run Daily Update"] = {"rc": rc, "out": out}
