from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from scenarioops.llm.client import LLMClient, MockLLMClient
from scenarioops.sources.policy import default_sources_for_policy
from scenarioops.graph.tools.web_retriever import RetrievedContent

def _default_sources() -> list[str]:
    return [
        "https://example.com/ai-policy",
        "https://example.com/chips",
        "https://example.com/c"
    ]


def _mock_retriever(url: str, **_: Any) -> RetrievedContent:
    text = f"Mock source content for {url}."
    excerpt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return RetrievedContent(
        url=url,
        title="Mock Source",
        date="2026-01-01T00:00:00+00:00",
        text=text,
        excerpt_hash=excerpt_hash,
        content_type="text/plain",
        http_status=200,
    )

def mock_retriever(url: str, **kwargs: Any) -> RetrievedContent:
    return _mock_retriever(url, **kwargs)

def _mock_payloads(sources: Sequence[str]) -> dict[str, dict[str, Any]]:
    safe_sources = list(sources)
    while len(safe_sources) < 3:
        safe_sources.append(_default_sources()[len(safe_sources)])
    
    # Reconstructing minimal mock payloads structure for brevity, 
    # ensuring it matches what tests/demos might expect.
    # (Simplified version of original huge mock payload function)
    
    return {
        "charter": {"json": {
            "id": "charter-001",
            "title": "ScenarioOps Charter",
            "purpose": "Assess operational resilience.",
            "decision_context": "Resilience investment prioritization.",
            "scope": "Supply chain",
            "time_horizon": "5 years (60 months)",
            "stakeholders": ["Operations", "Finance"],
            "constraints": ["No headcount increase"],
            "assumptions": ["Stable demand"],
            "success_criteria": ["Decision-ready scenario set"],
        }},
        "focal_issue": {"json": {
            "focal_issue": "How should the organization prioritize resilience investments?",
            "scope": {"geography": "UAE", "sectors": ["supply chain"], "time_horizon_months": 60},
            "decision_type": "strategic planning",
            "exclusions": [],
            "success_criteria": "Scenarios enable resilient investment.",
        }},
        # Add other nodes as needed for a full mock run if requested
    }


def _client_for(
    node_name: str,
    *,
    default_client: LLMClient | None = None,
    mock_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> LLMClient | None:
    if not mock_payloads:
        return default_client
    payload = mock_payloads.get(node_name, {})
    json_payload = payload.get("json")
    markdown_payload = payload.get("markdown")
    if json_payload or markdown_payload:
        return MockLLMClient(json_payload=json_payload, markdown_payload=markdown_payload)
    return default_client


def default_sources() -> list[str]:
    return _default_sources()


def mock_payloads_for_sources(sources: Sequence[str]) -> dict[str, dict[str, Any]]:
    return _mock_payloads(sources)


def client_for_node(
    node_name: str,
    *,
    default_client: LLMClient | None = None,
    mock_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> LLMClient | None:
    return _client_for(
        node_name, default_client=default_client, mock_payloads=mock_payloads
    )

def apply_node_result(
    run_id: str,
    base_dir: Any,
    state: Any,
    result: Any,
) -> Any:
    # Minimal impl or import if needed. 
    # Since apply_node_result was logic, maybe it stays in build_graph or moves here.
    # It updates state.
    from scenarioops.graph.tools.storage import write_artifact
    
    if hasattr(result, "state_updates"): # Check if it's a NodeResult
        for key, value in result.state_updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        
        for artifact in result.artifacts:
            write_artifact(
                run_id=run_id,
                artifact_name=artifact.name,
                payload=artifact.payload,
                ext=artifact.ext,
                input_values=artifact.input_values,
                prompt_values=artifact.prompt_values,
                tool_versions=artifact.tool_versions,
                base_dir=base_dir,
            )
    return state
