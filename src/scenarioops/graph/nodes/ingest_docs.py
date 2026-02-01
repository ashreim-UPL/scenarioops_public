from __future__ import annotations

import csv
import hashlib
import mimetypes
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scenarioops.app.config import LLMConfig, ScenarioOpsSettings, llm_config_from_settings
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.evidence_processing import fallback_summary, summarize_text
from scenarioops.graph.tools.schema_validate import validate_artifact
from scenarioops.graph.tools.storage import (
    ensure_run_dirs,
    read_run_json,
    update_run_json,
    write_artifact,
    write_bytes,
    write_text,
)
from scenarioops.graph.tools.traceability import build_run_metadata
from scenarioops.graph.tools.vectordb import open_run_vector_store
from scenarioops.llm.client import get_llm_client


_CHUNK_CHARS = int(os.environ.get("INGEST_CHUNK_CHARS", "1800"))
_CHUNK_OVERLAP = int(os.environ.get("INGEST_CHUNK_OVERLAP", "200"))
_MAX_CHUNKS = int(os.environ.get("INGEST_MAX_CHUNKS", "20"))


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return cleaned.strip("_") or "document"


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _detect_content_type(path: Path) -> str | None:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or None


def _chunk_text(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    overlap = min(_CHUNK_OVERLAP, max(_CHUNK_CHARS - 1, 0))
    step = max(1, _CHUNK_CHARS - overlap)
    chunks: list[str] = []
    for start in range(0, len(cleaned), step):
        if len(chunks) >= _MAX_CHUNKS:
            break
        chunk = cleaned[start : start + _CHUNK_CHARS]
        if chunk:
            chunks.append(chunk)
    return chunks


def _read_text_file(path: Path) -> tuple[str, str | None]:
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except Exception as exc:
        return "", f"text_read_failed:{exc}"


def _read_csv(path: Path) -> tuple[str, str | None]:
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            rows = ["\t".join(row) for row in reader if row]
        return "\n".join(rows), None
    except Exception as exc:
        return "", f"csv_read_failed:{exc}"


def _read_pdf(path: Path) -> tuple[str, str | None]:
    try:
        from pypdf import PdfReader
    except Exception:
        return "", "pdf_parser_unavailable"
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        return "", f"pdf_read_failed:{exc}"
    parts: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            parts.append(page_text)
    text = "\n".join(parts)
    if not text.strip():
        return "", "pdf_text_empty"
    return text, None


def _read_docx(path: Path) -> tuple[str, str | None]:
    try:
        import docx
    except Exception:
        return "", "docx_parser_unavailable"
    try:
        document = docx.Document(str(path))
    except Exception as exc:
        return "", f"docx_read_failed:{exc}"
    parts = [para.text for para in document.paragraphs if para.text]
    text = "\n".join(parts)
    if not text.strip():
        return "", "docx_text_empty"
    return text, None


def _extract_text(path: Path) -> tuple[str, str | None]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _read_text_file(path)
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    return "", "unsupported_extension"


def _dedupe_units(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        unit_id = str(unit.get("id") or unit.get("evidence_unit_id") or "")
        if not unit_id or unit_id in seen:
            continue
        seen.add(unit_id)
        deduped.append(dict(unit))
    return deduped


def run_ingest_docs_node(
    doc_paths: Sequence[str],
    *,
    run_id: str,
    state: ScenarioOpsState,
    user_params: Mapping[str, Any],
    base_dir: Path | None = None,
    settings: ScenarioOpsSettings | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    existing_units: list[Mapping[str, Any]] = []
    if state.evidence_units:
        prior_units = state.evidence_units.get("evidence_units")
        if isinstance(prior_units, list):
            existing_units = prior_units

    doc_list = [str(path) for path in (doc_paths or []) if path]
    if not doc_list and not existing_units:
        return state

    resolved_settings = settings or ScenarioOpsSettings()
    resolved_config = config or llm_config_from_settings(resolved_settings)
    summarizer_config = LLMConfig(
        model_name=resolved_settings.summarizer_model,
        temperature=resolved_config.temperature,
        timeouts=resolved_config.timeouts,
        mode=resolved_config.mode,
    )
    summarizer_client = get_llm_client(summarizer_config)

    vector_store = None
    try:
        vector_store = open_run_vector_store(
            run_id,
            base_dir=base_dir,
            embed_model=resolved_settings.embed_model,
            seed=int(resolved_settings.seed or 0),
        )
    except Exception:
        vector_store = None

    dirs = ensure_run_dirs(run_id, base_dir=base_dir)
    inputs_dir = dirs["inputs_dir"]
    derived_dir = dirs["derived_dir"]

    metadata = build_run_metadata(
        run_id=run_id,
        user_params=user_params,
        focal_issue=state.focal_issue if isinstance(state.focal_issue, Mapping) else None,
    )
    geography = metadata.get("geography", "")
    retrieved_at = datetime.now(timezone.utc).isoformat()

    evidence_units: list[dict[str, Any]] = []
    unit_counter = 1

    def _next_unit_id(prefix: str, file_hash: str) -> str:
        nonlocal unit_counter
        unit_id = f"{prefix}-{file_hash[:10]}-{unit_counter}"
        unit_counter += 1
        return unit_id

    for raw_path in doc_list:
        if not raw_path:
            continue
        source_path = Path(raw_path)
        if not source_path.exists():
            unit_id = _next_unit_id("upl", "missing")
            evidence_units.append(
                {
                    "id": unit_id,
                    "evidence_unit_id": unit_id,
                    "url": None,
                    "title": raw_path,
                    "status": "failed",
                    "failure_reason": "file_missing",
                    "summary": "",
                    "claims": [],
                    "metrics": [],
                    "tags": [],
                    "retrieved_at": retrieved_at,
                    "content_type": None,
                    "http_status": None,
                    "source_method": "upload",
                    "embedding_ref": None,
                    "raw_text": "",
                    "excerpt": "",
                    "publisher": "upload",
                    "date_published": retrieved_at,
                    "source_type": "primary",
                    "reliability_grade": "C",
                    "reliability_reason": "upload failed",
                    "reliability_notes": "File was missing at ingestion time.",
                    "geography_tags": [geography],
                    "domain_tags": [],
                    "simulated": False,
                    "file_name": source_path.name,
                    "sha256": "",
                }
            )
            continue

        try:
            file_bytes = source_path.read_bytes()
        except Exception as exc:
            unit_id = _next_unit_id("upl", "unreadable")
            evidence_units.append(
                {
                    "id": unit_id,
                    "evidence_unit_id": unit_id,
                    "url": None,
                    "title": source_path.name,
                    "status": "failed",
                    "failure_reason": f"file_read_failed:{exc}",
                    "summary": "",
                    "claims": [],
                    "metrics": [],
                    "tags": [],
                    "retrieved_at": retrieved_at,
                    "content_type": _detect_content_type(source_path),
                    "http_status": None,
                    "source_method": "upload",
                    "embedding_ref": None,
                    "raw_text": "",
                    "excerpt": "",
                    "publisher": "upload",
                    "date_published": retrieved_at,
                    "source_type": "primary",
                    "reliability_grade": "C",
                    "reliability_reason": "upload failed",
                    "reliability_notes": "File could not be read.",
                    "geography_tags": [geography],
                    "domain_tags": [],
                    "simulated": False,
                    "file_name": source_path.name,
                    "sha256": "",
                }
            )
            continue

        file_hash = _hash_bytes(file_bytes)
        safe_name = _safe_filename(source_path.stem)
        safe_suffix = source_path.suffix.lower()
        input_name = f"{safe_name}-{file_hash[:8]}{safe_suffix}"
        input_path = inputs_dir / input_name
        if not input_path.exists():
            write_bytes(input_path, file_bytes, base_dir=base_dir)

        text, extract_error = _extract_text(source_path)
        if extract_error:
            unit_id = _next_unit_id("upl", file_hash)
            evidence_units.append(
                {
                    "id": unit_id,
                    "evidence_unit_id": unit_id,
                    "url": None,
                    "title": source_path.name,
                    "status": "failed",
                    "failure_reason": extract_error,
                    "summary": "",
                    "claims": [],
                    "metrics": [],
                    "tags": [],
                    "retrieved_at": retrieved_at,
                    "content_type": _detect_content_type(source_path),
                    "http_status": None,
                    "source_method": "upload",
                    "embedding_ref": None,
                    "raw_text": "",
                    "excerpt": "",
                    "publisher": "upload",
                    "date_published": retrieved_at,
                    "source_type": "primary",
                    "reliability_grade": "C",
                    "reliability_reason": "upload failed",
                    "reliability_notes": "Unable to extract text from upload.",
                    "geography_tags": [geography],
                    "domain_tags": [],
                    "simulated": False,
                    "file_name": source_path.name,
                    "sha256": file_hash,
                }
            )
            continue

        derived_name = f"{safe_name}-{file_hash[:8]}.txt"
        derived_path = derived_dir / derived_name
        write_text(derived_path, text, base_dir=base_dir, content_type="text/plain")

        chunks = _chunk_text(text)
        if not chunks:
            unit_id = _next_unit_id("upl", file_hash)
            evidence_units.append(
                {
                    "id": unit_id,
                    "evidence_unit_id": unit_id,
                    "url": None,
                    "title": source_path.name,
                    "status": "failed",
                    "failure_reason": "empty_extracted_text",
                    "summary": "",
                    "claims": [],
                    "metrics": [],
                    "tags": [],
                    "retrieved_at": retrieved_at,
                    "content_type": _detect_content_type(source_path),
                    "http_status": None,
                    "source_method": "upload",
                    "embedding_ref": None,
                    "raw_text": "",
                    "excerpt": "",
                    "publisher": "upload",
                    "date_published": retrieved_at,
                    "source_type": "primary",
                    "reliability_grade": "C",
                    "reliability_reason": "upload failed",
                    "reliability_notes": "Extracted text was empty.",
                    "geography_tags": [geography],
                    "domain_tags": [],
                    "simulated": False,
                    "file_name": source_path.name,
                    "sha256": file_hash,
                }
            )
            continue

        published_at = retrieved_at
        try:
            published_at = datetime.fromtimestamp(
                source_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        except Exception:
            published_at = retrieved_at

        for index, chunk in enumerate(chunks, start=1):
            unit_id = _next_unit_id("upl", file_hash)
            title = source_path.name
            if len(chunks) > 1:
                title = f"{title} (chunk {index})"
            summary_payload, _, summary_error = summarize_text(
                client=summarizer_client,
                text=chunk,
                title=title,
                url=str(input_path),
            )
            status = "ok"
            failure_reason = None
            if summary_payload is None:
                status = "partial"
                failure_reason = f"summarizer_failed:{summary_error}"
                summary_payload = fallback_summary(chunk)

            summary = summary_payload.get("summary", "")
            claims = summary_payload.get("claims", [])
            metrics = summary_payload.get("metrics", [])
            tags = summary_payload.get("tags", [])
            reliability_notes = summary_payload.get("reliability_notes", "")

            unit = {
                "id": unit_id,
                "evidence_unit_id": unit_id,
                "url": None,
                "title": title,
                "status": status,
                "failure_reason": failure_reason,
                "summary": summary,
                "claims": claims,
                "metrics": metrics,
                "tags": tags,
                "retrieved_at": retrieved_at,
                "content_type": _detect_content_type(source_path),
                "http_status": None,
                "source_method": "upload",
                "embedding_ref": None,
                "raw_text": chunk[:4000],
                "excerpt": chunk[:300],
                "publisher": source_path.name,
                "date_published": published_at,
                "source_type": "primary",
                "reliability_grade": "B",
                "reliability_reason": "uploaded document",
                "reliability_notes": reliability_notes or "uploaded document",
                "geography_tags": [geography],
                "domain_tags": [],
                "simulated": False,
                "file_name": source_path.name,
                "sha256": file_hash,
            }
            if vector_store and summary:
                doc_id = f"upl:{unit_id}"
                unit["embedding_ref"] = doc_id
                doc = vector_store.build_document(
                    doc_id=doc_id,
                    text=summary,
                    metadata={
                        "evidence_unit": unit,
                        "company_name": metadata.get("company_name", ""),
                        "run_id": run_id,
                    },
                )
                vector_store.add_documents([doc])
            evidence_units.append(unit)

    combined_units = _dedupe_units([*existing_units, *evidence_units])
    if not combined_units:
        return state

    failed_units = [unit for unit in combined_units if unit.get("status") == "failed"]
    partial_units = [unit for unit in combined_units if unit.get("status") == "partial"]
    if failed_units or partial_units:
        warning_payload = {
            "failed": len(failed_units),
            "partial": len(partial_units),
            "total": len(combined_units),
        }
        current = read_run_json(run_id, base_dir=base_dir) or {}
        warnings = dict(current.get("warnings") or {})
        warnings["ingest_docs"] = warning_payload
        update_run_json(run_id=run_id, updates={"warnings": warnings}, base_dir=base_dir)

    payload = {
        **metadata,
        "schema_version": "2.0",
        "simulated": False,
        "evidence_units": combined_units,
    }
    validate_artifact("evidence_units.schema", payload)
    write_artifact(
        run_id=run_id,
        artifact_name="evidence_units_uploads",
        payload=payload,
        ext="json",
        input_values={
            "document_count": len(doc_list),
            "unit_count": len(combined_units),
        },
        tool_versions={"ingest_docs_node": "0.1.0"},
        base_dir=base_dir,
    )
    state.evidence_units = payload
    return state
