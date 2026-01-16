from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable, TypeVar

from scenarioops.graph.tools.storage import log_node_event

T = TypeVar("T")


def record_node_event(
    *,
    run_id: str,
    node_name: str,
    inputs: list[str],
    outputs: list[str],
    tools: list[str],
    base_dir: Path | None,
    schema_validated: bool = True,
    action: Callable[[], T],
) -> T:
    start = perf_counter()
    try:
        result = action()
    except Exception as exc:
        duration = perf_counter() - start
        log_node_event(
            run_id=run_id,
            node_name=node_name,
            inputs=inputs,
            outputs=outputs,
            schema_validated=False,
            duration_seconds=duration,
            base_dir=base_dir,
            error=str(exc),
            tools=tools,
            status="FAIL",
        )
        raise
    duration = perf_counter() - start
    log_node_event(
        run_id=run_id,
        node_name=node_name,
        inputs=inputs,
        outputs=outputs,
        schema_validated=schema_validated,
        duration_seconds=duration,
        base_dir=base_dir,
        tools=tools,
        status="OK",
    )
    return result
