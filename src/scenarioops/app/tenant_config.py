from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping


def _db_connect():
    try:
        import psycopg  # type: ignore
    except Exception as exc:
        raise RuntimeError("psycopg is required for tenant config storage.") from exc
    return psycopg.connect(os.environ["DATABASE_URL"])


def _db_init(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            create table if not exists scenarioops_tenant_config (
                tenant_id text primary key,
                settings jsonb,
                updated_at timestamptz
            );
            """
        )
    conn.commit()


def get_tenant_config(tenant_id: str) -> dict[str, Any]:
    if not os.environ.get("DATABASE_URL"):
        return {}
    conn = _db_connect()
    try:
        _db_init(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                select settings
                from scenarioops_tenant_config
                where tenant_id = %s
                limit 1;
                """,
                (tenant_id,),
            )
            row = cur.fetchone()
        if not row or row[0] is None:
            return {}
        if isinstance(row[0], dict):
            return dict(row[0])
        return {}
    finally:
        conn.close()


def update_tenant_config(tenant_id: str, settings: Mapping[str, Any]) -> None:
    if not os.environ.get("DATABASE_URL"):
        return
    conn = _db_connect()
    try:
        _db_init(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into scenarioops_tenant_config (tenant_id, settings, updated_at)
                values (%s, %s, %s)
                on conflict (tenant_id) do update set
                    settings = excluded.settings,
                    updated_at = excluded.updated_at;
                """,
                (
                    tenant_id,
                    dict(settings),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        conn.commit()
    finally:
        conn.close()


__all__ = ["get_tenant_config", "update_tenant_config"]
