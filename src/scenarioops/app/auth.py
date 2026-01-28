from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    user_id: str | None = None
    email: str | None = None

    @property
    def base_dir(self) -> Path:
        return Path("storage") / "runs" / self.tenant_id


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _db_connect():
    try:
        import psycopg  # type: ignore
    except Exception as exc:
        raise RuntimeError("psycopg is required for auth storage.") from exc
    return psycopg.connect(os.environ["DATABASE_URL"])


def _db_init(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            create table if not exists scenarioops_users (
                user_id text primary key,
                tenant_id text not null,
                email text,
                api_key_hash text not null,
                created_at timestamptz
            );
            """
        )
    conn.commit()


def _lookup_user(api_key: str) -> TenantContext | None:
    if not api_key:
        return None
    conn = _db_connect()
    try:
        _db_init(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                select user_id, tenant_id, email
                from scenarioops_users
                where api_key_hash = %s
                limit 1;
                """,
                (_hash_key(api_key),),
            )
            row = cur.fetchone()
        if not row:
            return None
        return TenantContext(tenant_id=row[1], user_id=row[0], email=row[2])
    finally:
        conn.close()


def ensure_default_user() -> None:
    if not os.environ.get("DATABASE_URL"):
        return
    default_key = os.environ.get("SCENARIOOPS_DEFAULT_API_KEY")
    default_tenant = os.environ.get("SCENARIOOPS_DEFAULT_TENANT", "public")
    if not default_key:
        return
    conn = _db_connect()
    try:
        _db_init(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into scenarioops_users (user_id, tenant_id, email, api_key_hash, created_at)
                values (%s, %s, %s, %s, %s)
                on conflict (user_id) do nothing;
                """,
                (
                    f"{default_tenant}-owner",
                    default_tenant,
                    None,
                    _hash_key(default_key),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def resolve_tenant(api_key: str | None) -> TenantContext:
    if not os.environ.get("SCENARIOOPS_AUTH_REQUIRED"):
        return TenantContext(
            tenant_id=os.environ.get("SCENARIOOPS_DEFAULT_TENANT", "public")
        )
    if not os.environ.get("DATABASE_URL"):
        raise RuntimeError("Auth required but DATABASE_URL not configured.")
    if not api_key:
        raise PermissionError("Missing API key.")
    user = _lookup_user(api_key)
    if user is None:
        raise PermissionError("Invalid API key.")
    return user


__all__ = ["TenantContext", "ensure_default_user", "resolve_tenant"]
