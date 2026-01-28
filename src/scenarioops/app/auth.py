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
    is_admin: bool = False

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
                role text,
                api_key_hash text not null,
                created_at timestamptz
            );
            """
        )
        cur.execute("alter table scenarioops_users add column if not exists role text;")
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
                select user_id, tenant_id, email, role
                from scenarioops_users
                where api_key_hash = %s
                limit 1;
                """,
                (_hash_key(api_key),),
            )
            row = cur.fetchone()
        if not row:
            return None
        is_admin = str(row[3] or "").lower() in {"admin", "owner"}
        return TenantContext(tenant_id=row[1], user_id=row[0], email=row[2], is_admin=is_admin)
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
                insert into scenarioops_users (user_id, tenant_id, email, role, api_key_hash, created_at)
                values (%s, %s, %s, %s, %s, %s)
                on conflict (user_id) do nothing;
                """,
                (
                    f"{default_tenant}-owner",
                    default_tenant,
                    None,
                    "owner",
                    _hash_key(default_key),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _admin_keys() -> set[str]:
    raw = os.environ.get("SCENARIOOPS_ADMIN_KEYS", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def resolve_tenant(api_key: str | None) -> TenantContext:
    if not os.environ.get("SCENARIOOPS_AUTH_REQUIRED"):
        is_admin = True
        if api_key and _admin_keys():
            is_admin = api_key in _admin_keys()
        return TenantContext(
            tenant_id=os.environ.get("SCENARIOOPS_DEFAULT_TENANT", "public"),
            is_admin=is_admin,
        )
    if not os.environ.get("DATABASE_URL"):
        raise RuntimeError("Auth required but DATABASE_URL not configured.")
    if not api_key:
        raise PermissionError("Missing API key.")
    user = _lookup_user(api_key)
    if user is None:
        raise PermissionError("Invalid API key.")
    if _admin_keys() and api_key in _admin_keys():
        return TenantContext(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            email=user.email,
            is_admin=True,
        )
    return user


__all__ = ["TenantContext", "ensure_default_user", "resolve_tenant"]
