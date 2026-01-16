from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ORG_RE = re.compile(
    r"\b([A-Z][\w&.-]+(?:\s+[A-Z][\w&.-]+)*)\s+"
    r"(Inc|LLC|Ltd|Corp|Company|Co\.?|PLC|AG|GmbH|S\.A\.|SAS|B\.V\.|BV)\b"
)
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")


@dataclass
class ScrubberState:
    email_map: dict[str, str] = field(default_factory=dict)
    org_map: dict[str, str] = field(default_factory=dict)
    name_map: dict[str, str] = field(default_factory=dict)
    email_count: int = 0
    org_count: int = 0
    name_count: int = 0

    def replace_email(self, match: re.Match[str]) -> str:
        value = match.group(0)
        if value not in self.email_map:
            self.email_count += 1
            self.email_map[value] = f"[EMAIL_{self.email_count}]"
        return self.email_map[value]

    def replace_org(self, match: re.Match[str]) -> str:
        value = match.group(0)
        if value not in self.org_map:
            self.org_count += 1
            self.org_map[value] = f"[ORG_{self.org_count}]"
        return self.org_map[value]

    def replace_name(self, match: re.Match[str]) -> str:
        value = match.group(0)
        if value not in self.name_map:
            self.name_count += 1
            self.name_map[value] = f"[NAME_{self.name_count}]"
        return self.name_map[value]


def scrub_text(text: str, *, private_mode: bool = True) -> str:
    if not private_mode:
        return text

    state = ScrubberState()
    scrubbed = _EMAIL_RE.sub(state.replace_email, text)
    scrubbed = _ORG_RE.sub(state.replace_org, scrubbed)
    scrubbed = _NAME_RE.sub(state.replace_name, scrubbed)
    return scrubbed


def scrub_payload(payload: Any, *, private_mode: bool = True) -> Any:
    if not private_mode:
        return payload

    if isinstance(payload, str):
        return scrub_text(payload, private_mode=private_mode)
    if isinstance(payload, list):
        return [scrub_payload(item, private_mode=private_mode) for item in payload]
    if isinstance(payload, tuple):
        return tuple(scrub_payload(item, private_mode=private_mode) for item in payload)
    if isinstance(payload, dict):
        return {
            scrub_payload(key, private_mode=private_mode): scrub_payload(
                value, private_mode=private_mode
            )
            for key, value in payload.items()
        }
    return payload
