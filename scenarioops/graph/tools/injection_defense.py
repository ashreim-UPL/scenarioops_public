from __future__ import annotations

import re

_INSTRUCTION_PATTERNS = [
    re.compile(r"(?i)\bignore (all|previous|earlier) instructions\b"),
    re.compile(r"(?i)\bdisregard (all|previous|earlier) instructions\b"),
    re.compile(r"(?i)\b(system|developer|assistant) (prompt|message|instruction)s?\b"),
    re.compile(r"(?i)\byou are (chatgpt|an ai|an assistant|a language model)\b"),
    re.compile(r"(?i)\bact as\b"),
    re.compile(r"(?i)\brole\s*:\s*(system|developer|assistant|user)\b"),
    re.compile(r"(?i)^#{1,6}\s*(system|developer|assistant) instructions\b"),
    re.compile(r"(?i)\bdo not (answer|respond|comply)\b"),
    re.compile(r"(?i)\bprompt injection\b"),
    re.compile(r"(?i)\bBEGIN (SYSTEM|INSTRUCTIONS)\b"),
    re.compile(r"(?i)\bEND (SYSTEM|INSTRUCTIONS)\b"),
]


def strip_instruction_patterns(text: str) -> str:
    """Remove instruction-like patterns while preserving surrounding content."""
    lines = text.splitlines()
    filtered = []
    for line in lines:
        sanitized = line
        for pattern in _INSTRUCTION_PATTERNS:
            sanitized = pattern.sub("", sanitized)
        if sanitized.strip():
            filtered.append(sanitized)

    collapsed = "\n".join(filtered)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed).strip()
    return collapsed
