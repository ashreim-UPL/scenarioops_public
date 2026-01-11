import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from scenarioops.graph.tools.web_retriever import retrieve_url


def _seed_cache(cache_dir: Path, url: str, text: str) -> None:
    payload = {
        "url": url,
        "title": "Cached Source",
        "date": "2026-01-01T00:00:00+00:00",
        "text": text,
        "excerpt_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_retriever_uses_cache_when_network_disabled(tmp_path: Path) -> None:
    url = "https://example.com/a"
    cache_dir = tmp_path / "cache"
    text = "Cached content for example."
    _seed_cache(cache_dir, url, text)

    result = retrieve_url(
        url,
        run_id="run-cache",
        base_dir=tmp_path / "runs",
        cache_dir=cache_dir,
        allow_network=False,
    )

    assert result.url == url
    assert result.text == text
