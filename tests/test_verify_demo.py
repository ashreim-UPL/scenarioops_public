from argparse import Namespace
import json
from pathlib import Path

from scenarioops.app.main import _run_verify


def test_verify_demo(tmp_path: Path) -> None:
    args = Namespace(
        demo=True,
        live=False,
        run_id="verify-demo",
        base_dir=str(tmp_path / "runs"),
        sources=None,
    )

    _run_verify(args)

    artifacts_dir = tmp_path / "runs" / "verify-demo" / "artifacts"
    for name in [
        "focal_issue.json",
        "driving_forces.json",
        "washout_report.json",
        "evidence_units.json",
        "certainty_uncertainty.json",
        "belief_sets.json",
        "effects.json",
    ]:
        assert (artifacts_dir / name).exists()

    report = json.loads((artifacts_dir / "washout_report.json").read_text(encoding="utf-8"))
    assert report.get("missing_categories") == []
    assert report.get("duplicate_ratio", 1.0) <= 0.2
