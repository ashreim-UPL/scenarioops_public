from argparse import Namespace
from pathlib import Path

from scenarioops.app.main import _run_verify


def test_verify_demo(tmp_path: Path) -> None:
    args = Namespace(
        demo=True,
        run_id="verify-demo",
        base_dir=str(tmp_path / "runs"),
    )

    _run_verify(args)
