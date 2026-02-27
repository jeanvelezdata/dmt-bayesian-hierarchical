import subprocess
import sys
from pathlib import Path


def test_cli_runs_fast_config(tmp_path):
    # run from repo root in CI; ensure results go to a temp folder
    cfg = Path("configs/fast.yaml")
    assert cfg.exists()

    outdir = tmp_path / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    # Run CLI via python -m to avoid Windows PATH issues in CI
    cmd = [
        sys.executable,
        "-m",
        "dmt_bayes.cli",
        "run",
        "--config",
        str(cfg),
    ]

    # Set an env var to redirect results if you add support later.
    # For now, we just run and then check default results paths.
    subprocess.run(cmd, check=True)

    # Check expected artifacts in default location
    # (If you want this to be isolated, we can add output override in Step 5.)
    assert Path("results/summary.csv").exists()
    assert Path("results/idata.nc").exists()