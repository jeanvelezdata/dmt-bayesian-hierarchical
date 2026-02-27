import subprocess
import sys
from pathlib import Path


def test_cli_runs_fast_config():
    subprocess.run([sys.executable, "-m", "dmt_bayes.cli", "run", "--config", "configs/fast.yaml"], check=True)

    assert Path("results/idata.nc").exists()

    subprocess.run([sys.executable, "-m", "dmt_bayes.cli", "summarize", "--idata", "results/idata.nc", "--outdir", "results"], check=True)
    subprocess.run([sys.executable, "-m", "dmt_bayes.cli", "plot", "--idata", "results/idata.nc", "--outdir", "results"], check=True)

    assert Path("results/tables/posterior_summary.csv").exists()
    assert Path("results/figures/trace.png").exists()