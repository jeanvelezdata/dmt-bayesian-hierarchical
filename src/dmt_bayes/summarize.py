import arviz as az
import pandas as pd
from pathlib import Path


def save_summary(idata, outdir="results"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    summary = az.summary(idata, hdi_prob=0.95)
    summary.to_csv(out / "summary.csv")

    idata.to_netcdf(out / "idata.nc")