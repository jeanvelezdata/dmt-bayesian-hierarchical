import arviz as az
from pathlib import Path


def save_summary(idata, outdir="results", save_idata=True, save_summary=True):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if save_summary:
        summary = az.summary(idata, hdi_prob=0.95)
        summary.to_csv(out / "summary.csv")

    if save_idata:
        idata.to_netcdf(out / "idata.nc")