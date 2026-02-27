from __future__ import annotations

from pathlib import Path
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ensure_dirs(outdir: str | Path):
    outdir = Path(outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    return outdir


def write_core_tables(idata, outdir: str | Path):
    outdir = ensure_dirs(outdir)

    # Full parameter summary (ArviZ)
    summ = az.summary(idata, hdi_prob=0.95)
    summ.to_csv(outdir / "tables" / "posterior_summary.csv")

    # Extract measure-specific effects if present
    if "beta_m" in idata.posterior:
        beta = idata.posterior["beta_m"]  # dims: chain, draw, measure
        beta_flat = beta.stack(sample=("chain", "draw")).values  # (measure, sample) or (sample, measure) depending
        # Ensure shape = (n_meas, n_samp)
        if beta_flat.shape[0] < beta_flat.shape[1]:
            beta_flat = beta_flat
        else:
            beta_flat = beta_flat.T

        rows = []
        for j in range(beta_flat.shape[0]):
            vals = beta_flat[j, :]
            rows.append(
                {
                    "measure_index": j,
                    "mean": float(np.mean(vals)),
                    "hdi_2.5%": float(np.quantile(vals, 0.025)),
                    "hdi_97.5%": float(np.quantile(vals, 0.975)),
                }
            )
        pd.DataFrame(rows).to_csv(outdir / "tables" / "beta_m_effects.csv", index=False)

    return outdir


def plot_trace(idata, outdir: str | Path):
    outdir = ensure_dirs(outdir)
    ax = az.plot_trace(idata)
    fig = ax.ravel()[0].figure
    fig.savefig(outdir / "figures" / "trace.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_forest_beta(idata, outdir: str | Path):
    outdir = ensure_dirs(outdir)
    if "beta_m" not in idata.posterior:
        return

    ax = az.plot_forest(idata, var_names=["beta_m"], combined=True, hdi_prob=0.95)
    fig = ax.ravel()[0].figure
    fig.savefig(outdir / "figures" / "beta_m_forest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)