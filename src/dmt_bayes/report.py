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

    # Full summary
    summ = az.summary(idata, hdi_prob=0.95)
    summ.to_csv(outdir / "tables" / "posterior_summary.csv")

    # Measure-labeled slope table (what reviewers actually care about)
    if "beta_m_by_measure" in idata.posterior:
        beta = idata.posterior["beta_m_by_measure"]  # dims: chain, draw, measure
        beta_df = az.summary(beta, hdi_prob=0.95).reset_index()

        # ArviZ uses "index" column names like "beta_m_by_measure[BFI-E]"
        # Keep it clean:
        beta_df = beta_df.rename(columns={"index": "parameter"})

        beta_df.to_csv(outdir / "tables" / "beta_by_measure.csv", index=False)

    return outdir


def plot_trace(idata, outdir: str | Path):
    outdir = ensure_dirs(outdir)

    # Plot only high-level parameters to avoid unreadable plots
    var_names = [
        "alpha",
        "beta_time",
        "sigma",
        "sd_u0",
        "sd_u1",
        "sigma_id",
        "corr_id",
    ]

    # Keep only variables that actually exist in the InferenceData
    existing = []
    post_vars = set(idata.posterior.data_vars)
    for v in var_names:
        if v in post_vars:
            existing.append(v)

    ax = az.plot_trace(
        idata,
        var_names=existing,
        compact=True,          # more compact layout
        combined=False,        # show chains separately
        figsize=(12, 2.2 * len(existing)),  # scale height with number of vars
    )

    fig = ax.ravel()[0].figure

    # Increase spacing so titles and ticks don't overlap
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.98,
        bottom=0.06,
        hspace=0.9,
        wspace=0.25,
    )

    fig.savefig(outdir / "figures" / "trace.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_forest_beta(idata, outdir: str | Path):
    outdir = ensure_dirs(outdir)
    if "beta_m_by_measure" not in idata.posterior:
        return

    ax = az.plot_forest(idata, var_names=["beta_m_by_measure"], combined=True, hdi_prob=0.95)
    fig = ax.ravel()[0].figure
    fig.savefig(outdir / "figures" / "beta_by_measure_forest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)