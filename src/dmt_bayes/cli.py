from __future__ import annotations
import arviz as az
from .report import write_core_tables, plot_trace, plot_forest_beta

import argparse
from pathlib import Path
import numpy as np
import yaml

from .download import download_file
from .data_prep import load_data, build_varmap, reshape_long, standardize_within_measure
from .model import build_model
from .fit import fit_model
from .summarize import save_summary


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_run(config_path: str) -> None:
    cfg = load_config(config_path)

    # Download data (Zenodo)
    data_cfg = cfg["data"]
    csv_path = download_file(
        url=data_cfg["url"],
        out_path=data_cfg["out_path"],
        expected_md5=data_cfg.get("md5"),
    )

    # Prepare data
    df = load_data(str(csv_path))
    varmap = build_varmap(df)
    dat_long = reshape_long(df, varmap)
    dat_long, sd_tbl = standardize_within_measure(dat_long)

    y = dat_long["z"].to_numpy()

    # Center time: pre=−0.5, post=+0.5 so alpha = grand mean
    time_raw = (dat_long["time"] == "post").astype(float).to_numpy()
    time_c = time_raw - time_raw.mean()

    # Encode IDs as contiguous integers
    id_cat   = dat_long["id"].astype("category")
    id_idx   = id_cat.cat.codes.to_numpy()
    id_labels = id_cat.cat.categories.tolist()
    n_id     = len(id_labels)

    # Encode measures as contiguous integers
    meas_cat   = dat_long["measure"].astype("category")
    meas_idx   = meas_cat.cat.codes.to_numpy()
    meas_labels = meas_cat.cat.categories.tolist()
    n_meas     = len(meas_labels)

    coords = {
        "id":     id_labels,
        "measure": meas_labels,
        "coef2":  ["intercept", "time"],
    }

    # Model + sample
    sampler = cfg["sampler"]
    model = build_model(y, time_c, id_idx, meas_idx, n_id, n_meas, coords)

    idata = fit_model(
        model,
        draws=int(sampler["draws"]),
        tune=int(sampler["tune"]),
        chains=int(sampler["chains"]),
        target_accept=float(sampler["target_accept"]),
        max_treedepth=int(sampler.get("max_treedepth", 12)),
    )

    # Outputs
    outs = cfg["outputs"]
    save_summary(
    idata,
    outdir=outs["outdir"],
    save_idata=bool(outs.get("save_idata", True)),
    save_summary=bool(outs.get("save_summary", True)),
    )

def cmd_summarize(idata_path: str, outdir: str) -> None:
    idata = az.from_netcdf(idata_path)
    write_core_tables(idata, outdir)


def cmd_plot(idata_path: str, outdir: str) -> None:
    idata = az.from_netcdf(idata_path)
    plot_trace(idata, outdir)
    plot_forest_beta(idata, outdir)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dmt-bayes")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Download data (if needed), fit model, and write outputs.")
    run.add_argument("--config", required=True, help="Path to YAML config, e.g. configs/default.yaml")
    summ = sub.add_parser("summarize", help="Generate tables from an existing idata.nc (no refit).")
    summ.add_argument("--idata", required=True, help="Path to idata.nc")
    summ.add_argument("--outdir", default="results", help="Output directory")

    plot = sub.add_parser("plot", help="Generate figures from an existing idata.nc (no refit).")
    plot.add_argument("--idata", required=True, help="Path to idata.nc")
    plot.add_argument("--outdir", default="results", help="Output directory")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    if args.cmd == "run":
        cmd_run(args.config)
    elif args.cmd == "summarize":
        cmd_summarize(args.idata, args.outdir)
    elif args.cmd == "plot":
        cmd_plot(args.idata, args.outdir)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()