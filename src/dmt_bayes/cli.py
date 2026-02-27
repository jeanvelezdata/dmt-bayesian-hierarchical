from __future__ import annotations
import arviz as az
from .report import write_core_tables, plot_trace, plot_forest_beta
from pathlib import Path
import json

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

    # dat_long and measure_order come from reshape_long()
    dat_long, measure_order = reshape_long(df, varmap)
    dat_long, sd_tbl = standardize_within_measure(dat_long)

    y = dat_long["z"].to_numpy()
    time_c = (dat_long["time"] == "post").astype(int).to_numpy()

    # ids are already integers 0..N-1 if you used np.arange in reshape_long()
    id_idx = dat_long["id"].to_numpy()

    # meas_idx must align with measure_order
    meas_idx = dat_long["measure"].cat.codes.to_numpy()

    n_id = int(np.unique(id_idx).size)
    n_meas = len(measure_order)

    coords = {
        "id": np.arange(n_id),
        "measure": measure_order,
        "coef2": ["intercept", "slope"],
    }

    # Model + sample
    sampler = cfg["sampler"]
    np.random.seed(cfg.get("seed", 42))
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
    outdir = Path(outs["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "measure_labels.json").write_text(
        json.dumps(measure_order, indent=2),
        encoding="utf-8",
    )
    save_summary(
        idata,
        outdir=outs["outdir"],
        save_idata=bool(outs.get("save_idata", True)),
        save_summary=bool(outs.get("save_summary", True)),
    )
    print(f"Model run complete. Outputs written to: {outs['outdir']}")

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