#!/usr/bin/env python3
"""
manuscript_analyses.py
======================
Four HIGH-priority analyses required before manuscript drafting:

  1. diagnostics_summary.csv  — R̂ and ESS for all parameters in idata.nc
  2. No-pooling baseline model — independent Bayesian regressions per measure
  3. loo_comparison.csv        — LOO-CV (PSIS-LOO) hierarchical vs. no-pooling
  4. sensitivity_results.csv  — β_m posterior summaries under three prior specs

Usage (from repo root, with pymc_env activated):
    python scripts/manuscript_analyses.py

All outputs are written to results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

# ── make dmt_bayes importable without `pip install -e .` ─────────────────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dmt_bayes.data_prep import (
    load_data,
    build_varmap,
    reshape_long,
    standardize_within_measure,
)
from dmt_bayes.model import build_model

RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: reconstruct analysis arrays from raw data
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data():
    """Load + reshape + standardize; return arrays and metadata."""
    df = load_data(str(REPO / "data" / "scales_results.csv"))
    varmap = build_varmap(df)
    dat_long, measure_order = reshape_long(df, varmap)
    dat_long, _ = standardize_within_measure(dat_long)

    dat_long["time_c"] = np.where(dat_long["time"] == "pre", -0.5, 0.5)

    id_cat   = pd.Categorical(dat_long["id"])
    meas_cat = dat_long["measure"].cat.codes.values

    y       = dat_long["z"].values.astype(float)
    time_c  = dat_long["time_c"].values.astype(float)
    id_idx  = id_cat.codes.astype(int)
    meas_idx = meas_cat.astype(int)
    n_id    = int(id_cat.categories.size)
    n_meas  = len(measure_order)

    coords = {
        "id":     list(id_cat.categories),
        "measure": measure_order,
        "coef2":  ["intercept", "slope"],
        "obs":    np.arange(len(y)),
    }

    return y, time_c, id_idx, meas_idx, n_id, n_meas, coords, measure_order


# ─────────────────────────────────────────────────────────────────────────────
# No-pooling model
# ─────────────────────────────────────────────────────────────────────────────

def build_nopooling_model(y, time_c, id_idx, meas_idx, n_id, n_meas, coords=None):
    """
    No-pooling baseline: independent Bayesian regression per measure.

    Structure
    ---------
    Each measure m gets its own:
      • alpha_m   ~ Normal(0, 1)          — per-measure intercept
      • beta_m    ~ Normal(0, 1)          — per-measure time effect (PRIMARY ESTIMAND)
      • sigma_b0  ~ HalfNormal(0.75)      — per-measure random-intercept SD
      • sigma_m   ~ HalfNormal(1.0)       — per-measure residual SD

    Subject random intercepts b0_im are measure-specific (non-centered):
      z_b0_im ~ Normal(0, 1)   →   b0_im = sigma_b0[m] * z_b0_im

    No information is shared across measures. This is equivalent in structure
    to running 12 separate Bayesian paired-t-test regressions.

    Note: random *slopes* per (subject, measure) are not included because with
    only 2 observations per subject-measure cell the slope is already captured
    by beta_m[m] (population) and b0_im (subject deviation from that mean);
    adding per-cell random slopes would be unidentified given the balanced
    pre/post design. This matches the structure of a Bayesian paired t-test.
    """
    with pm.Model(coords=coords) as model:

        # Mutable data containers so pm.set_data() works for CV / out-of-sample evaluation
        y_obs    = pm.Data("y_obs",     y)
        time_obs = pm.Data("time_c",   time_c)
        id_obs   = pm.Data("id_idx",   id_idx)
        meas_obs = pm.Data("meas_idx", meas_idx)

        # ── Per-measure fixed effects (independent priors, no hyperprior) ──
        alpha_m = pm.Normal("alpha_m", 0.0, 1.0, dims=("measure",))
        beta_m  = pm.Normal("beta_m",  0.0, 1.0, dims=("measure",))

        # ── Per-measure subject random intercepts (non-centered) ──
        sigma_b0 = pm.HalfNormal("sigma_b0", 0.75, dims=("measure",))
        z_b0     = pm.Normal("z_b0", 0.0, 1.0, dims=("id", "measure"))
        b0_im    = pm.Deterministic(
            "b0_im", sigma_b0[np.newaxis, :] * z_b0, dims=("id", "measure")
        )

        # ── Per-measure residual SD ──
        sigma_m = pm.HalfNormal("sigma_m", 1.0, dims=("measure",))

        # ── Linear predictor ──
        mu = (
            alpha_m[meas_obs]
            + beta_m[meas_obs] * time_obs
            + b0_im[id_obs, meas_obs]
        )

        pm.Normal("y", mu, sigma_m[meas_obs], observed=y_obs)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity model builder (parameterised priors)
# ─────────────────────────────────────────────────────────────────────────────

def build_sensitivity_model(
    y, time_c, id_idx, meas_idx, n_id, n_meas, coords,
    scale_fixed: float = 1.0,
    scale_re: float = 0.75,
    scale_sigma: float = 1.0,
):
    """
    Same hierarchical model as build_model() but with tunable prior widths.

    Parameters
    ----------
    scale_fixed  : SD for Normal priors on alpha and beta_time (default 1.0)
    scale_re     : SD for HalfNormal priors on all random-effect SDs (default 0.75)
    scale_sigma  : SD for HalfNormal prior on residual sigma (default 1.0)
    """
    with pm.Model(coords=coords) as model:

        # ── Global fixed effects ──
        alpha = pm.Normal("alpha",     0.0, scale_fixed)
        beta  = pm.Normal("beta_time", 0.0, scale_fixed)

        # ── Subject REs: manual Cholesky (mirrors original model) ──
        sigma_id = pm.HalfNormal("sigma_id", scale_re, shape=2)
        corr_id  = pm.Uniform("corr_id", -1.0, 1.0)

        L_id = pt.stack([
            pt.stack([sigma_id[0], pt.zeros(())]),
            pt.stack([corr_id * sigma_id[1],
                      sigma_id[1] * pt.sqrt(1.0 - corr_id**2)])
        ])

        z_id  = pm.Normal("z_id", 0.0, 1.0, dims=("id", "coef2"))
        RE_id = pm.Deterministic("RE_id", pt.dot(z_id, L_id.T), dims=("id", "coef2"))
        b0_i  = RE_id[:, 0]
        b1_i  = RE_id[:, 1]

        # ── Measure REs: non-centered, independent ──
        sd_u0 = pm.HalfNormal("sd_u0", scale_re)
        sd_u1 = pm.HalfNormal("sd_u1", scale_re)
        z_u0  = pm.Normal("z_u0", 0.0, 1.0, dims=("measure",))
        z_u1  = pm.Normal("z_u1", 0.0, 1.0, dims=("measure",))
        u0_m  = pm.Deterministic("u0_m", sd_u0 * z_u0, dims=("measure",))
        u1_m  = pm.Deterministic("u1_m", sd_u1 * z_u1, dims=("measure",))

        # ── Residual ──
        sigma = pm.HalfNormal("sigma", scale_sigma)

        # ── Linear predictor ──
        mu = (
            alpha
            + beta * time_c
            + b0_i[id_idx]
            + b1_i[id_idx] * time_c
            + u0_m[meas_idx]
            + u1_m[meas_idx] * time_c
        )

        pm.Normal("y", mu, sigma, observed=y)
        pm.Deterministic("beta_m_by_measure", beta + u1_m, dims=("measure",))

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helper
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_KWARGS = dict(
    draws=2000,
    tune=1000,
    chains=4,
    target_accept=0.99,
    nuts_sampler_kwargs={"max_treedepth": 15},
    return_inferencedata=True,
    random_seed=42,
)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — diagnostics_summary.csv
# ─────────────────────────────────────────────────────────────────────────────

def task1_diagnostics():
    print("\n" + "=" * 65)
    print("TASK 1: R̂ and ESS from existing idata.nc")
    print("=" * 65)

    idata = az.from_netcdf(str(RESULTS / "idata.nc"))

    diag_df = az.summary(idata, hdi_prob=0.94, round_to=6)
    out = RESULTS / "diagnostics_summary.csv"
    diag_df.to_csv(str(out))

    rhat_max = diag_df["r_hat"].max()
    ess_bulk_min = diag_df["ess_bulk"].min()
    ess_tail_min = diag_df["ess_tail"].min()

    print(f"  Parameters:     {len(diag_df)}")
    print(f"  R̂ max:          {rhat_max:.4f}  (target < 1.01)")
    print(f"  ESS bulk min:   {ess_bulk_min:.0f}  (target > 400)")
    print(f"  ESS tail min:   {ess_tail_min:.0f}  (target > 400)")
    print(f"  Saved → {out}")

    return idata


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 + 3 — No-pooling model and LOO-CV comparison
# ─────────────────────────────────────────────────────────────────────────────

def task2_3_nopooling_and_loo(idata_hier):
    print("\n" + "=" * 65)
    print("TASKS 2 + 3: No-pooling model and LOO-CV comparison")
    print("=" * 65)

    y, time_c, id_idx, meas_idx, n_id, n_meas, coords, measure_order = prepare_data()

    # ── 2a. Compute log-likelihood for the hierarchical model ────────────────
    # (needed for LOO; not stored in the original idata.nc)
    print("\n  [2a] Computing log-likelihood for hierarchical model …")
    hier_model = build_model(y, time_c, id_idx, meas_idx, n_id, n_meas, coords=coords)
    idata_hier = pm.compute_log_likelihood(idata_hier, model=hier_model)

    # ── 2b. Fit no-pooling model ─────────────────────────────────────────────
    print("\n  [2b] Fitting no-pooling model (independent regressions per measure) …")
    nopooling_model = build_nopooling_model(
        y, time_c, id_idx, meas_idx, n_id, n_meas, coords=coords
    )
    with nopooling_model:
        idata_nopool = pm.sample(**SAMPLE_KWARGS)

    # Compute log-likelihood for no-pooling model
    print("  [2b] Computing log-likelihood for no-pooling model …")
    idata_nopool = pm.compute_log_likelihood(idata_nopool, model=nopooling_model)

    # Save no-pooling idata for reference
    nopool_nc = RESULTS / "idata_nopooling.nc"
    idata_nopool.to_netcdf(str(nopool_nc))
    print(f"  No-pooling idata saved → {nopool_nc}")

    # ── 2c. Diagnostics for no-pooling model ────────────────────────────────
    nopool_diag = az.summary(idata_nopool, hdi_prob=0.94, round_to=6)
    print(f"  No-pooling R̂ max:        {nopool_diag['r_hat'].max():.4f}")
    print(f"  No-pooling ESS bulk min: {nopool_diag['ess_bulk'].min():.0f}")

    # ── 3. LOO-CV comparison ─────────────────────────────────────────────────
    print("\n  [3] Running LOO-CV comparison …")
    loo_hier   = az.loo(idata_hier,   pointwise=True)
    loo_nopool = az.loo(idata_nopool, pointwise=True)

    print(f"  Hierarchical  ELPD-LOO: {loo_hier.elpd_loo:.2f}  (SE={loo_hier.se:.2f})")
    print(f"  No-pooling    ELPD-LOO: {loo_nopool.elpd_loo:.2f}  (SE={loo_nopool.se:.2f})")

    comparison = az.compare(
        {"hierarchical": idata_hier, "no_pooling": idata_nopool},
        ic="loo",
        method="stacking",
    )
    out = RESULTS / "loo_comparison.csv"
    comparison.to_csv(str(out))
    print(f"  LOO comparison saved → {out}")
    print(comparison.to_string())

    return idata_nopool, loo_hier, loo_nopool

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3b — K-Fold Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import KFold

def make_folds(id_idx, K=10, seed=42):
    unique_ids = np.unique(id_idx)
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)

    folds = []
    for train_ids_idx, test_ids_idx in kf.split(unique_ids):
        train_ids = unique_ids[train_ids_idx]
        test_ids  = unique_ids[test_ids_idx]

        train_mask = np.isin(id_idx, train_ids)
        test_mask  = np.isin(id_idx, test_ids)

        folds.append((train_mask, test_mask))

    return folds

def compute_test_loglik(model, idata, y_test, time_test, id_test, meas_test):
    with model:
        pm.set_data({
            "y_obs": y_test,
            "time_c": time_test,
            "id_idx": id_test,
            "meas_idx": meas_test,
        })

        idata_ll = pm.compute_log_likelihood(idata)

    loglik = idata_ll.log_likelihood["y"].values  # (chain, draw, n_test)
    loglik = loglik.reshape(-1, loglik.shape[-1])  # (samples, n_test)

    lppd = np.sum(np.log(np.mean(np.exp(loglik), axis=0)))
    return lppd
    
def reindex_ids(id_array):
    unique = np.unique(id_array)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[i] for i in id_array]), len(unique)
    
def task3b_kfold_cv_comparison():
    print("\n[K-fold CV] Running 10-fold cross-validation...")

    y, time_c, id_idx, meas_idx, n_id, n_meas, coords, _ = prepare_data()
    folds = make_folds(id_idx, K=10)

    elpd_hier = []
    elpd_nopool = []

    for k, (train_mask, test_mask) in enumerate(folds):
        print(f"  Fold {k+1}/10")

        # TRAIN split
        y_train = y[train_mask]
        time_train = time_c[train_mask]
        id_train_raw = id_idx[train_mask]
        meas_train = meas_idx[train_mask]

        id_train, n_id_train = reindex_ids(id_train_raw)

        # TEST split
        y_test = y[test_mask]
        time_test = time_c[test_mask]
        id_test_raw = id_idx[test_mask]
        meas_test = meas_idx[test_mask]

        id_test, _ = reindex_ids(id_test_raw)

        # Build + sample
        model_h = build_model(
            y_train, time_train, id_train, meas_train,
            n_id_train, n_meas, coords=coords
        )

        with model_h:
            idata_h = pm.sample(**SAMPLE_KWARGS_KFOLD)

        # Evaluate on test
        elpd_hier.append(
            compute_test_loglik(
                model_h, idata_h,
                y_test, time_test, id_test, meas_test
            )
        )

        # --- No pooling ---
        model_np = build_nopooling_model(
            y_train, time_train, id_train, meas_train,
            n_id, n_meas, coords=coords
        )
        with model_np:
            idata_np = pm.sample(**SAMPLE_KWARGS)

        elpd_nopool.append(
            compute_test_loglik(
                model_np, idata_np,
                y_test, time_test, id_test, meas_test
            )
        )

    elpd_hier = np.array(elpd_hier)
    elpd_nopool = np.array(elpd_nopool)

    result = pd.DataFrame({
        "model": ["hierarchical", "no_pooling"],
        "elpd_mean": [elpd_hier.mean(), elpd_nopool.mean()],
        "elpd_se": [
            elpd_hier.std() / np.sqrt(10),
            elpd_nopool.std() / np.sqrt(10)
        ]
    })

    result["delta_elpd"] = result["elpd_mean"] - result["elpd_mean"].iloc[0]

    out = RESULTS / "kfold_comparison.csv"
    result.to_csv(out, index=False)

    print(result)
    return result

SAMPLE_KWARGS_KFOLD = dict(
    draws=1000,
    tune=1000,
    chains=2,
    target_accept=0.95,
    random_seed=42,
)
# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — Prior sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def task4_sensitivity(idata_hier, measure_order):
    """
    Three prior specifications:

      Spec A (default, already fitted):
        alpha, beta    ~ Normal(0, 1.00)
        sigma_id, sd_u ~ HalfNormal(0.75)
        sigma          ~ HalfNormal(1.00)

      Spec B (tighter — all SD-priors halved):
        alpha, beta    ~ Normal(0, 0.50)
        sigma_id, sd_u ~ HalfNormal(0.375)
        sigma          ~ HalfNormal(0.50)

      Spec C (wider — all SD-priors doubled):
        alpha, beta    ~ Normal(0, 2.00)
        sigma_id, sd_u ~ HalfNormal(1.50)
        sigma          ~ HalfNormal(2.00)

    For each spec we report, per measure:
      posterior mean, SD, 94% HDI lo/hi, and P(β_m < 0)
    """
    print("\n" + "=" * 65)
    print("TASK 4: Prior sensitivity analysis")
    print("=" * 65)

    y, time_c, id_idx, meas_idx, n_id, n_meas, coords, measure_order = prepare_data()

    # ── Spec A: extract from existing idata ──────────────────────────────────
    print("\n  [Spec A] Extracting β_m from existing idata.nc …")
    draws_a = idata_hier.posterior["beta_m_by_measure"]  # (chain, draw, measure)

    # ── Spec B: tighter priors ───────────────────────────────────────────────
    print("\n  [Spec B] Fitting tighter-prior model …")
    model_b = build_sensitivity_model(
        y, time_c, id_idx, meas_idx, n_id, n_meas, coords,
        scale_fixed=0.50,
        scale_re=0.375,
        scale_sigma=0.50,
    )
    with model_b:
        idata_b = pm.sample(**SAMPLE_KWARGS)
    idata_b.to_netcdf(str(RESULTS / "idata_spec_b.nc"))
    draws_b = idata_b.posterior["beta_m_by_measure"]

    # ── Spec C: wider priors ─────────────────────────────────────────────────
    print("\n  [Spec C] Fitting wider-prior model …")
    model_c = build_sensitivity_model(
        y, time_c, id_idx, meas_idx, n_id, n_meas, coords,
        scale_fixed=2.00,
        scale_re=1.50,
        scale_sigma=2.00,
    )
    with model_c:
        idata_c = pm.sample(**SAMPLE_KWARGS)
    idata_c.to_netcdf(str(RESULTS / "idata_spec_c.nc"))
    draws_c = idata_c.posterior["beta_m_by_measure"]

    # ── Compile results table ────────────────────────────────────────────────
    rows = []
    for i, meas in enumerate(measure_order):
        row = {"measure": meas}

        for label, draws in [("spec_a", draws_a), ("spec_b", draws_b), ("spec_c", draws_c)]:
            flat = draws.values[:, :, i].ravel()   # collapse chain × draw
            hdi  = az.hdi(flat, hdi_prob=0.94)
            row[f"{label}_mean"]   = float(np.mean(flat))
            row[f"{label}_sd"]     = float(np.std(flat))
            row[f"{label}_hdi_lo"] = float(hdi[0])
            row[f"{label}_hdi_hi"] = float(hdi[1])
            row[f"{label}_p_lt0"]  = float(np.mean(flat < 0))

        rows.append(row)

    sens_df = pd.DataFrame(rows)
    out = RESULTS / "sensitivity_results.csv"
    sens_df.to_csv(str(out), index=False, float_format="%.4f")
    print(f"\n  Sensitivity results saved → {out}")
    print(sens_df[["measure",
                   "spec_a_mean", "spec_b_mean", "spec_c_mean",
                   "spec_a_hdi_lo", "spec_b_hdi_lo", "spec_c_hdi_lo",
                   "spec_a_hdi_hi", "spec_b_hdi_hi", "spec_c_hdi_hi"]].to_string(index=False))

    return sens_df


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 — Posterior predictive checks: Bayesian p-values and Bayesian R²
# ─────────────────────────────────────────────────────────────────────────────

def task5_ppc_r2(idata_hier, measure_order):
    """
    Compute and export posterior predictive check statistics for the primary
    hierarchical model (Specification A):

    1. Per-measure Bayesian p-value:
       For each measure m, the proportion of posterior predictive replicates
       whose mean (across participants × time points for that measure) exceeds
       the observed mean. Values near 0 or 1 indicate systematic misfit.
       (Gabry et al., 2019, JRSS-A 182(2):389-402)

    2. Pooled Bayesian R² via ArviZ az.r2_score():
       Proportion of variance in the pooled 490-observation outcome vector
       explained by the posterior predictive mean.
       (Gelman et al., 2019, The American Statistician 73(3):307-309)

    Outputs: results/ppc_summary.csv
    """
    print("\n" + "=" * 65)
    print("TASK 5: Posterior predictive checks — Bayesian p-values and R²")
    print("=" * 65)

    # ── Load observed data and rebuild model ─────────────────────────────────
    y, time_c, id_idx, meas_idx, n_id, n_meas, coords, measure_order_local = (
        prepare_data()
    )
    # Use measure_order passed in (consistent with idata coords)
    if measure_order is None:
        measure_order = measure_order_local

    # ── Generate posterior predictive samples (not stored in idata.nc) ────────
    print("\n  [5a] Generating posterior predictive samples …")
    hier_model = build_model(y, time_c, id_idx, meas_idx, n_id, n_meas, coords=coords)
    with hier_model:
        idata_ppc = pm.sample_posterior_predictive(
            idata_hier, random_seed=42, progressbar=True
        )
    # idata_ppc.posterior_predictive["y"] shape: (chain, draw, obs)
    ppc_y_raw = idata_ppc.posterior_predictive["y"].values
    n_chain_r2, n_draw_r2, _ = ppc_y_raw.shape
    y_pred = ppc_y_raw.reshape(n_chain_r2 * n_draw_r2, -1)  # (samples, obs)
    y_true = y  # observed standardized scores, length 490

    # ── Bayesian R² (pooled, all 490 observations) ────────────────────────────
    print("\n  [5b] Computing pooled Bayesian R² …")
    # az.r2_score(y_true, y_pred) — y_pred shape: (n_samples, n_obs)
    # Compute per-draw R² manually so we have a distribution for HDI
    ss_res = np.sum((y_true - y_pred) ** 2, axis=1)          # (samples,)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2_samples = 1.0 - ss_res / ss_tot                        # (samples,)
    r2_mean = float(r2_samples.mean())
    r2_sd   = float(r2_samples.std())
    r2_hdi  = az.hdi(r2_samples, hdi_prob=0.94)
    print(f"  Bayesian R²: mean={r2_mean:.4f}, SD={r2_sd:.4f}, "
          f"94% HDI=[{r2_hdi[0]:.4f}, {r2_hdi[1]:.4f}]")

    # ── Per-measure Bayesian p-values ─────────────────────────────────────────
    print("\n  [5c] Computing per-measure Bayesian p-values …")

    # ppc_y reuses the same y_pred flat array
    ppc_y = ppc_y_raw  # (chain, draw, obs)
    ppc_flat = y_pred  # already (chain*draw, obs)

    rows = []
    for i, meas in enumerate(measure_order):
        mask = meas_idx == i
        pre_mask  = mask & (time_c == -0.5)
        post_mask = mask & (time_c ==  0.5)
        # align pre/post by subject so subtraction is paired
        pre_z  = y[pre_mask][np.argsort(id_idx[pre_mask])]
        post_z = y[post_mask][np.argsort(id_idx[post_mask])]
        obs_mean = float((post_z - pre_z).mean())

        # posterior predictive means for this measure
        ppc_means = ppc_flat[:, mask].mean(axis=1)  # (chain*draw,)
        # two-sided p-value: proportion of replicates as or more extreme than observed
        p_val = float(np.mean(np.abs(ppc_means - 0.0) >= np.abs(obs_mean - 0.0)))

        rows.append({
            "measure":       meas,
            "obs_mean":      round(obs_mean, 6),
            "ppc_mean_mean": round(float(ppc_means.mean()), 6),
            "ppc_mean_sd":   round(float(ppc_means.std()), 6),
            "bayesian_pval": round(p_val, 4),
        })
        print(f"    {meas:<25s}  obs_mean={obs_mean:+.4f}  "
              f"Bayes p-val={p_val:.3f}")

    ppc_df = pd.DataFrame(rows)

    # ── Append R² summary row ─────────────────────────────────────────────────
    r2_row = pd.DataFrame([{
        "measure":       "POOLED_R2",
        "obs_mean":      None,
        "ppc_mean_mean": round(r2_mean, 6),
        "ppc_mean_sd":   round(r2_sd, 6),
        "bayesian_pval": None,
        "r2_hdi_lo":     round(float(r2_hdi[0]), 6),
        "r2_hdi_hi":     round(float(r2_hdi[1]), 6),
    }])
    ppc_df = pd.concat([ppc_df, r2_row], ignore_index=True)

    out = RESULTS / "ppc_summary.csv"
    ppc_df.to_csv(str(out), index=False)
    print(f"\n  PPC summary saved → {out}")

    return ppc_df

def task6_rope_probability_directionality(idata, measure_order):
    """
    Compute ROPE-based classification for Specification A.

    Returns:
        DataFrame with:
        - measure
        - p_gt0
        - p_lt0
        - p_rope
        - classification
    """

    beta = idata.posterior["beta_m_by_measure"]
    beta = beta.stack(sample=("chain", "draw"))
    beta = beta.transpose("sample", "measure").values  # enforce correct shape

    rope_low, rope_high = -0.10, 0.10

    p_rope = ((beta >= rope_low) & (beta <= rope_high)).mean(axis=0)
    p_gt0  = (beta > 0).mean(axis=0)
    p_lt0  = (beta < 0).mean(axis=0)

    ROPE_HIGH = 0.60   # evidence for negligible
    ROPE_LOW  = 0.20   # evidence against negligible
    DIR_HIGH  = 0.90   # strong directional evidence

    classification = []
    for i in range(len(measure_order)):
        if p_rope[i] >= 0.45:
            classification.append("Likely negligible")
        elif p_rope[i] <= 0.40:
            if p_gt0[i] >= 0.80 or p_lt0[i] >= 0.80:
                classification.append("Potentially meaningful (directional tendency)")
            else:
                classification.append("Potentially meaningful (uncertain direction)")
        else:
            classification.append("Indeterminate")

    df = pd.DataFrame({
        "Measure": measure_order,
        "P(β>0)": np.round(p_gt0, 3),
        "P(β<0)": np.round(p_lt0, 3),
        "P(β∈ROPE)": np.round(p_rope, 3),
        "Classification": classification
    })
    print(df)
    out = RESULTS / "rope_probability_directionality.csv"
    df.to_csv(str(out), index=False)
    print(f"\n  ROPE probability directionality saved → {out}")

    return df
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DMT Bayesian Hierarchical — Manuscript Analyses")
    print("=" * 65)

    # 1. Diagnostics
    # idata_hier = task1_diagnostics()

    # # 2 + 3. No-pooling model and LOO-CV
    # idata_nopool, loo_hier, loo_nopool = task2_3_nopooling_and_loo(idata_hier)

    # # 4. Sensitivity analysis
    # import json
    # with open(str(RESULTS / "measure_labels.json")) as f:
    #     measure_order = json.load(f)
    # task4_sensitivity(idata_hier, measure_order)

    # # 5. Posterior predictive checks: Bayesian p-values and R²
    # task5_ppc_r2(idata_hier, measure_order)

    # # 6. ROPE Probability Directionality
    # y, time_c, id_idx, meas_idx, n_id, n_meas, coords, measure_order = prepare_data()
    # idata = az.from_netcdf(str(RESULTS / "idata.nc"))
    # task6_rope_probability_directionality(idata, measure_order)

    task3b_kfold_cv_comparison()

    print("\n" + "=" * 65)
    print("All analyses complete. Output files in results/:")
    print("  diagnostics_summary.csv")
    print("  idata_nopooling.nc")
    print("  loo_comparison.csv")
    print("  idata_spec_b.nc")
    print("  idata_spec_c.nc")
    print("  sensitivity_results.csv")
    print("  ppc_summary.csv")
    print("  rope_probability_directionality.csv")
    print("=" * 65)
