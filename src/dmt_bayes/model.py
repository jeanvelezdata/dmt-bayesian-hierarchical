from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model(y, time_c, id_idx, meas_idx, n_id, n_meas, coords=None):

    with pm.Model(coords=coords) as model:

        # --- Global fixed effects ---
        alpha = pm.Normal("alpha", 0.0, 1.0)
        beta  = pm.Normal("beta_time", 0.0, 1.0)

        # --- Subject REs: manual Cholesky (avoids LKJCholeskyCov einsum bug in PyMC 5.26) ---
        # Equivalent to LKJCholeskyCov(eta=5, sd_dist=HalfNormal(0.75))
        sigma_id = pm.HalfNormal("sigma_id", 0.75, shape=2)
        corr_id  = pm.Uniform("corr_id", -1.0, 1.0)

        # Build 2x2 lower Cholesky manually
        L_id = pt.stack([
            pt.stack([sigma_id[0],                                   pt.zeros(())]),
            pt.stack([corr_id * sigma_id[1], sigma_id[1] * pt.sqrt(1.0 - corr_id**2)])
        ])  # shape (2, 2)

        z_id  = pm.Normal("z_id", 0.0, 1.0, dims=("id", "coef2"))
        RE_id = pm.Deterministic("RE_id", pt.dot(z_id, L_id.T), dims=("id", "coef2"))

        b0_i = RE_id[:, 0]   # subject random intercepts
        b1_i = RE_id[:, 1]   # subject random slopes

        # --- Measure REs: noncentered, uncorrelated ---
        sd_u0 = pm.HalfNormal("sd_u0", 0.75)
        sd_u1 = pm.HalfNormal("sd_u1", 0.75)

        z_u0 = pm.Normal("z_u0", 0.0, 1.0, dims=("measure",))
        z_u1 = pm.Normal("z_u1", 0.0, 1.0, dims=("measure",))

        u0_m = pm.Deterministic("u0_m", sd_u0 * z_u0, dims=("measure",))
        u1_m = pm.Deterministic("u1_m", sd_u1 * z_u1, dims=("measure",))

        # --- Residual noise ---
        sigma = pm.HalfNormal("sigma", 1.0)

        # --- Linear predictor ---
        mu = (
            alpha
            + beta * time_c
            + b0_i[id_idx]
            + b1_i[id_idx] * time_c
            + u0_m[meas_idx]
            + u1_m[meas_idx] * time_c
        )

        pm.Normal("y", mu, sigma, observed=y)

        # Measure-specific total time slope (fixed + random)
        pm.Deterministic("beta_m_by_measure", beta + u1_m, dims=("measure",))

    return model