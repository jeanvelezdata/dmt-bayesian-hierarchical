"""Paired t and Cohen's d_z per measure (replicates Pallavicini et al. style on Zenodo CSV)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dmt_bayes.data_prep import build_varmap, load_data


def to_float(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return s.astype(float)


def main() -> None:
    df = load_data(str(REPO / "data" / "scales_results.csv"))
    varmap = build_varmap(df)
    rows = []
    for m, (pre, post) in varmap.items():
        pre_v = to_float(df[pre])
        post_v = to_float(df[post])
        d = post_v - pre_v
        t_stat, p_two = stats.ttest_rel(post_v, pre_v)
        dz = float(d.mean() / d.std(ddof=1))
        rows.append(
            {
                "measure": m,
                "t": t_stat,
                "p_two_sided": p_two,
                "cohens_dz": dz,
                "mean_raw_change": float(d.mean()),
            }
        )
    out = pd.DataFrame(rows)
    outp = REPO.parent / "data" / "original_paired_stats.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False, float_format="%.6f")
    print(out.to_string(index=False))
    print(f"\nSaved {outp}")


if __name__ == "__main__":
    main()
