import re
import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def find_col(columns, pattern):
    regex = re.compile(pattern, flags=re.I)
    matches = [c for c in columns if regex.search(c)]
    if len(matches) != 1:
        raise ValueError(f"Pattern {pattern} matched {matches}")
    return matches[0]


def build_varmap(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """
    Explicit pre/post column mapping for Zenodo record 3992359 (scales_results.csv).

    Seven outcome measures matching the manuscript analysis (methods §2.2):
      - Big Five Inventory (BFI): five subscales
      - Tellegen Absorption Scale (TAS): total score
      - State-Trait Anxiety Inventory — State subscale (STAI-E)

    Notes:
      - STAI1-R exists, but STAI2-R does NOT, so Trait anxiety is excluded.
      - MBS subscales are present in the dataset but are not part of this analysis.
    """
    varmap = {
        # Big Five Inventory (BFI)
        "BFI-E": ("BFI1-E", "BFI2-E"),
        "BFI-A": ("BFI1-A", "BFI2-A"),
        "BFI-C": ("BFI1-C", "BFI2-C"),
        "BFI-N": ("BFI1-N", "BFI2-N"),
        "BFI-O": ("BFI1-O", "BFI2-O"),

        # State anxiety (paired; STAI-Trait excluded — no post-session observation)
        "STAI-E": ("STAI1-E", "STAI2-E"),

        # Tellegen Absorption Scale
        "TAS": ("TAS1", "TAS2"),
    }

    # Hard fail if Zenodo schema changes or file is wrong
    missing = []
    for meas, (pre, post) in varmap.items():
        if pre not in df.columns:
            missing.append((meas, pre))
        if post not in df.columns:
            missing.append((meas, post))

    if missing:
        msg = "Missing expected columns in dataset:\n" + "\n".join([f"  {m}: {c}" for m, c in missing])
        raise ValueError(msg)

    return varmap


def reshape_long(df: pd.DataFrame, varmap: dict):
    """
    Convert wide pre/post format into long format for modeling.
    Preserves explicit measure ordering from varmap.
    """
    rows = []

    # stable measure order = insertion order of varmap
    measure_order = list(varmap.keys())

    for meas in measure_order:
        pre_col, post_col = varmap[meas]

        tmp = df[[pre_col, post_col]].copy()
        tmp.columns = ["pre", "post"]
        tmp["measure"] = meas

        # participant id = row number (assumption: each row is a subject)
        tmp["id"] = np.arange(len(tmp))

        long = tmp.melt(
            id_vars=["id", "measure"],
            value_vars=["pre", "post"],
            var_name="time",
            value_name="score",
        )
        rows.append(long)

    dat_long = pd.concat(rows, ignore_index=True)
    dat_long = dat_long.dropna(subset=["score"]).reset_index(drop=True)

    # lock measure order as categorical so codes match varmap order
    dat_long["measure"] = pd.Categorical(dat_long["measure"], categories=measure_order, ordered=True)

    return dat_long, measure_order


def standardize_within_measure(dat_long: pd.DataFrame):
    sd_tbl = (
        dat_long.groupby("measure")["score"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mu", "std": "sd"})
    )

    dat_long = dat_long.merge(sd_tbl, left_on="measure", right_index=True)
    dat_long["z"] = (dat_long["score"] - dat_long["mu"]) / dat_long["sd"]

    return dat_long, sd_tbl