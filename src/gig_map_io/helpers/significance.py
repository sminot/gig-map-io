"""
How a bin's association result becomes a call: higher in cases, lower in
cases, or unassociated.

The association model is a linear regression of bin abundance on disease
(plus batch terms for a combined study). gig-map reports its disease
coefficient divided by the intercept, the fitted control mean, as
``Estimate``, and divides ``Robust SE`` by the same intercept, so:

- the direction of an association is the sign of ``t value``, which is the
  sign of the coefficient itself. The sign of ``Estimate`` is not: a linear
  fit can put the control mean below zero, which flips it;
- the sign of the fitted control mean is the sign of ``Robust SE``;
- the fold change, fitted case mean over fitted control mean, is
  ``1 + Estimate``, and exists only while both fitted means are positive.
  Where exactly one is at or below zero the bin is present in one group and
  absent in the other, so its log2 fold change is infinite in the direction
  of the association.

Every place that calls bins, counts them or draws a threshold goes through
these functions, so that a bin is called the same way everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .style import FDR_THRESH, MIN_ABS_LOG2_FOLD_CHANGE, MIN_MEAN_RPKM

#: The three ways a bin can be called, as written in the association tables
DIRECTIONS = ["<", "=", ">"]


def log2_fold_change(association: pd.DataFrame) -> pd.Series:
    """
    log2 of fitted case mean over fitted control mean: infinite, in the
    direction of the association, where one fitted mean is at or below zero,
    and NaN where both are.
    """
    if (association["Robust SE"] == 0).any():
        raise ValueError("A Robust SE of zero leaves the sign of the fitted control mean unknown")
    control_positive = association["Robust SE"] > 0
    # The fitted case mean is control * (1 + Estimate)
    case_positive = np.where(control_positive, association["Estimate"] > -1, association["Estimate"] < -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        finite = np.log2(1 + association["Estimate"])
    infinite = np.sign(association["t value"]) * np.inf
    return pd.Series(
        np.select(
            [control_positive & case_positive, control_positive != case_positive],
            [finite, infinite],
            default=np.nan,
        ),
        index=association.index,
    )


def add_significance(association: pd.DataFrame, min_mean_rpkm: float = MIN_MEAN_RPKM) -> pd.DataFrame:
    """
    The association results with the abundance filter applied and the
    columns derived from it.

    A bin whose mean RPKM across the study's samples is below
    ``min_mean_rpkm`` is marked ``low_abundance`` and left out of the FDR
    correction, so it has no q-value. The filter does not look at which
    samples are cases, so the remaining p-values stay valid for it.
    """
    df = association.dropna(subset=["pvalue"])
    low_abundance = df["mean_abund"] < min_mean_rpkm
    qvalue = np.full(len(df), np.nan)
    qvalue[~low_abundance.values] = multipletests(df.loc[~low_abundance, "pvalue"], method="fdr_bh")[1]
    # A p-value of exactly zero would put a bin at infinity on a -log10 axis
    pvalue = df["pvalue"].clip(lower=df.loc[df["pvalue"] > 0, "pvalue"].min())
    direction = np.sign(df["t value"])
    return df.assign(
        low_abundance=low_abundance,
        log2_fold_change=log2_fold_change(df),
        pvalue=pvalue,
        qvalue=qvalue,
        neg_log10_pvalue=-np.log10(pvalue),
        neg_log10_qvalue=-np.log10(qvalue),
        signed_log10_pvalue=direction * -np.log10(pvalue),
        signed_log10_qvalue=direction * -np.log10(qvalue),
    )


def call_direction(
    association: pd.DataFrame,
    fdr: bool = True,
    sig_thresh: float = FDR_THRESH,
    min_abs_log2_fold_change: float = MIN_ABS_LOG2_FOLD_CHANGE,
) -> pd.Series:
    """
    ``">"`` or ``"<"`` for a bin called higher or lower in cases, ``"="``
    for one that is not.

    A bin is called when it passes the abundance filter, its q-value (or
    p-value, with ``fdr=False``) is below ``sig_thresh``, and its fold change
    is at least ``2 ** min_abs_log2_fold_change`` either way.
    """
    sig_col = "qvalue" if fdr else "pvalue"
    called = (
        ~association["low_abundance"]
        & (association[sig_col] < sig_thresh)
        & (association["log2_fold_change"].abs() >= min_abs_log2_fold_change)
    )
    return pd.Series(
        np.select(
            [called & (association["t value"] > 0), called & (association["t value"] < 0)],
            [">", "<"],
            default="=",
        ),
        index=association.index,
    )
