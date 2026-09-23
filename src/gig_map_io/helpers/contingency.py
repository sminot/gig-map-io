"""
Contingency testing between categorical sample annotations.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def chi2_contingency_test(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    dropna: bool = True,
    correction: bool = True,
) -> dict:
    """
    Run a chi-squared contingency test between two categorical columns.

    Builds a contingency table from the two columns and passes it to
    ``scipy.stats.chi2_contingency``.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Must contain ``col_a`` and ``col_b``.
    col_a : str
        Name of the first categorical column (rows of the contingency table).
    col_b : str
        Name of the second categorical column (columns of the contingency
        table).
    dropna : bool
        If True, rows where either column is NaN are excluded before building
        the contingency table. If False, NaN is treated as its own category.
    correction : bool
        If True, apply Yates' continuity correction when the contingency table
        is 2×2. Has no effect for larger tables. Passed directly to
        ``scipy.stats.chi2_contingency``.

    Returns
    -------
    dict with keys:
        ``chi2`` : float
            The test statistic.
        ``p_value`` : float
            Two-tailed p-value.
        ``dof`` : int
            Degrees of freedom.
        ``contingency_table`` : pd.DataFrame
            Observed counts with ``col_a`` categories as rows and ``col_b``
            categories as columns.
        ``expected`` : pd.DataFrame
            Expected counts under the null hypothesis of independence, with
            the same shape and labels as ``contingency_table``.
        ``cramers_v`` : float
            Cramér's V effect size (0 = no association, 1 = perfect
            association). Bias-corrected using the Bergsma–Wicher formula.
        ``n`` : int
            Number of observations included in the test (after NaN handling).

    Raises
    ------
    ValueError
        If either column is not found in ``df``, or if the contingency table
        has fewer than 2 rows or 2 columns after dropping NaNs.

    Notes
    -----
    Assumptions:
    - Observations are independent.
    - Expected cell counts should generally be ≥ 5. A warning is emitted if
      more than 20% of cells have expected counts below this threshold —
      consider Fisher's exact test for small samples or sparse tables.

    Cramér's V interpretation (approximate):
        < 0.1  negligible, 0.1–0.2  weak, 0.2–0.4  moderate, > 0.4  strong
    """
    for col in (col_a, col_b):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in DataFrame.")

    data = df[[col_a, col_b]].dropna() if dropna else df[[col_a, col_b]].fillna("NaN")

    contingency_table = pd.crosstab(data[col_a], data[col_b])

    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        raise ValueError(
            "Contingency table must be at least 2×2. "
            "Check that both columns have at least 2 distinct categories."
        )

    chi2, p_value, dof, expected_arr = chi2_contingency(
        contingency_table, correction=correction
    )

    expected = pd.DataFrame(
        expected_arr,
        index=contingency_table.index,
        columns=contingency_table.columns,
    )

    low_expected = (expected_arr < 5).mean()
    if low_expected > 0.2:
        warnings.warn(
            f"{low_expected:.0%} of cells have expected counts < 5. "
            "Chi-squared results may be unreliable — consider Fisher's exact "
            "test or collapsing sparse categories.",
            UserWarning,
            stacklevel=2,
        )

    # Bias-corrected Cramér's V (Bergsma & Wicher 2012)
    n = data.shape[0]
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2_tilde = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_tilde = r - (r - 1) ** 2 / (n - 1)
    k_tilde = k - (k - 1) ** 2 / (n - 1)
    cramers_v = np.sqrt(phi2_tilde / min(k_tilde - 1, r_tilde - 1))

    return {
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "contingency_table": contingency_table,
        "expected": expected,
        "cramers_v": cramers_v,
        "n": n,
    }
