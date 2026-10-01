"""
Contingency testing between categorical sample annotations.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def chi2_contingency_test(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """
    A chi-squared test of independence between two categorical columns,
    ignoring rows where either is missing.

    Returns a dict with the test (``chi2``, ``p_value``, ``dof``), the
    ``contingency_table`` and the ``expected`` counts under independence,
    the number of observations ``n``, and ``cramers_v``, the effect size
    with the Bergsma-Wicher bias correction: below 0.1 is negligible, 0.1
    to 0.2 weak, 0.2 to 0.4 moderate, above 0.4 strong. Warns when more
    than a fifth of the expected counts are below 5.
    """
    data = df[[col_a, col_b]].dropna()
    contingency_table = pd.crosstab(data[col_a], data[col_b])
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        raise ValueError(
            "Contingency table must be at least 2x2. "
            "Check that both columns have at least 2 distinct categories."
        )

    chi2, p_value, dof, expected_arr = chi2_contingency(contingency_table)

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
