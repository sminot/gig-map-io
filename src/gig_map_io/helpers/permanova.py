"""
PERMANOVA over a sample-by-feature table, one metadata variable at a time.
"""

from __future__ import annotations

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix
from skbio.stats.distance import permanova as skbio_permanova


def permanova(
    scalars_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    distance_metric: str = "euclidean",
    n_permutations: int = 999,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Run PERMANOVA independently for each metadata category against a shared
    distance matrix, returning the proportion of variance explained per variable.

    Note: R² values are marginal (each variable tested alone) rather than
    jointly partitioned. Variables are not adjusted for one another.

    Parameters
    ----------
    scalars_df : pd.DataFrame
        Numeric feature matrix. Shape (n_samples, n_features).
        Index must match metadata_df index.
    metadata_df : pd.DataFrame
        Categorical or grouping metadata. Shape (n_samples, n_categories).
        Each column is tested as an independent grouping variable.
        Index must match scalars_df index.
    distance_metric : str
        Any metric accepted by scipy.spatial.distance.pdist, e.g.
        "euclidean", "braycurtis", "jaccard". Default: "euclidean".
    n_permutations : int
        Number of permutations for significance testing. Default: 999.
    seed : int | None
        Seed for reproducibility. Default: 42.

    Returns
    -------
    pd.DataFrame
        One row per metadata category, sorted by R² descending, with columns:
          - category      : metadata column name
          - r_squared     : marginal proportion of variance explained
          - f_statistic   : pseudo-F statistic
          - p_value       : permutation-based p-value
          - n_groups      : number of unique groups
          - n_samples     : samples used (after dropping NaN for that column)
    """
    # --- Align indices -----------------------------------------------------
    shared_idx = scalars_df.index.intersection(metadata_df.index)
    if len(shared_idx) == 0:
        raise ValueError("scalars_df and metadata_df share no common index values.")
    if len(shared_idx) < len(scalars_df):
        print(f"Warning: {len(scalars_df) - len(shared_idx)} sample(s) dropped — not present in both DataFrames.")

    scalars_df  = scalars_df.loc[shared_idx]
    metadata_df = metadata_df.loc[shared_idx]

    # --- Drop samples missing any metadata column -------------------------
    complete_mask = metadata_df.notna().all(axis=1)
    n_dropped = (~complete_mask).sum()
    if n_dropped > 0:
        print(f"Warning: {n_dropped} sample(s) dropped due to missing metadata.")
    scalars_df  = scalars_df.loc[complete_mask]
    metadata_df = metadata_df.loc[complete_mask]

    if len(scalars_df) < 3:
        raise ValueError("Fewer than 3 complete samples remain after dropping missing values.")

    # --- Build distance matrix once on the complete sample set ------------
    dist_sq = squareform(pdist(scalars_df.values, metric=distance_metric))
    all_ids = scalars_df.index.astype(str).tolist()
    skbio_dm = DistanceMatrix(dist_sq, ids=all_ids)

    results = []

    for col in metadata_df.columns:
        grouping = metadata_df[col].astype(str)
        n_groups = grouping.nunique()

        if n_groups < 2:
            print(f"Skipping '{col}': only one unique group value.")
            continue

        result = skbio_permanova(
            distance_matrix=skbio_dm,
            grouping=grouping,
            permutations=n_permutations,
            seed=seed,
        )

        f = result["test statistic"]
        k = result["number of groups"]
        n = result["sample size"]
        r2 = (f * (k - 1)) / (f * (k - 1) + (n - k))

        results.append({
            "category":    col,
            "r_squared":   r2,
            "f_statistic": f,
            "p_value":     result["p-value"],
            "n_groups":    k,
            "n_samples":   n,
        })

    if not results:
        raise RuntimeError("No valid metadata columns found to run PERMANOVA.")

    return (
        pd.DataFrame(results)
        .sort_values("r_squared", ascending=False)
        .reset_index(drop=True)
    )
