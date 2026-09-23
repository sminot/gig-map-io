"""
Dimensionality reduction used to summarize sample-by-feature tables.
"""

from __future__ import annotations

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def tsne(
    df: pd.DataFrame,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
    scale: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Compute t-SNE coordinates from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Rows are samples, columns are features.
    n_components : int
        Number of t-SNE dimensions (2 or 3).
    perplexity : float
        t-SNE perplexity. Roughly the number of effective nearest neighbours.
        Typical values: 5–50.
    random_state : int
        Seed for reproducibility.
    scale : bool
        If True, standardise features to zero mean and unit variance before
        running t-SNE (strongly recommended).

    Any additional keyword arguments will be passed to TSNE()

    Returns
    -------
    pd.DataFrame
        Columns ``t-SNE 1``, ``t-SNE 2`` (and ``t-SNE 3`` if
        n_components=3), preserving the index of ``df``.
    """
    X = df.values

    if scale:
        X = StandardScaler().fit_transform(X)

    coords = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        **kwargs
    ).fit_transform(X)

    coord_cols = [f"t-SNE {i + 1}" for i in range(n_components)]
    tsne_df = pd.DataFrame(coords, index=df.index, columns=coord_cols)

    return tsne_df
