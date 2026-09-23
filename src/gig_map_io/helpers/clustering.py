"""
Community detection and hierarchical ordering of sample-by-feature tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def leiden(
    df: pd.DataFrame,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    random_state: int = 42,
    scale: bool = True,
    metric: str = "euclidean",
    **kwargs,
) -> pd.Series:
    """
    Compute Leiden cluster labels from a DataFrame.

    Builds a k-nearest-neighbour graph from the input features, then runs the
    Leiden community-detection algorithm to assign cluster membership.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Rows are samples, columns are features.
    resolution : float
        Resolution parameter for the Leiden algorithm. Higher values produce
        more, smaller clusters; lower values produce fewer, larger clusters.
        Typical values: 0.1–2.0.
    n_neighbors : int
        Number of nearest neighbours used to build the KNN graph.
        Larger values capture more global structure. Typical values: 5–50.
    random_state : int
        Seed for reproducibility.
    scale : bool
        If True, standardise features to zero mean and unit variance before
        building the KNN graph (strongly recommended).
    metric : str
        Distance metric used to compute nearest neighbours. Any metric
        supported by ``sklearn.neighbors.NearestNeighbors`` is valid
        (e.g. ``"cosine"``, ``"manhattan"``).
    **kwargs
        Any additional keyword arguments are forwarded to
        ``leidenalg.find_partition()``.

    Returns
    -------
    pd.Series
        Integer cluster labels (0-indexed), indexed to match ``df``.
        Series name is ``"leiden"``.

    Raises
    ------
    ImportError
        If ``leidenalg`` or ``igraph`` are not installed.
    ValueError
        If ``df`` contains NaN or infinite values after optional scaling.

    Notes
    -----
    Install dependencies with::

        pip install leidenalg igraph scikit-learn

    The number of clusters is not specified directly — it emerges from
    ``resolution`` and the graph topology. Run with a range of ``resolution``
    values and inspect cluster sizes to find a suitable granularity.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:
        raise ImportError(
            "leidenalg and igraph are required: pip install leidenalg igraph"
        ) from e

    X = df.values
    if scale:
        X = StandardScaler().fit_transform(X)

    if not np.isfinite(X).all():
        raise ValueError(
            "Input contains NaN or infinite values after scaling. "
            "Check your data for missing or invalid entries."
        )

    # Build KNN graph
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(X)
    distances, indices = knn.kneighbors(X)

    # Convert to igraph edge list (exclude self-loops at position 0)
    n_samples = X.shape[0]
    sources = np.repeat(np.arange(n_samples), n_neighbors - 1)
    targets = indices[:, 1:].flatten()
    weights = (1 - distances[:, 1:] / distances[:, 1:].max()).flatten()

    g = ig.Graph(n=n_samples, edges=list(zip(sources, targets)), directed=False)
    g.es["weight"] = weights.tolist()
    g.simplify(combine_edges="mean")

    # Run Leiden
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_state,
        **kwargs,
    )

    labels = np.array(partition.membership)
    clusters = pd.Series(labels, index=df.index, name="leiden", dtype=int)
    clusters = (
        clusters
        .apply(lambda i: f'Cluster {i + 1}')
    )
    return clusters


def linkage_order(matrix: np.ndarray) -> np.ndarray:
    """
    Return row indices sorted by average-linkage hierarchical clustering.

    Parameters
    ----------
    matrix : np.ndarray
        2D array with observations as rows and features as columns.
        Euclidean distance is used.

    Returns
    -------
    np.ndarray
        Integer indices that reorder rows by cluster proximity.
    """
    if matrix.shape[0] < 2:
        return np.arange(matrix.shape[0])
    Z = linkage(pdist(matrix, metric='euclidean'), method='average')
    return leaves_list(Z)
