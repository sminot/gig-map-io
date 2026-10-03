"""
Community detection and hierarchical ordering of sample-by-feature tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
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

    Builds a k-nearest-neighbor graph from the input features, then runs the
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
        Number of nearest neighbors used to build the KNN graph.
        Larger values capture more global structure. Typical values: 5–50.
    random_state : int
        Seed for reproducibility.
    scale : bool
        If True, standardize features to zero mean and unit variance before
        building the KNN graph. Suits Euclidean or cosine distances; turn it
        off for a dissimilarity defined only on non-negative values, such as
        Bray-Curtis.
    metric : str
        Distance metric used to compute nearest neighbors. Any metric
        supported by ``sklearn.neighbors.NearestNeighbors`` is valid
        (e.g. ``"cosine"``, ``"manhattan"``).
    **kwargs
        Any additional keyword arguments are forwarded to
        ``leidenalg.find_partition()``.

    Returns
    -------
    pd.Series
        Cluster labels "Cluster 1", "Cluster 2", ... indexed to match ``df``.

    The number of clusters is not specified directly: it emerges from
    ``resolution`` and the graph topology.
    """
    import igraph as ig
    import leidenalg

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


def linkage_order(
    matrix: np.ndarray,
    metric: str = "euclidean",
    method: str = "average",
    optimal: bool = False,
) -> np.ndarray:
    """
    Row indices in the order hierarchical clustering leaves them, so that
    similar rows sit together. ``optimal`` reorders the leaves to minimize
    the distance between neighbors, which costs more but reads better.
    Fewer than two rows come back as they are.
    """
    if matrix.shape[0] < 2:
        return np.arange(matrix.shape[0])
    distances = pdist(matrix, metric=metric)
    tree = linkage(distances, method=method)
    if optimal:
        tree = optimal_leaf_ordering(tree, distances)
    return leaves_list(tree)
