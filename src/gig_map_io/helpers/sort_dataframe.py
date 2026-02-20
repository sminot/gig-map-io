import pandas as pd
from scipy.cluster import hierarchy


def sort_dataframe(df: pd.DataFrame, metric: str = "euclidean", method: str = "average") -> pd.DataFrame:
    """
    Sort a dataframe using hierarchical clustering.

    Parameters
    ----------
    df: pd.DataFrame
    metric: str
    method: str

    Returns
    -------
    pd.DataFrame
    """
    return df.iloc[
        hierarchy.leaves_list(
            hierarchy.linkage(
                df.values,
                method=method,
                metric=metric
            )
        )
    ]
