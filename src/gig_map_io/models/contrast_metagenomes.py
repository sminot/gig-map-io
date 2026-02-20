"""
ContrastMetagenomes base class for gig-map-io.
"""

from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import Dataset


class ContrastMetagenomes(Dataset):
    """
    Representation of a contrast between metagenomes (e.g., case vs control).

    Reads key outputs from the contrast_metagenomes workflow: summary,
    association results, and optionally bin abundance (RPKM). DataFrame
    attributes are cached after first access.
    """

    @cached_property
    def summary(self) -> pd.DataFrame:
        """
        Summary DataFrame loaded from contrast_summary.csv.

        Returns
        -------
        DataFrame containing contrast summary data.
        """
        path = self.directory / "contrast_summary.csv"
        return pd.read_csv(path)

    @cached_property
    def association(self) -> pd.DataFrame:
        """
        Association results from association/association.csv.

        Returns
        -------
        DataFrame with columns such as feature, Estimate, SE, pvalue, etc.
        """
        path = self.directory / "association" / "association.csv"
        return pd.read_csv(path)

    @cached_property
    def rpkm(self) -> pd.DataFrame | None:
        """
        Bin abundance (RPKM) from bin_abundance/rpkm.csv or .csv.gz if present.

        Returns
        -------
        DataFrame with specimens as index and bins as columns, or None if missing.
        """
        for name in ("rpkm.csv.gz", "rpkm.csv"):
            path = self.directory / "bin_abundance" / name
            if path.exists():
                df = pd.read_csv(path)
                if "specimen" in df.columns:
                    return df.set_index("specimen")
                return df
        return None

    def load_summary(self, filename: str | Path = "contrast_summary.csv", **kwargs: Any) -> pd.DataFrame:
        """
        Load a summary table from a custom filename.

        Parameters
        ----------
        filename:
            Name of the CSV file relative to the directory.
        kwargs:
            Passed through to pandas `read_csv`.

        Returns
        -------
        DataFrame containing contrast summary data.
        """
        path = self.directory / filename
        return pd.read_csv(path, **kwargs)
