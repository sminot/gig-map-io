"""
PangenomeBin base class for gig-map-io.
"""

from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .dataset import Dataset


class PangenomeBin(Dataset):
    """
    Representation of a single pangenome bin (e.g., a MAG or species bin).

    DataFrame attributes are cached after first access to avoid reloading
    CSV files on subsequent accesses.
    """

    def __init__(self, directory: str | Path, bin_id: str) -> None:
        super().__init__(directory)
        self.bin_id = bin_id

    @cached_property
    def abundance(self) -> pd.DataFrame:
        """
        Abundance DataFrame loaded from <bin_id>_abundance.csv.

        Returns
        -------
        DataFrame containing abundance data for this bin.
        """
        filename = f"{self.bin_id}_abundance.csv"
        path = self.directory / filename
        return pd.read_csv(path)

    def load_abundance(self, filename: Optional[str | Path] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Load abundance information from a custom filename.

        Parameters
        ----------
        filename:
            Optional override for the filename; by default, will look up
            `<bin_id>_abundance.csv` in the directory.
        kwargs:
            Passed through to pandas `read_csv`.

        Returns
        -------
        DataFrame containing abundance data.
        """
        if filename is None:
            filename = f"{self.bin_id}_abundance.csv"
        path = self.directory / filename
        return pd.read_csv(path, **kwargs)
