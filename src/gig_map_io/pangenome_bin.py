"""
PangenomeBin base class for gig-map-io.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass
class PangenomeBin:
    """
    Representation of a single pangenome bin (e.g., a MAG or species bin).
    """

    directory: Path
    bin_id: str

    def abundance(self, filename: Optional[str | Path] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Load abundance information for this bin.

        Parameters
        ----------
        filename:
            Optional override for the filename; by default, will look up
            `<bin_id>_abundance.csv` in the directory.
        kwargs:
            Passed through to pandas `read_csv`.
        """
        if filename is None:
            filename = f"{self.bin_id}_abundance.csv"
        path = self.directory / filename
        return pd.read_csv(path, **kwargs)


