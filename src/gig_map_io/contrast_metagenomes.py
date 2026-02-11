"""
ContrastMetagenomes base class for gig-map-io.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ContrastMetagenomes:
    """
    Representation of a contrast between metagenomes (e.g., case vs control).
    """

    directory: Path

    def summary(self, filename: str | Path = "contrast_summary.csv", **kwargs: Any) -> pd.DataFrame:
        """
        Load a high-level summary table for the contrast.

        Parameters
        ----------
        filename:
            Name of the CSV file relative to the directory.
        kwargs:
            Passed through to pandas `read_csv`.
        """
        path = self.directory / filename
        return pd.read_csv(path, **kwargs)


