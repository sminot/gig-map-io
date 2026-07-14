"""
I/O helper utilities for gig-map-io.

The goal of this module is to provide a backend-agnostic interface for reading
tabular data (e.g., CSVs) that can later be extended to support remote
storage, databases, object stores, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class TableBackend(Protocol):
    """Abstract interface for reading tabular data from an arbitrary backend."""

    def read_table(self, resource: str | Path, *, fmt: str = "csv", **kwargs: Any) -> pd.DataFrame:  # pragma: no cover - protocol
        """
        Read a table identified by `resource`.

        Implementations are free to interpret `resource` as a local path,
        URL, database table, object store key, etc.
        """


@dataclass
class LocalCSVBackend:
    """
    Simple backend that reads CSV/TSV files from the local filesystem.

    This is intended as a reference implementation; other backends can be
    created for remote object stores, HTTP(S) endpoints, etc.
    """

    base: Path | None = None

    def _resolve(self, resource: str | Path) -> Path:
        path = Path(resource)
        if self.base is not None and not path.is_absolute():
            path = self.base / path
        return path

    def read_table(self, resource: str | Path, *, fmt: str = "csv", **kwargs: Any) -> pd.DataFrame:
        path = self._resolve(resource)

        if fmt == "csv":
            return pd.read_csv(path, **kwargs)
        if fmt in {"tsv", "tab"}:
            return pd.read_csv(path, sep="\t", **kwargs)

        raise ValueError(f"Unsupported format: {fmt!r}")


def default_local_backend(base: str | Path | None = None) -> LocalCSVBackend:
    """
    Convenience constructor for a local CSV backend.

    Parameters
    ----------
    base:
        Optional base directory for resolving relative resources.
    """
    return LocalCSVBackend(base=Path(base) if base is not None else None)

