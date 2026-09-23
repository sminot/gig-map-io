"""
Declarative recoding of contrast metadata columns into named sample groups.

A gig-map contrast carries whatever metadata columns the workflow was run
with, using whatever encoding the study used (``batch_1`` of 0/1, ``disease``
of 0/1, a ``study_name`` that is null for the study it was assembled around).
A ``SampleGroup`` describes how to turn one or more of those raw columns into
a single labelled categorical column, so that the mapping lives in the study
definition rather than in analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


def _key(value: Any) -> str:
    """Normalize a metadata value to the string form used to look up labels."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@dataclass
class SampleGroupSource:
    """
    One candidate source of values for a sample group.

    Exactly one of ``column`` (read from the metadata) or ``value`` (a
    constant) must be given. ``split``/``index`` extract a substring from the
    raw value, ``labels`` renames values, and ``default`` fills nulls.
    """

    column: str | None = None
    value: str | None = None
    split: str | None = None
    index: int | None = None
    labels: Dict[str, str] = field(default_factory=dict)
    default: str | None = None

    def __post_init__(self) -> None:
        if (self.column is None) == (self.value is None):
            raise ValueError("SampleGroupSource requires exactly one of 'column' or 'value'")
        if (self.split is None) != (self.index is None):
            raise ValueError("SampleGroupSource 'split' and 'index' must be given together")

    def resolve(self, metadata: pd.DataFrame) -> pd.Series:
        """Return the labelled values for this source, aligned to `metadata`."""
        if self.value is not None:
            return pd.Series(self.value, index=metadata.index, dtype=object)

        if self.column not in metadata.columns:
            raise KeyError(f"Metadata has no column {self.column!r}")

        values = metadata[self.column].map(
            lambda v: self._label(v) if pd.notnull(v) else None,
            na_action=None,
        )
        if self.default is not None:
            values = values.fillna(self.default)
        return values.astype(object)

    def _label(self, value: Any) -> str:
        if self.split is not None:
            value = str(value).split(self.split)[self.index]
        return self.labels.get(_key(value), _key(value))

    def to_dict(self) -> dict:
        out: dict = {}
        for key in ("column", "value", "split", "index", "default"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.labels:
            out["labels"] = self.labels
        return out


@dataclass
class SampleGroup:
    """
    A named categorical annotation of the samples in a study.

    ``sources`` are tried in order for each sample; the first one yielding a
    non-null value wins. ``order`` gives the category order to use in plots.
    """

    sources: List[SampleGroupSource]
    order: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, spec: dict) -> "SampleGroup":
        return cls(
            sources=[SampleGroupSource(**src) for src in spec["sources"]],
            order=list(spec.get("order", [])),
        )

    def to_dict(self) -> dict:
        out: dict = {"sources": [src.to_dict() for src in self.sources]}
        if self.order:
            out["order"] = self.order
        return out

    def resolve(self, metadata: pd.DataFrame) -> pd.Series:
        """Combine the sources into a single labelled column."""
        values = pd.Series(None, index=metadata.index, dtype=object)
        for source in self.sources:
            values = values.fillna(source.resolve(metadata))
        return values
