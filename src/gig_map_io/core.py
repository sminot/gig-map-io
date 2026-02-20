"""
Compatibility re-exports for core data model classes.

This module maintains backward compatibility for code that imports from
`gig_map_io.core`. The actual implementations live in `gig_map_io.models`.
"""

from __future__ import annotations

from .models import (
    CompareMultipleMetagenomes,
    ContrastMetagenomes,
    Pangenome,
    PangenomeBin,
    PangenomePhylogeny,
)

__all__ = [
    "CompareMultipleMetagenomes",
    "ContrastMetagenomes",
    "Pangenome",
    "PangenomeBin",
    "PangenomePhylogeny",
]
