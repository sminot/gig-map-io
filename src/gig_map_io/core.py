"""
Compatibility re-exports for core data model classes.

This module maintains backward compatibility for code that imports from
`gig_map_io.core`. The actual implementations are now in dedicated modules:
- `gig_map_io.pangenome` - Pangenome class
- `gig_map_io.contrast_metagenomes` - ContrastMetagenomes class
- `gig_map_io.pangenome_bin` - PangenomeBin class
"""

from __future__ import annotations

from .contrast_metagenomes import ContrastMetagenomes
from .pangenome import Pangenome
from .pangenome_bin import PangenomeBin

__all__ = [
    "Pangenome",
    "ContrastMetagenomes",
    "PangenomeBin",
]
