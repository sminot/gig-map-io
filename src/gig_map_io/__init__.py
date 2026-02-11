"""
Top-level package for gig-map-io.

This package provides utilities for parsing and analyzing outputs of the
gig-map (genes-in-genomes map) workflow.
"""

from .core import Pangenome, ContrastMetagenomes, PangenomeBin

__all__ = [
    "Pangenome",
    "ContrastMetagenomes",
    "PangenomeBin",
]

