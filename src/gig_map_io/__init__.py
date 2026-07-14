"""
gig-map-io: read outputs of the gig-map workflow and run common visualizations.

This library provides:

- **Reader objects** (one per gig-map workflow): each reads the key outputs of
  that workflow. Use ``Pangenome``, ``ContrastMetagenomes``, and
  ``PangenomePhylogeny`` for the main workflows.
  ``ContrastMetagenomesSet`` wraps a dict of ``ContrastMetagenomes``
  keyed by pangenome name to combine analyses across pangenomes.

- **Plotting functions**: take one or more reader objects and produce common
  plots (volcano, bin abundance, bin phylogeny).
"""

from .core import (
    ContrastMetagenomesSet,
    ContrastMetagenomes,
    Pangenome,
    PangenomeSet,
    PangenomePhylogeny,
    PangenomePhylogenySet,
)
from .parameters import Parameters

__all__ = [
    "ContrastMetagenomesSet",
    "ContrastMetagenomes",
    "Pangenome",
    "PangenomeSet",
    "PangenomePhylogeny",
    "PangenomePhylogenySet",
    "Parameters"
]

