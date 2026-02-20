"""
gig-map-io: read outputs of the gig-map workflow and run common visualizations.

This library provides:

- **Reader objects** (one per gig-map workflow): each reads the key outputs of
  that workflow. Use ``Pangenome``, ``ContrastMetagenomes``, and
  ``PangenomePhylogeny`` for the main workflows; ``PangenomeBin`` for per-bin
  data. ``CompareMultipleMetagenomes`` wraps a dict of ``ContrastMetagenomes``
  keyed by pangenome name to combine analyses across pangenomes.

- **Plotting functions**: take one or more reader objects and produce common
  plots (volcano, bin abundance, bin phylogeny).
"""

from .core import (
    CompareMultipleMetagenomes,
    ContrastMetagenomes,
    Pangenome,
    PangenomeBin,
    PangenomePhylogeny,
)
from .parameters import Parameters
from .plots import (
    plot_bin_abundance,
    plot_bin_phylogeny,
    plot_compare_contrasts,
    plot_double_volcano,
    plot_estimate_scatter,
    plot_volcano,
)

__all__ = [
    "CompareMultipleMetagenomes",
    "ContrastMetagenomes",
    "Pangenome",
    "PangenomeBin",
    "PangenomePhylogeny",
    "Parameters",
    "plot_bin_abundance",
    "plot_bin_phylogeny",
    "plot_compare_contrasts",
    "plot_double_volcano",
    "plot_estimate_scatter",
    "plot_volcano",
]

