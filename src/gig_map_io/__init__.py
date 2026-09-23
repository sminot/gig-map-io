"""
gig-map-io: read outputs of the gig-map workflow and run common visualizations.

This library provides:

- **Reader objects** (one per gig-map workflow): each reads the key outputs of
  that workflow. Use ``Pangenome``, ``ContrastMetagenomes``, and
  ``PangenomePhylogeny`` for the main workflows. The ``*Set`` variants wrap a
  dict of readers keyed by organism, to combine analyses across pangenomes.

- **Study definitions**: ``Study`` names the gig-map outputs that belong to one
  comparison and is serialized as JSON, so an analysis script loads a study by
  name and calls a single method on it. ``StudySet`` spans several studies.

- **``AnalysisScript``**: the command-line context for one analysis script --
  where its study definitions, its inputs from earlier steps, and its outputs
  live. Defaults work from the root of a checkout; a workflow overrides them.

- **Plotting and analysis methods** on those objects: volcano plots, contrast
  comparisons, pangenome summaries, community ordination, and clustering.
"""

from .models import (
    ContrastMetagenomesSet,
    ContrastMetagenomes,
    Pangenome,
    PangenomeSet,
    PangenomePhylogeny,
    PangenomePhylogenySet,
    SampleGroup,
    Study,
    StudySet,
)
from .parameters import Parameters
from .script import AnalysisScript

__all__ = [
    "ContrastMetagenomesSet",
    "ContrastMetagenomes",
    "Pangenome",
    "PangenomeSet",
    "PangenomePhylogeny",
    "PangenomePhylogenySet",
    "SampleGroup",
    "Study",
    "StudySet",
    "Parameters",
    "AnalysisScript",
]
