"""
Data models for gig-map workflow outputs.

Each module provides a reader object for one gig-map workflow. The ``*Set``
classes combine several readers keyed by organism, and ``Study`` /
``StudySet`` name the combinations that make up one analysis.
"""

from .contrast_metagenomes_set import ContrastMetagenomesSet
from .contrast_metagenomes import ContrastMetagenomes
from .pangenome import Pangenome
from .phylogeny import PangenomePhylogeny
from .phylogeny_set import PangenomePhylogenySet
from .pangenome_set import PangenomeSet
from .sample_group import SampleGroup
from .study import Study
from .study_set import StudySet

__all__ = [
    "ContrastMetagenomesSet",
    "ContrastMetagenomes",
    "Pangenome",
    "PangenomePhylogeny",
    "PangenomePhylogenySet",
    "PangenomeSet",
    "SampleGroup",
    "Study",
    "StudySet",
]
