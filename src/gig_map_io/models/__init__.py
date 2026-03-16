"""
Data models for gig-map workflow outputs.

Each module provides a reader object for one gig-map workflow, capable of
reading the key outputs of that workflow. ContrastMetagenomesSet combines
multiple ContrastMetagenomes keyed by pangenome name (no single directory).
"""

from .contrast_metagenomes_set import ContrastMetagenomesSet
from .contrast_metagenomes import ContrastMetagenomes
from .pangenome import Pangenome
from .phylogeny import PangenomePhylogeny
from .phylogeny_set import PangenomePhylogenySet
from .pangenome_set import PangenomeSet

__all__ = [
    "ContrastMetagenomesSet",
    "ContrastMetagenomes",
    "Pangenome",
    "PangenomePhylogeny",
    "PangenomePhylogenySet",
    "PangenomeSet",
]
