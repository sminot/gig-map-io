"""
Data models for gig-map workflow outputs.

Each module provides a reader object for one gig-map workflow, capable of
reading the key outputs of that workflow. CompareMultipleMetagenomes combines
multiple ContrastMetagenomes keyed by pangenome name (no single directory).
"""

from .compare_multiple_metagenomes import CompareMultipleMetagenomes
from .contrast_metagenomes import ContrastMetagenomes
from .pangenome import Pangenome
from .pangenome_phylogeny import PangenomePhylogeny
from .pangenome_set import PangenomeSet

__all__ = [
    "CompareMultipleMetagenomes",
    "ContrastMetagenomes",
    "Pangenome",
    "PangenomePhylogeny",
    "PangenomeSet",
]
