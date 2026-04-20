from functools import cached_property
from typing import Dict
from pathlib import Path

from .phylogeny import PangenomePhylogeny
from .dataset_dict import DatasetDict
from ..helpers.phylogeny import Phylogeny


class PangenomePhylogenySet(DatasetDict):
    """
    Representation of a set of pangenome phylogenies.
    """
    def __init__(self, directory_dict: Dict[str, str | Path]) -> None:
        super().__init__(directory_dict)

    @cached_property
    def phylogenies(self) -> Dict[str, PangenomePhylogeny]:
        return {key: PangenomePhylogeny(self.directory_dict[key]) for key in self.directory_dict.keys()}

    def __repr__(self) -> str:
        return f"PangenomePhylogenySet(directory_dict={self.directory_dict})"

    def __str__(self) -> str:
        return f"PangenomePhylogenySet(directory_dict={self.directory_dict})"

    def __format__(self, format_spec: str) -> str:
        return f"PangenomePhylogenySet(directory_dict={self.directory_dict})"

    def __getitem__(self, key: str) -> PangenomePhylogeny:
        return self.phylogenies[key]

    def tree(self, pangenome_name: str, bin_id: str) -> Phylogeny:
        return self.phylogenies[pangenome_name].tree(bin_id)

    def newick(self, pangenome_name: str, bin_id: str) -> str:
        return self.phylogenies[pangenome_name].newick(bin_id)