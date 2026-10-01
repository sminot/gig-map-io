from functools import cached_property
from typing import Dict

from .phylogeny import PangenomePhylogeny
from .dataset_dict import DatasetDict


class PangenomePhylogenySet(DatasetDict):
    """
    Representation of a set of pangenome phylogenies.
    """

    @cached_property
    def phylogenies(self) -> Dict[str, PangenomePhylogeny]:
        return {key: PangenomePhylogeny(self.directory_dict[key]) for key in self.directory_dict.keys()}

    def __repr__(self) -> str:
        return f"PangenomePhylogenySet(directory_dict={self.directory_dict})"

    def __getitem__(self, key: str) -> PangenomePhylogeny:
        return self.phylogenies[key]