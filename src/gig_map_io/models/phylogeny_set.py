from functools import cached_property
from typing import Dict
from pathlib import Path

from .phylogeny import Phylogeny
from .dataset_dict import DatasetDict


class PhylogenySet(DatasetDict):
    """
    Representation of a set of pangenome phylogenies.
    """
    def __init__(self, directory_dict: Dict[str, str | Path]) -> None:
        super().__init__(directory_dict)

    @cached_property
    def phylogenies(self) -> Dict[str, Phylogeny]:
        return {key: Phylogeny(self.directory_dict[key]) for key in self.directory_dict.keys()}

    def __repr__(self) -> str:
        return f"PhylogenySet(directory_dict={self.directory_dict})"

    def __str__(self) -> str:
        return f"PhylogenySet(directory_dict={self.directory_dict})"

    def __format__(self, format_spec: str) -> str:
        return f"PhylogenySet(directory_dict={self.directory_dict})"

    def __getitem__(self, key: str) -> Phylogeny:
        return self.phylogenies[key]
