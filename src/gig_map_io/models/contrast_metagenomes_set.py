from functools import cached_property
from logging import getLogger
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterator

from plotly.subplots import make_subplots
from plotly import graph_objects as go

from gig_map_io.helpers.save_image import save_image

from .contrast_metagenomes import ContrastMetagenomes
from .dataset_dict import DatasetDict

logger = getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

class ContrastMetagenomesSet(DatasetDict):
    """
    Representation of a set of contrast-metagenomes.
    """
    def __init__(self, directory_dict: Dict[str, str | Path]) -> None:
        super().__init__(directory_dict)

    @cached_property
    def contrast_metagenomes(self) -> Dict[str, ContrastMetagenomes]:
        return {key: ContrastMetagenomes(self.directory_dict[key]) for key in self.directory_dict.keys()}

    def __repr__(self) -> str:
        return f"ContrastMetagenomesSet(directory_dict={self.directory_dict})"

    def __str__(self) -> str:
        return f"ContrastMetagenomesSet(directory_dict={self.directory_dict})"

    def __format__(self, format_spec: str) -> str:
        return f"ContrastMetagenomesSet(directory_dict={self.directory_dict})"

    def __len__(self) -> int:
        return len(self.contrast_metagenomes)

    def __contains__(self, pangenome_name: str) -> bool:
        return pangenome_name in self.contrast_metagenomes

    def __getitem__(self, pangenome_name: str) -> ContrastMetagenomes:
        return self.contrast_metagenomes[pangenome_name]

    def __iter__(self) -> Iterator[tuple[str, ContrastMetagenomes]]:
        return self.contrast_metagenomes.items()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.contrast_metagenomes, name)