"""
Dataset base class for gig-map-io models that read from one directory per organism.
"""

from typing import Dict
from pathlib import Path

from .dataset import existing_directory


class DatasetDict:
    """
    Base class for models that represent gig-map output from a dictionary of
    directories keyed by organism. Subclasses use self.directory_dict to read
    workflow outputs.
    """
    directory_dict: Dict[str, Path]

    def __init__(self, directory_dict: Dict[str, str | Path]) -> None:
        self.directory_dict = {key: existing_directory(value) for key, value in directory_dict.items()}
