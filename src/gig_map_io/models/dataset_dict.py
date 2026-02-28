"""
Dataset base class for gig-map-io models that read from a single directory.
"""

from typing import Dict
from pathlib import Path


class DatasetDict:
    """
    Base class for models that represent gig-map output from a dictionary of directories.

    Accepts a dictionary of directories as a string or Path, resolves it to a Path, and
    validates that it exists and is a directory. Subclasses use self.directory_dict
    to read workflow outputs.
    """
    directory_dict: Dict[str, Path]

    def __init__(self, directory_dict: Dict[str, str | Path]) -> None:
        self.directory_dict = {}
        for key, value in directory_dict.items():
            path = Path(value).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Directory does not exist: {path}")
            if not path.is_dir():
                raise NotADirectoryError(f"Not a directory: {path}")
            self.directory_dict[key] = path
