"""
Dataset base class for gig-map-io models that read from a single directory.
"""

from pathlib import Path


class Dataset:
    """
    Base class for models that represent gig-map output from a single directory.

    Accepts a directory as a string or Path, resolves it to a Path, and
    validates that it exists and is a directory. Subclasses use self.directory
    to read workflow outputs.
    """
    directory: Path

    def __init__(self, directory: str | Path) -> None:
        path = Path(directory).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        self.directory = path
