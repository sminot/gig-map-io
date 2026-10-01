"""
Dataset base class for gig-map-io models that read from a single directory.
"""

from pathlib import Path


def existing_directory(directory: str | Path) -> Path:
    """The directory as a resolved path, which must exist."""
    path = Path(directory).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    return path


class Dataset:
    """
    Base class for models that represent gig-map output from a single directory.
    Subclasses use self.directory to read workflow outputs.
    """
    directory: Path

    def __init__(self, directory: str | Path) -> None:
        self.directory = existing_directory(directory)
