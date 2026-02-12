"""
Catalog class for tracking pangenomes, ContrastMetagenomes, and PangenomePhylogeny directories.

The Catalog automatically loads from and saves to `.gig-map/catalog.json` in the
current working directory. Every ContrastMetagenomes directory is linked to a
single Pangenome. Every PangenomePhylogeny directory is also linked to a single Pangenome.
"""

import json
import os
from pathlib import Path
from typing import Any


class Catalog:
    """
    Catalog for tracking pangenome and contrast metagenome directories.

    Automatically loads from `$PWD/.gig-map/catalog.json` on initialization
    and saves changes automatically when the catalog is modified.
    """

    def __init__(self, catalog_path: Path | None = None):
        """
        Initialize the catalog.

        Parameters
        ----------
        catalog_path:
            Optional path to catalog file. If not provided, uses the value of
            the ``GIG_MAP_CATALOG`` environment variable when set, falling back
            to ``$HOME/.gig-map/catalog.json``.
        """
        if catalog_path is None:
            env_catalog = os.environ.get("GIG_MAP_CATALOG")
            if env_catalog:
                catalog_path = Path(env_catalog)
            else:
                catalog_path = Path(os.environ.get("HOME", str(Path.home()))) / ".gig-map" / "catalog.json"
        else:
            catalog_path = Path(catalog_path)

        self._catalog_path = catalog_path
        self._pangenomes: dict[str, str] = {}  # pangenome_id -> directory path
        self._contrasts: dict[str, dict[str, str]] = {}  # contrast_id -> {directory, pangenome_id}
        self._phylogenies: dict[str, dict[str, str]] = {}  # phylogeny_id -> {directory, pangenome_id}

        # Load existing catalog if it exists
        self._load()

    def _load(self) -> None:
        """Load catalog data from JSON file."""
        if self._catalog_path.exists():
            with open(self._catalog_path, "r") as f:
                data = json.load(f)
                self._pangenomes = data.get("pangenomes", {})
                self._contrasts = data.get("contrasts", {})
                self._phylogenies = data.get("phylogenies", {})
        else:
            # Create directory if it doesn't exist
            self._catalog_path.parent.mkdir(parents=True, exist_ok=True)
            self._pangenomes = {}
            self._contrasts = {}
            self._phylogenies = {}

    def _save(self) -> None:
        """Save catalog data to JSON file."""
        # Ensure directory exists
        self._catalog_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "pangenomes": self._pangenomes,
            "contrasts": self._contrasts,
            "phylogenies": self._phylogenies,
        }

        with open(self._catalog_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_pangenome(self, directory: Path | str, pangenome_id: str | None = None) -> None:
        """
        Add a pangenome directory to the catalog.

        Parameters
        ----------
        directory:
            Path to the pangenome directory.
        pangenome_id:
            Optional identifier for the pangenome. If not provided, uses the
            directory name as the ID.
        """
        directory = Path(directory).resolve()
        if pangenome_id is None:
            pangenome_id = directory.name

        self._pangenomes[pangenome_id] = str(directory)
        self._save()

    def add_contrast(
        self, directory: Path | str, pangenome_id: str, contrast_id: str | None = None
    ) -> None:
        """
        Add a contrast metagenome directory to the catalog.

        Parameters
        ----------
        directory:
            Path to the contrast directory.
        pangenome_id:
            Identifier of the pangenome this contrast is linked to.
        contrast_id:
            Optional identifier for the contrast. If not provided, uses the
            directory name as the ID.
        """
        if pangenome_id not in self._pangenomes:
            raise ValueError(f"Pangenome '{pangenome_id}' not found in catalog")

        directory = Path(directory).resolve()
        if contrast_id is None:
            contrast_id = directory.name

        self._contrasts[contrast_id] = {
            "directory": str(directory),
            "pangenome_id": pangenome_id,
        }
        self._save()

    def add_phylogeny(
        self, directory: Path | str, pangenome_id: str, phylogeny_id: str | None = None
    ) -> None:
        """
        Add a pangenome phylogeny directory to the catalog.

        Parameters
        ----------
        directory:
            Path to the phylogeny directory.
        pangenome_id:
            Identifier of the pangenome this phylogeny is linked to.
        phylogeny_id:
            Optional identifier for the phylogeny. If not provided, uses the
            directory name as the ID.
        """
        if pangenome_id not in self._pangenomes:
            raise ValueError(f"Pangenome '{pangenome_id}' not found in catalog")

        directory = Path(directory).resolve()
        if phylogeny_id is None:
            phylogeny_id = directory.name

        self._phylogenies[phylogeny_id] = {
            "directory": str(directory),
            "pangenome_id": pangenome_id,
        }
        self._save()

    def remove_pangenome(self, pangenome_id: str) -> None:
        """
        Remove a pangenome from the catalog.

        Also removes all contrasts and phylogenies linked to this pangenome.

        Parameters
        ----------
        pangenome_id:
            Identifier of the pangenome to remove.
        """
        if pangenome_id not in self._pangenomes:
            raise ValueError(f"Pangenome '{pangenome_id}' not found in catalog")

        # Remove all contrasts linked to this pangenome
        contrasts_to_remove = [
            contrast_id
            for contrast_id, contrast_data in self._contrasts.items()
            if contrast_data["pangenome_id"] == pangenome_id
        ]
        for contrast_id in contrasts_to_remove:
            del self._contrasts[contrast_id]

        # Remove all phylogenies linked to this pangenome
        phylogenies_to_remove = [
            phylogeny_id
            for phylogeny_id, phylogeny_data in self._phylogenies.items()
            if phylogeny_data["pangenome_id"] == pangenome_id
        ]
        for phylogeny_id in phylogenies_to_remove:
            del self._phylogenies[phylogeny_id]

        del self._pangenomes[pangenome_id]
        self._save()

    def remove_contrast(self, contrast_id: str) -> None:
        """
        Remove a contrast from the catalog.

        Parameters
        ----------
        contrast_id:
            Identifier of the contrast to remove.
        """
        if contrast_id not in self._contrasts:
            raise ValueError(f"Contrast '{contrast_id}' not found in catalog")

        del self._contrasts[contrast_id]
        self._save()

    def get_pangenomes(self) -> dict[str, Path]:
        """
        Get all pangenomes in the catalog.

        Returns
        -------
        Dictionary mapping pangenome IDs to their directory paths.
        """
        return {pid: Path(path) for pid, path in self._pangenomes.items()}

    def get_contrasts(self) -> dict[str, dict[str, Any]]:
        """
        Get all contrasts in the catalog.

        Returns
        -------
        Dictionary mapping contrast IDs to their data (directory and pangenome_id).
        """
        return {
            cid: {
                "directory": Path(data["directory"]),
                "pangenome_id": data["pangenome_id"],
            }
            for cid, data in self._contrasts.items()
        }

    def get_contrasts_for_pangenome(self, pangenome_id: str) -> dict[str, Path]:
        """
        Get all contrasts linked to a specific pangenome.

        Parameters
        ----------
        pangenome_id:
            Identifier of the pangenome.

        Returns
        -------
        Dictionary mapping contrast IDs to their directory paths.
        """
        if pangenome_id not in self._pangenomes:
            raise ValueError(f"Pangenome '{pangenome_id}' not found in catalog")

        return {
            contrast_id: Path(contrast_data["directory"])
            for contrast_id, contrast_data in self._contrasts.items()
            if contrast_data["pangenome_id"] == pangenome_id
        }

    def get_pangenome(self, pangenome_id: str) -> Path:
        """
        Get the directory path for a specific pangenome.

        Parameters
        ----------
        pangenome_id:
            Identifier of the pangenome.

        Returns
        -------
        Path to the pangenome directory.
        """
        if pangenome_id not in self._pangenomes:
            raise ValueError(f"Pangenome '{pangenome_id}' not found in catalog")
        return Path(self._pangenomes[pangenome_id])

    def get_contrast(self, contrast_id: str) -> dict[str, Any]:
        """
        Get the data for a specific contrast.

        Parameters
        ----------
        contrast_id:
            Identifier of the contrast.

        Returns
        -------
        Dictionary with 'directory' (Path) and 'pangenome_id' (str).
        """
        if contrast_id not in self._contrasts:
            raise ValueError(f"Contrast '{contrast_id}' not found in catalog")
        data = self._contrasts[contrast_id]
        return {
            "directory": Path(data["directory"]),
            "pangenome_id": data["pangenome_id"],
        }

    def remove_phylogeny(self, phylogeny_id: str) -> None:
        """
        Remove a phylogeny from the catalog.

        Parameters
        ----------
        phylogeny_id:
            Identifier of the phylogeny to remove.
        """
        if phylogeny_id not in self._phylogenies:
            raise ValueError(f"Phylogeny '{phylogeny_id}' not found in catalog")

        del self._phylogenies[phylogeny_id]
        self._save()

    def get_phylogenies(self) -> dict[str, dict[str, Any]]:
        """
        Get all phylogenies in the catalog.

        Returns
        -------
        Dictionary mapping phylogeny IDs to their data (directory and pangenome_id).
        """
        return {
            pid: {
                "directory": Path(data["directory"]),
                "pangenome_id": data["pangenome_id"],
            }
            for pid, data in self._phylogenies.items()
        }

    def get_phylogenies_for_pangenome(self, pangenome_id: str) -> dict[str, Path]:
        """
        Get all phylogenies linked to a specific pangenome.

        Parameters
        ----------
        pangenome_id:
            Identifier of the pangenome.

        Returns
        -------
        Dictionary mapping phylogeny IDs to their directory paths.
        """
        if pangenome_id not in self._pangenomes:
            raise ValueError(f"Pangenome '{pangenome_id}' not found in catalog")

        return {
            phylogeny_id: Path(phylogeny_data["directory"])
            for phylogeny_id, phylogeny_data in self._phylogenies.items()
            if phylogeny_data["pangenome_id"] == pangenome_id
        }

    def get_phylogeny(self, phylogeny_id: str) -> dict[str, Any]:
        """
        Get the data for a specific phylogeny.

        Parameters
        ----------
        phylogeny_id:
            Identifier of the phylogeny.

        Returns
        -------
        Dictionary with 'directory' (Path) and 'pangenome_id' (str).
        """
        if phylogeny_id not in self._phylogenies:
            raise ValueError(f"Phylogeny '{phylogeny_id}' not found in catalog")
        data = self._phylogenies[phylogeny_id]
        return {
            "directory": Path(data["directory"]),
            "pangenome_id": data["pangenome_id"],
        }
