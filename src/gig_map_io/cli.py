[@@]
"""
Command-line interface for gig-map-io.

This module exposes a small CLI hook that can be used to launch marimo
notebooks and, in the future, additional utilities for working with gig-map
outputs.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Optional

import click


def _notebooks_dir() -> Path:
    """
    Return the path to the bundled marimo notebooks.

    The notebooks live under the `notebooks` package in `gig_map_io`.
    """
    # `files` returns an importlib.abc.Traversable; convert to Path when possible.
    try:
        from importlib.resources import files  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover - Python <3.9 fallback
        return Path(resources.path("gig_map_io", "notebooks").__enter__())

    return Path(files("gig_map_io") / "notebooks")


@click.group()
def main() -> None:
    """gig-map-io command-line interface."""


@main.command("launch-notebooks")
@click.option(
    "--notebook",
    "-n",
    "notebook_name",
    type=str,
    default=None,
    help="Specific marimo notebook file to run (defaults to index.py).",
)
def launch_notebooks(notebook_name: Optional[str]) -> None:
    """
    Launch bundled marimo notebooks.

    This is a thin wrapper around `python -m marimo run <notebook>`.
    """
    nb_dir = _notebooks_dir()
    if notebook_name is None:
        notebook_name = "index.py"

    notebook_path = nb_dir / notebook_name
    if not notebook_path.exists():
        raise SystemExit(f"Notebook not found: {notebook_path}")

    # Defer import so marimo is only required when this command is used.
    import marimo

    marimo.run(str(notebook_path))

