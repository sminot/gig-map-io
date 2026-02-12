"""
Command-line interface for gig-map-io.

This module exposes a small CLI hook that can be used to launch marimo
notebooks and, in the future, additional utilities for working with gig-map
outputs.
"""

from __future__ import annotations

from importlib import resources
import os
import subprocess
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
@click.option(
    "--catalog",
    "-c",
    "catalog_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(os.environ.get("HOME", str(Path.home()))) / ".gig-map" / "catalog.json",
    show_default=True,
    help="Path to gig-map catalog JSON file.",
)
@click.pass_context
def main(ctx: click.Context, catalog_path: Path) -> None:
    """gig-map-io command-line interface."""
    ctx.ensure_object(dict)
    ctx.obj["catalog"] = str(catalog_path)


@main.command("run")
@click.option(
    "--notebook",
    "-n",
    "notebook_name",
    type=str,
    default=None,
    help="Specific marimo notebook file to run (defaults to index.py).",
)
@click.pass_context
def run(ctx: click.Context, notebook_name: Optional[str]) -> None:
    """
    Run a bundled marimo notebook.

    This is a thin wrapper around `python -m marimo run <notebook>`.
    """
    nb_dir = _notebooks_dir()
    if notebook_name is None:
        notebook_name = "index"

    # Always treat the notebook name as a Python module file.
    if not notebook_name.endswith(".py"):
        notebook_name = f"{notebook_name}.py"

    notebook_path = nb_dir / notebook_name
    if not notebook_path.exists():
        raise SystemExit(f"Notebook not found: {notebook_path}")

    # Use marimo's CLI to run the notebook. This avoids depending on internal
    # Python APIs that may change between marimo versions.
    #
    # Equivalent to: python -m marimo run <notebook_path>
    env = os.environ.copy()
    env["GIG_MAP_CATALOG"] = ctx.obj.get("catalog", "")

    try:
        subprocess.run(
            ["python", "-m", "marimo", "run", str(notebook_path)],
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


@main.command("edit")
@click.option(
    "--notebook",
    "-n",
    "notebook_name",
    type=str,
    default=None,
    help="Specific marimo notebook file to edit (defaults to index.py).",
)
@click.pass_context
def edit(ctx: click.Context, notebook_name: Optional[str]) -> None:
    """
    Edit a bundled marimo notebook.

    This is a thin wrapper around `python -m marimo edit <notebook>`.
    """
    nb_dir = _notebooks_dir()
    if notebook_name is None:
        notebook_name = "index"

    # Always treat the notebook name as a Python module file.
    if not notebook_name.endswith(".py"):
        notebook_name = f"{notebook_name}.py"

    notebook_path = nb_dir / notebook_name
    if not notebook_path.exists():
        raise SystemExit(f"Notebook not found: {notebook_path}")

    # Use marimo's CLI to edit the notebook. This avoids depending on internal
    # Python APIs that may change between marimo versions.
    #
    # Equivalent to: python -m marimo edit <notebook_path>
    env = os.environ.copy()
    env["GIG_MAP_CATALOG"] = ctx.obj.get("catalog", "")

    try:
        subprocess.run(
            ["python", "-m", "marimo", "edit", str(notebook_path)],
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
