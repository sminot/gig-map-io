"""
Command-line context for an analysis script.

An analysis script turns study definitions, and sometimes the output of an
earlier script, into figures and tables. Where those things live depends on how
the script is being run: straight from a checkout, or inside a workflow that
stages its inputs into a scratch directory and collects the outputs afterwards.

``AnalysisScript`` holds that resolution in one place. A script declares what it
needs, gets defaults that work from the root of a checkout, and gets a command
line that lets a workflow point it somewhere else without the script knowing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping

from .models.study import Study
from .models.study_set import StudySet

#: Where study definitions and analysis outputs live in a checkout.
DEFAULT_ANALYSIS_DIR = Path("analysis")
DEFAULT_DATASETS_DIR = Path("datasets")
DEFAULT_STUDIES_DIR = DEFAULT_ANALYSIS_DIR / "organize_studies"


class AnalysisScript:
    """
    Resolve an analysis script's inputs and output location.

    Parameters
    ----------
    script:
        The script's own ``__file__``. Its directory is where outputs go by
        default, so moving the folder moves the outputs with it.
    inputs:
        Files this script reads from earlier steps, as a map of argument name
        to the path it has within ``analysis/``. Each becomes a ``--<name>``
        option defaulting to that path.
    description:
        Shown in ``--help``; pass the script's ``__doc__``.

    Examples
    --------
    ::

        script = AnalysisScript(__file__, description=__doc__)
        study = script.study("gvhd_combined")
        study.volcano_plot(file_prefix=script.output("figure"))

    ::

        script = AnalysisScript(
            __file__,
            inputs={"bins": "associated_bins/candidate_bins/02_negative_in_both/bins.csv"},
            description=__doc__,
        )
        bins = pd.read_csv(script.input("bins"), index_col=[0, 1])
    """

    def __init__(
        self,
        script: str | Path,
        inputs: Mapping[str, str] | None = None,
        description: str | None = None,
        argv: list[str] | None = None,
    ) -> None:
        self.script = Path(script).resolve()
        self.declared_inputs = dict(inputs or {})

        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--datasets",
            type=Path,
            default=DEFAULT_DATASETS_DIR,
            help="Root of the gig-map dataset collection (default: %(default)s)",
        )
        parser.add_argument(
            "--studies",
            type=Path,
            default=DEFAULT_STUDIES_DIR,
            help="Directory holding the study definitions (default: %(default)s)",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=self.script.parent,
            help="Where to write this script's outputs (default: the script's own directory)",
        )
        for name, default in self.declared_inputs.items():
            parser.add_argument(
                f"--{name}",
                type=Path,
                default=DEFAULT_ANALYSIS_DIR / default,
                help=f"Input read from an earlier step (default: %(default)s)",
            )

        self.args = parser.parse_args(argv)
        self.datasets: Path = self.args.datasets
        self.studies_dir: Path = self.args.studies
        self.output_dir: Path = self.args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Inputs -----------------------------------------------------------

    def study(self, name: str) -> Study:
        """Load one study definition by name."""
        return Study.from_json(self._study_path(name), datasets=self.datasets)

    def study_set(self, *names: str) -> StudySet:
        """Load several study definitions to be analyzed together."""
        return StudySet.from_json(
            *(self._study_path(name) for name in names), datasets=self.datasets
        )

    def _study_path(self, name: str) -> Path:
        path = self.studies_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No study definition at {path}. "
                "Run analysis/organize_studies/run.py first, or pass --studies."
            )
        return path

    def input(self, name: str) -> Path:
        """The resolved path of one of the declared inputs."""
        if name not in self.declared_inputs:
            raise KeyError(f"{self.script.name} declares no input named {name!r}")
        path: Path = getattr(self.args, name.replace("-", "_"))
        if not path.exists():
            raise FileNotFoundError(f"Input {name!r} not found at {path}")
        return path

    # --- Outputs ----------------------------------------------------------

    def output(self, name: str) -> str:
        """
        A file prefix under the output directory, for the plotting methods to
        write their several formats to.
        """
        return str(self.output_dir / name)

    def output_path(self, name: str) -> Path:
        """A single output file under the output directory."""
        return self.output_dir / name

    def describe(self) -> Dict[str, object]:
        """What this script reads and where it writes, for introspection."""
        return {
            "script": str(self.script),
            "inputs": self.declared_inputs,
            "datasets": str(self.datasets),
            "studies": str(self.studies_dir),
            "output_dir": str(self.output_dir),
        }
