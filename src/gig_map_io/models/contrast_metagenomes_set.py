from functools import cached_property
from logging import getLogger
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterator

from plotly import graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

from gig_map_io.helpers.make_lines import make_lines
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
    def __init__(self, directory_dict: Dict[str, str | Path], parameter: str) -> None:
        super().__init__(directory_dict)
        if not isinstance(parameter, str):
            raise ValueError("parameter must be a string")
        self.parameter = parameter

    @cached_property
    def contrast_metagenomes(self) -> Dict[str, ContrastMetagenomes]:
        return {key: ContrastMetagenomes(self.directory_dict[key], self.parameter) for key in self.directory_dict.keys()}

    def __repr__(self) -> str:
        return f"ContrastMetagenomesSet(directory_dict={self.directory_dict}, parameter={self.parameter})"

    def __str__(self) -> str:
        return f"ContrastMetagenomesSet(directory_dict={self.directory_dict}, parameter={self.parameter})"

    def __format__(self, format_spec: str) -> str:
        return f"ContrastMetagenomesSet(directory_dict={self.directory_dict}, parameter={self.parameter})"

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

    @cached_property
    def association(self) -> pd.DataFrame:
        # Combine the association results from all contrasts
        # and recompute the FDR-adjusted q-values
        df = pd.concat([
            contrast.association.assign(contrast=contrast_name)
            for contrast_name, contrast in self.contrast_metagenomes.items()
        ])
        qvalue = multipletests(df["pvalue"], method="fdr_bh")[1]
        df = df.assign(qvalue=qvalue, neg_log10_qvalue=-np.log10(qvalue))
        df = df.sort_values(by=["contrast", "pvalue"])
        return df

    def volcano_plot(
        self,
        estimate_thresh: float = 0.25,
        fdr_thresh: float = 0.2,
        max_abs_estimate: float = 5.0,
        width: int = 500,
        height: int = 400,
        file_prefix: str | None = None,
        **kwargs
    ) -> go.Figure:
        """
        Volcano plot from the association results.
        """

        df = (
            self.association
            .assign(Estimate_clipped=self.association["Estimate"].clip(lower=-max_abs_estimate, upper=max_abs_estimate))
        )

        fig = px.scatter(
            data_frame=df,
            x="Estimate_clipped",
            y="neg_log10_qvalue",
            hover_data=df.columns.values,
            hover_name="feature",
            color="contrast",
            template="plotly_white",
            labels=dict(
                Estimate_clipped="Effect Size",
                neg_log10_qvalue="-log10(q-value)",
                feature="Pangenome Bin",
                mean_abund="Mean Abundance (RPKM)",
                contrast="Contrast",
            ),
            size="mean_abund",
            width=width,
            height=height,
            **kwargs
        )
        make_lines(0, "black", fig)
        make_lines(estimate_thresh, "red", fig, hline=False)
        make_lines(-np.log10(fdr_thresh), "red", fig, vline=False, neg=False)

        # If save_image was provided, use the string as the file
        # prefix to write out HTML, PDF, PNG, and JSON
        save_image(fig, file_prefix)

        return fig