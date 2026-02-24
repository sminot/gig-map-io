"""
ContrastMetagenomes base class for gig-map-io.
"""

from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from .dataset import Dataset
from ..helpers.make_lines import make_lines


class ContrastMetagenomes(Dataset):
    """
    Representation of a contrast between metagenomes (e.g., case vs control).

    Reads key outputs from the contrast_metagenomes workflow: summary,
    association results, and optionally bin abundance (RPKM). DataFrame
    attributes are cached after first access.
    """

    parameter: str

    def __init__(self, directory: str | Path, parameter: str) -> None:
        Dataset.__init__(self, directory)
        if not isinstance(parameter, str):
            raise ValueError("parameter must be a string")
        self.parameter = parameter

    @cached_property
    def association(self) -> pd.DataFrame:
        """
        Association results from association/association.csv,
        filtered to the specified parameter.

        Returns
        -------
        DataFrame with columns such as feature, Estimate, SE, pvalue, etc.
        """
        path = self.directory / "association" / "association.csv"
        df = pd.read_csv(path)

        # Make sure that the parameter value is present in the parameter column
        if self.parameter not in df["parameter"].values:
            raise ValueError(f"parameter {self.parameter} not found in association.csv")

        # Filter to the parameter value
        df = df.loc[df["parameter"] == self.parameter].drop(columns=["parameter"])

        # Add the mean abundance to the dataframe
        df = df.assign(mean_abund=df["feature"].map(self.mean_abund).fillna(0))

        return df

    @cached_property
    def rpkm(self) -> pd.DataFrame:
        """
        Bin abundance (RPKM) from bin_abundance/rpkm.csv.gz if present.

        Returns
        -------
        DataFrame with specimens as index and bins as columns.
        """
        path = self.directory / "bin_abundance" / "rpkm.csv.gz"
        df = pd.read_csv(path)
        df.set_index("specimen", inplace=True)
        return df

    @cached_property
    def mean_abund(self) -> pd.Series:
        """
        Mean bin abundance (RPKM) for each bin.

        Returns
        -------
        Series with bins as index and mean abundance as values.
        """
        return self.rpkm.mean()

    def volcano_plot(
        self,
        estimate_thresh: float = 0.25,
        fdr_thresh: float = 0.2,
        max_abs_estimate: float = 5.0,
        width: int = 500,
        height: int = 400,
        **kwargs
    ) -> go.Figure:
        """
        Volcano plot from the association results.
        """

        df = self.association.assign(
            Estimate_clipped=self.association["Estimate"].clip(lower=-max_abs_estimate, upper=max_abs_estimate)
        )

        fig = px.scatter(
            data_frame=df,
            x="Estimate_clipped",
            y="neg_log10_qvalue",
            hover_data=df.columns.values,
            hover_name="feature",
            template="simple_white",
            labels=dict(
                Estimate_clipped="Effect Size",
                neg_log10_qvalue="-log10(q-value)",
                feature="Pangenome Bin",
                mean_abund="Mean Abundance (RPKM)",
            ),
            size="mean_abund",
            width=width,
            height=height,
            **kwargs
        )
        make_lines(0, "black", fig)
        make_lines(estimate_thresh, "red", fig, hline=False)
        make_lines(-np.log10(fdr_thresh), "red", fig, vline=False, neg=False)
        return fig