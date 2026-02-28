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
from ..helpers.save_image import save_image


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

    def __repr__(self) -> str:
        return f"ContrastMetagenomes(directory={self.directory}, parameter={self.parameter})"

    def __str__(self) -> str:
        return f"ContrastMetagenomes(directory={self.directory}, parameter={self.parameter})"

    def __format__(self, format_spec: str) -> str:
        return f"ContrastMetagenomes(directory={self.directory}, parameter={self.parameter})"

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
    def metadata(self) -> pd.DataFrame:
        """
        Metadata from metadata.csv.
        """
        path = self.directory / "association" / "metadata.csv"
        df = pd.read_csv(path, index_col=0)
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
        file_prefix: str | None = None,
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
            template="plotly_white",
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

        # If save_image was provided, use the string as the file
        # prefix to write out HTML, PDF, PNG, and JSON
        save_image(fig, file_prefix)

        return fig

    def plot_bin_abundance(
        self,
        bin: str,
        norm_bin: str | None = None,
        width: int = 500,
        height: int = 400,
        file_prefix: str | None = None,
        **kwargs
    ) -> go.Figure:
        """
        Plot the abundance of a bin.
        """
        assert bin in self.rpkm.columns, f"{bin} not found in rpkm.csv.gz"

        # The data used for plotting will be the metadata and the bin abundance
        df = self.metadata.assign(
            abundance=(
                self.rpkm.loc[:, bin]
                if norm_bin is None
                else self.rpkm.loc[:, bin] / self.rpkm.loc[:, norm_bin]
            )
        )

        fig = px.histogram(
            data_frame=df,
            y="abundance",
            template="plotly_white",
            width=width,
            height=height,
            **kwargs
        )
        fig.update_xaxes(title_text=f"{kwargs.get('histnorm', 'number').title()} of Samples")
        fig.update_yaxes(
            title_text=(
                f"Abundance of {bin} (RPKM)"
                if norm_bin is None
                else f"Abundance of {bin} / {norm_bin}"
            ),
            col=1
        )

        # If save_image was provided, use the string as the file
        # prefix to write out HTML, PDF, PNG, and JSON
        save_image(fig, file_prefix)

        return fig