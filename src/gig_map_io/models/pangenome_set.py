from functools import cached_property
from logging import getLogger
import logging
from pathlib import Path
import sys
from typing import Dict

from plotly.subplots import make_subplots
import plotly.express as px
from plotly import graph_objects as go
import pandas as pd
import numpy as np
from gig_map_io.helpers.save_image import save_image

from .pangenome import Pangenome
from .dataset_dict import DatasetDict

logger = getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

class PangenomeSet(DatasetDict):
    """
    Representation of a set of pangenomes.
    """
    def __init__(self, directory_dict: Dict[str, str | Path]) -> None:
        super().__init__(directory_dict)

    @cached_property
    def pangenomes(self) -> Dict[str, Pangenome]:
        return {key: Pangenome(self.directory_dict[key]) for key in self.directory_dict.keys()}

    def __repr__(self) -> str:
        return f"PangenomeSet(directory_dict={self.directory_dict})"

    def __str__(self) -> str:
        return f"PangenomeSet(directory_dict={self.directory_dict})"

    def __format__(self, format_spec: str) -> str:
        return f"PangenomeSet(directory_dict={self.directory_dict})"

    def bin_genome_heatmap(self,
        col_wrap: int = 3,
        width: int = 500,
        height: int = 400,
        horizontal_spacing: float = 0.05,
        vertical_spacing: float = 0.05,
        file_prefix: str | None = None
    ) -> go.Figure:
        """
        Heatmap of bin presence/absence for each genome, faceted by pangenome.
        """
        fig = make_subplots(
            rows=len(self.pangenomes) // col_wrap + 1,
            cols=col_wrap,
            shared_yaxes=False,
            shared_xaxes=False,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
            subplot_titles=[pangenome for pangenome in self.pangenomes.keys()]
        )
        for i, pangenome in enumerate(self.pangenomes.keys()):
            for trace in self.pangenomes[pangenome].bin_genome_heatmap().data:
                fig.add_trace(
                    trace,
                    row=i // col_wrap + 1,
                    col=i % col_wrap + 1
                )

        fig.update_layout(height=height, width=width)

        # Left-align the subplot titles
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].update(x=0.02, xanchor='left', xref=f'x{i+1}')

        save_image(fig, file_prefix)
        return fig

    def bin_size_histogram(self,
        bins: int = 30,
        width: int = 500,
        height: int = 400,
        file_prefix: str | None = None
    ) -> go.Figure:
        """
        Histogram of bin sizes, faceted by pangenome.
        """
        # Set the boundaries of the bins to be the same for all pangenomes
        max_bin_size = max([pg.bin_size.max() for pg in self.pangenomes.values()])
        min_bin_size = min([pg.bin_size.min() for pg in self.pangenomes.values()])
        bins = np.linspace(np.log10(min_bin_size), np.log10(max_bin_size), bins + 1)

        df = pd.concat([
            pangenome.bin_size_df(bins).assign(pangenome=pangenome_name)
            for pangenome_name, pangenome in self.pangenomes.items()
        ])

        fig = px.bar(
            data_frame=df,
            x="bin_size",
            y="count",
            color="pangenome",
            labels=dict(
                bin_size="Pangenome Bin Size (# of Genes)",
                count="Total Gene Content",
                pangenome="Pangenome"
            ),
            template="plotly_white",
            hover_name="bin_names",
            width=width,
            height=height
        )
        fig.update_xaxes(
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=["1", "10", "100", "1k", "10k", "100k"]
        )
        # If save_image was provided, use the string as the file
        # prefix to write out HTML, PDF, PNG, and JSON
        save_image(fig, file_prefix)

        return fig

    def rarefaction_curve(
        self,
        col_wrap: int = 3,
        width: int = 800,
        height: int = 800,
        horizontal_spacing: float = 0.05,
        file_prefix: str | None = None
    ) -> go.Figure:
        """
        Rarefaction curve of the pangenomes, faceted by pangenome.
        """
        fig = make_subplots(
            rows=len(self.pangenomes) // col_wrap + 1,
            cols=col_wrap,
            shared_yaxes=False,
            shared_xaxes=False,
            horizontal_spacing=horizontal_spacing,
            subplot_titles=[pangenome for pangenome in self.pangenomes.keys()]
        )
        for i, pangenome in enumerate(self.pangenomes.keys()):
            for trace in self.pangenomes[pangenome].rarefaction_curve().data:
                fig.add_trace(
                    trace,
                    row=i // col_wrap + 1,
                    col=i % col_wrap + 1
                )

        fig.update_layout(
            height=height,
            width=width,
            template="plotly_white"
        )
        fig.update_yaxes(
            range=[0, None]
        )
        save_image(fig, file_prefix)
        return fig