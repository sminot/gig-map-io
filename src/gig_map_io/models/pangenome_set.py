from functools import cached_property
from logging import getLogger
import logging
from pathlib import Path
import sys
from typing import Dict

from plotly.subplots import make_subplots
from plotly import graph_objects as go

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

        save_image(fig, file_prefix)
        return fig
