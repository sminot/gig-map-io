from functools import cached_property
from logging import getLogger
import logging
from scipy import stats
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
from gig_map_io.helpers.format_pvalue import format_pvalue
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

    @cached_property
    def pangenome_names(self) -> list[str]:
        return list(self.contrast_metagenomes.keys())

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
            contrast.association.assign(pangenome=pangenome_name)
            for pangenome_name, contrast in self.contrast_metagenomes.items()
        ])
        df = df.dropna(subset=["pvalue"])
        qvalue = multipletests(df["pvalue"], method="fdr_bh")[1]
        df = df.assign(
            qvalue=qvalue,
            pvalue=df["pvalue"].clip(lower=df.loc[df["pvalue"] > 0, "pvalue"].min()),
            neg_log10_qvalue=-np.log10(qvalue),
            signed_log10_qvalue=lambda d: np.sign(d["Estimate"]) * -np.log10(qvalue),
            signed_log10_pvalue=lambda d: np.sign(d["Estimate"]) * -np.log10(d["pvalue"]),
        )
        df = df.sort_values(by=["pangenome", "pvalue"])
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
            color="pangenome",
            template="plotly_white",
            labels=dict(
                Estimate_clipped="Effect Size",
                neg_log10_qvalue="-log10(q-value)",
                feature="Pangenome Bin",
                mean_abund="Mean Abundance (RPKM)",
                pangenome="Pangenome",
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

    def compare_association(self, comparitor: 'ContrastMetagenomesSet') -> pd.DataFrame:
        """
        Compare the association results of two contrast sets.
        """
        return (
            self.association
            .merge(
                comparitor.association,
                on=["pangenome", "feature"],
                suffixes=("_self", "_comparitor")
            )
            .assign(
                mean_abund=lambda x: x[["mean_abund_self", "mean_abund_comparitor"]].mean(axis=1),
            )
            .dropna(subset=["pvalue_self", "pvalue_comparitor"])
        )

    def compare_sig_categories(
        self,
        comparitor: 'ContrastMetagenomesSet',
        fdr: bool = True,
        sig_thresh: float = 0.2,
        self_label: str = "self",
        comparitor_label: str = "comparitor",
        width: int = 400,
        height: int = 400,
        file_prefix: str | None = None,
        **kwargs
    ) -> go.Figure:
        """
        Compare the significance categories of two contrast sets.
        """
        df = (
            self.compare_association(comparitor)
            .pipe(lambda d: _add_sig_categories(d, fdr, sig_thresh))
        )

        # Make a table comparing the significance categories
        sig_table = df.pivot_table(
            columns="self_sig",
            index="comparitor_sig",
            values="feature",
            aggfunc="count",
            fill_value=0,
        ).reindex(
            index=["<", "=", ">"],
            columns=["<", "=", ">"],
        )

        # Run a chi-squared test to compare the significance categories
        chi2, p, dof, expected = stats.chi2_contingency(sig_table)

        # Make a table showing the percentage difference between the significance categories
        # compared to the expected values
        expected_table = pd.DataFrame(expected, index=sig_table.index, columns=sig_table.columns)
        percent_diff_table = (sig_table - expected_table) / expected_table * 100

        # Make a heatmap showing the percentage difference
        # Include text in the cells showing the percentage difference
        # with the +/-, %, and number of features
        text = pd.DataFrame({
            cname: {
                iname: (
                    f"{v:.1f}%<br>n={sig_table.loc[iname, cname]:,}"
                    if v < 0
                    else f"+{v:.1f}%<br>n={sig_table.loc[iname, cname]:,}")
                    for iname, v in row.items()
            }
            for cname, row in percent_diff_table.iterrows()
        })
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=percent_diff_table.values,
                    x=percent_diff_table.columns.values,
                    y=percent_diff_table.index.values,
                    text=text.values,
                    colorscale="RdBu",
                    texttemplate="%{text}",
                    zmid=0,
                    showscale=False,
                )
            ]
        )
        fig.update_layout(
            title=f"Chi-squared test (p={format_pvalue(p)})",
            xaxis_title=self_label,
            yaxis_title=comparitor_label,
            width=width,
            height=height,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            plot_bgcolor="white",
            coloraxis_showscale=False
        )
        save_image(fig, file_prefix)

        return fig

    def compare_sig_scatter(
        self,
        comparitor: 'ContrastMetagenomesSet',
        self_label: str = "self",
        comparitor_label: str = "comparitor",
        fdr: bool = True,
        sig_thresh: float = 0.2,
        width: int = 500,
        height: int = 400,
        file_prefix: str | None = None,
        **kwargs
    ) -> go.Figure:
        """
        Scatter plot of q-values for two contrast sets.
        """
        df = self.compare_association(comparitor)
        value_col = "signed_log10_qvalue" if fdr else "signed_log10_pvalue"
        value_label = "signed -log10(q-value)" if fdr else "signed -log10(p-value)"

        fig = px.scatter(
            data_frame=df,
            x=f"{value_col}_self",
            y=f"{value_col}_comparitor",
            color="pangenome",
            template="plotly_white",
            labels={
                f"{value_col}_self": f"{value_label} ({self_label})",
                f"{value_col}_comparitor": f"{value_label} ({comparitor_label})",
                "pangenome": "Pangenome",
            },
            # size="mean_abund",
            width=width,
            height=height,
            **kwargs
        )

        make_lines(0, "black", fig)
        make_lines(-np.log10(sig_thresh), "red", fig)

        save_image(fig, file_prefix)

        return fig


def _add_sig_categories(df: pd.DataFrame, fdr: bool = True, sig_thresh: float = 0.2) -> pd.DataFrame:
    """
    Add the significance categories to the dataframe.
    """
    sig_col = "qvalue" if fdr else "pvalue"
    return df.assign(
        self_sig=df.apply(lambda row: (
            "=" if row[sig_col + "_self"] >= sig_thresh else (
                ">" if row["Estimate_self"] > 0 else "<"
            )
        ), axis=1),
        comparitor_sig=df.apply(lambda row: (
            "=" if row[sig_col + "_comparitor"] >= sig_thresh else (
                ">" if row["Estimate_comparitor"] > 0 else "<"
            )
        ), axis=1)
    )
