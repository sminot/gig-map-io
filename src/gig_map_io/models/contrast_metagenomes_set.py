from functools import cached_property
from logging import getLogger
import logging
from scipy import stats
from pathlib import Path
import sys
from typing import Any, Dict, Iterator

from plotly import graph_objects as go
from plotly.subplots import make_subplots
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
    def metadata(self) -> pd.DataFrame:
        return (
            pd.concat(
                [cm.metadata for cm in self.contrast_metagenomes.values()],
                join="outer",
            )
            .pipe(lambda df: df[~df.index.duplicated(keep="first")])
        )

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
        xlabel: str = "Effect Size",
        transpose: bool = False,
        **kwargs
    ) -> go.Figure:
        """
        Volcano plot from the association results.
        """

        df = (
            self.association
            .assign(Estimate_clipped=self.association["Estimate"].clip(lower=-max_abs_estimate, upper=max_abs_estimate))
        )

        # Make a nice hover name
        df = df.assign(
            hover_name=df.apply(lambda r: f"{r['pangenome']}<br>{r['feature']}", axis=1).astype(str)
        )

        _coords = (
            dict(x="Estimate_clipped", y="neg_log10_qvalue")
            if not transpose
            else dict(x="neg_log10_qvalue", y="Estimate_clipped")
        )

        fig = px.scatter(
            data_frame=df,
            x=_coords["x"],
            y=_coords["y"],
            hover_name="hover_name",
            color="pangenome",
            template="plotly_white",
            labels=dict(
                Estimate_clipped="Effect Size (Clipped)",
                Estimate="Effect Size",
                neg_log10_qvalue="-log10(q-value)",
                neg_log10_pvalue="-log10(p-value)",
                signed_log10_qvalue="Signed -log10(q-value)",
                signed_log10_pvalue="Signed -log10(p-value)",
                feature="Pangenome Bin",
                mean_abund="Mean Abundance (RPKM)",
                pangenome="Pangenome",
            ),
            hover_data=["mean_abund", "Estimate", "signed_log10_qvalue", "signed_log10_pvalue", "pvalue"],
            width=width,
            height=height,
            **kwargs
        )
        make_lines(0, "black", fig)
        make_lines(estimate_thresh, "red", fig, hline=False)
        make_lines(-np.log10(fdr_thresh), "red", fig, vline=False, neg=False)

        # Specify the x-axis title
        fig.update_xaxes(title_text=xlabel)

        # Center the title
        fig.update_layout(title_x=0.5)

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
        estimate_thresh: float = 0.25,
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
            .pipe(lambda d: _add_sig_categories(d, fdr, sig_thresh, estimate_thresh))
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
            title_x=0.5,
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

        # Display the pangenome name and feature name in the hover name
        df = df.assign(
            hover_name=df.apply(lambda r: f"{r['pangenome']}<br>{r['feature']}", axis=1).astype(str)
        )

        fig = px.scatter(
            data_frame=df,
            x=f"{value_col}_self",
            y=f"{value_col}_comparitor",
            color="pangenome",
            hover_name="hover_name",
            template="plotly_white",
            labels={
                f"{value_col}_self": f"{value_label} ({self_label})",
                f"{value_col}_comparitor": f"{value_label} ({comparitor_label})",
                "pvalue_self": f"p-value ({self_label})",
                "pvalue_comparitor": f"p-value ({comparitor_label})",
                "feature": "Pangenome Bin",
                "pangenome": "Pangenome",
            },
            hover_data=[f"{value_col}_self", f"{value_col}_comparitor", "pvalue_self", "pvalue_comparitor"],
            width=width,
            height=height,
            **kwargs
        )

        make_lines(0, "black", fig)
        make_lines(-np.log10(sig_thresh), "red", fig)

        save_image(fig, file_prefix)

        return fig

    def compare_association_scatter(
        self,
        comparitor: 'ContrastMetagenomesSet',
        self_label: str = "self",
        comparitor_label: str = "comparitor",
        fdr: bool = True,
        sig_thresh: float = 0.2,
        estimate_thresh: float = 0.25,
        width: int = 500,
        height: int = 400,
        file_prefix: str | None = None,
        **kwargs
    ) -> go.Figure:
        """
        Scatter plot of association values for two contrast sets.
        NOTE: Only show bins that are significant in both contrast sets.
        """
        sig_col = "qvalue" if fdr else "pvalue"
        df = self.compare_association(comparitor)
        df = df.loc[
            (df[sig_col + "_self"] <= sig_thresh)
            & (df[sig_col + "_comparitor"] <= sig_thresh)
            & (df["Estimate_self"].abs() >= estimate_thresh)
            & (df["Estimate_comparitor"].abs() >= estimate_thresh)
        ]

        # Display the pangenome name and feature name in the hover name
        df = df.assign(
            hover_name=df.apply(lambda r: f"{r['pangenome']}<br>{r['feature']}", axis=1).astype(str)
        )

        fig = px.scatter(
            data_frame=df,
            x="Estimate_self",
            y="Estimate_comparitor",
            color="pangenome",
            hover_name="hover_name",
            template="plotly_white",
            labels={
                "Estimate_self": f"Estimate ({self_label})",
                "Estimate_comparitor": f"Estimate ({comparitor_label})",
                "pvalue_self": f"p-value ({self_label})",
                "pvalue_comparitor": f"p-value ({comparitor_label})",
                "qvalue_self": f"q-value ({self_label})",
                "qvalue_comparitor": f"q-value ({comparitor_label})",
                "pangenome": "Pangenome",
                "feature": "Pangenome Bin",
            },
            hover_data=["Estimate_self", "Estimate_comparitor", "pvalue_self", "pvalue_comparitor", "qvalue_self", "qvalue_comparitor"],
            width=width,
            height=height,
            **kwargs
        )

        make_lines(0, "black", fig)

        save_image(fig, file_prefix)

        return fig

    def compare_volcano_with_estimate(
        self,
        comparitor: 'ContrastMetagenomesSet',
        self_label: str = "self",
        comparitor_label: str = "comparitor",
        fdr: bool = True,
        sig_thresh: float = 0.2,
        estimate_thresh: float = 0.25,
        max_abs_estimate: float = 2.5,
        width: int = 600,
        height: int = 600,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Multi-panel figure combining the volcano plot for each contrast set with the association scatter plot.
        """

        # Make a multi-panel figure combining the volcano plot for each contrast set with the association scatter plot.
        fig = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.06,
            vertical_spacing=0.06
        )
        fig.add_traces(
            self.volcano_plot(
                estimate_thresh=estimate_thresh,
                fdr_thresh=sig_thresh,
                max_abs_estimate=max_abs_estimate,
            ).data,
            rows=2,
            cols=2
        )
        fig.add_traces(
            comparitor.volcano_plot(
                estimate_thresh=estimate_thresh,
                fdr_thresh=sig_thresh,
                max_abs_estimate=max_abs_estimate,
                transpose=True
            ).data,
            rows=1,
            cols=1
        )
        fig.add_traces(
            self.compare_association_scatter(
                comparitor=comparitor,
                fdr=fdr,
                sig_thresh=sig_thresh,
            ).data,
            rows=1,
            cols=2
        )
        fig.update_layout(
            width=width,
            height=height,
            template="plotly_white",
            showlegend=False,
        )

        make_lines(0, "black", fig)
        make_lines(estimate_thresh, "red", fig, hline=False, row=2, col=2)
        make_lines(estimate_thresh, "red", fig, vline=False, row=1, col=1)
        make_lines(-np.log10(sig_thresh), "red", fig, vline=False, neg=False, row=2, col=2)
        make_lines(-np.log10(sig_thresh), "red", fig, hline=False, neg=False, row=1, col=1)

        sig_label = "q-value" if fdr else "p-value"
        fig.update_xaxes(title_text=f"-log10({sig_label})", row=1, col=1)
        fig.update_yaxes(title_text=f"Estimate ({comparitor_label})", row=1, col=1)
        fig.update_xaxes(title_text=f"Estimate ({self_label})", row=2, col=2)
        fig.update_yaxes(title_text=f"-log10({sig_label})", row=2, col=2)

        save_image(fig, file_prefix)

        return fig


def _add_sig_categories(
    df: pd.DataFrame,
    fdr: bool = True,
    sig_thresh: float = 0.2,
    estimate_thresh: float = 0.25,
) -> pd.DataFrame:
    """
    Add the significance categories to the dataframe.
    """

    return df.assign(
        self_sig=df.apply(lambda row: _add_sig_category(row, "self", fdr, sig_thresh, estimate_thresh), axis=1),
        comparitor_sig=df.apply(lambda row: _add_sig_category(row, "comparitor", fdr, sig_thresh, estimate_thresh), axis=1),
    )

def _add_sig_category(
    row: pd.Series,
    label: str,
    fdr: bool = True,
    sig_thresh: float = 0.2,
    estimate_thresh: float = 0.25,
) -> str:
    sig_col = ("qvalue" if fdr else "pvalue") + "_" + label
    est_col = "Estimate_" + label
    if row[sig_col] >= sig_thresh:
        return "="
    elif np.abs(row[est_col]) < estimate_thresh:
        return "="
    elif row[est_col] > 0:
        return ">"
    else:
        return "<"
