"""
ContrastMetagenomesSet: one case/control comparison across every organism.
"""

from functools import cached_property
import logging
from pathlib import Path
from typing import Dict

from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from gig_map_io.helpers.clustering import linkage_order
from gig_map_io.helpers.make_lines import make_lines
from gig_map_io.helpers.save_image import save_image
from gig_map_io.helpers.format_pvalue import format_pvalue
from gig_map_io.helpers.observed_expected import expected_counts, plot_observed_expected
from gig_map_io.helpers.sidebar import add_category_sidebar
from gig_map_io.helpers.style import (
    DENSE_MARKER_OPACITY, ESTIMATE_THRESH, FDR_THRESH, QUALITATIVE, TEMPLATE, THRESHOLD_LINE,
    ZERO_LINE, group_colors, organism_colors, organism_order,
)
from .contrast_metagenomes import ASSOCIATION_LABELS, ContrastMetagenomes, volcano_figure
from .dataset_dict import DatasetDict
from .sample_group import _key as _label_key

logger = logging.getLogger(__name__)


class ContrastMetagenomesSet(DatasetDict):
    """
    Representation of a set of contrast-metagenomes.
    """
    def __init__(self, directory_dict: Dict[str, str | Path], parameter: str) -> None:
        super().__init__(directory_dict)
        self.parameter = parameter

    @cached_property
    def contrast_metagenomes(self) -> Dict[str, ContrastMetagenomes]:
        return {key: ContrastMetagenomes(self.directory_dict[key], self.parameter) for key in self.directory_dict.keys()}

    def __repr__(self) -> str:
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

    @cached_property
    def metadata(self) -> pd.DataFrame:
        """
        The sample metadata of every contrast merged into one table. Where
        two contrasts disagree about a value, the first wins and a warning
        is logged.
        """
        long = (
            pd.concat([
                contrast.metadata.reset_index().melt(id_vars=["index"], var_name="variable", value_name="value")
                for contrast in self.contrast_metagenomes.values()
            ])
            .dropna(subset=["value"])
            .drop_duplicates()
            .sort_values(by=["index", "variable"])
        )
        for ix, val in long.groupby(["index", "variable"])["value"]:
            if len(val) > 1:
                logger.warning(f"Value {val} differs between contrasts for {ix}")
        return (
            long
            .groupby(["index", "variable"])
            .head(1)
            .pivot(index="index", columns="variable", values="value")
        )

    @cached_property
    def rpkm(self) -> pd.DataFrame:
        """
        RPKM from all contrasts.
        The pangenome name is added as the first level of the column index.
        """
        return (
            pd.concat([
                contrast.rpkm.T.assign(pangenome=pangenome_name).reset_index().set_index(['pangenome', 'index']).T
                for pangenome_name, contrast in self.contrast_metagenomes.items()
            ], axis=1).fillna(0)
        )

    @cached_property
    def association(self) -> pd.DataFrame:
        """The association results of every contrast, with the q-values recomputed across all of them."""
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
        return df.sort_values(by=["pangenome", "pvalue"])

    def _organism_scatter(self, df: pd.DataFrame, x: str, y: str, marker: dict, **kwargs) -> go.Figure:
        """
        A scatter of bins colored by organism, named in the hover by
        organism and bin, with the fixed organism colors and order.
        """
        df = df.assign(hover_name=df["pangenome"] + "<br>" + df["feature"])
        fig = px.scatter(
            data_frame=df,
            x=x,
            y=y,
            color="pangenome",
            color_discrete_map=organism_colors(self.pangenome_names),
            category_orders={"pangenome": organism_order(self.pangenome_names)},
            hover_name="hover_name",
            template=TEMPLATE,
            **kwargs,
        )
        fig.update_traces(marker=dict(marker, line_width=0))
        return fig

    def volcano_plot(
        self,
        estimate_thresh: float = ESTIMATE_THRESH,
        fdr_thresh: float = FDR_THRESH,
        max_abs_estimate: float = 5.0,
        width: int = 650,
        height: int = 450,
        file_prefix: str | None = None,
        **kwargs
    ) -> go.Figure:
        """Volcano plot from the association results, colored by organism."""
        df = self.association.assign(hover_name=lambda d: d["pangenome"] + "<br>" + d["feature"])
        fig = volcano_figure(
            df, estimate_thresh, fdr_thresh, max_abs_estimate,
            hover_name="hover_name",
            color="pangenome",
            color_discrete_map=organism_colors(self.pangenome_names),
            category_orders={"pangenome": organism_order(self.pangenome_names)},
            hover_data=["mean_abund", "Estimate", "signed_log10_qvalue", "signed_log10_pvalue", "pvalue", "qvalue"],
            width=width,
            height=height,
            **kwargs
        )
        fig.update_traces(marker=dict(size=6, opacity=DENSE_MARKER_OPACITY, line_width=0))
        save_image(fig, file_prefix)
        return fig

    def bin_abundance_heatmap(
        self,
        features: pd.MultiIndex,
        annotation_cols: dict[str, str] | None = None,
        annotation_labels: dict[str, dict] | None = None,
        annotation_orders: dict[str, list] | None = None,
        width: int = 1000,
        height: int = 800,
        rpkm_height_fraction: float = 0.65,
        annotation_width_fraction: float = 0.15,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        The abundance of a set of bins across samples, log10(RPKM + 1), with
        the Spearman correlation between every pair of bins beneath it and,
        optionally, a column of sample annotations beside it. Samples and
        bins are each ordered by hierarchical clustering.

        Parameters
        ----------
        features : pd.MultiIndex
            (pangenome, bin) pairs, each a bin of the corresponding contrast.
        annotation_cols : dict, optional
            Metadata columns to show beside the samples, mapped to the label
            each is displayed under.
        annotation_labels : dict, optional
            Display names for the values of an annotation column, keyed by
            the column's display label: ``{"GvHD": {"1": "Case", "0": "Control"}}``.
        annotation_orders : dict, optional
            Order of the (display) values of an annotation column, keyed the
            same way. A two-level column is colored as a contrast, with the
            first level the case-like one.
        rpkm_height_fraction : float
            Fraction of the figure height given to the abundance heatmap; the
            correlation heatmap takes the rest.
        annotation_width_fraction : float
            Fraction of the figure width given to the annotation column.
        """
        rpkm = pd.DataFrame({
            f"{pangenome} / {feature}": self.contrast_metagenomes[pangenome].rpkm[feature]
            for pangenome, feature in features
        })
        rpkm = rpkm.iloc[linkage_order(rpkm.fillna(0).values), linkage_order(rpkm.fillna(0).values.T)]
        sorted_samples = rpkm.index.tolist()
        sorted_features = rpkm.columns.tolist()
        corr_sorted = rpkm.corr(method="spearman").loc[sorted_features, sorted_features]

        annotations = None
        if annotation_cols:
            annotations = self.metadata.reindex(index=sorted_samples, columns=list(annotation_cols)).rename(columns=annotation_cols)
            for col, labels in (annotation_labels or {}).items():
                labels = {_label_key(k): v for k, v in labels.items()}
                annotations[col] = annotations[col].map(
                    lambda v: labels.get(_label_key(v), v) if pd.notnull(v) else v
                )
        annotation_orders = annotation_orders or {}

        # Row 1: [annotations | abundance], sharing the sample axis;
        # row 2: [            | correlation], sharing the bin axis with the abundance
        rpkm_col = 2 if annotations is not None else 1
        row_heights = [rpkm_height_fraction, 1.0 - rpkm_height_fraction]
        fig = make_subplots(
            rows=2,
            cols=rpkm_col,
            specs=[[{}, {}], [None, {}]] if annotations is not None else [[{}], [{}]],
            shared_xaxes=True,
            shared_yaxes=annotations is not None,
            column_widths=[annotation_width_fraction, 1.0 - annotation_width_fraction] if annotations is not None else None,
            row_heights=row_heights,
            horizontal_spacing=0.02,
            vertical_spacing=0.04,
        )
        fig.add_trace(
            go.Heatmap(
                z=np.log10(rpkm + 1).values,
                x=sorted_features,
                y=sorted_samples,
                colorscale="Blues",
                name="RPKM",
                colorbar=dict(title="log<sub>10</sub>(RPKM+1)", len=row_heights[0], yanchor="top", y=1.0, x=1.02, thickness=14),
            ),
            row=1, col=rpkm_col,
        )
        fig.add_trace(
            go.Heatmap(
                z=corr_sorted.values,
                x=sorted_features,
                y=sorted_features,
                colorscale="RdBu_r",
                zmid=0,
                name="Spearman r",
                colorbar=dict(title="Spearman r", len=row_heights[1], yanchor="bottom", y=0.0, x=1.02, thickness=14),
            ),
            row=2, col=rpkm_col,
        )

        if annotations is not None:
            for col_idx, col in enumerate(annotations.columns):
                present = annotations[col].dropna().unique().tolist()
                if col in annotation_orders:
                    categories = [v for v in annotation_orders[col] if v in present]
                    colors = group_colors(annotation_orders[col])
                else:
                    categories = sorted(present, key=str)
                    colors = dict(zip(categories, QUALITATIVE))
                add_category_sidebar(fig, annotations[col], categories, colors, row=1, col=1, x=col_idx, legend_group=col)
            # Integer positions with named ticks, so that the columns sit
            # flush with no categorical padding
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(len(annotations.columns))),
                ticktext=annotations.columns.tolist(),
                tickangle=90,
                range=[-0.5, len(annotations.columns) - 0.5],
                showticklabels=True,
                row=1, col=1,
            )
            fig.update_yaxes(showticklabels=False, row=1, col=1)

        # The categorical legend runs along the top, leaving the right margin
        # to the two color bars
        fig.update_layout(
            width=width,
            height=height,
            template=TEMPLATE,
            showlegend=annotations is not None,
            legend=dict(orientation="h", x=0.0, xanchor="left", y=1.0, yanchor="bottom"),
            margin=dict(t=50),
        )
        fig.update_yaxes(title_text="Samples", row=1, col=1)
        fig.update_yaxes(title_text="Pangenome bin", showticklabels=False, row=2, col=rpkm_col)
        fig.update_xaxes(title_text="Pangenome bin", row=2, col=rpkm_col)
        fig.update_xaxes(showticklabels=False, row=1, col=rpkm_col)
        fig.update_yaxes(showticklabels=False, row=1, col=rpkm_col)
        save_image(fig, file_prefix)
        return fig

    def compare_association(self, comparator: 'ContrastMetagenomesSet') -> pd.DataFrame:
        """The association results of the two sets side by side, one row per bin tested in both."""
        return (
            self.association
            .merge(
                comparator.association,
                on=["pangenome", "feature"],
                suffixes=("_self", "_comparator")
            )
            .assign(
                mean_abund=lambda x: x[["mean_abund_self", "mean_abund_comparator"]].mean(axis=1),
            )
            .dropna(subset=["pvalue_self", "pvalue_comparator"])
        )

    def compare_sig_categories(
        self,
        comparator: 'ContrastMetagenomesSet',
        fdr: bool = True,
        sig_thresh: float = FDR_THRESH,
        estimate_thresh: float = ESTIMATE_THRESH,
        self_label: str = "self",
        comparator_label: str = "comparator",
        width: int = 800,
        height: int = 500,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        How often two analyses call the same bin, against how often they would
        by chance.

        Every bin falls into one of nine categories: each analysis called it
        lower in cases, higher in cases, or not significant. The bars give the
        observed count of each category beside the count expected if the two
        analyses were independent, which is what the chi-squared test in the
        title compares.

        The categories are ordered by what they mean rather than by their
        position in the contingency table -- agreement first, then
        disagreement, then bins only one analysis called -- so that the
        question the figure answers can be read left to right.
        """
        df = self.compare_association(comparator)
        sig_col = "qvalue" if fdr else "pvalue"
        for side in ("self", "comparator"):
            df[f"{side}_sig"] = np.select(
                [
                    (df[f"{sig_col}_{side}"] >= sig_thresh) | (df[f"Estimate_{side}"].abs() < estimate_thresh),
                    df[f"Estimate_{side}"] > 0,
                ],
                ["=", ">"],
                default="<",
            )

        observed = df.pivot_table(
            columns="self_sig",
            index="comparator_sig",
            values="feature",
            aggfunc="count",
            fill_value=0,
        ).reindex(index=SIG_LEVELS, columns=SIG_LEVELS, fill_value=0)
        _, pvalue, _, _ = stats.chi2_contingency(observed)
        # Computed from the margins rather than taken from chi2_contingency,
        # which drops any row or column that is entirely empty and would then
        # not line up with the nine bars
        expected = expected_counts(observed)

        ticks, groups, obs, exp = [], [], [], []
        for self_sig, comparator_sig, group in _SIG_CATEGORY_ORDER:
            ticks.append(f"{_SIG_ARROW[self_sig]}<br>{_SIG_ARROW[comparator_sig]}")
            groups.append(group)
            obs.append(observed.loc[comparator_sig, self_sig])
            exp.append(expected.loc[comparator_sig, self_sig])

        return plot_observed_expected(
            ticks=ticks,
            groups=groups,
            observed=obs,
            expected=exp,
            title=f"Do {self_label} and {comparator_label} flag the same bins?",
            subtitle=f"chi-squared p = {format_pvalue(pvalue)}",
            y_title="Pangenome bins",
            caption=f"Each tick: direction in <b>{self_label}</b> (upper) "
                    f"and <b>{comparator_label}</b> (lower)",
            width=width,
            height=height,
            file_prefix=file_prefix,
        )

    def compare_sig_scatter(
        self,
        comparator: 'ContrastMetagenomesSet',
        self_label: str = "self",
        comparator_label: str = "comparator",
        fdr: bool = True,
        sig_thresh: float = FDR_THRESH,
        width: int = 650,
        height: int = 450,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """Signed significance of every bin in one set against the other."""
        value_col = "signed_log10_qvalue" if fdr else "signed_log10_pvalue"
        value_label = "signed -log10(q-value)" if fdr else "signed -log10(p-value)"
        fig = self._organism_scatter(
            self.compare_association(comparator),
            x=f"{value_col}_self",
            y=f"{value_col}_comparator",
            marker=dict(size=6, opacity=DENSE_MARKER_OPACITY),
            labels={
                **ASSOCIATION_LABELS,
                f"{value_col}_self": f"{value_label}<br>{self_label}",
                f"{value_col}_comparator": f"{value_label}<br>{comparator_label}",
                "pvalue_self": f"p-value ({self_label})",
                "pvalue_comparator": f"p-value ({comparator_label})",
            },
            hover_data=[f"{value_col}_self", f"{value_col}_comparator", "pvalue_self", "pvalue_comparator"],
            width=width,
            height=height,
        )
        make_lines(0, fig, **ZERO_LINE)
        make_lines(-np.log10(sig_thresh), fig, **THRESHOLD_LINE)
        save_image(fig, file_prefix)
        return fig

    def compare_association_scatter(
        self,
        comparator: 'ContrastMetagenomesSet',
        self_label: str = "self",
        comparator_label: str = "comparator",
        fdr: bool = True,
        sig_thresh: float = FDR_THRESH,
        estimate_thresh: float = ESTIMATE_THRESH,
        width: int = 650,
        height: int = 450,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """Effect size in one set against the other, for the bins significant in both."""
        sig_col = "qvalue" if fdr else "pvalue"
        df = self.compare_association(comparator)
        df = df.loc[
            (df[sig_col + "_self"] <= sig_thresh)
            & (df[sig_col + "_comparator"] <= sig_thresh)
            & (df["Estimate_self"].abs() >= estimate_thresh)
            & (df["Estimate_comparator"].abs() >= estimate_thresh)
        ]
        fig = self._organism_scatter(
            df,
            x="Estimate_self",
            y="Estimate_comparator",
            marker=dict(size=8, opacity=0.85),
            labels={
                **ASSOCIATION_LABELS,
                "Estimate_self": f"Effect size<br>{self_label}",
                "Estimate_comparator": f"Effect size<br>{comparator_label}",
                "pvalue_self": f"p-value ({self_label})",
                "pvalue_comparator": f"p-value ({comparator_label})",
                "qvalue_self": f"q-value ({self_label})",
                "qvalue_comparator": f"q-value ({comparator_label})",
            },
            hover_data=["Estimate_self", "Estimate_comparator", "pvalue_self", "pvalue_comparator", "qvalue_self", "qvalue_comparator"],
            width=width,
            height=height,
        )
        make_lines(0, fig, **ZERO_LINE)
        save_image(fig, file_prefix)
        return fig

    def compare_volcano_with_estimate(
        self,
        comparator: 'ContrastMetagenomesSet',
        self_label: str = "self",
        comparator_label: str = "comparator",
        fdr: bool = True,
        sig_thresh: float = FDR_THRESH,
        estimate_thresh: float = ESTIMATE_THRESH,
        max_abs_estimate: float = 2.5,
        width: int = 720,
        height: int = 720,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        The volcano plot of each contrast set as a margin of the effect-size
        scatter of the bins significant in both.

        The comparator's volcano is drawn on its side at the upper left, so
        that its effect-size axis runs alongside the scatter's vertical axis;
        this set's volcano sits below the scatter, sharing its horizontal
        axis. The legend takes the otherwise empty lower-left quadrant.
        """
        sig_label = "q-value" if fdr else "p-value"
        fig = make_subplots(
            rows=2,
            cols=2,
            horizontal_spacing=0.10,
            vertical_spacing=0.10,
            subplot_titles=(comparator_label, "Significant in both", None, self_label),
        )
        volcano = dict(estimate_thresh=estimate_thresh, fdr_thresh=sig_thresh, max_abs_estimate=max_abs_estimate)
        # The legend is drawn from this set's volcano, which shows every
        # organism; the other panels repeat the same names
        for trace in self.volcano_plot(**volcano).data:
            fig.add_trace(trace.update(showlegend=True, legendgroup=trace.name), row=2, col=2)
        for trace in comparator.volcano_plot(**volcano).data:
            fig.add_trace(trace.update(x=trace.y, y=trace.x, showlegend=False, legendgroup=trace.name), row=1, col=1)
        for trace in self.compare_association_scatter(
            comparator=comparator, fdr=fdr, sig_thresh=sig_thresh, estimate_thresh=estimate_thresh,
        ).data:
            fig.add_trace(trace.update(showlegend=False, legendgroup=trace.name), row=1, col=2)

        # Linked rather than shared, so every panel keeps its own tick labels
        fig.update_yaxes(matches="y", row=1, col=2)
        fig.update_xaxes(matches="x4", row=1, col=2)

        make_lines(0, fig, vline=False, row=1, col=1, **ZERO_LINE)
        make_lines(estimate_thresh, fig, vline=False, row=1, col=1, **THRESHOLD_LINE)
        make_lines(-np.log10(sig_thresh), fig, hline=False, neg=False, row=1, col=1, **THRESHOLD_LINE)
        make_lines(0, fig, row=1, col=2, **ZERO_LINE)
        make_lines(0, fig, hline=False, row=2, col=2, **ZERO_LINE)
        make_lines(estimate_thresh, fig, hline=False, row=2, col=2, **THRESHOLD_LINE)
        make_lines(-np.log10(sig_thresh), fig, vline=False, neg=False, row=2, col=2, **THRESHOLD_LINE)

        fig.update_layout(
            width=width,
            height=height,
            template=TEMPLATE,
            legend=dict(title_text="Organism", x=0.0, y=0.42, xanchor="left", yanchor="top"),
            margin=dict(t=50),
        )
        fig.update_xaxes(title_text=f"-log10({sig_label})", row=1, col=1)
        fig.update_yaxes(title_text=f"Effect size, {comparator_label}", row=1, col=1)
        fig.update_yaxes(title_text=f"Effect size, {comparator_label}", row=1, col=2)
        fig.update_xaxes(title_text=f"Effect size, {self_label}", row=2, col=2)
        fig.update_yaxes(title_text=f"-log10({sig_label})", row=2, col=2)
        save_image(fig, file_prefix)
        return fig


#: The three ways an analysis can call a bin, and how each is drawn
SIG_LEVELS = ["<", "=", ">"]
_SIG_ARROW = {"<": "&#8595;", ">": "&#8593;", "=": "ns"}

#: The nine categories, ordered by meaning: (self, comparator, group)
_SIG_CATEGORY_ORDER = [
    ("<", "<", "Same direction"),
    (">", ">", "Same direction"),
    ("<", ">", "Opposite"),
    (">", "<", "Opposite"),
    ("<", "=", "One study only"),
    (">", "=", "One study only"),
    ("=", "<", "One study only"),
    ("=", ">", "One study only"),
    ("=", "=", "Neither"),
]
