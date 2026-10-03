"""
ContrastMetagenomes: the association results and bin abundance of one
case/control comparison for one organism.
"""

from functools import cached_property
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from .dataset import Dataset
from ..helpers.format_pvalue import format_pvalue
from ..helpers.observed_expected import expected_counts, plot_observed_expected
from .sample_group import _key as _label_key
from ..helpers.make_lines import make_lines
from ..helpers.save_image import save_image
from ..helpers.style import (
    DENSE_MARKER_OPACITY, ESTIMATE_THRESH, FDR_THRESH, PRIMARY, TEMPLATE, THRESHOLD_LINE,
    ZERO_LINE, group_colors, legend_above,
)

#: How the association columns are labeled wherever they are plotted
ASSOCIATION_LABELS = dict(
    Estimate_clipped="Effect size",
    Estimate="Effect size",
    neg_log10_qvalue="-log10(q-value)",
    neg_log10_pvalue="-log10(p-value)",
    signed_log10_qvalue="Signed -log10(q-value)",
    signed_log10_pvalue="Signed -log10(p-value)",
    feature="Pangenome bin",
    mean_abund="Mean abundance (RPKM)",
    pangenome="Organism",
    qvalue="q-value",
    pvalue="p-value",
)


def volcano_figure(
    association: pd.DataFrame,
    estimate_thresh: float,
    fdr_thresh: float,
    max_abs_estimate: float,
    **scatter_kwargs,
) -> go.Figure:
    """
    Effect size against -log10 q-value for every bin in ``association``,
    with the effect size clipped to ``max_abs_estimate`` so that a few
    extreme bins do not crush the rest, and the significance thresholds
    drawn. ``scatter_kwargs`` go to ``px.scatter``.
    """
    df = association.assign(
        Estimate_clipped=association["Estimate"].clip(lower=-max_abs_estimate, upper=max_abs_estimate)
    )
    fig = px.scatter(
        data_frame=df,
        x="Estimate_clipped",
        y="neg_log10_qvalue",
        template=TEMPLATE,
        labels=ASSOCIATION_LABELS,
        **scatter_kwargs,
    )
    make_lines(0, fig, **ZERO_LINE)
    make_lines(estimate_thresh, fig, hline=False, **THRESHOLD_LINE)
    make_lines(-np.log10(fdr_thresh), fig, vline=False, neg=False, **THRESHOLD_LINE)
    return fig


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
        self.parameter = parameter

    def __repr__(self) -> str:
        return f"ContrastMetagenomes(directory={self.directory}, parameter={self.parameter})"

    @cached_property
    def association(self) -> pd.DataFrame:
        """
        Association results from association/association.csv, filtered to
        the specified parameter, with each bin's mean abundance added.
        """
        path = self.directory / "association" / "association.csv"
        df = pd.read_csv(path)
        if self.parameter not in df["parameter"].values:
            raise ValueError(f"parameter {self.parameter} not found in association.csv")
        df = df.loc[df["parameter"] == self.parameter].drop(columns=["parameter"])
        return df.assign(mean_abund=df["feature"].map(self.mean_abund).fillna(0))

    @cached_property
    def rpkm(self) -> pd.DataFrame:
        """Bin abundance (RPKM) from bin_abundance/rpkm.csv.gz: specimens by bins."""
        return pd.read_csv(self.directory / "bin_abundance" / "rpkm.csv.gz", index_col=0)

    @cached_property
    def abund(self) -> pd.DataFrame:
        """
        Per-bin abundance from association/abund.csv, as it was handed to the
        association model. Unlike `rpkm` this is not scaled by pangenome size.
        """
        return pd.read_csv(self.directory / "association" / "abund.csv", index_col=0)

    @cached_property
    def metadata(self) -> pd.DataFrame:
        """Sample metadata from association/metadata.csv."""
        return pd.read_csv(self.directory / "association" / "metadata.csv", index_col=0)

    @cached_property
    def mean_abund(self) -> pd.Series:
        """Mean bin abundance (RPKM) for each bin."""
        return self.rpkm.mean()

    def calc_odds_ratio(
        self,
        metadata_col: str,
        ref_group,
        comp_group,
        bin_id: str,
        samples: pd.Index | None = None,
        threshold="median",
    ) -> float:
        """
        The odds of detecting one bin in ``comp_group`` against ``ref_group``,
        two values of a metadata column, among ``samples`` (every sample by
        default). A bin is detected at or above ``threshold`` RPKM, or above
        the "median" or "mean" abundance across the samples. One is added to
        every cell so that an empty cell gives a finite ratio.
        """
        from scipy import stats

        metadata = self.metadata
        if samples is not None:
            metadata = metadata.loc[metadata.index.intersection(samples)]
        df = pd.DataFrame(dict(groups=metadata[metadata_col], rpkm=self.rpkm[bin_id])).dropna()
        df = df.loc[df["groups"].isin([ref_group, comp_group])]
        df = df.assign(x=df["groups"].map({ref_group: 0, comp_group: 1}))

        if threshold == "median":
            threshold = df["rpkm"].median()
        elif threshold == "mean":
            threshold = df["rpkm"].mean()
        table = (
            pd.crosstab(df["rpkm"] >= threshold, df["x"])
            .reindex(index=[False, True], columns=[0, 1])
            .fillna(0)
            .astype(int)
            + 1
        )
        return stats.contingency.odds_ratio(table.values).statistic

    def volcano_plot(
        self,
        estimate_thresh: float = ESTIMATE_THRESH,
        fdr_thresh: float = FDR_THRESH,
        max_abs_estimate: float = 5.0,
        label_bins: tuple = (),
        width: int = 560,
        height: int = 450,
        file_prefix: str | None = None,
        **kwargs
    ) -> go.Figure:
        """
        Volcano plot from the association results, each bin sized by its mean
        abundance. Each bin named in ``label_bins`` is pointed out with an
        arrow and its name.
        """
        fig = volcano_figure(
            self.association, estimate_thresh, fdr_thresh, max_abs_estimate,
            hover_data=self.association.columns.values,
            hover_name="feature",
            size="mean_abund",
            size_max=14,
            color_discrete_sequence=[PRIMARY],
            width=width,
            height=height,
            **kwargs
        )
        fig.update_traces(marker=dict(opacity=DENSE_MARKER_OPACITY, line_width=0))
        for bin in label_bins:
            point = self.association.loc[self.association["feature"] == bin]
            if point.empty:
                raise ValueError(f"{bin} is not among the bins this contrast tested")
            fig.add_annotation(
                x=float(point["Estimate"].clip(-max_abs_estimate, max_abs_estimate).iloc[0]),
                y=float(point["neg_log10_qvalue"].iloc[0]),
                text=bin, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
                arrowcolor="#333333", ax=45, ay=-25, font=dict(size=13),
            )
        save_image(fig, file_prefix)
        return fig

    def bin_contingency(
        self,
        bin: str,
        metadata_col: str,
        norm_bin: str | None = None,
        threshold: float = 0.25,
    ) -> dict:
        """
        Call a bin present or absent in each sample, and cross that against a
        metadata column.

        A bin is called present where its abundance is at least `threshold`.
        With `norm_bin` that abundance is relative to another bin -- normally
        the core genome, which makes the threshold a fraction of the genome
        copies carrying the bin rather than a raw depth.

        Returns the 2x2 table together with Fisher's exact test of it.
        """
        from scipy import stats

        abundance = (
            self.rpkm[bin] if norm_bin is None else self.rpkm[bin] / self.rpkm[norm_bin]
        )
        df = pd.DataFrame({
            "group": self.metadata[metadata_col],
            "present": (abundance >= threshold),
        }).dropna()

        groups = sorted(df["group"].unique())
        if len(groups) != 2:
            raise ValueError(
                f"{metadata_col} has {len(groups)} values {groups}; "
                "a 2x2 contingency table needs exactly two"
            )

        table = (
            pd.crosstab(df["present"], df["group"])
            .reindex(index=[True, False], columns=groups)
            .fillna(0)
            .astype(int)
        )
        odds_ratio, pvalue = stats.fisher_exact(table.values)

        return dict(
            table=table,
            expected=expected_counts(table),
            odds_ratio=odds_ratio,
            pvalue=pvalue,
            threshold=threshold,
            n=int(table.values.sum()),
        )

    def plot_bin_contingency(
        self,
        bin: str,
        metadata_col: str,
        norm_bin: str | None = None,
        threshold: float = 0.25,
        group_labels: dict | None = None,
        width: int = 660,
        height: int = 520,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        The 2x2 table from :meth:`bin_contingency`, drawn the same way as the
        study-concordance figure: the observed count of each combination
        beside the count expected if bin presence and the metadata column were
        independent.

        ``group_labels`` renames the metadata values for display.
        """
        result = self.bin_contingency(bin, metadata_col, norm_bin, threshold)
        table = result["table"]
        # Metadata values arrive as numbers while the study definition keys its
        # labels by string, so both are normalized the same way
        labels = {_label_key(k): v for k, v in (group_labels or {}).items()}

        # Grouped by presence, so the two metadata groups sit side by side and
        # the comparison the figure is for is the one between adjacent bars
        ticks, groups, observed, expected = [], [], [], []
        for present in (True, False):
            for group in table.columns:
                ticks.append(str(labels.get(_label_key(group), group)))
                groups.append(f"{bin} {'present' if present else 'absent'}")
                observed.append(int(table.loc[present, group]))
                expected.append(result["expected"].loc[present, group])

        measured = f"{bin} / {norm_bin}" if norm_bin is not None else f"{bin} (RPKM)"
        return plot_observed_expected(
            ticks=ticks,
            groups=groups,
            observed=observed,
            expected=expected,
            y_title="Samples",
            caption=(
                f"Present at {measured} &#8805; {threshold:g} &#183; n = {result['n']} "
                f"&#183; odds ratio {result['odds_ratio']:.2f} "
                f"&#183; Fisher's exact p = {format_pvalue(result['pvalue'])}"
            ),
            facet_groups=True,
            width=width,
            height=height,
            file_prefix=file_prefix,
        )

    def plot_bin_abundance(
        self,
        bin: str,
        norm_bin: str | None = None,
        group_labels: dict | None = None,
        group_order: list | None = None,
        x_title: str | None = None,
        width: int = 500,
        height: int = 400,
        file_prefix: str | None = None,
        **kwargs
    ) -> go.Figure:
        """
        A histogram of the bin's abundance across samples, optionally relative
        to ``norm_bin``; ``kwargs`` go to ``px.histogram`` (``color``,
        ``facet_row``, ``histnorm`` ...). ``x_title`` names the abundance
        axis, by default after the bins it measures.

        ``group_labels`` renames the values of the metadata column used for
        ``color`` / ``facet_row`` (say 1 to "BSI"), and ``group_order`` gives
        the order of those display names, the first being the case-like one.
        """
        df = self.metadata.assign(
            abundance=(
                self.rpkm.loc[:, bin]
                if norm_bin is None
                else self.rpkm.loc[:, bin] / self.rpkm.loc[:, norm_bin]
            )
        )
        labels = {_label_key(k): v for k, v in (group_labels or {}).items()}
        grouping = {kwargs.get(key) for key in ("color", "facet_row", "facet_col")} - {None}
        for column in grouping:
            df[column] = df[column].map(lambda v: labels.get(_label_key(v), v))
        if group_order is not None:
            kwargs.setdefault("category_orders", {column: list(group_order) for column in grouping})
            kwargs.setdefault("color_discrete_map", group_colors(group_order))

        fig = px.histogram(
            data_frame=df,
            x="abundance",
            template=TEMPLATE,
            width=width,
            height=height,
            **kwargs
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=", 1)[-1]))

        # Faceting repeats each axis title once per facet, which collides for
        # any title of a reasonable length: one centered title per axis
        # instead, sized as the template draws one
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="")
        if x_title is None:
            x_title = f"Abundance of {bin} (RPKM)" if norm_bin is None else f"Abundance of {bin} / {norm_bin}"
        fig.add_annotation(
            text=x_title, xref="paper", yref="paper", x=0.5, y=0, yshift=-30,
            yanchor="top", showarrow=False, font=dict(size=14),
        )
        fig.add_annotation(
            text=f"{kwargs.get('histnorm', 'number').capitalize()} of samples",
            xref="paper", yref="paper", x=0, y=0.5, xshift=-55, textangle=-90,
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(margin=dict(l=75, b=75))
        legend_above(fig)
        save_image(fig, file_prefix)
        return fig
