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
from ..helpers.observed_expected import expected_counts, plot_observed_expected, plot_observed_expected_heatmap
from .sample_group import _key as _label_key
from ..helpers.make_lines import make_lines
from ..helpers.save_image import save_image
from ..helpers.significance import add_significance
from ..helpers.style import (
    DENSE_MARKER_OPACITY, FDR_THRESH, MIN_ABS_LOG2_FOLD_CHANGE, MIN_MEAN_RPKM, NEUTRAL, PRIMARY,
    TEMPLATE, THRESHOLD_LINE, ZERO_LINE, group_colors, legend_above,
)

LOG2_FOLD_CHANGE_LABEL = "log<sub>2</sub>(cases / controls)"

#: How the association columns are labeled wherever they are plotted
ASSOCIATION_LABELS = dict(
    log2_fold_change_clipped=LOG2_FOLD_CHANGE_LABEL,
    log2_fold_change=LOG2_FOLD_CHANGE_LABEL,
    Estimate="Estimate (fold change - 1)",
    neg_log10_qvalue="-log10(q-value)",
    neg_log10_pvalue="-log10(p-value)",
    signed_log10_qvalue="Signed -log10(q-value)",
    signed_log10_pvalue="Signed -log10(p-value)",
    feature="Pangenome bin",
    mean_abund_clipped="Mean abundance (RPKM)",
    mean_abund="Mean abundance (RPKM)",
    pangenome="Organism",
    qvalue="q-value",
    pvalue="p-value",
)

#: Volcano and M-A points are sized by mean abundance, by area; the floor
#: keeps the many rare bins visible beside a few that run to thousands of RPKM
SIZE_BY_ABUNDANCE = dict(size="mean_abund", size_max=14)
MIN_MARKER_PX = 4
#: Where a volcano's bin label sits relative to its point, in pixels
LABEL_OFFSET_PX = (45, -25)

#: One facet row of an abundance histogram, and the gap between rows
HISTOGRAM_ROW_PX = 150
HISTOGRAM_ROW_GAP_PX = 30

#: The columns every association plot shows on hover
ASSOCIATION_HOVER = ["mean_abund", "log2_fold_change", "Estimate", "pvalue", "qvalue"]

#: Where a bin with a mean RPKM below this is drawn on a log axis, which
#: cannot show zero
MIN_PLOTTED_RPKM = 0.01

#: The legend entry for the bins drawn at an axis limit
BEYOND_AXIS = "Beyond the axis"


def clip_log2_fold_change(association: pd.DataFrame, max_abs_log2_fold_change: float) -> pd.DataFrame:
    """
    ``log2_fold_change_clipped``: the log2 fold change held within the axis,
    so that a few extreme bins -- including those absent from one group,
    which are infinite -- do not crush the rest.
    """
    return association.assign(
        log2_fold_change_clipped=association["log2_fold_change"].clip(-max_abs_log2_fold_change, max_abs_log2_fold_change)
    )


#: The per-point properties of a scatter trace, which travel with a point
#: when a trace is split
_POINT_PROPERTIES = [("x",), ("y",), ("customdata",), ("hovertext",), ("text",), ("ids",),
                     ("marker", "size"), ("marker", "color")]

#: ``meta`` of the traces holding the points drawn at an axis limit
_BEYOND_META = "beyond"


def _take_points(trace, keep: np.ndarray) -> None:
    """Keep only the points of ``trace`` where ``keep`` is true, in place."""
    for path in _POINT_PROPERTIES:
        owner = trace if len(path) == 1 else trace[path[0]]
        value = owner[path[-1]]
        if value is not None and not isinstance(value, (str, int, float)):
            owner[path[-1]] = np.asarray(value)[keep]


def mark_beyond_axis(fig: go.Figure, limit: float, axis: str) -> None:
    """
    Draw every point sitting at ``±limit`` on ``axis`` ("x" or "y") as a
    triangle pointing off the axis, and add a legend entry saying so where
    there are any.

    The points move to a trace of their own beside the one they came from,
    in its legend group but not in the legend, since a legend icon is drawn
    from a trace's first point. A point already moved for the other axis
    keeps its triangle.
    """
    low, high = ("triangle-left", "triangle-right") if axis == "x" else ("triangle-down", "triangle-up")
    beyond_traces = []
    for trace in fig.data:
        if trace.name == BEYOND_AXIS or trace.meta == _BEYOND_META:
            continue
        values = np.asarray(trace[axis], dtype=float)
        beyond = np.abs(values) >= limit
        if not beyond.any():
            continue
        part = type(trace)(trace)
        _take_points(part, beyond)
        _take_points(trace, ~beyond)
        part.update(
            showlegend=False,
            legendgroup=trace.legendgroup or trace.name,
            meta=_BEYOND_META,
            marker_symbol=np.where(values[beyond] > 0, high, low),
        )
        trace.legendgroup = trace.legendgroup or trace.name
        beyond_traces.append(part)
    fig.add_traces(beyond_traces)
    if beyond_traces and BEYOND_AXIS not in [trace.name for trace in fig.data]:
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers", name=BEYOND_AXIS, legendrank=1002,
                       marker=dict(symbol=high, color=NEUTRAL, size=8)),
        )


def add_low_abundance(fig: go.Figure, low: pd.DataFrame, x: str, y: str, hover_name: str) -> None:
    """Draw the bins below the abundance filter in grey, beneath the rest and last in the legend."""
    fig.add_trace(
        go.Scattergl(
            x=low[x], y=low[y], mode="markers", name="Low abundance", legendrank=1001,
            # The least abundant bins of all, so drawn at the smallest size
            marker=dict(color=NEUTRAL, opacity=0.35, size=MIN_MARKER_PX, line_width=0),
            hovertext=low[hover_name], hoverinfo="text",
        ),
    )
    fig.data = (fig.data[-1],) + fig.data[:-1]


def volcano_figure(
    association: pd.DataFrame,
    fdr_thresh: float,
    max_abs_log2_fold_change: float,
    show_low_abundance: bool,
    fold_change_label: str = LOG2_FOLD_CHANGE_LABEL,
    **scatter_kwargs,
) -> go.Figure:
    """
    log2 fold change against significance, one point per bin, drawn one of
    two ways.

    By default, the bins that pass the abundance filter against their
    q-value, with the thresholds that call a bin drawn. With
    ``show_low_abundance``, every bin against its p-value -- the only
    measure the bins below the filter have -- with those bins in grey and
    no thresholds, since calls are made on q-values among the bins kept.
    ``scatter_kwargs`` go to ``px.scatter``, and must name ``hover_name``.
    """
    df = clip_log2_fold_change(association, max_abs_log2_fold_change)
    y = "neg_log10_pvalue" if show_low_abundance else "neg_log10_qvalue"
    fig = px.scatter(
        data_frame=df.loc[~df["low_abundance"]],
        x="log2_fold_change_clipped",
        y=y,
        template=TEMPLATE,
        labels={**ASSOCIATION_LABELS, "log2_fold_change_clipped": fold_change_label},
        **scatter_kwargs,
    )
    if show_low_abundance:
        add_low_abundance(fig, df.loc[df["low_abundance"]], "log2_fold_change_clipped", y, scatter_kwargs["hover_name"])
    mark_beyond_axis(fig, max_abs_log2_fold_change, "x")
    make_lines(0, fig, **ZERO_LINE)
    if not show_low_abundance:
        make_lines(MIN_ABS_LOG2_FOLD_CHANGE, fig, hline=False, **THRESHOLD_LINE)
        make_lines(-np.log10(fdr_thresh), fig, vline=False, neg=False, **THRESHOLD_LINE)
    return fig


def ma_figure(
    association: pd.DataFrame,
    max_abs_log2_fold_change: float,
    fold_change_label: str = LOG2_FOLD_CHANGE_LABEL,
    **scatter_kwargs,
) -> go.Figure:
    """
    log2 fold change against mean abundance, one point per bin: the bins
    below the abundance filter in grey, the filter as a vertical line, and
    the fold-change thresholds as horizontal ones. ``scatter_kwargs`` go to
    ``px.scatter``, and must name ``hover_name``.
    """
    df = clip_log2_fold_change(association, max_abs_log2_fold_change).assign(
        mean_abund_clipped=association["mean_abund"].clip(lower=MIN_PLOTTED_RPKM)
    )
    fig = px.scatter(
        data_frame=df.loc[~df["low_abundance"]],
        x="mean_abund_clipped",
        y="log2_fold_change_clipped",
        log_x=True,
        template=TEMPLATE,
        labels={**ASSOCIATION_LABELS, "log2_fold_change_clipped": fold_change_label},
        **scatter_kwargs,
    )
    add_low_abundance(fig, df.loc[df["low_abundance"]], "mean_abund_clipped", "log2_fold_change_clipped", scatter_kwargs["hover_name"])
    mark_beyond_axis(fig, max_abs_log2_fold_change, "y")
    make_lines(0, fig, vline=False, neg=False, **ZERO_LINE)
    make_lines(MIN_ABS_LOG2_FOLD_CHANGE, fig, vline=False, **THRESHOLD_LINE)
    make_lines(MIN_MEAN_RPKM, fig, hline=False, neg=False, **THRESHOLD_LINE)
    # An annotation on a log axis is placed in log10 units
    fig.add_annotation(
        x=np.log10(MIN_MEAN_RPKM), y=1, yref="paper", text=f"{MIN_MEAN_RPKM:g} RPKM", showarrow=False,
        xanchor="left", yanchor="bottom", font=dict(color=THRESHOLD_LINE["line_color"]),
    )
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
        the specified parameter, with each bin's mean abundance added and the
        abundance filter, q-values and fold changes of
        ``helpers.significance.add_significance``.
        """
        path = self.directory / "association" / "association.csv"
        df = pd.read_csv(path)
        if self.parameter not in df["parameter"].values:
            raise ValueError(f"parameter {self.parameter} not found in association.csv")
        df = df.loc[df["parameter"] == self.parameter].drop(columns=["parameter"])
        return add_significance(df.assign(mean_abund=df["feature"].map(self.mean_abund).fillna(0)))

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
        fdr_thresh: float = FDR_THRESH,
        max_abs_log2_fold_change: float = 4.0,
        show_low_abundance: bool = False,
        label_bins: tuple | dict = (),
        fold_change_label: str = LOG2_FOLD_CHANGE_LABEL,
        width: int = 560,
        height: int = 450,
        size_max: int = SIZE_BY_ABUNDANCE["size_max"],
        file_prefix: str | None = None,
        **kwargs
    ) -> go.Figure:
        """
        Volcano plot from the association results (see ``volcano_figure``
        for its two versions), each bin sized by its mean abundance up to
        ``size_max`` pixels across. Each bin named in ``label_bins`` is
        pointed out with an arrow and its name, its label up and to the right
        unless ``label_bins`` maps the bin to its own (x, y) offset in pixels,
        for labels that would otherwise collide. ``fold_change_label`` names
        the two groups where they are not cases and controls.
        """
        fig = volcano_figure(
            self.association, fdr_thresh, max_abs_log2_fold_change, show_low_abundance, fold_change_label,
            hover_data=ASSOCIATION_HOVER,
            hover_name="feature",
            color_discrete_sequence=[PRIMARY],
            opacity=DENSE_MARKER_OPACITY,
            width=width,
            height=height,
            **{**SIZE_BY_ABUNDANCE, "size_max": size_max},
            **kwargs
        )
        fig.update_traces(marker_line_width=0, marker_sizemin=MIN_MARKER_PX)
        y = "neg_log10_pvalue" if show_low_abundance else "neg_log10_qvalue"
        offsets = label_bins if isinstance(label_bins, dict) else dict.fromkeys(label_bins, LABEL_OFFSET_PX)
        for bin, (label_x, label_y) in offsets.items():
            point = self.association.loc[self.association["feature"] == bin]
            if point.empty:
                raise ValueError(f"{bin} is not among the bins this contrast tested")
            if point["low_abundance"].iloc[0] and not show_low_abundance:
                raise ValueError(f"{bin} is below the abundance filter, so this version of the plot does not show it")
            fig.add_annotation(
                x=float(point["log2_fold_change"].clip(-max_abs_log2_fold_change, max_abs_log2_fold_change).iloc[0]),
                y=float(point[y].iloc[0]),
                text=bin, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
                arrowcolor="#333333", ax=label_x, ay=label_y, font=dict(size=13),
            )
        save_image(fig, file_prefix)
        return fig

    def ma_plot(
        self,
        max_abs_log2_fold_change: float = 4.0,
        fold_change_label: str = LOG2_FOLD_CHANGE_LABEL,
        width: int = 560,
        height: int = 450,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """M-A plot from the association results (see ``ma_figure``)."""
        fig = ma_figure(
            self.association, max_abs_log2_fold_change, fold_change_label,
            hover_data=ASSOCIATION_HOVER,
            hover_name="feature",
            color_discrete_sequence=[PRIMARY],
            opacity=DENSE_MARKER_OPACITY,
            width=width,
            height=height,
            **SIZE_BY_ABUNDANCE,
        )
        fig.update_traces(marker_line_width=0, marker_sizemin=MIN_MARKER_PX)
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
        width: int | None = None,
        height: int | None = None,
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

        return plot_observed_expected(
            ticks=ticks,
            groups=groups,
            observed=observed,
            expected=expected,
            y_title="Samples",
            caption=_contingency_caption(result, bin, norm_bin),
            facet_groups=True,
            width=width,
            height=height,
            file_prefix=file_prefix,
        )

    def plot_bin_contingency_heatmap(
        self,
        bin: str,
        metadata_col: str,
        norm_bin: str | None = None,
        threshold: float = 0.25,
        group_labels: dict | None = None,
        width: int = 460,
        height: int = 340,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        The 2x2 table of :meth:`plot_bin_contingency` as a grid, drawn the
        same way as the study-concordance heatmap: present above absent, the
        metadata groups across, each cell colored by observed over expected
        and labeled with both counts. The caption carries the same test.
        """
        result = self.bin_contingency(bin, metadata_col, norm_bin, threshold)
        labels = {_label_key(k): v for k, v in (group_labels or {}).items()}
        # Plotly draws the first row at the bottom
        rows = {False: "Absent", True: "Present"}
        columns = {group: str(labels.get(_label_key(group), group)) for group in result["table"].columns}
        return plot_observed_expected_heatmap(
            # reindex, not loc: a list of booleans given to loc is a mask
            result["table"].reindex(list(rows)).rename(index=rows, columns=columns),
            result["expected"].reindex(list(rows)).rename(index=rows, columns=columns),
            x_title="",
            y_title=bin,
            caption=_contingency_caption(result, bin, norm_bin),
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
        width: int = 560,
        height: int | None = None,
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
        The height follows the number of ``facet_row`` rows unless given. A
        legend that would only repeat the row labels, because ``color`` and
        ``facet_row`` name the same column, is left out.
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

        rows = df[kwargs["facet_row"]].nunique() if "facet_row" in kwargs else 1
        legend = kwargs.get("color") is not None and kwargs.get("color") != kwargs.get("facet_row")
        # The abundance axis title may run to more than one line
        top, bottom = (70 if legend else 30), 75 + 16 * (x_title or "").count("<br>")
        if height is None:
            height = top + bottom + HISTOGRAM_ROW_PX * rows + HISTOGRAM_ROW_GAP_PX * (rows - 1)
        if rows > 1:
            kwargs.setdefault("facet_row_spacing", HISTOGRAM_ROW_GAP_PX / (height - top - bottom))
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
        fig.update_layout(margin=dict(l=75, b=bottom, t=top), showlegend=legend)
        if legend:
            legend_above(fig)
        save_image(fig, file_prefix)
        return fig


def _contingency_caption(result: dict, bin: str, norm_bin: str | None) -> str:
    """What a bin-presence figure was called on, and Fisher's exact test of it."""
    measured = f"{bin} / {norm_bin}" if norm_bin is not None else f"{bin} (RPKM)"
    return (
        f"Present at {measured} &#8805; {result['threshold']:g} &#183; n = {result['n']} "
        f"<br>odds ratio {result['odds_ratio']:.2f} "
        f"&#183; Fisher's exact p = {format_pvalue(result['pvalue'])}"
    )
