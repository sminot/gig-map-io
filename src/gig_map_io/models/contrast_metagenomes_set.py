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

from gig_map_io.helpers.clustering import linkage_order
from gig_map_io.helpers.make_lines import make_lines
from gig_map_io.helpers.save_image import save_image
from gig_map_io.helpers.format_pvalue import format_pvalue
from gig_map_io.helpers.observed_expected import (
    expected_counts, plot_observed_expected, plot_observed_expected_heatmap,
)
from gig_map_io.helpers.sidebar import add_category_sidebar
from gig_map_io.helpers.significance import DIRECTIONS, add_significance, call_direction
from gig_map_io.helpers.style import (
    DENSE_MARKER_OPACITY, FDR_THRESH, MIN_ABS_LOG2_FOLD_CHANGE, QUALITATIVE, SIMPLE_TEMPLATE, TEMPLATE,
    THRESHOLD_LINE, TOP_LEGEND, ZERO_LINE, group_colors, organism_colors, organism_order,
)
from .contrast_metagenomes import (
    ASSOCIATION_HOVER, ASSOCIATION_LABELS, LOG2_FOLD_CHANGE_LABEL, MIN_MARKER_PX, SIZE_BY_ABUNDANCE,
    ContrastMetagenomes,
    _BEYOND_META, add_low_abundance, ma_figure, mark_beyond_axis, volcano_figure,
)
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
        """
        The association results of every contrast, with the q-values
        recomputed across all of them, among the bins that pass the
        abundance filter.
        """
        df = pd.concat([
            contrast.association.assign(pangenome=pangenome_name)
            for pangenome_name, contrast in self.contrast_metagenomes.items()
        ], ignore_index=True)
        return add_significance(df).sort_values(by=["pangenome", "pvalue", "feature"])

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

    def _organism_traces(self) -> dict:
        """``px.scatter`` arguments that color bins by organism, as everywhere, and size them by abundance."""
        return dict(
            hover_name="hover_name",
            color="pangenome",
            color_discrete_map=organism_colors(self.pangenome_names),
            category_orders={"pangenome": organism_order(self.pangenome_names)},
            hover_data=ASSOCIATION_HOVER,
            opacity=DENSE_MARKER_OPACITY,
            **SIZE_BY_ABUNDANCE,
        )

    def volcano_plot(
        self,
        fdr_thresh: float = FDR_THRESH,
        max_abs_log2_fold_change: float = 4.0,
        show_low_abundance: bool = False,
        width: int = 650,
        height: int = 450,
        file_prefix: str | None = None,
        **kwargs
    ) -> go.Figure:
        """
        Volcano plot from the association results, colored by organism; see
        ``volcano_figure`` for its two versions.
        """
        df = self.association.assign(hover_name=lambda d: d["pangenome"] + "<br>" + d["feature"])
        fig = volcano_figure(
            df, fdr_thresh, max_abs_log2_fold_change, show_low_abundance,
            width=width,
            height=height,
            **self._organism_traces(),
            **kwargs
        )
        fig.update_traces(marker_line_width=0, marker_sizemin=MIN_MARKER_PX)
        save_image(fig, file_prefix)
        return fig

    def ma_plot(
        self,
        max_abs_log2_fold_change: float = 4.0,
        width: int = 650,
        height: int = 450,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """M-A plot from the association results, colored by organism (see ``ma_figure``)."""
        df = self.association.assign(hover_name=lambda d: d["pangenome"] + "<br>" + d["feature"])
        fig = ma_figure(df, max_abs_log2_fold_change, width=width, height=height, **self._organism_traces())
        fig.update_traces(marker_line_width=0, marker_sizemin=MIN_MARKER_PX)
        save_image(fig, file_prefix)
        return fig

    def bin_abundance_heatmap(
        self,
        features: pd.MultiIndex,
        annotation_cols: dict[str, str] | None = None,
        annotation_labels: dict[str, dict] | None = None,
        annotation_orders: dict[str, list] | None = None,
        width: int = 520,
        height: int | None = None,
        annotation_width_fraction: float = 0.1,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        The abundance of a set of bins across samples, log10(RPKM + 1), with
        the Spearman correlation between every pair of bins beneath it and,
        optionally, a column of sample annotations beside it. Samples and
        bins are each ordered by hierarchical clustering. The correlation
        cells are square, and the height follows the number of bins unless
        given.

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
        # row 2: [            | correlation], sharing the bin axis with the abundance.
        # The correlation panel is as tall as the bin columns are wide, so
        # that its cells come out square
        rpkm_col = 2 if annotations is not None else 1
        longest = max(len(name) for name in sorted_features)
        margin = dict(l=70, r=110, t=50, b=45 + int(LABEL_CHAR_PX * longest))
        bins_px = (width - margin["l"] - margin["r"]) * (1.0 - annotation_width_fraction if annotations is not None else 1.0)
        if height is None:
            height = int(margin["t"] + margin["b"] + ABUNDANCE_PANEL_PX + HEATMAP_GAP_PX + bins_px)
        plot_height = height - margin["t"] - margin["b"]
        row_heights = [(plot_height - HEATMAP_GAP_PX - bins_px) / plot_height, bins_px / plot_height]
        fig = make_subplots(
            rows=2,
            cols=rpkm_col,
            specs=[[{}, {}], [None, {}]] if annotations is not None else [[{}], [{}]],
            shared_xaxes=True,
            shared_yaxes=annotations is not None,
            column_widths=[annotation_width_fraction, 1.0 - annotation_width_fraction] if annotations is not None else None,
            row_heights=row_heights,
            horizontal_spacing=0.02,
            vertical_spacing=HEATMAP_GAP_PX / plot_height,
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
                tickangle=0,
                side="top",
                ticks="",
                range=[-0.5, len(annotations.columns) - 0.5],
                showticklabels=True,
                row=1, col=1,
            )
            fig.update_yaxes(showticklabels=False, ticks="", row=1, col=1)

        # The categorical legend runs along the top, leaving the right margin
        # to the two color bars
        fig.update_layout(
            width=width,
            height=height,
            template=TEMPLATE,
            showlegend=annotations is not None,
            # One row, the annotation column's name leading its entries
            # rather than stacked above them
            legend=dict(TOP_LEGEND, traceorder="normal", y=1.04),
            margin=margin,
        )
        fig.update_yaxes(title_text="Samples", row=1, col=1)
        fig.update_yaxes(title_text="Pangenome bin", showticklabels=False, ticks="", row=2, col=rpkm_col)
        fig.update_xaxes(title_text="Pangenome bin", row=2, col=rpkm_col)
        fig.update_xaxes(showticklabels=False, ticks="", row=1, col=rpkm_col)
        fig.update_yaxes(showticklabels=False, ticks="", row=1, col=rpkm_col)
        save_image(fig, file_prefix)
        return fig

    def compare_association(
        self,
        comparator: 'ContrastMetagenomesSet',
        fdr: bool = True,
        sig_thresh: float = FDR_THRESH,
    ) -> pd.DataFrame:
        """
        The association results of the two sets side by side, one row per bin
        tested in both, with each set's call in ``call_self`` and
        ``call_comparator``.
        """
        return (
            self.association
            .assign(call=lambda d: call_direction(d, fdr=fdr, sig_thresh=sig_thresh))
            .merge(
                comparator.association.assign(call=lambda d: call_direction(d, fdr=fdr, sig_thresh=sig_thresh)),
                on=["pangenome", "feature"],
                suffixes=("_self", "_comparator")
            )
            .assign(
                mean_abund=lambda x: x[["mean_abund_self", "mean_abund_comparator"]].mean(axis=1),
            )
        )

    def _call_contingency(
        self,
        comparator: 'ContrastMetagenomesSet',
        fdr: bool,
        sig_thresh: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """
        The nine-category table of the two sets' calls (rows the comparator's,
        columns this set's), the counts expected if the calls were
        independent, and a caption line giving the direction agreement and the
        chi-squared test described in ``compare_sig_categories``.
        """
        df = self.compare_association(comparator, fdr=fdr, sig_thresh=sig_thresh)
        observed = df.pivot_table(
            columns="call_self",
            index="call_comparator",
            values="feature",
            aggfunc="count",
            fill_value=0,
        ).reindex(index=DIRECTIONS, columns=DIRECTIONS, fill_value=0)
        expected = expected_counts(observed)

        directions = observed.loc[["<", ">"], ["<", ">"]]
        agree = int(directions.loc["<", "<"] + directions.loc[">", ">"])
        both = int(directions.values.sum())
        # A call no bin received leaves an empty row or column, which has no
        # expected count to test against
        tested = observed.loc[observed.sum(axis=1) > 0, observed.sum(axis=0) > 0]
        if min(tested.shape) < 2:
            test = "too few calls to test"
        else:
            chi2, pvalue, dof, expected_tested = stats.chi2_contingency(tested.values)
            if expected_tested.min() < MIN_EXPECTED:
                tables = stats.random_table(tested.sum(axis=1), tested.sum(axis=0), seed=0).rvs(MONTE_CARLO_TABLES)
                exceeding = int(np.sum(np.array([_chi2_statistic(table) for table in tables]) >= chi2))
                pvalue = (1 + exceeding) / (1 + MONTE_CARLO_TABLES)
                # With no random table as extreme, the estimate is only a bound
                method = "Monte Carlo p " + ("≤" if exceeding == 0 else "=")
            else:
                method = "p ="
            test = f"chi-squared = {chi2:.1f}, {dof} df, {method} {format_pvalue(pvalue)}"
        subtitle = f"Direction agrees for {agree:,} of the {both:,} bins both call<br>{test}"
        return observed, expected, subtitle

    def compare_sig_categories(
        self,
        comparator: 'ContrastMetagenomesSet',
        fdr: bool = True,
        sig_thresh: float = FDR_THRESH,
        self_label: str = "self",
        comparator_label: str = "comparator",
        width: int | None = None,
        height: int | None = None,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Whether two analyses agree on the direction of the bins they both call.

        Every bin falls into one of nine categories: each analysis called it
        lower in cases, higher in cases, or neither -- including the bins too
        scarce to test. The bars give the observed count of each category
        beside the count expected if the two analyses were independent, and
        the caption gives the chi-squared test of that independence over all
        nine cells. It answers whether the two analyses call the same bins in
        the same direction more than chance would, which a test confined to
        the bins both call cannot do when one analysis calls nearly every
        such bin in one direction. Where any expected count is below
        ``MIN_EXPECTED`` the p-value is taken from tables drawn at random with
        the same margins rather than from the chi-squared distribution.

        The test counts bins as independent, which they are not: bins of one
        pangenome rise and fall together. The p-value is optimistic.

        The categories are ordered by what they mean rather than by their
        position in the contingency table -- agreement first, then
        disagreement, then bins only one analysis called -- so that the
        question the figure answers can be read left to right.
        """
        observed, expected, subtitle = self._call_contingency(comparator, fdr, sig_thresh)

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
            y_title="Pangenome bins",
            caption=f"{subtitle}<br>Ticks: <b>{self_label}</b> (upper), <b>{comparator_label}</b> (lower)",
            width=width,
            height=height,
            file_prefix=file_prefix,
        )

    def compare_sig_heatmap(
        self,
        comparator: 'ContrastMetagenomesSet',
        fdr: bool = True,
        sig_thresh: float = FDR_THRESH,
        self_label: str = "self",
        comparator_label: str = "comparator",
        width: int = 560,
        height: int = 380,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        The nine categories of ``compare_sig_categories`` as a 3x3 grid, one
        set's call across and the other's up: each cell colored by how many
        times more (warm) or fewer (cool) bins it holds than independence
        predicts, on a log scale centered on 1x, and labeled with its observed
        and expected counts. Bins agreeing in direction lie on the diagonal
        from lower left to upper right. The caption carries the same test.
        """
        observed, expected, subtitle = self._call_contingency(comparator, fdr, sig_thresh)
        # Plotly draws the first row at the bottom
        order = ["<", "=", ">"]
        names = {"<": "Lower in cases", "=": "Not called", ">": "Higher in cases"}
        return plot_observed_expected_heatmap(
            observed.loc[order, order].rename(index=names, columns=names),
            expected.loc[order, order].rename(index=names, columns=names),
            x_title=self_label, y_title=comparator_label, caption=subtitle,
            width=width, height=height, file_prefix=file_prefix,
        )

    def compare_sig_scatter(
        self,
        comparator: 'ContrastMetagenomesSet',
        self_label: str = "self",
        comparator_label: str = "comparator",
        width: int = 650,
        height: int = 450,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Signed significance of every bin in one set against the other:
        -log10 p-value, signed by the direction of association. p-values
        rather than q-values, so that the bins below the abundance filter in
        either set, which have no q-value, can be shown, in grey. No
        threshold is drawn, since calls are made on q-values among the bins
        kept.
        """
        value_col = "signed_log10_pvalue"
        df = self.compare_association(comparator).assign(
            hover_name=lambda d: d["pangenome"] + "<br>" + d["feature"]
        )
        low = df["low_abundance_self"] | df["low_abundance_comparator"]
        fig = self._organism_scatter(
            df.loc[~low],
            x=f"{value_col}_self",
            y=f"{value_col}_comparator",
            marker=dict(size=6, opacity=DENSE_MARKER_OPACITY),
            labels={
                **ASSOCIATION_LABELS,
                f"{value_col}_self": f"signed -log10(p-value)<br>{self_label}",
                f"{value_col}_comparator": f"signed -log10(p-value)<br>{comparator_label}",
                "pvalue_self": f"p-value ({self_label})",
                "pvalue_comparator": f"p-value ({comparator_label})",
            },
            hover_data=["pvalue_self", "pvalue_comparator"],
            width=width,
            height=height,
        )
        add_low_abundance(fig, df.loc[low], f"{value_col}_self", f"{value_col}_comparator", "hover_name")
        make_lines(0, fig, **ZERO_LINE)
        save_image(fig, file_prefix)
        return fig

    def compare_association_scatter(
        self,
        comparator: 'ContrastMetagenomesSet',
        self_label: str = "self",
        comparator_label: str = "comparator",
        fdr: bool = True,
        sig_thresh: float = FDR_THRESH,
        max_abs_log2_fold_change: float = 4.0,
        width: int = 650,
        height: int = 450,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        log2 fold change in one set against the other, for the bins called
        in both, each beyond the axis drawn at its limit as a triangle.
        """
        df = self.compare_association(comparator, fdr=fdr, sig_thresh=sig_thresh)
        df = df.loc[(df["call_self"] != "=") & (df["call_comparator"] != "=")]
        df = df.assign(
            log2_fold_change_self=df["log2_fold_change_self"].clip(-max_abs_log2_fold_change, max_abs_log2_fold_change),
            log2_fold_change_comparator=df["log2_fold_change_comparator"].clip(-max_abs_log2_fold_change, max_abs_log2_fold_change),
        )
        fig = self._organism_scatter(
            df,
            x="log2_fold_change_self",
            y="log2_fold_change_comparator",
            marker=dict(size=8, opacity=0.85),
            labels={
                **ASSOCIATION_LABELS,
                "log2_fold_change_self": f"{LOG2_FOLD_CHANGE_LABEL}<br>{self_label}",
                "log2_fold_change_comparator": f"{LOG2_FOLD_CHANGE_LABEL}<br>{comparator_label}",
                "pvalue_self": f"p-value ({self_label})",
                "pvalue_comparator": f"p-value ({comparator_label})",
                "qvalue_self": f"q-value ({self_label})",
                "qvalue_comparator": f"q-value ({comparator_label})",
            },
            hover_data=["Estimate_self", "Estimate_comparator", "pvalue_self", "pvalue_comparator", "qvalue_self", "qvalue_comparator"],
            width=width,
            height=height,
        )
        mark_beyond_axis(fig, max_abs_log2_fold_change, "x")
        mark_beyond_axis(fig, max_abs_log2_fold_change, "y")
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
        max_abs_log2_fold_change: float = 4.0,
        show_low_abundance: bool = False,
        width: int = 720,
        height: int = 720,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        The volcano plot of each contrast set as a margin of the fold-change
        scatter of the bins called in both. ``show_low_abundance`` chooses
        the version of the two volcanoes (see ``volcano_figure``).

        The comparator's volcano is drawn on its side at the upper left, so
        that its fold-change axis runs alongside the scatter's vertical axis;
        this set's volcano sits below the scatter, sharing its horizontal
        axis. The legend takes the otherwise empty lower-left quadrant.
        """
        sig_label = "p-value" if show_low_abundance else "q-value"
        fig = make_subplots(
            rows=2,
            cols=2,
            horizontal_spacing=0.10,
            vertical_spacing=0.10,
            subplot_titles=(comparator_label, "Called in both", None, self_label),
        )
        volcano = dict(fdr_thresh=sig_thresh, max_abs_log2_fold_change=max_abs_log2_fold_change, show_low_abundance=show_low_abundance)
        # The legend is drawn from this set's volcano, which shows every
        # organism; the other panels repeat the same names
        for trace in self.volcano_plot(**volcano).data:
            fig.add_trace(trace.update(showlegend=trace.meta != _BEYOND_META, legendgroup=trace.name), row=2, col=2)
        for trace in comparator.volcano_plot(**volcano).data:
            fig.add_trace(trace.update(x=trace.y, y=trace.x, showlegend=False, legendgroup=trace.name), row=1, col=1)
        # The comparator's volcano lies on its side, so its triangles turn with it
        for trace in fig.select_traces(row=1, col=1):
            if trace.marker.symbol is not None and not isinstance(trace.marker.symbol, str):
                trace.marker.symbol = [
                    {"triangle-left": "triangle-down", "triangle-right": "triangle-up"}.get(s, s) for s in trace.marker.symbol
                ]
        for trace in self.compare_association_scatter(
            comparator=comparator, fdr=fdr, sig_thresh=sig_thresh, max_abs_log2_fold_change=max_abs_log2_fold_change,
        ).data:
            fig.add_trace(trace.update(showlegend=False, legendgroup=trace.name), row=1, col=2)

        # Linked rather than shared, so every panel keeps its own tick labels
        fig.update_yaxes(matches="y", row=1, col=2)
        fig.update_xaxes(matches="x4", row=1, col=2)

        make_lines(0, fig, vline=False, row=1, col=1, **ZERO_LINE)
        make_lines(0, fig, row=1, col=2, **ZERO_LINE)
        make_lines(0, fig, hline=False, row=2, col=2, **ZERO_LINE)
        if not show_low_abundance:
            make_lines(MIN_ABS_LOG2_FOLD_CHANGE, fig, vline=False, row=1, col=1, **THRESHOLD_LINE)
            make_lines(-np.log10(sig_thresh), fig, hline=False, neg=False, row=1, col=1, **THRESHOLD_LINE)
            make_lines(MIN_ABS_LOG2_FOLD_CHANGE, fig, hline=False, row=2, col=2, **THRESHOLD_LINE)
            make_lines(-np.log10(sig_thresh), fig, vline=False, neg=False, row=2, col=2, **THRESHOLD_LINE)

        fig.update_layout(
            width=width,
            height=height,
            template=TEMPLATE,
            # Rows as compact as the single volcano's, which make_subplots does
            # not carry over, so that every organism fits the empty quadrant
            legend=dict(
                title_text="Organism", x=0.0, y=0.42, xanchor="left", yanchor="top",
                tracegroupgap=0, itemsizing="constant",
            ),
            margin=dict(t=50),
        )
        # The scatter shares the comparator volcano's vertical axis, so only
        # the volcano names it
        fig.update_xaxes(title_text=f"-log10({sig_label})", row=1, col=1)
        fig.update_yaxes(title_text=f"log<sub>2</sub>(cases / controls), {comparator_label}", row=1, col=1)
        fig.update_yaxes(title_text=None, row=1, col=2)
        fig.update_xaxes(title_text=f"log<sub>2</sub>(cases / controls), {self_label}", row=2, col=2)
        fig.update_yaxes(title_text=f"-log10({sig_label})", row=2, col=2)
        save_image(fig, file_prefix)
        return fig


#: The abundance heatmap's sample panel and the gap above the correlation
#: panel, in px, and the width of one character of a bin label at 13 px
ABUNDANCE_PANEL_PX = 300
HEATMAP_GAP_PX = 30
LABEL_CHAR_PX = 6


#: Below this expected count in any cell, the chi-squared distribution is a
#: poor guide and the p-value comes from random tables with the same margins
MIN_EXPECTED = 5
MONTE_CARLO_TABLES = 10_000


def _chi2_statistic(table: np.ndarray) -> float:
    """Pearson's chi-squared statistic of a contingency table with no empty margin."""
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    return float(((table - expected) ** 2 / expected).sum())


#: How each way an analysis can call a bin is drawn
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
