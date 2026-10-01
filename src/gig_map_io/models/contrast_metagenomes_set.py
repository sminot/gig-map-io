from functools import cached_property
from logging import getLogger
import logging
from scipy import stats
from scipy.cluster import hierarchy
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
from gig_map_io.helpers.observed_expected import plot_observed_expected
from gig_map_io.helpers.style import (
    DENSE_MARKER_OPACITY, QUALITATIVE, TEMPLATE, THRESHOLD_LINE, ZERO_LINE,
    group_colors, organism_colors, organism_order,
)
from .contrast_metagenomes import ContrastMetagenomes
from .dataset_dict import DatasetDict
from .sample_group import _key as _label_key

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
    def n_samples(self) -> int:
        """
        Number of samples in the contrast set.
        """
        return self.metadata.shape[0]

    @cached_property
    def metadata(self) -> pd.DataFrame:
        # To merge the metadata from all contrasts, melt the wide form
        # of the metadata into a long form, and then merge the long form
        # with the metadata from all contrasts.
        # Check for any values that differ between contrasts, and log a warning.
        # If any values differ, log a warning.
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
        # Make a wide table
        df = (
            long
            .groupby(["index", "variable"])
            .head(1)
            .pivot(index="index", columns="variable", values="value")
        )
        return df

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
        width: int = 650,
        height: int = 450,
        file_prefix: str | None = None,
        xlabel: str = "Effect size",
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
            color_discrete_map=organism_colors(self.pangenome_names),
            category_orders={"pangenome": organism_order(self.pangenome_names)},
            template=TEMPLATE,
            labels=dict(
                Estimate_clipped="Effect size (clipped)",
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
            ),
            hover_data=["mean_abund", "Estimate", "signed_log10_qvalue", "signed_log10_pvalue", "pvalue", "qvalue"],
            width=width,
            height=height,
            **kwargs
        )
        fig.update_traces(marker=dict(size=6, opacity=DENSE_MARKER_OPACITY, line_width=0))
        make_lines(0, fig, **ZERO_LINE)
        make_lines(estimate_thresh, fig, hline=False, **THRESHOLD_LINE)
        make_lines(-np.log10(fdr_thresh), fig, vline=False, neg=False, **THRESHOLD_LINE)

        # Specify the x-axis title
        if not transpose:
            fig.update_xaxes(title_text=xlabel)
        else:
            fig.update_yaxes(title_text=xlabel)

        # If save_image was provided, use the string as the file
        # prefix to write out HTML, PDF, PNG, and JSON
        save_image(fig, file_prefix)

        return fig

    def bin_abundance_heatmap(
        self,
        features: pd.MultiIndex,
        annotation_cols: list[str] | dict[str, str] | None = None,
        annotation_labels: dict[str, dict] | None = None,
        annotation_orders: dict[str, list] | None = None,
        metadata: pd.DataFrame | None = None,
        log_transform: bool = True,
        width: int = 1000,
        height: int = 800,
        rpkm_height_fraction: float = 0.65,
        annotation_width_fraction: float = 0.15,
        rpkm_colorscale: str = "Blues",
        corr_colorscale: str = "RdBu_r",
        annotation_colorscale: str = "Viridis",
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Multi-panel heatmap figure showing bin abundance patterns across samples.

        Panels:
          a) Heatmap of RPKM values (samples × features)
          b) Heatmap of pairwise Spearman correlation coefficients (features × features)
          c) [Optional] Heatmap of sample-level annotations (aligned with sample rows)

        Parameters
        ----------
        features : pd.MultiIndex
            MultiIndex with level names 'pangenome' and 'feature'. Each entry
            references a bin from the corresponding ContrastMetagenomes object.
        annotation_cols : list of str or dict, optional
            Columns from self.metadata (or the provided `metadata`) to display as
            sample annotations. If a dict, keys are original column names and values
            are display labels. If None, no annotation heatmap is shown.
        annotation_labels : dict, optional
            Display names for the values of an annotation column, keyed by
            the column's display label: ``{"GvHD": {"1": "Case", "0": "Control"}}``.
        annotation_orders : dict, optional
            Order of the (display) values of an annotation column, keyed the
            same way. A two-level column is coloured as a contrast, with the
            first level the case-like one.
        metadata : pd.DataFrame, optional
            Sample-level metadata. Overrides self.metadata when provided. Index
            must match specimen names; each column becomes a column in the annotation
            heatmap (shown to the left of the RPKM heatmap). Categorical/string
            columns are encoded as integer codes.
        log_transform : bool
            If True (default), display log₁₀(RPKM + 1) in the RPKM heatmap.
        width : int
            Figure width in pixels.
        height : int
            Figure height in pixels.
        rpkm_height_fraction : float
            Fraction of the figure height devoted to the RPKM heatmap (default 0.65).
            The Spearman correlation heatmap takes the remaining fraction.
        annotation_width_fraction : float
            Fraction of the figure width devoted to the annotation heatmap (default 0.15).
            The RPKM heatmap takes the remaining fraction. Ignored when no annotations
            are shown.
        rpkm_colorscale : str
            Plotly colorscale name for the RPKM heatmap.
        corr_colorscale : str
            Plotly colorscale name for the Spearman correlation heatmap.
        annotation_colorscale : str
            Plotly colorscale name for the annotation heatmap.
        file_prefix : str, optional
            If provided, save the figure as HTML/PDF/PNG/JSON with this prefix.

        Returns
        -------
        go.Figure
        """

        # ── 1. Build the combined RPKM DataFrame ─────────────────────────────
        rpkm_dict = {}
        for pangenome, feature in features:
            if pangenome not in self.contrast_metagenomes:
                logger.warning(f"Pangenome '{pangenome}' not found, skipping")
                continue
            cm = self.contrast_metagenomes[pangenome]
            if feature not in cm.rpkm.columns:
                logger.warning(
                    f"Feature '{feature}' not found in pangenome '{pangenome}', skipping"
                )
                continue
            label = f"{pangenome} / {feature}"
            rpkm_dict[label] = cm.rpkm[feature]

        if not rpkm_dict:
            raise ValueError("No valid features found in the provided index")

        rpkm = pd.DataFrame(rpkm_dict)

        # ── 2. Sort rows (samples) and columns (features) by hierarchical clustering
        def _hclust_order(df: pd.DataFrame) -> list:
            """Return leaf order from hierarchical clustering; NaN filled with 0."""
            if df.shape[0] < 2:
                return list(range(df.shape[0]))
            lnk = hierarchy.linkage(df.fillna(0).values, method="average", metric="euclidean")
            return hierarchy.leaves_list(lnk).tolist()

        rpkm = rpkm.iloc[_hclust_order(rpkm)]
        rpkm = rpkm.T.iloc[_hclust_order(rpkm.T)].T

        sorted_samples = rpkm.index.tolist()
        sorted_features = rpkm.columns.tolist()

        # ── 3. Optional log transform for display ─────────────────────────────
        rpkm_display = np.log10(rpkm + 1) if log_transform else rpkm.copy()
        rpkm_label = "log₁₀(RPKM+1)" if log_transform else "RPKM"

        # ── 4. Spearman correlation between features (pairwise, ignores NaN) ──
        corr_sorted = rpkm.corr(method="spearman").loc[sorted_features, sorted_features]

        # ── 5. Encode annotations for display ────────────────────────────────
        if annotation_cols is not None:
            if metadata is None:
                metadata = self.metadata
            if isinstance(annotation_cols, dict):
                metadata = metadata.reindex(columns=list(annotation_cols.keys())).rename(columns=annotation_cols)
            else:
                metadata = metadata.reindex(columns=annotation_cols)
            for col, labels in (annotation_labels or {}).items():
                labels = {_label_key(k): v for k, v in labels.items()}
                metadata[col] = metadata[col].map(
                    lambda v: labels.get(_label_key(v), v) if pd.notnull(v) else v
                )
        else:
            metadata = None
        has_annotations = metadata is not None and not metadata.empty
        annotation_orders = annotation_orders or {}

        # For each annotation column, determine if it is categorical (< 12 unique values).
        # ann_cat_info maps col -> (ordered unique values, list of hex colors).
        ann_cat_info: dict[str, tuple[list, list]] = {}
        if has_annotations:
            ann = metadata.reindex(index=sorted_samples)
            for col in ann.columns:
                s = ann[col]
                n_unique = s.nunique(dropna=True)
                if (not pd.api.types.is_numeric_dtype(s)) or n_unique < 12:
                    unique_vals = sorted(s.dropna().unique().tolist(), key=str)
                    if col in annotation_orders:
                        unique_vals = [v for v in annotation_orders[col] if v in unique_vals]
                        palette = group_colors(annotation_orders[col])
                        colors = [palette[v] for v in unique_vals]
                    else:
                        colors = QUALITATIVE[:len(unique_vals)]
                    ann_cat_info[col] = (unique_vals, colors)

        # ── 6. Build subplot layout ───────────────────────────────────────────
        #  Layout (with annotations):
        #    Row 1 (tall):   [annot hm  | RPKM heatmap  ]  ← shared y
        #    Row 2 (medium): [empty     | corr heatmap   ]
        #
        #  Column 2 subplots share x-axis (features aligned top-to-bottom).
        #  Row 1 subplots share y-axis (samples aligned left-to-right).
        #
        #  Without annotations: single-column, 2 rows.

        row_heights = [rpkm_height_fraction, 1.0 - rpkm_height_fraction]

        if has_annotations:
            specs = [
                [{},  {}],
                [None, {}],
            ]
            col_widths = [annotation_width_fraction, 1.0 - annotation_width_fraction]
            rpkm_col = 2
        else:
            specs = [[{}], [{}]]
            col_widths = None
            rpkm_col = 1

        n_cols = 2 if has_annotations else 1

        fig = make_subplots(
            rows=2,
            cols=n_cols,
            specs=specs,
            shared_xaxes=True,
            shared_yaxes=has_annotations,
            column_widths=col_widths,
            row_heights=row_heights,
            horizontal_spacing=0.02,
            vertical_spacing=0.04,
        )

        # ── 7. Add traces ─────────────────────────────────────────────────────

        # a) RPKM heatmap
        fig.add_trace(
            go.Heatmap(
                z=rpkm_display.values,
                x=sorted_features,
                y=sorted_samples,
                colorscale=rpkm_colorscale,
                name="RPKM",
                colorbar=dict(
                    title=rpkm_label,
                    len=row_heights[0],
                    yanchor="top",
                    y=1.0,
                    x=1.02,
                    thickness=14,
                ),
            ),
            row=1, col=rpkm_col,
        )

        # b) Spearman correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=corr_sorted.values,
                x=sorted_features,
                y=sorted_features,
                colorscale=corr_colorscale,
                zmid=0,
                name="Spearman r",
                colorbar=dict(
                    title="Spearman r",
                    len=row_heights[1],
                    yanchor="bottom",
                    y=0.0,
                    x=1.02,
                    thickness=14,
                ),
            ),
            row=2, col=rpkm_col,
        )

        # c) Sample annotations heatmap — one trace per column, using integer x
        #    positions to avoid Plotly's categorical axis padding gap.
        if has_annotations:
            for col_idx, col in enumerate(ann.columns):
                s = ann[col]
                if col in ann_cat_info:
                    unique_vals, colors = ann_cat_info[col]
                    code_map = {v: i for i, v in enumerate(unique_vals)}
                    n = len(unique_vals)
                    z_col = [[code_map.get(v, np.nan)] for v in s]
                    # Stepped discrete colorscale: each band covers [j/n, (j+1)/n].
                    # Using zmin=-0.5, zmax=n-0.5 centers each code within its band.
                    colorscale = []
                    for j, color in enumerate(colors):
                        colorscale.append([j / n, color])
                        colorscale.append([(j + 1) / n, color])
                    fig.add_trace(
                        go.Heatmap(
                            z=z_col,
                            x=[col_idx],
                            y=sorted_samples,
                            colorscale=colorscale,
                            zmin=-0.5,
                            zmax=n - 0.5,
                            showscale=False,
                            name=col,
                        ),
                        row=1, col=1,
                    )
                    # Invisible scatter traces to drive the color legend
                    for val, color in zip(unique_vals, colors):
                        fig.add_trace(
                            go.Scatter(
                                x=[None],
                                y=[None],
                                mode="markers",
                                marker=dict(color=color, symbol="square", size=10),
                                name=str(val),
                                legendgroup=col,
                                legendgrouptitle=dict(text=col),
                                showlegend=True,
                            ),
                            row=1, col=1,
                        )
                else:
                    fig.add_trace(
                        go.Heatmap(
                            z=[[v] for v in s],
                            x=[col_idx],
                            y=sorted_samples,
                            colorscale=annotation_colorscale,
                            showscale=False,
                            name=col,
                        ),
                        row=1, col=1,
                    )

        # ── 8. Update layout ──────────────────────────────────────────────────
        has_cat_annotations = has_annotations and bool(ann_cat_info)
        # The categorical legend runs along the top, leaving the right margin
        # to the two colour bars
        fig.update_layout(
            width=width,
            height=height,
            template=TEMPLATE,
            showlegend=has_cat_annotations,
            legend=dict(
                orientation="h", x=0.0, xanchor="left", y=1.0, yanchor="bottom",
            ),
            margin=dict(t=50),
        )

        fig.update_yaxes(title_text="Samples", row=1, col=1 if has_annotations else rpkm_col)
        fig.update_yaxes(title_text="Pangenome bin", row=2, col=rpkm_col)
        fig.update_xaxes(title_text="Pangenome bin", row=2, col=rpkm_col)

        # Hide x-tick labels on the top row (shared axis shows them at bottom)
        fig.update_xaxes(showticklabels=False, row=1, col=rpkm_col)

        # Hide y-tick labels on the RPKM, Spearman, and annotation heatmaps
        fig.update_yaxes(showticklabels=False, row=1, col=rpkm_col)
        fig.update_yaxes(showticklabels=False, row=2, col=rpkm_col)
        if has_annotations:
            fig.update_yaxes(showticklabels=False, row=1, col=1)
            # Use integer x positions with named ticks and a tight range so that
            # columns sit flush against each other with no categorical padding.
            n_ann_cols = len(ann.columns)
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(n_ann_cols)),
                ticktext=ann.columns.tolist(),
                tickangle=90,
                range=[-0.5, n_ann_cols - 0.5],
                row=1, col=1,
            )

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

    def calc_auc(
        self,
        pangenome_name: str,
        metadata_col: str,
        ref_group,
        comp_group,
        bin_id: str,
        query_str=None,
        samples: pd.Index | None = None
    ):
        """
        For an organism, calculate the AUC for one bin with respect to a particular metadata column.
        The user specifies a reference group and comparison group, both of which must be
        values present in the metadata column.
        """
        contrast: ContrastMetagenomes = self[pangenome_name]
        return contrast.calc_auc(
            metadata_col=metadata_col,
            ref_group=ref_group,
            comp_group=comp_group,
            bin_id=bin_id,
            query_str=query_str,
            samples=samples
        )

    def calc_odds_ratio(
        self,
        pangenome_name: str,
        metadata_col: str,
        ref_group,
        comp_group,
        bin_id: str,
        query_str=None,
        samples: pd.Index | None = None,
        threshold="median"
    ):
        """
        For an organism, calculate the odds ratio for one bin with respect to a particular metadata column.
        The user specifies a reference group and comparison group, both of which must be
        values present in the metadata column.
        The threshold can be set as the "median", "mean", or with a specific RPKM value.
        """
        contrast: ContrastMetagenomes = self[pangenome_name]
        return contrast.calc_odds_ratio(
            metadata_col=metadata_col,
            ref_group=ref_group,
            comp_group=comp_group,
            bin_id=bin_id,
            query_str=query_str,
            samples=samples,
            threshold=threshold
        )

    def compare_sig_categories(
        self,
        comparitor: 'ContrastMetagenomesSet',
        fdr: bool = True,
        sig_thresh: float = 0.2,
        estimate_thresh: float = 0.25,
        self_label: str = "self",
        comparitor_label: str = "comparitor",
        width: int = 800,
        height: int = 500,
        file_prefix: str | None = None,
        **kwargs
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
        df = (
            self.compare_association(comparitor)
            .pipe(lambda d: _add_sig_categories(d, fdr, sig_thresh, estimate_thresh))
        )

        observed = df.pivot_table(
            columns="self_sig",
            index="comparitor_sig",
            values="feature",
            aggfunc="count",
            fill_value=0,
        ).reindex(index=SIG_LEVELS, columns=SIG_LEVELS, fill_value=0)

        chi2, pvalue, dof, _ = stats.chi2_contingency(observed)

        # Computed from the margins rather than taken from chi2_contingency,
        # which drops any row or column that is entirely empty and would then
        # not line up with the nine bars
        total = observed.values.sum()
        expected = pd.DataFrame(
            np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total,
            index=observed.index,
            columns=observed.columns,
        )

        return _plot_sig_categories(
            observed=observed,
            expected=expected,
            pvalue=pvalue,
            self_label=self_label,
            comparitor_label=comparitor_label,
            width=width,
            height=height,
            file_prefix=file_prefix,
        )

    def compare_sig_scatter(
        self,
        comparitor: 'ContrastMetagenomesSet',
        self_label: str = "self",
        comparitor_label: str = "comparitor",
        fdr: bool = True,
        sig_thresh: float = 0.2,
        width: int = 650,
        height: int = 450,
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
            color_discrete_map=organism_colors(self.pangenome_names),
            category_orders={"pangenome": organism_order(self.pangenome_names)},
            hover_name="hover_name",
            template=TEMPLATE,
            labels={
                f"{value_col}_self": f"{value_label}<br>{self_label}",
                f"{value_col}_comparitor": f"{value_label}<br>{comparitor_label}",
                "pvalue_self": f"p-value ({self_label})",
                "pvalue_comparitor": f"p-value ({comparitor_label})",
                "feature": "Pangenome bin",
                "pangenome": "Organism",
            },
            hover_data=[f"{value_col}_self", f"{value_col}_comparitor", "pvalue_self", "pvalue_comparitor"],
            width=width,
            height=height,
            **kwargs
        )
        fig.update_traces(marker=dict(size=6, opacity=DENSE_MARKER_OPACITY, line_width=0))

        make_lines(0, fig, **ZERO_LINE)
        make_lines(-np.log10(sig_thresh), fig, **THRESHOLD_LINE)

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
        width: int = 650,
        height: int = 450,
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
            color_discrete_map=organism_colors(self.pangenome_names),
            category_orders={"pangenome": organism_order(self.pangenome_names)},
            hover_name="hover_name",
            template=TEMPLATE,
            labels={
                "Estimate_self": f"Effect size<br>{self_label}",
                "Estimate_comparitor": f"Effect size<br>{comparitor_label}",
                "pvalue_self": f"p-value ({self_label})",
                "pvalue_comparitor": f"p-value ({comparitor_label})",
                "qvalue_self": f"q-value ({self_label})",
                "qvalue_comparitor": f"q-value ({comparitor_label})",
                "pangenome": "Organism",
                "feature": "Pangenome bin",
            },
            hover_data=["Estimate_self", "Estimate_comparitor", "pvalue_self", "pvalue_comparitor", "qvalue_self", "qvalue_comparitor"],
            width=width,
            height=height,
            **kwargs
        )
        fig.update_traces(marker=dict(size=8, opacity=0.85, line_width=0))

        make_lines(0, fig, **ZERO_LINE)

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
        width: int = 720,
        height: int = 720,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        The volcano plot of each contrast set as a margin of the effect-size
        scatter of the bins significant in both.

        The comparitor's volcano is drawn on its side at the upper left, so
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
            subplot_titles=(comparitor_label, "Significant in both", None, self_label),
        )
        # The legend is drawn from this set's volcano, which shows every
        # organism; the other panels repeat the same names
        for trace in self.volcano_plot(
            estimate_thresh=estimate_thresh,
            fdr_thresh=sig_thresh,
            max_abs_estimate=max_abs_estimate,
        ).data:
            fig.add_trace(trace.update(showlegend=True, legendgroup=trace.name), row=2, col=2)
        for trace in comparitor.volcano_plot(
            estimate_thresh=estimate_thresh,
            fdr_thresh=sig_thresh,
            max_abs_estimate=max_abs_estimate,
            transpose=True,
        ).data:
            fig.add_trace(trace.update(showlegend=False, legendgroup=trace.name), row=1, col=1)
        for trace in self.compare_association_scatter(
            comparitor=comparitor,
            fdr=fdr,
            sig_thresh=sig_thresh,
            estimate_thresh=estimate_thresh,
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
            legend=dict(
                title_text="Organism", x=0.0, y=0.42, xanchor="left", yanchor="top",
            ),
            margin=dict(t=50),
        )
        fig.update_xaxes(title_text=f"-log10({sig_label})", row=1, col=1)
        fig.update_yaxes(title_text=f"Effect size, {comparitor_label}", row=1, col=1)
        fig.update_yaxes(title_text=f"Effect size, {comparitor_label}", row=1, col=2)
        fig.update_xaxes(title_text=f"Effect size, {self_label}", row=2, col=2)
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


#: The three ways an analysis can call a bin, and how each is drawn
SIG_LEVELS = ["<", "=", ">"]
_SIG_ARROW = {"<": "&#8595;", ">": "&#8593;", "=": "ns"}

#: The nine categories, ordered by meaning: (self, comparitor, group)
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


def _plot_sig_categories(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    pvalue: float,
    self_label: str,
    comparitor_label: str,
    width: int,
    height: int,
    file_prefix: str | None,
) -> go.Figure:
    """Draw the nine categories as observed-against-expected bars."""
    ticks, groups, obs, exp = [], [], [], []
    for self_sig, comparitor_sig, group in _SIG_CATEGORY_ORDER:
        ticks.append(f"{_SIG_ARROW[self_sig]}<br>{_SIG_ARROW[comparitor_sig]}")
        groups.append(group)
        obs.append(observed.loc[comparitor_sig, self_sig])
        exp.append(expected.loc[comparitor_sig, self_sig])

    return plot_observed_expected(
        ticks=ticks,
        groups=groups,
        observed=obs,
        expected=exp,
        title=f"Do {self_label} and {comparitor_label} flag the same bins?",
        subtitle=f"chi-squared p = {format_pvalue(pvalue)}",
        y_title="Pangenome bins",
        caption=f"Each tick: direction in <b>{self_label}</b> (upper) "
                f"and <b>{comparitor_label}</b> (lower)",
        width=width,
        height=height,
        file_prefix=file_prefix,
    )
