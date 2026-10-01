"""
StudySet: analyses that span more than one study.

Where a :class:`~gig_map_io.models.study.Study` covers one contrast, a
``StudySet`` stacks the samples of several studies into a single table so that
community-level structure can be compared across them.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go

from ..helpers.clustering import leiden
from ..helpers.contingency import chi2_contingency_test
from ..helpers.enrichment import enrich_organisms, plot_enrichment
from ..helpers.ordination import tsne
from ..helpers.permanova import permanova
from ..helpers.positivity import plot_feature_positivity, plot_positivity_heatmap
from ..helpers.save_image import save_image
from ..helpers.style import (
    DENSE_MARKER_OPACITY, ESTIMATE_THRESH, FDR_THRESH, LARGE_QUALITATIVE, PRIMARY, TEMPLATE,
    THRESHOLD_LINE, TOP_LEGEND, group_colors, legend_above, organism_colors, organism_order,
)
from ..helpers.supervised import fit_classifier
from .study import Study

#: Sample groups every study in a set is expected to define.
REQUIRED_SAMPLE_GROUPS = ("study", "cohort", "disease", "participant")


class StudySet:
    """
    Several studies analyzed together over a shared sample-by-bin table.

    Parameters
    ----------
    studies:
        The studies to combine. Sample indices must be disjoint.
    """

    def __init__(self, studies: Iterable[Study]) -> None:
        self.studies = list(studies)
        if not self.studies:
            raise ValueError("StudySet requires at least one study")

    @classmethod
    def from_json(cls, *paths: str | Path, datasets: str | Path = "datasets") -> "StudySet":
        """Read several study definitions from JSON files."""
        return cls([Study.from_json(path, datasets=datasets) for path in paths])

    def __repr__(self) -> str:
        return f"StudySet(studies={[study.name for study in self.studies]})"

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, name: str) -> Study:
        for study in self.studies:
            if study.name == name:
                return study
        raise KeyError(f"No study named {name!r} in this set")


    @cached_property
    def rpkm(self) -> pd.DataFrame:
        """Bin RPKM for every sample, columns keyed by (organism, bin)."""
        return pd.concat([study.contrasts.rpkm for study in self.studies]).fillna(0)

    @cached_property
    def bin_size(self) -> pd.Series:
        """Number of genes in each bin of the combined table."""
        pangenomes = {}
        for study in self.studies:
            if study.pangenome_dirs:
                pangenomes.update(study.pangenomes.pangenomes)
        if not pangenomes:
            raise ValueError("No study in this set defines pangenomes")
        return pd.Series(
            [pangenomes[organism].bin_size[bin_id] for organism, bin_id in self.rpkm.columns],
            index=self.rpkm.columns,
        )

    @cached_property
    def weighted_rpkm(self) -> pd.DataFrame:
        """Bin RPKM scaled by the number of genes in each bin."""
        return self.rpkm * self.bin_size

    @cached_property
    def sample_metadata(self) -> pd.DataFrame:
        """
        Sample annotations for every study, one column per shared sample
        group.
        """
        return pd.concat(
            [study.sample_metadata(REQUIRED_SAMPLE_GROUPS) for study in self.studies]
        )

    def _order(self, group: str) -> List[str]:
        order: List[str] = []
        for study in self.studies:
            for name in study.group_order(group):
                if name not in order:
                    order.append(name)
        return order

    @cached_property
    def study_order(self) -> List[str]:
        """Study names in the order declared by the studies."""
        return self._order("study")

    @cached_property
    def cohort_order(self) -> List[str]:
        """Cohort names in the order declared by the studies."""
        return self._order("cohort")

    @cached_property
    def disease_order(self) -> List[str]:
        """Disease states in the order declared by the studies, the case-like one first."""
        return self._order("disease")

    def features(self, features: pd.MultiIndex | pd.DataFrame | None = None) -> pd.DataFrame:
        """The combined RPKM table, optionally restricted to a subset of bins."""
        if features is None:
            return self.rpkm
        index = features.index if isinstance(features, pd.DataFrame) else features
        return self.rpkm.reindex(columns=index)


    def significant_bins(
        self,
        estimate_thresh: float = ESTIMATE_THRESH,
        fdr_thresh: float = FDR_THRESH,
    ) -> pd.DataFrame:
        """
        Bins passing the significance thresholds in *every* study in the set.

        Association columns are suffixed with each study's name.
        """
        merged: pd.DataFrame | None = None
        for study in self.studies:
            df = study.significant_bins(estimate_thresh, fdr_thresh).add_suffix(f"_{study.name}")
            merged = df if merged is None else merged.join(df, how="inner")
        return merged


    def tsne_plot(
        self,
        color: str,
        features: pd.MultiIndex | pd.DataFrame | None = None,
        width: int = 600,
        height: int = 450,
        show_legend: bool = True,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        t-SNE ordination of the combined samples, colored by a sample group.

        The levels of a group are colored in the order the studies declare
        them, so that a group shared between figures is colored alike; a
        group with more levels than the palette (participants) is not given
        a legend.
        """
        coords = tsne(self.features(features)).merge(
            self.sample_metadata, left_index=True, right_index=True
        )
        order = self._order(color)
        fig = px.scatter(
            data_frame=coords,
            x="t-SNE 1",
            y="t-SNE 2",
            color=color,
            color_discrete_map=group_colors(order) if order else None,
            color_discrete_sequence=None if order else LARGE_QUALITATIVE,
            category_orders={color: order} if order else None,
            template=TEMPLATE,
            width=width,
            height=height,
            labels={"study": "Study", "disease": "Disease", "participant": "Participant"},
        )
        fig.update_traces(marker=dict(size=6, opacity=DENSE_MARKER_OPACITY, line_width=0))
        fig.update_xaxes(showticklabels=False, ticks="")
        fig.update_yaxes(showticklabels=False, ticks="")
        fig.update_layout(showlegend=show_legend)
        save_image(fig, file_prefix)
        return fig

    def permanova(
        self,
        features: pd.MultiIndex | pd.DataFrame | None = None,
        categories: Iterable[str] = ("disease", "participant"),
        file_prefix: str | None = None,
    ) -> pd.DataFrame:
        """
        Run PERMANOVA separately within each study, for each sample group.

        Testing within study rather than across all samples keeps differences
        between studies from being attributed to the variable of interest.

        Returns the long-form results (one row per study per category). When
        ``file_prefix`` is given, a wide table is also written as CSV.
        """
        rpkm = self.features(features)
        metadata = self.sample_metadata

        results = pd.concat([
            permanova(
                scalars_df=rpkm.reindex(index=study_metadata.index),
                metadata_df=study_metadata.reindex(columns=list(categories)),
            ).assign(study=study)
            for study, study_metadata in metadata.groupby("study")
        ])

        if file_prefix is not None:
            path = Path(file_prefix + ".csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._permanova_wide(results).to_csv(path)

        return results

    def _permanova_wide(self, results: pd.DataFrame) -> pd.DataFrame:
        renamed = {
            "r_squared": "R^2",
            "p_value": "p-value",
            "f_statistic": "F statistic",
            "n_groups": "# Groups",
            "n_samples": "# Samples",
        }
        # The sample count is the same for every category within a study, so it
        # belongs on the index rather than in the body. Whether pandas drops an
        # index level that also appears in `values` has changed between
        # versions, so it is excluded here rather than left to that.
        index = ["Study", "# Samples"]
        wide = (
            results
            .assign(category=results["category"].str.title())
            .rename(columns=lambda v: renamed.get(v, v.title()))
            .pivot_table(
                columns="Category",
                index=index,
                values=[name for name in renamed.values() if name not in index],
            )
        )
        wide.columns = pd.MultiIndex.from_tuples(
            [(metric, category) for category, metric in wide.columns],
            names=["Metric", "Category"],
        )
        return (
            wide
            .sort_index(axis=1)
            .reset_index()
            .set_index("Study")
            .reindex(index=self.study_order)
            .reset_index()
            .set_index(["Study", "# Samples"])
        )

    def compare_permanova(
        self,
        results: Dict[str, pd.DataFrame],
        metric: str = "r_squared",
        width: int = 760,
        height: int = 360,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Bar chart comparing PERMANOVA results from several feature subsets.

        ``results`` maps a label (e.g. "All bins") to the long-form output of
        :meth:`permanova`. p-values are drawn on a log axis, since the
        interesting ones are the small ones, with the 0.05 line marked.
        """
        df = pd.concat([
            result.assign(Bins=label) for label, result in results.items()
        ])
        fig = px.bar(
            data_frame=df,
            x="study",
            y=metric,
            facet_col="category",
            color="Bins",
            barmode="group",
            template=TEMPLATE,
            facet_col_spacing=0.1,
            labels={
                "r_squared": "Variance explained (R&#178;)",
                "p_value": "p-value",
                "study": "Study",
                "disease": "Disease",
                "participant": "Participant",
            },
            width=width,
            height=height,
            category_orders={"study": self.study_order},
        )
        if metric == "p_value":
            fig.add_hline(y=0.05, **THRESHOLD_LINE)
            fig.update_yaxes(type="log", tickvals=[0.001, 0.01, 0.05, 0.1, 0.5, 1])
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_xaxes(title_text="")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].title()))
        fig.update_layout(margin=dict(t=90))
        legend_above(fig)
        save_image(fig, file_prefix)
        return fig


    def pangenome_clusters(
        self,
        organism: str,
        resolution: float = 1.0,
        metric: str = "braycurtis",
    ) -> pd.DataFrame:
        """
        Cluster the samples on one organism's bin composition.

        Samples with no detectable signal for the organism are dropped.
        Returns the t-SNE coordinates, the Leiden cluster label, and the
        sample metadata.
        """
        rpkm = self.weighted_rpkm[organism]
        rpkm = rpkm.loc[rpkm.max(axis=1) > 0]
        coords = tsne(rpkm).assign(cluster=leiden(rpkm, resolution=resolution, metric=metric))
        return pd.concat([coords, self.sample_metadata.reindex(index=rpkm.index)], axis=1)

    def cluster_disease_contingency(
        self,
        clusters: Dict[str, pd.DataFrame],
        width: int = 950,
        height: int = 460,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Strength of the association between community type and disease state,
        for each organism and study, as grouped bars of Cramér's V.

        One group of bars per organism, one bar per study, so that the
        cohorts can be read against each other within an organism and the
        same cohort followed across organisms by its color. The chi-squared
        significance is printed over each bar as stars.
        """
        rows = []
        for organism, df in clusters.items():
            for study, study_df in df.groupby("study"):
                result = chi2_contingency_test(df=study_df, col_a="cluster", col_b="disease")
                rows.append({
                    "organism": organism,
                    "study": study,
                    "cramers_v": result["cramers_v"],
                    "p_value": result["p_value"],
                    "stars": _significance_stars(result["p_value"]),
                })

        results = pd.DataFrame(rows)
        fig = self._organism_study_bars(
            results, y="cramers_v", study_order=self.study_order,
            labels={"cramers_v": "Cram&#233;r's V"},
            hover_data={"p_value": ":.2e", "cramers_v": ":.2f"},
            title="Association between community type and disease state",
            footnote="Chi-squared test: * p < 0.05, ** p < 0.01, *** p < 0.001",
            width=width, height=height, text="stars",
        )
        fig.update_traces(textposition="outside", textangle=-90, textfont_size=11, cliponaxis=False)
        fig.update_yaxes(range=[0, results["cramers_v"].max() * 1.2])
        save_image(fig, file_prefix)
        return fig

    def _organism_study_bars(
        self,
        summary: pd.DataFrame,
        y: str,
        study_order: List[str],
        labels: dict,
        hover_data: dict,
        title: str,
        width: int,
        height: int,
        footnote: str | None = None,
        **bar_kwargs,
    ) -> go.Figure:
        """
        One group of bars per organism, one bar per study, in the fixed
        organism order and study colors, with a legend above and an optional
        footnote below the tilted organism labels.
        """
        fig = px.bar(
            data_frame=summary,
            x="organism",
            y=y,
            color="study",
            barmode="group",
            template=TEMPLATE,
            color_discrete_map=group_colors(study_order),
            category_orders={"study": study_order, "organism": organism_order(summary["organism"])},
            labels={"organism": "Organism", "study": "Study", **labels},
            hover_data=hover_data,
            title=title,
            width=width,
            height=height,
            **bar_kwargs,
        )
        fig.update_xaxes(title_text="", tickangle=-25)
        fig.update_layout(
            bargroupgap=0.05,
            legend=TOP_LEGEND,
            margin=dict(t=95 if title else 60, b=130 if footnote else 80),
        )
        if footnote:
            fig.add_annotation(
                text=footnote, x=0.5, xref="paper", y=0, yref="paper", yshift=-105, showarrow=False,
                font=dict(size=11, color="#555555"),
            )
        return fig

    def cluster_tsne_plot(
        self,
        clusters: pd.DataFrame,
        width: int = 600,
        height: int | None = None,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        t-SNE ordination of one organism's samples, colored by community type.

        There can be more community types than the standard palette has
        colors, so a larger one is used here and nowhere else, and the
        height grows with the legend unless given.
        """
        order = _sorted_clusters(clusters["cluster"])
        if height is None:
            height = max(500, 20 * len(order) + 160)
        fig = px.scatter(
            data_frame=clusters,
            x="t-SNE 1",
            y="t-SNE 2",
            color="cluster",
            color_discrete_sequence=LARGE_QUALITATIVE,
            template=TEMPLATE,
            labels={"study": "Study", "disease": "Disease", "cluster": "Community type"},
            category_orders={"study": self.study_order, "cluster": order},
            width=width,
            height=height,
        )
        fig.update_traces(marker=dict(size=6, opacity=DENSE_MARKER_OPACITY, line_width=0))
        fig.update_xaxes(showticklabels=False, ticks="")
        fig.update_yaxes(showticklabels=False, ticks="")
        save_image(fig, file_prefix)
        return fig

    def cluster_composition_bars(
        self,
        clusters: pd.DataFrame,
        width: int = 650,
        height: int | None = None,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Percentage of each cohort's cases and controls falling in each
        community type, one row per cohort. The height follows the number of
        cohorts unless given.
        """
        composition = (
            clusters
            .assign(count=1)
            .pivot_table(columns=["study", "disease"], index="cluster", values="count", aggfunc="sum")
            .fillna(0)
            .apply(lambda c: c / c.sum())
            .multiply(100.0)
            .melt(ignore_index=False)
            .reset_index()
        )
        studies = [study for study in self.study_order if study in set(composition["study"])]
        fig = px.bar(
            data_frame=composition,
            x="cluster",
            y="value",
            color="disease",
            color_discrete_map=group_colors(self.disease_order),
            facet_col="study",
            facet_col_wrap=1,
            facet_row_spacing=0.06,
            barmode="group",
            template=TEMPLATE,
            category_orders={
                "study": studies,
                "cluster": _sorted_clusters(clusters["cluster"]),
                "disease": self.disease_order,
            },
            labels={"value": "% of samples", "cluster": "", "disease": "Disease"},
            width=width,
            height=height if height is not None else 120 * len(studies) + 130,
        )
        fig.update_yaxes(matches=None)
        fig.update_xaxes(tickangle=-90)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(margin=dict(t=80))
        legend_above(fig)
        save_image(fig, file_prefix)
        return fig


    def bin_proportion_boxplot(
        self,
        features: pd.MultiIndex | pd.DataFrame,
        floor: float = 0.001,
        width: int = 560,
        height: int | None = None,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Abundance of each bin relative to its organism's core genome, split by
        cohort and by case/control status, one row per bin.

        Dividing by the core genome turns RPKM into the proportion of that
        organism's genomes in the sample which carry the bin, which is
        comparable across cohorts of differing sequencing depth. The height
        follows the number of bins unless given.
        """
        index = features.index if isinstance(features, pd.DataFrame) else features

        panels = []
        for study in self.studies:
            core_genomes = study.core_genomes
            metadata = study.sample_metadata(["cohort", "disease"])
            for organism, bin_id in index:
                rpkm = study.contrast(organism).rpkm
                panels.append(pd.DataFrame({
                    "Cohort": metadata["cohort"],
                    "Group": metadata["disease"],
                    "Proportion": (rpkm[bin_id] / rpkm[core_genomes[organism]]).clip(lower=floor),
                    "Feature": f"{organism} - {bin_id}",
                }))

        n_bins = len(index)
        fig = px.box(
            data_frame=pd.concat(panels),
            x="Cohort",
            y="Proportion",
            color="Group",
            color_discrete_map=group_colors(self.disease_order),
            template=TEMPLATE,
            log_y=True,
            facet_col="Feature",
            facet_col_wrap=1,
            facet_row_spacing=min(0.05, 0.6 / max(n_bins, 1)),
            category_orders={"Cohort": self.cohort_order, "Group": self.disease_order},
            labels={"Proportion": "Proportion of genomes", "Group": ""},
            width=width,
            height=height if height is not None else 150 * n_bins + 160,
            range_y=(floor, 1),
        )
        fig.update_traces(marker=dict(size=3, opacity=0.6), line_width=1)
        # Only the decades are worth a tick on a three-decade axis
        decades = [10.0 ** i for i in range(int(np.log10(floor)), 1)]
        fig.update_yaxes(tickvals=decades, ticktext=[f"{d:g}" for d in decades])
        fig.update_xaxes(tickangle=-40)
        fig.for_each_annotation(
            lambda a: a.update(x=0.02, xanchor="left", text=a.text.split("=", 1)[1])
        )
        # One axis title for the column of facets, not one per row
        fig.update_yaxes(title_text="")
        fig.add_annotation(
            text="Proportion of genomes", x=0, xref="paper", y=0.5, yref="paper",
            xshift=-58, textangle=-90, showarrow=False, font=dict(size=14),
        )
        fig.update_layout(margin=dict(t=80))
        legend_above(fig)
        save_image(fig, file_prefix)
        return fig


    def plot_feature_positivity(
        self,
        features: pd.MultiIndex | pd.DataFrame | None = None,
        samples: pd.Index | None = None,
        group: str = "disease",
        **kwargs: Any,
    ) -> go.Figure:
        """Samples ranked by how many of the given bins they carry."""
        rpkm, metadata = self._subset(features, samples)
        kwargs.setdefault("group_order", self._order(group))
        return plot_feature_positivity(rpkm, metadata[group], **kwargs)

    def plot_positivity_heatmap(
        self,
        features: pd.MultiIndex | pd.DataFrame | None = None,
        samples: pd.Index | None = None,
        groups: Iterable[str] = ("disease",),
        **kwargs: Any,
    ) -> go.Figure:
        """Per-sample detection of each bin, annotated by sample group."""
        rpkm, metadata = self._subset(features, samples)
        kwargs.setdefault("group_orders", {group: self._order(group) for group in groups})
        return plot_positivity_heatmap(rpkm, metadata.reindex(columns=list(groups)), **kwargs)

    def _subset(
        self,
        features: pd.MultiIndex | pd.DataFrame | None,
        samples: pd.Index | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rpkm = self.features(features)
        metadata = self.sample_metadata
        if samples is not None:
            rpkm = rpkm.reindex(index=samples)
            metadata = metadata.reindex(index=samples)
        rpkm = rpkm.set_axis(rpkm.columns.map(" - ".join), axis=1)
        return rpkm, metadata



    def classify_by_organism(
        self,
        n_replicates: int = 10,
        n_interaction_features: int = 10,
    ) -> Dict[str, pd.DataFrame]:
        """
        For each organism, fit a classifier per study separating cases from
        controls on that organism's bin abundance.

        Returns three tables: ``replicates`` (validation ROC-AUC per split),
        ``shap`` (mean absolute SHAP per bin, one column per study, plus their
        geometric mean), and ``interactions`` (pairwise SHAP interaction
        strengths among each model's most important bins).
        """
        organisms = [
            organism for organism in self.studies[0].organisms
            if all(organism in study.contrast_dirs for study in self.studies)
        ]

        replicates, shap_panels, interactions = [], [], []
        for organism in organisms:
            per_study = {}
            for study in self.studies:
                contrast = study.contrast(organism)
                result = fit_classifier(
                    contrast.abund,
                    contrast.metadata[study.parameter],
                    n_replicates=n_replicates,
                    n_interaction_features=n_interaction_features,
                )
                if result is None:
                    continue
                per_study[study.label] = result.shap
                replicates.extend(
                    {
                        "organism": organism,
                        "study": study.label,
                        "replicate": replicate,
                        "roc_auc": auc,
                        "n_samples": result.n_samples,
                    }
                    for replicate, auc in enumerate(result.replicate_auc)
                )
                if result.interactions is not None:
                    interactions.append(
                        result.interactions
                        .rename_axis(index="bin_a", columns="bin_b")
                        .stack()
                        .rename("mean_abs_interaction")
                        .reset_index()
                        .assign(organism=organism, study=study.label)
                    )

            # A bin can only be compared across studies if every study modelled it
            if len(per_study) == len(self.studies):
                shap_panels.append(
                    pd.DataFrame(per_study).fillna(0.0).rename_axis(index="bin")
                    .assign(organism=organism)
                    .reset_index()
                )

        shap = pd.concat(shap_panels, ignore_index=True)
        labels = [study.label for study in self.studies]
        shap["combined"] = np.exp(np.log(shap[labels].clip(lower=1e-12)).mean(axis=1))

        return {
            "replicates": pd.DataFrame(replicates),
            "shap": shap.sort_values(
                ["combined", "organism", "bin"], ascending=[False, True, True]
            ),
            "interactions": pd.concat(interactions, ignore_index=True),
        }

    def plot_classifier_performance(
        self,
        replicates: pd.DataFrame,
        width: int = 1000,
        height: int = 520,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """ROC-AUC per organism and study, with the spread across replicates."""
        summary = (
            replicates
            .groupby(["organism", "study"])
            .agg(auc_mean=("roc_auc", "mean"), auc_std=("roc_auc", "std"))
            .reset_index()
        )
        fig = self._organism_study_bars(
            summary, y="auc_mean", study_order=[study.label for study in self.studies],
            labels={"auc_mean": "Validation ROC-AUC"}, hover_data={"auc_std": ":.3f"}, title="",
            width=width, height=height, error_y="auc_std",
        )
        fig.update_traces(error_y=dict(thickness=1, width=4))
        fig.add_hline(
            y=0.5, annotation_text="chance", annotation_position="bottom right",
            annotation_font_color="#555555", **THRESHOLD_LINE,
        )
        fig.update_yaxes(range=[0.4, 1.05])
        save_image(fig, file_prefix)
        return fig

    def plot_shap_comparison(
        self,
        shap: pd.DataFrame,
        organism: str,
        n_labelled: int = 5,
        width: int = 650,
        height: int = 560,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Importance of one organism's bins in each study's model, against each
        other. Bins near the diagonal carry signal in both.
        """
        df = shap.loc[shap["organism"] == organism].sort_values(["combined", "bin"], ascending=[False, True])
        fig, x_label, y_label, _ = self._shap_scatter(
            df, title=f"{organism}: bin importance in each study", width=width, height=height,
        )
        fig.update_traces(marker=dict(size=6, opacity=DENSE_MARKER_OPACITY, color=PRIMARY))
        top = df.head(n_labelled)
        fig.add_trace(go.Scatter(
            x=top[x_label], y=top[y_label], mode="text", text=top["bin"],
            textposition="top right", textfont=dict(size=11),
            showlegend=False, hoverinfo="skip",
        ))
        save_image(fig, file_prefix)
        return fig

    def _shap_scatter(self, df: pd.DataFrame, title: str, width: int, height: int, **scatter_kwargs):
        """
        Importance to the first study's model against importance to the
        second's, on equal axes with the diagonal drawn. Returns the figure,
        the two axis columns and the axis limit.
        """
        if len(self.studies) != 2:
            raise ValueError("SHAP comparisons need exactly two studies")
        x_label, y_label = (study.label for study in self.studies)
        limit = max(df[x_label].max(), df[y_label].max()) * 1.08
        fig = px.scatter(
            data_frame=df,
            x=x_label,
            y=y_label,
            hover_name="bin",
            template=TEMPLATE,
            labels={
                x_label: f"Mean |SHAP|, {x_label}",
                y_label: f"Mean |SHAP|, {y_label}",
                "organism": "Organism",
            },
            title=title,
            width=width,
            height=height,
            range_x=[0, limit],
            range_y=[0, limit],
            **scatter_kwargs,
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=limit, y1=limit,
                      line=dict(color="#b0b0b0", dash="dash", width=1))
        return fig, x_label, y_label, limit

    def plot_shap_comparison_all(
        self,
        shap: pd.DataFrame,
        n_labelled: int = 8,
        width: int = 720,
        height: int = 600,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        Every organism's bins on one plot: importance to the first study's
        model against importance to the second's, colored by organism, with
        the bins ranking highest on the combined importance named.
        """
        df = shap.sort_values(["combined", "organism", "bin"], ascending=[False, True, True])
        fig, x_label, y_label, limit = self._shap_scatter(
            df, title="Bin importance in each study, all organisms", width=width, height=height,
            color="organism",
            color_discrete_map=organism_colors(df["organism"]),
            category_orders={"organism": organism_order(df["organism"])},
            hover_data={"combined": ":.4f", "organism": True},
        )
        fig.update_traces(marker=dict(size=6, opacity=DENSE_MARKER_OPACITY, line_width=0))
        # The named bins crowd the lower-left corner, so each label is set
        # out to the right, spaced evenly down the empty side of the plot,
        # and tied back to its point with an arrow
        top = df.head(n_labelled).reset_index(drop=True)
        top = top.sort_values(y_label, ascending=False).reset_index(drop=True)
        for i, row in top.iterrows():
            fig.add_annotation(
                x=row[x_label], y=row[y_label],
                ax=limit * 0.55,
                ay=limit * (0.95 - 0.5 * i / max(len(top) - 1, 1)),
                axref="x", ayref="y",
                text=f"{row['organism']} {row['bin']}",
                showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor="#444444",
                font=dict(size=11), bgcolor="rgba(255,255,255,0.85)", xanchor="left",
            )
        save_image(fig, file_prefix)
        return fig

    def importance_enrichment(
        self,
        shap: pd.DataFrame,
        top_fraction: float = 0.1,
    ) -> pd.DataFrame:
        """
        Whether each organism is over-represented among the bins each study's
        models rely on most: for each study, the bins in the top
        ``top_fraction`` of mean |SHAP| across every modelled bin, tested
        against the organisms' shares of all the modelled bins. One row per
        organism per study, labeled by a ``group`` column.
        """
        universe = pd.MultiIndex.from_frame(shap[["organism", "bin"]])
        organisms = organism_order(shap["organism"])
        n_top = max(1, int(round(top_fraction * len(shap))))
        panels = []
        for study in self.studies:
            top = shap.nlargest(n_top, study.label, keep="all")
            foreground = pd.MultiIndex.from_frame(top[["organism", "bin"]])
            panels.append(enrich_organisms(foreground, universe, organisms).assign(group=study.label))
        return pd.concat(panels, ignore_index=True)

    def plot_importance_enrichment(
        self,
        enrichment: pd.DataFrame,
        qvalue_threshold: float = 0.2,
        width: int = 800,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """The output of :meth:`importance_enrichment`, drawn like the association enrichment."""
        return plot_enrichment(
            enrichment,
            label="organism",
            axis_title="Organism",
            title="Organisms among the bins the models rely on most",
            qvalue_threshold=qvalue_threshold,
            order=organism_order(enrichment["organism"]),
            width=width,
            height=None,
            file_prefix=file_prefix,
        )

    def plot_interaction_share(
        self,
        interactions: pd.DataFrame,
        width: int = 900,
        height: int = 460,
        file_prefix: str | None = None,
    ) -> go.Figure:
        """
        How much of each model's attribution among its top bins comes from
        bins acting jointly rather than on their own: the off-diagonal SHAP
        interaction attribution as a share of the total, one bar per organism
        and study. The strongest pair of bins in each model is in the hover.

        A share near zero means the bins add up independently; the higher it
        is, the more the model's signal lies in combinations of bins. With a
        ``file_prefix`` the shares and strongest pairs are also written as CSV.
        """
        rows = []
        for (organism, study), model in interactions.groupby(["organism", "study"]):
            own = model["bin_a"] == model["bin_b"]
            main = model.loc[own, "mean_abs_interaction"].sum()
            joint = model.loc[~own, "mean_abs_interaction"].sum()
            pairs = model.loc[model["bin_a"] < model["bin_b"]].sort_values(
                ["mean_abs_interaction", "bin_a", "bin_b"], ascending=[False, True, True]
            )
            strongest = pairs.iloc[0]
            rows.append({
                "organism": organism,
                "study": study,
                "share": joint / (main + joint),
                "strongest_pair": f"{strongest['bin_a']} x {strongest['bin_b']}",
                "strongest_over_main": strongest["mean_abs_interaction"] / model.loc[own, "mean_abs_interaction"].max(),
            })
        summary = pd.DataFrame(rows)
        if file_prefix is not None:
            summary.to_csv(file_prefix + ".csv", index=False)
        n_top = interactions.groupby(["organism", "study"])["bin_a"].nunique().max()
        fig = self._organism_study_bars(
            summary, y="share", study_order=[study.label for study in self.studies],
            labels={
                "share": "Share of attribution from bin interactions",
                "strongest_over_main": "Strongest pair / strongest main effect",
                "strongest_pair": "Strongest pair",
            },
            hover_data={"share": ":.2f", "strongest_pair": True, "strongest_over_main": ":.2f"},
            title="How much each model relies on bins acting together",
            footnote=f"Attribution among each model's {n_top} most important bins, split into main effects and pairwise interactions",
            width=width, height=height,
        )
        fig.update_yaxes(range=[0, min(1.0, summary["share"].max() * 1.3)], tickformat=".0%")
        save_image(fig, file_prefix)
        return fig



def _significance_stars(p_value: float) -> str:
    for threshold, stars in [(0.001, "***"), (0.01, "**"), (0.05, "*")]:
        if p_value < threshold:
            return stars
    return ""


def _sorted_clusters(values: pd.Series) -> List[str]:
    labels = values.drop_duplicates().tolist()
    labels.sort(key=lambda name: int(name.split(" ")[1]))
    return labels
