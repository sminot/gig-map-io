# gig-map-io

Python library for reading the outputs of the [gig-map](https://github.com/fredhutch/gig-map)
workflow (genes-in-genomes map) and running the analyses built on them.

## Overview

- **Reader objects** — one per gig-map workflow, each reading that workflow's
  key outputs:
  - **`Pangenome`** — gene bins, genome content, gene coordinates, ANI distances
  - **`ContrastMetagenomes`** — association results, bin abundance, sample metadata
  - **`PangenomePhylogeny`** — per-bin gene trees

  Each has a `*Set` counterpart (`PangenomeSet`, `ContrastMetagenomesSet`,
  `PangenomePhylogenySet`) wrapping a dict of readers keyed by organism, so an
  analysis can span several pangenomes at once.

- **`Study`** — names the gig-map outputs belonging to one comparison, keyed by
  organism, along with the recoding that turns raw contrast metadata into
  labeled sample groups. A study is serialized as JSON holding relative paths,
  so a definition can be checked in next to the analysis code and re-resolved
  against a downloaded copy of the data. **`StudySet`** spans several studies
  for community-level analyses.

- **Analysis methods** on those objects, rather than free functions: volcano
  plots and contrast comparisons, pangenome summaries and gene maps, community
  ordination and PERMANOVA, Leiden community types, and supervised models with
  SHAP attribution. Every plotting method takes a `file_prefix` and writes PNG,
  trimmed PNG, PDF, HTML and JSON (PNG and PDF for the matplotlib gene maps).

## Installation

```bash
pip install "gig-map-io @ git+https://github.com/sminot/gig-map-io.git"
```

For development:

```bash
git clone https://github.com/sminot/gig-map-io.git
cd gig-map-io
pip install -e ".[dev]"
```

## Quick start

### Reader objects

```python
from pathlib import Path
from gig_map_io import Pangenome, ContrastMetagenomes, PangenomePhylogeny

pangenome = Pangenome(directory=Path("/path/to/pangenome"))
pangenome.gene_bins
pangenome.genome_content
pangenome.core_genome()

contrast = ContrastMetagenomes(directory=Path("/path/to/contrast"), parameter="disease")
contrast.association
contrast.rpkm
contrast.metadata
```

### Studies

A study definition lists the dataset directories that belong together:

```json
{
  "name": "gvhd_combined",
  "label": "GvHD - Combined",
  "parameter": "disease",
  "contrasts": {"Alistipes": "Contrast - GvHD Cohorts - Alistipes (n=414) - .../data"},
  "pangenomes": {"Alistipes": "Pangenome - Alistipes (n=414) (9540b5)/data"},
  "phylogenies": {"Alistipes": "Phylogenies - Alistipes (n=414) (a057ba)/data"},
  "sample_groups": {
    "disease": {
      "sources": [{"column": "disease", "labels": {"1": "Case", "0": "Control"}}],
      "order": ["Case", "Control"]
    }
  }
}
```

Each dataset is named relative to the root of the collection, and that root is
supplied when the study is loaded (`datasets="datasets"` by default). The same
definition therefore resolves against a checkout, a download somewhere else, or
a workflow task with the data staged beside it.

```python
from gig_map_io import Study

study = Study.from_json("studies/gvhd_combined.json", datasets="/data/pangenomes")

study.volcano_plot(max_abs_estimate=2.5, file_prefix="out/volcano")
study.significant_bins(estimate_thresh=0.25, fdr_thresh=0.2, direction="negative")
study.describe_bins(bins, odds_by="cohort")
study.bin_gene_map("Alistipes", "Bin 100", file_prefix="out/gene_map")
```

Comparisons take another study, and label themselves from the two studies'
labels:

```python
ibd = Study.from_json("studies/ibd_combined.json")

study.compare_sig_scatter(ibd, file_prefix="out/scatter")
study.compare_sig_categories(ibd, file_prefix="out/categories")
study.compare_volcano_with_estimate(ibd, file_prefix="out/joint_volcano")
```

### Sample groups

Cohorts encode the same facts differently — one study's `batch_1` is another's
`study_name`, and the study a contrast was assembled around often leaves its
own samples unlabelled. A `SampleGroup` describes that recoding declaratively:
one or more source columns tried in order, the labels to map their values onto,
an optional `default` for nulls, and the category order to use in plots.

```python
study.sample_metadata(["study", "cohort", "disease", "participant"])
study.group_order("cohort")
```

### Several studies at once

```python
from gig_map_io import StudySet

studies = StudySet.from_json("studies/gvhd_combined.json", "studies/ibd_combined.json")

studies.significant_bins()                          # bins significant in every study
studies.tsne_plot("disease", file_prefix="out/tsne")
studies.permanova(file_prefix="out/permanova")
studies.pangenome_clusters("Alistipes")             # Leiden community types
studies.classify_by_organism()                      # XGBoost + SHAP per organism
```

### Running an analysis as a script

`AnalysisScript` gives a script the same command line whether it runs from a
checkout or inside a workflow that stages its inputs somewhere else:

```python
from gig_map_io import AnalysisScript

script = AnalysisScript(
    __file__,
    inputs={"bins": "associated_bins/candidate_bins/02_negative_in_both/bins.csv"},
    description=__doc__,
)

study = script.study("gvhd_combined")
bins = pd.read_csv(script.input("bins"), index_col=[0, 1])
study.bin_abundance_heatmap(bins.index, file_prefix=script.output("figure"))
```

It exposes `--datasets`, `--studies`, `--output-dir`, and one option per
declared input, each defaulting to where that thing sits in a checkout.

## Gene maps and their context

`Pangenome.bin_gene_map` draws the genes of one bin along a line, placed by
their median position across the contigs that carry them.
`Pangenome.bin_context_map` puts that map over one row per genome: the stretch
of contig around the bin with every gene on it drawn as an arrow, colored by
gene identity so that conserved synteny shows as columns of matching color,
with the bin's genes outlined. The genomes whose contigs span most of the
window are shown, a GenBank copy of a RefSeq assembly is dropped, and rows are
ordered by clustering on the genes they carry. Both use `helpers.coords.Coords`
to build the shared coordinate space.

## Genome names

`helpers.ncbi.genome_names` gives every assembly in a pangenome a readable
name, "A. finegoldii CE91-St15 (GCF_022846055.1)", from NCBI's Datasets record
of the accession: the genus reduced to its initial, the strain from the strain
field, the isolate, a designation the organism name carries, or a
submitter-given assembly name, in that order. Records are fetched once and
kept in a JSON cache the caller names, so a complete cache needs no network.
`Study.genome_names(cache)` covers a study's pangenomes; the context map and
`Phylogeny.newick` take the resulting mapping.

## Multi-panel figures

`gig_map_io.helpers.panels.compose_panels` places the PDFs that several
plotting methods wrote onto one page as lettered panels, laid out left to right
in a grid of a given number of columns and wrapping into rows. Each panel is
scaled to its cell's width (optionally spanning several columns) and, if it
would be too tall, to a height cap, so a tall narrow panel does not stretch its
row.

A plotly figure is drawn for the screen, and shrinking its PDF into a 55 mm
cell leaves the text unreadable. So when the `.json` that `save_image` wrote
sits beside a panel's PDF, the panel is drawn again for print: on a canvas
larger than its cell by the ratio of the figure's font size to `font_pt`
(default 7), then scaled into the cell, so the text lands at `font_pt` and
everything else keeps its on-screen proportions. A panel may carry its own
`font_pt`, and `height_mm` to be drawn that tall rather than in the
proportions it had, which gives a legend room in a narrow cell. A PDF with no
specification beside it, such as a matplotlib figure drawn in inches, is placed
as it is. Either way the content stays vector.

```python
from gig_map_io.helpers.panels import compose_panels

compose_panels(
    {
        "width_mm": 170,
        "columns": 3,
        "max_panel_height_mm": 60,
        "panels": [
            {"pdf": "analysis/pangenome_database/rarefaction_curve/figure.pdf"},
            {"pdf": "analysis/combined_studies/gvhd_volcano/figure.pdf", "span": 2},
        ],
    },
    root="path/to/analysis/repo",
    output="figure.pdf",
)
```

## Figure style

Every plotting method draws with the template and colors in
`gig_map_io.helpers.style`, so that figures from different methods read as one
set. `TEMPLATE` (a light grid) and `SIMPLE_TEMPLATE` (axis lines only, for
heatmaps and trees) layer the shared fonts and layout over plotly's own
templates. `organism_order` lists organisms alphabetically by display name, which is the
order every figure uses, `organism_colors` gives every organism a fixed color
by name, and
`group_colors` colors the levels of a sample group in the order the study
declares them, with the first level of a two-level group -- the case-like one
-- in the warm color. `save_image` exports PNGs at twice the layout size.

## Reproducibility

Anything that subsamples takes a `random_state` and defaults to a fixed seed,
so re-running a figure reproduces it. Note that t-SNE and Leiden are both
sensitive to the order of rows and columns in the input, so a study definition
that lists its organisms in a stable order matters for more than tidiness.

Plotly's static image export renders text through a browser, so PNG and PDF
output differs slightly between machines with different font stacks even when
the underlying figure is identical, and PDFs carry a creation timestamp so they
differ on every run regardless. The JSON that `save_image` writes alongside them
is the exact figure specification: compare that when you need to know whether a
figure really changed.

## License

See the LICENSE file for details.
