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
  labelled sample groups. A study is serialized as JSON holding relative paths,
  so a definition can be checked in next to the analysis code and re-resolved
  against a downloaded copy of the data. **`StudySet`** spans several studies
  for community-level analyses.

- **Analysis methods** on those objects, rather than free functions: volcano
  plots and contrast comparisons, pangenome summaries and gene maps, community
  ordination and PERMANOVA, Leiden community types, and supervised models with
  SHAP attribution. Every plotting method takes a `file_prefix` and writes PNG,
  trimmed PNG, PDF, HTML and JSON.

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

## Reproducibility

Anything that subsamples takes a `random_state` and defaults to a fixed seed,
so re-running a figure reproduces it. Note that t-SNE and Leiden are both
sensitive to the order of rows and columns in the input, so a study definition
that lists its organisms in a stable order matters for more than tidiness.

Plotly's static image export renders text through a browser, so PNG and PDF
output differs slightly between machines with different font stacks even when
the underlying figure is identical. The JSON that `save_image` writes alongside
them is the exact figure specification, and does not.

## License

See the LICENSE file for details.
