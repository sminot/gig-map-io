# gig-map-io

Python library for **reading the outputs of the [gig-map](https://github.com/fredhutch/gig-map) workflow** (genes-in-genomes map). It provides reader objects for each gig-map workflow and plotting functions that use those objects for common visualizations.

## Overview

- **Reader objects** — One per gig-map workflow; each reads the key outputs of that workflow:
  - **`Pangenome`** — pangenome workflow outputs (e.g. gene bins, genome content)
  - **`ContrastMetagenomes`** — contrast-metagenomes workflow (summary, association results, optional RPKM)
  - **`PangenomePhylogeny`** — phylogeny workflow (e.g. bin trees)

- **Plotting functions** — Take one or more reader objects and return Plotly figures. **Volcano** (single contrast), **compare-contrasts** (signed log q-value scatter for two contrasts), and **estimate-scatter** (Estimate ± SE for two contrasts) are implemented; **double-volcano** (two volcano plots comparing two contrasts), bin abundance, and bin phylogeny are stubbed.

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

### Using reader objects

```python
from pathlib import Path
from gig_map_io import Pangenome, ContrastMetagenomes, PangenomePhylogeny

# Read pangenome outputs
pang = Pangenome(directory=Path("/path/to/pangenome"))
gene_bins = pang.gene_bins
genome_content = pang.genome_content

# Read contrast outputs (summary, association, optional RPKM)
contrast = ContrastMetagenomes(directory=Path("/path/to/contrast"), parameter="my_parameter")
summary = contrast.summary
association = contrast.association   # association/association.csv
rpkm = contrast.rpkm                 # bin_abundance/rpkm.csv.gz if present
```

### Plotting

```python
from pathlib import Path
from gig_map_io import (
    ContrastMetagenomes,
    plot_volcano,
    plot_compare_contrasts,
    plot_estimate_scatter,
    plot_bin_abundance,
    plot_bin_phylogeny,
)

# Volcano (single contrast)
fig = plot_volcano(contrast, fdr_thresh=0.05, estimate_thresh=0.5)

# Compare two contrasts: concordance scatter (signed log10 q-value) and estimate scatter (Estimate ± SE)
c1 = ContrastMetagenomes(directory=Path("/path/to/contrast1"), parameter="param1")
c2 = ContrastMetagenomes(directory=Path("/path/to/contrast2"), parameter="param2")
fig = plot_compare_contrasts(c1, c2, label1="Study A", label2="Study B")
fig = plot_estimate_scatter(c1, c2, fdr_thresh=0.05)

# Bin abundance and bin phylogeny: stubbed (NotImplementedError)
```

## License

See the LICENSE file for details.
