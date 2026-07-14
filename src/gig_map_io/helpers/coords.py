from collections import defaultdict
from typing import Dict

import pandas as pd
import numpy as np


class Coords:
    start_coords: Dict[str, float]
    stop_coords: Dict[str, float]
    n: Dict[str, int]

    def __init__(self, aln: pd.DataFrame):
        self.start_coords = defaultdict(list)
        self.stop_coords = defaultdict(list)
        self.seen = set([])

        # Make sure there is only one gene alignment per contig
        aln = aln.groupby(["sseqid", "qseqid", "genome"]).head(1).assign(
            contig=lambda d: d.apply(
                lambda r: f"{r['genome']} - {r['qseqid']}",
                axis=1
            ),
            length=lambda d: d.apply(
                lambda r: np.abs(r["qend"] - r["qstart"]),
                axis=1
            )
        )

        # Get the median gene length from the input
        self._input_aln_len = aln.groupby("sseqid")["length"].median()

        # Get the list of contigs, sorted by the number of genes
        contig_sizes = aln["contig"].value_counts()

        for contig in contig_sizes.index.values:
            contig_aln = aln.query(f"contig == '{contig}'").set_index("sseqid")
            self.add_contig(
                contig_aln["qstart"].to_dict(),
                contig_aln["qend"].to_dict()
            )

    def to_df(self):

        df = pd.DataFrame([
            dict(
                gene=gene,
                start=np.median(self.start_coords[gene]),
                stop=np.median(self.stop_coords[gene]),
                len=np.abs(np.median(self.start_coords[gene]) - np.median(self.stop_coords[gene])),
                n=len(self.start_coords[gene]),
                input_len=self._input_aln_len[gene]
            )
            for gene in list(self.seen)
        ]).assign(
            dir=lambda d: d.apply(lambda r: "fwd" if r['stop'] > r['start'] else "rev", axis=1)
        ).sort_values(by="start").query("len > 0")

        # If all of the genes are in the reverse direction, flip the coordinates by multiplying by -1
        if df["dir"].value_counts().get("rev", 0) == df.shape[0]:
            df = df.assign(
                start=df["start"] * -1,
                stop=df["stop"] * -1
            )
            df = df.assign(
                dir=lambda d: d.apply(lambda r: "fwd" if r['stop'] > r['start'] else "rev", axis=1)
            ).sort_values(by="start")

        # Make it start at 0
        min_val = np.min([df["start"].min(), df["stop"].min()])

        df = df.assign(
            start=df["start"] - min_val,
            stop=df["stop"] - min_val
        ).reset_index(drop=True)

        return df

    def add_contig(self, start_coords: Dict[str, int], stop_coords: Dict[str, int]):

        # If this is the first contig
        if len(self.seen) == 0:
            # Just add the coordinates as given
            self.add_coords(start_coords, stop_coords, reverse=False)

        else:

            # For every gene which has been seen, calculate the offset of start and stop coordinates
            # Once for the forward and once for the reverse
            fwd_offset = self.calc_offset(start_coords, stop_coords, reverse=False)
            rev_offset = self.calc_offset(start_coords, stop_coords, reverse=True)

            # If there is some overlap
            if fwd_offset["n"] >= 1:

                # If the reverse direction has a more consistent offset, use that
                reverse = fwd_offset["std"] > rev_offset["std"]

                # Add the new set of coordinates
                self.add_coords(
                    start_coords,
                    stop_coords,
                    reverse=reverse,
                    offset=(
                        rev_offset["mean"]
                        if reverse
                        else fwd_offset["mean"]
                    )
                )

    def add_coords(
        self,
        start_coords: Dict[str, int],
        stop_coords: Dict[str, int],
        reverse: bool,
        offset=0
    ):
        for gene in start_coords:
            self.start_coords[gene].append((start_coords[gene] + offset) * (-1 if reverse else 1))
            self.stop_coords[gene].append((stop_coords[gene] + offset) * (-1 if reverse else 1))
            self.seen.add(gene)

    def calc_offset(self, start_coords: Dict[str, int], stop_coords: Dict[str, int], reverse: bool):
        start_offset = [
            np.median(self.start_coords[gene]) - ((-1 if reverse else 1) * start_coords[gene])
            for gene in start_coords
            if gene in self.seen
        ]
        stop_offset = [
            np.median(self.stop_coords[gene]) - ((-1 if reverse else 1) * stop_coords[gene])
            for gene in stop_coords
            if gene in self.seen
        ]
        n = len(start_offset)
        if n == 0:
            return dict(n=0)

        else:
            return dict(
                mean=np.mean(start_offset + stop_offset),
                std=np.std(start_offset + stop_offset),
                n=n
            )