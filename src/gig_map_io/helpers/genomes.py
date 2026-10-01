"""
Naming conventions of the assemblies a pangenome was built from.
"""

from __future__ import annotations

from typing import Iterable


def genbank_duplicates(genomes: Iterable[str]) -> set:
    """
    The GenBank (GCA_) assemblies among ``genomes`` that are also present as
    their RefSeq (GCF_) copy. NCBI publishes the same assembly under both
    accessions, and a pangenome built from both carries it twice.
    """
    genomes = list(genomes)
    refseq = {name[4:].split("_")[0] for name in genomes if name.startswith("GCF_")}
    return {
        name for name in genomes
        if name.startswith("GCA_") and name[4:].split("_")[0] in refseq
    }
