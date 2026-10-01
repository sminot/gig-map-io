"""
Readable names for the assemblies a pangenome was built from.

A pangenome names its genomes by assembly file, which carries the GenBank or
RefSeq accession and little else. NCBI's Datasets API knows the organism and
strain behind each accession; this module fetches that once, keeps it in a
JSON cache beside the analysis so that later runs (and the workflow) need no
network, and turns it into a display name like
"A. finegoldii CE91-St15 (GCF_022846055.1)".
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

DATASETS_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/dataset_report"
#: Accessions per request; the API accepts up to 1000
BATCH_SIZE = 200
#: Assembly names that name nothing: NCBI's own (ASM...), project
#: identifiers, pipeline outputs, and versioned organism abbreviations
GENERIC_ASSEMBLY_NAME = re.compile(r"^ASM\d+v\d+$|IMG-taxon|^\d+_\d+#|^PRJ|MAG|hifiasm|_\d+\.\d+$", re.I)
ACCESSION = re.compile(r"(GC[AF]_\d+\.\d+)")

#: What is kept of each report
FIELDS = ("organism_name", "strain", "isolate", "assembly_name", "assembly_level")
#: A strain designation longer than this (a MAG pipeline's scaffold name, say) is cut
MAX_STRAIN_CHARS = 24
SEROTYPE = re.compile(r"^O\d+:?H?\d*$|^O\d+$|^H\d+$")


def assembly_accession(genome: str) -> str | None:
    """The GenBank or RefSeq accession in an assembly file name, if any."""
    match = ACCESSION.search(genome)
    return match.group(1) if match else None


def fetch_assembly_metadata(accessions: Iterable[str], cache: Path) -> Dict[str, dict]:
    """
    The organism, strain and assembly name of each accession, from the cache
    where it has them and from NCBI otherwise, writing what it fetched back to
    the cache. An accession NCBI has no report for (a suppressed assembly) is
    cached as empty so it is not asked for again.
    """
    cache = Path(cache)
    known = json.loads(cache.read_text()) if cache.exists() else {}
    missing = sorted(set(accessions) - set(known))
    fetched = _fetch(missing)
    # An assembly NCBI no longer reports under one accession may still be
    # reported under its GenBank/RefSeq twin, which describes the same genome
    twins = {accession: _twin(accession) for accession in missing if not fetched.get(accession)}
    for accession, fields in _fetch(sorted(set(twins.values()))).items():
        for original, twin in twins.items():
            if twin == accession and fields:
                fetched[original] = fields
    known.update(fetched)
    if missing:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(dict(sorted(known.items())), indent=1) + "\n")
    return known


def _fetch(accessions: list) -> Dict[str, dict]:
    """The kept fields of each accession's report, empty where NCBI has none."""
    found = {accession: {} for accession in accessions}
    for start in range(0, len(accessions), BATCH_SIZE):
        for report in _dataset_reports(accessions[start:start + BATCH_SIZE]):
            organism = report.get("organism", {})
            info = report.get("assembly_info", {})
            found[report["accession"]] = {
                "organism_name": organism.get("organism_name"),
                "strain": organism.get("infraspecific_names", {}).get("strain"),
                "isolate": organism.get("infraspecific_names", {}).get("isolate"),
                "assembly_name": info.get("assembly_name"),
                "assembly_level": info.get("assembly_level"),
            }
        time.sleep(0.4)
    return found


def _twin(accession: str) -> str:
    """The same assembly's other accession: GCA_ for a GCF_ and the reverse."""
    return ("GCA_" if accession.startswith("GCF_") else "GCF_") + accession[4:]


def _dataset_reports(accessions: list) -> list:
    request = urllib.request.Request(
        DATASETS_URL,
        data=json.dumps({"accessions": accessions, "page_size": BATCH_SIZE}).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response).get("reports", [])


def display_name(accession: str, fields: dict) -> str:
    """
    "G. species strain (accession)", from whatever the record carries.

    The genus is reduced to its initial. The organism name may already end
    with the strain, or with a strain designation that differs from the
    strain field, or be "Genus sp. X"; a strain comes from the strain field,
    then the isolate, then the organism name's own suffix, then a non-generic
    assembly name. With no report at all, the accession stands alone.
    """
    name = (fields.get("organism_name") or "").replace("[", "").replace("]", "").strip()
    if not name:
        return accession
    words = name.split()
    prefix = ""
    if words[0] == "Candidatus" and len(words) > 1:
        prefix, words = "Ca. ", words[1:]
    elif words[0] == "uncultured" and len(words) > 1:
        words = words[1:]
    genus, rest = words[0], words[1:]
    species = []
    if rest and rest[0] == "sp.":
        # "Alistipes sp. cv1": the designation after sp. names the isolate
        species = rest[:2] if len(rest) > 1 else rest[:1]
        rest = rest[len(species):]
    else:
        species = rest[:1]
        rest = rest[1:]
        if rest[:1] == ["subsp."] and len(rest) > 1:
            species += rest[:2]
            rest = rest[2:]

    strain = _clean_strain(fields.get("strain") or fields.get("isolate") or "")
    suffix = re.sub(r"\bstr\.\s+", "", " ".join(rest).split(" = ")[0]).strip()
    if suffix and suffix != strain:
        # The organism name carries its own designation, e.g. "timonensis
        # JC136"; a serotype such as "coli O157:H7" is kept beside the strain
        strain = f"{suffix} {strain}".strip() if SEROTYPE.match(suffix) and strain not in suffix else suffix
    if not strain and species[:1] != ["sp."] or (species[:1] == ["sp."] and len(species) == 1 and not strain):
        assembly = fields.get("assembly_name") or ""
        if assembly and not GENERIC_ASSEMBLY_NAME.search(assembly):
            strain = assembly
    if species[:1] == ["sp."] and len(species) > 1:
        # The designation already names the isolate
        strain = ""

    label = f"{prefix}{genus[0]}. {' '.join(species)}"
    if len(strain) > MAX_STRAIN_CHARS:
        strain = strain[:MAX_STRAIN_CHARS - 1] + "\u2026"
    if strain:
        label += f" {strain}"
    return f"{label} ({accession})"


def _clean_strain(strain: str) -> str:
    strain = re.sub(r"^(strain|str\.)\s+", "", strain.strip(), flags=re.I)
    return " ".join(strain.split())


def genome_names(genomes: Iterable[str], cache: Path) -> pd.DataFrame:
    """
    One row per genome file name: its accession, display name, and the
    fields the name was built from.
    """
    genomes = sorted(set(genomes))
    accessions = {genome: assembly_accession(genome) for genome in genomes}
    metadata = fetch_assembly_metadata([a for a in accessions.values() if a], cache)
    rows = []
    for genome in genomes:
        accession = accessions[genome]
        fields = metadata.get(accession, {}) if accession else {}
        rows.append({
            "genome": genome,
            "accession": accession,
            "display_name": display_name(accession, fields) if accession else genome,
            **{field: fields.get(field) for field in FIELDS},
        })
    return pd.DataFrame(rows)
