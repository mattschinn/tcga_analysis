"""
Gene length fetching for TPM normalization.

Tries ENSEMBL BioMart (urllib, no extra packages). If unavailable, falls back
to a constant 2000 bp for all genes (effectively RPM). Documents which path
was taken.
"""

import os
import urllib.request
import urllib.parse
import io
import csv
import numpy as np
import pandas as pd

CACHE_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'gene_lengths_cache.csv'
)
FALLBACK_LENGTH_BP = 2000
BIOMART_TIMEOUT = 60  # seconds


_BIOMART_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<!DOCTYPE Query>'
    '<Query virtualSchemaName="default" formatter="TSV" header="1"'
    ' uniqueRows="0" count="" datasetConfigVersion="0.6">'
    '<Dataset name="hsapiens_gene_ensembl" interface="default">'
    '<Attribute name="hgnc_symbol"/>'
    '<Attribute name="transcript_length"/>'
    '</Dataset>'
    '</Query>'
)

_BIOMART_URL = 'https://www.ensembl.org/biomart/martservice'


def _fetch_from_biomart():
    """Query ENSEMBL BioMart for hgnc_symbol -> transcript_length (all rows).
    Returns DataFrame with columns ['gene', 'length']. May take 30-60 s.
    """
    print("  Fetching gene lengths from ENSEMBL BioMart (this may take ~60s)...")
    params = urllib.parse.urlencode({'query': _BIOMART_XML}).encode('utf-8')
    req = urllib.request.Request(_BIOMART_URL, data=params, method='POST')
    with urllib.request.urlopen(req, timeout=BIOMART_TIMEOUT) as resp:
        raw = resp.read().decode('utf-8')

    # Parse TSV -- first line is header: Gene name, Transcript length
    reader = csv.reader(io.StringIO(raw), delimiter='\t')
    header = next(reader)

    rows = []
    for row in reader:
        if len(row) < 2:
            continue
        gene, length_str = row[0].strip(), row[1].strip()
        if not gene or not length_str:
            continue
        try:
            rows.append((gene, int(length_str)))
        except ValueError:
            continue

    df = pd.DataFrame(rows, columns=['gene', 'length'])
    return df


def get_gene_lengths(gene_list, cache_path=CACHE_PATH):
    """
    Return a dict mapping gene symbol -> median transcript length in bp.

    Strategy:
    1. If cache exists, load it.
    2. Else, fetch from ENSEMBL BioMart and save to cache.
    3. If fetch fails (no internet, timeout, etc.), use constant FALLBACK_LENGTH_BP.

    Parameters
    ----------
    gene_list  : list of gene symbols to look up
    cache_path : path to CSV cache file (gene, length)

    Returns
    -------
    lengths : dict  {gene_symbol -> length_bp (float)}
    source  : str describing where lengths came from
    """
    cache_path = os.path.normpath(cache_path)

    # -- Try cache first
    if os.path.exists(cache_path):
        print(f"  Loading gene lengths from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        median_lengths = df.groupby('gene')['length'].median()
        lengths = {g: median_lengths.get(g, FALLBACK_LENGTH_BP) for g in gene_list}
        n_found = sum(1 for g in gene_list if g in median_lengths.index)
        print(f"  Found lengths for {n_found}/{len(gene_list)} genes from cache.")
        return lengths, f"ENSEMBL BioMart (cached at {cache_path})"

    # -- Try live fetch
    try:
        raw_df = _fetch_from_biomart()
        if len(raw_df) < 1000:
            raise ValueError(f"BioMart returned only {len(raw_df)} rows -- likely an error")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        raw_df.to_csv(cache_path, index=False)
        print(f"  Cached {len(raw_df)} transcript records to {cache_path}")

        median_lengths = raw_df.groupby('gene')['length'].median()
        lengths = {g: median_lengths.get(g, FALLBACK_LENGTH_BP) for g in gene_list}
        n_found = sum(1 for g in gene_list if g in median_lengths.index)
        print(f"  Found lengths for {n_found}/{len(gene_list)} genes from BioMart.")
        return lengths, "ENSEMBL BioMart (live fetch)"

    except Exception as exc:
        print(
            f"  WARNING: BioMart fetch failed ({exc}). "
            f"Falling back to constant {FALLBACK_LENGTH_BP} bp for all genes. "
            "TPM becomes equivalent to RPM under this approximation."
        )
        lengths = {g: FALLBACK_LENGTH_BP for g in gene_list}
        return lengths, f"Constant {FALLBACK_LENGTH_BP} bp fallback (BioMart unavailable)"
