"""
Phase 2A + Phase 3: TPM normalization and downstream analysis.

Loads raw RSEM expected counts, divides by gene lengths (median transcript
length from ENSEMBL BioMart, or constant 2000 bp fallback), scales to
1 million, applies log2(x+1), applies TSS regression correction, then runs
the full analysis pipeline.

Outputs:
    reports/norm_comparison/tpm/report.md
    reports/norm_comparison/tpm/report_metrics.json
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.utils import load_intermediate, load_gene_cols
from scripts.normalization_comparison.gene_lengths import get_gene_lengths
from scripts.normalization_comparison.tss_correction import apply_tss_correction
from scripts.normalization_comparison.analysis_pipeline import run_analysis


def compute_tpm(counts_df, gene_cols, gene_lengths_dict):
    """
    Compute TPM from raw counts.

    Parameters
    ----------
    counts_df       : DataFrame with 'pid' + gene_cols containing RSEM expected counts
    gene_cols       : list of gene column names
    gene_lengths_dict: dict gene_symbol -> length_bp

    Returns
    -------
    DataFrame with 'pid' + gene_cols containing log2(TPM+1) values
    """
    counts = counts_df[gene_cols].values.astype(np.float64)  # (n_samples, n_genes)

    lengths = np.array([gene_lengths_dict.get(g, 2000) for g in gene_cols],
                       dtype=np.float64)  # (n_genes,)
    lengths_kb = lengths / 1000.0

    # RPK = counts / length_kb
    rpk = counts / lengths_kb[np.newaxis, :]

    # TPM = RPK / sum(RPK) * 1e6
    rpk_sum = rpk.sum(axis=1, keepdims=True)
    rpk_sum = np.where(rpk_sum == 0, 1.0, rpk_sum)  # avoid division by zero
    tpm = rpk / rpk_sum * 1e6

    # Verify: column sums should be ~1e6 before log
    sample_sums = tpm.sum(axis=1)
    print(f"  TPM per-sample sum: mean={sample_sums.mean():.0f}, "
          f"min={sample_sums.min():.0f}, max={sample_sums.max():.0f} (expected ~1e6)")

    # log2(TPM + 1)
    log_tpm = np.log2(tpm + 1.0)

    result = counts_df[['pid']].copy().reset_index(drop=True)
    expr_df = pd.DataFrame(log_tpm, columns=gene_cols)
    return pd.concat([result, expr_df], axis=1)


def main():
    print("=" * 70)
    print("PHASE 2A: TPM NORMALIZATION")
    print("=" * 70)

    # -- Load raw counts
    print("\nLoading raw RSEM expected counts...")
    raw_counts = load_intermediate('01_tumor_raw_filtered')
    clinical = load_intermediate('01_clinical_qc')
    cn = load_intermediate('01_cn_qc')
    gene_cols = load_gene_cols()

    # Filter gene_cols to those present in raw_counts
    gene_cols = [g for g in gene_cols if g in raw_counts.columns]
    print(f"  Raw counts: {len(raw_counts)} samples x {len(gene_cols)} genes")
    print(f"  Clinical: {len(clinical)} patients")
    print(f"  CN: {len(cn)} patients")

    # -- Verify ERBB2 raw values look like raw counts (not already normalized)
    if 'ERBB2' in raw_counts.columns:
        erbb2_median = raw_counts['ERBB2'].median()
        print(f"  ERBB2 median raw count: {erbb2_median:.0f} "
              f"(expected ~5000-15000 for RSEM expected counts)")

    # -- Fetch gene lengths
    print("\n--- Getting gene lengths ---")
    gene_lengths, gl_source = get_gene_lengths(gene_cols)
    print(f"  Gene length source: {gl_source}")

    # Spot-check known genes
    for g in ['ERBB2', 'ACTB', 'GAPDH', 'ESR1', 'TP53']:
        if g in gene_lengths:
            print(f"    {g}: {gene_lengths[g]:.0f} bp")

    # -- Compute TPM
    print("\n--- Computing TPM ---")
    tpm_df = compute_tpm(raw_counts, gene_cols, gene_lengths)
    print(f"  TPM matrix: {tpm_df.shape}")

    # Spot-check ERBB2 TPM values
    if 'ERBB2' in tpm_df.columns:
        print(f"  ERBB2 log2(TPM+1) stats: "
              f"mean={tpm_df['ERBB2'].mean():.3f}, "
              f"std={tpm_df['ERBB2'].std():.3f}, "
              f"min={tpm_df['ERBB2'].min():.3f}, "
              f"max={tpm_df['ERBB2'].max():.3f}")

    # -- Apply TSS correction
    print("\n--- Applying TSS correction ---")
    tpm_tss = apply_tss_correction(tpm_df, clinical, gene_cols)
    print(f"  TSS-corrected TPM matrix: {tpm_tss.shape}")

    # -- Run analysis pipeline
    output_dir = PROJECT_ROOT / 'reports' / 'norm_comparison' / 'tpm'
    metrics = run_analysis(
        tpm_tss, clinical, cn, gene_cols,
        method_name='TPM-TSS',
        output_dir=output_dir,
        gene_length_source=gl_source
    )

    print("\nPhase 2A + 3 (TPM) complete.")
    return metrics


if __name__ == '__main__':
    main()
