"""
Phase 2B + Phase 3: TMM normalization (pure Python) and downstream analysis.

Implements TMM following Robinson & Oshlack (2010) Genome Biology.
Does NOT require R or edgeR. Algorithm:
  1. Reference sample = sample whose upper quartile is closest to mean UQ.
  2. For each sample vs reference: compute M (log-ratio) and A (average log)
     values per gene.
  3. Trim: bottom/top 30% of M, bottom/top 5% of A.
  4. Weighted mean of remaining M-values = log2(TMM_factor).
  5. Effective lib size = lib_size * TMM_factor.
  6. log-CPM = log2(count / eff_lib_size * 1e6 + prior.count).

Outputs:
    reports/norm_comparison/tmm_edger/report.md
    reports/norm_comparison/tmm_edger/report_metrics.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.utils import load_intermediate, load_gene_cols
from scripts.normalization_comparison.tss_correction import apply_tss_correction
from scripts.normalization_comparison.analysis_pipeline import run_analysis

PRIOR_COUNT = 1.0  # adds to CPM before log2, same as edgeR default


def compute_tmm_factors(counts, lib_sizes, m_trim=0.30, a_trim=0.05):
    """
    Compute TMM normalization factors for all samples.

    Parameters
    ----------
    counts    : ndarray (n_samples, n_genes) -- non-negative integer-ish counts
    lib_sizes : ndarray (n_samples,)

    Returns
    -------
    factors : ndarray (n_samples,) -- multiplicative scaling factors
              Normalized for geometric mean = 1 (standard edgeR convention).
    """
    n_samples, n_genes = counts.shape

    # Step 1: pick reference sample (UQ closest to mean UQ)
    uqs = np.array([
        np.percentile(counts[i, counts[i, :] > 0], 75)
        if (counts[i, :] > 0).any() else 0.0
        for i in range(n_samples)
    ])
    mean_uq = uqs.mean()
    ref_idx = int(np.argmin(np.abs(uqs - mean_uq)))
    print(f"  TMM reference sample index: {ref_idx} (UQ={uqs[ref_idx]:.0f}, mean_UQ={mean_uq:.0f})")

    ref_counts = counts[ref_idx, :].astype(np.float64)
    ref_lib = lib_sizes[ref_idx]

    log_factors = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        if i == ref_idx:
            log_factors[i] = 0.0
            continue

        sam_counts = counts[i, :].astype(np.float64)
        sam_lib = lib_sizes[i]

        # Keep only genes expressed in both samples
        keep = (sam_counts > 0) & (ref_counts > 0)
        if keep.sum() < 10:
            log_factors[i] = 0.0
            continue

        s = sam_counts[keep]
        r = ref_counts[keep]
        ls = sam_lib
        lr = ref_lib

        # M = log2(s/ls) - log2(r/lr) = log2-fold-change in CPM
        # A = (log2(s/ls) + log2(r/lr)) / 2 = average log-CPM
        with np.errstate(divide='ignore', invalid='ignore'):
            M = np.log2(s / ls) - np.log2(r / lr)
            A = 0.5 * (np.log2(s / ls) + np.log2(r / lr))

        # Precision weight per gene
        w = (ls - s) / (ls * s) + (lr - r) / (lr * r)
        w = np.where(w <= 0, 1e-10, w)

        # Trim
        m_lo, m_hi = np.percentile(M, [m_trim * 100, (1 - m_trim) * 100])
        a_lo, a_hi = np.percentile(A, [a_trim * 100, (1 - a_trim) * 100])
        trimmed = (M >= m_lo) & (M <= m_hi) & (A >= a_lo) & (A <= a_hi)

        if trimmed.sum() < 5:
            log_factors[i] = 0.0
            continue

        # Weighted mean of M
        log_factors[i] = np.sum(w[trimmed] * M[trimmed]) / np.sum(w[trimmed])

    # Normalize so geometric mean of factors = 1 (standard edgeR convention)
    log_factors -= log_factors.mean()
    factors = 2.0 ** log_factors
    return factors


def compute_tmm(counts_df, gene_cols, prior_count=PRIOR_COUNT):
    """
    Apply TMM normalization and return log2-CPM matrix.

    Parameters
    ----------
    counts_df   : DataFrame with 'pid' + gene_cols (RSEM expected counts)
    gene_cols   : gene column names

    Returns
    -------
    DataFrame with 'pid' + gene_cols containing log2(CPM+prior) values
    """
    counts_mat = counts_df[gene_cols].values.astype(np.float64)
    counts_int = np.round(counts_mat).astype(np.float64)  # TMM expects integer-like

    lib_sizes = counts_int.sum(axis=1).astype(np.float64)
    print(f"  Library sizes: mean={lib_sizes.mean():.0f}, "
          f"min={lib_sizes.min():.0f}, max={lib_sizes.max():.0f}")

    # Compute TMM factors
    factors = compute_tmm_factors(counts_int, lib_sizes)
    print(f"  TMM factors: mean={factors.mean():.4f}, "
          f"min={factors.min():.4f}, max={factors.max():.4f} "
          f"(expected ~0.9-1.1 range)")

    # Effective library sizes
    eff_lib = lib_sizes * factors  # (n_samples,)

    # CPM with prior count: log2( (count + prior*sum(factors)/n_samples) / eff_lib * 1e6 )
    # This matches edgeR's cpm(log=TRUE, prior.count=1)
    # Simplified: log2( (count + prior) / eff_lib * 1e6 )
    n_samples = counts_int.shape[0]
    adjusted = counts_int + prior_count
    cpm = adjusted / eff_lib[:, np.newaxis] * 1e6
    log_cpm = np.log2(cpm)

    print(f"  ERBB2 log2-CPM stats (if present):")
    erbb2_idx = gene_cols.index('ERBB2') if 'ERBB2' in gene_cols else None
    if erbb2_idx is not None:
        erbb2_vals = log_cpm[:, erbb2_idx]
        print(f"    mean={erbb2_vals.mean():.3f}, std={erbb2_vals.std():.3f}, "
              f"min={erbb2_vals.min():.3f}, max={erbb2_vals.max():.3f}")

    result = counts_df[['pid']].copy().reset_index(drop=True)
    expr_df = pd.DataFrame(log_cpm, columns=gene_cols)
    return pd.concat([result, expr_df], axis=1)


def main():
    print("=" * 70)
    print("PHASE 2B: TMM NORMALIZATION (pure Python)")
    print("=" * 70)

    # -- Load raw counts
    print("\nLoading raw RSEM expected counts...")
    raw_counts = load_intermediate('01_tumor_raw_filtered')
    clinical = load_intermediate('01_clinical_qc')
    cn = load_intermediate('01_cn_qc')
    gene_cols = load_gene_cols()

    gene_cols = [g for g in gene_cols if g in raw_counts.columns]
    print(f"  Raw counts: {len(raw_counts)} samples x {len(gene_cols)} genes")

    # Verify values look like raw counts
    if 'ERBB2' in raw_counts.columns:
        print(f"  ERBB2 raw median: {raw_counts['ERBB2'].median():.0f}")

    # -- Compute TMM
    print("\n--- Computing TMM normalization ---")
    tmm_df = compute_tmm(raw_counts, gene_cols)
    print(f"  TMM matrix: {tmm_df.shape}")

    # -- Apply TSS correction
    print("\n--- Applying TSS correction ---")
    tmm_tss = apply_tss_correction(tmm_df, clinical, gene_cols)
    print(f"  TSS-corrected TMM matrix: {tmm_tss.shape}")

    # -- Run analysis pipeline
    output_dir = PROJECT_ROOT / 'reports' / 'norm_comparison' / 'tmm_edger'
    metrics = run_analysis(
        tmm_tss, clinical, cn, gene_cols,
        method_name='TMM-edgeR (pure Python)',
        output_dir=output_dir,
        gene_length_source='N/A (TMM does not require gene lengths)'
    )

    print("\nPhase 2B + 3 (TMM) complete.")
    return metrics


if __name__ == '__main__':
    main()
