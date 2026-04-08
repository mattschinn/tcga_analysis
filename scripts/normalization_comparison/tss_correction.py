"""
TSS batch correction for normalization comparison.

Uses ComBat (parametric empirical Bayes; Johnson et al. 2007) with biological
covariates (HER2, ER) to correct tissue source site batch effects while
preserving biological signal.

ComBat was selected over OLS regression after a systematic comparison
(see scripts/01s_combat_vs_regression.py). Both methods perform comparably
on batch removal (TSS eta2 on PC1) and signal preservation (ERBB2 Cohen's d).
ComBat was chosen as the conventional default in genomics batch correction,
with additional advantages:
  - Corrects both location (mean shift) and scale (variance) per batch
  - Empirical Bayes shrinkage stabilizes estimates for small batch sizes
  - Robust to the 22 TSS groups with variable sample counts

The TSS-HER2 confound (chi2=177.9, p=3.73e-10) is handled by including
HER2 and ER as biological covariates in the ComBat model, so the method
estimates batch effects conditional on known biology.
"""

import numpy as np
import pandas as pd
from combat.pycombat import pycombat


MIN_SAMPLES_PER_TSS = 5


def _collapse_rare_sites(tss_series, min_samples=MIN_SAMPLES_PER_TSS):
    counts = tss_series.value_counts()
    rare = counts[counts < min_samples].index
    collapsed = tss_series.copy()
    collapsed[collapsed.isin(rare)] = "Other"
    return collapsed


def apply_tss_correction(expr_df, clinical, gene_cols):
    """
    ComBat batch correction for TSS with HER2/ER as protected covariates.

    Parameters
    ----------
    expr_df  : DataFrame with 'pid' column + gene_cols
    clinical : DataFrame with 'pid', 'tss', 'her2_composite',
               'ER Status By IHC' columns
    gene_cols: list of gene column names to correct

    Returns
    -------
    DataFrame with same structure as expr_df but TSS batch effects removed.
    """
    clin_dedup = clinical.drop_duplicates(subset='pid')
    df = expr_df.merge(
        clin_dedup[['pid', 'tss', 'her2_composite', 'ER Status By IHC']],
        on='pid', how='left'
    )

    if df['tss'].isna().all():
        print("  WARNING: no TSS data available -- skipping TSS correction")
        return expr_df.copy()

    df['tss_collapsed'] = _collapse_rare_sites(df['tss'])
    n_batches = df['tss_collapsed'].nunique()

    # Expression matrix: genes x samples (ComBat convention)
    expr_matrix = df[gene_cols].T
    expr_matrix.columns = df['pid'].values
    batch = df['tss_collapsed'].values.tolist()

    # Biological covariates (list of lists, one per covariate)
    her2 = df['her2_composite'].fillna('Unknown')
    her2_dummies = pd.get_dummies(her2, prefix='her2', drop_first=True)
    er = (df.get('ER Status By IHC', pd.Series(dtype=str)) == 'Positive').astype(float)

    mod = []
    for col in her2_dummies.columns:
        mod.append(her2_dummies[col].values.tolist())
    mod.append(er.values.tolist())
    n_covariates = len(mod)

    print(f"  ComBat TSS correction: {len(gene_cols)} genes x {len(df)} samples, "
          f"{n_batches} batches, {n_covariates} protected covariates")

    corrected = pycombat(expr_matrix, batch, mod=mod)

    result = df[['pid']].copy().reset_index(drop=True)
    corrected_df = pd.DataFrame(corrected.T.values, columns=gene_cols)
    result = pd.concat([result, corrected_df], axis=1)

    return result
