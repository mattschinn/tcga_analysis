"""
TSS residualization for normalization comparison.

Applies the same regression-based TSS correction used in
src/01s_tss_batch_assessment.py phase2_correct(), parameterized to accept
any normalized expression matrix. Protects HER2 and ER covariates.
"""

import numpy as np
import pandas as pd


MIN_SAMPLES_PER_TSS = 5


def _collapse_rare_sites(tss_series, min_samples=MIN_SAMPLES_PER_TSS):
    counts = tss_series.value_counts()
    rare = counts[counts < min_samples].index
    collapsed = tss_series.copy()
    collapsed[collapsed.isin(rare)] = "Other"
    return collapsed


def apply_tss_correction(expr_df, clinical, gene_cols):
    """
    Regress out TSS while preserving HER2 and ER signal.

    Parameters
    ----------
    expr_df  : DataFrame with 'pid' column + gene_cols
    clinical : DataFrame with 'pid', 'tss', 'her2_composite',
               'ER Status By IHC' columns
    gene_cols: list of gene column names to correct

    Returns
    -------
    DataFrame with same structure as expr_df but TSS effect subtracted.
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

    # TSS dummies (drop_first for identifiability)
    tss_dummies = pd.get_dummies(df['tss_collapsed'], prefix='tss', drop_first=True)
    n_tss = tss_dummies.shape[1]

    # Protected covariates
    her2 = df['her2_composite'].fillna('Unknown')
    her2_dummies = pd.get_dummies(her2, prefix='her2', drop_first=True)
    er = (df.get('ER Status By IHC', pd.Series(dtype=str)) == 'Positive').astype(float)

    protected = pd.concat([her2_dummies, er.rename('ER_pos')], axis=1)
    n_protected = protected.shape[1]

    X_full = np.column_stack([
        np.ones(len(df)),
        tss_dummies.values,
        protected.values
    ]).astype(np.float64)

    Y = df[gene_cols].fillna(0).values.astype(np.float64)
    B, _, rank, _ = np.linalg.lstsq(X_full, Y, rcond=None)

    print(
        f"  TSS correction: design={X_full.shape[1]} cols "
        f"(1 intercept + {n_tss} TSS + {n_protected} protected), rank={rank}"
    )

    B_tss = B[1:1 + n_tss, :]
    X_tss = X_full[:, 1:1 + n_tss]
    corrected_values = Y - X_tss @ B_tss

    result = df[['pid']].copy().reset_index(drop=True)
    corrected_df = pd.DataFrame(corrected_values, columns=gene_cols)
    result = pd.concat([result, corrected_df], axis=1)

    return result
