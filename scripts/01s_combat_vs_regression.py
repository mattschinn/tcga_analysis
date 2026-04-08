"""
01s_combat_vs_regression.py -- Exploratory comparison of batch correction methods.

Compares the existing OLS regression-based TSS correction (from src/01s_tss_batch_assessment.py)
with Johnson-Rabinovic ComBat (empirical Bayes) on the same normalized expression data.

Decision metrics:
  Primary:   TSS eta2 on PC1 (lower = less residual batch in dominant axis).
  Guardrail: ERBB2 Cohen's d must not drop below uncorrected baseline.
  Secondary: Pearson r(PC1, read-depth proxy) reported for reference.

ComBat assumptions & trade-offs
-------------------------------
Assumptions:
  1. Batch effects are additive (location) and multiplicative (scale) on the
     expression scale.  The parametric variant assumes gene-level batch shifts
     are drawn from Normal (location) and Inverse-Gamma (scale) priors.
  2. Batches are *orthogonal* to the biology of interest -- i.e., biological
     covariates are balanced across batches, or at minimum the design matrix
     can separate batch from biology.
  3. Batch sizes are large enough for the EB shrinkage to be well-estimated.

Advantages over OLS regression:
  - Borrows strength across genes via empirical Bayes shrinkage, so batch
    estimates are more stable for low-expression or noisy genes.
  - Corrects both location (mean shift) and scale (variance inflation/deflation)
    per batch, whereas OLS only removes the mean shift.
  - Widely used in genomics; reviewers and collaborators expect it.

Disadvantages / risks for this dataset:
  - TSS is confounded with HER2 status (chi2=177.9, p=3.73e-10).  ComBat
    cannot cleanly separate batch from biology when they are non-orthogonal,
    even when HER2/ER are included as model covariates.  The EB shrinkage may
    attenuate real biological signal that co-varies with batch.
  - Many TSS groups are small (<5 samples).  EB hyperparameter estimation can
    be unstable with tiny batches, and the prior may dominate.
  - ComBat is a black-box parametric model; OLS regression is transparent and
    lets us subtract only the batch component of a joint model, preserving
    protected covariates more explicitly.

Usage:
    python scripts/01s_combat_vs_regression.py

Outputs:
    outputs/01s_combat_comparison.parquet  -- per-method metrics table
    outputs/figures/fig_01s_combat_vs_regression.png
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -- Project setup ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils import (
    load_intermediate, load_gene_cols, save_intermediate, savefig,
    ensure_dirs, OUTPUT_DIR, FIGURE_DIR
)

# Suppress combat's internal warnings about small batches
warnings.filterwarnings("ignore", category=RuntimeWarning)

N_PCS = 10
MIN_SAMPLES_PER_TSS = 5  # Match the main pipeline


# -- Helpers ---------------------------------------------------------------------

def read_depth_proxy(expr_df, gene_cols, n_top=3000):
    """Sum of top-N MAD genes as a read-depth proxy (matches E3 metric)."""
    gene_mad = expr_df[gene_cols].apply(
        lambda x: np.median(np.abs(x - np.median(x))), axis=0
    )
    top_genes = gene_mad.nlargest(min(n_top, len(gene_cols))).index.tolist()
    return expr_df[top_genes].sum(axis=1).values


def run_pca_metrics(expr_df, gene_cols, tss_groups, her2_groups, label=""):
    """Run PCA and compute the key comparison metrics."""
    X = expr_df[gene_cols].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=N_PCS, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    pc1 = pcs[:, 0]

    # E3: PC1 vs read-depth proxy
    depth = read_depth_proxy(expr_df, gene_cols)
    r_depth, p_depth = stats.pearsonr(pc1, depth)

    # TSS eta2 on PC1
    valid_tss = tss_groups.notna()
    # Inline eta2
    grand_mean = pc1[valid_tss].mean()
    ss_total = ((pc1[valid_tss] - grand_mean) ** 2).sum()
    ss_between = 0.0
    for g in tss_groups[valid_tss].unique():
        mask = (tss_groups == g).values & valid_tss.values
        n_g = mask.sum()
        if n_g > 0:
            ss_between += n_g * (pc1[mask].mean() - grand_mean) ** 2
    eta2_tss = ss_between / ss_total if ss_total > 0 else np.nan

    # HER2 eta2 on PC1
    valid_h = her2_groups.notna()
    gm_h = pc1[valid_h].mean()
    ss_t_h = ((pc1[valid_h] - gm_h) ** 2).sum()
    ss_b_h = 0.0
    for g in her2_groups[valid_h].unique():
        mask = (her2_groups == g).values & valid_h.values
        n_g = mask.sum()
        if n_g > 0:
            ss_b_h += n_g * (pc1[mask].mean() - gm_h) ** 2
    eta2_her2 = ss_b_h / ss_t_h if ss_t_h > 0 else np.nan

    # ERBB2 Cohen's d (HER2+ vs HER2-)
    if 'ERBB2' in gene_cols:
        pos = expr_df.loc[her2_groups == 'Positive', 'ERBB2'].dropna()
        neg = expr_df.loc[her2_groups == 'Negative', 'ERBB2'].dropna()
        if len(pos) > 1 and len(neg) > 1:
            pooled_sd = np.sqrt(
                ((len(pos) - 1) * pos.var() + (len(neg) - 1) * neg.var())
                / (len(pos) + len(neg) - 2)
            )
            cohens_d = (pos.mean() - neg.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        else:
            cohens_d = np.nan
    else:
        cohens_d = np.nan

    var_explained_pc1 = pca.explained_variance_ratio_[0]

    print(f"\n  [{label}]")
    print(f"    PC1 variance explained: {var_explained_pc1:.1%}")
    print(f"    PC1 vs depth proxy:  r = {r_depth:.4f}  (p = {p_depth:.2e})")
    print(f"    TSS eta2 on PC1:     {eta2_tss:.4f}")
    print(f"    HER2 eta2 on PC1:    {eta2_her2:.4f}")
    print(f"    ERBB2 Cohen's d:     {cohens_d:.4f}")

    return {
        'method': label,
        'pc1_var_explained': var_explained_pc1,
        'pc1_depth_r': r_depth,
        'pc1_depth_p': p_depth,
        'pc1_tss_eta2': eta2_tss,
        'pc1_her2_eta2': eta2_her2,
        'erbb2_cohens_d': cohens_d,
    }, pcs, pca


# -- Correction methods ----------------------------------------------------------

def ols_tss_correction(df, gene_cols):
    """OLS regression-based TSS correction (mirrors src/01s_tss_batch_assessment.py)."""
    tss_dummies = pd.get_dummies(df['tss_collapsed'], prefix='tss', drop_first=True)
    n_tss = tss_dummies.shape[1]

    protected = pd.DataFrame(index=df.index)
    her2 = df['her2_composite'].fillna('Unknown')
    her2_dummies = pd.get_dummies(her2, prefix='her2', drop_first=True)
    protected = pd.concat([protected, her2_dummies], axis=1)
    er = (df.get('ER Status By IHC', pd.Series(dtype=str)) == 'Positive').astype(float)
    protected['ER_pos'] = er

    X_design = np.column_stack([
        np.ones(len(df)),
        tss_dummies.values,
        protected.values
    ]).astype(np.float64)

    Y = df[gene_cols].fillna(0).values.astype(np.float64)
    B, _, _, _ = np.linalg.lstsq(X_design, Y, rcond=None)

    B_tss = B[1:1+n_tss, :]
    X_tss = X_design[:, 1:1+n_tss]
    corrected_values = Y - X_tss @ B_tss

    result = df[['pid']].copy()
    corrected_df = pd.DataFrame(corrected_values, columns=gene_cols, index=df.index)
    result = pd.concat([result, corrected_df], axis=1)
    return result


def combat_correction(df, gene_cols):
    """ComBat (parametric empirical Bayes) TSS correction with biological covariates."""
    from combat.pycombat import pycombat

    # Expression matrix: genes x samples (ComBat convention)
    expr_matrix = df[gene_cols].T  # (n_genes, n_samples)
    expr_matrix.columns = df['pid'].values

    batch = df['tss_collapsed'].values.tolist()

    # pycombat mod: list of lists, each inner list is one covariate across samples
    her2 = df['her2_composite'].fillna('Unknown')
    her2_dummies = pd.get_dummies(her2, prefix='her2', drop_first=True)
    er = (df.get('ER Status By IHC', pd.Series(dtype=str)) == 'Positive').astype(float)

    mod = []
    for col in her2_dummies.columns:
        mod.append(her2_dummies[col].values.tolist())
    mod.append(er.values.tolist())

    print("  Running ComBat (parametric)...")
    print(f"    {len(gene_cols)} genes x {len(df)} samples, "
          f"{df['tss_collapsed'].nunique()} batches")
    print(f"    {len(mod)} biological covariates protected")

    corrected = pycombat(expr_matrix, batch, mod=mod)

    result = df[['pid']].copy()
    corrected_df = pd.DataFrame(corrected.T.values, columns=gene_cols, index=df.index)
    result = pd.concat([result, corrected_df], axis=1)
    return result


def combat_no_covariates(df, gene_cols):
    """ComBat without biological covariates -- naive version for comparison."""
    from combat.pycombat import pycombat

    expr_matrix = df[gene_cols].T
    expr_matrix.columns = df['pid'].values
    batch = df['tss_collapsed'].values

    print("  Running ComBat (no covariates)...")
    corrected = pycombat(expr_matrix, batch)

    result = df[['pid']].copy()
    corrected_df = pd.DataFrame(corrected.T.values, columns=gene_cols, index=df.index)
    result = pd.concat([result, corrected_df], axis=1)
    return result


# -- Main -----------------------------------------------------------------------

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("COMBAT vs OLS REGRESSION: TSS BATCH CORRECTION COMPARISON")
    print("=" * 70)

    # -- Load data ---------------------------------------------------------------
    # Pre-TSS UQ data (input to any correction method)
    tumor_uq = load_intermediate("01_tumor_norm_uq")
    # Pipeline-produced TSS-corrected outputs (ground truth baselines)
    uq_tss = load_intermediate("01_tumor_norm_uq_tss")
    tmm_tss = load_intermediate("01_tumor_norm_tmm_tss")
    clinical = load_intermediate("01_clinical_qc")
    gene_cols = load_gene_cols("01_gene_cols")

    # -- Align to a common sample set -------------------------------------------
    # UQ-TSS has 7 duplicate pids from the clinical merge; deduplicate for
    # apples-to-apples comparison.  Use the intersection of all three datasets.
    uq_tss_dedup = uq_tss.drop_duplicates(subset='pid')
    common_pids = sorted(
        set(tumor_uq['pid']) & set(uq_tss_dedup['pid']) & set(tmm_tss['pid'])
    )
    print(f"\nCommon samples across all intermediates: {len(common_pids)}")

    tumor_uq = (tumor_uq[tumor_uq['pid'].isin(common_pids)]
                .sort_values('pid').reset_index(drop=True))
    uq_tss_dedup = (uq_tss_dedup[uq_tss_dedup['pid'].isin(common_pids)]
                    .sort_values('pid').reset_index(drop=True))
    tmm_tss_aligned = (tmm_tss[tmm_tss['pid'].isin(common_pids)]
                       .sort_values('pid').reset_index(drop=True))

    n_samples = len(common_pids)
    print(f"Aligned: {n_samples} samples x {len(gene_cols)} genes")

    # Merge metadata onto the pre-TSS data for ComBat.
    # Use a left join on the deduplicated clinical to avoid expanding rows
    # (some pids have duplicate clinical rows, which inflated the old pipeline
    # from 1093 to 1100).
    clinical_dedup = clinical.drop_duplicates(subset='pid')
    df = tumor_uq.merge(
        clinical_dedup[['pid', 'tss', 'her2_composite', 'ER Status By IHC']],
        on='pid', how='left'
    )
    assert len(df) == n_samples, (
        f"Merge changed row count: {len(df)} != {n_samples}")

    # Collapse rare TSS sites (same logic as main pipeline)
    counts = df['tss'].value_counts()
    rare = counts[counts < MIN_SAMPLES_PER_TSS].index
    df['tss_collapsed'] = df['tss'].copy()
    df.loc[df['tss_collapsed'].isin(rare), 'tss_collapsed'] = 'Other'
    n_groups = df['tss_collapsed'].nunique()
    print(f"TSS groups after collapsing rare sites: {n_groups}")

    # Shared metadata vectors (aligned by pid sort order, length = n_samples)
    tss_groups = df['tss_collapsed'].reset_index(drop=True)
    her2_groups = df['her2_composite'].reset_index(drop=True)

    # -- 1. Uncorrected baseline (UQ, no TSS correction) -------------------------
    print("\n" + "-" * 70)
    print("1. UNCORRECTED (UQ, no TSS correction)")
    print("-" * 70)
    uncorr_expr = tumor_uq[['pid'] + gene_cols].copy().reset_index(drop=True)
    m_uncorr, pcs_uncorr, pca_uncorr = run_pca_metrics(
        uncorr_expr, gene_cols, tss_groups, her2_groups,
        "Uncorrected (UQ)"
    )

    # -- 2. OLS regression -- loaded from saved pipeline output ------------------
    print("\n" + "-" * 70)
    print("2. OLS REGRESSION (pipeline output: 01_tumor_norm_uq_tss)")
    print("-" * 70)
    ols_expr = uq_tss_dedup[['pid'] + gene_cols].copy().reset_index(drop=True)
    m_ols, pcs_ols, pca_ols = run_pca_metrics(
        ols_expr, gene_cols, tss_groups, her2_groups,
        "OLS Regression (UQ-TSS)"
    )

    # -- 3. TMM-TSS (canonical pipeline output, for reference) -------------------
    print("\n" + "-" * 70)
    print("3. TMM-TSS (canonical pipeline output: 01_tumor_norm_tmm_tss)")
    print("-" * 70)
    tmm_expr = tmm_tss_aligned[['pid'] + gene_cols].copy().reset_index(drop=True)
    m_tmm, pcs_tmm, pca_tmm = run_pca_metrics(
        tmm_expr, gene_cols, tss_groups, her2_groups,
        "OLS Regression (TMM-TSS)"
    )

    # -- 4. ComBat with biological covariates ------------------------------------
    print("\n" + "-" * 70)
    print("4. COMBAT (with HER2 + ER covariates, applied to UQ)")
    print("-" * 70)
    try:
        combat_corrected = combat_correction(df, gene_cols)
        combat_corrected = combat_corrected.reset_index(drop=True)
        m_combat, pcs_combat, pca_combat = run_pca_metrics(
            combat_corrected, gene_cols, tss_groups, her2_groups,
            "ComBat + covariates (UQ)"
        )
    except Exception as e:
        print(f"  ComBat with covariates FAILED: {e}")
        m_combat = None
        pcs_combat = None
        pca_combat = None

    # -- 5. ComBat without covariates (naive) ------------------------------------
    print("\n" + "-" * 70)
    print("5. COMBAT (no covariates -- naive, applied to UQ)")
    print("-" * 70)
    try:
        combat_naive = combat_no_covariates(df, gene_cols)
        combat_naive = combat_naive.reset_index(drop=True)
        m_naive, pcs_naive, pca_naive = run_pca_metrics(
            combat_naive, gene_cols, tss_groups, her2_groups,
            "ComBat naive (UQ)"
        )
    except Exception as e:
        print(f"  ComBat naive FAILED: {e}")
        m_naive = None
        pcs_naive = None
        pca_naive = None

    # -- Summary table -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY (all on same %d samples, %d genes)" % (n_samples, len(gene_cols)))
    print("=" * 70)

    rows = [m_uncorr, m_ols, m_tmm]
    expr_dfs = [uncorr_expr, ols_expr, tmm_expr]
    if m_combat is not None:
        rows.append(m_combat)
        expr_dfs.append(combat_corrected)
    if m_naive is not None:
        rows.append(m_naive)
        expr_dfs.append(combat_naive)

    summary = pd.DataFrame(rows)
    summary = summary.set_index('method')

    print("\n" + summary.to_string(float_format='{:.4f}'.format))

    # Decision metrics:
    #   Primary:   TSS eta2 on PC1 (lower = less residual batch in dominant axis)
    #   Guardrail: ERBB2 Cohen's d (must not drop below uncorrected baseline)
    corrected_only = summary.iloc[1:]  # skip uncorrected
    uncorr_d = m_uncorr['erbb2_cohens_d']

    print("\n--- Primary Metric: TSS eta2 on PC1 (lower = better batch removal) ---")
    print(f"--- Guardrail: ERBB2 Cohen's d >= {uncorr_d:.2f} (uncorrected baseline) ---")
    best_eta2 = corrected_only['pc1_tss_eta2'].min()
    for _, row in summary.iterrows():
        is_corrected = row.name != summary.index[0]
        is_best = is_corrected and row['pc1_tss_eta2'] == best_eta2
        signal_ok = row['erbb2_cohens_d'] >= uncorr_d
        flag = ""
        if is_best and signal_ok:
            flag = " <-- BEST"
        elif is_best and not signal_ok:
            flag = " <-- lowest eta2 but FAILS guardrail"
        elif not signal_ok and is_corrected:
            flag = " (FAILS guardrail)"
        print(f"  {row.name:<30s}  eta2 = {row['pc1_tss_eta2']:.4f}"
              f"  ERBB2 d = {row['erbb2_cohens_d']:.4f}"
              f"  |r_depth| = {abs(row['pc1_depth_r']):.4f}{flag}")

    # Identify best method that passes the guardrail
    passing = corrected_only[corrected_only['erbb2_cohens_d'] >= uncorr_d]
    if not passing.empty:
        best = passing['pc1_tss_eta2'].idxmin()
        print(f"\n  >> Best corrected method (lowest eta2, passes guardrail): {best}")
    else:
        print("\n  >> WARNING: No corrected method passes the ERBB2 guardrail.")

    save_intermediate(summary.reset_index(), "01s_combat_comparison")

    # -- Figure ------------------------------------------------------------------
    print("\nGenerating comparison figure...")

    all_methods = [
        ("Uncorrected (UQ)", pcs_uncorr, pca_uncorr, uncorr_expr),
        ("OLS (UQ-TSS)", pcs_ols, pca_ols, ols_expr),
        ("OLS (TMM-TSS)", pcs_tmm, pca_tmm, tmm_expr),
    ]
    if pcs_combat is not None:
        all_methods.append(("ComBat+cov (UQ)", pcs_combat, pca_combat, combat_corrected))
    if pcs_naive is not None:
        all_methods.append(("ComBat naive (UQ)", pcs_naive, pca_naive, combat_naive))

    n_methods = len(all_methods)
    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 10))
    if n_methods == 1:
        axes = axes.reshape(2, 1)

    # Color maps
    her2_cmap = {'Positive': '#e74c3c', 'Negative': '#3498db',
                 'Equivocal': '#f39c12'}
    her2_colors = [her2_cmap.get(h, '#95a5a6') for h in her2_groups.values]

    for col, (name, pcs, pca_obj, expr_data) in enumerate(all_methods):
        var1 = pca_obj.explained_variance_ratio_[0]
        var2 = pca_obj.explained_variance_ratio_[1]

        # Top row: PCA colored by HER2
        axes[0, col].scatter(pcs[:, 0], pcs[:, 1], c=her2_colors, s=12, alpha=0.5)
        axes[0, col].set_xlabel(f"PC1 ({var1:.1%})")
        axes[0, col].set_ylabel(f"PC2 ({var2:.1%})")
        axes[0, col].set_title(f"{name}\n(colored by HER2)")

        # Bottom row: PC1 vs depth proxy from the same corrected expression
        dp = read_depth_proxy(expr_data, gene_cols)
        r_val, _ = stats.pearsonr(pcs[:, 0], dp)
        axes[1, col].scatter(dp, pcs[:, 0], c=her2_colors, s=12, alpha=0.5)
        axes[1, col].set_xlabel("Read-Depth Proxy\n(sum top-3k MAD genes)")
        axes[1, col].set_ylabel("PC1")
        axes[1, col].set_title(f"PC1 vs Depth (r = {r_val:.3f})")

    fig.suptitle(
        "TSS Batch Correction: ComBat vs OLS Regression\n"
        "Top: PCA by HER2 status | Bottom: PC1 vs read-depth proxy (secondary metric)",
        fontsize=13, fontweight='bold', y=1.02
    )
    fig.tight_layout()
    savefig(fig, "fig_01s_combat_vs_regression")
    plt.close(fig)

    print("\n" + "=" * 70)
    print("DONE -- see outputs/01s_combat_comparison.parquet")
    print("        and outputs/figures/fig_01s_combat_vs_regression.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
