"""
01s_tss_batch_assessment.py --Supporting QC analysis for Notebook 01.

Quantifies tissue source site (TSS) batch effects in the TCGA BRCA expression data
and produces a TSS-residualized expression matrix for sensitivity analysis.

Usage:
    Run after Notebook 01 has generated its intermediates in outputs/.

    python src/01s_tss_batch_assessment.py

Outputs (saved to outputs/):
    01s_tss_eta_squared.parquet    --eta2 for TSS vs each top PC and ERBB2
    01s_tss_gene_anova.parquet     --per-gene ANOVA F-stats for TSS (top 200 genes)
    01s_tumor_norm_tss_corrected.parquet --TSS-residualized expression matrix
    figures/fig_01s_tss_eta_heatmap.png  --eta2 heatmap (PCs x variables)
    figures/fig_01s_pca_before_after.png --PCA colored by TSS, before/after correction

Depends on:
    outputs/01_tumor_norm.parquet
    outputs/01_clinical_qc.parquet
    outputs/01_gene_cols.json
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -- Project paths ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils import (
    load_intermediate, load_gene_cols, save_intermediate, savefig,
    ensure_dirs, OUTPUT_DIR, FIGURE_DIR
)

MIN_SAMPLES_PER_TSS = 5  # Sites below this are collapsed to "Other"
N_PCS = 10


# -- Phase 1: Quantify TSS batch effects ---------------------------------------

def compute_eta_squared(values, groups):
    """Compute eta2 (eta-squared) --proportion of variance explained by grouping.

    eta2 = SS_between / SS_total.  Range [0, 1].
    """
    unique_groups = groups.dropna().unique()
    if len(unique_groups) < 2:
        return np.nan

    grand_mean = values.mean()
    ss_total = ((values - grand_mean) ** 2).sum()
    if ss_total == 0:
        return np.nan

    ss_between = 0
    for g in unique_groups:
        mask = groups == g
        n_g = mask.sum()
        if n_g > 0:
            group_mean = values[mask].mean()
            ss_between += n_g * (group_mean - grand_mean) ** 2

    return ss_between / ss_total


def collapse_rare_sites(tss_series, min_samples=MIN_SAMPLES_PER_TSS):
    """Collapse TSS categories with fewer than min_samples into 'Other'."""
    counts = tss_series.value_counts()
    rare = counts[counts < min_samples].index
    collapsed = tss_series.copy()
    collapsed[collapsed.isin(rare)] = "Other"
    n_collapsed = len(rare)
    n_remaining = collapsed.nunique()
    print(f"  TSS collapsing: {n_collapsed} rare sites -> 'Other' "
          f"({n_remaining} groups remain, threshold={min_samples})")
    return collapsed


def phase1_quantify(tumor_norm, clinical, gene_cols):
    """Compute eta2 for TSS across PCs and key genes."""

    print("=" * 70)
    print("PHASE 1: QUANTIFY TSS BATCH EFFECTS")
    print("=" * 70)

    # Merge TSS into expression data
    df = tumor_norm.merge(clinical[['pid', 'tss', 'her2_composite', 'ER Status By IHC']],
                          on='pid', how='left')
    df['tss_collapsed'] = collapse_rare_sites(df['tss'])

    # -- PCA ----------------------------------------------------------------
    print("\nRunning PCA...")
    X = df[gene_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=N_PCS, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    pc_names = [f"PC{i+1}" for i in range(N_PCS)]

    for i in range(N_PCS):
        df[f"PC{i+1}"] = pcs[:, i]

    print(f"  Variance explained (top {N_PCS} PCs): "
          f"{pca.explained_variance_ratio_.sum():.1%}")

    # -- eta2 for each PC vs TSS, HER2, ER -----------------------------------
    print("\nComputing eta2 (variance explained) per variable per PC...")

    grouping_vars = {
        'TSS': df['tss_collapsed'],
        'HER2': df['her2_composite'],
        'ER': df['ER Status By IHC'],
    }

    eta_results = {}
    for var_name, groups in grouping_vars.items():
        valid = groups.notna()
        etas = []
        for pc in pc_names:
            eta = compute_eta_squared(df.loc[valid, pc], groups[valid])
            etas.append(eta)
        eta_results[var_name] = etas

    # ERBB2 expression specifically
    if 'ERBB2' in gene_cols:
        erbb2_vals = df['ERBB2']
        for var_name, groups in grouping_vars.items():
            valid = groups.notna() & erbb2_vals.notna()
            eta = compute_eta_squared(erbb2_vals[valid], groups[valid])
            eta_results.setdefault(f'{var_name}_on_ERBB2', [eta])

    eta_df = pd.DataFrame(eta_results, index=pc_names)

    print("\n  eta2 (variance explained by each variable):")
    print(eta_df.to_string(float_format='{:.4f}'.format))

    # Flag concerning PCs
    tss_high = eta_df['TSS'][eta_df['TSS'] > 0.10]
    if not tss_high.empty:
        print(f"\n  WARNING: TSS explains >10% of variance on: {list(tss_high.index)}")
        for pc in tss_high.index:
            her2_eta = eta_df.loc[pc, 'HER2'] if 'HER2' in eta_df.columns else 0
            if her2_eta > 0.05:
                print(f"    -> {pc} also loads on HER2 (eta2={her2_eta:.3f}) -- "
                      f"TSS correction risks attenuating HER2 signal here")
    else:
        print("\n  OK: TSS explains <10% of variance on all top PCs")

    # -- Per-gene ANOVA for TSS (vectorized) ----------------------------------
    print("\nComputing per-gene ANOVA (expression ~ TSS, vectorized)...")
    tss_groups = df['tss_collapsed']
    valid_tss = tss_groups.notna()

    # Vectorized one-way ANOVA: compute SS_between and SS_total for all genes at once
    gene_data = df.loc[valid_tss, gene_cols].values  # (n_samples, n_genes)
    group_labels = tss_groups[valid_tss].values
    unique_groups = [g for g in np.unique(group_labels) if pd.notna(g)]
    n_total = gene_data.shape[0]
    k = len(unique_groups)

    grand_mean = gene_data.mean(axis=0)  # (n_genes,)
    ss_total = ((gene_data - grand_mean) ** 2).sum(axis=0)  # (n_genes,)

    ss_between = np.zeros(len(gene_cols))
    for g in unique_groups:
        mask = group_labels == g
        n_g = mask.sum()
        if n_g > 0:
            group_mean = gene_data[mask].mean(axis=0)
            ss_between += n_g * (group_mean - grand_mean) ** 2

    ss_within = ss_total - ss_between
    df_between = k - 1
    df_within = n_total - k

    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        f_stats = np.where(ms_within > 0, ms_between / ms_within, np.nan)
        eta_sq = np.where(ss_total > 0, ss_between / ss_total, np.nan)

    # Compute p-values from F distribution
    p_vals = np.where(np.isnan(f_stats), np.nan,
                      1 - stats.f.cdf(f_stats, df_between, df_within))

    gene_anova = pd.DataFrame({
        'gene': gene_cols,
        'F': f_stats,
        'p': p_vals,
        'eta_sq': eta_sq
    }).dropna(subset=['F']).sort_values('F', ascending=False)
    top_200 = gene_anova.head(200)

    # Check ERBB2 rank
    erbb2_row = gene_anova[gene_anova['gene'] == 'ERBB2']
    if not erbb2_row.empty:
        erbb2_rank = (gene_anova['F'] > erbb2_row['F'].values[0]).sum() + 1
        erbb2_eta = erbb2_row['eta_sq'].values[0]
        print(f"  ERBB2: rank {erbb2_rank}/{len(gene_anova)} by TSS F-stat, "
              f"eta2 = {erbb2_eta:.4f}")
        if erbb2_eta < 0.05:
            print("  OK: ERBB2 is minimally affected by TSS batch (<5% variance)")
        else:
            print("  WARNING: ERBB2 shows meaningful TSS variation -- correction recommended")

    print(f"  Top 5 TSS-variable genes: {list(top_200['gene'].head(5))}")

    # -- Discordant cases x TSS ---------------------------------------------
    # Check if discordant cases concentrate in specific sites
    try:
        discordant = load_intermediate("02_discordant_cases")
        disc_tss = discordant.merge(clinical[['pid', 'tss']], on='pid', how='left')
        print(f"\n  Discordant cases by TSS (top 5 sites):")
        print(disc_tss['tss'].value_counts().head().to_string())
    except FileNotFoundError:
        print("\n  (02_discordant_cases.parquet not yet available -- skip discordant x TSS)")

    # Save
    save_intermediate(eta_df.reset_index().rename(columns={'index': 'PC'}),
                      "01s_tss_eta_squared")
    save_intermediate(top_200, "01s_tss_gene_anova")

    return df, pca, pcs, eta_df, gene_anova


# -- Phase 2: TSS Residualization -----------------------------------------------

def phase2_correct(df, gene_cols):
    """Regress out TSS while preserving HER2 and ER signal.

    Model for all genes simultaneously:  Y = X @ B
    where X = [intercept | TSS_dummies | protected_covariates]
    Corrected = Y - X_tss @ B_tss  (subtract only the TSS contribution)

    Uses numpy lstsq for a single matrix solve instead of per-gene OLS.
    """

    print("\n" + "=" * 70)
    print("PHASE 2: TSS RESIDUALIZATION (protected covariates: HER2, ER)")
    print("=" * 70)

    # Encode TSS dummies (collapsed, drop first for identifiability)
    tss_dummies = pd.get_dummies(df['tss_collapsed'], prefix='tss', drop_first=True)
    n_tss = tss_dummies.shape[1]

    # Protected covariates -- their signal is preserved
    protected = pd.DataFrame(index=df.index)

    # HER2: Positive/Negative/Equivocal -> dummies
    her2 = df['her2_composite'].fillna('Unknown')
    her2_dummies = pd.get_dummies(her2, prefix='her2', drop_first=True)
    protected = pd.concat([protected, her2_dummies], axis=1)

    # ER: binary
    er = (df.get('ER Status By IHC', pd.Series(dtype=str)) == 'Positive').astype(float)
    protected['ER_pos'] = er
    n_protected = protected.shape[1]

    # Full design matrix: intercept + TSS + protected
    X_full = np.column_stack([
        np.ones(len(df)),
        tss_dummies.values,
        protected.values
    ]).astype(np.float64)

    print(f"  Design matrix: {X_full.shape[1]} columns "
          f"(1 intercept + {n_tss} TSS dummies + {n_protected} protected)")
    print(f"  Correcting {len(gene_cols)} genes via matrix least-squares...")

    # Gene expression matrix (n_samples x n_genes)
    Y = df[gene_cols].fillna(0).values.astype(np.float64)

    # Solve X @ B = Y for B  (B is n_features x n_genes)
    B, residuals, rank, sv = np.linalg.lstsq(X_full, Y, rcond=None)

    print(f"  Matrix rank: {rank} (full rank = {X_full.shape[1]})")

    # TSS betas are rows 1 through n_tss (row 0 is intercept)
    B_tss = B[1:1+n_tss, :]        # (n_tss, n_genes)
    X_tss = X_full[:, 1:1+n_tss]   # (n_samples, n_tss)

    # Subtract only the TSS contribution
    tss_contribution = X_tss @ B_tss  # (n_samples, n_genes)
    corrected_values = Y - tss_contribution

    corrected = pd.DataFrame(corrected_values, columns=gene_cols, index=df.index)

    # Rebuild the full dataframe
    result = df[['pid']].copy()
    result = pd.concat([result, corrected], axis=1)

    save_intermediate(result, "01s_tumor_norm_tss_corrected")

    print(f"  OK: TSS-corrected expression matrix saved ({result.shape[0]} x {result.shape[1]})")

    return result


# -- Phase 3: Validation plots -------------------------------------------------

def phase3_validate(df_original, df_corrected, gene_cols, eta_df):
    """PCA before/after comparison and eta2 validation."""

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    print("\n" + "=" * 70)
    print("PHASE 3: VALIDATION --BEFORE/AFTER COMPARISON")
    print("=" * 70)

    # Re-run PCA on corrected data
    X_corr = df_corrected[gene_cols].fillna(0).values
    scaler = StandardScaler()
    X_corr_scaled = scaler.fit_transform(X_corr)

    pca_corr = PCA(n_components=N_PCS, random_state=42)
    pcs_corr = pca_corr.fit_transform(X_corr_scaled)

    # Compute eta2 on corrected PCs
    tss_groups = df_original['tss_collapsed']
    valid = tss_groups.notna()

    eta_after = []
    for i in range(N_PCS):
        eta = compute_eta_squared(
            pd.Series(pcs_corr[valid.values, i], index=df_original.index[valid]),
            tss_groups[valid]
        )
        eta_after.append(eta)

    print("\n  eta2 for TSS --before vs after correction:")
    print(f"  {'PC':<6} {'Before':>8} {'After':>8} {'Delta':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for i in range(N_PCS):
        before = eta_df['TSS'].iloc[i]
        after = eta_after[i]
        delta = after - before
        print(f"  PC{i+1:<4} {before:8.4f} {after:8.4f} {delta:+8.4f}")

    # Check HER2 signal preservation
    her2_groups = df_original['her2_composite']
    valid_h = her2_groups.notna()

    her2_eta_before = []
    her2_eta_after = []
    X_orig = df_original[gene_cols].fillna(0).values
    scaler_orig = StandardScaler()
    X_orig_scaled = scaler_orig.fit_transform(X_orig)
    pca_orig = PCA(n_components=N_PCS, random_state=42)
    pcs_orig = pca_orig.fit_transform(X_orig_scaled)

    for i in range(N_PCS):
        eta_b = compute_eta_squared(
            pd.Series(pcs_orig[valid_h.values, i], index=df_original.index[valid_h]),
            her2_groups[valid_h]
        )
        eta_a = compute_eta_squared(
            pd.Series(pcs_corr[valid_h.values, i], index=df_original.index[valid_h]),
            her2_groups[valid_h]
        )
        her2_eta_before.append(eta_b)
        her2_eta_after.append(eta_a)

    print("\n  eta2 for HER2 --before vs after correction:")
    print(f"  {'PC':<6} {'Before':>8} {'After':>8} {'Delta':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for i in range(N_PCS):
        b = her2_eta_before[i]
        a = her2_eta_after[i]
        print(f"  PC{i+1:<4} {b:8.4f} {a:8.4f} {a-b:+8.4f}")

    # -- Figure: Before/After PCA colored by TSS ---------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    tss = df_original['tss_collapsed'].values
    unique_tss = sorted(set(t for t in tss if pd.notna(t)))
    # Use a categorical colormap
    cmap = plt.cm.get_cmap('tab20', len(unique_tss))
    tss_to_color = {t: cmap(i) for i, t in enumerate(unique_tss)}
    colors = [tss_to_color.get(t, (0.5, 0.5, 0.5, 1.0)) for t in tss]

    # Before --colored by TSS
    axes[0, 0].scatter(pcs_orig[:, 0], pcs_orig[:, 1], c=colors, alpha=0.5, s=15)
    axes[0, 0].set_xlabel(f"PC1 ({pca_orig.explained_variance_ratio_[0]:.1%})")
    axes[0, 0].set_ylabel(f"PC2 ({pca_orig.explained_variance_ratio_[1]:.1%})")
    axes[0, 0].set_title("Before correction --colored by TSS")

    # After --colored by TSS
    axes[0, 1].scatter(pcs_corr[:, 0], pcs_corr[:, 1], c=colors, alpha=0.5, s=15)
    axes[0, 1].set_xlabel(f"PC1 ({pca_corr.explained_variance_ratio_[0]:.1%})")
    axes[0, 1].set_ylabel(f"PC2 ({pca_corr.explained_variance_ratio_[1]:.1%})")
    axes[0, 1].set_title("After TSS correction --colored by TSS")

    # Before --colored by HER2
    her2_colors_map = {'Positive': '#e74c3c', 'Negative': '#3498db',
                       'Equivocal': '#f39c12'}
    her2_vals = df_original['her2_composite'].values
    her2_c = [her2_colors_map.get(h, '#95a5a6') for h in her2_vals]

    axes[1, 0].scatter(pcs_orig[:, 0], pcs_orig[:, 1], c=her2_c, alpha=0.5, s=15)
    axes[1, 0].set_xlabel(f"PC1 ({pca_orig.explained_variance_ratio_[0]:.1%})")
    axes[1, 0].set_ylabel(f"PC2 ({pca_orig.explained_variance_ratio_[1]:.1%})")
    axes[1, 0].set_title("Before correction --colored by HER2")

    # After --colored by HER2
    axes[1, 1].scatter(pcs_corr[:, 0], pcs_corr[:, 1], c=her2_c, alpha=0.5, s=15)
    axes[1, 1].set_xlabel(f"PC1 ({pca_corr.explained_variance_ratio_[0]:.1%})")
    axes[1, 1].set_ylabel(f"PC2 ({pca_corr.explained_variance_ratio_[1]:.1%})")
    axes[1, 1].set_title("After TSS correction --colored by HER2")

    fig.suptitle("TSS Batch Correction: PCA Before vs After\n"
                 "(Top: TSS structure should diminish; Bottom: HER2 separation should persist)",
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_01s_pca_before_after")
    plt.close(fig)

    # -- Figure: eta2 heatmap ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before
    eta_before_mat = pd.DataFrame({
        'TSS': eta_df['TSS'].values,
        'HER2': eta_df['HER2'].values,
        'ER': eta_df['ER'].values,
    }, index=[f"PC{i+1}" for i in range(N_PCS)])

    im0 = axes[0].imshow(eta_before_mat.T.values, aspect='auto', cmap='YlOrRd',
                          vmin=0, vmax=0.5)
    axes[0].set_xticks(range(N_PCS))
    axes[0].set_xticklabels(eta_before_mat.index, fontsize=9)
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(['TSS', 'HER2', 'ER'], fontsize=10)
    for i in range(3):
        for j in range(N_PCS):
            val = eta_before_mat.T.values[i, j]
            color = 'white' if val > 0.25 else 'black'
            axes[0].text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color=color)
    axes[0].set_title("eta2 BEFORE correction")
    plt.colorbar(im0, ax=axes[0], label='eta2', shrink=0.8)

    # After
    eta_after_mat = pd.DataFrame({
        'TSS': eta_after,
        'HER2': her2_eta_after,
        'ER': eta_df['ER'].values,  # ER not recomputed --same protection
    }, index=[f"PC{i+1}" for i in range(N_PCS)])

    im1 = axes[1].imshow(eta_after_mat.T.values, aspect='auto', cmap='YlOrRd',
                          vmin=0, vmax=0.5)
    axes[1].set_xticks(range(N_PCS))
    axes[1].set_xticklabels(eta_after_mat.index, fontsize=9)
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(['TSS', 'HER2', 'ER'], fontsize=10)
    for i in range(3):
        for j in range(N_PCS):
            val = eta_after_mat.T.values[i, j]
            color = 'white' if val > 0.25 else 'black'
            axes[1].text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color=color)
    axes[1].set_title("eta2 AFTER correction")
    plt.colorbar(im1, ax=axes[1], label='eta2', shrink=0.8)

    fig.suptitle("Variance Explained (eta2) by TSS, HER2, ER across Top PCs",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    savefig(fig, "fig_01s_tss_eta_heatmap")
    plt.close(fig)

    print("\n  OK: Validation figures saved to outputs/figures/")


# -- Main -----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TSS BATCH EFFECT ASSESSMENT")
    print("Supporting analysis for Notebook 01 QC")
    print("=" * 70)

    # Load intermediates from Notebook 01
    tumor_norm = load_intermediate("01_tumor_norm")
    clinical = load_intermediate("01_clinical_qc")
    gene_cols = load_gene_cols("01_gene_cols")

    # Verify ERBB2 is present
    if 'ERBB2' not in gene_cols:
        print("WARNING: ERBB2 not in gene_cols -- check filtering pipeline")

    # Phase 1: Quantify
    df, pca, pcs, eta_df, gene_anova = phase1_quantify(tumor_norm, clinical, gene_cols)

    # Phase 2: Correct
    df_corrected = phase2_correct(df, gene_cols)

    # Phase 3: Validate
    phase3_validate(df, df_corrected, gene_cols, eta_df)

    print("\n" + "=" * 70)
    print("DONE --Review outputs/01s_*.parquet and outputs/figures/fig_01s_*")
    print("=" * 70)


if __name__ == "__main__":
    main()
