"""
Shared analysis pipeline for normalization comparison (Phases 2/3).

Accepts a normalized expression matrix and runs identical analyses across
all normalization methods: Sections A-E matching the report template in
skills/user/shared/normalization_comparison_plan.md.

Usage:
    from scripts.normalization_comparison.analysis_pipeline import run_analysis
    metrics = run_analysis(expr_df, clinical, cn, gene_cols,
                           method_name='tpm', output_dir=Path(...))
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Section A: ERBB2 RNA vs Copy Number
# ---------------------------------------------------------------------------

def section_a(cohort_df):
    """
    cohort_df must contain: ERBB2_expr, erbb2_copy_number, her2_composite.
    Returns dict with keys A1..A7.
    """
    r = {}

    labeled = cohort_df[cohort_df['her2_composite'].isin(['Positive', 'Negative'])].copy()
    pos = labeled[labeled['her2_composite'] == 'Positive']
    neg = labeled[labeled['her2_composite'] == 'Negative']

    pos_expr = pos['ERBB2_expr'].dropna()
    neg_expr = neg['ERBB2_expr'].dropna()
    all_expr = labeled['ERBB2_expr'].dropna()
    all_cn = labeled.loc[all_expr.index, 'erbb2_copy_number']

    r['A1'], _ = stats.pearsonr(all_expr, all_cn)
    r['A2'], _ = stats.spearmanr(all_expr, all_cn)

    pos_both = pos.dropna(subset=['ERBB2_expr', 'erbb2_copy_number'])
    if len(pos_both) > 2:
        r['A3'], _ = stats.pearsonr(pos_both['ERBB2_expr'], pos_both['erbb2_copy_number'])
    else:
        r['A3'] = float('nan')

    neg_both = neg.dropna(subset=['ERBB2_expr', 'erbb2_copy_number'])
    if len(neg_both) > 2:
        r['A4'], _ = stats.pearsonr(neg_both['ERBB2_expr'], neg_both['erbb2_copy_number'])
    else:
        r['A4'] = float('nan')

    pooled_std = np.sqrt(
        ((len(pos_expr) - 1) * pos_expr.std() ** 2 + (len(neg_expr) - 1) * neg_expr.std() ** 2)
        / (len(pos_expr) + len(neg_expr) - 2)
    )
    r['A5'] = (pos_expr.mean() - neg_expr.mean()) / pooled_std

    _, r['A6'] = stats.mannwhitneyu(pos_expr, neg_expr, alternative='two-sided')

    denom = neg_expr.median()
    r['A7'] = pos_expr.median() / denom if denom != 0 else float('nan')

    for k, v in r.items():
        print(f"  {k}: {v:.4f}" if k != 'A6' else f"  {k}: {v:.2e}")
    return r


# ---------------------------------------------------------------------------
# Section B: Logistic Regression
# ---------------------------------------------------------------------------

def section_b(cohort_df):
    """
    cohort_df must contain: ERBB2_expr, erbb2_copy_number, her2_composite.
    Returns dict with keys B1..B7.
    """
    r = {}

    model_df = cohort_df.dropna(subset=['her2_composite', 'erbb2_copy_number', 'ERBB2_expr'])
    model_df = model_df[model_df['her2_composite'].isin(['Positive', 'Negative'])].copy()
    model_df['y'] = (model_df['her2_composite'] == 'Positive').astype(int)

    X_rna = model_df[['ERBB2_expr']].values
    X_cn = model_df[['erbb2_copy_number']].values
    X_both = model_df[['ERBB2_expr', 'erbb2_copy_number']].values
    y = model_df['y'].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_kwargs = dict(class_weight='balanced', random_state=42, max_iter=1000)

    keys = [('ERBB2 RNA only', X_rna, 'B1', 'B2'),
            ('ERBB2 CN only', X_cn, 'B3', 'B4'),
            ('RNA + CN combined', X_both, 'B5', 'B6')]

    y_probs = {}
    for name, X, roc_key, pr_key in keys:
        X_s = StandardScaler().fit_transform(X)
        y_prob = cross_val_predict(LogisticRegression(**lr_kwargs), X_s, y,
                                   cv=cv, method='predict_proba')[:, 1]
        r[roc_key] = roc_auc_score(y, y_prob)
        r[pr_key] = average_precision_score(y, y_prob)
        y_probs[name] = y_prob

    r['B7'] = r['B1'] - r['B3']

    for k, v in r.items():
        print(f"  {k}: {v:.4f}")
    return r, {'y_true': y, 'y_probs': y_probs}


# ---------------------------------------------------------------------------
# Section C: Unsupervised Clustering
# ---------------------------------------------------------------------------

def section_c(expr_df, clinical, gene_cols):
    """
    Runs PCA + KMeans on top 3000 MAD genes.
    Returns (metrics_dict, k4_labels, patient_ids, X_pca).
    """
    r = {}

    gene_mad = expr_df[gene_cols].apply(
        lambda x: np.median(np.abs(x - np.median(x))), axis=0
    )
    top_genes = gene_mad.nlargest(min(3000, len(gene_cols))).index.tolist()

    X_cluster = expr_df[top_genes].fillna(0).values
    patient_ids = expr_df['pid'].values

    X_scaled = StandardScaler().fit_transform(X_cluster)
    n_pcs = min(20, *X_scaled.shape)
    X_pca = PCA(n_components=n_pcs, random_state=42).fit_transform(X_scaled)
    X_for_cl = X_pca[:, :min(10, X_pca.shape[1])]

    # Silhouette for k=2..7
    sil_scores = {}
    for k in range(2, 8):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_for_cl)
        sil_scores[k] = silhouette_score(X_for_cl, labels)
    r['C1'] = sil_scores
    r['C2'] = max(sil_scores, key=sil_scores.get)
    print(f"  C1: {sil_scores}")
    print(f"  C2 best k: {r['C2']}")

    k4_labels = KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(X_for_cl)
    r['C3'] = silhouette_score(X_for_cl, k4_labels)
    print(f"  C3 silhouette k=4: {r['C3']:.4f}")

    # ARI at k=4
    pid_to_her2 = (clinical.drop_duplicates('pid')
                   .set_index('pid')['her2_composite'].to_dict())
    pid_to_er = (clinical.drop_duplicates('pid')
                 .set_index('pid')['ER Status By IHC'].to_dict())

    her2_vals = [pid_to_her2.get(p) for p in patient_ids]
    er_vals = [pid_to_er.get(p) for p in patient_ids]

    her2_valid = [(k4_labels[i], her2_vals[i])
                  for i in range(len(k4_labels))
                  if her2_vals[i] in ('Positive', 'Negative')]
    r['C4'] = (adjusted_rand_score(*zip(*her2_valid))
               if her2_valid else float('nan'))

    er_valid = [(k4_labels[i], er_vals[i])
                for i in range(len(k4_labels))
                if er_vals[i] in ('Positive', 'Negative')]
    r['C5'] = (adjusted_rand_score(*zip(*er_valid))
               if er_valid else float('nan'))

    print(f"  C4 ARI HER2: {r['C4']:.4f}")
    print(f"  C5 ARI ER:   {r['C5']:.4f}")

    return r, k4_labels, patient_ids, X_pca


# ---------------------------------------------------------------------------
# Section D: Subtype Marker Separation
# ---------------------------------------------------------------------------

_MARKER_GENES = {
    'ESR1': 'luminal', 'PGR': 'luminal', 'GATA3': 'luminal', 'FOXA1': 'luminal',
    'ERBB2': 'her2', 'GRB7': 'her2', 'STARD3': 'her2',
    'KRT5': 'basal', 'KRT14': 'basal', 'KRT17': 'basal', 'EGFR': 'basal',
    'MKI67': 'prolif', 'CCNB1': 'prolif', 'AURKA': 'prolif', 'TOP2A': 'prolif',
}


def section_d(expr_df, clinical, k4_labels, patient_ids):
    """
    Returns (metrics_dict, subtype_map dict {cluster -> name}).
    """
    r = {}

    char_df = pd.DataFrame({'pid': patient_ids, 'cluster': k4_labels})
    expr_map = expr_df.drop_duplicates(subset='pid').set_index('pid')
    clin_map = clinical.drop_duplicates('pid').set_index('pid')

    for gene in _MARKER_GENES:
        if gene in expr_map.columns:
            char_df[gene] = char_df['pid'].map(expr_map[gene])

    for col in ['ER Status By IHC', 'her2_composite']:
        if col in clin_map.columns:
            char_df[col] = char_df['pid'].map(clin_map[col])

    score_df = _compute_subtype_scores(char_df)

    for panel, key in [('luminal', 'D1'), ('her2', 'D2'), ('basal', 'D3')]:
        if panel in score_df.columns:
            r[key] = score_df[panel].max() - score_df[panel].min()
        else:
            r[key] = float('nan')
    r['D4'] = np.mean([r['D1'], r['D2'], r['D3']])

    print(f"  D1 luminal spread:  {r['D1']:.4f}")
    print(f"  D2 HER2 spread:     {r['D2']:.4f}")
    print(f"  D3 basal spread:    {r['D3']:.4f}")
    print(f"  D4 mean gap:        {r['D4']:.4f}")

    subtype_map = _assign_subtypes(score_df)
    print(f"  Subtype assignments: {subtype_map}")

    her2_enr = [c for c, s in subtype_map.items() if s == 'HER2-enriched']
    if her2_enr:
        her2_pos = char_df[char_df['her2_composite'] == 'Positive']
        total = len(her2_pos)
        in_enr = her2_pos['cluster'].isin(her2_enr).sum()
        r['D5'] = in_enr / max(total, 1)
    else:
        r['D5'] = float('nan')
    print(f"  D5 HER2+ in HER2-enriched: {r['D5']:.4f}")

    return r, subtype_map, score_df, char_df


def _compute_subtype_scores(char_df):
    score_rows = {}
    for c in sorted(char_df['cluster'].unique()):
        cl = char_df[char_df['cluster'] == c]
        row = {}
        for panel in ['luminal', 'her2', 'basal', 'prolif']:
            genes = [g for g, p in _MARKER_GENES.items() if p == panel and g in cl.columns]
            if genes:
                row[panel] = cl[genes].median().mean()
        er_pos = (cl.get('ER Status By IHC', pd.Series()) == 'Positive').sum()
        er_tot = cl.get('ER Status By IHC', pd.Series()).isin(['Positive', 'Negative']).sum()
        her2_pos = (cl.get('her2_composite', pd.Series()) == 'Positive').sum()
        her2_tot = cl.get('her2_composite', pd.Series()).isin(['Positive', 'Negative']).sum()
        row['er_pct'] = er_pos / max(er_tot, 1)
        row['her2_pct'] = her2_pos / max(her2_tot, 1)
        score_rows[c] = row
    score_df = pd.DataFrame(score_rows).T

    for panel in ['luminal', 'her2', 'basal', 'prolif']:
        if panel in score_df.columns:
            mu = score_df[panel].mean()
            sd = max(score_df[panel].std(), 1e-6)
            score_df[f'{panel}_z'] = (score_df[panel] - mu) / sd
    return score_df


def _assign_subtypes(score_df):
    subtype_map = {}
    for c in score_df.index:
        s = score_df.loc[c]
        er_pct = s.get('er_pct', 0.5)
        her2_pct = s.get('her2_pct', 0.0)
        lum_z = s.get('luminal_z', 0.0)
        her2_z = s.get('her2_z', 0.0)
        bas_z = s.get('basal_z', 0.0)
        pro_z = s.get('prolif_z', 0.0)

        if er_pct < 0.3 and bas_z > 0 and her2_z < 0.5:
            subtype_map[c] = 'Basal-like'
        elif her2_z > 0.5 and her2_pct > 0.2:
            subtype_map[c] = 'HER2-enriched'
        elif er_pct >= 0.5 and pro_z > 0.3:
            subtype_map[c] = 'Luminal B'
        elif er_pct >= 0.5:
            subtype_map[c] = 'Luminal A'
        else:
            best = max([('Basal-like', bas_z), ('HER2-enriched', her2_z),
                        ('Luminal', lum_z)], key=lambda x: x[1])
            subtype_map[c] = best[0]
    return subtype_map


# ---------------------------------------------------------------------------
# Section E: Normalization Diagnostics
# ---------------------------------------------------------------------------

def section_e(expr_df, clinical, gene_cols, X_pca, patient_ids):
    """
    Returns dict with keys E1..E4.
    """
    r = {}
    pc1 = X_pca[:, 0]

    clin_dedup = clinical.drop_duplicates('pid').set_index('pid')
    pid_to_tss = clin_dedup['tss'].to_dict() if 'tss' in clin_dedup else {}

    # E1: CV of median expression across TSS sites
    tss_map = {p: pid_to_tss.get(p) for p in expr_df['pid']}
    tss_series = pd.Series(tss_map.values(), index=expr_df['pid'])
    tss_valid = tss_series.dropna()

    if len(tss_valid) > 0:
        tmp = expr_df.set_index('pid').loc[tss_valid.index, gene_cols]
        tmp['_tss'] = tss_valid
        tss_medians = tmp.groupby('_tss')[gene_cols].median().median(axis=1)
        r['E1'] = (tss_medians.std() / tss_medians.mean()
                   if tss_medians.mean() != 0 else float('nan'))
    else:
        r['E1'] = float('nan')
    print(f"  E1 CV TSS: {r['E1']:.4f}")

    # E2: PC1 vs TSS (Kruskal-Wallis)
    tss_cluster = [pid_to_tss.get(p) for p in patient_ids]
    tss_grp = {}
    for i, t in enumerate(tss_cluster):
        if pd.notna(t):
            tss_grp.setdefault(t, []).append(pc1[i])
    valid_grps = [v for v in tss_grp.values() if len(v) >= 2]
    if len(valid_grps) >= 3:
        _, kw_p = stats.kruskal(*valid_grps)
    else:
        kw_p = float('nan')
    r['E2'] = kw_p
    print(f"  E2 KW p: {kw_p:.2e}")

    # E3: PC1 vs read-depth proxy (sum of top genes in clustering)
    gene_mad = expr_df[gene_cols].apply(
        lambda x: np.median(np.abs(x - np.median(x))), axis=0
    )
    top_genes = gene_mad.nlargest(min(3000, len(gene_cols))).index.tolist()
    lib_proxy = expr_df[top_genes].sum(axis=1).values
    if len(lib_proxy) == len(pc1):
        r['E3'], _ = stats.pearsonr(pc1, lib_proxy)
    else:
        r['E3'] = float('nan')
    print(f"  E3 PC1-vs-depth r: {r['E3']:.4f}")

    # E4: ERBB2 CV within HER2+ group
    her2_pos_pids = set(
        clinical[clinical['her2_composite'] == 'Positive']['pid']
    )
    her2_expr = expr_df[expr_df['pid'].isin(her2_pos_pids)]
    if 'ERBB2' in her2_expr.columns and len(her2_expr) > 1:
        vals = her2_expr['ERBB2'].dropna()
        r['E4'] = vals.std() / vals.mean() if vals.mean() != 0 else float('nan')
    else:
        r['E4'] = float('nan')
    print(f"  E4 ERBB2 CV HER2+: {r['E4']:.4f}")

    return r


# ---------------------------------------------------------------------------
# Plot generation (P1-P5)
# ---------------------------------------------------------------------------

def generate_plots(cohort_c, b_plot_data, k4_labels, X_pca, char_df,
                   score_df, sil_scores, method_name, output_dir):
    """
    Generate P1-P5 plots to output_dir.
    Silently skips on import errors (matplotlib not critical).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  WARNING: matplotlib/seaborn not available -- skipping plots")
        return

    output_dir = Path(output_dir)
    COLORS = {'Positive': '#d62728', 'Negative': '#1f77b4',
              'Equivocal': '#ff7f0e', 'Unknown': '#7f7f7f'}

    # P1: ERBB2 RNA vs CN scatter colored by HER2 status
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        labeled = cohort_c[cohort_c['her2_composite'].isin(['Positive', 'Negative'])]
        for status, grp in labeled.groupby('her2_composite'):
            ax.scatter(grp['ERBB2_expr'], grp['erbb2_copy_number'],
                       c=COLORS.get(status, 'gray'), alpha=0.4, s=12,
                       label=f'HER2 {status}')
        ax.set_xlabel('ERBB2 log-expression')
        ax.set_ylabel('ERBB2 copy number')
        ax.set_title(f'P1: ERBB2 RNA vs CN -- {method_name}')
        ax.legend(markerscale=2)
        fig.tight_layout()
        fig.savefig(output_dir / 'scatter_rna_vs_cn.png', dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  WARNING: P1 failed: {e}")

    # P2: ROC curves overlay
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        y_true = b_plot_data['y_true']
        line_styles = ['-', '--', ':']
        for (name, y_prob), ls in zip(b_plot_data['y_probs'].items(), line_styles):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, ls, label=f'{name} (AUC={auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title(f'P2: ROC curves -- {method_name}')
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / 'roc_curves.png', dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  WARNING: P2 failed: {e}")

    # P3: Silhouette profile across k
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        ks = list(sil_scores.keys())
        sils = list(sil_scores.values())
        ax.plot(ks, sils, 'o-', color='steelblue')
        ax.axvline(x=4, color='gray', linestyle='--', alpha=0.6, label='k=4')
        ax.set_xlabel('k (number of clusters)')
        ax.set_ylabel('Silhouette score')
        ax.set_title(f'P3: Silhouette profile -- {method_name}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / 'silhouette_profile.png', dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  WARNING: P3 failed: {e}")

    # P4: PCA scatter colored by k=4 clusters
    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        unique_clusters = np.unique(k4_labels)
        palette = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))

        for ax_idx, ax in enumerate(axes):
            for ci, color in zip(unique_clusters, palette):
                mask = k4_labels == ci
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=[color], alpha=0.4, s=10, label=f'Cluster {ci}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')

        axes[0].set_title(f'P4: PCA (k=4 clusters) -- {method_name}')
        axes[0].legend(markerscale=2, fontsize=7)
        axes[1].set_title('PC1 vs PC3')
        for ci, color in zip(unique_clusters, palette):
            mask = k4_labels == ci
            axes[1].scatter(X_pca[mask, 0],
                            X_pca[mask, 2] if X_pca.shape[1] > 2 else X_pca[mask, 1],
                            c=[color], alpha=0.4, s=10)

        fig.tight_layout()
        fig.savefig(output_dir / 'umap_clusters.png', dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  WARNING: P4 failed: {e}")

    # P5: Marker gene heatmap by cluster
    try:
        marker_order = [
            'ESR1', 'PGR', 'GATA3', 'FOXA1',
            'ERBB2', 'GRB7', 'STARD3',
            'KRT5', 'KRT14', 'KRT17', 'EGFR',
        ]
        available = [g for g in marker_order if g in char_df.columns]
        if available:
            heatmap_data = (char_df.groupby('cluster')[available]
                            .median().T)
            fig, ax = plt.subplots(figsize=(max(5, len(heatmap_data.columns) + 2),
                                            max(5, len(available) * 0.5 + 1)))
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdBu_r',
                        center=heatmap_data.values.mean(), ax=ax)
            ax.set_title(f'P5: Marker gene heatmap (median per cluster) -- {method_name}')
            ax.set_xlabel('Cluster')
            fig.tight_layout()
            fig.savefig(output_dir / 'marker_heatmap.png', dpi=120)
            plt.close(fig)
    except Exception as e:
        print(f"  WARNING: P5 failed: {e}")

    print(f"  Plots saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(report, method_name, output_dir, cohort_size,
                 subtype_map, score_df, char_df, k4_labels,
                 gene_length_source=None):
    """
    Write report.md and report_metrics.json to output_dir.
    Returns path to report.md.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sil_str = ', '.join(
        f"k={k}: {v:.4f}" for k, v in report['C1'].items()
    )

    # Build subtype block text
    subtype_lines = []
    for c in sorted(subtype_map):
        n = (k4_labels == c).sum()
        er_pct = score_df.loc[c, 'er_pct'] if 'er_pct' in score_df.columns else float('nan')
        her2_pct = score_df.loc[c, 'her2_pct'] if 'her2_pct' in score_df.columns else float('nan')
        subtype_lines.append(
            f"- Cluster {c} -> {subtype_map[c]} (n={n}): "
            f"ER%={er_pct:.0%}, HER2%={her2_pct:.0%}"
        )
    subtype_block = '\n'.join(subtype_lines)

    r = report
    gl_note = f"\n**Gene length source:** {gene_length_source}" if gene_length_source else ''

    def sig_str(v):
        if v > 1.5:
            return 'Strong'
        if v > 0.8:
            return 'Moderate'
        return 'Weak'

    def corr_str(v):
        av = abs(v)
        if av > 0.5:
            return 'strong'
        if av > 0.3:
            return 'moderate'
        return 'weak'

    def depth_str(v):
        av = abs(v)
        if av < 0.1:
            return 'negligible'
        if av < 0.3:
            return 'weak'
        if av < 0.5:
            return 'moderate'
        return 'strong'

    def cv_str(v):
        if v < 0.15:
            return 'low'
        if v < 0.3:
            return 'moderate'
        return 'high'

    report_text = f"""# Normalization Comparison Report: {method_name.upper()}

**Method:** {method_name}
**Date:** 2026-04-05
**Input:** TCGA BRCA expression matrix normalized by {method_name}, log2(x+1) transformed,
with TSS batch regression (protecting HER2/ER covariates).
Multimodal cohort: {cohort_size} patients with clinical + RNA-seq + copy number data.{gl_note}

---

## Section A: ERBB2 RNA vs Copy Number

| Metric | Value |
|--------|-------|
| A1. Pearson r (all samples) | {r['A1']:.4f} |
| A2. Spearman rho (all samples) | {r['A2']:.4f} |
| A3. Pearson r (HER2+ only) | {r['A3']:.4f} |
| A4. Pearson r (HER2- only) | {r['A4']:.4f} |
| A5. Cohen's d (HER2+ vs HER2-) | {r['A5']:.4f} |
| A6. Mann-Whitney U p-value | {r['A6']:.2e} |
| A7. Fold-change (median Pos/Neg) | {r['A7']:.4f} |

**Interpretation:** {sig_str(r['A5'])} effect size (Cohen's d = {r['A5']:.2f}) separating
HER2+ from HER2- on ERBB2 RNA. Pearson r of {r['A1']:.3f} between RNA and CN indicates
{corr_str(r['A1'])} linear concordance across all samples.

---

## Section B: Logistic Regression (RNA/CN -> HER2 IHC)

| Metric | Value |
|--------|-------|
| B1. AUC-ROC, RNA only (5-fold CV) | {r['B1']:.4f} |
| B2. AUC-PR, RNA only (5-fold CV) | {r['B2']:.4f} |
| B3. AUC-ROC, CN only (5-fold CV) | {r['B3']:.4f} |
| B4. AUC-PR, CN only (5-fold CV) | {r['B4']:.4f} |
| B5. AUC-ROC, RNA + CN (5-fold CV) | {r['B5']:.4f} |
| B6. AUC-PR, RNA + CN (5-fold CV) | {r['B6']:.4f} |
| B7. Delta AUC-ROC (RNA - CN) | {r['B7']:.4f} |

**Interpretation:** RNA {'outperforms' if r['B7'] > 0.01 else 'matches'} CN for HER2
prediction (delta AUC = {r['B7']:+.3f}). Combined model
{'improves' if r['B5'] > r['B1'] + 0.005 else 'does not improve'} over RNA alone,
suggesting CN's predictive information is
{'partially independent of' if r['B5'] > r['B1'] + 0.005 else 'largely captured by'} RNA.

---

## Section C: Unsupervised Clustering

| Metric | Value |
|--------|-------|
| C1. Silhouette scores (k=2..7) | {sil_str} |
| C2. Best k (argmax silhouette) | {r['C2']} |
| C3. Silhouette at k=4 | {r['C3']:.4f} |
| C4. ARI (k=4 vs HER2 label) | {r['C4']:.4f} |
| C5. ARI (k=4 vs ER status) | {r['C5']:.4f} |

**Interpretation:** Best k = {r['C2']} by silhouette. k=4 silhouette = {r['C3']:.3f};
ARI with ER ({r['C5']:.3f}) {'>' if r['C5'] > r['C4'] else '<'} ARI with HER2 ({r['C4']:.3f}),
indicating clustering structure is
{'more aligned with ER' if r['C5'] > r['C4'] else 'more aligned with HER2'} status.

---

## Section D: Subtype Marker Separation

| Metric | Value |
|--------|-------|
| D1. Luminal score spread (range) | {r['D1']:.4f} |
| D2. HER2 score spread (range) | {r['D2']:.4f} |
| D3. Basal score spread (range) | {r['D3']:.4f} |
| D4. Mean subtype-score gap | {r['D4']:.4f} |
| D5. Frac HER2+ in HER2-enriched cluster | {r['D5']:.4f} |

**Subtype assignments (k=4):**
{subtype_block}

**Interpretation:** Mean subtype-score gap = {r['D4']:.2f}.
{'Marker genes separate cleanly across clusters' if r['D4'] > 1.0 else 'Moderate marker separation'}.
{r['D5']:.0%} of HER2+ patients fall in the HER2-enriched cluster.

---

## Section E: Normalization Diagnostics

| Metric | Value |
|--------|-------|
| E1. CV of median expr across TSS | {r['E1']:.4f} |
| E2. PC1 vs TSS (Kruskal-Wallis p) | {r['E2']:.2e} |
| E3. PC1 vs read-depth proxy (r) | {r['E3']:.4f} |
| E4. ERBB2 CV within HER2+ group | {r['E4']:.4f} |

**Interpretation:** {'TSS batch effects remain significant (p < 0.05 on PC1)' if r['E2'] < 0.05 else 'TSS batch effects not significant on PC1'}.
PC1 correlation with read-depth proxy = {r['E3']:.3f}
({depth_str(r['E3'])} read-depth confound in primary axis of variation).
ERBB2 CV within HER2+ = {r['E4']:.3f} ({cv_str(r['E4'])} within-group heterogeneity).

---

## Summary

| Category | Key Metric | Value |
|----------|-----------|-------|
| Signal: Effect size | Cohen's d (A5) | {r['A5']:.3f} |
| Signal: RNA predictiveness | AUC-ROC RNA (B1) | {r['B1']:.3f} |
| Signal: RNA advantage | Delta AUC (B7) | {r['B7']:+.3f} |
| Signal: Cluster quality | Silhouette k=4 (C3) | {r['C3']:.3f} |
| Signal: Marker separation | Mean gap (D4) | {r['D4']:.3f} |
| Signal: HER2 cluster purity | D5 | {r['D5']:.3f} |
| Noise: TSS batch | KW p (E2) | {r['E2']:.2e} |
| Noise: Read-depth confound | PC1 r (E3) | {r['E3']:.3f} |
| Noise: HER2+ heterogeneity | ERBB2 CV (E4) | {r['E4']:.3f} |

---

*Report generated by scripts/normalization_comparison/ -- Phase 2/3 of normalization comparison.*
"""

    report_path = output_dir / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  Report written to: {report_path}")

    # Save JSON for comparison summary
    metrics_json = {}
    for k, v in report.items():
        if k == 'C1':
            metrics_json[k] = {str(kk): float(vv) for kk, vv in v.items()}
        elif isinstance(v, (int, float)):
            metrics_json[k] = float(v)
        else:
            metrics_json[k] = v
    metrics_json['cohort_size'] = cohort_size
    metrics_json['method'] = method_name
    if gene_length_source:
        metrics_json['gene_length_source'] = gene_length_source

    json_path = output_dir / 'report_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Metrics JSON written to: {json_path}")

    return report_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_analysis(expr_df, clinical, cn, gene_cols, method_name, output_dir,
                 gene_length_source=None):
    """
    Run the full sections A-E analysis and write the report.

    Parameters
    ----------
    expr_df     : DataFrame with 'pid' + gene expression columns (normalized, log-scale)
    clinical    : DataFrame from 01_clinical_qc
    cn          : DataFrame with 'pid', 'erbb2_copy_number'
    gene_cols   : list of gene column names present in expr_df
    method_name : string label for this normalization (e.g. 'tpm', 'tmm_edger')
    output_dir  : Path to write report.md and report_metrics.json
    gene_length_source : optional description of gene length data source

    Returns
    -------
    dict of all metrics (A1..E4)
    """
    print(f"\n{'='*70}")
    print(f"ANALYSIS PIPELINE: {method_name.upper()}")
    print(f"{'='*70}")

    # Build multimodal cohort
    clin_rna = clinical.merge(expr_df[['pid'] + gene_cols], on='pid', how='inner')
    cohort_c = clin_rna.merge(cn[['pid', 'erbb2_copy_number']], on='pid', how='inner')
    cohort_c['ERBB2_expr'] = cohort_c['ERBB2'] if 'ERBB2' in cohort_c.columns else float('nan')
    print(f"Multimodal cohort: {len(cohort_c)} patients")

    report = {}

    print("\n--- Section A ---")
    report.update(section_a(cohort_c))

    print("\n--- Section B ---")
    b_metrics, b_plot_data = section_b(cohort_c)
    report.update(b_metrics)

    print("\n--- Section C ---")
    c_metrics, k4_labels, patient_ids, X_pca = section_c(expr_df, clinical, gene_cols)
    report.update(c_metrics)

    print("\n--- Section D ---")
    d_metrics, subtype_map, score_df, char_df = section_d(
        expr_df, clinical, k4_labels, patient_ids
    )
    report.update(d_metrics)

    print("\n--- Section E ---")
    e_metrics = section_e(expr_df, clinical, gene_cols, X_pca, patient_ids)
    report.update(e_metrics)

    write_report(
        report, method_name, output_dir, len(cohort_c),
        subtype_map, score_df, char_df, k4_labels,
        gene_length_source=gene_length_source
    )

    print("\n--- Generating plots ---")
    generate_plots(cohort_c, b_plot_data, k4_labels, X_pca, char_df,
                   score_df, report['C1'], method_name, output_dir)

    return report
