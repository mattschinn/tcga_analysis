"""
Phase 1: Extract RSEM-UQ-TSS baseline report from Notebook 02a pipeline.

Loads the same intermediates, runs identical analyses, and computes all
metrics defined in the normalization comparison report template (Phase 0).
Outputs a structured report to skills/user/shared/.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    silhouette_score, adjusted_rand_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from src.utils import load_intermediate, load_gene_cols

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("PHASE 1: RSEM-UQ-TSS BASELINE REPORT EXTRACTION")
print("=" * 70)

clinical = load_intermediate('01_clinical_qc')
tumor_norm = load_intermediate('01_tumor_norm')
cn = load_intermediate('01_cn_qc')
gene_cols = load_gene_cols()

print(f"\nClinical: {len(clinical)} patients")
print(f"Tumor norm: {len(tumor_norm)} samples x {len(gene_cols)} genes")
print(f"Copy number: {len(cn)} patients")

# ============================================================================
# 2. MERGE (identical to Notebook 02a Cell 5)
# ============================================================================
clin_rna = clinical.merge(tumor_norm, on='pid', how='inner')
cohort_c = clin_rna.merge(cn[['pid', 'erbb2_copy_number']], on='pid', how='inner')
print(f"Multimodal cohort: {len(cohort_c)} patients")

analysis_df = cohort_c.copy()
if 'ERBB2' in gene_cols:
    analysis_df['ERBB2_expr'] = analysis_df['ERBB2']

for gene in ['GRB7', 'ESR1', 'PGR', 'MKI67', 'EGFR', 'ERBB3']:
    if gene in gene_cols:
        analysis_df[f'{gene}_expr'] = analysis_df[gene]

# ============================================================================
# 3. SECTION A: ERBB2 RNA vs Copy Number
# ============================================================================
print("\n--- SECTION A: ERBB2 RNA vs Copy Number ---")
report = {}

labeled = analysis_df[analysis_df['her2_composite'].isin(['Positive', 'Negative'])].copy()
pos = labeled[labeled['her2_composite'] == 'Positive']
neg = labeled[labeled['her2_composite'] == 'Negative']

pos_expr = pos['ERBB2_expr'].dropna()
neg_expr = neg['ERBB2_expr'].dropna()
all_expr = labeled['ERBB2_expr'].dropna()
all_cn = labeled.loc[all_expr.index, 'erbb2_copy_number']

# A1-A2: Correlations (all samples)
r_pearson, _ = stats.pearsonr(all_expr, all_cn)
r_spearman, _ = stats.spearmanr(all_expr, all_cn)
report['A1'] = r_pearson
report['A2'] = r_spearman
print(f"  A1. Pearson r (all):    {r_pearson:.4f}")
print(f"  A2. Spearman rho (all): {r_spearman:.4f}")

# A3: Pearson r (HER2+ only)
pos_both = pos.dropna(subset=['ERBB2_expr', 'erbb2_copy_number'])
if len(pos_both) > 2:
    r_pos, _ = stats.pearsonr(pos_both['ERBB2_expr'], pos_both['erbb2_copy_number'])
else:
    r_pos = float('nan')
report['A3'] = r_pos
print(f"  A3. Pearson r (HER2+):  {r_pos:.4f}")

# A4: Pearson r (HER2- only)
neg_both = neg.dropna(subset=['ERBB2_expr', 'erbb2_copy_number'])
if len(neg_both) > 2:
    r_neg, _ = stats.pearsonr(neg_both['ERBB2_expr'], neg_both['erbb2_copy_number'])
else:
    r_neg = float('nan')
report['A4'] = r_neg
print(f"  A4. Pearson r (HER2-):  {r_neg:.4f}")

# A5: Cohen's d
pooled_std = np.sqrt(((len(pos_expr)-1)*pos_expr.std()**2 +
                       (len(neg_expr)-1)*neg_expr.std()**2) /
                      (len(pos_expr) + len(neg_expr) - 2))
cohens_d = (pos_expr.mean() - neg_expr.mean()) / pooled_std
report['A5'] = cohens_d
print(f"  A5. Cohen's d (Pos-Neg): {cohens_d:.4f}")

# A6: Mann-Whitney U
u_stat, mw_p = stats.mannwhitneyu(pos_expr, neg_expr, alternative='two-sided')
report['A6'] = mw_p
print(f"  A6. Mann-Whitney p:      {mw_p:.2e}")

# A7: Fold-change (median)
fc = pos_expr.median() / neg_expr.median() if neg_expr.median() != 0 else float('nan')
report['A7'] = fc
print(f"  A7. Fold-change (median): {fc:.4f}")

# ============================================================================
# 4. SECTION B: Logistic Regression
# ============================================================================
print("\n--- SECTION B: Logistic Regression ---")

model_df = analysis_df.dropna(subset=['her2_composite', 'erbb2_copy_number']).copy()
model_df = model_df[model_df['her2_composite'].isin(['Positive', 'Negative'])]
model_df['y'] = (model_df['her2_composite'] == 'Positive').astype(int)
model_df = model_df.dropna(subset=['ERBB2_expr'])

X_rna = model_df[['ERBB2_expr']].values
X_cn = model_df[['erbb2_copy_number']].values
X_both = model_df[['ERBB2_expr', 'erbb2_copy_number']].values
y = model_df['y'].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_lr = {}
for name, X in [('ERBB2 RNA only', X_rna), ('ERBB2 CN only', X_cn),
                ('RNA + CN combined', X_both)]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    y_prob = cross_val_predict(lr, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    auc_roc = roc_auc_score(y, y_prob)
    auc_pr = average_precision_score(y, y_prob)
    results_lr[name] = {'auc_roc': auc_roc, 'auc_pr': auc_pr}

report['B1'] = results_lr['ERBB2 RNA only']['auc_roc']
report['B2'] = results_lr['ERBB2 RNA only']['auc_pr']
report['B3'] = results_lr['ERBB2 CN only']['auc_roc']
report['B4'] = results_lr['ERBB2 CN only']['auc_pr']
report['B5'] = results_lr['RNA + CN combined']['auc_roc']
report['B6'] = results_lr['RNA + CN combined']['auc_pr']
report['B7'] = report['B1'] - report['B3']

for key in ['B1','B2','B3','B4','B5','B6','B7']:
    print(f"  {key}. {report[key]:.4f}")

# ============================================================================
# 5. SECTION C: Unsupervised Clustering
# ============================================================================
print("\n--- SECTION C: Unsupervised Clustering ---")

tumor_for_clustering = tumor_norm.copy()
gene_mad = tumor_for_clustering[gene_cols].apply(
    lambda x: np.median(np.abs(x - np.median(x))), axis=0
)
gene_mad_sorted = gene_mad.sort_values(ascending=False)
n_top_genes = min(len(gene_cols), 3000)
top_genes = gene_mad_sorted.head(n_top_genes).index.tolist()

X_cluster = tumor_for_clustering[top_genes].fillna(0).values
patient_ids_cluster = tumor_for_clustering['pid'].values

scaler_cl = StandardScaler()
X_scaled_cl = scaler_cl.fit_transform(X_cluster)

n_pcs = min(20, X_scaled_cl.shape[0], X_scaled_cl.shape[1])
pca_cl = PCA(n_components=n_pcs)
X_pca = pca_cl.fit_transform(X_scaled_cl)

n_pcs_for_clustering = min(10, X_pca.shape[1])
X_for_clustering = X_pca[:, :n_pcs_for_clustering]

# C1: Silhouette scores for k=2..7
k_range = range(2, min(8, len(X_for_clustering) // 5))
silhouette_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_for_clustering)
    sil = silhouette_score(X_for_clustering, labels)
    silhouette_scores.append(sil)

report['C1'] = dict(zip(list(k_range), silhouette_scores))
report['C2'] = list(k_range)[np.argmax(silhouette_scores)]
print(f"  C1. Silhouette scores: {report['C1']}")
print(f"  C2. Best k: {report['C2']}")

# C3: Silhouette at k=4
km4 = KMeans(n_clusters=4, random_state=42, n_init=20)
k4_labels = km4.fit_predict(X_for_clustering)
sil_k4 = silhouette_score(X_for_clustering, k4_labels)
report['C3'] = sil_k4
print(f"  C3. Silhouette at k=4: {sil_k4:.4f}")

# C4-C5: ARI at k=4
pid_to_her2 = dict(zip(clinical['pid'], clinical['her2_composite']))
pid_to_er = dict(zip(clinical['pid'], clinical['ER Status By IHC']))

cluster_her2 = [pid_to_her2.get(pid, np.nan) for pid in patient_ids_cluster]
cluster_er = [pid_to_er.get(pid, np.nan) for pid in patient_ids_cluster]

# ARI requires non-NaN labels -- filter
her2_valid = [(k4_labels[i], cluster_her2[i])
              for i in range(len(k4_labels))
              if cluster_her2[i] in ('Positive', 'Negative')]
if len(her2_valid) > 0:
    k4_sub, her2_sub = zip(*her2_valid)
    ari_her2 = adjusted_rand_score(her2_sub, k4_sub)
else:
    ari_her2 = float('nan')

er_valid = [(k4_labels[i], cluster_er[i])
            for i in range(len(k4_labels))
            if cluster_er[i] in ('Positive', 'Negative')]
if len(er_valid) > 0:
    k4_sub_er, er_sub = zip(*er_valid)
    ari_er = adjusted_rand_score(er_sub, k4_sub_er)
else:
    ari_er = float('nan')

report['C4'] = ari_her2
report['C5'] = ari_er
print(f"  C4. ARI (k=4 vs HER2): {ari_her2:.4f}")
print(f"  C5. ARI (k=4 vs ER):   {ari_er:.4f}")

# ============================================================================
# 6. SECTION D: Subtype Marker Separation
# ============================================================================
print("\n--- SECTION D: Subtype Marker Separation ---")

# Build characterization df (same as Notebook 02a cells 33/37)
char_df = pd.DataFrame({
    'pid': patient_ids_cluster,
    'cluster': k4_labels,
})

tumor_expr_map = tumor_norm.set_index('pid')
marker_genes = {
    'ESR1': 'Luminal', 'PGR': 'Luminal', 'GATA3': 'Luminal', 'FOXA1': 'Luminal',
    'ERBB2': 'HER2', 'GRB7': 'HER2', 'STARD3': 'HER2',
    'KRT5': 'Basal', 'KRT14': 'Basal', 'KRT17': 'Basal', 'EGFR': 'Basal',
    'MKI67': 'Proliferation', 'CCNB1': 'Proliferation',
    'AURKA': 'Proliferation', 'TOP2A': 'Proliferation',
}

for gene in marker_genes:
    if gene in tumor_expr_map.columns:
        char_df[gene] = char_df['pid'].map(tumor_expr_map[gene])

clin_map = clinical.drop_duplicates(subset='pid').set_index('pid')
for col in ['ER Status By IHC', 'her2_composite']:
    if col in clin_map.columns:
        char_df[col] = char_df['pid'].map(clin_map[col])

# Compute subtype scores per cluster
subtype_scores = {}
for c in sorted(char_df['cluster'].unique()):
    cl = char_df[char_df['cluster'] == c]
    scores = {}
    luminal_genes = [g for g in ['ESR1', 'PGR', 'GATA3', 'FOXA1'] if g in cl.columns]
    if luminal_genes:
        scores['luminal'] = cl[luminal_genes].median().mean()
    her2_genes = [g for g in ['ERBB2', 'GRB7', 'STARD3'] if g in cl.columns]
    if her2_genes:
        scores['her2'] = cl[her2_genes].median().mean()
    basal_genes = [g for g in ['KRT5', 'KRT14', 'KRT17', 'EGFR'] if g in cl.columns]
    if basal_genes:
        scores['basal'] = cl[basal_genes].median().mean()
    subtype_scores[c] = scores

score_df = pd.DataFrame(subtype_scores).T

# D1-D3: IQR of subtype scores across clusters
if 'luminal' in score_df.columns:
    report['D1'] = score_df['luminal'].max() - score_df['luminal'].min()
else:
    report['D1'] = float('nan')
if 'her2' in score_df.columns:
    report['D2'] = score_df['her2'].max() - score_df['her2'].min()
else:
    report['D2'] = float('nan')
if 'basal' in score_df.columns:
    report['D3'] = score_df['basal'].max() - score_df['basal'].min()
else:
    report['D3'] = float('nan')

# D4: Mean subtype-score gap
report['D4'] = np.mean([report['D1'], report['D2'], report['D3']])

print(f"  D1. Luminal score spread:  {report['D1']:.4f}")
print(f"  D2. HER2 score spread:     {report['D2']:.4f}")
print(f"  D3. Basal score spread:    {report['D3']:.4f}")
print(f"  D4. Mean score gap:        {report['D4']:.4f}")

# D5: Fraction of HER2+ in HER2-enriched cluster
# Assign subtypes using same logic as notebook
for col in ['luminal', 'her2', 'basal']:
    if col in score_df.columns:
        score_df[f'{col}_z'] = (
            (score_df[col] - score_df[col].mean()) / max(score_df[col].std(), 1e-6)
        )

# Compute proliferation score and er_pct/her2_pct per cluster
for c in score_df.index:
    cl = char_df[char_df['cluster'] == c]
    prolif_genes = [g for g in ['MKI67', 'CCNB1', 'AURKA', 'TOP2A'] if g in cl.columns]
    if prolif_genes:
        score_df.loc[c, 'proliferation'] = cl[prolif_genes].median().mean()
    er_vals = (cl['ER Status By IHC'] == 'Positive').sum()
    er_total = cl['ER Status By IHC'].isin(['Positive', 'Negative']).sum()
    score_df.loc[c, 'er_pct'] = er_vals / max(er_total, 1)
    her2_vals = (cl['her2_composite'] == 'Positive').sum()
    her2_total = cl['her2_composite'].isin(['Positive', 'Negative']).sum()
    score_df.loc[c, 'her2_pct'] = her2_vals / max(her2_total, 1)

if 'proliferation' in score_df.columns:
    score_df['proliferation_z'] = (
        (score_df['proliferation'] - score_df['proliferation'].mean()) /
        max(score_df['proliferation'].std(), 1e-6)
    )

subtype_map = {}
for c in score_df.index:
    s = score_df.loc[c]
    er_pct = s.get('er_pct', 0.5)
    her2_pct = s.get('her2_pct', 0)
    lum_z = s.get('luminal_z', 0)
    her2_z = s.get('her2_z', 0)
    bas_z = s.get('basal_z', 0)
    pro_z = s.get('proliferation_z', 0)

    if er_pct < 0.3 and bas_z > 0 and her2_z < 0.5:
        subtype_map[c] = 'Basal-like'
    elif her2_z > 0.5 and her2_pct > 0.2:
        subtype_map[c] = 'HER2-enriched'
    elif er_pct >= 0.5 and pro_z > 0.3:
        subtype_map[c] = 'Luminal B'
    elif er_pct >= 0.5:
        subtype_map[c] = 'Luminal A'
    else:
        max_axis = max([('Basal-like', bas_z), ('HER2-enriched', her2_z),
                        ('Luminal B', lum_z)], key=lambda x: x[1])
        subtype_map[c] = max_axis[0]

print(f"\n  Subtype assignments: {subtype_map}")

# Find HER2-enriched cluster(s)
her2_enriched_clusters = [c for c, s in subtype_map.items() if s == 'HER2-enriched']
if her2_enriched_clusters:
    her2_pos_patients = char_df[char_df['her2_composite'] == 'Positive']
    in_enriched = her2_pos_patients['cluster'].isin(her2_enriched_clusters).sum()
    total_her2_pos = len(her2_pos_patients)
    report['D5'] = in_enriched / max(total_her2_pos, 1)
else:
    report['D5'] = float('nan')
print(f"  D5. Frac HER2+ in HER2-enriched: {report['D5']:.4f}")

# ============================================================================
# 7. SECTION E: Normalization Diagnostics
# ============================================================================
print("\n--- SECTION E: Normalization Diagnostics ---")

# E1: CV of median gene expression across TSS sites
if 'tss' in tumor_norm.columns or 'tss' in clinical.columns:
    # Map TSS to tumor_norm
    pid_to_tss = dict(zip(clinical['pid'], clinical.get('tss', pd.Series(dtype=str))))
    tn_tss = tumor_norm['pid'].map(pid_to_tss)

    tss_valid = tn_tss.dropna()
    if len(tss_valid) > 0:
        tss_groups = tumor_norm.loc[tss_valid.index]
        tss_groups = tss_groups.copy()
        tss_groups['_tss'] = tss_valid.values
        # Median expression per TSS
        tss_medians = tss_groups.groupby('_tss')[gene_cols].median().median(axis=1)
        cv_tss = tss_medians.std() / tss_medians.mean() if tss_medians.mean() != 0 else float('nan')
    else:
        cv_tss = float('nan')
else:
    cv_tss = float('nan')

report['E1'] = cv_tss
print(f"  E1. CV of median expr across TSS: {cv_tss:.4f}")

# E2: PC1 correlation with TSS (Kruskal-Wallis)
pc1 = X_pca[:, 0]
cluster_tss_vals = [pid_to_tss.get(pid, np.nan) for pid in patient_ids_cluster]
tss_groups_pc1 = {}
for i, t in enumerate(cluster_tss_vals):
    if pd.notna(t):
        tss_groups_pc1.setdefault(t, []).append(pc1[i])

# Need at least 3 groups with >= 2 samples each
valid_groups = [v for v in tss_groups_pc1.values() if len(v) >= 2]
if len(valid_groups) >= 3:
    kw_stat, kw_p = stats.kruskal(*valid_groups)
else:
    kw_p = float('nan')
report['E2'] = kw_p
print(f"  E2. PC1 vs TSS (Kruskal-Wallis p): {kw_p:.2e}")

# E3: PC1 correlation with read-depth proxy
# Use sum of expression as proxy for library size
lib_proxy = tumor_for_clustering[top_genes].sum(axis=1).values
if len(lib_proxy) == len(pc1):
    r_lib, _ = stats.pearsonr(pc1, lib_proxy)
else:
    r_lib = float('nan')
report['E3'] = r_lib
print(f"  E3. PC1 vs read-depth proxy (r):   {r_lib:.4f}")

# E4: ERBB2 CV within HER2+ group
her2_pos_pids = set(clinical[clinical['her2_composite'] == 'Positive']['pid'])
her2_pos_expr = tumor_norm[tumor_norm['pid'].isin(her2_pos_pids)]
if 'ERBB2' in her2_pos_expr.columns and len(her2_pos_expr) > 1:
    erbb2_vals = her2_pos_expr['ERBB2'].dropna()
    cv_erbb2 = erbb2_vals.std() / erbb2_vals.mean() if erbb2_vals.mean() != 0 else float('nan')
else:
    cv_erbb2 = float('nan')
report['E4'] = cv_erbb2
print(f"  E4. ERBB2 CV (HER2+ group):        {cv_erbb2:.4f}")

# ============================================================================
# 8. FORMAT AND WRITE REPORT
# ============================================================================
print("\n" + "=" * 70)
print("WRITING REPORT")
print("=" * 70)

# Format silhouette scores
sil_str = ", ".join([f"k={k}: {v:.4f}" for k, v in report['C1'].items()])

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
report_path = project_root / "skills" / "user" / "shared" / "rsem_uq_tss_report.md"

report_text = f"""# Normalization Comparison Report: RSEM-UQ-TSS

**Method:** RSEM-UQ-TSS (upper-quartile normalized RSEM + log2(x+1) + TSS-targeted correction)
**Date:** 2026-04-05
**Input:** TCGA BRCA RSEM expected counts, upper-quartile normalized per sample,
log2(x+1) transformed, with TSS batch regression (protecting HER2/ER covariates).
Multimodal cohort: {len(cohort_c)} patients with clinical + RNA-seq + copy number data.

---

## Section A: ERBB2 RNA vs Copy Number

| Metric | Value |
|--------|-------|
| A1. Pearson r (all samples) | {report['A1']:.4f} |
| A2. Spearman rho (all samples) | {report['A2']:.4f} |
| A3. Pearson r (HER2+ only) | {report['A3']:.4f} |
| A4. Pearson r (HER2- only) | {report['A4']:.4f} |
| A5. Cohen's d (HER2+ vs HER2-) | {report['A5']:.4f} |
| A6. Mann-Whitney U p-value | {report['A6']:.2e} |
| A7. Fold-change (median Pos/Neg) | {report['A7']:.4f} |

**Interpretation:** {"Strong" if report['A5'] > 1.5 else "Moderate" if report['A5'] > 0.8 else "Weak"} \
effect size (Cohen's d = {report['A5']:.2f}) separating HER2+ from HER2- on ERBB2 RNA.
Pearson r of {report['A1']:.3f} between RNA and CN indicates \
{"strong" if abs(report['A1']) > 0.5 else "moderate" if abs(report['A1']) > 0.3 else "weak"} \
linear concordance across all samples.

---

## Section B: Logistic Regression (RNA/CN -> HER2 IHC)

| Metric | Value |
|--------|-------|
| B1. AUC-ROC, RNA only (5-fold CV) | {report['B1']:.4f} |
| B2. AUC-PR, RNA only (5-fold CV) | {report['B2']:.4f} |
| B3. AUC-ROC, CN only (5-fold CV) | {report['B3']:.4f} |
| B4. AUC-PR, CN only (5-fold CV) | {report['B4']:.4f} |
| B5. AUC-ROC, RNA + CN (5-fold CV) | {report['B5']:.4f} |
| B6. AUC-PR, RNA + CN (5-fold CV) | {report['B6']:.4f} |
| B7. Delta AUC-ROC (RNA - CN) | {report['B7']:.4f} |

**Interpretation:** RNA {"outperforms" if report['B7'] > 0.01 else "matches"} CN for HER2 prediction \
(delta AUC = {report['B7']:+.3f}). Combined model \
{"improves" if report['B5'] > report['B1'] + 0.005 else "does not improve"} over RNA alone, \
suggesting CN's predictive information is {"largely captured by" if report['B5'] <= report['B1'] + 0.005 else "partially independent of"} \
RNA expression.

---

## Section C: Unsupervised Clustering

| Metric | Value |
|--------|-------|
| C1. Silhouette scores (k=2..7) | {sil_str} |
| C2. Best k (argmax silhouette) | {report['C2']} |
| C3. Silhouette at k=4 | {report['C3']:.4f} |
| C4. ARI (k=4 vs HER2 label) | {report['C4']:.4f} |
| C5. ARI (k=4 vs ER status) | {report['C5']:.4f} |

**Interpretation:** Best k = {report['C2']} by silhouette \
({"which maps primarily to ER status" if report['C2'] == 2 else ""}). \
k=4 silhouette = {report['C3']:.3f}; ARI with ER ({report['C5']:.3f}) \
{">" if report['C5'] > report['C4'] else "<"} ARI with HER2 ({report['C4']:.3f}), \
indicating clustering structure is {"more aligned with ER" if report['C5'] > report['C4'] else "more aligned with HER2"} \
status.

---

## Section D: Subtype Marker Separation

| Metric | Value |
|--------|-------|
| D1. Luminal score spread (range) | {report['D1']:.4f} |
| D2. HER2 score spread (range) | {report['D2']:.4f} |
| D3. Basal score spread (range) | {report['D3']:.4f} |
| D4. Mean subtype-score gap | {report['D4']:.4f} |
| D5. Frac HER2+ in HER2-enriched cluster | {report['D5']:.4f} |

**Subtype assignments (k=4):**
"""

for c in sorted(subtype_map):
    n = (k4_labels == c).sum()
    s = score_df.loc[c]
    report_text += (
        f"- Cluster {c} -> {subtype_map[c]} (n={n}): "
        f"ER%={s.get('er_pct',0):.0%}, HER2%={s.get('her2_pct',0):.0%}\n"
    )

report_text += f"""
**Interpretation:** Mean subtype-score gap = {report['D4']:.2f}. \
{"Marker genes separate cleanly across clusters" if report['D4'] > 1.0 else "Moderate marker separation across clusters"}\
. {report['D5']:.0%} of HER2+ patients fall in the HER2-enriched cluster, \
with the remainder distributed across other subtypes (primarily Luminal B).

---

## Section E: Normalization Diagnostics

| Metric | Value |
|--------|-------|
| E1. CV of median expr across TSS | {report['E1']:.4f} |
| E2. PC1 vs TSS (Kruskal-Wallis p) | {report['E2']:.2e} |
| E3. PC1 vs read-depth proxy (r) | {report['E3']:.4f} |
| E4. ERBB2 CV within HER2+ group | {report['E4']:.4f} |

**Interpretation:** {"TSS batch effects remain significant (p < 0.05 on PC1)" if report['E2'] < 0.05 else "TSS batch effects not significant on PC1"}. \
PC1 correlation with read-depth proxy = {report['E3']:.3f} \
({"negligible" if abs(report['E3']) < 0.1 else "weak" if abs(report['E3']) < 0.3 else "moderate" if abs(report['E3']) < 0.5 else "strong"} \
read-depth confound in primary axis of variation). \
ERBB2 CV within HER2+ = {report['E4']:.3f} \
({"low" if report['E4'] < 0.15 else "moderate" if report['E4'] < 0.3 else "high"} \
within-group heterogeneity).

---

## Summary

| Category | Key Metric | Value |
|----------|-----------|-------|
| Signal: Effect size | Cohen's d (A5) | {report['A5']:.3f} |
| Signal: RNA predictiveness | AUC-ROC RNA (B1) | {report['B1']:.3f} |
| Signal: RNA advantage | Delta AUC (B7) | {report['B7']:+.3f} |
| Signal: Cluster quality | Silhouette k=4 (C3) | {report['C3']:.3f} |
| Signal: Marker separation | Mean gap (D4) | {report['D4']:.3f} |
| Signal: HER2 cluster purity | D5 | {report['D5']:.3f} |
| Noise: TSS batch | KW p (E2) | {report['E2']:.2e} |
| Noise: Read-depth confound | PC1 r (E3) | {report['E3']:.3f} |
| Noise: HER2+ heterogeneity | ERBB2 CV (E4) | {report['E4']:.3f} |

---

*Report generated by scripts/extract_rsem_report.py -- Phase 1 of normalization comparison.*
"""

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\nReport written to: {report_path}")
print("Done.")
