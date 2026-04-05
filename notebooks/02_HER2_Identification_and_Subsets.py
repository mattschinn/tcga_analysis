# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 02: HER2 Identification & Subsets
#
# **Tempus HER2 Coding Challenge**
#
# This notebook addresses:
# 1. Clinical HER2 definition recap (from Notebook 01)
# 2. Multimodal definition — ERBB2 RNA vs. copy number
# 3. RNA vs. DNA: which is more predictive of clinical IHC?
# 4. Discordant case identification and characterization
# 5. Unsupervised clustering for biologically distinct subsets
#
# **Inputs:** Intermediates from Notebook 01
# **Outputs:**
# - `02_multimodal_cohort.parquet` — merged clinical + RNA + CN
# - `02_analysis_df.parquet` — analysis-ready with key gene expression
# - `02_discordant_cases.parquet` — IHC/FISH/RNA/CN discordant patients
# - `02_cluster_assignments.parquet` — cluster labels for each k
# - `02_pca_embeddings.parquet` — PCA coordinates
# - `02_umap_embeddings.parquet` — UMAP coordinates (if available)

# %% [markdown]
# ---
# ## 1. Setup and Data Loading

# %%
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.utils import (
    load_intermediate, load_gene_cols, save_intermediate, savefig,
    setup_plotting, get_color, classify_her2_spectrum,
    COLORS, HER2_PATHWAY_GENES
)

setup_plotting()

# %%
# Load intermediates from Notebook 01
clinical = load_intermediate('01_clinical_qc')
tumor_norm = load_intermediate('01_tumor_norm')
cn = load_intermediate('01_cn_qc')
gene_cols = load_gene_cols()

print(f"\nClinical: {len(clinical)} patients")
print(f"Tumor norm: {len(tumor_norm)} samples × {len(gene_cols)} genes")
print(f"Copy number: {len(cn)} patients")

# %% [markdown]
# ---
# ## 2. Dataset Merging

# %%
print("=" * 70)
print("SEQUENTIAL DATASET MERGE")
print("=" * 70)

# Step 1: Clinical alone
print(f"\n1. Clinical data: {clinical['pid'].nunique()} unique patients")

# Step 2: Clinical ∩ RNA-Seq
clin_rna = clinical.merge(tumor_norm, on='pid', how='inner')
print(f"2. Clinical ∩ RNA-Seq: {len(clin_rna)} patients")

# Step 3: Clinical ∩ RNA-Seq ∩ Copy Number
clin_rna_cn = clin_rna.merge(cn[['pid', 'erbb2_copy_number']], on='pid', how='inner')
print(f"3. Clinical ∩ RNA-Seq ∩ CN: {len(clin_rna_cn)} patients (multimodal cohort)")

# Overlap analysis
rna_pids = set(tumor_norm['pid'])
cn_pids = set(cn['pid'])
clin_pids = set(clinical['pid'])

print(f"\nOverlap analysis:")
print(f"  In clinical but not RNA-Seq: {len(clin_pids - rna_pids)}")
print(f"  In RNA-Seq but not clinical: {len(rna_pids - clin_pids)}")
print(f"  In clinical but not CN: {len(clin_pids - cn_pids)}")
print(f"  Three-way intersection: {len(clin_pids & rna_pids & cn_pids)}")

# Define cohorts
cohort_c = clin_rna_cn.copy()  # Full multimodal cohort
cohort_a = cohort_c[cohort_c['her2_composite'].isin(['Positive', 'Negative'])]
equivocal = cohort_c[cohort_c['her2_composite'] == 'Equivocal']

print(f"\nCohort A (labeled, multimodal): {len(cohort_a)} patients")
pos_n = (cohort_a['her2_composite'] == 'Positive').sum()
neg_n = (cohort_a['her2_composite'] == 'Negative').sum()
print(f"  Positive: {pos_n}, Negative: {neg_n}, Ratio: {neg_n/max(pos_n,1):.1f}:1")
print(f"Equivocal (multimodal): {len(equivocal)} patients")

# %% [markdown]
# ---
# ## 3. ERBB2-Specific Analysis
#
# ### 3.1 ERBB2 Expression Distribution by HER2 Status

# %%
# Create analysis dataframe with key gene expression columns
analysis_df = cohort_c.copy()

# Add ERBB2 expression as a named column for easy access
if 'ERBB2' in gene_cols:
    analysis_df['ERBB2_expr'] = analysis_df['ERBB2']
    print(f"ERBB2 expression available: {analysis_df['ERBB2_expr'].notna().sum()} samples")
else:
    print("⚠ ERBB2 not found in gene columns!")

# Add other key genes
for gene in ['GRB7', 'ESR1', 'PGR', 'MKI67', 'EGFR', 'ERBB3']:
    if gene in gene_cols:
        analysis_df[f'{gene}_expr'] = analysis_df[gene]

# %%
# ERBB2 expression distribution by HER2 status
if 'ERBB2_expr' in analysis_df.columns:
    labeled = analysis_df[analysis_df['her2_composite'].isin(['Positive', 'Negative', 'Equivocal'])]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Violin + strip plot
    order = ['Negative', 'Equivocal', 'Positive']
    palette = {k: get_color(k) for k in order}
    
    parts = axes[0].violinplot(
        [labeled.loc[labeled['her2_composite'] == lab, 'ERBB2_expr'].dropna().values 
         for lab in order if (labeled['her2_composite'] == lab).sum() > 0],
        showmedians=True, showextrema=True
    )
    
    # Simpler approach: use seaborn
    axes[0].clear()
    sns.violinplot(data=labeled, x='her2_composite', y='ERBB2_expr', order=order,
                   palette=palette, inner='quartile', ax=axes[0], cut=0)
    sns.stripplot(data=labeled, x='her2_composite', y='ERBB2_expr', order=order,
                  color='black', size=2, alpha=0.3, jitter=True, ax=axes[0])
    
    # Mann-Whitney test
    pos_vals = labeled.loc[labeled['her2_composite'] == 'Positive', 'ERBB2_expr'].dropna()
    neg_vals = labeled.loc[labeled['her2_composite'] == 'Negative', 'ERBB2_expr'].dropna()
    if len(pos_vals) > 0 and len(neg_vals) > 0:
        u_stat, mw_p = stats.mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
        axes[0].set_title(f'ERBB2 Expression by HER2 Status\n(Mann-Whitney p = {mw_p:.2e})',
                         fontsize=13, fontweight='bold')
    else:
        axes[0].set_title('ERBB2 Expression by HER2 Status', fontsize=13, fontweight='bold')
    
    axes[0].set_xlabel('Clinical HER2 Status')
    axes[0].set_ylabel('ERBB2 Expression (log2 normalized)')
    
    # Histogram overlay
    axes[1].hist(neg_vals, bins=40, alpha=0.6, color=get_color('Negative'), 
                label=f'Negative (n={len(neg_vals)})', density=True)
    axes[1].hist(pos_vals, bins=40, alpha=0.6, color=get_color('Positive'),
                label=f'Positive (n={len(pos_vals)})', density=True)
    equi_vals = labeled.loc[labeled['her2_composite'] == 'Equivocal', 'ERBB2_expr'].dropna()
    if len(equi_vals) > 0:
        axes[1].hist(equi_vals, bins=20, alpha=0.6, color=get_color('Equivocal'),
                    label=f'Equivocal (n={len(equi_vals)})', density=True)
    axes[1].set_xlabel('ERBB2 Expression (log2 normalized)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Expression Distribution Overlap', fontsize=13, fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    savefig(fig, 'fig09_erbb2_by_her2_status')
    plt.show()

# %% [markdown]
# ### 3.2 ERBB2 Copy Number vs. IHC Concordance

# %%
print("=" * 70)
print("ERBB2 CN vs IHC CONCORDANCE")
print("=" * 70)

cn_labeled = analysis_df.dropna(subset=['her2_composite', 'erbb2_copy_number'])
cn_labeled = cn_labeled[cn_labeled['her2_composite'].isin(['Positive', 'Negative', 'Equivocal'])]

ct = pd.crosstab(cn_labeled['erbb2_copy_number'], cn_labeled['her2_composite'],
                 margins=True, margins_name='Total')
print("\nCrosstab: GISTIC CN × HER2 Status")
print(ct.to_string())

# Concordance rates
print("\nKey concordance rates:")
for gistic_val in sorted(cn_labeled['erbb2_copy_number'].unique()):
    subset = cn_labeled[cn_labeled['erbb2_copy_number'] == gistic_val]
    n_pos = (subset['her2_composite'] == 'Positive').sum()
    labels = ['Deep Del', 'Shallow Del', 'Diploid', 'Gain', 'Amp']
    label = labels[int(gistic_val) + 2] if -2 <= gistic_val <= 2 else str(gistic_val)
    print(f"  GISTIC {gistic_val:+d} ({label:>10s}): {n_pos}/{len(subset)} HER2+ ({100*n_pos/max(len(subset),1):.1f}%)")

# %%
# Heatmap visualization
fig, ax = plt.subplots(figsize=(8, 5))
ct_plot = pd.crosstab(cn_labeled['erbb2_copy_number'], cn_labeled['her2_composite'],
                       normalize='index') * 100
ct_plot = ct_plot[['Negative', 'Equivocal', 'Positive']] if 'Equivocal' in ct_plot.columns \
    else ct_plot[['Negative', 'Positive']]

sns.heatmap(ct_plot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
            cbar_kws={'label': '% of GISTIC category'})
ax.set_xlabel('Clinical HER2 Status')
ax.set_ylabel('GISTIC Copy Number')
ax.set_title('ERBB2 CN → HER2 Status Concordance (%)', fontsize=13, fontweight='bold')
plt.tight_layout()
savefig(fig, 'fig10_cn_ihc_concordance')
plt.show()

# %% [markdown]
# ### 3.3 ERBB2 RNA vs. Copy Number Scatter

# %%
if 'ERBB2_expr' in analysis_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, (subset_name, subset_df) in zip(axes, [
        ('Labeled (Pos/Neg)', analysis_df[analysis_df['her2_composite'].isin(['Positive', 'Negative'])]),
        ('All (incl. Equivocal)', analysis_df[analysis_df['her2_composite'].notna()])
    ]):
        for label in ['Negative', 'Equivocal', 'Positive']:
            mask = subset_df['her2_composite'] == label
            if mask.sum() > 0:
                ax.scatter(subset_df.loc[mask, 'erbb2_copy_number'] + np.random.uniform(-0.15, 0.15, mask.sum()),
                          subset_df.loc[mask, 'ERBB2_expr'],
                          s=20, alpha=0.5, label=f'{label} (n={mask.sum()})',
                          color=get_color(label))
        ax.set_xlabel('ERBB2 Copy Number (GISTIC)')
        ax.set_ylabel('ERBB2 RNA Expression (log2 normalized)')
        ax.set_title(subset_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
    
    plt.suptitle('ERBB2 RNA vs. Copy Number by Clinical HER2 Status', fontsize=14, fontweight='bold')
    plt.tight_layout()
    savefig(fig, 'fig11_rna_vs_cn_scatter')
    plt.show()

# %% [markdown]
# ---
# ## 4. RNA vs. DNA: Which Is More Predictive of Clinical IHC?
#
# We compare ERBB2 RNA expression, ERBB2 copy number, and a combined model for 
# predicting the clinical IHC-HER2 label using ROC-AUC analysis with 5-fold 
# stratified cross-validation.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("RNA vs DNA: PREDICTIVE COMPARISON FOR CLINICAL HER2")
print("=" * 70)

model_df = analysis_df.dropna(subset=['her2_composite', 'erbb2_copy_number']).copy()
model_df = model_df[model_df['her2_composite'].isin(['Positive', 'Negative'])]
model_df['y'] = (model_df['her2_composite'] == 'Positive').astype(int)

has_erbb2_expr = 'ERBB2_expr' in model_df.columns and model_df['ERBB2_expr'].notna().sum() > 10

if has_erbb2_expr:
    model_df = model_df.dropna(subset=['ERBB2_expr'])
    
    X_rna = model_df[['ERBB2_expr']].values
    X_cn = model_df[['erbb2_copy_number']].values
    X_both = model_df[['ERBB2_expr', 'erbb2_copy_number']].values
    y = model_df['y'].values
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"\nSamples: {len(model_df)} (Pos: {y.sum()}, Neg: {(1-y).sum()})")
    
    results = {}
    for name, X in [('ERBB2 RNA only', X_rna), ('ERBB2 CN only', X_cn), ('RNA + CN combined', X_both)]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        y_prob = cross_val_predict(lr, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
        
        auc_roc = roc_auc_score(y, y_prob)
        auc_pr = average_precision_score(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)
        
        results[name] = {'auc_roc': auc_roc, 'auc_pr': auc_pr, 'fpr': fpr, 'tpr': tpr, 'y_prob': y_prob}
        print(f"  {name:25s}: AUC-ROC = {auc_roc:.3f}, AUC-PR = {auc_pr:.3f}")
    
    best = max(results, key=lambda k: results[k]['auc_roc'])
    print(f"\n→ Best predictor: {best} (AUC-ROC = {results[best]['auc_roc']:.3f})")
    print(f"→ Combined model {'improves' if results['RNA + CN combined']['auc_roc'] > results[best]['auc_roc'] + 0.005 else 'does not improve'} over best single modality.")

# %%
# ROC curves
if has_erbb2_expr:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors_roc = {'ERBB2 RNA only': '#e74c3c', 'ERBB2 CN only': '#3498db', 'RNA + CN combined': '#2ecc71'}
    
    for name, res in results.items():
        axes[0].plot(res['fpr'], res['tpr'], color=colors_roc[name], linewidth=2,
                    label=f"{name} (AUC = {res['auc_roc']:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC: RNA vs CN for HER2 Prediction', fontsize=13, fontweight='bold')
    axes[0].legend(loc='lower right')
    
    # AUC comparison bar chart
    names = list(results.keys())
    auc_rocs = [results[n]['auc_roc'] for n in names]
    auc_prs = [results[n]['auc_pr'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    axes[1].bar(x - width/2, auc_rocs, width, label='AUC-ROC', color='#3498db')
    axes[1].bar(x + width/2, auc_prs, width, label='AUC-PR', color='#e74c3c')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Model Comparison', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    savefig(fig, 'fig12_rna_vs_cn_roc')
    plt.show()

# %% [markdown]
# **Interpretation:** ERBB2 RNA expression outperforms copy number for predicting clinical
# HER2 status. The combined model offers no improvement, indicating that copy number's 
# predictive information is largely captured by RNA expression. This is consistent with 
# the known literature: RNA captures not only amplification-driven overexpression but also 
# transcriptional regulation effects that CN misses (epigenetic modulation, enhancer 
# activation). The notably lower AUC-PR for copy number reflects poor precision in the 
# minority class, due to the coarse granularity of GISTIC discrete values (-2 to +2).

# %% [markdown]
# ### 4.1 Subgroup Analysis: IHC 2+ (Equivocal) Patients
#
# Among IHC 2+ patients with FISH data, does RNA expression better predict FISH 
# outcome than CN? This is the clinically relevant question — RNA's added value 
# is greatest in the equivocal zone.

# %%
print("=" * 70)
print("SUBGROUP ANALYSIS: IHC 2+ PATIENTS")
print("=" * 70)

# Identify IHC 2+ patients with FISH results
ihc2_mask = analysis_df['HER2 ihc score'].apply(
    lambda x: str(x).strip().rstrip('+') in ('2', '2.0') if pd.notna(x) else False
)
fish_mask = analysis_df['HER2 fish status'].isin(['Positive', 'Negative'])
ihc2_with_fish = analysis_df[ihc2_mask & fish_mask].copy()

print(f"IHC 2+ patients with FISH result: {len(ihc2_with_fish)}")

if len(ihc2_with_fish) >= 10 and 'ERBB2_expr' in ihc2_with_fish.columns:
    ihc2_with_fish['fish_positive'] = (ihc2_with_fish['HER2 fish status'] == 'Positive').astype(int)
    
    print(f"  FISH Positive: {ihc2_with_fish['fish_positive'].sum()}")
    print(f"  FISH Negative: {(1 - ihc2_with_fish['fish_positive']).sum()}")
    
    if ihc2_with_fish['fish_positive'].nunique() > 1:
        # Compare RNA vs CN for predicting FISH in this subset
        rna_vals = ihc2_with_fish['ERBB2_expr'].dropna()
        cn_vals_sub = ihc2_with_fish['erbb2_copy_number']
        fish_vals = ihc2_with_fish.loc[rna_vals.index, 'fish_positive']
        
        if len(rna_vals) > 5:
            rna_auc = roc_auc_score(fish_vals, rna_vals)
            cn_auc = roc_auc_score(fish_vals, cn_vals_sub.loc[rna_vals.index])
            print(f"\n  RNA AUC for predicting FISH (in IHC 2+ subset): {rna_auc:.3f}")
            print(f"  CN AUC for predicting FISH (in IHC 2+ subset):  {cn_auc:.3f}")
            print(f"\n  → {'RNA' if rna_auc > cn_auc else 'CN'} is more predictive in the equivocal zone.")
else:
    print("  Insufficient IHC 2+ patients with FISH for subgroup analysis.")

# %% [markdown]
# ---
# ## 5. Discordant Case Identification
#
# We identify patients with discordance between modalities — these are the most
# clinically interesting cases and potential targets for RNA-based reclassification.

# %%
print("=" * 70)
print("DISCORDANT CASE IDENTIFICATION")
print("=" * 70)

discordant_records = []

if 'ERBB2_expr' in analysis_df.columns:
    # Define thresholds based on labeled population
    pos_expr = analysis_df.loc[analysis_df['her2_composite'] == 'Positive', 'ERBB2_expr'].dropna()
    neg_expr = analysis_df.loc[analysis_df['her2_composite'] == 'Negative', 'ERBB2_expr'].dropna()
    
    if len(pos_expr) > 0 and len(neg_expr) > 0:
        pos_median = pos_expr.median()
        neg_median = neg_expr.median()
        # Threshold: midpoint between medians, or negative's 95th percentile
        rna_threshold_high = neg_expr.quantile(0.95)
        rna_threshold_low = pos_expr.quantile(0.05)
        
        print(f"ERBB2 expression thresholds:")
        print(f"  Positive median: {pos_median:.2f}")
        print(f"  Negative median: {neg_median:.2f}")
        print(f"  High threshold (Neg 95th pctl): {rna_threshold_high:.2f}")
        print(f"  Low threshold (Pos 5th pctl): {rna_threshold_low:.2f}")
        
        for idx, row in analysis_df.iterrows():
            pid = row['pid']
            ihc = row.get('her2_composite', np.nan)
            expr = row.get('ERBB2_expr', np.nan)
            cn_val = row.get('erbb2_copy_number', np.nan)
            ihc_score = row.get('HER2 ihc score', np.nan)
            fish = row.get('HER2 fish status', np.nan)
            grb7 = row.get('GRB7_expr', np.nan) if 'GRB7_expr' in row.index else np.nan
            
            disc_types = []
            
            # IHC+ / RNA-low
            if ihc == 'Positive' and pd.notna(expr) and expr < rna_threshold_low:
                disc_types.append('IHC+/RNA-low')
            
            # IHC- / RNA-high
            if ihc == 'Negative' and pd.notna(expr) and expr > rna_threshold_high:
                disc_types.append('IHC-/RNA-high')
            
            # CN-high / RNA-low
            if pd.notna(cn_val) and cn_val >= 2 and pd.notna(expr) and expr < neg_median:
                disc_types.append('CN-high/RNA-low')
            
            # CN-low / RNA-high
            if pd.notna(cn_val) and cn_val <= 0 and pd.notna(expr) and expr > rna_threshold_high:
                disc_types.append('CN-low/RNA-high')
            
            # IHC 3+ / FISH-
            if str(ihc_score).strip().rstrip('+') in ('3', '3.0') and str(fish).lower() == 'negative':
                disc_types.append('IHC3+/FISH-')
            
            # IHC 0-1+ / FISH+
            if str(ihc_score).strip().rstrip('+') in ('0', '1', '0.0', '1.0') and str(fish).lower() == 'positive':
                disc_types.append('IHC-low/FISH+')
            
            for dt in disc_types:
                discordant_records.append({
                    'pid': pid,
                    'discordance_type': dt,
                    'her2_composite': ihc,
                    'ERBB2_expr': expr,
                    'erbb2_copy_number': cn_val,
                    'HER2_ihc_score': ihc_score,
                    'HER2_fish_status': fish,
                    'GRB7_expr': grb7,
                })

discordant_df = pd.DataFrame(discordant_records)

if len(discordant_df) > 0:
    print(f"\nDiscordant cases identified: {discordant_df['pid'].nunique()} unique patients")
    print(f"\nDiscordance type breakdown:")
    for dtype, count in discordant_df['discordance_type'].value_counts().items():
        n_unique = discordant_df.loc[discordant_df['discordance_type'] == dtype, 'pid'].nunique()
        print(f"  {dtype:20s}: {n_unique} patients")
        
        # Biological interpretation
        if dtype == 'IHC-/RNA-high':
            print(f"    → Potential missed HER2+ patients — candidates for RNA-based reclassification")
        elif dtype == 'IHC+/RNA-low':
            print(f"    → IHC positive but low RNA — possible antibody artifact or post-translational regulation")
        elif dtype == 'IHC3+/FISH-':
            print(f"    → Possible polysomy 17 (centromere gain without ERBB2 amplification)")
        elif dtype == 'IHC-low/FISH+':
            print(f"    → Amplification without protein overexpression (epigenetic silencing?)")
        elif dtype == 'CN-high/RNA-low':
            print(f"    → Gene amplification without transcription")
        elif dtype == 'CN-low/RNA-high':
            print(f"    → Transcriptional upregulation without amplification")
else:
    print("No discordant cases identified.")

# %%
# GRB7 co-expression check for discordant cases
if len(discordant_df) > 0 and 'GRB7_expr' in discordant_df.columns:
    print("\nGRB7 co-expression in discordant cases (17q12 amplicon validation):")
    for dtype in discordant_df['discordance_type'].unique():
        subset = discordant_df[discordant_df['discordance_type'] == dtype]
        grb7_vals = subset['GRB7_expr'].dropna()
        if len(grb7_vals) > 0:
            print(f"  {dtype:20s}: GRB7 median = {grb7_vals.median():.2f} "
                  f"(cohort median = {analysis_df.get('GRB7_expr', pd.Series()).median():.2f})")

# %% [markdown]
# ---
# ## 6. Unsupervised Clustering
#
# ### 6.1 Feature Selection and Dimensionality Reduction

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("=" * 70)
print("UNSUPERVISED CLUSTERING")
print("=" * 70)

# Use normalized tumor expression for clustering
# Select most variable genes by MAD
tumor_for_clustering = tumor_norm.copy()
gene_mad = tumor_for_clustering[gene_cols].apply(lambda x: np.median(np.abs(x - np.median(x))), axis=0)
gene_mad_sorted = gene_mad.sort_values(ascending=False)

n_top_genes = min(len(gene_cols), 3000)
top_genes = gene_mad_sorted.head(n_top_genes).index.tolist()
print(f"Using top {len(top_genes)} most variable genes (by MAD)")

# Prepare and standardize
X_cluster = tumor_for_clustering[top_genes].fillna(0).values
patient_ids_cluster = tumor_for_clustering['pid'].values

scaler_cl = StandardScaler()
X_scaled_cl = scaler_cl.fit_transform(X_cluster)

# PCA
n_pcs = min(20, X_scaled_cl.shape[0], X_scaled_cl.shape[1])
pca_cl = PCA(n_components=n_pcs)
X_pca = pca_cl.fit_transform(X_scaled_cl)

cumvar = np.cumsum(pca_cl.explained_variance_ratio_) * 100
n_pcs_use = np.argmax(cumvar >= 90) + 1
n_pcs_use = max(n_pcs_use, 5)  # at least 5 PCs
print(f"\nPCA: {n_pcs} components")
print(f"  Variance explained by first 5 PCs: {cumvar[4]:.1f}%")
print(f"  Variance explained by first 10 PCs: {cumvar[min(9, n_pcs-1)]:.1f}%")
print(f"  PCs for ≥90% variance: {n_pcs_use}")

# %%
# UMAP for visualization
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = reducer.fit_transform(X_pca[:, :min(10, n_pcs_use)])
    has_umap = True
    print("UMAP embedding computed successfully.")
except ImportError:
    print("umap-learn not available. Using PCA for visualization.")
    X_umap = X_pca[:, :2]
    has_umap = False

# %% [markdown]
# ### 6.2 Optimal k Selection

# %%
n_pcs_for_clustering = min(10, X_pca.shape[1])
X_for_clustering = X_pca[:, :n_pcs_for_clustering]

k_range = range(2, min(8, len(X_for_clustering) // 5))
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_for_clustering)
    sil = silhouette_score(X_for_clustering, labels)
    silhouette_scores.append(sil)

best_k = list(k_range)[np.argmax(silhouette_scores)]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(k_range), silhouette_scores, 'o-', color='#3498db', linewidth=2, markersize=8)
ax.axvline(best_k, color='red', linestyle='--', alpha=0.5, label=f'Best k = {best_k}')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Optimal k: Silhouette Analysis', fontsize=13, fontweight='bold')
ax.set_xticks(list(k_range))
ax.legend()
plt.tight_layout()
savefig(fig, 'fig13_silhouette_analysis')
plt.show()

print(f"Best k by silhouette: {best_k} (score = {max(silhouette_scores):.3f})")
print(f"\nNote: In breast cancer, we expect ~4-5 intrinsic subtypes (Luminal A, Luminal B,")
print(f"HER2-enriched, Basal-like, Normal-like). Testing both data-driven k and k=4.")

# %% [markdown]
# ### 6.3 Clustering Results

# %%
# Map pids to clinical labels for characterization
pid_to_her2 = dict(zip(clinical['pid'], clinical['her2_composite']))
pid_to_er = dict(zip(clinical['pid'], clinical['ER Status By IHC']))

cluster_her2 = [pid_to_her2.get(pid, np.nan) for pid in patient_ids_cluster]
cluster_er = [pid_to_er.get(pid, np.nan) for pid in patient_ids_cluster]

# Get TSS for batch check
pid_to_tss = dict(zip(clinical['pid'], clinical.get('tss', pd.Series(dtype=str))))
cluster_tss = [pid_to_tss.get(pid, np.nan) for pid in patient_ids_cluster]

# Run clustering for multiple k values
k_values = [best_k, 4, 5]
k_values = sorted(set([k for k in k_values if 2 <= k <= 7]))

all_cluster_labels = {}

for k_use in k_values:
    label_name = f'k{k_use}'
    km = KMeans(n_clusters=k_use, random_state=42, n_init=20)
    cluster_labels = km.fit_predict(X_for_clustering)
    all_cluster_labels[label_name] = cluster_labels
    
    print(f"\n{'=' * 70}")
    print(f"CLUSTERING RESULTS: k = {k_use}")
    print(f"{'=' * 70}")
    
    for c in range(k_use):
        n_c = (cluster_labels == c).sum()
        print(f"  Cluster {c}: {n_c} samples ({100*n_c/len(cluster_labels):.1f}%)")
    
    # HER2 enrichment
    print(f"\nHER2 Enrichment by Cluster (Fisher's exact test):")
    for c in range(k_use):
        in_cluster = cluster_labels == c
        her2_in = np.array([h == 'Positive' for h, ic in zip(cluster_her2, in_cluster) if h in ('Positive', 'Negative') and ic])
        her2_out = np.array([h == 'Positive' for h, ic in zip(cluster_her2, in_cluster) if h in ('Positive', 'Negative') and not ic])
        
        if len(her2_in) > 0 and len(her2_out) > 0:
            table = np.array([[her2_in.sum(), len(her2_in) - her2_in.sum()],
                             [her2_out.sum(), len(her2_out) - her2_out.sum()]])
            odds_ratio, p_val = stats.fisher_exact(table)
            pct_pos = 100 * her2_in.sum() / len(her2_in)
            print(f"  Cluster {c}: {her2_in.sum()}/{len(her2_in)} HER2+ ({pct_pos:.0f}%), "
                  f"Fisher p = {p_val:.4g}, OR = {odds_ratio:.2f}")
    
    # ER status distribution
    print(f"\nER Status by Cluster:")
    for c in range(k_use):
        in_cluster = cluster_labels == c
        er_in = [e for e, ic in zip(cluster_er, in_cluster) if ic and e in ('Positive', 'Negative')]
        if len(er_in) > 0:
            pct_er_pos = 100 * sum(1 for e in er_in if e == 'Positive') / len(er_in)
            print(f"  Cluster {c}: {pct_er_pos:.0f}% ER+ (n={len(er_in)})")
    
    # TSS distribution check (batch effect)
    print(f"\nTSS diversity by Cluster (batch check):")
    for c in range(k_use):
        in_cluster = cluster_labels == c
        tss_in = [t for t, ic in zip(cluster_tss, in_cluster) if ic and pd.notna(t)]
        n_tss_in = len(set(tss_in))
        print(f"  Cluster {c}: {n_tss_in} unique TSS")

# %%
# Visualize clustering (best k and k=4)
dim_label = 'UMAP' if has_umap else 'PCA'

for k_use in [best_k, 4]:
    if f'k{k_use}' not in all_cluster_labels:
        continue
    labels = all_cluster_labels[f'k{k_use}']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # By cluster
    for c in range(k_use):
        mask = labels == c
        axes[0].scatter(X_umap[mask, 0], X_umap[mask, 1], s=15, alpha=0.6, label=f'Cluster {c}')
    axes[0].set_xlabel(f'{dim_label} 1')
    axes[0].set_ylabel(f'{dim_label} 2')
    axes[0].set_title(f'Clusters (k={k_use})')
    axes[0].legend(fontsize=8)
    
    # By HER2 status
    for label in ['Negative', 'Equivocal', 'Positive']:
        mask = np.array([h == label for h in cluster_her2])
        if mask.sum() > 0:
            axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1], s=15, alpha=0.5,
                          label=label, color=get_color(label))
    axes[1].set_xlabel(f'{dim_label} 1')
    axes[1].set_title('HER2 Status')
    axes[1].legend(fontsize=8)
    
    # By ER status
    er_colors = {'Positive': '#2ecc71', 'Negative': '#9b59b6'}
    for label in ['Positive', 'Negative']:
        mask = np.array([e == label for e in cluster_er])
        if mask.sum() > 0:
            axes[2].scatter(X_umap[mask, 0], X_umap[mask, 1], s=15, alpha=0.5,
                          label=f'ER {label}', color=er_colors.get(label, 'gray'))
    axes[2].set_xlabel(f'{dim_label} 1')
    axes[2].set_title('ER Status')
    axes[2].legend(fontsize=8)
    
    plt.suptitle(f'Unsupervised Clustering (k={k_use})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    savefig(fig, f'fig14_clustering_k{k_use}')
    plt.show()

# %% [markdown]
# ---
# ## 7. Save Intermediates

# %%
print("=" * 70)
print("SAVING INTERMEDIATES")
print("=" * 70)

# Multimodal cohort
save_intermediate(cohort_c, '02_multimodal_cohort')

# Analysis dataframe
save_intermediate(analysis_df, '02_analysis_df')

# Discordant cases
if len(discordant_df) > 0:
    save_intermediate(discordant_df, '02_discordant_cases')

# Cluster assignments
cluster_df = pd.DataFrame({'pid': patient_ids_cluster})
for label_name, labels in all_cluster_labels.items():
    cluster_df[f'cluster_{label_name}'] = labels
save_intermediate(cluster_df, '02_cluster_assignments')

# PCA embeddings
pca_embed_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_embed_df['pid'] = patient_ids_cluster
save_intermediate(pca_embed_df, '02_pca_embeddings')

# UMAP embeddings
umap_embed_df = pd.DataFrame(X_umap, columns=[f'{dim_label}1', f'{dim_label}2'])
umap_embed_df['pid'] = patient_ids_cluster
save_intermediate(umap_embed_df, '02_umap_embeddings')

print("\n✓ All Notebook 02 intermediates saved.")

# %% [markdown]
# ---
# ## Summary
#
# **ERBB2 Expression by HER2 Status:**
# - Clear separation between HER2-Positive and HER2-Negative populations.
# - Equivocal cases show intermediate, heterogeneous expression.
#
# **RNA vs. DNA Predictiveness:**
# - ERBB2 RNA outperforms copy number for predicting clinical HER2 status.
# - Combined model does not improve over RNA alone — CN's information is captured in RNA.
# - This is consistent with literature: RNA captures transcriptional regulation beyond amplification.
#
# **Discordant Cases:**
# - Multiple discordance patterns identified with distinct biological mechanisms.
# - IHC-/RNA-high cases are potential missed HER2+ patients — candidates for reclassification.
# - GRB7 co-expression provides 17q12 amplicon validation.
#
# **Unsupervised Clustering:**
# - Clusters tested at data-driven k and biology-motivated k=4.
# - HER2 enrichment assessed per cluster with Fisher's exact test.
# - TSS diversity checked per cluster to flag potential batch artifacts.
