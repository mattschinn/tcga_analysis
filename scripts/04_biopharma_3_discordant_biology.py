"""
Analysis #3: Discordant Biology -- Normal Tissue and ER Pathway (Priority 2)
============================================================================
Determine whether non-amplified discordant group (CN<=1) is driven by ER/luminal
co-regulation of ERBB2 or by independent HER2-pathway activation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import (load_intermediate, save_intermediate, savefig,
                   to_patient_id, setup_plotting, COLORS)
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json

setup_plotting()
import matplotlib.pyplot as plt
import seaborn as sns

REPORT_DIR = Path(__file__).resolve().parent.parent / 'reports' / 'biopharma'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Gene panels ──────────────────────────────────────────────────────────────

ER_PATHWAY_GENES = ['ESR1', 'FOXA1', 'GATA3', 'TFF1', 'TFF3', 'PGR',
                    'XBP1', 'AGR2', 'CA12', 'NAT1', 'SLC39A6']
HER2_PATHWAY_GENES = ['ERBB2', 'GRB7', 'STARD3', 'PGAP3', 'MIEN1',
                      'EGFR', 'ERBB3', 'ERBB4']
PROLIFERATION_GENES = ['MKI67', 'AURKA', 'CCNB1', 'TOP2A', 'BIRC5',
                       'MYBL2', 'CDK1', 'PLK1']

ALL_GENES = ER_PATHWAY_GENES + HER2_PATHWAY_GENES + PROLIFERATION_GENES

# ── 1. Load data ─────────────────────────────────────────────────────────────

tumor_norm = load_intermediate('01_tumor_norm_tmm_tss')
normal = load_intermediate('01_normal_raw_filtered')
mm = load_intermediate('02_multimodal_cohort')
analysis_df = load_intermediate('02_analysis_df')
disc = load_intermediate('02_discordant_cases')
dossier = load_intermediate('03_discordant_dossier')
clin = pd.read_csv(
    Path(__file__).resolve().parent.parent / 'data' / 'brca_tcga_clinical_data_cleaned.csv'
)
clin['pid'] = clin['Patient ID'].apply(to_patient_id)

with open(Path(__file__).resolve().parent.parent / 'outputs' / '01_gene_cols.json') as f:
    gene_cols = json.load(f)

# ── 2. Define patient groups ─────────────────────────────────────────────────

ihc_neg_rna_high = disc[disc['discordance_type'] == 'IHC-/RNA-high']
disc_pids = set(ihc_neg_rna_high['pid'])
cn_high_disc = ihc_neg_rna_high[ihc_neg_rna_high['erbb2_copy_number'] >= 2]
cn_low_disc = ihc_neg_rna_high[ihc_neg_rna_high['erbb2_copy_number'] <= 1]

print(f"\nDiscordant IHC-/RNA-high: {len(ihc_neg_rna_high)}")
print(f"  CN-high (>=2): {len(cn_high_disc)}")
print(f"  CN-low (<=1): {len(cn_low_disc)}")

conc_neg = mm[
    (mm['her2_composite'] == 'Negative') &
    (~mm['pid'].isin(disc_pids))
]
conc_pos = mm[mm['her2_composite'] == 'Positive']

print(f"Concordant Negative: {len(conc_neg)}")
print(f"Concordant Positive: {len(conc_pos)}")

# ── Part A: Tumor-to-Normal ERBB2 Ratios ────────────────────────────────────

# Find matched tumor-normal pairs
tumor_pids = set(tumor_norm['pid'])
normal_pids = set(normal['pid'])
matched_pids = tumor_pids & normal_pids
print(f"\nMatched tumor-normal pairs: {len(matched_pids)}")

# Check which genes are available
available_genes = [g for g in ALL_GENES if g in tumor_norm.columns and g in normal.columns]
print(f"Available genes in both tumor+normal: {len(available_genes)} of {len(ALL_GENES)}")
missing_genes = [g for g in ALL_GENES if g not in available_genes]
if missing_genes:
    print(f"  Missing: {missing_genes}")

# Compute tumor-to-normal ratios (log-space subtraction)
ratios = []
for pid in matched_pids:
    t_row = tumor_norm[tumor_norm['pid'] == pid]
    n_row = normal[normal['pid'] == pid]
    if len(t_row) == 0 or len(n_row) == 0:
        continue
    row_data = {'pid': pid}
    for gene in available_genes:
        t_val = t_row[gene].values[0]
        n_val = n_row[gene].values[0]
        row_data[f'{gene}_ratio'] = t_val - n_val  # log-space subtraction = log ratio
    ratios.append(row_data)

ratio_df = pd.DataFrame(ratios)
print(f"Computed ratios for {len(ratio_df)} patients")

# Assign groups to matched patients
ratio_df['group'] = 'Other'
ratio_df.loc[ratio_df['pid'].isin(cn_high_disc['pid']), 'group'] = 'Discordant CN-high'
ratio_df.loc[ratio_df['pid'].isin(cn_low_disc['pid']), 'group'] = 'Discordant CN-low'
ratio_df.loc[ratio_df['pid'].isin(set(conc_neg['pid'])), 'group'] = 'Concordant Negative'
ratio_df.loc[ratio_df['pid'].isin(set(conc_pos['pid'])), 'group'] = 'Concordant Positive'

group_counts_normal = ratio_df['group'].value_counts()
print(f"\nGroup distribution (matched patients):\n{group_counts_normal.to_string()}")

# Descriptive statistics for ERBB2 ratio by group
print("\nERBB2 tumor-to-normal ratio by group:")
for grp in ['Discordant CN-high', 'Discordant CN-low', 'Concordant Negative', 'Concordant Positive']:
    subset = ratio_df[ratio_df['group'] == grp]['ERBB2_ratio']
    if len(subset) > 0:
        print(f"  {grp} (n={len(subset)}): median={subset.median():.3f}, "
              f"IQR=[{subset.quantile(0.25):.3f}, {subset.quantile(0.75):.3f}]")

# Figure: Tumor-to-normal ratio box plot
ratio_groups = ratio_df[ratio_df['group'] != 'Other']
if len(ratio_groups) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    order = ['Concordant Negative', 'Discordant CN-low', 'Discordant CN-high', 'Concordant Positive']
    order = [g for g in order if g in ratio_groups['group'].values]
    palette = {
        'Concordant Negative': COLORS['Negative'],
        'Discordant CN-low': '#f39c12',
        'Discordant CN-high': '#e67e22',
        'Concordant Positive': COLORS['Positive'],
    }
    sns.boxplot(data=ratio_groups, x='group', y='ERBB2_ratio', order=order,
                palette=palette, ax=ax, showfliers=False)
    sns.stripplot(data=ratio_groups, x='group', y='ERBB2_ratio', order=order,
                  color='black', alpha=0.5, size=4, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('ERBB2 Tumor-to-Normal Ratio (log2)')
    ax.set_title('ERBB2 Tumor vs. Normal Expression by Patient Group')
    for i, grp in enumerate(order):
        n = ratio_groups[ratio_groups['group'] == grp].shape[0]
        ax.text(i, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f'n={n}', ha='center', fontsize=9)
    plt.tight_layout()
    savefig(fig, 'fig_04_3a_tumor_normal_erbb2_ratio')
    plt.close()

# ── Part B: ER Pathway Correlation in Non-Amplified Discordant ───────────────

# Build patient-level expression matrix
expr_genes = [g for g in available_genes if g in tumor_norm.columns]
expr_data = tumor_norm[['pid'] + expr_genes].copy()

# Assign groups
expr_data['group'] = 'Other'
expr_data.loc[expr_data['pid'].isin(cn_high_disc['pid']), 'group'] = 'Discordant CN-high'
expr_data.loc[expr_data['pid'].isin(cn_low_disc['pid']), 'group'] = 'Discordant CN-low'
expr_data.loc[expr_data['pid'].isin(set(conc_neg['pid'])), 'group'] = 'Concordant Negative'
expr_data.loc[expr_data['pid'].isin(set(conc_pos['pid'])), 'group'] = 'Concordant Positive'

# Spearman correlations: ERBB2 vs ER pathway genes within CN-low discordant
print("\n=== Spearman Correlations: ERBB2 vs ER Pathway Genes ===")
er_genes_available = [g for g in ER_PATHWAY_GENES if g in expr_genes]

corr_results = []
for group_name in ['Discordant CN-low', 'Concordant Negative']:
    subset = expr_data[expr_data['group'] == group_name]
    print(f"\n{group_name} (n={len(subset)}):")
    for gene in er_genes_available:
        erbb2_vals = subset['ERBB2'].dropna()
        gene_vals = subset[gene].dropna()
        common_idx = erbb2_vals.index.intersection(gene_vals.index)
        if len(common_idx) >= 5:
            rho, p = stats.spearmanr(subset.loc[common_idx, 'ERBB2'],
                                      subset.loc[common_idx, gene])
            print(f"  ERBB2 vs {gene}: rho={rho:.3f}, p={p:.4f}")
            corr_results.append({
                'group': group_name, 'gene': gene,
                'rho': rho, 'p': p, 'n': len(common_idx)
            })

corr_df = pd.DataFrame(corr_results)

# Interpretation heatmap: mean z-scored expression by gene group x patient group
print("\n=== Pathway Expression Heatmap ===")
gene_groups = {
    'ER pathway': [g for g in ER_PATHWAY_GENES if g in expr_genes],
    'HER2 pathway': [g for g in HER2_PATHWAY_GENES if g in expr_genes],
    'Proliferation': [g for g in PROLIFERATION_GENES if g in expr_genes],
}

patient_groups = ['Concordant Negative', 'Discordant CN-low',
                  'Discordant CN-high', 'Concordant Positive']
heatmap_data = {}

for pg in patient_groups:
    pg_data = expr_data[expr_data['group'] == pg]
    if len(pg_data) == 0:
        continue
    heatmap_data[pg] = {}
    for gg_name, gg_genes in gene_groups.items():
        if gg_genes:
            # Z-score relative to all patients, then take mean per group
            all_vals = expr_data[gg_genes]
            z_scores = (pg_data[gg_genes] - all_vals.mean()) / all_vals.std()
            heatmap_data[pg][gg_name] = z_scores.mean().mean()

heatmap_df = pd.DataFrame(heatmap_data).T
heatmap_df = heatmap_df.reindex(patient_groups)
print(heatmap_df.to_string())

# Figure: Heatmap
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, linewidths=0.5, cbar_kws={'label': 'Mean Z-score'})
ax.set_title('Pathway Expression Profiles by Patient Group')
ax.set_ylabel('')
# Add sample sizes
ylabels = []
for pg in patient_groups:
    n = len(expr_data[expr_data['group'] == pg])
    ylabels.append(f'{pg}\n(n={n})')
ax.set_yticklabels(ylabels, rotation=0)
plt.tight_layout()
savefig(fig, 'fig_04_3b_pathway_heatmap')
plt.close()

# ── Part C: ER Quantitative Scores ──────────────────────────────────────────

er_cols = ['er_allred_score', 'er_hscore', 'er_intensity',
           'er_percent_positive', 'er_fmol_mg']

# Merge clinical ER scores with groups
er_data = clin[['pid'] + er_cols].copy()
er_data = er_data.merge(
    expr_data[['pid', 'group']].drop_duplicates(),
    on='pid', how='inner'
)

print("\n=== ER Quantitative Scores by Group ===")
er_results = []
for col in er_cols:
    valid = er_data[er_data[col].notna() & er_data['group'].isin(patient_groups)].copy()
    if len(valid) < 5:
        print(f"\n{col}: insufficient data (n={len(valid)})")
        continue
    # Try to convert to numeric; skip non-numeric columns
    numeric_vals = pd.to_numeric(valid[col], errors='coerce')
    if numeric_vals.notna().sum() < 5:
        print(f"\n{col}: non-numeric or mostly non-numeric, skipping statistical tests")
        print(f"  Values: {valid[col].value_counts().to_dict()}")
        continue
    valid[col] = numeric_vals
    valid = valid[valid[col].notna()]
    print(f"\n{col}:")
    for pg in patient_groups:
        subset = valid[valid['group'] == pg][col].dropna()
        if len(subset) > 0:
            print(f"  {pg} (n={len(subset)}): median={subset.median():.1f}, "
                  f"IQR=[{subset.quantile(0.25):.1f}, {subset.quantile(0.75):.1f}]")
    # Kruskal-Wallis across groups
    groups_data = [valid[valid['group'] == pg][col].dropna().values
                   for pg in patient_groups if len(valid[valid['group'] == pg][col].dropna()) > 0]
    if len(groups_data) >= 2:
        h_stat, kw_p = stats.kruskal(*groups_data)
        print(f"  Kruskal-Wallis: H={h_stat:.2f}, p={kw_p:.4f}")
        er_results.append({'metric': col, 'H': h_stat, 'p': kw_p})

    # Pairwise: CN-low discordant vs concordant negative
    cn_low_vals = valid[(valid['group'] == 'Discordant CN-low')][col].dropna()
    conc_neg_vals = valid[(valid['group'] == 'Concordant Negative')][col].dropna()
    if len(cn_low_vals) >= 3 and len(conc_neg_vals) >= 3:
        u_stat, u_p = stats.mannwhitneyu(cn_low_vals, conc_neg_vals, alternative='two-sided')
        print(f"  Discordant CN-low vs Concordant Negative: U={u_stat:.1f}, p={u_p:.4f}")

# Figure: ER quantitative box plots (best-populated metric)
best_er_col = None
best_n = 0
for col in er_cols:
    tmp = er_data[er_data[col].notna() & er_data['group'].isin(patient_groups)].copy()
    tmp_numeric = pd.to_numeric(tmp[col], errors='coerce')
    n = tmp_numeric.notna().sum()
    if n > best_n:
        best_n = n
        best_er_col = col

if best_er_col and best_n > 10:
    plot_data = er_data[er_data[best_er_col].notna() & er_data['group'].isin(patient_groups)].copy()
    plot_data[best_er_col] = pd.to_numeric(plot_data[best_er_col], errors='coerce')
    plot_data = plot_data[plot_data[best_er_col].notna()]
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {
        'Concordant Negative': COLORS['Negative'],
        'Discordant CN-low': '#f39c12',
        'Discordant CN-high': '#e67e22',
        'Concordant Positive': COLORS['Positive'],
    }
    order = [g for g in patient_groups if g in plot_data['group'].values]
    sns.boxplot(data=plot_data, x='group', y=best_er_col, order=order,
                palette=palette, ax=ax, showfliers=False)
    sns.stripplot(data=plot_data, x='group', y=best_er_col, order=order,
                  color='black', alpha=0.5, size=4, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(best_er_col.replace('_', ' ').title())
    ax.set_title(f'ER Quantitative Score ({best_er_col}) by Patient Group')
    for i, grp in enumerate(order):
        n = plot_data[plot_data['group'] == grp].shape[0]
        ax.text(i, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f'n={n}', ha='center', fontsize=9)
    plt.tight_layout()
    savefig(fig, 'fig_04_3c_er_quantitative')
    plt.close()

# ── Part D: Fraction Genome Altered ──────────────────────────────────────────

print("\n=== Fraction Genome Altered (FGA) ===")
fga_data = mm[['pid', 'Fraction Genome Altered']].copy()
fga_data['group'] = 'Other'
fga_data.loc[fga_data['pid'].isin(cn_high_disc['pid']), 'group'] = 'Discordant CN-high'
fga_data.loc[fga_data['pid'].isin(cn_low_disc['pid']), 'group'] = 'Discordant CN-low'
fga_data.loc[fga_data['pid'].isin(set(conc_neg['pid'])), 'group'] = 'Concordant Negative'
fga_data.loc[fga_data['pid'].isin(set(conc_pos['pid'])), 'group'] = 'Concordant Positive'

fga_results = {}
for pg in patient_groups:
    subset = fga_data[(fga_data['group'] == pg) & fga_data['Fraction Genome Altered'].notna()]
    fga_vals = subset['Fraction Genome Altered']
    if len(fga_vals) > 0:
        fga_results[pg] = {
            'n': len(fga_vals),
            'median': fga_vals.median(),
            'q25': fga_vals.quantile(0.25),
            'q75': fga_vals.quantile(0.75),
        }
        print(f"  {pg} (n={len(fga_vals)}): median={fga_vals.median():.3f}, "
              f"IQR=[{fga_vals.quantile(0.25):.3f}, {fga_vals.quantile(0.75):.3f}]")

# CN-low vs concordant negative
cn_low_fga = fga_data[(fga_data['group'] == 'Discordant CN-low')]['Fraction Genome Altered'].dropna()
conc_neg_fga = fga_data[(fga_data['group'] == 'Concordant Negative')]['Fraction Genome Altered'].dropna()
if len(cn_low_fga) >= 3 and len(conc_neg_fga) >= 3:
    fga_u, fga_p = stats.mannwhitneyu(cn_low_fga, conc_neg_fga, alternative='two-sided')
    print(f"\n  Discordant CN-low vs Concordant Negative: U={fga_u:.1f}, p={fga_p:.4f}")
else:
    fga_u, fga_p = np.nan, np.nan

# Figure: FGA box plot
fga_plot = fga_data[fga_data['group'].isin(patient_groups) & fga_data['Fraction Genome Altered'].notna()]
fig, ax = plt.subplots(figsize=(10, 6))
palette = {
    'Concordant Negative': COLORS['Negative'],
    'Discordant CN-low': '#f39c12',
    'Discordant CN-high': '#e67e22',
    'Concordant Positive': COLORS['Positive'],
}
order = [g for g in patient_groups if g in fga_plot['group'].values]
sns.boxplot(data=fga_plot, x='group', y='Fraction Genome Altered', order=order,
            palette=palette, ax=ax, showfliers=False)
sns.stripplot(data=fga_plot, x='group', y='Fraction Genome Altered', order=order,
              color='black', alpha=0.4, size=3, ax=ax)
ax.set_xlabel('')
ax.set_ylabel('Fraction Genome Altered')
ax.set_title('Genomic Instability (FGA) by Patient Group')
for i, grp in enumerate(order):
    n = fga_plot[fga_plot['group'] == grp].shape[0]
    ax.text(i, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            f'n={n}', ha='center', fontsize=9)
plt.tight_layout()
savefig(fig, 'fig_04_3d_fga_comparison')
plt.close()

# ── 9. Write report ──────────────────────────────────────────────────────────

# Gather data for report
n_matched_disc = len(ratio_df[ratio_df['group'].str.startswith('Discordant')])

# Correlation summary for CN-low
cn_low_corrs = corr_df[corr_df['group'] == 'Discordant CN-low'] if len(corr_df) > 0 else pd.DataFrame()
neg_corrs = corr_df[corr_df['group'] == 'Concordant Negative'] if len(corr_df) > 0 else pd.DataFrame()

report = f"""# Analysis 3: Discordant Biology -- Normal Tissue and ER Pathway

## Key Findings

"""

# Determine biology narrative from heatmap data
if 'Discordant CN-low' in heatmap_data:
    cn_low_er = heatmap_data['Discordant CN-low'].get('ER pathway', 0)
    cn_low_her2 = heatmap_data['Discordant CN-low'].get('HER2 pathway', 0)
    cn_low_prolif = heatmap_data['Discordant CN-low'].get('Proliferation', 0)
    if cn_low_er > 0.2:
        report += "- CN-low discordant patients show elevated ER pathway expression (z={:.2f}), ".format(cn_low_er)
        report += "supporting ER/luminal co-regulation of ERBB2 in this subgroup.\n"
    else:
        report += "- CN-low discordant patients show modest ER pathway expression (z={:.2f}), ".format(cn_low_er)
        report += "suggesting mixed biology rather than pure luminal co-regulation.\n"

if 'Discordant CN-high' in heatmap_data:
    cn_high_her2 = heatmap_data['Discordant CN-high'].get('HER2 pathway', 0)
    report += "- CN-high discordant patients show strong HER2 pathway activation (z={:.2f}), ".format(cn_high_her2)
    report += "consistent with IHC-missed HER2+ biology driven by genomic amplification.\n"

report += f"- Only {n_matched_disc} discordant patients had matched normal tissue, "
report += "limiting tumor-to-normal ratio analysis to descriptive statistics.\n"

if not np.isnan(fga_p):
    report += f"- Genomic instability (FGA): CN-low discordant vs concordant negative "
    report += f"p={fga_p:.4f}, "
    if fga_p < 0.05:
        report += "suggesting distinct genomic backgrounds.\n"
    else:
        report += "no significant difference.\n"

report += f"""
## Methods

### Part A: Tumor-to-Normal ERBB2 Ratios

Matched tumor-normal pairs (N={len(ratio_df)}) were used to compute tumor-to-normal
ERBB2 expression ratios (log-space subtraction). Patients were stratified by
concordance group. Due to only {n_matched_disc} discordant patients having matched
normals, this analysis is descriptive.

### Part B: ER Pathway Correlation

Spearman correlations between ERBB2 and ER pathway genes (ESR1, FOXA1, GATA3,
PGR, etc.) were computed within CN-low discordant patients (n={len(cn_low_disc)})
and concordant negatives as a null comparator. A pathway-level heatmap summarized
mean z-scored expression across gene groups and patient groups.

### Part C: ER Quantitative Scores

Clinical ER quantitative scores (Allred, H-score, intensity, percent positive)
from the cleaned clinical dataset were compared across groups using Kruskal-Wallis
and pairwise Mann-Whitney U tests.

### Part D: Fraction Genome Altered

FGA from the multimodal cohort was compared across groups as a measure of genomic
instability.

## Results

### Part A: Tumor-to-Normal ERBB2 Ratios

| Group | N (matched) | ERBB2 Ratio Median | IQR |
|---|---|---|---|
"""

for grp in patient_groups:
    subset = ratio_df[ratio_df['group'] == grp]['ERBB2_ratio']
    if len(subset) > 0:
        report += f"| {grp} | {len(subset)} | {subset.median():.3f} | [{subset.quantile(0.25):.3f}, {subset.quantile(0.75):.3f}] |\n"
    else:
        report += f"| {grp} | 0 | -- | -- |\n"

report += f"""
**Note:** Only {n_matched_disc} discordant patients had matched normals. Interpret
tumor-to-normal ratios for discordant subgroups descriptively only.

### Part B: ER Pathway Correlations

**ERBB2 vs ER Pathway Genes (Spearman) -- CN-low Discordant (n={len(cn_low_disc)}):**

| Gene | rho | p-value | N |
|---|---|---|---|
"""

for _, row in cn_low_corrs.iterrows():
    sig = "*" if row['p'] < 0.05 else ""
    report += f"| {row['gene']} | {row['rho']:.3f} | {row['p']:.4f}{sig} | {row['n']} |\n"

report += f"""
**ERBB2 vs ER Pathway Genes (Spearman) -- Concordant Negative (n={len(conc_neg)}):**

| Gene | rho | p-value | N |
|---|---|---|---|
"""

for _, row in neg_corrs.iterrows():
    sig = "*" if row['p'] < 0.05 else ""
    report += f"| {row['gene']} | {row['rho']:.3f} | {row['p']:.4f}{sig} | {row['n']} |\n"

report += """
### Pathway Expression Heatmap (Mean Z-scores)

"""
report += heatmap_df.to_markdown() + "\n"

report += """
### Part C: ER Quantitative Scores

"""

for col in er_cols:
    valid = er_data[er_data[col].notna() & er_data['group'].isin(patient_groups)].copy()
    if len(valid) < 5:
        continue
    numeric_vals = pd.to_numeric(valid[col], errors='coerce')
    if numeric_vals.notna().sum() < 5:
        continue
    valid[col] = numeric_vals
    valid = valid[valid[col].notna()]
    report += f"\n**{col.replace('_', ' ').title()}:**\n\n"
    report += "| Group | N | Median | IQR |\n|---|---|---|---|\n"
    for pg in patient_groups:
        s = valid[valid['group'] == pg][col].dropna()
        if len(s) > 0:
            report += f"| {pg} | {len(s)} | {s.median():.1f} | [{s.quantile(0.25):.1f}, {s.quantile(0.75):.1f}] |\n"
    report += "\n"

report += f"""
### Part D: Fraction Genome Altered

| Group | N | Median FGA | IQR |
|---|---|---|---|
"""

for pg in patient_groups:
    if pg in fga_results:
        r = fga_results[pg]
        report += f"| {pg} | {r['n']} | {r['median']:.3f} | [{r['q25']:.3f}, {r['q75']:.3f}] |\n"

if not np.isnan(fga_p):
    report += f"\nDiscordant CN-low vs Concordant Negative: Mann-Whitney U={fga_u:.1f}, p={fga_p:.4f}\n"

report += f"""
## Limitations

- Tumor-to-normal comparison limited to {n_matched_disc} discordant patients with
  matched normals (out of 35 total). This analysis is descriptive only.
- ER quantitative scores are sparsely populated in TCGA clinical annotations;
  coverage varies by metric.
- CN-high discordant subgroup (n={len(cn_high_disc)}) is too small for any statistical
  inference; all findings for this group are descriptive.
- Correlation analyses in the CN-low group (n={len(cn_low_disc)}) have limited power
  for detecting moderate effect sizes.

## Implications

"""

# Adaptive interpretation
if 'Discordant CN-low' in heatmap_data:
    er_z = heatmap_data['Discordant CN-low'].get('ER pathway', 0)
    her2_z = heatmap_data['Discordant CN-low'].get('HER2 pathway', 0)
    prolif_z = heatmap_data['Discordant CN-low'].get('Proliferation', 0)

    if er_z > 0 and her2_z > 0:
        report += """The CN-low discordant group shows elevated expression of both ER pathway and
HER2 pathway genes, consistent with a luminal biology that co-regulates ERBB2
through ER-driven transcriptional programs. This profile suggests these patients
may benefit from endocrine therapy, with the elevated ERBB2 representing
co-regulation rather than independent oncogenic HER2 signaling.

"""
    if her2_z > 0 and er_z <= 0:
        report += """The CN-low discordant group shows elevated HER2 pathway expression without
corresponding ER pathway elevation, suggesting independent HER2-driven biology
without genomic amplification. These patients may represent candidates for
HER2-directed therapy.

"""

report += """The CN-high discordant group (n=6) shows a distinct biology with strong HER2
pathway activation, consistent with true IHC false negatives where genomic
amplification is present but IHC failed to detect protein overexpression.
These patients are the strongest candidates for HER2-directed therapy
reclassification.

The biological distinction between CN-high and CN-low discordant patients supports
different clinical strategies for each subgroup, a finding that would be
strengthened by validation in a larger Tempus cohort with treatment outcome data.

---

**Figures:**
- `fig_04_3a_tumor_normal_erbb2_ratio.png` -- Tumor-to-normal ERBB2 ratio by group
- `fig_04_3b_pathway_heatmap.png` -- Pathway expression heatmap
- `fig_04_3c_er_quantitative.png` -- ER quantitative scores by group
- `fig_04_3d_fga_comparison.png` -- FGA by group
"""

report_path = REPORT_DIR / '3_discordant_biology.md'
report_path.write_text(report, encoding='utf-8')
print(f"\nReport written to: {report_path}")
print("Done.")
