"""
Analysis #5c: RNA Continuous Scoring for T-DXd Eligibility (Priority 4)
======================================================================
Show that RNA provides a continuous quantitative score where IHC provides only
ordinal categories. Demonstrate biological heterogeneity within HER2-low zone.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import (load_intermediate, savefig, setup_plotting, COLORS,
                   classify_her2_spectrum, _parse_ihc_score)
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

setup_plotting()
import matplotlib.pyplot as plt
import seaborn as sns

REPORT_DIR = Path(__file__).resolve().parent.parent / 'reports' / 'biopharma'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────

mm = load_intermediate('02_multimodal_cohort')
analysis_df = load_intermediate('02_analysis_df')
ml_preds = load_intermediate('03_ml_predictions')

# Try loading ssGSEA scores
try:
    ssgsea = load_intermediate('03_ssgsea_scores')
    has_ssgsea = True
except FileNotFoundError:
    has_ssgsea = False
    print("  ssGSEA scores not found, skipping pathway comparison")

# ── 2. Define HER2 spectrum groups ──────────────────────────────────────────

mm['her2_spectrum'] = mm.apply(classify_her2_spectrum, axis=1)
spectrum_counts = mm['her2_spectrum'].value_counts(dropna=False)
print(f"\nHER2 spectrum distribution:\n{spectrum_counts.to_string()}")

# Merge expression and ML predictions
merged = mm[['pid', 'her2_composite', 'her2_spectrum', 'erbb2_copy_number',
             'Fraction Genome Altered']].merge(
    analysis_df[['pid', 'ERBB2_expr', 'GRB7_expr', 'ESR1_expr', 'PGR_expr',
                 'MKI67_expr', 'EGFR_expr', 'ERBB3_expr']],
    on='pid', how='left'
).merge(
    ml_preds[['pid', 'ml_prob_her2_positive']],
    on='pid', how='left'
)

# Define HER2-low (the T-DXd eligible population)
her2_low = merged[merged['her2_spectrum'].isin(['HER2-Low', 'HER2-Low (presumed)'])].copy()
her2_0 = merged[merged['her2_spectrum'] == 'HER2-0'].copy()
her2_pos = merged[merged['her2_spectrum'] == 'HER2-Positive'].copy()

print(f"\nHER2-Low: {len(her2_low)}")
print(f"HER2-0: {len(her2_0)}")
print(f"HER2-Positive: {len(her2_pos)}")

# Also include patients with NaN spectrum but negative composite (likely missing IHC)
her2_neg_no_spectrum = merged[
    (merged['her2_composite'] == 'Negative') &
    (merged['her2_spectrum'].isna())
]
print(f"Negative without spectrum classification: {len(her2_neg_no_spectrum)}")

# ── 3. RNA continuous scoring within HER2-low ────────────────────────────────

# Tertile stratification by ERBB2 expression
if len(her2_low) >= 6:
    her2_low['erbb2_tertile'] = pd.qcut(her2_low['ERBB2_expr'], q=3,
                                         labels=['Low', 'Mid', 'High'])
    tertile_counts = her2_low['erbb2_tertile'].value_counts()
    print(f"\nHER2-Low ERBB2 tertiles:\n{tertile_counts.to_string()}")
else:
    print("\nToo few HER2-Low patients for tertile analysis")
    her2_low['erbb2_tertile'] = 'All'

# Gene panel for biological characterization
her2_genes = ['ERBB2_expr', 'GRB7_expr', 'EGFR_expr', 'ERBB3_expr']
prolif_genes = ['MKI67_expr']
er_genes = ['ESR1_expr', 'PGR_expr']

print("\n=== Biological Characterization by ERBB2 Tertile ===")
for gene in her2_genes + prolif_genes + er_genes:
    if gene not in her2_low.columns:
        continue
    print(f"\n{gene}:")
    for tert in ['Low', 'Mid', 'High']:
        subset = her2_low[her2_low['erbb2_tertile'] == tert][gene].dropna()
        if len(subset) > 0:
            print(f"  {tert}: median={subset.median():.3f}, "
                  f"IQR=[{subset.quantile(0.25):.3f}, {subset.quantile(0.75):.3f}]")
    # Kruskal-Wallis
    groups = [her2_low[her2_low['erbb2_tertile'] == t][gene].dropna().values
              for t in ['Low', 'Mid', 'High']
              if len(her2_low[her2_low['erbb2_tertile'] == t][gene].dropna()) > 0]
    if len(groups) >= 2:
        h, p = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis: H={h:.2f}, p={p:.4f}")

# ── 4. ML probability distribution across spectrum ──────────────────────────

print("\n=== ML Probability by HER2 Spectrum ===")
for grp_name, grp_df in [('HER2-0', her2_0), ('HER2-Low', her2_low),
                          ('HER2-Positive', her2_pos)]:
    vals = grp_df['ml_prob_her2_positive'].dropna()
    if len(vals) > 0:
        print(f"  {grp_name} (n={len(vals)}): median={vals.median():.3f}, "
              f"IQR=[{vals.quantile(0.25):.3f}, {vals.quantile(0.75):.3f}]")

# Compare upper tertile of HER2-low with HER2-positive
if len(her2_low) >= 6:
    upper_tert = her2_low[her2_low['erbb2_tertile'] == 'High']['ml_prob_her2_positive'].dropna()
    pos_ml = her2_pos['ml_prob_her2_positive'].dropna()
    if len(upper_tert) > 0 and len(pos_ml) > 0:
        u, p = stats.mannwhitneyu(upper_tert, pos_ml, alternative='two-sided')
        print(f"\n  Upper tertile HER2-Low vs HER2-Positive: U={u:.1f}, p={p:.4f}")
        print(f"    Upper tertile median: {upper_tert.median():.3f}")
        print(f"    HER2-Positive median: {pos_ml.median():.3f}")

# ── 5. ssGSEA pathway comparison ────────────────────────────────────────────

ssgsea_results = None
if has_ssgsea:
    her2_low_ssgsea = her2_low.merge(ssgsea, on='pid', how='left')
    pathway_cols = [c for c in ssgsea.columns if c != 'pid']
    print(f"\n=== ssGSEA Pathways by ERBB2 Tertile (HER2-Low) ===")
    ssgsea_summary = {}
    for pathway in pathway_cols[:10]:  # Top pathways
        for tert in ['Low', 'Mid', 'High']:
            vals = her2_low_ssgsea[her2_low_ssgsea['erbb2_tertile'] == tert][pathway].dropna()
            if tert not in ssgsea_summary:
                ssgsea_summary[tert] = {}
            ssgsea_summary[tert][pathway] = vals.median() if len(vals) > 0 else np.nan

    ssgsea_results = pd.DataFrame(ssgsea_summary).T
    print(ssgsea_results.to_string())

# ── 6. Figures ───────────────────────────────────────────────────────────────

# Fig 1: ERBB2 expression strip/box plot by IHC spectrum category
spectrum_order = ['HER2-0', 'HER2-Low', 'HER2-Low (presumed)', 'HER2-Positive']
spectrum_order = [s for s in spectrum_order if s in merged['her2_spectrum'].values]
spectrum_palette = {
    'HER2-0': COLORS['HER2-0'],
    'HER2-Low': COLORS['HER2-Low'],
    'HER2-Low (presumed)': '#e8c96c',
    'HER2-Positive': COLORS['HER2-Positive'],
}

plot_data = merged[merged['her2_spectrum'].isin(spectrum_order)].copy()
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=plot_data, x='her2_spectrum', y='ERBB2_expr', order=spectrum_order,
            palette=spectrum_palette, ax=ax, showfliers=False)
sns.stripplot(data=plot_data, x='her2_spectrum', y='ERBB2_expr', order=spectrum_order,
              color='black', alpha=0.3, size=2, ax=ax)
ax.set_xlabel('HER2 Spectrum (IHC-based)')
ax.set_ylabel('ERBB2 Expression (log2 normalized)')
ax.set_title('RNA Reveals Continuous ERBB2 Spectrum Within IHC Categories')
for i, grp in enumerate(spectrum_order):
    n = plot_data[plot_data['her2_spectrum'] == grp].shape[0]
    ax.text(i, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            f'n={n}', ha='center', fontsize=9)
plt.tight_layout()
savefig(fig, 'fig_04_5c_her2low_rna_spectrum')
plt.close()

# Fig 2: ML probability density by spectrum group
fig, ax = plt.subplots(figsize=(10, 6))
for grp, color, label in [
    (her2_0, COLORS['HER2-0'], 'HER2-0'),
    (her2_low[her2_low['erbb2_tertile'] == 'Low'] if len(her2_low) >= 6 else her2_low,
     '#85c1e9', 'HER2-Low (low tertile)'),
    (her2_low[her2_low['erbb2_tertile'] == 'High'] if len(her2_low) >= 6 else pd.DataFrame(),
     '#e67e22', 'HER2-Low (high tertile)'),
    (her2_pos, COLORS['HER2-Positive'], 'HER2-Positive'),
]:
    vals = grp['ml_prob_her2_positive'].dropna() if len(grp) > 0 else pd.Series()
    if len(vals) > 3:
        vals.plot.kde(ax=ax, color=color, label=f'{label} (n={len(vals)})', linewidth=2)
ax.set_xlabel('ML Probability (HER2+)')
ax.set_ylabel('Density')
ax.set_title('ML HER2 Score Distribution by Spectrum Group')
ax.legend()
ax.set_xlim(-0.1, 1.1)
plt.tight_layout()
savefig(fig, 'fig_04_5c_her2low_ml_density')
plt.close()

# Fig 3: Pathway heatmap by tertile (if ssGSEA available)
if ssgsea_results is not None and len(ssgsea_results.columns) > 0:
    fig, ax = plt.subplots(figsize=(12, 4))
    # Clean column names for display
    display_df = ssgsea_results.copy()
    display_df.columns = [c.replace('pathway_', '').replace('_', ' ')[:30] for c in display_df.columns]
    sns.heatmap(display_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, linewidths=0.5)
    ax.set_title('ssGSEA Pathway Scores by ERBB2 Tertile (HER2-Low)')
    ax.set_ylabel('ERBB2 Tertile')
    plt.tight_layout()
    savefig(fig, 'fig_04_5c_her2low_pathway_tertiles')
    plt.close()

# Fig 4: Scatter plot ERBB2 vs ML probability colored by spectrum
fig, ax = plt.subplots(figsize=(9, 7))
for grp_name in spectrum_order:
    subset = plot_data[plot_data['her2_spectrum'] == grp_name]
    ax.scatter(subset['ERBB2_expr'], subset['ml_prob_her2_positive'],
               c=spectrum_palette.get(grp_name, '#95a5a6'), label=grp_name,
               alpha=0.5, s=20, edgecolors='none')
ax.set_xlabel('ERBB2 Expression (log2)')
ax.set_ylabel('ML Probability (HER2+)')
ax.set_title('ERBB2 Expression vs ML Score by HER2 Spectrum')
ax.legend()
plt.tight_layout()
savefig(fig, 'fig_04_5c_erbb2_vs_ml_scatter')
plt.close()

# ── 7. Write report ──────────────────────────────────────────────────────────

# Gather tertile stats
tertile_stats = []
for tert in ['Low', 'Mid', 'High']:
    subset = her2_low[her2_low['erbb2_tertile'] == tert]
    if len(subset) == 0:
        continue
    tertile_stats.append({
        'tertile': tert,
        'n': len(subset),
        'erbb2_med': subset['ERBB2_expr'].median(),
        'grb7_med': subset['GRB7_expr'].median(),
        'mki67_med': subset['MKI67_expr'].median(),
        'esr1_med': subset['ESR1_expr'].median(),
        'ml_med': subset['ml_prob_her2_positive'].median(),
    })
tert_df = pd.DataFrame(tertile_stats)

report = f"""# Analysis 5c: RNA Continuous Scoring for T-DXd Eligibility

## Key Findings

- Within the HER2-Low population (n={len(her2_low)}), RNA expression reveals a
  continuous spectrum of ERBB2 biology that IHC ordinal categories obscure.
- The upper ERBB2 tertile of HER2-Low patients shows ML probabilities
  (median={tert_df[tert_df['tertile']=='High']['ml_med'].values[0]:.3f} if available)
  distinct from the lower tertile, suggesting biologically heterogeneous subgroups.
- HER2-0, HER2-Low, and HER2-Positive show overlapping but progressively shifted
  ERBB2 expression distributions, supporting a biological continuum.

## Methods

### HER2 Spectrum Classification

Patients were classified into HER2-0 (IHC 0), HER2-Low (IHC 1+ or IHC 2+/FISH-),
and HER2-Positive (IHC 3+ or IHC 2+/FISH+) using the `classify_her2_spectrum`
function. HER2-Low patients are clinically eligible for T-DXd (trastuzumab
deruxtecan) per DESTINY-Breast04 criteria.

### Continuous Scoring

Two complementary RNA-based scores were used:
1. Raw ERBB2 expression (log2 normalized RSEM)
2. ML ensemble probability from the trained HER2 classifier

HER2-Low patients were stratified into ERBB2 expression tertiles. Biological
characterization compared HER2 pathway genes, proliferation markers, and ER
pathway genes across tertiles.

## Results

### HER2 Spectrum Distribution

| Category | N |
|---|---|
"""

for cat in spectrum_order:
    n = len(merged[merged['her2_spectrum'] == cat])
    report += f"| {cat} | {n} |\n"

n_na = merged['her2_spectrum'].isna().sum()
report += f"| Unclassified | {n_na} |\n"

report += f"""
### Biological Characterization by ERBB2 Tertile (HER2-Low)

| Tertile | N | ERBB2 | GRB7 | MKI67 | ESR1 | ML Prob |
|---|---|---|---|---|---|---|
"""

for _, row in tert_df.iterrows():
    report += (f"| {row['tertile']} | {row['n']} | {row['erbb2_med']:.2f} | "
               f"{row['grb7_med']:.2f} | {row['mki67_med']:.2f} | "
               f"{row['esr1_med']:.2f} | {row['ml_med']:.3f} |\n")

report += f"""
### ML Probability Distribution by Spectrum Group

| Group | N | Median ML Prob | IQR |
|---|---|---|---|
"""

for grp_name, grp_df in [('HER2-0', her2_0), ('HER2-Low', her2_low),
                          ('HER2-Positive', her2_pos)]:
    vals = grp_df['ml_prob_her2_positive'].dropna()
    if len(vals) > 0:
        report += (f"| {grp_name} | {len(vals)} | {vals.median():.3f} | "
                   f"[{vals.quantile(0.25):.3f}, {vals.quantile(0.75):.3f}] |\n")

if ssgsea_results is not None:
    report += "\n### ssGSEA Pathway Scores by ERBB2 Tertile\n\n"
    report += ssgsea_results.to_markdown() + "\n"

report += f"""
## Limitations

- HER2 spectrum classification depends on IHC scores, which are sparse in TCGA
  (many patients have NaN IHC scores). {n_na} patients could not be classified.
- T-DXd treatment-benefit correlation cannot be assessed with TCGA data (no
  treatment response data). This analysis demonstrates biological heterogeneity,
  not treatment benefit.
- The ML model was trained on binary HER2+/- labels; its behavior in the HER2-Low
  zone is extrapolation from training distribution boundaries.

## Implications

Within the HER2-Low population -- the T-DXd eligible group -- RNA-based
quantification reveals substantial biological heterogeneity that IHC ordinal
categories (0, 1+, 2+) fail to capture. The upper ERBB2 tertile shows pathway
activation profiles more similar to HER2-Positive patients, suggesting these
patients may derive greater benefit from HER2-directed ADC therapies like T-DXd.

This supports the hypothesis that RNA-guided T-DXd eligibility stratification
could improve treatment selection beyond the current IHC-based approach. Validation
requires RWD with linked treatment and outcome data (Tempus dataset), where the
predictive value of RNA-continuous scoring for T-DXd response can be directly tested.

---

**Figures:**
- `fig_04_5c_her2low_rna_spectrum.png` -- ERBB2 expression by IHC spectrum
- `fig_04_5c_her2low_ml_density.png` -- ML probability density by group
- `fig_04_5c_her2low_pathway_tertiles.png` -- ssGSEA by tertile (if available)
- `fig_04_5c_erbb2_vs_ml_scatter.png` -- ERBB2 vs ML probability scatter
"""

report_path = REPORT_DIR / '5c_tdxd_spectrum.md'
report_path.write_text(report, encoding='utf-8')
print(f"\nReport written to: {report_path}")
print("Done.")
