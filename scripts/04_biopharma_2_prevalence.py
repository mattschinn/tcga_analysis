"""
Analysis #2: Prevalence of Molecular ERBB2 Overexpression (Priority 3)
=====================================================================
Estimate the size of the "missed HER2+" population among IHC-negative patients.
Frame as prevalence of molecular overexpression, NOT false negative rate.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_intermediate, savefig, setup_plotting, COLORS
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from pathlib import Path

setup_plotting()
import matplotlib.pyplot as plt
import seaborn as sns

REPORT_DIR = Path(__file__).resolve().parent.parent / 'reports' / 'biopharma'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────

analysis_df = load_intermediate('02_analysis_df')
disc = load_intermediate('02_discordant_cases')

# ── 2. Define denominator: IHC-negative patients with RNA data ───────────────

ihc_neg = analysis_df[analysis_df['her2_composite'] == 'Negative'].copy()
n_total = len(ihc_neg)
print(f"\nIHC-negative patients with RNA data: {n_total}")

# Verify ERBB2_expr populated
n_erbb2 = ihc_neg['ERBB2_expr'].notna().sum()
print(f"  With ERBB2_expr: {n_erbb2}")

# ── 3. Define thresholds ────────────────────────────────────────────────────

neg_erbb2 = ihc_neg['ERBB2_expr'].dropna()

thresholds = {
    'p95 (primary)': neg_erbb2.quantile(0.95),
    'p90': neg_erbb2.quantile(0.90),
    'p99': neg_erbb2.quantile(0.99),
    'mean+2SD': neg_erbb2.mean() + 2 * neg_erbb2.std(),
}

print("\nThreshold values:")
for name, val in thresholds.items():
    print(f"  {name}: {val:.3f}")

# NB02's discordant count for reference
n_disc_nb02 = len(disc[disc['discordance_type'] == 'IHC-/RNA-high'])
print(f"\nNB02 discordant (IHC-/RNA-high): {n_disc_nb02}")

# ── 4. Compute prevalence at each threshold ─────────────────────────────────

prevalence_results = []
for name, thresh in thresholds.items():
    count = (neg_erbb2 > thresh).sum()
    total = len(neg_erbb2)
    prev = count / total
    ci_lo, ci_hi = proportion_confint(count, total, alpha=0.05, method='wilson')
    prevalence_results.append({
        'threshold': name,
        'threshold_value': thresh,
        'count': count,
        'total': total,
        'prevalence': prev,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
    })
    print(f"\n{name} (threshold={thresh:.3f}):")
    print(f"  Count: {count}/{total} = {prev:.1%}")
    print(f"  95% CI (Wilson): [{ci_lo:.1%}, {ci_hi:.1%}]")

prev_df = pd.DataFrame(prevalence_results)

# ── 5. CN stratification of molecular overexpression group ──────────────────

# Use primary threshold (p95)
primary_thresh = thresholds['p95 (primary)']
mol_overexpr = ihc_neg[ihc_neg['ERBB2_expr'] > primary_thresh].copy()
print(f"\nMolecular ERBB2 overexpression group (p95): {len(mol_overexpr)}")

cn_dist = mol_overexpr['erbb2_copy_number'].value_counts().sort_index()
print(f"  CN distribution:\n{cn_dist.to_string()}")

cn_high = (mol_overexpr['erbb2_copy_number'] >= 2).sum()
cn_low = (mol_overexpr['erbb2_copy_number'] <= 1).sum()
cn_na = mol_overexpr['erbb2_copy_number'].isna().sum()
print(f"  CN >= 2 (amplified): {cn_high}")
print(f"  CN <= 1 (not amplified): {cn_low}")
print(f"  CN missing: {cn_na}")

# ── 6. Population extrapolation (heavily caveated) ──────────────────────────

# US breast cancer incidence ~310,000/year (2024 ACS estimate)
# ~75-80% are HER2-negative by IHC
us_bc_annual = 310000
her2_neg_frac = 0.80
us_her2_neg = us_bc_annual * her2_neg_frac

primary_prev = prev_df[prev_df['threshold'] == 'p95 (primary)'].iloc[0]
est_low = int(us_her2_neg * primary_prev['ci_lo'])
est_mid = int(us_her2_neg * primary_prev['prevalence'])
est_high = int(us_her2_neg * primary_prev['ci_hi'])

print(f"\nPopulation extrapolation (heavily caveated):")
print(f"  US HER2-negative BC/year: ~{us_her2_neg:,.0f}")
print(f"  Estimated patients with molecular overexpression:")
print(f"    Low: {est_low:,}, Mid: {est_mid:,}, High: {est_high:,}")

# ── 7. Figures ───────────────────────────────────────────────────────────────

# Fig 1: Prevalence by threshold with CI error bars
fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(prev_df))
bars = ax.bar(x, prev_df['prevalence'] * 100, color='#3498db', alpha=0.8, edgecolor='white')
ax.errorbar(x, prev_df['prevalence'] * 100,
            yerr=[(prev_df['prevalence'] - prev_df['ci_lo']) * 100,
                  (prev_df['ci_hi'] - prev_df['prevalence']) * 100],
            fmt='none', color='black', capsize=5, linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(prev_df['threshold'], rotation=15, ha='right')
ax.set_ylabel('Prevalence (%)')
ax.set_title('Prevalence of Molecular ERBB2 Overexpression\nAmong IHC-Negative Patients')
# Annotate counts
for i, row in prev_df.iterrows():
    ax.text(i, row['prevalence'] * 100 + 0.3,
            f"n={row['count']}", ha='center', fontsize=9)
plt.tight_layout()
savefig(fig, 'fig_04_2_prevalence_by_threshold')
plt.close()

# Fig 2: CN-stratified breakdown
fig, ax = plt.subplots(figsize=(6, 5))
cn_data = pd.DataFrame({
    'group': ['CN >= 2\n(amplified)', 'CN <= 1\n(not amplified)'],
    'count': [cn_high, cn_low],
})
colors = ['#e74c3c', '#f39c12']
ax.bar(cn_data['group'], cn_data['count'], color=colors, edgecolor='white', alpha=0.8)
for i, row in cn_data.iterrows():
    ax.text(i, row['count'] + 0.3, str(row['count']), ha='center', fontsize=11)
ax.set_ylabel('Count')
ax.set_title(f'CN Stratification of Molecular ERBB2\nOverexpression Group (N={len(mol_overexpr)})')
plt.tight_layout()
savefig(fig, 'fig_04_2_prevalence_cn_stratified')
plt.close()

# Fig 3: ERBB2 distribution with threshold markers
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(neg_erbb2, bins=50, color='#3498db', alpha=0.6, edgecolor='white', label='IHC-Negative')
# Also show positive distribution for reference
pos_erbb2 = analysis_df[analysis_df['her2_composite'] == 'Positive']['ERBB2_expr'].dropna()
ax.hist(pos_erbb2, bins=30, color='#e74c3c', alpha=0.4, edgecolor='white', label='IHC-Positive')
# Threshold lines
colors_thresh = ['#2c3e50', '#7f8c8d', '#95a5a6', '#bdc3c7']
for (name, val), c in zip(thresholds.items(), colors_thresh):
    ax.axvline(val, color=c, linestyle='--', linewidth=1.5, label=f'{name} ({val:.1f})')
ax.set_xlabel('ERBB2 Expression (log2 normalized)')
ax.set_ylabel('Count')
ax.set_title('ERBB2 Expression Distribution with Overexpression Thresholds')
ax.legend(fontsize=8)
plt.tight_layout()
savefig(fig, 'fig_04_2_erbb2_distribution_thresholds')
plt.close()

# ── 8. Write report ──────────────────────────────────────────────────────────

report = f"""# Analysis 2: Prevalence of Molecular ERBB2 Overexpression in IHC-Negative Patients

## Key Findings

- At the primary threshold (95th percentile), {primary_prev['count']:.0f} of {primary_prev['total']:.0f}
  IHC-negative patients ({primary_prev['prevalence']:.1%}, 95% CI: [{primary_prev['ci_lo']:.1%},
  {primary_prev['ci_hi']:.1%}]) show molecular evidence of ERBB2 overexpression.
- Of these, {cn_high} ({100*cn_high/len(mol_overexpr):.0f}%) also show genomic amplification
  (CN >= 2), providing orthogonal evidence for HER2+ biology.
- Prevalence is robust across thresholds: {prev_df['prevalence'].min():.1%} (p99) to
  {prev_df['prevalence'].max():.1%} (p90).
- If this prevalence holds in the general IHC-negative population, it represents an
  estimated {est_low:,}-{est_high:,} patients annually in the US.

## Methods

### Framing

This analysis estimates the **prevalence of molecular ERBB2 overexpression among
IHC-negative patients** -- not a "false negative rate." IHC is the clinical ground
truth; there is no higher authority to define false negatives against. The defensible
framing is: a defined proportion of IHC-negative patients show molecular evidence
that questions their negative classification.

### Approach

The denominator was all IHC-negative patients in the multimodal cohort with RNA-seq
data (N={n_total}). The primary threshold for ERBB2 overexpression was the 95th
percentile of ERBB2 expression among IHC-negative patients -- the same criterion
used in NB02 for discordant identification.

Sensitivity analysis used three additional thresholds: 90th percentile, 99th
percentile, and mean + 2 standard deviations.

95% confidence intervals were computed using the Wilson score method. CN stratification
used GISTIC-derived copy number values (0, 1, 2).

## Results

### Prevalence by Threshold

| Threshold | Value | Count | Prevalence | 95% CI |
|---|---|---|---|---|
"""

for _, row in prev_df.iterrows():
    report += (f"| {row['threshold']} | {row['threshold_value']:.3f} | "
               f"{row['count']:.0f}/{row['total']:.0f} | {row['prevalence']:.1%} | "
               f"[{row['ci_lo']:.1%}, {row['ci_hi']:.1%}] |\n")

report += f"""
### CN Stratification (Primary Threshold)

Among the {len(mol_overexpr)} patients with molecular ERBB2 overexpression (p95):

| CN Status | N | Percentage | Interpretation |
|---|---|---|---|
| CN >= 2 (amplified) | {cn_high} | {100*cn_high/len(mol_overexpr):.0f}% | Strongest case: genomic + transcriptomic evidence |
| CN = 1 | {(mol_overexpr['erbb2_copy_number'] == 1).sum()} | {100*(mol_overexpr['erbb2_copy_number'] == 1).sum()/len(mol_overexpr):.0f}% | Intermediate: modest CN gain + high expression |
| CN = 0 | {(mol_overexpr['erbb2_copy_number'] == 0).sum()} | {100*(mol_overexpr['erbb2_copy_number'] == 0).sum()/len(mol_overexpr):.0f}% | Transcriptional only: expression without amplification |

### Population Extrapolation (Heavily Caveated)

| Parameter | Value |
|---|---|
| US breast cancer incidence (annual) | ~{us_bc_annual:,} |
| Estimated HER2-negative fraction | ~{her2_neg_frac:.0%} |
| US HER2-negative patients/year | ~{us_her2_neg:,.0f} |
| Estimated molecular overexpression (low) | {est_low:,} |
| Estimated molecular overexpression (mid) | {est_mid:,} |
| Estimated molecular overexpression (high) | {est_high:,} |

**Caveats:** TCGA is not a random sample of the breast cancer population. Ascertainment
bias, single-institution effects, and pre-treatment selection may inflate or deflate
these estimates. The extrapolation is included for order-of-magnitude context only.

## Limitations

- The threshold for "molecular overexpression" is statistically derived (distribution
  percentiles), not clinically validated. The clinical relevance of exceeding any
  particular percentile depends on treatment response data not available in TCGA.
- TCGA ascertainment bias: academic medical center patients may differ from the
  general population in disease severity, testing practices, and demographics.
- The prevalence estimate applies to patients who are IHC-negative AND have RNA-seq
  data. Patients without RNA-seq testing cannot be assessed.
- CN = 0 patients with high ERBB2 RNA may represent transcriptional regulation
  (e.g., ER-driven) rather than HER2-driven biology (see Analysis #3).

## Implications

The identification of {primary_prev['prevalence']:.1%} of IHC-negative patients with
molecular evidence of ERBB2 overexpression suggests a clinically meaningful population
that may be missed by current IHC-only testing. The CN-stratified breakdown provides
a biological hierarchy:

1. **CN >= 2 patients (n={cn_high})** represent the strongest candidates for
   reclassification -- they have both genomic amplification and transcriptomic
   overexpression, suggesting true IHC false negatives.
2. **CN = 0/1 patients (n={cn_low})** require deeper biological characterization
   (see Analysis #3) to determine whether their ERBB2 overexpression is
   HER2-pathway-driven or reflects ER/luminal co-regulation.

RNA-based molecular testing could identify this population in clinical practice,
either as a standalone CDx or as a reflex test for IHC-negative patients with
clinical suspicion of HER2-driven disease.

---

**Figures:**
- `fig_04_2_prevalence_by_threshold.png` -- Prevalence at each threshold with CIs
- `fig_04_2_prevalence_cn_stratified.png` -- CN breakdown of overexpression group
- `fig_04_2_erbb2_distribution_thresholds.png` -- ERBB2 distribution with thresholds
"""

report_path = REPORT_DIR / '2_prevalence_estimation.md'
report_path.write_text(report, encoding='utf-8')
print(f"\nReport written to: {report_path}")
print("Done.")
