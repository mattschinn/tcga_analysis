"""
Analysis #4a: Clinical Correlates (Priority 5)
==============================================
Table 1 characterization of discordant IHC-/RNA-high population vs concordant.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import (load_intermediate, savefig, setup_plotting, COLORS,
                   to_patient_id, _parse_ihc_score)
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path

setup_plotting()
import matplotlib.pyplot as plt
import seaborn as sns

REPORT_DIR = Path(__file__).resolve().parent.parent / 'reports' / 'biopharma'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load and merge data ──────────────────────────────────────────────────

mm = load_intermediate('02_multimodal_cohort')
disc = load_intermediate('02_discordant_cases')
clin = pd.read_csv(
    Path(__file__).resolve().parent.parent / 'data' / 'brca_tcga_clinical_data_cleaned.csv'
)
clin['pid'] = clin['Patient ID'].apply(to_patient_id)

# Define groups
ihc_neg_rna_high = disc[disc['discordance_type'] == 'IHC-/RNA-high']
disc_pids = set(ihc_neg_rna_high['pid'])
cn_high_pids = set(ihc_neg_rna_high[ihc_neg_rna_high['erbb2_copy_number'] >= 2]['pid'])
cn_low_pids = set(ihc_neg_rna_high[ihc_neg_rna_high['erbb2_copy_number'] <= 1]['pid'])

# Merge multimodal with clinical
merged = mm.merge(clin.drop(columns=['pid']), left_on='pid',
                  right_on=clin['pid'], how='left')
# Remove duplicate pid column if created
if 'key_0' in merged.columns:
    merged = merged.drop(columns=['key_0'])

merged['group'] = 'Other'
merged.loc[merged['pid'].isin(cn_high_pids), 'group'] = 'Discordant CN-high'
merged.loc[merged['pid'].isin(cn_low_pids), 'group'] = 'Discordant CN-low'
conc_neg_mask = (merged['her2_composite'] == 'Negative') & (~merged['pid'].isin(disc_pids))
merged.loc[conc_neg_mask, 'group'] = 'Concordant Negative'
merged.loc[merged['her2_composite'] == 'Positive', 'group'] = 'Concordant Positive'

groups_of_interest = ['Concordant Negative', 'Discordant CN-low',
                      'Discordant CN-high', 'Concordant Positive']
df = merged[merged['group'].isin(groups_of_interest)].copy()

print(f"\nGroup sizes:")
print(df['group'].value_counts().to_string())

# ── 2. Define clinical variables ────────────────────────────────────────────

continuous_vars = [
    ('Diagnosis Age', 'Age at Diagnosis'),
    ('Fraction Genome Altered', 'FGA'),
]

categorical_vars = [
    ('ER Status By IHC', 'ER Status'),
    ('Cancer Type Detailed', 'Histologic Type'),
]

# Try to get staging columns
stage_cols = [c for c in df.columns if 'stage' in c.lower() or 'Stage' in c]
print(f"\nStaging columns available: {stage_cols[:5]}")

# ER quantitative from clinical
er_quant_cols = ['er_allred_score', 'er_hscore']
for col in er_quant_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Add ER quantitative as continuous vars
for col in er_quant_cols:
    if col in df.columns and df[col].notna().sum() > 10:
        continuous_vars.append((col, col.replace('_', ' ').title()))

# ── 3. Build Table 1 ────────────────────────────────────────────────────────

table1_rows = []
p_values = []
test_names = []

# Continuous variables
for col, label in continuous_vars:
    if col not in df.columns:
        continue
    row = {'Variable': label}
    valid_groups = []
    for grp in groups_of_interest:
        subset = df[df['group'] == grp][col].dropna()
        if len(subset) > 0:
            row[grp] = f"{subset.median():.1f} ({subset.quantile(0.25):.1f}-{subset.quantile(0.75):.1f})"
            row[f'{grp}_n'] = len(subset)
            valid_groups.append(subset.values)
        else:
            row[grp] = '--'
            row[f'{grp}_n'] = 0

    # Kruskal-Wallis
    if len(valid_groups) >= 2:
        h, p = stats.kruskal(*[g for g in valid_groups if len(g) > 0])
        row['p'] = p
        p_values.append(p)
        test_names.append(label)
    else:
        row['p'] = np.nan
    table1_rows.append(row)

# Categorical variables
for col, label in categorical_vars:
    if col not in df.columns:
        continue
    row = {'Variable': label}
    for grp in groups_of_interest:
        subset = df[df['group'] == grp][col].dropna()
        if len(subset) > 0:
            mode_val = subset.mode().iloc[0] if len(subset.mode()) > 0 else '--'
            mode_count = (subset == mode_val).sum()
            row[grp] = f"{mode_val} ({mode_count}/{len(subset)})"
            row[f'{grp}_n'] = len(subset)
        else:
            row[grp] = '--'
            row[f'{grp}_n'] = 0

    # Chi-squared (discordant combined vs concordant negative)
    disc_all = df[df['group'].isin(['Discordant CN-low', 'Discordant CN-high'])][col].dropna()
    conc_neg_vals = df[df['group'] == 'Concordant Negative'][col].dropna()
    if len(disc_all) > 0 and len(conc_neg_vals) > 0:
        combined = pd.concat([disc_all, conc_neg_vals])
        group_labels = ['Discordant'] * len(disc_all) + ['Concordant Neg'] * len(conc_neg_vals)
        ct = pd.crosstab(group_labels, combined)
        # Use Fisher's for 2x2, chi2 otherwise
        if ct.shape == (2, 2):
            odds, p = stats.fisher_exact(ct.values)
        else:
            # Drop columns with zero totals
            ct = ct.loc[:, ct.sum() > 0]
            if ct.shape[1] >= 2:
                chi2, p, dof, expected = stats.chi2_contingency(ct.values)
            else:
                p = np.nan
        row['p'] = p
        if not np.isnan(p):
            p_values.append(p)
            test_names.append(label)
    else:
        row['p'] = np.nan
    table1_rows.append(row)

# FDR correction
if len(p_values) > 0:
    reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    fdr_map = dict(zip(test_names, pvals_corrected))
    for row in table1_rows:
        row['p_fdr'] = fdr_map.get(row['Variable'], np.nan)
else:
    for row in table1_rows:
        row['p_fdr'] = np.nan

# Print Table 1
print("\n=== Table 1: Clinical Characteristics ===")
for row in table1_rows:
    print(f"\n{row['Variable']}:")
    for grp in groups_of_interest:
        print(f"  {grp}: {row.get(grp, '--')} (n={row.get(f'{grp}_n', 0)})")
    p = row.get('p', np.nan)
    pfdr = row.get('p_fdr', np.nan)
    if not np.isnan(p):
        print(f"  p={p:.4f}, FDR={pfdr:.4f}" if not np.isnan(pfdr) else f"  p={p:.4f}")

# ── 4. Polysomy 17 check ────────────────────────────────────────────────────

print("\n=== Polysomy 17 Check ===")
cent17_col = 'Cent17 Copy Number'
her2_cent17_col = 'HER2 cent17 ratio'

for col in [cent17_col, her2_cent17_col]:
    if col in df.columns:
        disc_vals = df[df['group'].str.startswith('Discordant')][['pid', 'group', col]].dropna(subset=[col])
        print(f"\n{col} -- discordant patients with data: {len(disc_vals)}")
        if len(disc_vals) > 0:
            print(disc_vals.to_string(index=False))

# Also check IHC+/RNA-low for polysomy
ihc_pos_rna_low = disc[disc['discordance_type'] == 'IHC+/RNA-low']
if len(ihc_pos_rna_low) > 0:
    ihc_pos_rna_low_mm = mm[mm['pid'].isin(ihc_pos_rna_low['pid'])]
    print(f"\nIHC+/RNA-low patients: {len(ihc_pos_rna_low)}")
    for col in [cent17_col, her2_cent17_col]:
        if col in ihc_pos_rna_low_mm.columns:
            vals = ihc_pos_rna_low_mm[['pid', col]].dropna(subset=[col])
            print(f"  {col} data: {len(vals)} patients")
            if len(vals) > 0:
                print(vals.to_string(index=False))

# ── 5. ER scoring method variation ──────────────────────────────────────────

print("\n=== ER Scoring Method ===")
er_method_col = 'er_scoring_method_detail'
if er_method_col in df.columns:
    for grp in groups_of_interest:
        subset = df[df['group'] == grp][er_method_col].dropna()
        if len(subset) > 0:
            print(f"\n{grp}:")
            print(subset.value_counts().to_string())
else:
    print(f"Column '{er_method_col}' not found")

# ── 6. Figure: Forest plot of effect sizes ───────────────────────────────────

# Compute standardized mean differences (Cohen's d) for continuous variables
# Discordant (combined) vs Concordant Negative
disc_combined = df[df['group'].isin(['Discordant CN-low', 'Discordant CN-high'])]
conc_neg = df[df['group'] == 'Concordant Negative']

effect_sizes = []
for col, label in continuous_vars:
    if col not in df.columns:
        continue
    d_vals = disc_combined[col].dropna()
    n_vals = conc_neg[col].dropna()
    if len(d_vals) >= 3 and len(n_vals) >= 3:
        pooled_std = np.sqrt(((len(d_vals)-1)*d_vals.std()**2 + (len(n_vals)-1)*n_vals.std()**2) /
                            (len(d_vals) + len(n_vals) - 2))
        if pooled_std > 0:
            d = (d_vals.mean() - n_vals.mean()) / pooled_std
            se = np.sqrt(1/len(d_vals) + 1/len(n_vals) + d**2/(2*(len(d_vals)+len(n_vals))))
            effect_sizes.append({
                'variable': label,
                'd': d,
                'ci_lo': d - 1.96*se,
                'ci_hi': d + 1.96*se,
            })

if len(effect_sizes) > 0:
    es_df = pd.DataFrame(effect_sizes)
    fig, ax = plt.subplots(figsize=(8, max(4, len(es_df)*0.6 + 1)))
    y_pos = range(len(es_df))
    ax.errorbar(es_df['d'], y_pos,
                xerr=[es_df['d'] - es_df['ci_lo'], es_df['ci_hi'] - es_df['d']],
                fmt='o', color='#2c3e50', capsize=5, markersize=8)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(es_df['variable'])
    ax.set_xlabel("Cohen's d (Discordant vs Concordant Negative)")
    ax.set_title('Effect Sizes: Discordant vs Concordant Negative')
    ax.invert_yaxis()
    plt.tight_layout()
    savefig(fig, 'fig_04_4a_table1_forest')
    plt.close()

# ── 7. Write report ──────────────────────────────────────────────────────────

report = """# Analysis 4a: Clinical Correlates of Discordant Patients

## Key Findings

"""

# Summarize significant findings
sig_findings = [r for r in table1_rows if r.get('p_fdr', 1) < 0.05]
if sig_findings:
    for r in sig_findings:
        report += f"- {r['Variable']}: significantly different across groups (FDR p={r['p_fdr']:.4f})\n"
else:
    report += "- No clinical variables reached statistical significance after FDR correction.\n"

report += f"""- Discordant group (n={len(disc_combined)}) compared against concordant negative (n={len(conc_neg)}).
- CN-high discordant (n={len(cn_high_pids)}) is too small for independent statistical inference.

## Methods

Clinical characteristics were compared across four groups: Concordant Negative,
Discordant CN-low, Discordant CN-high, and Concordant Positive. Continuous variables
were summarized as median (IQR) and compared using Kruskal-Wallis. Categorical
variables used chi-squared or Fisher's exact test (discordant combined vs concordant
negative). All p-values were FDR-corrected (Benjamini-Hochberg).

## Results

### Table 1: Clinical Characteristics

| Variable | Concordant Neg | Discordant CN-low | Discordant CN-high | Concordant Pos | p | FDR p |
|---|---|---|---|---|---|---|
"""

for row in table1_rows:
    p_str = f"{row['p']:.4f}" if not np.isnan(row.get('p', np.nan)) else '--'
    fdr_str = f"{row['p_fdr']:.4f}" if not np.isnan(row.get('p_fdr', np.nan)) else '--'
    report += (f"| {row['Variable']} | {row.get('Concordant Negative', '--')} | "
               f"{row.get('Discordant CN-low', '--')} | "
               f"{row.get('Discordant CN-high', '--')} | "
               f"{row.get('Concordant Positive', '--')} | "
               f"{p_str} | {fdr_str} |\n")

report += """
### Polysomy 17

"""

for col in [cent17_col, her2_cent17_col]:
    if col in df.columns:
        disc_vals = df[df['group'].str.startswith('Discordant')][['pid', 'group', col]].dropna(subset=[col])
        if len(disc_vals) > 0:
            report += f"**{col}** -- {len(disc_vals)} discordant patient(s) with data:\n\n"
            report += "| pid | Group | Value |\n|---|---|---|\n"
            for _, r in disc_vals.iterrows():
                report += f"| {r['pid']} | {r['group']} | {r[col]} |\n"
            report += "\n"
        else:
            report += f"**{col}** -- No discordant patients with data.\n\n"

report += """
### ER Scoring Method

"""
if er_method_col in df.columns:
    for grp in ['Discordant CN-low', 'Discordant CN-high', 'Concordant Negative']:
        subset = df[df['group'] == grp][er_method_col].dropna()
        if len(subset) > 0:
            report += f"**{grp}:** {subset.value_counts().to_dict()}\n\n"
        else:
            report += f"**{grp}:** No data.\n\n"

report += f"""
## Limitations

- CN-high discordant group (n={len(cn_high_pids)}) is severely underpowered for any
  independent statistical comparison. All CN-high findings are descriptive.
- TCGA clinical annotations are variably complete across institutions; missing data
  is not random (certain TSS sites have more complete records).
- FDR correction across {len(p_values)} tests is conservative given the small sample
  sizes in the discordant group.

## Implications

"""

if sig_findings:
    report += "Statistically significant differences in clinical characteristics "
    report += "suggest that the discordant population is clinically distinguishable "
    report += "from concordant negatives, which could inform prospective study "
    report += "inclusion criteria.\n"
else:
    report += "The lack of significant clinical differences after FDR correction "
    report += "suggests that the discordant population is clinically similar to "
    report += "concordant negatives. This implies that clinical features alone "
    report += "cannot identify these patients -- molecular testing is required.\n"

report += """
---

**Figures:**
- `fig_04_4a_table1_forest.png` -- Forest plot of effect sizes
"""

report_path = REPORT_DIR / '4a_clinical_correlates.md'
report_path.write_text(report, encoding='utf-8')
print(f"\nReport written to: {report_path}")
print("Done.")
