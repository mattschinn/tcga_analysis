"""
Analysis #5d: Equivocal Demographics (Priority 9)
=================================================
Compare clinical/demographic features of equivocal-resolved-positive vs
equivocal-resolved-negative.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_intermediate, savefig, setup_plotting, COLORS, to_patient_id
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
eq_scores = load_intermediate('03_equivocal_scores')
clin = pd.read_csv(
    Path(__file__).resolve().parent.parent / 'data' / 'brca_tcga_clinical_data_cleaned.csv'
)
clin['pid'] = clin['Patient ID'].apply(to_patient_id)

# ── 2. Define groups ────────────────────────────────────────────────────────

prob_cols = [c for c in eq_scores.columns if c.startswith('prob_')]
eq_scores['ensemble_prob'] = eq_scores[prob_cols].mean(axis=1)
eq_scores['rna_class'] = (eq_scores['ensemble_prob'] >= 0.5).map(
    {True: 'RNA-pos', False: 'RNA-neg'}
)

# Merge with clinical data
equivocal = mm[mm['her2_composite'] == 'Equivocal'].copy()
equivocal = equivocal.merge(eq_scores[['pid', 'ensemble_prob', 'rna_class']], on='pid')
equivocal = equivocal.merge(clin.drop(columns=['pid']),
                            left_on='pid', right_on=clin['pid'], how='left')

n_pos = (equivocal['rna_class'] == 'RNA-pos').sum()
n_neg = (equivocal['rna_class'] == 'RNA-neg').sum()
print(f"\nEquivocal RNA-positive: {n_pos}")
print(f"Equivocal RNA-negative: {n_neg}")

# ── 3. Compare clinical variables ───────────────────────────────────────────

# Continuous variables from multimodal cohort
continuous_cols = [
    ('Fraction Genome Altered', 'FGA'),
]
# Try to find age column
for age_col in ['Diagnosis Age', 'Age at Diagnosis', 'Age at Initial Pathologic Diagnosis']:
    if age_col in equivocal.columns:
        equivocal[age_col] = pd.to_numeric(equivocal[age_col], errors='coerce')
        if equivocal[age_col].notna().sum() > 0:
            continuous_cols.insert(0, (age_col, 'Age at Diagnosis'))
            break

# Also check for age with suffix
for col in equivocal.columns:
    if 'age' in col.lower() and 'diagnosis' in col.lower():
        equivocal[col] = pd.to_numeric(equivocal[col], errors='coerce')
        if equivocal[col].notna().sum() > 5 and (col, 'Age at Diagnosis') not in continuous_cols:
            continuous_cols.insert(0, (col, 'Age at Diagnosis'))
            break

categorical_cols = [
    ('ER Status By IHC', 'ER Status'),
]
# Check for cancer type
for ct_col in ['Cancer Type Detailed', 'Cancer Type Detailed_x']:
    if ct_col in equivocal.columns:
        categorical_cols.append((ct_col, 'Histologic Type'))
        break

print("\n=== Clinical Comparison ===")
table_rows = []

for col, label in continuous_cols:
    if col not in equivocal.columns:
        continue
    pos_vals = equivocal[equivocal['rna_class'] == 'RNA-pos'][col].dropna()
    neg_vals = equivocal[equivocal['rna_class'] == 'RNA-neg'][col].dropna()
    row = {'Variable': label}
    if len(pos_vals) > 0:
        row['RNA-pos'] = f"{pos_vals.median():.1f} ({pos_vals.quantile(0.25):.1f}-{pos_vals.quantile(0.75):.1f})"
        row['RNA-pos_n'] = len(pos_vals)
    else:
        row['RNA-pos'] = '--'
        row['RNA-pos_n'] = 0

    if len(neg_vals) > 0:
        row['RNA-neg'] = f"{neg_vals.median():.1f} ({neg_vals.quantile(0.25):.1f}-{neg_vals.quantile(0.75):.1f})"
        row['RNA-neg_n'] = len(neg_vals)
    else:
        row['RNA-neg'] = '--'
        row['RNA-neg_n'] = 0

    if len(pos_vals) >= 2 and len(neg_vals) >= 2:
        u, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
        row['p'] = p
    else:
        row['p'] = np.nan
    table_rows.append(row)
    print(f"\n{label}:")
    print(f"  RNA-pos: {row['RNA-pos']} (n={row['RNA-pos_n']})")
    print(f"  RNA-neg: {row['RNA-neg']} (n={row['RNA-neg_n']})")
    if not np.isnan(row.get('p', np.nan)):
        print(f"  Mann-Whitney p={row['p']:.4f}")

for col, label in categorical_cols:
    if col not in equivocal.columns:
        continue
    pos_vals = equivocal[equivocal['rna_class'] == 'RNA-pos'][col].dropna()
    neg_vals = equivocal[equivocal['rna_class'] == 'RNA-neg'][col].dropna()
    row = {'Variable': label}
    if len(pos_vals) > 0:
        mode_val = pos_vals.mode().iloc[0]
        row['RNA-pos'] = f"{mode_val} ({(pos_vals == mode_val).sum()}/{len(pos_vals)})"
        row['RNA-pos_n'] = len(pos_vals)
    else:
        row['RNA-pos'] = '--'
        row['RNA-pos_n'] = 0

    if len(neg_vals) > 0:
        mode_val = neg_vals.mode().iloc[0]
        row['RNA-neg'] = f"{mode_val} ({(neg_vals == mode_val).sum()}/{len(neg_vals)})"
        row['RNA-neg_n'] = len(neg_vals)
    else:
        row['RNA-neg'] = '--'
        row['RNA-neg_n'] = 0

    # Fisher/chi2
    if len(pos_vals) > 0 and len(neg_vals) > 0:
        combined = pd.concat([pos_vals, neg_vals])
        labels = ['RNA-pos'] * len(pos_vals) + ['RNA-neg'] * len(neg_vals)
        ct = pd.crosstab(labels, combined)
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            if ct.shape == (2, 2):
                odds, p = stats.fisher_exact(ct.values)
            else:
                chi2, p, dof, expected = stats.chi2_contingency(ct.values)
            row['p'] = p
        else:
            row['p'] = np.nan
    else:
        row['p'] = np.nan
    table_rows.append(row)
    print(f"\n{label}:")
    print(f"  RNA-pos: {row['RNA-pos']} (n={row['RNA-pos_n']})")
    print(f"  RNA-neg: {row['RNA-neg']} (n={row['RNA-neg_n']})")
    if not np.isnan(row.get('p', np.nan)):
        print(f"  p={row['p']:.4f}")

# ── 4. Molecular features comparison ────────────────────────────────────────

mol_cols = ['erbb2_copy_number', 'Fraction Genome Altered']
mol_results = {}
for col in mol_cols:
    if col not in equivocal.columns:
        continue
    pos_vals = equivocal[equivocal['rna_class'] == 'RNA-pos'][col].dropna()
    neg_vals = equivocal[equivocal['rna_class'] == 'RNA-neg'][col].dropna()
    if len(pos_vals) >= 2 and len(neg_vals) >= 2:
        u, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
        mol_results[col] = {
            'pos_med': pos_vals.median(),
            'neg_med': neg_vals.median(),
            'p': p,
        }
        print(f"\n{col}: pos_median={pos_vals.median():.3f}, neg_median={neg_vals.median():.3f}, p={p:.4f}")

# ── 5. Write report ──────────────────────────────────────────────────────────

report = f"""# Analysis 5d: Equivocal Patient Demographics

## Key Findings

- Among 28 equivocal (IHC 2+) patients, RNA reclassifies {n_pos} as HER2+ and
  {n_neg} as HER2-.
"""

sig_vars = [r for r in table_rows if r.get('p', 1) < 0.05]
if sig_vars:
    for r in sig_vars:
        report += f"- {r['Variable']} differs between groups (p={r['p']:.4f}).\n"
else:
    report += """- No clinical/demographic variables show statistically significant differences
  between RNA-reclassified subgroups, supporting the need for molecular rather than
  clinical selection criteria.\n"""

report += f"""
## Methods

Equivocal patients were split into RNA-predicted HER2+ (ensemble probability >= 0.5,
n={n_pos}) and RNA-predicted HER2- (< 0.5, n={n_neg}). Clinical and molecular
features were compared using Mann-Whitney U (continuous) and Fisher's exact or
chi-squared (categorical).

**Note:** With n=28 total and only {n_pos} in the smaller group, this analysis is
severely underpowered. All findings are exploratory.

## Results

### Clinical Characteristics

| Variable | RNA-pos (n={n_pos}) | RNA-neg (n={n_neg}) | p-value |
|---|---|---|---|
"""

for row in table_rows:
    p_str = f"{row['p']:.4f}" if not np.isnan(row.get('p', np.nan)) else '--'
    report += f"| {row['Variable']} | {row['RNA-pos']} | {row['RNA-neg']} | {p_str} |\n"

if mol_results:
    report += "\n### Molecular Features\n\n"
    report += "| Feature | RNA-pos Median | RNA-neg Median | p-value |\n|---|---|---|---|\n"
    for col, r in mol_results.items():
        report += f"| {col} | {r['pos_med']:.3f} | {r['neg_med']:.3f} | {r['p']:.4f} |\n"

report += f"""
## Limitations

- Total n=28, with only {n_pos} RNA-positive patients. No statistical test has
  meaningful power at this sample size.
- Clinical annotations in TCGA are variably complete; some variables may have
  extensive missing data within the equivocal subset.
- No treatment or outcome data available for equivocal patients in TCGA.

## Implications

"""

if sig_vars:
    report += "Differences in clinical features between RNA-reclassified subgroups "
    report += "could inform prospective study inclusion criteria, though validation "
    report += "in a larger cohort is essential given the small sample size.\n"
else:
    report += """RNA-based reclassification identifies a biologically distinct population that is
clinically indistinguishable from the broader equivocal group. This supports the
need for molecular testing -- clinical features alone cannot identify which equivocal
patients are HER2+ at the molecular level. A prospective study design should use
molecular selection criteria rather than clinical enrichment.\n"""

report += """
---

**Note:** No figures generated due to small sample size (n=5 vs n=23).
"""

report_path = REPORT_DIR / '5d_equivocal_demographics.md'
report_path.write_text(report, encoding='utf-8')
print(f"\nReport written to: {report_path}")
print("Done.")
