"""
Analysis #1: HER2 Testing Method as Confounder (Priority 6)
===========================================================
Due diligence: assess whether testing method is a confounder in discordant cases.
Severely data-limited in TCGA.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_intermediate, savefig, setup_plotting, to_patient_id
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

setup_plotting()
import matplotlib.pyplot as plt

REPORT_DIR = Path(__file__).resolve().parent.parent / 'reports' / 'biopharma'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────

disc = load_intermediate('02_discordant_cases')
mm = load_intermediate('02_multimodal_cohort')
clin = pd.read_csv(
    Path(__file__).resolve().parent.parent / 'data' / 'brca_tcga_clinical_data_cleaned.csv'
)
clin['pid'] = clin['Patient ID'].apply(to_patient_id)

# ── 2. Test method distribution ──────────────────────────────────────────────

method_col = 'her2_test_method'
n_total = len(clin)
n_with_method = clin[method_col].notna().sum()
print(f"\nTest method data: {n_with_method}/{n_total} patients ({100*n_with_method/n_total:.1f}%)")
print(f"\nTest method distribution:\n{clin[method_col].value_counts().to_string()}")

# ── 3. Discordant patients with test method ─────────────────────────────────

ihc_neg_rna_high = disc[disc['discordance_type'] == 'IHC-/RNA-high']
disc_pids = set(ihc_neg_rna_high['pid'])

disc_with_method = clin[clin['pid'].isin(disc_pids) & clin[method_col].notna()]
n_disc_method = len(disc_with_method)
print(f"\nDiscordant IHC-/RNA-high with test method: {n_disc_method}/{len(disc_pids)}")

if n_disc_method > 0:
    # Merge with expression data
    disc_detail = disc_with_method.merge(
        ihc_neg_rna_high[['pid', 'ERBB2_expr', 'erbb2_copy_number', 'GRB7_expr']],
        on='pid', how='left'
    )
    print("\nDiscordant patients with test method data:")
    display_cols = ['pid', method_col, 'HER2 ihc score', 'ERBB2_expr',
                    'erbb2_copy_number', 'GRB7_expr']
    display_cols = [c for c in display_cols if c in disc_detail.columns]
    print(disc_detail[display_cols].to_string(index=False))

# ── 4. Cross-tabulation: method vs HER2 status ──────────────────────────────

# Among patients with method data, cross-tab with composite label
clin_with_method = clin[clin[method_col].notna()].copy()
clin_with_method = clin_with_method.merge(
    mm[['pid', 'her2_composite']].drop_duplicates(), on='pid', how='left'
)

# Fill in composite from clinical data for those not in multimodal
if 'her2_composite' not in clin_with_method.columns or clin_with_method['her2_composite'].isna().any():
    pass  # Use what we have

ct_status = pd.crosstab(clin_with_method[method_col],
                         clin_with_method['her2_composite'], margins=True)
print(f"\nTest method vs HER2 composite status:\n{ct_status.to_string()}")

# Cross-tab: method vs discordant status
clin_with_method['is_discordant'] = clin_with_method['pid'].isin(disc_pids)
ct_disc = pd.crosstab(clin_with_method[method_col],
                       clin_with_method['is_discordant'], margins=True)
ct_disc.columns = ['Concordant', 'Discordant', 'All'] if len(ct_disc.columns) == 3 else ct_disc.columns
print(f"\nTest method vs discordant status:\n{ct_disc.to_string()}")

# Fisher's exact (if 2x2) or chi2
conc_neg_pids = set(mm[(mm['her2_composite'] == 'Negative') & (~mm['pid'].isin(disc_pids))]['pid'])
method_analysis = clin_with_method[
    clin_with_method['pid'].isin(disc_pids | conc_neg_pids) &
    clin_with_method[method_col].notna()
].copy()
method_analysis['group'] = method_analysis['pid'].apply(
    lambda x: 'Discordant' if x in disc_pids else 'Concordant Neg'
)

if len(method_analysis) > 0:
    ct_test = pd.crosstab(method_analysis['group'], method_analysis[method_col])
    print(f"\nDiscordant vs Concordant Neg by method:\n{ct_test.to_string()}")
    if ct_test.shape[0] >= 2 and ct_test.shape[1] >= 2:
        if ct_test.shape == (2, 2):
            odds, p = stats.fisher_exact(ct_test.values)
            print(f"Fisher's exact: odds={odds:.2f}, p={p:.4f}")
        else:
            chi2, p, dof, expected = stats.chi2_contingency(ct_test.values)
            print(f"Chi-squared: chi2={chi2:.2f}, p={p:.4f}")
    else:
        p = np.nan
        print("Insufficient categories for statistical test")
else:
    p = np.nan

# ── 5. Write report ──────────────────────────────────────────────────────────

report = f"""# Analysis 1: HER2 Testing Method as Confounder

## Key Findings

- Testing method data was available for only {n_disc_method}/{len(disc_pids)}
  ({100*n_disc_method/len(disc_pids):.0f}%) of discordant (IHC-/RNA-high) patients,
  precluding a powered confounder analysis.
- Among the {n_with_method} patients with test method annotations ({100*n_with_method/n_total:.0f}%
  of full cohort), we describe the distribution below.
- This analysis would be substantially more informative in a Tempus dataset with
  standardized testing metadata.

## Methods

HER2 testing method (`her2_test_method`) was extracted from the cleaned clinical
dataset. Cross-tabulations compared method distribution across HER2 composite status
and discordant vs concordant groups. Statistical testing was limited by sparse data.

## Results

### Test Method Distribution (Full Cohort)

| Method | Count |
|---|---|
"""

for method, count in clin[method_col].value_counts().items():
    report += f"| {method} | {count} |\n"
report += f"| Missing | {n_total - n_with_method} |\n"

report += f"""
### Test Method vs HER2 Status

{ct_status.to_markdown()}

### Test Method vs Discordant Status

{ct_disc.to_markdown()}

"""

if n_disc_method > 0:
    report += "### Discordant Patients with Test Method Data\n\n"
    report += "| pid | Method | IHC Score | ERBB2 | CN | GRB7 |\n|---|---|---|---|---|---|\n"
    for _, r in disc_detail.iterrows():
        report += (f"| {r['pid']} | {r.get(method_col, '--')} | "
                   f"{r.get('HER2 ihc score', '--')} | "
                   f"{r.get('ERBB2_expr', np.nan):.2f} | "
                   f"{r.get('erbb2_copy_number', '--')} | "
                   f"{r.get('GRB7_expr', np.nan):.2f} |\n")
    report += "\n"

report += f"""
## Limitations

- Testing method data was available for only {100*n_with_method/n_total:.0f}% of the
  cohort and {100*n_disc_method/len(disc_pids):.0f}% of discordant patients.
- The sparse data precludes any meaningful statistical inference about testing method
  as a confounder.
- TCGA samples were processed across multiple institutions over several years; testing
  method variation likely reflects institutional and temporal heterogeneity rather than
  systematic bias.

## Implications

The near-complete absence of testing method data in TCGA highlights a critical gap:
method-level metadata is essential for investigating IHC performance variability.
In a Tempus dataset with standardized testing metadata (assay platform, antibody
clone, fixation protocol), this analysis could directly test whether specific testing
configurations are associated with higher discordance rates -- a finding that would
have immediate clinical quality implications.

---

**Note:** No figures generated for this analysis due to insufficient data for
meaningful visualization.
"""

report_path = REPORT_DIR / '1_test_method_confounder.md'
report_path.write_text(report, encoding='utf-8')
print(f"\nReport written to: {report_path}")
print("Done.")
