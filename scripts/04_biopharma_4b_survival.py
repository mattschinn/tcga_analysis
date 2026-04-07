"""
Analysis #4b: Survival Analysis (Priority 8)
=============================================
KM survival for discordant vs concordant HER2-negative.
Severely underpowered -- hypothesis-generating only.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_intermediate, savefig, setup_plotting, COLORS
import pandas as pd
import numpy as np
from pathlib import Path

setup_plotting()
import matplotlib.pyplot as plt

REPORT_DIR = Path(__file__).resolve().parent.parent / 'reports' / 'biopharma'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────

mm = load_intermediate('02_multimodal_cohort')
disc = load_intermediate('02_discordant_cases')

# ── 2. Prepare survival data ────────────────────────────────────────────────

# Parse survival status
def parse_os_status(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if '1:' in s or 'DECEASED' in s or 'DEAD' in s:
        return 1
    if '0:' in s or 'LIVING' in s or 'ALIVE' in s:
        return 0
    return np.nan

mm['os_event'] = mm['Overall Survival Status'].apply(parse_os_status)
mm['os_time'] = pd.to_numeric(mm['Overall Survival (Months)'], errors='coerce')
mm['dfs_event'] = mm['Disease Free Status'].apply(parse_os_status)
mm['dfs_time'] = pd.to_numeric(mm['Disease Free (Months)'], errors='coerce')

print(f"\nOS data available: {mm['os_time'].notna().sum()} time, {mm['os_event'].notna().sum()} event")
print(f"DFS data available: {mm['dfs_time'].notna().sum()} time, {mm['dfs_event'].notna().sum()} event")

# ── 3. Define groups ────────────────────────────────────────────────────────

ihc_neg_rna_high = disc[disc['discordance_type'] == 'IHC-/RNA-high']
disc_pids = set(ihc_neg_rna_high['pid'])
cn_high_pids = set(ihc_neg_rna_high[ihc_neg_rna_high['erbb2_copy_number'] >= 2]['pid'])
cn_low_pids = set(ihc_neg_rna_high[ihc_neg_rna_high['erbb2_copy_number'] <= 1]['pid'])

mm['surv_group'] = 'Other'
conc_neg_mask = (mm['her2_composite'] == 'Negative') & (~mm['pid'].isin(disc_pids))
mm.loc[conc_neg_mask, 'surv_group'] = 'Concordant Negative'
mm.loc[mm['pid'].isin(disc_pids), 'surv_group'] = 'Discordant'
mm.loc[mm['her2_composite'] == 'Positive', 'surv_group'] = 'Concordant Positive'

# More granular discordant
mm['surv_group_cn'] = mm['surv_group']
mm.loc[mm['pid'].isin(cn_high_pids), 'surv_group_cn'] = 'Discordant CN-high'
mm.loc[mm['pid'].isin(cn_low_pids), 'surv_group_cn'] = 'Discordant CN-low'

print(f"\nSurvival group sizes:")
print(mm['surv_group'].value_counts().to_string())

# ── 4. KM analysis ──────────────────────────────────────────────────────────

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# Filter to valid OS data
os_data = mm[mm['os_time'].notna() & mm['os_event'].notna()].copy()
print(f"\nPatients with complete OS data: {len(os_data)}")

# Events per group
for grp in ['Concordant Negative', 'Discordant', 'Concordant Positive']:
    subset = os_data[os_data['surv_group'] == grp]
    n_events = subset['os_event'].sum()
    print(f"  {grp}: n={len(subset)}, events={n_events:.0f}")

# KM curves: OS
groups_to_plot = ['Concordant Negative', 'Discordant', 'Concordant Positive']
group_colors_km = {
    'Concordant Negative': COLORS['Negative'],
    'Discordant': '#f39c12',
    'Concordant Positive': COLORS['Positive'],
}

fig, ax = plt.subplots(figsize=(10, 7))
kmf = KaplanMeierFitter()

km_results = {}
for grp in groups_to_plot:
    subset = os_data[os_data['surv_group'] == grp]
    if len(subset) < 3:
        continue
    kmf.fit(subset['os_time'], event_observed=subset['os_event'], label=grp)
    kmf.plot_survival_function(ax=ax, color=group_colors_km[grp], linewidth=2)

    # Median and 5-year survival
    median_surv = kmf.median_survival_time_
    surv_5yr = kmf.predict(60) if 60 <= subset['os_time'].max() else np.nan
    km_results[grp] = {
        'n': len(subset),
        'events': int(subset['os_event'].sum()),
        'median_os': median_surv,
        'surv_5yr': float(surv_5yr) if not isinstance(surv_5yr, (float, int)) or not np.isnan(surv_5yr) else np.nan,
    }

ax.set_xlabel('Time (Months)')
ax.set_ylabel('Overall Survival Probability')
ax.set_title('Kaplan-Meier: Overall Survival by HER2 Concordance Group')
ax.legend(loc='lower left')
plt.tight_layout()
savefig(fig, 'fig_04_4b_km_os')
plt.close()

# Log-rank: Concordant Negative vs Discordant
cn_data = os_data[os_data['surv_group'] == 'Concordant Negative']
disc_data = os_data[os_data['surv_group'] == 'Discordant']
if len(cn_data) > 0 and len(disc_data) > 0:
    lr = logrank_test(cn_data['os_time'], disc_data['os_time'],
                      cn_data['os_event'], disc_data['os_event'])
    lr_p = lr.p_value
    lr_stat = lr.test_statistic
    print(f"\nLog-rank (Concordant Neg vs Discordant): chi2={lr_stat:.2f}, p={lr_p:.4f}")
else:
    lr_p, lr_stat = np.nan, np.nan

# Cox PH: univariate
cox_data = os_data[os_data['surv_group'].isin(['Concordant Negative', 'Discordant'])].copy()
cox_data['is_discordant'] = (cox_data['surv_group'] == 'Discordant').astype(int)
hr, hr_lo, hr_hi, cox_p = np.nan, np.nan, np.nan, np.nan

if cox_data['os_event'].sum() >= 5 and cox_data['is_discordant'].nunique() > 1:
    cph = CoxPHFitter()
    try:
        cph.fit(cox_data[['os_time', 'os_event', 'is_discordant']],
                duration_col='os_time', event_col='os_event')
        hr = np.exp(cph.params_['is_discordant'])
        ci = cph.confidence_intervals_.loc['is_discordant']
        hr_lo = np.exp(ci.iloc[0])
        hr_hi = np.exp(ci.iloc[1])
        cox_p = cph.summary['p']['is_discordant']
        print(f"Cox PH: HR={hr:.2f} ({hr_lo:.2f}-{hr_hi:.2f}), p={cox_p:.4f}")
    except Exception as e:
        print(f"Cox PH failed: {e}")

# ── 5. DFS analysis (if sufficient data) ────────────────────────────────────

dfs_data = mm[mm['dfs_time'].notna() & mm['dfs_event'].notna()].copy()
dfs_n = len(dfs_data)
print(f"\nPatients with complete DFS data: {dfs_n}")

dfs_results = {}
if dfs_n > 50:
    fig, ax = plt.subplots(figsize=(10, 7))
    for grp in groups_to_plot:
        subset = dfs_data[dfs_data['surv_group'] == grp]
        if len(subset) < 3:
            continue
        kmf.fit(subset['dfs_time'], event_observed=subset['dfs_event'], label=grp)
        kmf.plot_survival_function(ax=ax, color=group_colors_km[grp], linewidth=2)
        dfs_results[grp] = {
            'n': len(subset),
            'events': int(subset['dfs_event'].sum()),
            'median_dfs': kmf.median_survival_time_,
        }
    ax.set_xlabel('Time (Months)')
    ax.set_ylabel('Disease-Free Survival Probability')
    ax.set_title('Kaplan-Meier: Disease-Free Survival by HER2 Concordance Group')
    ax.legend(loc='lower left')
    plt.tight_layout()
    savefig(fig, 'fig_04_4b_km_dfs')
    plt.close()

# ── 6. Write report ──────────────────────────────────────────────────────────

report = f"""# Analysis 4b: Survival Analysis

## Key Findings

- **This analysis is hypothesis-generating.** With n={len(disc_data)} discordant
  patients and {disc_data['os_event'].sum():.0f} events, we are severely underpowered
  to detect clinically meaningful survival differences.
"""

if not np.isnan(hr):
    report += f"- Hazard ratio (discordant vs concordant negative): HR={hr:.2f} "
    report += f"(95% CI: {hr_lo:.2f}-{hr_hi:.2f}, p={cox_p:.4f}).\n"
    if hr_hi - hr_lo > 2:
        report += "- The wide confidence interval reflects severe underpowering.\n"

report += f"- Log-rank test: chi2={lr_stat:.2f}, p={lr_p:.4f}\n"

report += f"""
## Methods

Kaplan-Meier survival curves were generated for Overall Survival (OS) comparing
Concordant Negative, Discordant (IHC-/RNA-high), and Concordant Positive groups.
Log-rank test compared Concordant Negative vs Discordant. A univariate Cox
proportional hazards model estimated the hazard ratio for discordant status.

Survival status was parsed from TCGA encoding (1:DECEASED = event, 0:LIVING = censored).

## Results

### Overall Survival

| Group | N | Events | Median OS (months) |
|---|---|---|---|
"""

for grp in groups_to_plot:
    if grp in km_results:
        r = km_results[grp]
        med_str = f"{r['median_os']:.1f}" if not np.isinf(r['median_os']) else "Not reached"
        report += f"| {grp} | {r['n']} | {r['events']} | {med_str} |\n"

report += f"""
**Log-rank test (Concordant Neg vs Discordant):** chi2={lr_stat:.2f}, p={lr_p:.4f}
"""

if not np.isnan(hr):
    report += f"""
**Cox Proportional Hazards (univariate):**
- HR (discordant vs concordant negative) = {hr:.2f} (95% CI: {hr_lo:.2f}-{hr_hi:.2f})
- p = {cox_p:.4f}
"""

if dfs_results:
    report += "\n### Disease-Free Survival\n\n"
    report += "| Group | N | Events | Median DFS (months) |\n|---|---|---|---|\n"
    for grp in groups_to_plot:
        if grp in dfs_results:
            r = dfs_results[grp]
            med_str = f"{r['median_dfs']:.1f}" if not np.isinf(r['median_dfs']) else "Not reached"
            report += f"| {grp} | {r['n']} | {r['events']} | {med_str} |\n"

report += f"""
## Limitations

- **Severely underpowered.** The discordant group has n={len(disc_data)} patients with
  {disc_data['os_event'].sum():.0f} events. Standard power calculations suggest n>100 per arm
  for detecting HR differences of 1.5 with 80% power. Our analysis has <10% of this
  requirement.
- TCGA follow-up is variable; censoring patterns may differ across institutions.
- Treatment data is not available; survival differences (or lack thereof) cannot be
  attributed to HER2 status vs treatment vs confounders.
- No adjustment for age, stage, ER status, or other prognostic factors (insufficient
  sample for multivariable modeling in the discordant group).

## Implications

"""

if not np.isnan(hr):
    if hr > 1.3:
        report += f"""The point estimate (HR={hr:.2f}) suggests a potential survival disadvantage
for discordant patients, which could reflect undertreated HER2-driven disease. However,
the wide confidence interval ({hr_lo:.2f}-{hr_hi:.2f}) spans both no effect and
clinically large effects, making any conclusion premature. This trend would merit
investigation in a larger Tempus cohort with linked treatment data.\n"""
    elif hr < 0.7:
        report += f"""The point estimate (HR={hr:.2f}) suggests a potential survival advantage
for discordant patients. This could reflect favorable biology in the ER-driven CN-low
subgroup. However, the wide CI makes this inconclusive.\n"""
    else:
        report += f"""The point estimate (HR={hr:.2f}) is near null, suggesting no large survival
difference between discordant and concordant negative patients. This does not rule out
a clinically meaningful effect -- the study is severely underpowered.\n"""

report += """
---

**Figures:**
- `fig_04_4b_km_os.png` -- Kaplan-Meier curves for OS
"""
if dfs_results:
    report += "- `fig_04_4b_km_dfs.png` -- Kaplan-Meier curves for DFS\n"

report_path = REPORT_DIR / '4b_survival.md'
report_path.write_text(report, encoding='utf-8')
print(f"\nReport written to: {report_path}")
print("Done.")
