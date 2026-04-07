"""
Analysis #5b: Multi-Modal Concordance Tiers (Priority 7)
========================================================
Stratify equivocal patients by agreement across modalities (RNA, CN, FISH).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_intermediate, savefig, setup_plotting, COLORS
import pandas as pd
import numpy as np
from pathlib import Path

setup_plotting()
import matplotlib.pyplot as plt
import seaborn as sns

REPORT_DIR = Path(__file__).resolve().parent.parent / 'reports' / 'biopharma'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────

mm = load_intermediate('02_multimodal_cohort')
analysis_df = load_intermediate('02_analysis_df')
eq_scores = load_intermediate('03_equivocal_scores')

# ── 2. Build per-patient multimodal status table ────────────────────────────

# Ensemble RNA probability
prob_cols = [c for c in eq_scores.columns if c.startswith('prob_')]
eq_scores['ensemble_prob'] = eq_scores[prob_cols].mean(axis=1)
eq_scores['rna_call'] = (eq_scores['ensemble_prob'] >= 0.5).map(
    {True: 'Positive', False: 'Negative'}
)

# Merge with multimodal data
equivocal = mm[mm['her2_composite'] == 'Equivocal'][['pid', 'erbb2_copy_number',
    'HER2 fish status', 'ER Status By IHC', 'Fraction Genome Altered']].copy()
equivocal = equivocal.merge(eq_scores[['pid', 'ensemble_prob', 'rna_call']], on='pid')
equivocal = equivocal.merge(
    analysis_df[['pid', 'ERBB2_expr', 'GRB7_expr']], on='pid', how='left'
)

# CN call
equivocal['cn_call'] = equivocal['erbb2_copy_number'].apply(
    lambda x: 'Amplified' if x >= 2 else 'Not amplified' if pd.notna(x) else 'Unknown'
)

# FISH call
equivocal['fish_call'] = equivocal['HER2 fish status'].apply(
    lambda x: x.strip().capitalize() if pd.notna(x) and x.strip().lower() in ['positive', 'negative']
    else 'NA'
)

print(f"\nEquivocal patients: {len(equivocal)}")
print(f"RNA calls: {equivocal['rna_call'].value_counts().to_string()}")
print(f"CN calls: {equivocal['cn_call'].value_counts().to_string()}")
print(f"FISH calls: {equivocal['fish_call'].value_counts().to_string()}")

# ── 3. Assign concordance tiers ─────────────────────────────────────────────

def assign_tier(row):
    rna_pos = row['rna_call'] == 'Positive'
    cn_amp = row['cn_call'] == 'Amplified'
    fish_pos = row['fish_call'] == 'Positive'
    fish_na = row['fish_call'] == 'NA'

    if rna_pos and (cn_amp or fish_pos):
        return 'Tier 1: High confidence HER2+'
    elif rna_pos and not cn_amp and not fish_pos:
        return 'Tier 2: RNA-only HER2+'
    elif not rna_pos and not cn_amp and not fish_pos:
        return 'Tier 3: Concordant HER2-'
    else:
        return 'Tier 4: Mixed signals'

equivocal['tier'] = equivocal.apply(assign_tier, axis=1)
tier_counts = equivocal['tier'].value_counts().sort_index()
print(f"\nConcordance tiers:\n{tier_counts.to_string()}")

# ── 4. Characterize each tier biologically ──────────────────────────────────

print("\n=== Biological Characterization by Tier ===")
tier_stats = []
for tier in sorted(equivocal['tier'].unique()):
    subset = equivocal[equivocal['tier'] == tier]
    stat = {
        'tier': tier,
        'n': len(subset),
        'erbb2_med': subset['ERBB2_expr'].median(),
        'grb7_med': subset['GRB7_expr'].median(),
        'ml_prob_med': subset['ensemble_prob'].median(),
        'er_pos_pct': (subset['ER Status By IHC'] == 'Positive').mean() * 100 if len(subset) > 0 else np.nan,
        'fga_med': subset['Fraction Genome Altered'].median(),
        'cn_amp_pct': (subset['cn_call'] == 'Amplified').mean() * 100,
    }
    tier_stats.append(stat)
    print(f"\n{tier} (n={stat['n']}):")
    print(f"  ERBB2 median: {stat['erbb2_med']:.2f}")
    print(f"  GRB7 median: {stat['grb7_med']:.2f}")
    print(f"  ML prob median: {stat['ml_prob_med']:.3f}")
    print(f"  ER+ rate: {stat['er_pos_pct']:.0f}%")
    print(f"  FGA median: {stat['fga_med']:.3f}")

tier_df = pd.DataFrame(tier_stats)

# ── 5. Figures ───────────────────────────────────────────────────────────────

tier_order = sorted(equivocal['tier'].unique())
tier_colors = {
    'Tier 1: High confidence HER2+': '#e74c3c',
    'Tier 2: RNA-only HER2+': '#e67e22',
    'Tier 3: Concordant HER2-': '#3498db',
    'Tier 4: Mixed signals': '#95a5a6',
}

# Fig: Grouped bar/dot plot
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, (metric, label) in zip(axes, [
    ('ERBB2_expr', 'ERBB2 Expression'),
    ('GRB7_expr', 'GRB7 Expression'),
    ('ensemble_prob', 'ML Probability'),
]):
    plot_data = equivocal[equivocal[metric].notna()]
    if len(plot_data) == 0:
        continue
    order = [t for t in tier_order if t in plot_data['tier'].values]
    sns.boxplot(data=plot_data, x='tier', y=metric, order=order,
                palette=tier_colors, ax=ax, showfliers=False)
    sns.stripplot(data=plot_data, x='tier', y=metric, order=order,
                  color='black', alpha=0.5, size=5, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(label)
    ax.set_xticklabels([t.split(':')[0] for t in order], rotation=0)
    for i, t in enumerate(order):
        n = len(plot_data[plot_data['tier'] == t])
        ax.text(i, ax.get_ylim()[0], f'n={n}', ha='center', fontsize=8)

fig.suptitle('Biological Characterization by Concordance Tier', fontsize=13, y=1.02)
plt.tight_layout()
savefig(fig, 'fig_04_5b_concordance_tiers')
plt.close()

# ── 6. Patient-level table ──────────────────────────────────────────────────

patient_table = equivocal[['pid', 'rna_call', 'cn_call', 'fish_call', 'tier',
                            'ERBB2_expr', 'ensemble_prob']].sort_values('tier')
print("\n=== Patient-Level Concordance Table ===")
print(patient_table.to_string(index=False))

# ── 7. Write report ──────────────────────────────────────────────────────────

report = f"""# Analysis 5b: Multi-Modal Concordance Tiers for Equivocal Patients

## Key Findings

"""

for _, row in tier_df.iterrows():
    report += f"- **{row['tier']}:** {row['n']:.0f} patients"
    if row['n'] > 0:
        report += f" (ERBB2 median={row['erbb2_med']:.2f}, ML prob={row['ml_prob_med']:.3f})"
    report += "\n"

report += f"""
## Methods

The 28 equivocal (IHC 2+) patients were classified into concordance tiers based on
agreement across three modalities:

- **RNA call:** Ensemble ML probability >= 0.5 -> Positive
- **CN call:** GISTIC copy number >= 2 -> Amplified
- **FISH call:** Definitive FISH result (Positive/Negative) or NA

Tier definitions:
- **Tier 1 (High confidence HER2+):** RNA+ AND (CN amplified OR FISH+)
- **Tier 2 (RNA-only HER2+):** RNA+ AND CN not amplified AND FISH not positive
- **Tier 3 (Concordant HER2-):** RNA- AND CN not amplified AND FISH not positive
- **Tier 4 (Mixed signals):** Any other combination

## Results

### Tier Distribution

| Tier | N | ERBB2 Median | GRB7 Median | ML Prob Median | ER+ Rate | FGA Median |
|---|---|---|---|---|---|---|
"""

for _, row in tier_df.iterrows():
    report += (f"| {row['tier']} | {row['n']:.0f} | {row['erbb2_med']:.2f} | "
               f"{row['grb7_med']:.2f} | {row['ml_prob_med']:.3f} | "
               f"{row['er_pos_pct']:.0f}% | {row['fga_med']:.3f} |\n")

report += """
### Patient-Level Table

| pid | RNA Call | CN Call | FISH Call | Tier | ERBB2 | ML Prob |
|---|---|---|---|---|---|---|
"""

for _, row in patient_table.iterrows():
    report += (f"| {row['pid']} | {row['rna_call']} | {row['cn_call']} | "
               f"{row['fish_call']} | {row['tier'].split(':')[0]} | "
               f"{row['ERBB2_expr']:.2f} | {row['ensemble_prob']:.3f} |\n")

report += f"""
## Limitations

- FISH data is absent for all 28 equivocal patients (all NA or Equivocal/Indeterminate),
  so FISH does not contribute to tier assignment in this dataset.
- Tier assignments rely primarily on RNA + CN agreement.
- Small sample size (n=28) limits statistical power for between-tier comparisons.

## Implications

Tier 1 patients (RNA+ and CN amplified) represent the highest-confidence candidates
for HER2 reclassification -- multiple independent modalities agree on HER2+ biology.
Tier 2 patients (RNA-only) may still be HER2+ but require additional validation.
Tier 3 patients are concordantly HER2- across available modalities.
Tier 4 patients show mixed signals and would be candidates for additional testing
in a clinical setting.

In a Tempus dataset with complete FISH data, tier assignment would gain a third
modality, further sharpening the confidence stratification.

---

**Figures:**
- `fig_04_5b_concordance_tiers.png` -- Biological characterization by tier
"""

report_path = REPORT_DIR / '5b_concordance_tiers.md'
report_path.write_text(report, encoding='utf-8')
print(f"\nReport written to: {report_path}")
print("Done.")
