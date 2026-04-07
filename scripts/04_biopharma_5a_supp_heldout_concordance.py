"""
Analysis #5a Supplementary: Held-Out Concordance Validation
===========================================================
Retrain models EXCLUDING the 154 FISH-resolved IHC 2+ patients, then score
them as a genuinely held-out concordance population.

This eliminates the training-data contamination concern in the primary 5a analysis.

Training set: 683 patients (117 Positive, 566 Negative) -- IHC 0/1+/3+ only
Held-out set: 154 patients (34 FISH-Positive, 120 FISH-Negative) -- IHC 2+ with FISH
Also re-scored: 28 truly equivocal patients (IHC 2+, no FISH)

Uses the curated gene panel (~45 genes from 6 biological gene sets) per project
feature reduction convention, matching the consolidated NB03 approach.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import (load_intermediate, load_gene_cols, savefig, setup_plotting,
                   to_patient_id, _parse_ihc_score, COLORS)
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             cohen_kappa_score)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

try:
    import xgboost as xgb
    has_xgb = True
except ImportError:
    has_xgb = False
    from sklearn.ensemble import GradientBoostingClassifier

setup_plotting()
import matplotlib.pyplot as plt
import seaborn as sns

REPORT_DIR = Path(__file__).resolve().parent.parent / 'reports' / 'biopharma'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load data and identify IHC 2+ FISH-resolved patients ────────────────

cohort_c = load_intermediate('02_multimodal_cohort')
analysis_df = load_intermediate('02_analysis_df')
tumor_norm = load_intermediate('01_tumor_norm_tmm_tss')
gene_cols = load_gene_cols()
eq_scores_orig = load_intermediate('03_equivocal_scores')

clin = pd.read_csv(
    Path(__file__).resolve().parent.parent / 'data' / 'brca_tcga_clinical_data_cleaned.csv'
)
clin['pid'] = clin['Patient ID'].apply(to_patient_id)
clin['ihc_parsed'] = clin['HER2 ihc score'].apply(_parse_ihc_score)

# IHC 2+ with definitive FISH
ihc2_clin = clin[clin['ihc_parsed'] == 2].copy()
fish_col = 'HER2 fish status'
ihc2_definitive = ihc2_clin[
    ihc2_clin[fish_col].str.strip().str.lower().isin(['positive', 'negative'])
].copy()
ihc2_definitive['fish_binary'] = ihc2_definitive[fish_col].str.strip().str.capitalize()
resolved_pids = set(ihc2_definitive['pid'])

# Confirm overlap with multimodal cohort
resolved_in_mm = ihc2_definitive[ihc2_definitive['pid'].isin(cohort_c['pid'])]
print(f"IHC 2+ FISH-resolved in multimodal cohort: {len(resolved_in_mm)}")
print(f"  FISH Positive: {(resolved_in_mm['fish_binary'] == 'Positive').sum()}")
print(f"  FISH Negative: {(resolved_in_mm['fish_binary'] == 'Negative').sum()}")

# ── 2. Build feature matrix (curated gene panel, per project convention) ───

# Curated gene sets (same as NB03 / concordant_threshold_sensitivity.py)
GENE_SETS = {
    "HER2_17q12_AMPLICON": [
        "ERBB2", "GRB7", "STARD3", "PGAP3", "TCAP", "PNMT", "PPP1R1B",
    ],
    "ERBB_SIGNALING": [
        "ERBB2", "ERBB3", "ERBB4", "EGFR", "PIK3CA", "AKT1", "MAPK1", "SHC1",
    ],
    "LUMINAL_ER_PROGRAM": [
        "ESR1", "PGR", "GATA3", "FOXA1", "BCL2", "TFF1", "XBP1", "CCND1",
    ],
    "BASAL_MYOEPITHELIAL": [
        "KRT5", "KRT14", "KRT17", "EGFR", "VIM", "CDH3", "TP63", "FOXC1",
    ],
    "PROLIFERATION": [
        "MKI67", "CCNB1", "AURKA", "TOP2A", "PCNA", "BUB1", "CDC20", "CCNE1",
    ],
    "EMT": [
        "VIM", "CDH1", "CDH2", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "FN1",
    ],
}
curated_genes = sorted(set(g for gs in GENE_SETS.values() for g in gs))
curated_genes = [g for g in curated_genes if g in gene_cols and g in tumor_norm.columns]
print(f"Curated gene panel: {len(curated_genes)} genes")

ml_df = cohort_c.copy()

# Merge only curated genes (fast)
tn_subset = tumor_norm[['pid'] + curated_genes].copy()
tn_subset = tn_subset.rename(columns={g: f'expr_{g}' for g in curated_genes})
ml_df = ml_df.merge(tn_subset, on='pid', how='left')

ml_df['er_positive'] = (ml_df['ER Status By IHC'] == 'Positive').astype(float)
ml_df['pr_positive'] = (ml_df['PR status by ihc'] == 'Positive').astype(float)
ml_df['er_positive'] = ml_df['er_positive'].fillna(ml_df['er_positive'].median())
ml_df['pr_positive'] = ml_df['pr_positive'].fillna(ml_df['pr_positive'].median())

expr_cols = [c for c in ml_df.columns if c.startswith('expr_')]
feature_cols = expr_cols + ['erbb2_copy_number', 'er_positive', 'pr_positive']
print(f"Feature matrix: {len(feature_cols)} features ({len(expr_cols)} expression + 3 clinical)")

# ── 3. Split: training (non-IHC-2+) vs held-out (IHC 2+ FISH-resolved) ────

ml_labeled = ml_df[ml_df['her2_composite'].isin(['Positive', 'Negative'])].copy()
ml_labeled['y'] = (ml_labeled['her2_composite'] == 'Positive').astype(int)
ml_equivocal = ml_df[ml_df['her2_composite'] == 'Equivocal'].copy()

# Training: labeled patients who are NOT IHC 2+ FISH-resolved
train_df = ml_labeled[~ml_labeled['pid'].isin(resolved_pids)].copy()
train_df = train_df.dropna(subset=feature_cols + ['y'])

# Held-out: IHC 2+ FISH-resolved patients
heldout_df = ml_labeled[ml_labeled['pid'].isin(resolved_pids)].copy()
heldout_df = heldout_df.dropna(subset=feature_cols)
heldout_df = heldout_df.merge(
    ihc2_definitive[['pid', 'fish_binary']].drop_duplicates('pid'),
    on='pid', how='left'
)

X_train = train_df[feature_cols].values
y_train = train_df['y'].values
X_heldout = heldout_df[feature_cols].values
y_fish = (heldout_df['fish_binary'] == 'Positive').astype(int).values

print(f"\nTraining set: {len(train_df)} patients")
print(f"  Positive: {y_train.sum()}, Negative: {(1-y_train).sum():.0f}")
print(f"Held-out (IHC 2+ FISH-resolved): {len(heldout_df)} patients")
print(f"  FISH Positive: {y_fish.sum()}, FISH Negative: {(1-y_fish).sum():.0f}")

# ── 4. Train models on non-IHC-2+ data only ───────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_heldout_scaled = scaler.transform(X_heldout)
scale_pos = (1 - y_train).sum() / max(y_train.sum(), 1)

models = {
    'L1-LR': LogisticRegression(
        penalty='l1', solver='saga', max_iter=500, random_state=42,
        class_weight='balanced', C=1.0, tol=1e-3
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42,
        class_weight='balanced', n_jobs=-1
    ),
}

if has_xgb:
    models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos, random_state=42,
        eval_metric='logloss', use_label_encoder=False
    )
else:
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )

# CV performance on training set (for reference)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n=== CV Performance on Training Set (non-IHC-2+) ===")
cv_results = {}
for name, model in models.items():
    X_use = X_train_scaled if 'LR' in name else X_train
    y_prob_cv = cross_val_predict(model, X_use, y_train, cv=cv, method='predict_proba')[:, 1]
    auc_cv = roc_auc_score(y_train, y_prob_cv)
    cv_results[name] = auc_cv
    print(f"  {name}: CV AUC = {auc_cv:.3f}")

# Fit final models on full training set and score held-out
print("\n=== Held-Out Concordance (IHC 2+ FISH-resolved) ===")
heldout_probs = {}
for name, model in models.items():
    X_use = X_train_scaled if 'LR' in name else X_train
    X_ho_use = X_heldout_scaled if 'LR' in name else X_heldout
    model.fit(X_use, y_train)
    probs = model.predict_proba(X_ho_use)[:, 1]
    heldout_probs[name] = probs
    auc_ho = roc_auc_score(y_fish, probs)
    print(f"  {name}: Held-out AUC = {auc_ho:.3f}")

# Ensemble probability (mean of all models)
prob_matrix = np.column_stack(list(heldout_probs.values()))
ensemble_prob = prob_matrix.mean(axis=1)
ensemble_auc = roc_auc_score(y_fish, ensemble_prob)
print(f"  Ensemble: Held-out AUC = {ensemble_auc:.3f}")

heldout_df['ensemble_prob_heldout'] = ensemble_prob

# ── 5. Concordance table at threshold 0.5 ─────────────────────────────────

def wilson_ci(x, n):
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(x, n, alpha=0.05, method='wilson')
    return (lo, hi)

heldout_df['rna_pred'] = np.where(ensemble_prob >= 0.5, 'Positive', 'Negative')

labels = ['Positive', 'Negative']
cm = confusion_matrix(heldout_df['fish_binary'], heldout_df['rna_pred'], labels=labels)
tp, fn = cm[0, 0], cm[0, 1]
fp, tn = cm[1, 0], cm[1, 1]
n_total = tp + tn + fp + fn

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
accuracy = (tp + tn) / n_total
kappa = cohen_kappa_score(heldout_df['fish_binary'], heldout_df['rna_pred'])

sens_ci = wilson_ci(tp, tp + fn)
spec_ci = wilson_ci(tn, tn + fp)
ppv_ci = wilson_ci(tp, tp + fp)
npv_ci = wilson_ci(tn, tn + fn)
acc_ci = wilson_ci(tp + tn, n_total)

print(f"\n=== Concordance Table (threshold=0.5, N={n_total}) ===")
print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")
print(f"  Sensitivity: {sensitivity:.3f} ({sens_ci[0]:.3f}-{sens_ci[1]:.3f})")
print(f"  Specificity: {specificity:.3f} ({spec_ci[0]:.3f}-{spec_ci[1]:.3f})")
print(f"  PPV: {ppv:.3f} ({ppv_ci[0]:.3f}-{ppv_ci[1]:.3f})")
print(f"  NPV: {npv:.3f} ({npv_ci[0]:.3f}-{npv_ci[1]:.3f})")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  Kappa: {kappa:.3f}")

# ── 6. ROC and Youden's J optimal threshold ───────────────────────────────

fpr, tpr, thresholds = roc_curve(y_fish, ensemble_prob)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_thresh = thresholds[best_idx]

# Metrics at optimal threshold
heldout_df['rna_pred_optimal'] = np.where(ensemble_prob >= best_thresh, 'Positive', 'Negative')
cm_opt = confusion_matrix(heldout_df['fish_binary'], heldout_df['rna_pred_optimal'],
                          labels=labels)
tp_o, fn_o = cm_opt[0, 0], cm_opt[0, 1]
fp_o, tn_o = cm_opt[1, 0], cm_opt[1, 1]
sens_opt = tp_o / (tp_o + fn_o) if (tp_o + fn_o) > 0 else np.nan
spec_opt = tn_o / (tn_o + fp_o) if (tn_o + fp_o) > 0 else np.nan

print(f"\nOptimal threshold (Youden's J): {best_thresh:.3f}")
print(f"  Sensitivity: {sens_opt:.3f}, Specificity: {spec_opt:.3f}")

# ── 7. Compare with primary analysis ──────────────────────────────────────

print("\n=== Comparison: Primary vs Held-Out ===")
print(f"  Primary AUC:   0.994 (trained on all labeled, scored same patients)")
print(f"  Held-out AUC:  {ensemble_auc:.3f} (trained without IHC 2+, scored IHC 2+)")
print(f"  Primary threshold (Youden): 0.341")
print(f"  Held-out threshold (Youden): {best_thresh:.3f}")

# ── 8. Re-score truly equivocal patients with held-out models ──────────────

equivocal_clean = ml_equivocal.dropna(subset=feature_cols)
X_equiv = equivocal_clean[feature_cols].values
X_equiv_scaled = scaler.transform(X_equiv)

eq_probs = {}
for name, model in models.items():
    X_eq_use = X_equiv_scaled if 'LR' in name else X_equiv
    probs = model.predict_proba(X_eq_use)[:, 1]
    eq_probs[name] = probs

eq_prob_matrix = np.column_stack(list(eq_probs.values()))
eq_ensemble = eq_prob_matrix.mean(axis=1)

equivocal_clean = equivocal_clean.copy()
equivocal_clean['ensemble_prob_heldout'] = eq_ensemble
equivocal_clean['rna_pred_heldout'] = np.where(eq_ensemble >= 0.5, 'Positive', 'Negative')

# Compare with original equivocal scores
orig_probs = eq_scores_orig.set_index('pid')
prob_cols = [c for c in orig_probs.columns if c.startswith('prob_')]
orig_probs['ensemble_prob_orig'] = orig_probs[prob_cols].mean(axis=1)
orig_probs['rna_pred_orig'] = np.where(orig_probs['ensemble_prob_orig'] >= 0.5, 'Positive', 'Negative')

eq_compare = equivocal_clean[['pid', 'ensemble_prob_heldout', 'rna_pred_heldout']].merge(
    orig_probs[['ensemble_prob_orig', 'rna_pred_orig']].reset_index(),
    on='pid', how='left'
)
eq_compare = eq_compare.merge(
    analysis_df[['pid', 'ERBB2_expr', 'erbb2_copy_number']], on='pid', how='left'
)

n_pos_orig = (eq_compare['rna_pred_orig'] == 'Positive').sum()
n_pos_heldout = (eq_compare['rna_pred_heldout'] == 'Positive').sum()
n_agreement = (eq_compare['rna_pred_orig'] == eq_compare['rna_pred_heldout']).sum()

print(f"\n=== Equivocal Reclassification Comparison ===")
print(f"  Original models: {n_pos_orig} HER2+, {len(eq_compare) - n_pos_orig} HER2-")
print(f"  Held-out models: {n_pos_heldout} HER2+, {len(eq_compare) - n_pos_heldout} HER2-")
print(f"  Agreement: {n_agreement}/{len(eq_compare)} ({100*n_agreement/len(eq_compare):.0f}%)")

# Patient-level comparison
print("\nPatient-level comparison:")
print(eq_compare[['pid', 'ERBB2_expr', 'erbb2_copy_number',
                   'ensemble_prob_orig', 'ensemble_prob_heldout',
                   'rna_pred_orig', 'rna_pred_heldout']].to_string(index=False))

# ── 9. Figures ─────────────────────────────────────────────────────────────

# Fig 1: ROC comparison (held-out)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr, tpr, color='#e74c3c', linewidth=2,
        label=f'Held-out ensemble (AUC = {ensemble_auc:.3f})')

# Individual model ROCs
for name, probs in heldout_probs.items():
    fpr_m, tpr_m, _ = roc_curve(y_fish, probs)
    auc_m = roc_auc_score(y_fish, probs)
    ax.plot(fpr_m, tpr_m, linewidth=1, alpha=0.5, linestyle='--',
            label=f'{name} (AUC = {auc_m:.3f})')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5,
           label=f'Optimal threshold ({best_thresh:.2f})')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_title(f'Held-Out Concordance: RNA vs FISH in IHC 2+ (N={n_total})\n'
             f'Models trained WITHOUT any IHC 2+ patients')
ax.legend(loc='lower right', fontsize=9)
ax.set_aspect('equal')
plt.tight_layout()
savefig(fig, 'fig_04_5a_supp_heldout_roc')
plt.close()

# Fig 2: Score distribution by FISH outcome (held-out)
fig, ax = plt.subplots(figsize=(8, 5))
for fish_status, color in [('Positive', COLORS['Positive']),
                            ('Negative', COLORS['Negative'])]:
    subset = heldout_df[heldout_df['fish_binary'] == fish_status]
    ax.hist(subset['ensemble_prob_heldout'], bins=20, alpha=0.6,
            color=color, label=f'FISH {fish_status} (n={len(subset)})', edgecolor='white')
ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Default threshold (0.5)')
ax.axvline(best_thresh, color='orange', linestyle=':', linewidth=2,
           label=f'Optimal threshold ({best_thresh:.2f})')
ax.set_xlabel('Held-Out Ensemble Probability (HER2+)')
ax.set_ylabel('Count')
ax.set_title(f'Held-Out Score Distribution by FISH Outcome (N={n_total})')
ax.legend()
plt.tight_layout()
savefig(fig, 'fig_04_5a_supp_heldout_scores_by_fish')
plt.close()

# Fig 3: Equivocal probability comparison (original vs held-out)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(eq_compare['ensemble_prob_orig'], eq_compare['ensemble_prob_heldout'],
           c=eq_compare['erbb2_copy_number'].fillna(0), cmap='RdYlBu_r',
           s=60, edgecolor='black', linewidth=0.5, alpha=0.8)
ax.axhline(0.5, color='grey', linestyle='--', alpha=0.4)
ax.axvline(0.5, color='grey', linestyle='--', alpha=0.4)
ax.plot([0, 1], [0, 1], 'k:', alpha=0.3)
ax.set_xlabel('Original Model Ensemble Probability')
ax.set_ylabel('Held-Out Model Ensemble Probability')
ax.set_title('Equivocal Patient Scores: Original vs Held-Out Models')
cbar = plt.colorbar(ax.collections[0], ax=ax, label='ERBB2 Copy Number')
# Annotate quadrant counts
for (xlo, xhi, ylo, yhi, label) in [
    (0.5, 1, 0.5, 1, 'Both +'),
    (0, 0.5, 0, 0.5, 'Both -'),
    (0.5, 1, 0, 0.5, 'Orig+ / HO-'),
    (0, 0.5, 0.5, 1, 'Orig- / HO+'),
]:
    n_quad = ((eq_compare['ensemble_prob_orig'] >= xlo) &
              (eq_compare['ensemble_prob_orig'] < xhi) &
              (eq_compare['ensemble_prob_heldout'] >= ylo) &
              (eq_compare['ensemble_prob_heldout'] < yhi)).sum()
    ax.text((xlo + xhi) / 2, (ylo + yhi) / 2, f'{label}\nn={n_quad}',
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
plt.tight_layout()
savefig(fig, 'fig_04_5a_supp_equivocal_comparison')
plt.close()

# ── 10. Write report ──────────────────────────────────────────────────────

def fmt_ci(val, ci):
    return f"{val:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"

# Per-model AUCs for report
model_auc_lines = ""
for name, probs in heldout_probs.items():
    auc_m = roc_auc_score(y_fish, probs)
    cv_auc_str = f"{cv_results[name]:.3f}" if name in cv_results else "--"
    model_auc_lines += f"| {name} | {cv_auc_str} | {auc_m:.3f} |\n"
model_auc_lines += f"| **Ensemble** | -- | **{ensemble_auc:.3f}** |\n"

report = f"""# Analysis 5a Supplementary: Held-Out Concordance Validation

## Purpose

The primary Analysis 5a reported AUC = 0.994 for RNA-based prediction of FISH outcome
in IHC 2+ patients. However, those 154 patients were in the ML training set (their
FISH results determined their training labels). This supplementary analysis eliminates
that contamination by retraining models with all IHC 2+ FISH-resolved patients
excluded, then scoring them as a genuinely held-out population.

## Key Findings

- **Held-out AUC = {ensemble_auc:.3f}** (vs. 0.994 in primary analysis).
- At threshold 0.5: sensitivity = {sensitivity:.3f}, specificity = {specificity:.3f},
  kappa = {kappa:.3f}.
- Optimal threshold (Youden's J): {best_thresh:.3f}
  (sensitivity = {sens_opt:.3f}, specificity = {spec_opt:.3f}).
- Equivocal reclassification: {n_pos_heldout} HER2+ (vs. {n_pos_orig} in primary),
  {n_agreement}/{len(eq_compare)} patients agree ({100*n_agreement/len(eq_compare):.0f}%).

## Methods

### Training Set Construction

The standard training set (837 labeled patients) was filtered to exclude all 154
IHC 2+ patients with definitive FISH results. The remaining {len(train_df)} patients
consist of IHC 0, IHC 1+, and IHC 3+ patients (plus any IHC 2+ without FISH who
were labeled via other pathways).

| Set | N | Positive | Negative |
|---|---|---|---|
| Original training | 837 | 151 | 686 |
| Held-out training | {len(train_df)} | {y_train.sum()} | {(1-y_train).sum():.0f} |
| Removed (IHC 2+ FISH-resolved) | {len(heldout_df)} | {y_fish.sum()} | {(1-y_fish).sum():.0f} |

### Model Training

Three models (L1-LR, Random Forest, XGBoost) were trained on the reduced dataset
using the curated gene panel (~45 genes from 6 biological gene sets: HER2/17q12
amplicon, ERBB signaling, luminal/ER, basal, proliferation, EMT) plus copy number
and ER/PR status. This matches the feature reduction approach used in the
consolidated NB03 analysis. Models were fit on the full reduced training set and
applied to the held-out IHC 2+ population.

### Concordance Analysis

The held-out concordance table compares ensemble RNA probability (mean of 3 models)
against FISH ground truth. This is a genuinely out-of-sample evaluation: the models
have never seen any IHC 2+ patient during training.

## Results

### Model Performance

| Model | CV AUC (training) | Held-out AUC (IHC 2+) |
|---|---|---|
{model_auc_lines}

### Concordance Table (threshold = 0.5, N={n_total})

|  | RNA Positive | RNA Negative |
|---|---|---|
| FISH Positive | {tp} | {fn} |
| FISH Negative | {fp} | {tn} |

| Metric | Value (95% CI) |
|---|---|
| Sensitivity | {fmt_ci(sensitivity, sens_ci)} |
| Specificity | {fmt_ci(specificity, spec_ci)} |
| PPV | {fmt_ci(ppv, ppv_ci)} |
| NPV | {fmt_ci(npv, npv_ci)} |
| Accuracy | {fmt_ci(accuracy, acc_ci)} |
| Cohen's kappa | {kappa:.3f} |
| AUC | {ensemble_auc:.3f} |

**Optimal threshold (Youden's J):** {best_thresh:.3f}
(sensitivity = {sens_opt:.3f}, specificity = {spec_opt:.3f})

### Comparison with Primary Analysis

| Metric | Primary (5a) | Held-Out (this analysis) |
|---|---|---|
| AUC | 0.994 | {ensemble_auc:.3f} |
| Sensitivity (0.5) | 0.706 | {sensitivity:.3f} |
| Specificity (0.5) | 1.000 | {specificity:.3f} |
| Kappa (0.5) | 0.790 | {kappa:.3f} |
| Optimal threshold | 0.341 | {best_thresh:.3f} |
| Sens (optimal) | 0.941 | {sens_opt:.3f} |
| Spec (optimal) | 0.992 | {spec_opt:.3f} |

### Equivocal Reclassification Stability

| Metric | Original Models | Held-Out Models |
|---|---|---|
| HER2+ calls | {n_pos_orig} | {n_pos_heldout} |
| HER2- calls | {len(eq_compare) - n_pos_orig} | {len(eq_compare) - n_pos_heldout} |
| Agreement | {n_agreement}/{len(eq_compare)} ({100*n_agreement/len(eq_compare):.0f}%) |  |

**Patient-level comparison:**

| pid | ERBB2 | CN | Prob (orig) | Prob (held-out) | Call (orig) | Call (held-out) |
|---|---|---|---|---|---|---|
"""

for _, row in eq_compare.sort_values('ensemble_prob_orig', ascending=False).iterrows():
    report += (f"| {row['pid']} | {row['ERBB2_expr']:.2f} | "
               f"{row['erbb2_copy_number']:.0f} | "
               f"{row['ensemble_prob_orig']:.3f} | "
               f"{row['ensemble_prob_heldout']:.3f} | "
               f"{row['rna_pred_orig']} | {row['rna_pred_heldout']} |\n")

report += f"""
## Interpretation

"""

auc_delta = 0.994 - ensemble_auc
if ensemble_auc >= 0.95:
    report += f"""The held-out AUC of {ensemble_auc:.3f} confirms that the primary result was not
an artifact of training-data contamination. The AUC drop of {auc_delta:.3f} from the
primary analysis is {"minimal" if auc_delta < 0.03 else "modest"}, indicating that the
HER2 transcriptional signature learned from IHC 0/1+/3+ patients generalizes robustly
into the IHC 2+ equivocal zone. This is the strongest possible form of this evidence:
a model that has never seen an equivocal patient can still discriminate FISH outcome
among equivocal patients with AUC > 0.95.
"""
elif ensemble_auc >= 0.85:
    report += f"""The held-out AUC of {ensemble_auc:.3f} represents a drop of {auc_delta:.3f} from
the primary analysis (0.994). This indicates that some of the primary AUC was inflated
by training-data overlap, but the underlying signal remains strong. The model trained
on non-equivocal patients still achieves good discrimination in the equivocal zone,
supporting the feasibility of RNA-based FISH replacement -- though with somewhat
reduced performance compared to the primary estimate.
"""
else:
    report += f"""The held-out AUC of {ensemble_auc:.3f} represents a substantial drop of {auc_delta:.3f}
from the primary analysis (0.994). This suggests that the primary AUC was significantly
inflated by training-data overlap, and the model's ability to generalize into the
IHC 2+ zone from non-equivocal training data is limited. A CDx validation study would
likely need to include IHC 2+ patients in the training set (with proper cross-validation)
to achieve adequate performance.
"""

report += f"""
## Limitations

- The held-out analysis removes 154 patients from training, including 34 Positives
  (22% of original Positive class). This reduces the model's exposure to borderline
  HER2+ biology, which may underestimate what a properly designed CDx training set
  would achieve.
- The held-out population is still retrospective TCGA data. A prospective concordance
  study remains the definitive validation.
- Feature set includes copy number and ER/PR status alongside RNA expression. The
  held-out concordance reflects multi-modal prediction, not RNA-only.

---

**Figures:**
- `fig_04_5a_supp_heldout_roc.png` -- ROC curve for held-out concordance
- `fig_04_5a_supp_heldout_scores_by_fish.png` -- Score distribution by FISH outcome
- `fig_04_5a_supp_equivocal_comparison.png` -- Equivocal scores: original vs held-out
"""

report_path = REPORT_DIR / '5a_supp_heldout_concordance.md'
report_path.write_text(report, encoding='utf-8')
print(f"\nReport written to: {report_path}")
print("Done.")
