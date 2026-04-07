"""
03_exploratory_binary_ml.py
===========================
Exploratory binary HER2 ML analysis extracted from Notebook 03a.

Contains:
- Full calibration analysis (calibration curves, reliability diagrams)
- Threshold sweep with sensitivity/specificity tradeoff table
- Detailed equivocal sample scoring (all models)
- All-patient scoring with reclassification summary
- Biological interpretation of top features (annotated)

This script is self-contained: it loads intermediates from outputs/ and can be
run independently. The consolidated notebook (03_ML_and_Discordant_Biology.ipynb)
uses a condensed version of this analysis; this script preserves the full detail.

Outputs:
- outputs/03_ml_predictions.parquet
- outputs/03_feature_importance.parquet
- outputs/03_equivocal_scores.parquet
- outputs/figures/fig_03s_calibration_confusion.png
- outputs/figures/fig_03s_threshold_sweep.png
- outputs/figures/fig_03s_equivocal_scores.png
"""

import sys
import os

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

from src.utils import (
    load_intermediate, load_gene_cols, save_intermediate, savefig,
    setup_plotting, HER2_PATHWAY_GENES
)

try:
    import xgboost as xgb
    has_xgb = True
except ImportError:
    has_xgb = False
    from sklearn.ensemble import GradientBoostingClassifier

try:
    import shap
    has_shap = True
except ImportError:
    has_shap = False

setup_plotting()


# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("=" * 70)
print("EXPLORATORY BINARY HER2 ML")
print("=" * 70)

clinical = load_intermediate('01_clinical_qc')
tumor_norm = load_intermediate('01_tumor_norm_tmm_tss')
cn = load_intermediate('01_cn_qc')
gene_cols = load_gene_cols()
cohort_c = load_intermediate('02_multimodal_cohort')


# ============================================================================
# 2. FEATURE MATRIX -- FULL TRANSCRIPTOME + CN + CLINICAL
# ============================================================================
# NOTE: This script uses a broad feature set for exploratory purposes.
# The consolidated notebook uses a curated gene panel per the feature
# reduction constraint.

ml_df = cohort_c.copy()

tumor_norm_dict = {}
for gene in gene_cols:
    if gene in tumor_norm.columns:
        gene_map = tumor_norm.set_index('pid')[gene].to_dict()
        ml_df[f'expr_{gene}'] = ml_df['pid'].map(gene_map)

ml_df['er_positive'] = (ml_df['ER Status By IHC'] == 'Positive').astype(float)
ml_df['pr_positive'] = (ml_df['PR status by ihc'] == 'Positive').astype(float)
ml_df['er_positive'] = ml_df['er_positive'].fillna(ml_df['er_positive'].median())
ml_df['pr_positive'] = ml_df['pr_positive'].fillna(ml_df['pr_positive'].median())

expr_cols = [c for c in ml_df.columns if c.startswith('expr_')]
feature_cols = expr_cols + ['erbb2_copy_number', 'er_positive', 'pr_positive']

ml_labeled = ml_df[ml_df['her2_composite'].isin(['Positive', 'Negative'])].copy()
ml_labeled['y'] = (ml_labeled['her2_composite'] == 'Positive').astype(int)
ml_equivocal = ml_df[ml_df['her2_composite'] == 'Equivocal'].copy()

ml_clean = ml_labeled.dropna(subset=feature_cols + ['y'])
X = ml_clean[feature_cols].values
y = ml_clean['y'].values
feature_names = [c.replace('expr_', '') for c in feature_cols]
labeled_pids = ml_clean['pid'].values

scaler_ml = StandardScaler()
X_scaled = scaler_ml.fit_transform(X)
scale_pos = (1 - y).sum() / max(y.sum(), 1)

print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
print(f"  Expression features: {len(expr_cols)}")
print(f"  Class balance: {y.sum()} Positive, {(1-y).sum():.0f} Negative ({scale_pos:.1f}:1)")
print(f"  Equivocal patients: {len(ml_equivocal)}")


# ============================================================================
# 3. MODEL TRAINING (3 MODELS, 5-FOLD CV)
# ============================================================================
models = {
    'L1-LR': LogisticRegression(
        penalty='l1', solver='saga', max_iter=2000, random_state=42,
        class_weight='balanced', C=1.0
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 70)
print("MODEL COMPARISON (Stratified 5-Fold CV)")
print("=" * 70)

model_results = {}
for name, model in models.items():
    X_use = X_scaled if 'LR' in name or 'Logistic' in name else X
    y_prob = cross_val_predict(model, X_use, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc_roc = roc_auc_score(y, y_prob)
    auc_pr = average_precision_score(y, y_prob)
    fpr, tpr, _ = roc_curve(y, y_prob)
    model_results[name] = {
        'auc_roc': auc_roc, 'auc_pr': auc_pr,
        'fpr': fpr, 'tpr': tpr, 'y_prob': y_prob, 'y_pred': y_pred
    }
    print(f"  {name}: AUC-ROC={auc_roc:.3f}, AUC-PR={auc_pr:.3f}")

best_model_name = max(model_results, key=lambda k: model_results[k]['auc_roc'])
print(f"\nBest model: {best_model_name} (AUC-ROC = {model_results[best_model_name]['auc_roc']:.3f})")


# ============================================================================
# 4. FEATURE IMPORTANCE (SHAP OR GAIN)
# ============================================================================
if has_xgb:
    best_tree = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos, random_state=42,
        eval_metric='logloss', use_label_encoder=False
    )
else:
    best_tree = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
best_tree.fit(X, y)

if has_shap:
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(best_tree)
    shap_values = explainer.shap_values(X)
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_type = 'shap'
else:
    mean_shap = best_tree.feature_importances_ if hasattr(best_tree, 'feature_importances_') else np.zeros(len(feature_names))
    importance_type = 'gain'

sorted_idx = np.argsort(mean_shap)[::-1]


# ============================================================================
# 5. BIOLOGICAL INTERPRETATION OF TOP FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("BIOLOGICAL INTERPRETATION OF TOP FEATURES")
print("=" * 70)

her2_pathway_annotations = {
    'ERBB2': 'HER2 receptor itself -- primary oncogene target',
    'GRB7': '17q12 amplicon neighbor, co-amplified with ERBB2',
    'ESR1': 'Estrogen receptor -- inverse correlation with HER2',
    'PGR': 'Progesterone receptor -- co-expressed with ESR1',
    'MKI67': 'Proliferation marker -- elevated in HER2+ and basal',
    'EGFR': 'ERBB family member (HER1)',
    'ERBB3': 'HER3 -- heterodimerization partner for HER2',
    'PIK3CA': 'PI3K pathway -- downstream of HER2',
    'AKT1': 'AKT pathway -- downstream effector of PI3K/HER2',
    'CCND1': 'Cyclin D1 -- cell cycle driver',
    'FOXA1': 'Luminal transcription factor',
    'TOP2A': '17q12-q21 region -- sometimes co-amplified with ERBB2',
    'STARD3': '17q12 amplicon -- co-amplified with ERBB2',
    'PGAP3': '17q12 amplicon -- directly adjacent to ERBB2',
    'erbb2_copy_number': 'ERBB2 GISTIC copy number',
    'er_positive': 'ER status by IHC',
    'pr_positive': 'PR status by IHC',
}

top_features = [feature_names[i] for i in sorted_idx[:20]]
n_pathway = 0
for rank, feat in enumerate(top_features, 1):
    annotation = her2_pathway_annotations.get(feat, '(not in curated HER2 pathway list)')
    marker = 'Y' if feat in her2_pathway_annotations else ' '
    print(f"  {rank:2d}. [{marker}] {feat:20s}: {annotation}")
    if feat in her2_pathway_annotations:
        n_pathway += 1

print(f"\n{n_pathway}/20 top features have known HER2 pathway connections.")


# ============================================================================
# 6. CALIBRATION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("CALIBRATION ANALYSIS")
print("=" * 70)

colors_model = {'L1-LR': '#e74c3c', 'Random Forest': '#2ecc71',
                'XGBoost': '#3498db', 'Gradient Boosting': '#3498db'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfectly calibrated')
for name, res in model_results.items():
    prob_true, prob_pred = calibration_curve(y, res['y_prob'], n_bins=10, strategy='uniform')
    axes[0].plot(prob_pred, prob_true, 's-', color=colors_model.get(name, 'gray'),
                label=name, linewidth=2)
axes[0].set_xlabel('Mean Predicted Probability')
axes[0].set_ylabel('Fraction of Positives')
axes[0].set_title('Calibration Curves', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=8)

y_prob_best = model_results[best_model_name]['y_prob']
y_pred_best = model_results[best_model_name]['y_pred']
cm = confusion_matrix(y, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title(f'Confusion Matrix ({best_model_name}, threshold=0.5)',
                 fontsize=12, fontweight='bold')

plt.tight_layout()
savefig(fig, 'fig_03s_calibration_confusion')
plt.close()


# ============================================================================
# 7. THRESHOLD SWEEP
# ============================================================================
print("\n" + "=" * 70)
print("THRESHOLD SWEEP: Sensitivity/Specificity Tradeoff")
print("=" * 70)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"\n{'Threshold':>10s}  {'Sensitivity':>12s}  {'Specificity':>12s}  {'PPV':>8s}  {'NPV':>8s}  {'FP':>5s}  {'FN':>5s}")
print("-" * 75)

for thresh in thresholds:
    y_pred_t = (y_prob_best >= thresh).astype(int)
    tp = ((y_pred_t == 1) & (y == 1)).sum()
    tn = ((y_pred_t == 0) & (y == 0)).sum()
    fp = ((y_pred_t == 1) & (y == 0)).sum()
    fn = ((y_pred_t == 0) & (y == 1)).sum()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    print(f"  {thresh:>8.2f}  {sens:>12.3f}  {spec:>12.3f}  {ppv:>8.3f}  {npv:>8.3f}  {fp:>5d}  {fn:>5d}")

# Find threshold for 90% sensitivity
for thresh in np.arange(0.01, 1.0, 0.01):
    y_pred_t = (y_prob_best >= thresh).astype(int)
    tp = ((y_pred_t == 1) & (y == 1)).sum()
    fn = ((y_pred_t == 0) & (y == 1)).sum()
    sens = tp / max(tp + fn, 1)
    if sens >= 0.90:
        fp = ((y_pred_t == 1) & (y == 0)).sum()
        tn = ((y_pred_t == 0) & (y == 0)).sum()
        spec = tn / max(tn + fp, 1)
        print(f"\n-> For 90% sensitivity: threshold = {thresh:.2f}, specificity = {spec:.3f}")
        print(f"  Accepting {fp} false positives to detect {tp}/{tp+fn} HER2+ patients.")
        break


# ============================================================================
# 8. EQUIVOCAL SAMPLE SCORING (DETAILED)
# ============================================================================
print("\n" + "=" * 70)
print("EQUIVOCAL SAMPLE SCORING (DETAILED)")
print("=" * 70)

if len(ml_equivocal) > 0:
    X_equiv = ml_equivocal[feature_cols].dropna()
    equiv_pids = ml_equivocal.loc[X_equiv.index, 'pid'].values
    X_equiv_vals = X_equiv.values

    equiv_scores = pd.DataFrame({'pid': equiv_pids})
    for name, model in models.items():
        X_use_full = X_scaled if 'LR' in name or 'Logistic' in name else X
        model.fit(X_use_full, y)
        X_equiv_use = scaler_ml.transform(X_equiv_vals) if ('LR' in name or 'Logistic' in name) else X_equiv_vals
        probs = model.predict_proba(X_equiv_use)[:, 1]
        equiv_scores[f'prob_{name}'] = probs

    best_col = f'prob_{best_model_name}'
    print(f"Equivocal patients scored: {len(equiv_scores)}")
    print(f"\nPredicted P(HER2+) distribution ({best_model_name}):")
    print(f"  > 0.7 (likely positive):  {(equiv_scores[best_col] > 0.7).sum()}")
    print(f"  0.3-0.7 (ambiguous):      {((equiv_scores[best_col] >= 0.3) & (equiv_scores[best_col] <= 0.7)).sum()}")
    print(f"  < 0.3 (likely negative):  {(equiv_scores[best_col] < 0.3).sum()}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(equiv_scores[best_col], bins=15, color='#f39c12', edgecolor='white', alpha=0.8)
    ax.axvline(0.5, color='red', linestyle='--', label='Decision threshold (0.5)')
    ax.axvline(0.3, color='gray', linestyle=':', alpha=0.5, label='Low confidence zone')
    ax.axvline(0.7, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Predicted P(HER2+)')
    ax.set_ylabel('Count')
    ax.set_title(f'ML-Predicted HER2 Probability for Equivocal Patients (n={len(equiv_scores)})',
                fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    savefig(fig, 'fig_03s_equivocal_scores')
    plt.close()

    save_intermediate(equiv_scores, '03_equivocal_scores')
else:
    print("No equivocal patients in multimodal cohort.")


# ============================================================================
# 9. SCORE ALL PATIENTS
# ============================================================================
print("\n" + "=" * 70)
print("SCORING ALL PATIENTS")
print("=" * 70)

all_scorable = ml_df[feature_cols].dropna()
all_pids = ml_df.loc[all_scorable.index, 'pid'].values
all_labels = ml_df.loc[all_scorable.index, 'her2_composite'].values
all_probs = best_tree.predict_proba(all_scorable.values)[:, 1]

predictions_df = pd.DataFrame({
    'pid': all_pids,
    'her2_composite': all_labels,
    'ml_prob_her2_positive': all_probs,
    'ml_pred_her2': (all_probs >= 0.5).astype(int),
})

for name, model in models.items():
    X_full_use = scaler_ml.transform(all_scorable.values) if ('LR' in name or 'Logistic' in name) else all_scorable.values
    predictions_df[f'prob_{name}'] = model.predict_proba(X_full_use)[:, 1]

print(f"All patients scored: {len(predictions_df)}")
print(f"\nML reclassification summary:")
for label in ['Positive', 'Negative', 'Equivocal']:
    subset = predictions_df[predictions_df['her2_composite'] == label]
    if len(subset) > 0:
        n_ml_pos = (subset['ml_pred_her2'] == 1).sum()
        print(f"  Clinical {label:10s} -> ML Positive: {n_ml_pos}/{len(subset)} ({100*n_ml_pos/len(subset):.1f}%)")

save_intermediate(predictions_df, '03_ml_predictions')

# Feature importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    f'{importance_type}_importance': mean_shap,
})
importance_df = importance_df.sort_values(f'{importance_type}_importance', ascending=False)
save_intermediate(importance_df, '03_feature_importance')

print("\nAll exploratory binary ML outputs saved.")
