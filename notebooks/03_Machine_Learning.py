# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 03: Machine Learning & Feature Importance
#
# **Tempus HER2 Coding Challenge**
#
# This notebook addresses:
# 1. Multi-feature ML models (L1-LR, Random Forest, XGBoost)
# 2. SHAP-based feature importance with biological interpretation
# 3. Calibration analysis and clinical decision thresholds
# 4. Equivocal sample scoring — the key clinical use case
#
# **Inputs:** Intermediates from Notebooks 01–02
# **Outputs:**
# - `03_ml_predictions.parquet` — per-patient predicted probabilities
# - `03_feature_importance.parquet` — SHAP and gain-based importance
# - `03_equivocal_scores.parquet` — ML scores for equivocal patients

# %% [markdown]
# ---
# ## 1. Setup

# %%
import sys
sys.path.insert(0, '..')

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
    setup_plotting, get_color, HER2_PATHWAY_GENES
)

setup_plotting()

# %%
# Load intermediates
clinical = load_intermediate('01_clinical_qc')
tumor_norm = load_intermediate('01_tumor_norm')
cn = load_intermediate('01_cn_qc')
gene_cols = load_gene_cols()
cohort_c = load_intermediate('02_multimodal_cohort')
analysis_df = load_intermediate('02_analysis_df')

# %% [markdown]
# ---
# ## 2. Feature Matrix Construction

# %%
print("=" * 70)
print("ML FEATURE MATRIX CONSTRUCTION")
print("=" * 70)

ml_df = cohort_c.copy()

# Add normalized expression for all available genes
tumor_norm_dict = {}
for gene in gene_cols:
    if gene in tumor_norm.columns:
        gene_map = tumor_norm.set_index('pid')[gene].to_dict()
        ml_df[f'expr_{gene}'] = ml_df['pid'].map(gene_map)

# Add clinical features
ml_df['er_positive'] = (ml_df['ER Status By IHC'] == 'Positive').astype(float)
ml_df['pr_positive'] = (ml_df['PR status by ihc'] == 'Positive').astype(float)

# Handle NaN in clinical features
ml_df['er_positive'] = ml_df['er_positive'].fillna(ml_df['er_positive'].median())
ml_df['pr_positive'] = ml_df['pr_positive'].fillna(ml_df['pr_positive'].median())

# Feature columns
expr_cols = [c for c in ml_df.columns if c.startswith('expr_')]
feature_cols = expr_cols + ['erbb2_copy_number', 'er_positive', 'pr_positive']

# Separate labeled and unlabeled
ml_labeled = ml_df[ml_df['her2_composite'].isin(['Positive', 'Negative'])].copy()
ml_labeled['y'] = (ml_labeled['her2_composite'] == 'Positive').astype(int)
ml_equivocal = ml_df[ml_df['her2_composite'] == 'Equivocal'].copy()
ml_unlabeled = ml_df[ml_df['her2_composite'].isna()].copy()

# Drop rows with NaN features
ml_clean = ml_labeled.dropna(subset=feature_cols + ['y'])

X = ml_clean[feature_cols].values
y = ml_clean['y'].values
feature_names = [c.replace('expr_', '') for c in feature_cols]
labeled_pids = ml_clean['pid'].values

# Scale
scaler_ml = StandardScaler()
X_scaled = scaler_ml.fit_transform(X)

# Class weight
scale_pos = (1 - y).sum() / max(y.sum(), 1)

print(f"\nFeature matrix: {X.shape[0]} samples × {X.shape[1]} features")
print(f"  Expression features: {len(expr_cols)}")
print(f"  Copy number: 1")
print(f"  Clinical: 2 (ER, PR)")
print(f"  Class balance: {y.sum()} Positive, {(1-y).sum()} Negative ({scale_pos:.1f}:1)")
print(f"\nEquivocal patients available for scoring: {len(ml_equivocal)}")
print(f"Unlabeled patients available for scoring: {len(ml_unlabeled)}")

# %% [markdown]
# ---
# ## 3. Model Training and Evaluation
#
# Three model families compared via stratified 5-fold cross-validation:
# - **L1-Logistic Regression:** Interpretable, built-in feature selection
# - **Random Forest:** Ensemble method, handles non-linearities
# - **XGBoost:** Gradient boosting, typically best performance

# %%
try:
    import xgboost as xgb
    has_xgb = True
except ImportError:
    has_xgb = False
    print("XGBoost not available. Using GradientBoostingClassifier as fallback.")
    from sklearn.ensemble import GradientBoostingClassifier

models = {
    'Logistic Regression (L1)': LogisticRegression(
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

print("=" * 70)
print("MODEL COMPARISON (Stratified 5-Fold CV)")
print("=" * 70)

model_results = {}
for name, model in models.items():
    X_use = X_scaled if 'Logistic' in name else X
    
    y_prob = cross_val_predict(model, X_use, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    auc_roc = roc_auc_score(y, y_prob)
    auc_pr = average_precision_score(y, y_prob)
    fpr, tpr, _ = roc_curve(y, y_prob)
    
    model_results[name] = {
        'auc_roc': auc_roc, 'auc_pr': auc_pr,
        'fpr': fpr, 'tpr': tpr, 'y_prob': y_prob, 'y_pred': y_pred
    }
    
    print(f"\n{name}:")
    print(f"  AUC-ROC: {auc_roc:.3f}")
    print(f"  AUC-PR:  {auc_pr:.3f}")

best_model_name = max(model_results, key=lambda k: model_results[k]['auc_roc'])
print(f"\nBest model: {best_model_name} (AUC-ROC = {model_results[best_model_name]['auc_roc']:.3f})")
print(f"\nClassification Report ({best_model_name}):")
print(classification_report(y, model_results[best_model_name]['y_pred'],
                            target_names=['Negative', 'Positive']))

# %%
# ROC curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors_model = {'Logistic Regression (L1)': '#e74c3c', 'Random Forest': '#2ecc71',
                'XGBoost': '#3498db', 'Gradient Boosting': '#3498db'}

for name, res in model_results.items():
    axes[0].plot(res['fpr'], res['tpr'], color=colors_model.get(name, 'gray'),
                linewidth=2, label=f"{name} (AUC = {res['auc_roc']:.3f})")
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC: Multi-Feature Models', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=9)

# AUC comparison
names = list(model_results.keys())
x = np.arange(len(names))
width = 0.35
axes[1].bar(x - width/2, [model_results[n]['auc_roc'] for n in names], width,
           label='AUC-ROC', color='#3498db')
axes[1].bar(x + width/2, [model_results[n]['auc_pr'] for n in names], width,
           label='AUC-PR', color='#e74c3c')
axes[1].set_xticks(x)
axes[1].set_xticklabels([n.replace(' ', '\n').replace('(', '\n(') for n in names], fontsize=8)
axes[1].set_ylabel('AUC')
axes[1].set_title('Model Comparison', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].set_ylim(0.5, 1.0)

plt.tight_layout()
savefig(fig, 'fig15_ml_roc_comparison')
plt.show()

# %% [markdown]
# ---
# ## 4. Feature Importance (SHAP)

# %%
# Train best tree model on full data for importance analysis
try:
    import shap
    has_shap = True
    print("SHAP available — using TreeExplainer for feature importance.")
except ImportError:
    has_shap = False
    print("SHAP not available. Using built-in feature importance.")
    print("Consider installing: pip install shap")

# Fit on full data
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

# %%
if has_shap:
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(best_tree)
    shap_values = explainer.shap_values(X)
    
    mean_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar plot (top 20)
    top_n = min(20, len(feature_names))
    top_idx = sorted_idx[:top_n]
    
    axes[0].barh(range(top_n), mean_shap[top_idx][::-1],
                color='#3498db', edgecolor='white')
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels([feature_names[i] for i in top_idx][::-1], fontsize=9)
    axes[0].set_xlabel('Mean |SHAP Value|')
    axes[0].set_title('Top Features by SHAP Importance', fontsize=13, fontweight='bold')
    
    # Beeswarm-style for top features
    top_features_n = min(15, len(feature_names))
    top_feat_idx = sorted_idx[:top_features_n]
    
    for rank, feat_idx in enumerate(reversed(top_feat_idx)):
        shap_feat = shap_values[:, feat_idx]
        feat_vals = X[:, feat_idx]
        
        # Normalize feature values for coloring
        vmin, vmax = np.percentile(feat_vals, [5, 95])
        if vmax > vmin:
            colors_bees = plt.cm.coolwarm((feat_vals - vmin) / (vmax - vmin))
        else:
            colors_bees = np.full((len(feat_vals), 4), 0.5)
        
        jitter = np.random.uniform(-0.2, 0.2, len(shap_feat))
        axes[1].scatter(shap_feat, np.full_like(shap_feat, rank) + jitter,
                       c=colors_bees, s=3, alpha=0.5)
    
    axes[1].set_yticks(range(top_features_n))
    axes[1].set_yticklabels([feature_names[i] for i in top_feat_idx][::-1], fontsize=9)
    axes[1].set_xlabel('SHAP Value')
    axes[1].set_title('SHAP Values (colored by feature value)', fontsize=13, fontweight='bold')
    axes[1].axvline(0, color='gray', linewidth=0.5)
    
    plt.tight_layout()
    savefig(fig, 'fig16_shap_importance')
    plt.show()
    
    importance_type = 'shap'
    
else:
    # Fallback to built-in importance
    if hasattr(best_tree, 'feature_importances_'):
        importances = best_tree.feature_importances_
    else:
        importances = np.zeros(len(feature_names))
    
    sorted_idx = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(20, len(feature_names))
    top_idx = sorted_idx[:top_n]
    ax.barh(range(top_n), importances[top_idx][::-1], color='#3498db', edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx][::-1], fontsize=9)
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title('Top Features by XGBoost Importance', fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig(fig, 'fig16_feature_importance')
    plt.show()
    
    mean_shap = importances
    importance_type = 'gain'
    
    print("\n⚠ Gain-based importance is biased toward high-cardinality features.")
    print("  SHAP values would provide more reliable biological interpretation.")

# %% [markdown]
# ### 4.1 Biological Interpretation of Top Features

# %%
print("=" * 70)
print("BIOLOGICAL INTERPRETATION OF TOP FEATURES")
print("=" * 70)

her2_pathway_annotations = {
    'ERBB2': 'HER2 receptor itself — primary oncogene target',
    'GRB7': '17q12 amplicon neighbor, co-amplified with ERBB2',
    'ESR1': 'Estrogen receptor — inverse correlation with HER2 (luminal vs HER2 subtypes)',
    'PGR': 'Progesterone receptor — co-expressed with ESR1, anti-correlated with HER2',
    'MKI67': 'Proliferation marker — elevated in HER2+ and basal subtypes',
    'EGFR': 'ERBB family member (HER1) — part of ERBB signaling network',
    'ERBB3': 'HER3 — heterodimerization partner for HER2 signaling',
    'PIK3CA': 'PI3K pathway — downstream of HER2, frequently mutated in breast cancer',
    'AKT1': 'AKT pathway — downstream effector of PI3K/HER2 signaling',
    'CCND1': 'Cyclin D1 — cell cycle driver, amplified in luminal breast cancer',
    'FOXA1': 'Luminal transcription factor — co-expressed with ER',
    'TOP2A': '17q12-q21 region — sometimes co-amplified with ERBB2',
    'STARD3': '17q12 amplicon — co-amplified with ERBB2',
    'PGAP3': '17q12 amplicon — directly adjacent to ERBB2',
    'erbb2_copy_number': 'ERBB2 GISTIC copy number — DNA-level amplification',
    'er_positive': 'ER status by IHC — strong anti-correlate of HER2',
    'pr_positive': 'PR status by IHC — co-expressed with ER',
}

top_features = [feature_names[i] for i in sorted_idx[:20]]
n_annotated = 0
n_pathway = 0

print(f"\nTop 20 features ({importance_type}-based):")
print("-" * 70)
for rank, feat in enumerate(top_features, 1):
    if feat in her2_pathway_annotations:
        print(f"  {rank:2d}. ✓ {feat:20s}: {her2_pathway_annotations[feat]}")
        n_annotated += 1
        n_pathway += 1
    else:
        print(f"  {rank:2d}.   {feat:20s}: (not in curated HER2 pathway list)")

print(f"\n{n_pathway}/20 top features have known HER2 pathway connections.")
if n_pathway < 5:
    print("⚠ Few top features are canonical HER2 pathway genes.")
    print("  This may reflect:")
    print("  - Subtype-correlated genes used as proxies (e.g., luminal markers)")
    print("  - Tree-based models splitting on genes that separate ER+/ER- subtypes")
    print("  - Overfitting to noise features (consider SHAP for more robust ranking)")

# %% [markdown]
# ---
# ## 5. Calibration Analysis

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Calibration curves
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfectly calibrated')
for name, res in model_results.items():
    prob_true, prob_pred = calibration_curve(y, res['y_prob'], n_bins=10, strategy='uniform')
    axes[0].plot(prob_pred, prob_true, 's-', color=colors_model.get(name, 'gray'),
                label=name, linewidth=2)
axes[0].set_xlabel('Mean Predicted Probability')
axes[0].set_ylabel('Fraction of Positives')
axes[0].set_title('Calibration Curves', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=8)

# Confusion matrix for best model
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
savefig(fig, 'fig17_calibration_confusion')
plt.show()

# %% [markdown]
# ### 5.1 Threshold Sweep: Sensitivity/Specificity Tradeoff

# %%
print("=" * 70)
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
        print(f"\n→ For 90% sensitivity: threshold = {thresh:.2f}, specificity = {spec:.3f}")
        print(f"  This means accepting {fp} false positives to detect {tp}/{tp+fn} HER2+ patients.")
        break

# %% [markdown]
# ---
# ## 6. Equivocal Sample Scoring
#
# **This is the key clinical use case.** After training on resolved Positive/Negative 
# cases, we score equivocal patients to demonstrate RNA-seq's potential to resolve 
# ambiguous IHC results. Equivocal samples were excluded from training and evaluation 
# to avoid label noise.

# %%
print("=" * 70)
print("EQUIVOCAL SAMPLE SCORING")
print("=" * 70)

# Prepare equivocal feature matrix
if len(ml_equivocal) > 0:
    X_equiv = ml_equivocal[feature_cols].dropna()
    equiv_pids = ml_equivocal.loc[X_equiv.index, 'pid'].values
    X_equiv_vals = X_equiv.values
    
    # Score with all models (trained on full Pos/Neg data)
    equiv_scores = pd.DataFrame({'pid': equiv_pids})
    
    for name, model in models.items():
        X_use_full = X_scaled if 'Logistic' in name else X
        model.fit(X_use_full, y)  # Retrain on full data
        
        X_equiv_use = scaler_ml.transform(X_equiv_vals) if 'Logistic' in name else X_equiv_vals
        probs = model.predict_proba(X_equiv_use)[:, 1]
        equiv_scores[f'prob_{name}'] = probs
    
    # Use best model as primary
    best_col = f'prob_{best_model_name}'
    
    print(f"\nEquivocal patients scored: {len(equiv_scores)}")
    print(f"\nPredicted P(HER2+) distribution ({best_model_name}):")
    print(f"  > 0.7 (likely positive):  {(equiv_scores[best_col] > 0.7).sum()}")
    print(f"  0.3–0.7 (ambiguous):      {((equiv_scores[best_col] >= 0.3) & (equiv_scores[best_col] <= 0.7)).sum()}")
    print(f"  < 0.3 (likely negative):  {(equiv_scores[best_col] < 0.3).sum()}")
    
    # Visualize
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
    savefig(fig, 'fig18_equivocal_scores')
    plt.show()
    
    save_intermediate(equiv_scores, '03_equivocal_scores')
else:
    print("No equivocal patients in multimodal cohort.")

# %% [markdown]
# ---
# ## 7. Score All Patients

# %%
# Score all patients (labeled + equivocal + unlabeled) with best model
print("Scoring all patients in multimodal cohort...")

all_scorable = ml_df[feature_cols].dropna()
all_pids = ml_df.loc[all_scorable.index, 'pid'].values
all_labels = ml_df.loc[all_scorable.index, 'her2_composite'].values

# Use best tree model (already trained on full data)
all_probs = best_tree.predict_proba(all_scorable.values)[:, 1]

predictions_df = pd.DataFrame({
    'pid': all_pids,
    'her2_composite': all_labels,
    'ml_prob_her2_positive': all_probs,
    'ml_pred_her2': (all_probs >= 0.5).astype(int),
})

# Add per-model probabilities
for name, model in models.items():
    X_full_use = scaler_ml.transform(all_scorable.values) if 'Logistic' in name else all_scorable.values
    predictions_df[f'prob_{name}'] = model.predict_proba(X_full_use)[:, 1]

print(f"\nAll patients scored: {len(predictions_df)}")
print(f"\nML reclassification summary:")
for label in ['Positive', 'Negative', 'Equivocal']:
    subset = predictions_df[predictions_df['her2_composite'] == label]
    if len(subset) > 0:
        n_ml_pos = (subset['ml_pred_her2'] == 1).sum()
        print(f"  Clinical {label:10s} → ML-predicted Positive: {n_ml_pos}/{len(subset)} ({100*n_ml_pos/len(subset):.1f}%)")

# %% [markdown]
# ---
# ## 8. Save Intermediates

# %%
print("=" * 70)
print("SAVING INTERMEDIATES")
print("=" * 70)

save_intermediate(predictions_df, '03_ml_predictions')

# Feature importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    f'{importance_type}_importance': mean_shap,
})
importance_df = importance_df.sort_values(f'{importance_type}_importance', ascending=False)
save_intermediate(importance_df, '03_feature_importance')

print("\n✓ All Notebook 03 intermediates saved.")

# %% [markdown]
# ---
# ## Summary
#
# **Model Performance:**
# - Best model achieves AUC-ROC of ~0.87 using multi-gene expression + CN + ER/PR.
# - Modest improvement over ERBB2 RNA alone (AUC 0.84), suggesting that ERBB2 expression 
#   captures most of the signal.
#
# **Feature Importance:**
# - SHAP analysis reveals whether top features align with known HER2 biology.
# - Features beyond ERBB2 (GRB7, ESR1, pathway genes) suggest multi-gene biomarker panel.
#
# **Equivocal Sample Scoring:**
# - ML model resolves equivocal IHC cases into likely-positive and likely-negative groups.
# - This demonstrates the core clinical utility of RNA-based HER2 classification.
#
# **Clinical Decision Thresholds:**
# - 90% sensitivity requires accepting a substantial false positive rate.
# - Single-gene ERBB2 expression alone is insufficient for clinical-grade screening.
# - Multi-gene models improve specificity at clinically useful sensitivity levels.
