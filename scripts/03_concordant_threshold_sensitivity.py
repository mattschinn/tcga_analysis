"""
03_concordant_threshold_sensitivity.py
======================================
Sensitivity analysis for concordant group threshold definitions.

The concordant-only model (NB03 Section 4) defines "concordant" patients using
three thresholds: ERBB2 expression floor for positives, CN floor for positives,
and ERBB2 expression ceiling for negatives. This script tests three named
configurations to assess whether the CN-stratified discordant biology finding
is robust to reasonable threshold variation.

Configurations:
  Strict   (current default): pos ERBB2 >= 25th pctl, CN >= 1, neg ERBB2 <= 75th pctl
  Relaxed:                     pos ERBB2 >= 10th pctl, CN >= 0, neg ERBB2 <= 90th pctl
  Stringent:                   pos ERBB2 >= 40th pctl, CN >= 2, neg ERBB2 <= 50th pctl

Key metric: stability of CN-stratified group assignment for ~35 discordant
patients across threshold configurations.

Outputs:
  outputs/03_threshold_sensitivity.parquet
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from src.utils import load_intermediate, load_gene_cols, save_intermediate

# ============================================================================
# DATA LOADING
# ============================================================================
print("=" * 70)
print("CONCORDANT THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 70)

cohort_c = load_intermediate("02_multimodal_cohort")
tumor_norm = load_intermediate("01_tumor_norm_tmm_tss")
gene_cols = load_gene_cols()
discordant_df = load_intermediate("02_discordant_cases")

tumor_expr = tumor_norm.set_index("pid")

# Curated gene sets (same as NB03)
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
curated_genes = [g for g in curated_genes if g in gene_cols]

# ============================================================================
# BUILD ML DATA FRAME (same as NB03 cell 6)
# ============================================================================
ml_df = cohort_c.copy()
for gene in curated_genes:
    col = f"expr_{gene}"
    if gene in tumor_expr.columns:
        ml_df[col] = ml_df["pid"].map(tumor_expr[gene])

ml_df["er_positive"] = (ml_df["ER Status By IHC"] == "Positive").astype(float)
ml_df["pr_positive"] = (ml_df["PR status by ihc"] == "Positive").astype(float)
ml_df["er_positive"] = ml_df["er_positive"].fillna(ml_df["er_positive"].median())
ml_df["pr_positive"] = ml_df["pr_positive"].fillna(ml_df["pr_positive"].median())

fs_cols = [f"expr_{g}" for g in curated_genes if f"expr_{g}" in ml_df.columns]
fs_cols += ["erbb2_copy_number", "er_positive", "pr_positive"]

ml_labeled = ml_df[ml_df["her2_composite"].isin(["Positive", "Negative"])].copy()
ml_labeled["y"] = (ml_labeled["her2_composite"] == "Positive").astype(int)
ml_clean = ml_labeled.dropna(subset=fs_cols + ["y"])

# Discordant patient IDs
disc_rna_high = discordant_df[discordant_df["discordance_type"] == "IHC-/RNA-high"]
disc_pids = set(disc_rna_high["pid"].unique())

# Pre-computed full-data model predictions (for consensus score)
try:
    predictions_full = load_intermediate("03_ml_predictions")
    has_full_preds = True
except FileNotFoundError:
    has_full_preds = False
    predictions_full = pd.DataFrame()

print(f"ML data: {len(ml_clean)} labeled patients, {len(fs_cols)} features")
print(f"Discordant (IHC-/RNA-high): {len(disc_pids)} patients")

# ============================================================================
# THRESHOLD CONFIGURATIONS
# ============================================================================
# Each config: (pos_erbb2_quantile, pos_cn_floor, neg_erbb2_quantile)
# pos_erbb2_quantile: quantile of positive ERBB2 expr used as floor
# neg_erbb2_quantile: quantile of negative ERBB2 expr used as ceiling

CONFIGS = {
    "Strict (default)": {"pos_erbb2_q": 0.25, "pos_cn_floor": 1, "neg_erbb2_q": 0.75},
    "Relaxed":          {"pos_erbb2_q": 0.10, "pos_cn_floor": 0, "neg_erbb2_q": 0.90},
    "Stringent":        {"pos_erbb2_q": 0.40, "pos_cn_floor": 2, "neg_erbb2_q": 0.50},
}


def classify_discordant(row):
    """Same classification logic as NB03 cell 37."""
    cn = row["erbb2_copy_number"]
    consensus = row["consensus_score"]
    grb7 = row.get("GRB7_expr", np.nan)
    grb7_elevated = grb7 > 10.0 if pd.notna(grb7) else False

    if cn >= 2 and consensus > 0.3:
        return "IHC-missed HER2+"
    elif cn >= 2:
        return "Amplified, low confidence"
    elif consensus > 0.4 and grb7_elevated:
        return "Transcriptional HER2 activation"
    elif consensus > 0.3:
        return "Moderate molecular HER2 signal"
    else:
        return "Isolated ERBB2 elevation"


# ============================================================================
# RUN SENSITIVITY ANALYSIS
# ============================================================================
results = []

analysis = ml_clean[["pid", "her2_composite", "erbb2_copy_number"]].copy()
analysis["ERBB2_expr"] = analysis["pid"].map(tumor_expr["ERBB2"])
analysis = analysis.dropna(subset=["ERBB2_expr"])

pos_expr = analysis.loc[analysis["her2_composite"] == "Positive", "ERBB2_expr"]
neg_expr = analysis.loc[analysis["her2_composite"] == "Negative", "ERBB2_expr"]

# Per-patient classification across configs (for stability analysis)
patient_classifications = {}

for config_name, params in CONFIGS.items():
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name}")
    print(f"  pos ERBB2 floor: {params['pos_erbb2_q']*100:.0f}th pctl of positives")
    print(f"  pos CN floor: >= {params['pos_cn_floor']}")
    print(f"  neg ERBB2 ceiling: {params['neg_erbb2_q']*100:.0f}th pctl of negatives")
    print("=" * 70)

    # 1. Define concordant sets
    pos_floor = pos_expr.quantile(params["pos_erbb2_q"])
    neg_ceiling = neg_expr.quantile(params["neg_erbb2_q"])

    concordant_pos = analysis[
        (analysis["her2_composite"] == "Positive") &
        (analysis["ERBB2_expr"] >= pos_floor) &
        (analysis["erbb2_copy_number"] >= params["pos_cn_floor"])
    ]
    concordant_neg = analysis[
        (analysis["her2_composite"] == "Negative") &
        (analysis["ERBB2_expr"] <= neg_ceiling)
    ]

    n_conc_pos = len(concordant_pos)
    n_conc_neg = len(concordant_neg)
    n_excluded = len(analysis) - n_conc_pos - n_conc_neg
    print(f"  Concordant pos: {n_conc_pos}, neg: {n_conc_neg}, excluded: {n_excluded}")

    # 2. Train concordant-only RF (5-fold CV)
    concordant_pids = set(concordant_pos["pid"]) | set(concordant_neg["pid"])
    ml_conc = ml_clean[ml_clean["pid"].isin(concordant_pids)].copy()
    ml_conc = ml_conc.dropna(subset=fs_cols)
    ml_conc_y = (ml_conc["her2_composite"] == "Positive").astype(int)

    X_conc = ml_conc[fs_cols].values
    y_conc = ml_conc_y.values

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_cv = cross_val_predict(rf, X_conc, y_conc, cv=cv, method="predict_proba")[:, 1]
    auc_cv = roc_auc_score(y_conc, y_prob_cv)
    print(f"  CV AUC-ROC: {auc_cv:.3f}")

    # Retrain on full concordant set
    rf.fit(X_conc, y_conc)

    # 3. Score discordant patients
    ml_disc = ml_clean[ml_clean["pid"].isin(disc_pids)].dropna(subset=fs_cols).copy()
    X_disc = ml_disc[fs_cols].values
    ml_disc["conc_model_prob"] = rf.predict_proba(X_disc)[:, 1]

    # Add GRB7 expression for classification
    ml_disc["GRB7_expr"] = ml_disc["pid"].map(tumor_expr["GRB7"])

    # Build consensus score (average of full-data model probs + concordant-only)
    if has_full_preds:
        prob_cols_full = [c for c in predictions_full.columns if c.startswith("prob_")]
        if prob_cols_full:
            ml_disc = ml_disc.merge(
                predictions_full[["pid"] + prob_cols_full], on="pid", how="left"
            )
            all_prob_cols = prob_cols_full + ["conc_model_prob"]
        else:
            all_prob_cols = ["conc_model_prob"]
    else:
        all_prob_cols = ["conc_model_prob"]

    ml_disc["consensus_score"] = ml_disc[all_prob_cols].mean(axis=1)

    # 4. Classify discordant patients
    ml_disc["classification"] = ml_disc.apply(classify_discordant, axis=1)

    # Track per-patient classifications
    for _, row in ml_disc.iterrows():
        pid = row["pid"]
        if pid not in patient_classifications:
            patient_classifications[pid] = {}
        patient_classifications[pid][config_name] = {
            "classification": row["classification"],
            "consensus_score": row["consensus_score"],
            "conc_model_prob": row["conc_model_prob"],
        }

    # 5. Summarize
    cn2_patients = ml_disc[ml_disc["erbb2_copy_number"] >= 2]
    cn_low_patients = ml_disc[ml_disc["erbb2_copy_number"] < 2]
    cn2_missed = (cn2_patients["classification"] == "IHC-missed HER2+").sum()

    class_counts = ml_disc["classification"].value_counts().to_dict()

    print(f"  CN>=2 patients: {len(cn2_patients)}, classified as IHC-missed: {cn2_missed}")
    print(f"  CN<=1 patients: {len(cn_low_patients)}")
    print(f"  Classification breakdown:")
    for cls, count in sorted(class_counts.items()):
        print(f"    {cls}: {count}")

    results.append({
        "config": config_name,
        "pos_erbb2_q": params["pos_erbb2_q"],
        "pos_cn_floor": params["pos_cn_floor"],
        "neg_erbb2_q": params["neg_erbb2_q"],
        "n_concordant_pos": n_conc_pos,
        "n_concordant_neg": n_conc_neg,
        "n_excluded": n_excluded,
        "cv_auc_roc": auc_cv,
        "n_disc_scored": len(ml_disc),
        "cn2_ihc_missed": cn2_missed,
        "cn2_total": len(cn2_patients),
        "consensus_mean": ml_disc["consensus_score"].mean(),
        "consensus_std": ml_disc["consensus_score"].std(),
        **{f"n_{cls}": count for cls, count in class_counts.items()},
    })

# ============================================================================
# STABILITY ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("CN-STRATIFIED GROUP ASSIGNMENT STABILITY")
print("=" * 70)

config_names = list(CONFIGS.keys())
stable_count = 0
total_count = 0

for pid, configs in patient_classifications.items():
    if len(configs) == len(CONFIGS):
        total_count += 1
        classes = [configs[c]["classification"] for c in config_names]
        if len(set(classes)) == 1:
            stable_count += 1

print(f"Patients scored in all configs: {total_count}")
print(f"Stable classification (same across all configs): {stable_count} "
      f"({100*stable_count/max(total_count,1):.0f}%)")
print(f"Shifted classification: {total_count - stable_count}")

# Show patients that shifted
print("\nPatients with shifted classification:")
for pid, configs in sorted(patient_classifications.items()):
    if len(configs) == len(CONFIGS):
        classes = [configs[c]["classification"] for c in config_names]
        if len(set(classes)) > 1:
            cn_info = ml_clean.loc[ml_clean["pid"] == pid, "erbb2_copy_number"].iloc[0]
            shifts = " -> ".join(
                f"{c}: {configs[c]['classification']} ({configs[c]['consensus_score']:.3f})"
                for c in config_names
            )
            print(f"  {pid} (CN={cn_info:.0f}): {shifts}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
results_df = pd.DataFrame(results)
save_intermediate(results_df, "03_threshold_sensitivity")

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
summary_cols = ["config", "n_concordant_pos", "n_concordant_neg", "cv_auc_roc",
                "cn2_ihc_missed", "cn2_total", "consensus_mean"]
print(results_df[summary_cols].to_string(index=False, float_format="{:.3f}".format))
print("\nSaved: outputs/03_threshold_sensitivity.parquet")
