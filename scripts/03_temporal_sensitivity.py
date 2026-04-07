"""
03_temporal_sensitivity.py
==========================
Sensitivity analysis: diagnosis year as a potential confounder.

NB01 (Section 2.3) identified that HER2 label availability correlates with
diagnosis year (chi2=11.5, p=7e-04) due to the 2007 ASCO/CAP guideline change.
Pre-2007 samples had less standardized IHC testing. This script asks:

  1. Are pre-2007 samples overrepresented in the discordant or equivocal groups?
  2. Does excluding pre-2007 samples change the concordant-only model's
     performance or its scoring of discordant patients?

If results are stable across eras, diagnosis year is not driving the findings.

Outputs:
  outputs/03_temporal_sensitivity.parquet   (summary table)
  outputs/03_temporal_crosstab.parquet      (year x group cross-tabulation)
  outputs/figures/fig_03s_temporal_sensitivity.png
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from src.utils import (
    load_intermediate, load_gene_cols, save_intermediate, savefig,
    setup_plotting,
)

setup_plotting()

YEAR_CUTOFF = 2007  # ASCO/CAP HER2 testing guidelines published

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("=" * 70)
print("TEMPORAL SENSITIVITY ANALYSIS: DIAGNOSIS YEAR AS CONFOUNDER")
print("=" * 70)

clinical = load_intermediate("01_clinical_qc")
tumor_norm = load_intermediate("01_tumor_norm_tmm_tss")
cn = load_intermediate("01_cn_qc")
gene_cols = load_gene_cols()
cohort_c = load_intermediate("02_multimodal_cohort")
discordant_df = load_intermediate("02_discordant_cases")

# Equivocal scores (from exploratory ML script)
try:
    equivocal_scores = load_intermediate("03_equivocal_scores")
    has_equivocal = True
except FileNotFoundError:
    has_equivocal = False
    equivocal_scores = pd.DataFrame()

tumor_expr = tumor_norm.set_index("pid")

# Curated gene sets (same as NB03 / threshold sensitivity script)
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
# 2. CROSS-TABULATION: DIAGNOSIS YEAR x HER2 GROUP
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: DIAGNOSIS YEAR DISTRIBUTION BY HER2 GROUP")
print("=" * 70)

# dx_year is already in multimodal cohort (carried from clinical_qc)
cohort_year = cohort_c.copy()

# Identify group membership
disc_pids = set(
    discordant_df[discordant_df["discordance_type"] == "IHC-/RNA-high"]["pid"].unique()
)
equiv_pids = set(cohort_year[cohort_year["her2_composite"] == "Equivocal"]["pid"])

def assign_group(row):
    if row["pid"] in disc_pids:
        return "Discordant (IHC-/RNA-high)"
    elif row["pid"] in equiv_pids:
        return "Equivocal"
    elif row["her2_composite"] == "Positive":
        return "Concordant Positive"
    elif row["her2_composite"] == "Negative":
        return "Concordant Negative"
    else:
        return "Other"

cohort_year["group"] = cohort_year.apply(assign_group, axis=1)
cohort_year["era"] = np.where(
    cohort_year["dx_year"] < YEAR_CUTOFF, "Pre-2007", "2007+"
)
cohort_year.loc[cohort_year["dx_year"].isna(), "era"] = "Unknown"

# Cross-tabulation
crosstab = pd.crosstab(
    cohort_year["group"], cohort_year["era"],
    margins=True, margins_name="Total",
)
print("\nYear-era x HER2 group cross-tabulation:")
print(crosstab.to_string())

# Percentage pre-2007 by group
print("\nPre-2007 fraction by group:")
for grp in ["Concordant Positive", "Concordant Negative",
            "Discordant (IHC-/RNA-high)", "Equivocal"]:
    grp_rows = cohort_year[cohort_year["group"] == grp]
    n_total = len(grp_rows)
    n_pre = (grp_rows["dx_year"] < YEAR_CUTOFF).sum()
    n_known = grp_rows["dx_year"].notna().sum()
    if n_known > 0:
        pct = 100 * n_pre / n_known
        print(f"  {grp}: {n_pre}/{n_known} ({pct:.1f}%) pre-2007  [of {n_total} total]")
    else:
        print(f"  {grp}: no year data available")

# Save cross-tab
crosstab_df = crosstab.reset_index()
crosstab_df.columns.name = None
save_intermediate(crosstab_df, "03_temporal_crosstab")
print("\nSaved: outputs/03_temporal_crosstab.parquet")

# ============================================================================
# 3. BUILD ML DATA (same as NB03 / threshold sensitivity)
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: CONCORDANT-ONLY MODEL -- POST-2007 SENSITIVITY")
print("=" * 70)

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

# dx_year already present from cohort_c
# Concordant group definition (same thresholds as NB03 Section 4 / strict default)
analysis = ml_clean[["pid", "her2_composite", "erbb2_copy_number", "dx_year"]].copy()
analysis["ERBB2_expr"] = analysis["pid"].map(tumor_expr["ERBB2"])
analysis = analysis.dropna(subset=["ERBB2_expr"])

pos_expr = analysis.loc[analysis["her2_composite"] == "Positive", "ERBB2_expr"]
neg_expr = analysis.loc[analysis["her2_composite"] == "Negative", "ERBB2_expr"]

concordant_pos = analysis[
    (analysis["her2_composite"] == "Positive") &
    (analysis["ERBB2_expr"] >= pos_expr.quantile(0.25)) &
    (analysis["erbb2_copy_number"] >= 1)
]
concordant_neg = analysis[
    (analysis["her2_composite"] == "Negative") &
    (analysis["ERBB2_expr"] <= neg_expr.quantile(0.75))
]

concordant_pids = set(concordant_pos["pid"]) | set(concordant_neg["pid"])

# ============================================================================
# 4. TRAIN MODELS: ALL-ERA vs POST-2007-ONLY
# ============================================================================

def train_concordant_rf(pid_set, label, ml_data, feature_cols):
    """Train concordant-only RF with 5-fold CV, return metrics and fitted model."""
    ml_conc = ml_data[ml_data["pid"].isin(pid_set)].copy()
    ml_conc = ml_conc.dropna(subset=feature_cols)
    y = (ml_conc["her2_composite"] == "Positive").astype(int).values
    X = ml_conc[feature_cols].values

    if len(np.unique(y)) < 2:
        print(f"  {label}: only one class present -- skipping")
        return None, None, None, None

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, y_prob)

    # Refit on full set for scoring
    rf.fit(X, y)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    print(f"  {label}: n={len(y)} (pos={n_pos}, neg={n_neg}), CV AUC={auc:.3f}")

    return rf, auc, n_pos, n_neg


# -- Model A: All eras (baseline, replicates NB03 Section 4) --
print("\nModel A: All eras (baseline)")
rf_all, auc_all, npos_all, nneg_all = train_concordant_rf(
    concordant_pids, "All eras", ml_clean, fs_cols
)

# -- Model B: Post-2007 only --
post2007_pids = set(
    analysis[analysis["dx_year"] >= YEAR_CUTOFF]["pid"]
)
concordant_post2007 = concordant_pids & post2007_pids

# How many concordant patients are pre-2007?
concordant_pre2007 = concordant_pids - post2007_pids - set(
    analysis[analysis["dx_year"].isna()]["pid"]
)
concordant_unknown_year = concordant_pids & set(
    analysis[analysis["dx_year"].isna()]["pid"]
)
print(f"\nConcordant training set breakdown:")
print(f"  Post-2007: {len(concordant_post2007)}")
print(f"  Pre-2007:  {len(concordant_pre2007)}")
print(f"  Unknown:   {len(concordant_unknown_year)}")

print("\nModel B: Post-2007 only")
rf_post, auc_post, npos_post, nneg_post = train_concordant_rf(
    concordant_post2007, "Post-2007", ml_clean, fs_cols
)

# ============================================================================
# 5. SCORE DISCORDANT PATIENTS WITH BOTH MODELS
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: DISCORDANT PATIENT SCORING -- MODEL COMPARISON")
print("=" * 70)

ml_disc = ml_clean[ml_clean["pid"].isin(disc_pids)].dropna(subset=fs_cols).copy()
X_disc = ml_disc[fs_cols].values

if rf_all is not None and rf_post is not None and len(ml_disc) > 0:
    ml_disc = ml_disc.copy()
    ml_disc["prob_all_era"] = rf_all.predict_proba(X_disc)[:, 1]
    ml_disc["prob_post2007"] = rf_post.predict_proba(X_disc)[:, 1]
    ml_disc["prob_delta"] = ml_disc["prob_post2007"] - ml_disc["prob_all_era"]

    print(f"\nDiscordant patients scored: {len(ml_disc)}")
    print(f"  Mean prob (all-era model):  {ml_disc['prob_all_era'].mean():.3f}")
    print(f"  Mean prob (post-2007 model): {ml_disc['prob_post2007'].mean():.3f}")
    print(f"  Mean delta:                  {ml_disc['prob_delta'].mean():.3f}")
    print(f"  Max |delta|:                 {ml_disc['prob_delta'].abs().max():.3f}")

    # Correlation between the two models' scores
    from scipy import stats
    r, p = stats.pearsonr(ml_disc["prob_all_era"], ml_disc["prob_post2007"])
    print(f"  Pearson r (all-era vs post-2007): {r:.3f} (p={p:.2e})")

    # How many discordant patients are themselves pre-2007?
    n_disc_pre = (ml_disc["dx_year"] < YEAR_CUTOFF).sum()
    n_disc_known = ml_disc["dx_year"].notna().sum()
    print(f"\n  Discordant patients pre-2007: {n_disc_pre}/{n_disc_known}")

    # CN-stratified comparison
    print("\n  CN-stratified scoring delta:")
    for cn_label, cn_mask in [("CN >= 2", ml_disc["erbb2_copy_number"] >= 2),
                               ("CN < 2",  ml_disc["erbb2_copy_number"] < 2)]:
        subset = ml_disc[cn_mask]
        if len(subset) > 0:
            print(f"    {cn_label} (n={len(subset)}): "
                  f"delta mean={subset['prob_delta'].mean():.3f}, "
                  f"delta std={subset['prob_delta'].std():.3f}")

else:
    print("Could not score discordant patients (model training failed).")

# ============================================================================
# 6. SCORE EQUIVOCAL PATIENTS (if available)
# ============================================================================
if has_equivocal and len(equivocal_scores) > 0 and rf_post is not None:
    print("\n" + "=" * 70)
    print("PART 4: EQUIVOCAL PATIENT SCORING -- MODEL COMPARISON")
    print("=" * 70)

    ml_equiv = ml_df[ml_df["her2_composite"] == "Equivocal"].copy()
    ml_equiv = ml_equiv.dropna(subset=fs_cols)
    if len(ml_equiv) > 0:
        X_eq = ml_equiv[fs_cols].values
        ml_equiv = ml_equiv.copy()
        ml_equiv["prob_all_era"] = rf_all.predict_proba(X_eq)[:, 1]
        ml_equiv["prob_post2007"] = rf_post.predict_proba(X_eq)[:, 1]
        ml_equiv["prob_delta"] = ml_equiv["prob_post2007"] - ml_equiv["prob_all_era"]

        print(f"Equivocal patients scored: {len(ml_equiv)}")
        print(f"  Mean prob (all-era model):  {ml_equiv['prob_all_era'].mean():.3f}")
        print(f"  Mean prob (post-2007 model): {ml_equiv['prob_post2007'].mean():.3f}")
        print(f"  Mean delta:                  {ml_equiv['prob_delta'].mean():.3f}")

        # Reclassification stability at threshold 0.5
        reclass_all = (ml_equiv["prob_all_era"] >= 0.5).astype(int)
        reclass_post = (ml_equiv["prob_post2007"] >= 0.5).astype(int)
        agreement = (reclass_all == reclass_post).sum()
        print(f"  Reclassification agreement (threshold=0.5): "
              f"{agreement}/{len(ml_equiv)} ({100*agreement/len(ml_equiv):.0f}%)")

# ============================================================================
# 7. SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

summary_rows = []

if rf_all is not None:
    summary_rows.append({
        "model": "All eras (baseline)",
        "n_train": npos_all + nneg_all,
        "n_pos": npos_all,
        "n_neg": nneg_all,
        "cv_auc": auc_all,
        "disc_prob_mean": ml_disc["prob_all_era"].mean() if len(ml_disc) > 0 else np.nan,
        "disc_prob_std": ml_disc["prob_all_era"].std() if len(ml_disc) > 0 else np.nan,
    })

if rf_post is not None:
    summary_rows.append({
        "model": "Post-2007 only",
        "n_train": npos_post + nneg_post,
        "n_pos": npos_post,
        "n_neg": nneg_post,
        "cv_auc": auc_post,
        "disc_prob_mean": ml_disc["prob_post2007"].mean() if len(ml_disc) > 0 else np.nan,
        "disc_prob_std": ml_disc["prob_post2007"].std() if len(ml_disc) > 0 else np.nan,
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False, float_format="{:.3f}".format))

save_intermediate(summary_df, "03_temporal_sensitivity")
print("\nSaved: outputs/03_temporal_sensitivity.parquet")

# ============================================================================
# 8. FIGURE: SIDE-BY-SIDE COMPARISON
# ============================================================================
if rf_all is not None and rf_post is not None and len(ml_disc) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Cross-tab bar chart (pre-2007 fraction by group)
    ax = axes[0]
    groups_for_plot = ["Concordant Positive", "Concordant Negative",
                       "Discordant (IHC-/RNA-high)", "Equivocal"]
    short_labels = ["Conc+", "Conc-", "Discord", "Equiv"]
    pre_fracs = []
    for grp in groups_for_plot:
        grp_rows = cohort_year[cohort_year["group"] == grp]
        n_known = grp_rows["dx_year"].notna().sum()
        n_pre = (grp_rows["dx_year"] < YEAR_CUTOFF).sum()
        pre_fracs.append(100 * n_pre / n_known if n_known > 0 else 0)

    colors = ["#e74c3c", "#3498db", "#9b59b6", "#f39c12"]
    bars = ax.bar(short_labels, pre_fracs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("% Pre-2007")
    ax.set_title("A. Pre-2007 Fraction by Group", fontweight="bold")
    ax.set_ylim(0, max(pre_fracs) * 1.3 if max(pre_fracs) > 0 else 10)
    for bar, frac in zip(bars, pre_fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{frac:.1f}%", ha="center", va="bottom", fontsize=9)

    # Panel B: Scatter -- all-era vs post-2007 discordant scores
    ax = axes[1]
    ax.scatter(ml_disc["prob_all_era"], ml_disc["prob_post2007"],
               c=ml_disc["erbb2_copy_number"], cmap="RdYlBu_r",
               edgecolors="black", linewidth=0.5, s=50, zorder=3)
    lims = [0, 1]
    ax.plot(lims, lims, "--", color="gray", linewidth=0.8, zorder=1)
    ax.set_xlabel("P(HER2+) -- All-Era Model")
    ax.set_ylabel("P(HER2+) -- Post-2007 Model")
    ax.set_title(f"B. Discordant Scores (r={r:.2f})", fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    # Panel C: Delta distribution
    ax = axes[2]
    ax.hist(ml_disc["prob_delta"], bins=15, color="#95a5a6", edgecolor="black",
            linewidth=0.5)
    ax.axvline(0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Delta P(HER2+)  [post-2007 minus all-era]")
    ax.set_ylabel("Count")
    mean_d = ml_disc["prob_delta"].mean()
    ax.set_title(f"C. Score Shift (mean={mean_d:+.3f})", fontweight="bold")

    fig.suptitle(
        "Temporal Sensitivity: Excluding Pre-2007 Samples\n"
        f"Concordant-only RF  |  AUC: all-era={auc_all:.3f}, post-2007={auc_post:.3f}",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    savefig(fig, "fig_03s_temporal_sensitivity")
    print("Saved: outputs/figures/fig_03s_temporal_sensitivity.png")

print("\nDone.")
