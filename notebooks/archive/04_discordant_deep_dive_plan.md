# Notebook 04: Discordant Case Deep Dive

## Motivation

Notebook 03a's binary HER2 classifier, trained on IHC labels, predicts nearly all 35
IHC-negative/RNA-high discordant patients as negative (33/35, probabilities 0.12–0.49;
only 2 above 0.5). This is expected — the model learned to reproduce IHC, so it
faithfully reproduces IHC's blind spots. The question is no longer "are these patients
secretly HER2-positive?" but "what *are* these patients, and is there a coherent
subgroup among them that might benefit from HER2-targeted therapy?"

This notebook approaches that question from three angles.

---

## Inputs (all from prior notebooks)

| Intermediate | Source | Key columns / contents |
|---|---|---|
| `01_clinical_qc` | NB01 | Full clinical metadata (1108 × 148) |
| `01_tumor_norm` | NB01 | Normalized expression matrix (1093 × ~17K genes) |
| `02_multimodal_cohort` | NB02 | Merged clinical + RNA + CN (966 × ~17.8K) |
| `02_discordant_cases` | NB02 | 69 rows: pid, discordance_type, her2_composite, ERBB2_expr, erbb2_copy_number, HER2_ihc_score, HER2_fish_status, GRB7_expr |
| `02_subtype_assignments` | NB02 | pid, cluster_k4, provisional_subtype (1093 rows) |
| `02_umap_embeddings` | NB02 | pid, UMAP1, UMAP2 |
| `03_ml_predictions` | NB03a | pid, her2_composite, ml_prob_her2_positive, ml_pred_her2, prob_L1-LR, prob_Random_Forest, prob_XGBoost (960 rows) |
| `03_ssgsea_scores` | NB03a | pid + 30 pathway score columns (960 rows) |
| `03_gsea_results` | NB03a | GSEA results table (30 pathways) |

---

## Section 1: Setup & Cohort Definition

Load all intermediates. Extract the 35 IHC-/RNA-high discordant patients by filtering
`02_discordant_cases` on `discordance_type == 'IHC-/RNA-high'`.

Merge onto these patients:
- Expression data (from `01_tumor_norm`)
- ML predicted probabilities (from `03_ml_predictions`)
- ssGSEA pathway scores (from `03_ssgsea_scores`)
- CN status, subtype, UMAP coordinates

Define comparison groups (all IHC-negative, matched on available covariates where possible):
- **Concordant negatives**: IHC-negative AND ERBB2 expression below the negative
  population median. These are the "clean" negatives. (expected n ≈ 340)
- **Discordant IHC-/RNA-high**: the 35 patients of interest.

Split the discordant group by CN:
- **Discordant-amplified**: CN = 2 (expected ~2–5 patients based on NB02 cell 41)
- **Discordant-non-amplified**: CN ≤ 1 (the majority, ~30 patients)

---

## Section 2: Concordant-Only Model (Option 1 — Anomaly-Aware Scoring)

### 2.1 Rationale

The 03a model was trained on all labeled patients, including discordant cases that may
be pulling the decision boundary. Here we retrain on "concordant-only" samples — patients
where IHC and RNA agree — so the model learns the *typical* IHC-positive and IHC-negative
signatures without contamination from the ambiguous tails.

### 2.2 Implementation

Define concordant training set:
- **Concordant Positive**: IHC-positive AND ERBB2 expression ≥ positive population
  25th percentile AND CN ≥ 1.
- **Concordant Negative**: IHC-negative AND ERBB2 expression ≤ negative population
  75th percentile.

Use the same feature set that performed best in 03a (Feature Set B: Curated Panel,
32 features) and the same model (Random Forest, same hyperparameters). Retrain with
stratified 5-fold CV on the concordant subset only. Report AUC on concordant data
(should be higher than 03a since we removed ambiguous cases).

### 2.3 Score the discordant cases

Apply the concordant-trained model to:
1. The 35 IHC-/RNA-high discordant patients
2. All other patients excluded from concordant training (for context)

Report predicted P(HER2+) for each discordant patient alongside their CN status, GRB7
expression, and provisional subtype.

### 2.4 Per-patient SHAP

For discordant patients with model probability > 0.3, compute per-patient SHAP values
using the concordant-trained model. This reveals *which features* push each individual
patient toward positive — is it ERBB2 alone, or is there a broader multi-gene signature?

Visualize as a SHAP waterfall plot for 3–5 representative patients (pick one
discordant-amplified and several discordant-non-amplified).

---

## Section 3: CN-Stratified Analysis (Option 2 — Amplicon vs. Non-Amplicon)

### 3.1 Discordant-Amplified Subgroup (CN = 2)

These are the strongest "IHC-missed" candidates. Compare them to confirmed
IHC-positive/CN=2 patients on:
- ERBB2 expression (are they comparable?)
- GRB7/STARD3 co-expression (is the full 17q12 amplicon active?)
- ssGSEA pathway profile (ERBB2/HER2 signaling, PI3K-AKT, mTOR)
- Concordant-model predicted probability

If these patients match the IHC+/CN=2 profile across all modalities, they are strong
candidates for reclassification. Report as a small-n case series with per-patient detail.

### 3.2 Discordant-Non-Amplified Subgroup (CN ≤ 1)

This is the larger, more interesting group. ERBB2 is transcriptionally upregulated
without genomic amplification. The question: is this a coherent biology or noise?

**Differential expression analysis:**
Compare discordant-non-amplified (n ≈ 30) vs. a matched set of concordant negatives
with the same CN range (CN ≤ 1). Match on ER status if possible to avoid confounding
the DE with ER-driven differences.

- Compute t-statistics for all ~17K genes.
- Rank by significance. Report top 50 upregulated and top 50 downregulated genes.
- Check: is ERBB2 the only differentially expressed gene, or are there co-regulated
  genes? If the DE signature is ERBB2-only, this is likely stochastic transcriptional
  noise. If there's a broader program, it's biologically coherent.

**Pathway analysis on the DE signature:**
Run GSEA (preranked by t-statistic) against the same Hallmark gene sets used in 03a.
Key questions:
- Is ERBB2/HER2 signaling enriched? (would indicate downstream pathway activation
  despite no amplification)
- Are there unexpected pathways? (e.g., immune, EMT — could indicate a distinct
  biology)

**17q12 amplicon gene check:**
Beyond ERBB2 and GRB7, check the full set of 17q12 amplicon genes (STARD3, PGAP3,
TCAP, MIEN1/ORMDL3). If these are NOT co-upregulated, the ERBB2 elevation is not
amplicon-driven — consistent with the CN data and pointing toward transcriptional
regulation.

---

## Section 4: Continuous Molecular HER2 Score (Option 3 — Probability Reframing)

### 4.1 Construct a composite score

Rather than a binary call, combine the three model probabilities from 03a
(L1-LR, Random Forest, XGBoost) into a consensus molecular HER2 score for each
discordant patient. Simple approach: mean of the three probabilities.

### 4.2 Stratify within the discordant group

Plot the consensus score against CN status (x-axis: CN, y-axis: consensus probability,
point size or color: GRB7 expression). This should visually separate the discordant
group into interpretable quadrants:
- High score + high CN → "IHC-missed HER2+" (strongest reclassification candidates)
- High score + low CN → "Transcriptional HER2 activation" (biologically interesting)
- Low score + any CN → "ERBB2 expression outlier without broader HER2 program"

### 4.3 Cross-reference with Section 2 and Section 3

Annotate each patient with:
- Concordant-model probability (from Section 2)
- Consensus probability (from Section 4)
- DE group membership (amplified vs non-amplified, from Section 3)
- Provisional subtype (from NB02)

Present as a single patient-level table — the final "discordant case dossier."

---

## Section 5: Synthesis & Visualization

### 5.1 Summary figure

A multi-panel figure that ties the analysis together:

- **Panel A**: UMAP of the full cohort, with discordant patients highlighted by CN
  subgroup (amplified vs non-amplified), sized by concordant-model probability.
- **Panel B**: Heatmap of key gene expression (ERBB2, GRB7, STARD3, ESR1, MKI67,
  top DE genes from Section 3) for: concordant positives, concordant negatives,
  discordant-amplified, discordant-non-amplified. Ordered by concordant-model
  probability within each group.
- **Panel C**: ssGSEA pathway comparison (bar or dot plot) across the four groups,
  focusing on ERBB2/HER2 signaling, PI3K-AKT, estrogen response.

### 5.2 Patient-level summary table

One row per discordant patient. Columns:
- pid, CN, ERBB2_expr, GRB7_expr
- IHC score, FISH status (from clinical data)
- 03a model probability (RF), concordant-model probability
- Consensus score
- Provisional subtype
- Classification: "IHC-missed" / "Transcriptional upregulation" / "Inconclusive"

### 5.3 Interpretive text

Write a concluding interpretation covering:
1. How many discordant patients show evidence of genuine HER2 pathway activation vs.
   isolated ERBB2 expression elevation.
2. Whether the non-amplified subgroup represents a coherent biology (based on DE/GSEA)
   or stochastic noise.
3. Clinical implications: which patients, if any, might be candidates for HER2-targeted
   therapy (especially T-DXd, which has shown efficacy in HER2-low populations).
4. Limitations: sample size (n=35 total, split further by CN), TCGA-specific biases,
   lack of protein-level validation.

---

## Section 6: Save Intermediates

Save for potential downstream use:
- `04_discordant_dossier`: patient-level summary table
- `04_concordant_model_predictions`: full-cohort predictions from the concordant-trained model
- `04_discordant_de_results`: DE genes for non-amplified subgroup
- `04_discordant_gsea`: GSEA results for non-amplified DE signature

---

## Execution Notes

- All gene expression data comes from the upper-quartile normalized, log2-transformed
  matrix produced in NB01. No re-normalization needed.
- DE analysis uses Welch's t-test (same as NB03a/03b GSEA preranking). With n ≈ 30 vs
  n ≈ 340, power is limited; we focus on effect sizes and pathway-level results rather
  than individual gene significance after multiple testing correction.
- The concordant-only model retraining is the most computationally intensive step but
  should run in under a minute with the same Random Forest configuration.
- SHAP waterfall plots require the `shap` library; fall back to feature importance bars
  if unavailable.
