# ML Notebook Consolidation Plan

**Objective:** Merge Notebooks 03a, 03b, and 04 into a single notebook that tells
one coherent story: "ML applied to TCGA HER2 dataset reveals complex heterogeneity
in discordant group biology."

**Date:** 2026-04-06
**Status:** Active (supersedes Revision Plan_260403.md)

**Hard constraint:** No ML without feature reduction. All models use either a
curated HER2-relevant gene list or GSEA-derived feature sets -- never raw
full-transcriptome features. This applies to binary classification, multi-class,
and concordant-only models alike.

---

## 1. Diagnosis: What Each Notebook Contributes

### 03a (03_Machine_Learning.ipynb) -- Binary HER2 Classification
- **Keep (as brief setup):** Feature matrix construction, 3-model comparison
  (L1-LR, RF, XGBoost) on binary HER2 label, key finding that ERBB2 alone is
  nearly as predictive as full panel (AUC ~0.84 vs ~0.87).
- **Move to script:** Calibration analysis, threshold sweep, equivocal scoring
  (these are useful exploratory work but tangential to the discordant biology story).
- **Key insight to preserve:** Full-dataset binary model gives near-universal negative
  predictions for discordant cases -- this motivates the concordant-only approach in NB04.

### 03b (03b_Machine_Learning(1).ipynb) -- Multi-Class Subtype ML + GSEA
- **Keep (condensed):** The finding that multi-class subtype classification on
  cluster-derived labels doesn't cleanly capture discordant biology. GSEA results
  (one-vs-rest by subtype, HER2+ vs HER2-) are valuable context.
- **Move to script:** Full multi-class model comparison (3 models x N feature sets),
  per-subtype SHAP, survival overlay, equivocal scoring by subtype, ssGSEA computation,
  clinical implications summary. These are extensive analyses that don't serve the
  consolidated narrative.
- **Key insight to preserve:** Cluster-derived subtypes are approximations; discordant
  patients don't map cleanly to any single cluster, confirming that the discordant
  group requires dedicated analysis (NB04's approach).

### 04 (04_Discordant_Deep_Dive.ipynb) -- The Core Story
- **Keep (nearly all):** Concordant-only model, discordant scoring, per-patient SHAP,
  CN-stratified biology (amplified vs non-amplified), differential expression,
  pathway analysis on DE signature, consensus molecular HER2 score, synthesis
  visualization, expression heatmap, pathway score comparison.
- **Minor edits:** Add explicit comparison of GSEA findings against NB02's
  unsupervised cluster GSEA. Tighten the narrative framing.

---

## 2. Consolidated Notebook Structure

**Filename:** `03_ML_and_Discordant_Biology.ipynb`

### Section 1: Setup and Data Loading
- Single import block (deduplicated from 03a/03b/04)
- Load all intermediates from NB01 and NB02
- Brief cohort summary (N patients, label distribution)

### Section 2: Binary HER2 Classification (from 03a, condensed)
**Purpose:** Establish baseline -- how well can ML predict HER2 from expression?

- **Feature selection (curated, not full-transcriptome):**
  - Start with a curated HER2-relevant gene panel (ERBB2, GRB7, STARD3, PGAP3,
    TCAP, EGFR, ERBB3, ERBB4, PIK3CA, AKT1, ESR1, PGR, FOXA1, GATA3, MKI67, etc.)
  - **NB02 GSEA cross-reference:** Load NB02's unsupervised cluster GSEA results.
    Identify any gene sets / pathways that emerged as discriminating across clusters
    but are NOT represented in the curated panel. If NB02 flagged pathways like
    EMT, interferon, or proliferation signatures, add representative genes from
    those sets. Document which genes were added and why (data-driven augmentation
    of the curated panel).
  - Also include CN and clinical features (ER, PR status)
- 3-model comparison: L1-LR, RF, XGBoost (single training cell, single results table)
- ROC curves (one figure)
- Key finding: ERBB2 alone AUC ~0.84 vs curated panel AUC ~0.87
  - Interpretation: modest improvement suggests ERBB2 dominates the signal
  - **But:** this is on full dataset including discordant cases in training --
    the discordant population may be suppressing heterogeneous biology
- SHAP feature importance (one summary figure -- beeswarm or bar)
- Score discordant patients with the full-data model
  - Show that most discordant cases get predicted negative
  - This is the pivot point: "the full-data model fails to see the discordant group"

### Section 3: Multi-Class Context (from 03b, heavily condensed)
**Purpose:** Show that unsupervised cluster labels don't resolve the discordant
question either.

- Brief recap: NB02 identified k=4 clusters approximating PAM50 subtypes
- GSEA summary (from 03b): which pathways define each cluster?
  - Present as a single heatmap (NES by subtype)
  - **New addition:** Compare against NB02's cluster-level GSEA findings
    (load `02_cluster_profiles` or relevant intermediate). Note concordance
    and any discrepancies.
- Where do discordant patients fall in cluster space?
  - UMAP/PCA colored by concordant/discordant status
  - Distribution table: discordant patients by cluster assignment
  - Finding: discordant patients are scattered, not concentrated in one cluster
- One-paragraph interpretation: cluster labels are noisy proxies; dedicated
  analysis needed for the discordant group

### Section 4: Concordant-Only Model (from 04, kept mostly intact)
**Purpose:** Retrain on unambiguous cases to get a cleaner reference frame.

- Cohort definition (concordant pos, concordant neg, discordant by CN status)
- Feature set: same curated gene panel from Section 2 (+ any NB02-GSEA-augmented
  genes) + CN + clinical. Must match Section 2 features for comparability.
- Train RF on concordant-only data, 5-fold CV
- Score all patients including discordant
- Comparison figure: concordant-model prob vs full-data model prob
  - Highlight CN-stratification in the discordant group
- Per-patient SHAP for discordant cases with prob > 0.3

### Section 5: CN-Stratified Discordant Biology (from 04, kept intact)
**Purpose:** The discordant group is not one population -- CN status reveals
two distinct biological mechanisms.

- 5.1 Amplified subgroup (CN=2): case series, expression profiles
  - Likely IHC technical failures -- genuine 17q12 amplification
- 5.2 Non-amplified subgroup (CN<=1): differential expression analysis
  - Volcano plot
  - Top DE genes -- is this isolated ERBB2 or a broader program?
- 5.3 Pathway analysis on DE signature
  - GSEA preranked by t-statistic
  - **New addition:** Explicitly compare these enriched pathways against:
    (a) the curated breast cancer gene sets used in 03b
    (b) the pathways that emerged from NB02's unsupervised cluster GSEA
    Note which pathways overlap and which are unique to the discordant group

### Section 6: Consensus Molecular HER2 Score (from 04, kept intact)
- Combine probabilities from full-data models + concordant-only model
- Graded score rather than binary call
- Classification of discordant patients into tiers

### Section 7: Equivocal Sample Scoring
**Purpose:** Demonstrate clinical utility -- RNA-based resolution of ambiguous IHC.

- Score equivocal patients (IHC 2+ without FISH resolution) using both:
  (a) the full-data binary model (Section 2)
  (b) the concordant-only model (Section 4)
- Distribution of predicted probabilities for equivocal patients
- How many resolve to Positive (prob > 0.5/0.7) vs Negative (prob < 0.3)?
- Brief comparison: do the two models agree on equivocal scoring?
- One-paragraph clinical framing: RNA-seq as reflex test for equivocal cases,
  reducing time-to-treatment-decision
- Note: Full calibration analysis and threshold sweeps are in
  `scripts/03_exploratory_binary_ml.py`

### Section 8: Synthesis Visualization (from 04, kept intact)
- UMAP with discordant patients highlighted
- Expression heatmap (key genes by group: Conc Pos, Disc CN=2, Disc CN<=1, Conc Neg)
- Pathway score comparison (ssGSEA across groups)
- Note: ssGSEA scores are pre-computed in `scripts/03_exploratory_multiclass_ml.py`.
  This section loads them from `outputs/03_ssgsea_scores.parquet`. A brief methods
  note describes the ssGSEA computation (gene sets used, scoring method, software)
  and points to the script for reproducibility.

### Section 9: Interpretation and Conclusions
- Narrative synthesis of all findings
- The 35 IHC-/RNA-high discordant patients are NOT a single population
- CN-stratified biology: amplicon-driven vs transcriptionally-driven
- Clinical implications (brief)
- Limitations (sample size, cluster labels as noisy targets, etc.)

### Section 10: Save Intermediates
- Consolidated outputs (predictions, feature importance, discordant dossier)

---

## 3. What Moves to Scripts

### scripts/03_exploratory_binary_ml.py
From 03a:
- Full calibration analysis (calibration curves, reliability diagrams)
- Threshold sweep with sensitivity/specificity tradeoff table
- Equivocal sample scoring (full detail)
- All-patient scoring
- Biological interpretation of top features (detailed annotation block)

### scripts/03_exploratory_multiclass_ml.py
From 03b:
- Full multi-class model comparison (3 models x feature sets)
- ssGSEA score computation
- Per-subtype SHAP analysis
- GSEA-vs-SHAP biological cross-validation
- Survival overlay (Kaplan-Meier by subtype)
- Equivocal scoring by subtype
- Clinical implications summary
- Full GSEA computation (one-vs-rest, HER2+ vs HER2-)

Note: The consolidated notebook loads pre-computed outputs from these scripts
(e.g., ssGSEA scores, full-data model predictions). Each script is self-contained
and runnable independently. The notebook includes a brief methods description
for each pre-computed result, with a pointer to the script for full details
and reproducibility.

---

## 4. Archive Strategy

```
notebooks/
  archive/
    03_Machine_Learning.ipynb          (was 03a)
    03b_Machine_Learning(1).ipynb      (was 03b)
    04_Discordant_Deep_Dive.ipynb      (original NB04)
    Revision Plan_260403.md            (superseded by this plan)
  03_ML_and_Discordant_Biology.ipynb   (new consolidated)
```

The .py script exports of old notebooks already exist and serve as additional
archive. Do not delete them.

---

## 5. NB02 GSEA Cross-Reference (Threaded Through Notebook)

This addresses the weakness that NB02's unsupervised cluster GSEA findings are
not leveraged in the ML notebooks. The cross-reference appears in THREE places:

**Implementation:**
1. Load NB02's cluster characterization outputs (cluster profiles, marker
   expression, subtype assignments, GSEA results) at notebook startup.
2. **In Section 2 (binary classification feature selection):** Use NB02's
   cluster GSEA to augment the curated gene panel. If NB02's unsupervised
   analysis flagged pathways (e.g., EMT, interferon, proliferation) that
   discriminate between clusters but are absent from the default curated panel,
   add representative genes. This ensures the ML feature set is informed by
   what the data itself highlighted, not just prior knowledge.
3. **In Section 3 (multi-class context):** Compare 03b's supervised GSEA
   (pathways enriched per predicted subtype) against NB02's unsupervised GSEA
   (pathways enriched per cluster). Present as a concordance table or
   Venn-style summary.
4. **In Section 5.3 (discordant pathway analysis):** After running GSEA on the
   discordant DE signature, explicitly note which of those pathways were also
   flagged in NB02's cluster GSEA and which are novel to the discordant group.

This connects the unsupervised and supervised threads and strengthens the
narrative that discordant biology is partially captured but not fully resolved
by clustering alone.

---

## 6. Intermediate Dependencies

The consolidated notebook loads:
- From NB01: `01_clinical_qc`, `01_tumor_norm`, `01_cn_qc`, `01_gene_cols.json`
- From NB02: `02_multimodal_cohort`, `02_analysis_df`, `02_discordant_cases`,
  `02_subtype_assignments`, `02_umap_embeddings`, `02_cluster_assignments`,
  `02_pca_embeddings`
- From scripts (pre-computed): `03_ssgsea_scores`, `03_ml_predictions` (full-data
  model predictions, needed for consensus score)

The notebook produces:
- `03_ml_predictions.parquet` (updated: includes both full-data and concordant-only
  model probabilities)
- `03_feature_importance.parquet`
- `03_concordant_model_predictions.parquet`
- `03_discordant_de_results.parquet`
- `03_discordant_gsea.parquet`
- `03_discordant_dossier.parquet`

Note: Output numbering changes from 04_* to 03_* since the consolidated notebook
is now NB03. This may require updating any downstream references.

---

## 7. Execution Sequence

1. **Archive** old notebooks into `notebooks/archive/`
2. **Create scripts** (03_exploratory_binary_ml.py, 03_exploratory_multiclass_ml.py)
   by extracting the relevant cells from 03a and 03b
3. **Run scripts** to confirm they produce the expected intermediates
   (ssGSEA scores, full-data predictions)
4. **Build consolidated notebook** section by section, testing each section
   against existing outputs for consistency
5. **Add NB02 GSEA cross-reference** (Sections 3 and 5.3)
6. **Full run** of consolidated notebook end-to-end
7. **Verify** outputs match or improve upon originals
8. **Update CLAUDE.md** data flow diagram and any references to old notebook numbers

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Upstream normalization changes alter results | You said to proceed without re-evaluation. If results look qualitatively different during build, flag but continue. |
| NB02 GSEA outputs not saved as intermediates | Load NB02's cluster profiles + marker expression; recompute GSEA summary inline if needed (lightweight). |
| Output renumbering breaks downstream | Search for all references to `04_*` outputs; update to `03_*`. Check `04_Deep_Dive_and_Clinical.ipynb` (which currently is NB04 in the revision plan but becomes NB04 in the new numbering). |
| Script extraction misses dependencies | Each script should be self-contained: own imports, own data loading from outputs/. Test in isolation before relying on their outputs. |

---

## 9. Resolved Decisions

1. **Naming:** `03_ML_and_Discordant_Biology.ipynb` -- confirmed.
2. **Revision plan:** `Revision Plan_260403.md` is superseded by this plan.
   Archived to `notebooks/archive/`.
3. **Equivocal scoring:** Included as Section 7 in the consolidated notebook.
   Full calibration/threshold detail remains in scripts.
4. **ssGSEA:** Pre-computed in `scripts/03_exploratory_multiclass_ml.py`.
   Notebook describes the method and points to the script for reproducibility.
5. **Feature reduction (hard constraint):** All ML uses curated gene panels
   or GSEA-derived features. No full-transcriptome models. NB02's unsupervised
   GSEA informs the curated panel (Section 2 feature selection).
