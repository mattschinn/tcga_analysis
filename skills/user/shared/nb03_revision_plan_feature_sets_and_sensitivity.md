# NB03 Revision Plan: Feature Set Comparison & Concordant Threshold Sensitivity

**Date:** 2026-04-06
**Status:** Part B complete; Parts A and C proposed
**Triggered by:** Analyst review findings (nb03_analyst_review.md)
**Affects:** `notebooks/03_ML_and_Discordant_Biology.ipynb`, `/notebooks` folder cleanup

---

## Part A: Incorporate 3-Feature-Set Comparison into NB03

### Problem

The analyst review (Section 2) discusses a 3x3 model comparison across three
feature sets (ERBB2-only, Curated Panel, Pathway Scores). This comparison
exists in NB03a (`notebooks/03a_Machine_Learning.ipynb`, cells 10-21) and its
results are in `outputs/03_model_comparison.parquet`. But the consolidated NB03
only trains on the curated panel -- it never builds the ERBB2-only or
pathway-score feature sets and therefore cannot show the ERBB2 dominance
finding (AUC 0.840 vs 0.852, delta = 0.012).

This is an important result for the scientific argument:
- It quantifies how much signal ERBB2 alone carries (98% of AUC)
- It shows pathway scores *underperform* gene-level features (AUC 0.839)
- The small delta motivates the question: if ERBB2 dominates, why do
  discordant patients (high ERBB2 but IHC-) get missed?

### What to add to NB03 Section 2

**New cells between current cells 5-6 (feature construction) and cell 7
(markdown header for 3-model comparison):**

1. **Feature Set A: ERBB2-only baseline** (~10 lines)
   - 3 features: `expr_ERBB2`, `erbb2_copy_number`, `er_positive`
   - This is a trivial construction from columns already in `ml_df`

2. **Feature Set C: ssGSEA Pathway Scores** (~30 lines)
   - Load pre-computed `03_ssgsea_scores.parquet` from
     `scripts/03_exploratory_multiclass_ml.py`
   - Select top ~30 Hallmark pathway scores + CN + ER/PR
   - If ssGSEA scores not available (script not run), skip gracefully with
     a print warning -- the ERBB2-only vs Curated comparison alone is still
     valuable

3. **Expand model comparison cell** (modify current cell 8)
   - Loop over 3 feature sets (A: ERBB2-only, B: Curated Panel, C: Pathway
     Scores) x 3 models
   - Store all results in `03_model_comparison.parquet`
   - Add a comparison summary table (printed) and heatmap figure

4. **Interpretation markdown cell** (new, ~5 lines)
   - ERBB2 dominance finding: delta = 0.012 means ERBB2 is ~98% of signal
   - Pathway scores dilute the sharp ERBB2 signal
   - But the curated panel's marginal improvement matters for discordant
     biology interpretation, not for overall classification
   - This frames the pivot to Section 4 (concordant-only model)

**ROC curves cell** (modify current cell 9):
- Show one ROC per feature set (best model for each), not one per model

### What NOT to change

- Section 4 (concordant-only model) still uses only the curated panel.
  The ERBB2-only and pathway score feature sets are for the binary
  classification comparison only.
- Sections 5-10 are unaffected.
- The curated gene panel definition (GENE_SETS dict in cell 3) is unchanged.

### Estimate

~4 new/modified cells, ~80-100 lines of code. The logic is straightforward
since the ERBB2-only feature set is trivial and ssGSEA scores are pre-computed.

---

## Part B: Clean Up /notebooks -- COMPLETE (2026-04-06)

Archived: 00_Exploratory-draft, 03a, 03b, 04_Discordant_Deep_Dive,
04_Deep_Dive_and_Clinical (.ipynb + .py), 04_discordant_deep_dive_plan.md.

```
notebooks/                            # current state
  01_QC_and_Normalization.ipynb
  01_QC_and_Normalization.py
  02_HER2_Identification_and_Subsets.py
  02a_HER2_Identification_and_Subsets.ipynb
  03_ML_and_Discordant_Biology.ipynb   # to be expanded with Part A
  03_Machine_Learning.py
  archive/
    (all old notebooks + plans)
```

---

## Part C: Concordant Threshold Sensitivity Analysis

### Motivation

The concordant-only model (NB03 Section 4) is the centerpiece of the
scientific argument. It defines "concordant" using three thresholds:

| Parameter | Current Value | What it controls |
|-----------|--------------|-----------------|
| Concordant-positive ERBB2 expression floor | `pos_expr.quantile(0.25)` | Excludes IHC+ patients with low ERBB2 expression |
| Concordant-positive CN floor | `CN >= 1` | Excludes IHC+ patients without copy-number gain |
| Concordant-negative ERBB2 expression ceiling | `neg_expr.quantile(0.75)` | Excludes IHC- patients with elevated ERBB2 |

These thresholds are reasonable defaults but arbitrary. If the CN-stratified
discordant biology finding (the paper's main result) is sensitive to small
threshold changes, the argument weakens. If it is robust, that strengthens
the claim.

### Design: Three threshold configurations

Rather than a full grid sweep (which would be excessive for ~185 patients),
test three named configurations that represent meaningful alternatives:

| Config | Pos ERBB2 floor | Pos CN floor | Neg ERBB2 ceiling | Intent |
|--------|----------------|-------------|-------------------|--------|
| **Strict** (current) | 25th pctl of pos | CN >= 1 | 75th pctl of neg | Current default |
| **Relaxed** | 10th pctl of pos | CN >= 0 | 90th pctl of neg | Wider concordant set: more training data, but noisier |
| **Stringent** | 40th pctl of pos | CN >= 2 | 50th pctl of neg (median) | Narrower concordant set: cleaner training, fewer samples |

### What to measure at each threshold

For each configuration, record:

1. **Concordant set sizes** -- N concordant-positive, N concordant-negative,
   N excluded (discordant + borderline)

2. **Concordant-only model CV AUC** -- does the model still perform well
   when the training population shifts?

3. **Discordant patient scoring** -- how do the 35 discordant patients'
   concordant-model probabilities change?

4. **CN-stratified group assignment stability** (the key metric):
   - How many of the 6 CN=2 patients remain classified as "IHC-missed HER2+"?
   - How many of the 29 CN<=1 patients shift between "Transcriptional HER2
     activation" / "Moderate signal" / "Isolated elevation"?
   - Does the fundamental CN=2 vs CN<=1 biological distinction hold?

5. **Consensus score distribution shift** -- do the score distributions
   for each classification tier overlap more or less under different thresholds?

### What NOT to test

- The CN threshold for the amplified/non-amplified split (CN=2 vs CN<=1)
  is a GISTIC definition, not a modeling choice. Do not vary it.
- The consensus score classification cutoffs (0.3, 0.4, GRB7 > 10.0) are
  downstream of the concordant model. They could be varied too, but that
  is a separate concern. Keep them fixed for this analysis so we isolate
  the concordant-definition effect.

### Where this lives

**Script:** `scripts/03_concordant_threshold_sensitivity.py`
**Output:** `outputs/03_threshold_sensitivity.parquet`

The script is self-contained: loads intermediates from `outputs/`, runs the
3 configs, saves a summary table. Same pattern as the other exploratory
scripts.

**NB03 reference (Section 4):** One new markdown cell after the concordant
model results, loading and displaying the summary table from the parquet.
Brief interpretation: "CN-stratified finding is robust across threshold
definitions (see `scripts/03_concordant_threshold_sensitivity.py`)."

### Estimate

Script: ~80-120 lines. Most logic is lifted from NB03 cells 19-21
(concordant model training/scoring), wrapped in a loop over configs.

NB03 change: 1 markdown cell + ~5 lines of code to load and display the
summary table.

---

## Execution Order

1. ~~**Part B** (cleanup).~~ DONE.
2. **Part A** (feature set expansion). Edit NB03 `.ipynb` directly --
   add/modify cells in Section 2 for 3-feature-set comparison.
3. **Part C** (sensitivity analysis). Write
   `scripts/03_concordant_threshold_sensitivity.py`. Run it to produce
   `outputs/03_threshold_sensitivity.parquet`. Then add a brief reference
   cell in NB03 Section 4.
4. **Re-run NB03 end-to-end** to verify all outputs.
5. **Update CLAUDE.md** data flow diagram and output list.

---

## Resolved Questions

1. **04_Deep_Dive_and_Clinical.ipynb** -- archived (2026-04-06). Superseded
   by NB03 consolidation. Moved to `notebooks/archive/` along with its .py
   export.

2. **00_Exploratory-draft_QC-ML.ipynb** -- archived (2026-04-06). Original
   exploratory draft. Moved to `notebooks/archive/`.

## Design Decisions

### NB03 edits: direct .ipynb, not build script

Per user direction, Parts A and C edit the notebook `.ipynb` directly
rather than going through `scripts/build_consolidated_notebook.py`. The
build script is a one-time consolidation tool; ongoing edits are direct.

### Sensitivity analysis: script, not notebook

`scripts/03_concordant_threshold_sensitivity.py`, not inline in NB03.

Rationale:
- The sensitivity analysis is confirmatory, not part of the discovery
  narrative. It answers "is the finding robust?" -- a supporting claim.
- Inserting a threshold sweep between the concordant model (Section 4) and
  discordant biology (Section 5) interrupts the story arc.
- Matches the existing pattern: `scripts/03_exploratory_binary_ml.py` and
  `scripts/03_exploratory_multiclass_ml.py` hold supplemental analyses that
  NB03 references without inlining.
- NB03 Section 4 gets a brief markdown cell + ~5 lines loading a summary
  table from `outputs/03_threshold_sensitivity.parquet`.
