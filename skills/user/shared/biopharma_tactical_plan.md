# Biopharma Analyses -- Tactical Implementation Plan

**Date:** 2026-04-06
**Status:** Ready for implementation
**Audience:** LLM coding agent (Coder persona)

---

## General Instructions for All Analyses

### Project Setup

- All scripts go in `scripts/` with prefix `04_biopharma_`.
- All reports go in `reports/biopharma/` (create this directory).
- Every script should start with the same boilerplate:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_intermediate, save_intermediate, savefig, to_patient_id, setup_plotting
import pandas as pd
import numpy as np
```

- Load data from `outputs/` parquet files. Never re-derive what upstream notebooks
  already computed.
- ASCII only in all code and output. No Unicode arrows, em dashes, Greek letters,
  or mathematical symbols.
- Every script ends by writing a markdown report to `reports/biopharma/`.
- Reports use this structure:

```markdown
# [Analysis Title]

## Key Findings
[2-4 bullet points, the headline results]

## Methods
[Brief description of what was done]

## Results
[Tables, statistics, interpretation]

## Limitations
[Sample size, data gaps, caveats]

## Implications
[What this means for the biopharma narrative]
```

### Data Loading Recipes

These are the common data loading patterns. Copy the relevant ones into each script.

```python
# Multimodal cohort (clinical + RNA + CN, n=966)
mm = load_intermediate('02_multimodal_cohort')

# Analysis dataframe (multimodal + derived marker columns)
analysis_df = load_intermediate('02_analysis_df')
# Derived columns: ERBB2_expr, GRB7_expr, ESR1_expr, PGR_expr, MKI67_expr, EGFR_expr, ERBB3_expr

# Discordant cases (n=71, all types)
disc = load_intermediate('02_discordant_cases')
# Columns: pid, discordance_type, her2_composite, ERBB2_expr, erbb2_copy_number, HER2_ihc_score, HER2_fish_status, GRB7_expr

# Discordant dossier (n=35, IHC-/RNA-high only, with concordant model scores)
dossier = load_intermediate('03_discordant_dossier')
# Columns: pid, erbb2_copy_number, expr_ERBB2, GRB7_expr, conc_model_prob, consensus_score, provisional_subtype, classification

# Equivocal ML scores (n=28)
eq_scores = load_intermediate('03_equivocal_scores')
# Columns: pid, prob_L1-LR, prob_Random Forest, prob_XGBoost

# ML predictions for all patients (n=960)
ml_preds = load_intermediate('03_ml_predictions')
# Columns: pid, her2_composite, ml_prob_her2_positive, ml_pred_her2, prob_L1-LR, prob_Random Forest, prob_XGBoost

# Tumor expression (normalized, n=966 x ~17K genes)
tumor_norm = load_intermediate('01_tumor_norm')
# Gene columns stored separately:
import json
with open('outputs/01_gene_cols.json') as f:
    gene_cols = json.load(f)

# Normal tissue expression (n=112 x ~17K genes)
normal = load_intermediate('01_normal_raw_filtered')
# Has 'pid' column. 112 unique patients.

# Cleaned clinical data (full cohort, n=1108)
clin = pd.read_csv('data/brca_tcga_clinical_data_cleaned.csv')
# Patient ID column is 'Patient ID'

# ssGSEA pathway scores
ssgsea = load_intermediate('03_ssgsea_scores')
# Columns: pid, pathway_PI3K_AKT_MTOR_SIGNALING, pathway_MTORC1_SIGNALING, etc.
```

### Key Column Mappings

| What you need | Where it lives | Column name |
|---|---|---|
| Patient ID | All parquet files | `pid` |
| Patient ID | Cleaned CSV | `Patient ID` |
| HER2 composite label | `02_multimodal_cohort` | `her2_composite` (Positive/Negative/Equivocal) |
| ERBB2 RNA expression | `02_analysis_df` | `ERBB2_expr` (derived) or `ERBB2` (raw gene col) |
| ERBB2 copy number (GISTIC) | `02_analysis_df` or `02_discordant_cases` | `erbb2_copy_number` (0, 1, 2) |
| GRB7 expression | `02_analysis_df` | `GRB7_expr` |
| IHC score | `02_multimodal_cohort` | `HER2 ihc score` or `IHC Score` |
| FISH status | `02_multimodal_cohort` | `HER2 fish status` |
| HER2 test method | `brca_tcga_clinical_data_cleaned.csv` | `her2_test_method` |
| ER quantitative | `brca_tcga_clinical_data_cleaned.csv` | `er_allred_score`, `er_hscore`, `er_intensity`, `er_percent_positive`, `er_fmol_mg` |
| ER status | `02_multimodal_cohort` | `ER Status By IHC` |
| Cent17 | `02_multimodal_cohort` | `Cent17 Copy Number`, `HER2 cent17 ratio` |
| FGA | `02_multimodal_cohort` | `Fraction Genome Altered` |
| Overall survival | `02_multimodal_cohort` | `Overall Survival (Months)`, `Overall Survival Status` |
| Disease-free survival | `02_multimodal_cohort` | `Disease Free (Months)`, `Disease Free Status` |

### CN Stratification for Discordant Patients

The 35 IHC-/RNA-high discordant patients split into:
- **CN-high (CN >= 2):** 6 patients. Classification: "IHC-missed HER2+" -- likely
  false negatives by IHC, have genomic amplification.
- **CN-low (CN <= 1):** 29 patients. Mixed classifications: "Transcriptional HER2
  activation", "Moderate molecular HER2 signal", "Isolated ERBB2 elevation".
  Subdivide further by CN=1 (n=19) vs CN=0 (n=10) if sample size permits.

---

## Analysis #5a: Equivocal Concordance Table (PRIORITY 1)

**Script:** `scripts/04_biopharma_5a_equivocal_concordance.py`
**Report:** `reports/biopharma/5a_equivocal_concordance.md`

### Objective

Build a formal concordance table: RNA-predicted HER2 status vs. FISH outcome in
IHC 2+ patients. This is the single most important biopharma deliverable -- it
directly supports a CDx filing argument.

### Data Availability Reality Check

Before coding, understand the constraint: Among the 28 equivocal patients in the
multimodal cohort, FISH status is sparse:
- `HER2 fish status` = None: 20 patients
- `HER2 fish status` = Equivocal: 5 patients
- `HER2 fish status` = Indeterminate: 3 patients
- `HER2 fish status` = Positive or Negative: **0 patients**

This means a direct RNA-vs-FISH concordance table **cannot be built from the
multimodal cohort alone.** The full clinical dataset (n=1108) has 38 IHC 2+
patients, of which 15 have definitive FISH results (14 Negative, 1 Positive).
But these 15 may not have RNA data.

### Step-by-Step Instructions

1. **Identify all IHC 2+ patients across both datasets.**

   - From `02_multimodal_cohort.parquet`: filter where `her2_composite == 'Equivocal'`.
     Result: 28 patients with RNA data but almost no FISH.
   - From `brca_tcga_clinical_data_cleaned.csv`: filter where `IHC Score` contains "2"
     (use `_parse_ihc_score` logic or string matching -- watch for float 2.0 vs string "2+").
     Result: 38 patients, 15 with definitive FISH.

2. **Find the intersection: IHC 2+ patients who have BOTH RNA data and definitive FISH.**

   - Merge on patient ID (use `to_patient_id` for the clinical CSV's `Patient ID`,
     match against `pid` in multimodal cohort).
   - Filter to those where `HER2 fish status` is in ['Positive', 'Negative']
     (exclude None, Equivocal, Indeterminate).
   - **Expected N: very small, possibly 0-5.** Report the exact number.

3. **If N >= 5: Build the concordance table.**

   - For each patient in the intersection, get the RNA-predicted HER2 probability
     from `03_equivocal_scores.parquet` (columns: `prob_L1-LR`, `prob_Random Forest`,
     `prob_XGBoost`).
   - Compute an ensemble probability: mean of the three model probabilities.
   - Binarize at threshold 0.5: `RNA_predicted_HER2 = ensemble_prob >= 0.5`.
   - Build a 2x2 confusion matrix: RNA-predicted vs. FISH-actual.
   - Compute: sensitivity, specificity, PPV, NPV, accuracy, Cohen's kappa.
   - Use `scipy.stats` for confidence intervals (Wilson score interval for proportions).

4. **If N < 5: Pivot to the alternative approach.**

   Since we cannot build a powered concordance table, do the following instead:

   a. **RNA score distribution in equivocal patients (n=28).**
      - Load `03_equivocal_scores.parquet`.
      - Compute ensemble probability (mean of three models).
      - Create a histogram/density plot of ensemble probabilities.
      - Overlay the threshold at 0.5.
      - Report: how many equivocal patients would be reclassified as HER2+
        vs HER2- by the RNA model.

   b. **Biological validation of RNA reclassification.**
      - Split the 28 equivocal patients into RNA-predicted-positive (ensemble >= 0.5)
        and RNA-predicted-negative (ensemble < 0.5).
      - For each group, compute mean ERBB2 expression, mean GRB7 expression, and
        mean erbb2_copy_number.
      - Run a Mann-Whitney U test comparing ERBB2 expression between the two groups.
      - If the RNA-predicted-positive group has significantly higher ERBB2 AND GRB7,
        this supports the RNA model's biological validity.

   c. **Comparison with concordant positive/negative patients.**
      - From `02_multimodal_cohort.parquet`, get ERBB2 expression for:
        - Concordant Positive (her2_composite == 'Positive')
        - Concordant Negative (her2_composite == 'Negative')
        - Equivocal RNA-predicted-positive
        - Equivocal RNA-predicted-negative
      - Create a box plot or violin plot showing these four groups.
      - If equivocal-RNA-positive patients have ERBB2 levels overlapping with
        concordant positives, this is strong evidence for the CDx argument.

   d. **Available FISH data descriptive table.**
      - For the 15 IHC 2+ patients with FISH results (from full clinical),
        report their FISH outcomes in a simple table.
      - Note how many of these 15 also have RNA data (the intersection from step 2).
      - This demonstrates that FISH data sparsity is the bottleneck, not analytical
        capability.

5. **Figure generation.**
   - `fig_04_5a_equivocal_score_distribution.png`: Histogram of ensemble RNA
     probabilities for equivocal patients, with 0.5 threshold line.
   - `fig_04_5a_equivocal_erbb2_comparison.png`: Box/violin plot of ERBB2 expression
     across the four groups (concordant pos, concordant neg, equivocal-RNA-pos,
     equivocal-RNA-neg).
   - Use `savefig(fig, 'fig_04_5a_...')` from utils.

6. **Report writing.**

   The report should frame the finding as: "We demonstrate that an RNA-based model
   can resolve equivocal (IHC 2+) cases into biologically distinct HER2+ and HER2-
   subgroups, with expression profiles consistent with concordant positive and negative
   patients respectively. Direct FISH concordance validation was limited by FISH data
   availability in TCGA (N=X with both RNA and FISH). In a Tempus real-world dataset
   with paired RNA-seq and FISH results, this concordance analysis would be immediately
   executable."

   Include a table like:

   | Metric | Value | Note |
   |---|---|---|
   | IHC 2+ patients (multimodal) | 28 | Have RNA data |
   | IHC 2+ with definitive FISH | X | From full clinical |
   | IHC 2+ with RNA + FISH | Y | Intersection |
   | RNA-reclassified as HER2+ | Z | Ensemble prob >= 0.5 |
   | RNA-reclassified as HER2- | W | Ensemble prob < 0.5 |
   | ERBB2 expression (RNA-pos vs RNA-neg) | p=... | Mann-Whitney U |

---

## Analysis #3: Discordant Biology -- Normal Tissue and ER Pathway (PRIORITY 2)

**Script:** `scripts/04_biopharma_3_discordant_biology.py`
**Report:** `reports/biopharma/3_discordant_biology.md`

### Objective

Determine whether the non-amplified discordant group (CN <= 1) is driven by
ER/luminal co-regulation of ERBB2 or by independent HER2-pathway activation. This
is the mechanistic foundation for the "expand" play.

### Data Availability

- Normal tissue matched samples: 112 patients total, but only **4 of 35** IHC-/RNA-high
  discordant patients have matched normals. This is too few for a powered tumor-vs-normal
  comparison within discordant patients.
- Concordant HER2-Negative patients with matched normals: **67 patients.** This is the
  comparison baseline.
- ER quantitative scores: In `brca_tcga_clinical_data_cleaned.csv`, columns
  `er_allred_score`, `er_hscore`, `er_intensity`, `er_percent_positive`, `er_fmol_mg`.
  Coverage needs to be checked per analysis.

### Step-by-Step Instructions

#### Part A: Tumor-to-Normal ERBB2 Ratios

1. **Load tumor and normal expression data.**

   ```python
   tumor_norm = load_intermediate('01_tumor_norm')
   normal = load_intermediate('01_normal_raw_filtered')
   ```

2. **Identify patients with matched tumor-normal pairs.**

   - Get the set of pids in both `tumor_norm` and `normal`.
   - Expected: 112 patients.

3. **Compute tumor-to-normal ERBB2 ratio for each matched patient.**

   - For each matched pid, extract ERBB2 expression from tumor and normal.
   - Compute: `erbb2_ratio = tumor_ERBB2 - normal_ERBB2` (log-space, so subtraction
     = log ratio).
   - Also compute for GRB7 (amplicon co-member) and ESR1 (ER pathway marker).

4. **Stratify by patient group.**

   Define groups among the 112 matched patients:
   - Concordant Negative: `her2_composite == 'Negative'` AND not in discordant list
   - Discordant (if any of the 4 are present): `pid in discordant IHC-/RNA-high pids`
   - Concordant Positive: `her2_composite == 'Positive'`

5. **Statistical comparison.**

   - Compare ERBB2 tumor-to-normal ratios across groups using Kruskal-Wallis
     (if 3+ groups) or Mann-Whitney (if only 2 groups have sufficient N).
   - **Flag: n=4 discordant patients with normals is severely underpowered.**
     Report descriptive statistics (median, IQR) but do not over-interpret p-values.

6. **Figure:** Box plot of tumor-to-normal ERBB2 ratio by group.
   `fig_04_3a_tumor_normal_erbb2_ratio.png`

#### Part B: ER Pathway Correlation in Non-Amplified Discordant

7. **Define the gene panel for ER pathway correlation.**

   ```python
   er_pathway_genes = ['ESR1', 'FOXA1', 'GATA3', 'TFF1', 'TFF3', 'PGR',
                        'XBP1', 'AGR2', 'CA12', 'NAT1', 'SLC39A6']
   her2_pathway_genes = ['ERBB2', 'GRB7', 'STARD3', 'PGAP3', 'MIEN1',
                          'EGFR', 'ERBB3', 'ERBB4']
   proliferation_genes = ['MKI67', 'AURKA', 'CCNB1', 'TOP2A', 'BIRC5',
                           'MYBL2', 'CDK1', 'PLK1']
   ```

8. **Build a patient-level expression matrix for these genes.**

   - Load `01_tumor_norm.parquet` and subset to these genes (check which are present).
   - Merge with `02_discordant_cases.parquet` to identify discordant patients.
   - Define groups:
     - **Discordant CN-high (CN >= 2):** 6 patients
     - **Discordant CN-low (CN <= 1):** 29 patients
     - **Concordant Negative:** sample 50 randomly as comparison (or use all)
     - **Concordant Positive:** sample 50 randomly as comparison (or use all)

9. **Compute pairwise Spearman correlations within the CN-low discordant group.**

   - Specifically: ERBB2 vs ESR1, ERBB2 vs FOXA1, ERBB2 vs GATA3, ERBB2 vs PGR.
   - Use `scipy.stats.spearmanr`. Report rho and p-value for each pair.
   - If ERBB2 correlates positively with ESR1/FOXA1/GATA3 in the CN-low group,
     this supports ER-driven co-regulation.

10. **Compute the same correlations in concordant negative patients.**

    - This is the null expectation. In concordant negatives, ERBB2 should NOT correlate
      with ER pathway genes (both are low).
    - If the correlation is unique to the discordant CN-low group, the finding is
      specific and not an artifact.

11. **Interpretation framework heatmap.**

    - Create a heatmap with rows = gene groups (ER pathway, HER2 pathway, proliferation)
      and columns = patient groups (CN-high disc, CN-low disc, concordant neg, concordant pos).
    - Cell values = mean z-scored expression.
    - This visually shows the biology: CN-low discordant should cluster with luminal
      biology (high ER, high ERBB2, low proliferation) vs CN-high discordant clustering
      with HER2-enriched biology (high HER2, high proliferation, low ER).

12. **Figure:** `fig_04_3b_pathway_heatmap.png`

#### Part C: ER Quantitative Scores

13. **Merge ER quantitative scores from cleaned clinical data.**

    ```python
    clin = pd.read_csv('data/brca_tcga_clinical_data_cleaned.csv')
    er_cols = ['er_allred_score', 'er_hscore', 'er_intensity',
               'er_percent_positive', 'er_fmol_mg']
    ```

    - Merge on patient ID (clin uses `Patient ID`, parquet files use `pid`; both are
      12-character TCGA IDs, so direct string match works).

14. **Compare ER quantitative scores across groups.**

    - Groups: CN-high discordant, CN-low discordant, concordant negative, concordant positive.
    - For each ER metric, compute median and IQR per group.
    - Kruskal-Wallis test across groups (if N permits).
    - **Key hypothesis:** CN-low discordant patients should have HIGHER ER quantitative
      scores than concordant negatives, supporting the ER-driven biology narrative.

15. **Figure:** Grouped bar chart or box plots of ER quantitative scores by group.
    `fig_04_3c_er_quantitative.png`

#### Part D: Fraction Genome Altered (Quick Win)

16. **Compare FGA across discordant subgroups.**

    - FGA is in `02_multimodal_cohort.parquet` column `Fraction Genome Altered`.
    - Groups: CN-high discordant, CN-low discordant, concordant negative, concordant positive.
    - Mann-Whitney U test: CN-low discordant vs concordant negative.
    - **Interpretation:** Genomically quiet (low FGA) supports "normal-ish biology with
      incidental ERBB2 elevation." Genomically unstable (high FGA) supports
      "disease-driven biology."

17. **Figure:** Box plot of FGA by group. `fig_04_3d_fga_comparison.png`

18. **Report writing.**

    Structure the report around the interpretation framework:
    - High ERBB2 + high FOXA1 + high ESR1 = luminal biology with incidental ERBB2
      co-regulation -> favors endocrine therapy
    - High ERBB2 + low ESR1 + high proliferation = HER2-driven phenotype without
      amplification -> favors HER2-directed therapy

    Report the N explicitly for every comparison. Flag Part A (tumor-normal) as
    n=4 and therefore descriptive only.

---

## Analysis #2: Prevalence of Molecular ERBB2 Overexpression (PRIORITY 3)

**Script:** `scripts/04_biopharma_2_prevalence.py`
**Report:** `reports/biopharma/2_prevalence_estimation.md`

### Objective

Estimate the size of the "missed HER2+" population -- patients classified as
HER2-negative by IHC but showing molecular evidence of ERBB2 overexpression.
Frame as prevalence, not false-negative rate.

### Critical Framing Note

Do NOT use the phrase "false negative rate." IHC is the clinical ground truth;
there is no higher authority to define FN against. The defensible framing is:

> "Prevalence of molecular ERBB2 overexpression among IHC-negative patients."

The argument becomes: "X% of IHC-negative patients show molecular evidence that
questions their negative classification."

### Step-by-Step Instructions

1. **Define the denominator: all IHC-negative patients with RNA data.**

   ```python
   mm = load_intermediate('02_multimodal_cohort')
   ihc_neg = mm[mm['her2_composite'] == 'Negative']  # n=686
   ```

   But only those with RNA data (i.e., in the multimodal cohort with gene expression).
   The multimodal cohort already has RNA data, so all 686 qualify.

   Wait -- the multimodal cohort is 966 patients but the analysis_df (with derived
   expression columns) is also 966. Verify that ERBB2_expr is populated for all.

2. **Define the numerator: IHC-negative patients with RNA evidence of ERBB2 overexpression.**

   Use the same threshold that NB02 used for discordant identification.
   From `02_discordant_cases.parquet`, the IHC-/RNA-high type has 35 patients.

   But also compute alternative thresholds for sensitivity analysis:
   - Primary: NB02's threshold (95th percentile of negative ERBB2 expression)
   - Sensitivity 1: 90th percentile
   - Sensitivity 2: 99th percentile
   - Sensitivity 3: Mean + 2 SD of negative ERBB2 expression

3. **Compute prevalence at each threshold.**

   ```python
   neg_erbb2 = analysis_df[analysis_df['her2_composite'] == 'Negative']['ERBB2_expr']
   thresholds = {
       'p95 (primary)': neg_erbb2.quantile(0.95),
       'p90': neg_erbb2.quantile(0.90),
       'p99': neg_erbb2.quantile(0.99),
       'mean+2SD': neg_erbb2.mean() + 2 * neg_erbb2.std(),
   }
   ```

   For each threshold, count how many IHC-negative patients exceed it.
   Prevalence = count / total IHC-negative.

4. **Compute 95% confidence intervals for prevalence.**

   Use Wilson score interval:
   ```python
   from statsmodels.stats.proportion import proportion_confint
   ci_low, ci_high = proportion_confint(count, total, alpha=0.05, method='wilson')
   ```

5. **Stratify prevalence by CN status.**

   Among the "molecular ERBB2 overexpression" patients (using primary threshold):
   - CN >= 2: amplicon-driven (strongest case for missed HER2+)
   - CN = 1: intermediate
   - CN = 0: no amplification (weakest case, likely transcriptional)

   Report: "Of the X% with molecular overexpression, Y% also show genomic
   amplification, providing orthogonal evidence for HER2+ biology."

6. **Extrapolation to population.**

   - TCGA is not a random sample, so do not make point estimates for US breast
     cancer incidence. Instead:
   - Report: "If this prevalence (X%, 95% CI: [a%, b%]) holds in the general
     IHC-negative breast cancer population, and ~250,000 new breast cancers are
     diagnosed annually in the US with ~80% IHC-negative, this represents
     approximately N patients per year who may benefit from further molecular testing."
   - Use the CI bounds for a range estimate, not a point estimate.
   - **Caveat heavily:** TCGA ascertainment bias, single-institution effects, etc.

7. **Figure:**
   - `fig_04_2_prevalence_by_threshold.png`: Bar chart showing prevalence at each
     threshold, with CI error bars.
   - `fig_04_2_prevalence_cn_stratified.png`: Stacked bar showing CN-high vs CN-low
     breakdown of the molecular overexpression group.

8. **Report writing.**

   Lead with the primary prevalence estimate and CI. Then present the threshold
   sensitivity analysis to show robustness. Include the CN stratification to show
   that a subset has orthogonal genomic evidence. Close with the population
   extrapolation (heavily caveated).

---

## Analysis #5c: RNA Continuous Scoring for T-DXd Eligibility (PRIORITY 4)

**Script:** `scripts/04_biopharma_5c_tdxd_spectrum.py`
**Report:** `reports/biopharma/5c_tdxd_spectrum.md`

### Objective

Demonstrate that RNA provides a continuous quantitative score where IHC provides
only ordinal categories (0, 1+, 2+, 3+). Show that RNA score stratifies the
equivocal population into biologically distinct subgroups, setting up the hypothesis
that RNA-guided T-DXd eligibility could improve treatment selection.

### Context

T-DXd (trastuzumab deruxtecan) is approved for HER2-low breast cancer
(IHC 1+ or IHC 2+/FISH-). DESTINY-Breast04 showed benefit in the HER2-low
population, but the magnitude of benefit may vary with actual HER2 expression level.
RNA provides a continuous quantification that IHC cannot.

We cannot answer the treatment-benefit question with TCGA (no treatment data), but
we can show biological heterogeneity within the HER2-low zone.

### Step-by-Step Instructions

1. **Define the HER2-low population.**

   HER2-low = IHC 1+ OR (IHC 2+ AND FISH-negative).

   ```python
   mm = load_intermediate('02_multimodal_cohort')
   # Parse IHC scores carefully (float/string mismatch!)
   # IHC 1+ patients:
   ihc_1plus = mm[mm['IHC Score'].astype(str).isin(['1', '1.0', '1+'])]
   # IHC 2+ / FISH- patients:
   ihc_2plus_fish_neg = mm[
       (mm['IHC Score'].astype(str).isin(['2', '2.0', '2+'])) &
       (mm['HER2 fish status'] == 'Negative')
   ]
   her2_low = pd.concat([ihc_1plus, ihc_2plus_fish_neg]).drop_duplicates(subset='pid')
   ```

   Note: `IHC Score` may be mostly NaN in the multimodal cohort (27 of 28 equivocal
   are NaN). Use `HER2 ihc score` column instead if `IHC Score` is sparse.
   Also check `her2_composite` and construct using the classify_her2_spectrum function
   from utils.py if available.

   Alternative approach: use `classify_her2_spectrum(row)` from utils.py which
   produces HER2-0 / HER2-Low / HER2-Positive categories. Apply it to each row
   of the multimodal cohort.

2. **Get RNA-based continuous HER2 scores for these patients.**

   Two complementary scores:
   - **Raw ERBB2 expression:** From `02_analysis_df`, column `ERBB2_expr`.
   - **ML probability:** From `03_ml_predictions.parquet`, column `ml_prob_her2_positive`.
     This is the trained model's P(HER2+) -- a composite score integrating multiple
     features beyond just ERBB2.

3. **Show the continuous spectrum within HER2-low.**

   - Create a scatter plot: x = ERBB2 expression, y = ML probability, colored by
     IHC score (0 vs 1+ vs 2+/FISH-).
   - Overlay the equivocal patients (IHC 2+, including those without FISH) as
     distinct markers.
   - Show that within IHC categories, there is wide variation in RNA/ML scores.

4. **Biological characterization of RNA-stratified subgroups within HER2-low.**

   - Divide HER2-low patients into tertiles based on ERBB2 expression (or ML probability).
   - For each tertile, compute:
     - Mean expression of HER2 pathway genes (ERBB2, GRB7, EGFR, ERBB3)
     - Mean expression of proliferation markers (MKI67, AURKA, TOP2A)
     - Mean expression of ER pathway genes (ESR1, FOXA1, PGR)
   - Run Kruskal-Wallis across tertiles for each gene.
   - **Key finding to look for:** Do the highest-RNA-score HER2-low patients resemble
     HER2-positive patients more than they resemble low-scoring HER2-low patients?

5. **Comparison with HER2-0 and HER2-positive.**

   - Include HER2-0 (IHC 0) and HER2-positive (IHC 3+ or IHC 2+/FISH+) as reference
     groups.
   - Create a ridge plot or overlapping density plot showing ML probability distribution
     for: HER2-0, HER2-low (lower tertile), HER2-low (upper tertile), HER2-positive.
   - If the upper tertile of HER2-low overlaps with HER2-positive in RNA space, this
     supports the hypothesis that these patients may derive greater T-DXd benefit.

6. **ssGSEA pathway comparison.**

   - If `03_ssgsea_scores.parquet` has the relevant patients, merge pathway scores.
   - Compare pathway activity (PI3K/AKT/mTOR, E2F, G2M) across HER2-low tertiles.
   - Patients with higher pathway activity may represent a biologically distinct
     subgroup within HER2-low.

7. **Figures:**
   - `fig_04_5c_her2low_rna_spectrum.png`: Scatter/strip plot of ERBB2 expression
     by IHC category.
   - `fig_04_5c_her2low_ml_density.png`: Overlapping density plot of ML probability
     by HER2 spectrum group.
   - `fig_04_5c_her2low_pathway_tertiles.png`: Heatmap of pathway scores by tertile.

8. **Report writing.**

   Frame as: "Within the HER2-low population, RNA-based quantification reveals a
   continuous spectrum of HER2 biology that IHC ordinal categories obscure. The
   upper tertile of HER2-low patients shows pathway activation profiles similar to
   HER2-positive patients, suggesting that RNA-guided stratification could identify
   patients most likely to benefit from T-DXd and related ADC therapies."

   Note that this is hypothesis-generating; treatment-benefit validation requires
   RWD with linked treatment and outcome data (Tempus dataset).

---

## Analysis #4a: Clinical Correlates (PRIORITY 5)

**Script:** `scripts/04_biopharma_4a_clinical_correlates.py`
**Report:** `reports/biopharma/4a_clinical_correlates.md`

### Objective

Characterize the discordant IHC-/RNA-high population clinically. Identify features
that distinguish them from concordant negatives -- useful for targeting a prospective
study or building inclusion criteria.

### Step-by-Step Instructions

1. **Define groups.**

   ```python
   disc = load_intermediate('02_discordant_cases')
   ihc_neg_rna_high = disc[disc['discordance_type'] == 'IHC-/RNA-high']  # n=35
   cn_high_disc = ihc_neg_rna_high[ihc_neg_rna_high['erbb2_copy_number'] >= 2]  # n=6
   cn_low_disc = ihc_neg_rna_high[ihc_neg_rna_high['erbb2_copy_number'] <= 1]   # n=29
   ```

   Concordant negative = her2_composite == 'Negative' AND pid NOT in discordant list.
   Concordant positive = her2_composite == 'Positive'.

2. **Clinical variables to compare.**

   From `02_multimodal_cohort.parquet` (merged with cleaned clinical CSV where needed):

   | Variable | Column | Type | Test |
   |---|---|---|---|
   | Age at diagnosis | `Diagnosis Age` | Continuous | Kruskal-Wallis |
   | AJCC stage | `Neoplasm Disease Stage American Joint Committee on Cancer Code` | Ordinal | Chi-squared or Fisher's exact |
   | T stage | `American Joint Committee on Cancer Tumor Stage Code` | Ordinal | Chi-squared/Fisher |
   | N stage | `Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code` | Ordinal | Chi-squared/Fisher |
   | ER status | `ER Status By IHC` | Binary | Chi-squared/Fisher |
   | PR status | `PR status by ihc` | Binary | Chi-squared/Fisher |
   | Histologic type | `Cancer Type Detailed` | Categorical | Fisher's exact |
   | FGA | `Fraction Genome Altered` | Continuous | Kruskal-Wallis |
   | ER quantitative (allred) | `er_allred_score` (from cleaned CSV) | Ordinal | Kruskal-Wallis |
   | ER quantitative (H-score) | `er_hscore` (from cleaned CSV) | Continuous | Kruskal-Wallis |

3. **Build a "Table 1" style summary.**

   For each variable, report:
   - N (non-missing) per group
   - Median (IQR) for continuous, N (%) for categorical
   - Test statistic and p-value for comparison across groups
   - Use `scipy.stats.kruskal` for continuous, `scipy.stats.fisher_exact` (2x2) or
     `scipy.stats.chi2_contingency` (larger) for categorical.
   - Apply Benjamini-Hochberg FDR correction across all tests.

4. **Polysomy 17 check.**

   - Column `Cent17 Copy Number` in multimodal cohort. Sparsely populated (likely
     only 1-2 non-null even in equivocal group).
   - For any discordant patient with non-null Cent17, report the value.
   - Also check `HER2 cent17 ratio`.
   - If any IHC+/RNA-low discordant cases (from `02_discordant_cases`, type
     `IHC+/RNA-low`, n=8) have Cent17 data, elevated Cent17 would indicate polysomy
     rather than true amplification. Report descriptively.

5. **ER scoring method variation check.**

   - Column `er_scoring_method_detail` in cleaned clinical CSV.
   - Tabulate: does the ER scoring method vary across discordant vs concordant groups?
   - If it does, this is a potential confounder for Analysis #3's ER pathway correlation.
   - Report as a cross-tabulation.

6. **Figure:**
   - `fig_04_4a_table1_forest.png`: Forest plot of effect sizes (standardized mean
     differences or odds ratios) for each clinical variable, discordant vs concordant
     negative. Visually shows which variables differ.

7. **Report writing.**

   Present as a clinical characterization. Highlight any statistically significant
   differences after FDR correction, but also highlight clinically meaningful
   differences that may not reach significance due to small N. Include the Table 1
   as a formatted markdown table.

---

## Analysis #1: HER2 Testing Method as Confounder (PRIORITY 6)

**Script:** `scripts/04_biopharma_1_test_method.py`
**Report:** `reports/biopharma/1_test_method_confounder.md`

### Objective

Assess whether the HER2 testing method (IHC platform/protocol) is a confounder in
discordant cases -- i.e., were certain testing methods more likely to produce
false-negative IHC results?

### Data Availability

`her2_test_method` in `brca_tcga_clinical_data_cleaned.csv`:
- NaN: 1018 patients
- Non-null: 90 patients total
- Of 35 IHC-/RNA-high discordant patients: only **4 have test method data.**

This analysis is severely data-limited. Treat as due diligence.

### Step-by-Step Instructions

1. **Load and merge.**

   ```python
   clin = pd.read_csv('data/brca_tcga_clinical_data_cleaned.csv')
   disc = load_intermediate('02_discordant_cases')
   ihc_neg_rna_high_pids = disc[disc['discordance_type'] == 'IHC-/RNA-high']['pid'].tolist()
   ```

   Merge clin (using `Patient ID`) with discordant status.

2. **Descriptive table of test methods.**

   Among all patients with `her2_test_method` populated (n~90):
   - Cross-tabulate test method vs her2_composite (Positive/Negative/Equivocal).
   - Cross-tabulate test method vs discordant status (discordant Y/N).

3. **For the 4 discordant patients with test method data:**

   - List their test methods, IHC scores, ERBB2 expression, and CN.
   - Note whether any pattern is visible (e.g., all tested with an older method).

4. **Chi-squared or Fisher's exact test:**

   - Test whether test method distribution differs between discordant and concordant
     negative patients (among those with method data).
   - **Expect:** underpowered, likely non-significant.

5. **Report writing.**

   Frame as: "Testing method data was available for only 4/35 (11%) of discordant
   patients, precluding a powered confounder analysis. Among the 90 patients with
   test method annotations, we observed [describe distribution]. This analysis
   would be substantially more informative in a Tempus dataset with standardized
   testing metadata."

   Include the cross-tabulation tables even if N is small.

---

## Analysis #5b: Multi-Modal Concordance Tiers (PRIORITY 7)

**Script:** `scripts/04_biopharma_5b_concordance_tiers.py`
**Report:** `reports/biopharma/5b_concordance_tiers.md`

### Objective

Among equivocal patients, stratify by agreement across modalities (RNA, CN, FISH)
to identify high-confidence vs uncertain reclassifications.

### Step-by-Step Instructions

1. **Build a per-patient multimodal status table for equivocal patients.**

   For each of the 28 equivocal patients:
   - RNA call: `prob >= 0.5` from `03_equivocal_scores.parquet` (ensemble of 3 models) -> Positive/Negative
   - CN call: `erbb2_copy_number >= 2` from `02_multimodal_cohort` -> Amplified/Not amplified
   - FISH call: `HER2 fish status` from `02_multimodal_cohort` -> Positive/Negative/NA

   To get `erbb2_copy_number` for equivocal patients, load `02_multimodal_cohort`
   and use column `erbb2_copy_number` (GISTIC-derived, values 0/1/2).

2. **Assign concordance tiers.**

   - **Tier 1 (High confidence HER2+):** RNA+ AND (CN amplified OR FISH+)
   - **Tier 2 (RNA-only HER2+):** RNA+ AND CN not amplified AND FISH not positive
   - **Tier 3 (Concordant HER2-):** RNA- AND CN not amplified AND FISH not positive
   - **Tier 4 (Mixed signals):** Any other combination (e.g., RNA- but CN amplified)

3. **Characterize each tier biologically.**

   For each tier, compute:
   - Mean ERBB2 expression, GRB7 expression
   - Mean ML probability (from equivocal scores)
   - ER status distribution
   - FGA

4. **Figure:**
   - `fig_04_5b_concordance_tiers.png`: Grouped bar or dot plot showing ERBB2
     expression, GRB7 expression, and ML probability by tier.

5. **Report writing.**

   Report the N per tier. Emphasize Tier 1 as the most actionable (multiple
   modalities agree -> highest confidence for reclassification). Note Tier 4 as
   candidates for additional testing in a clinical setting.

---

## Analysis #4b: Survival Analysis (PRIORITY 8)

**Script:** `scripts/04_biopharma_4b_survival.py`
**Report:** `reports/biopharma/4b_survival.md`

### Objective

Kaplan-Meier survival comparison for discordant vs concordant HER2-negative patients.
Severely underpowered -- frame as hypothesis-generating.

### Step-by-Step Instructions

1. **Prepare survival data.**

   ```python
   mm = load_intermediate('02_multimodal_cohort')
   ```

   - Time: `Overall Survival (Months)` -- all 966 patients have this.
   - Event: `Overall Survival Status` -- parse to binary (1 = dead, 0 = alive/censored).
     Typical TCGA encoding: "1:DECEASED" = 1, "0:LIVING" = 0. Check actual values.
   - Also: `Disease Free (Months)` and `Disease Free Status` for DFS analysis.

2. **Define groups.**

   - Concordant Negative (not discordant, her2_composite == 'Negative')
   - Discordant IHC-/RNA-high (n=35, further split by CN if N permits)
   - Concordant Positive (reference)

3. **Kaplan-Meier curves.**

   Use `lifelines`:
   ```python
   from lifelines import KaplanMeierFitter
   from lifelines.statistics import logrank_test
   ```

   - Plot KM curves for OS, comparing concordant negative vs discordant.
   - Report median OS per group (if estimable) and 5-year OS rate.
   - Log-rank test for concordant negative vs discordant.

4. **CN-stratified survival (if N permits).**

   - Split discordant into CN-high (n=6) and CN-low (n=29).
   - Only if CN-low has >= 15 events should you plot separate curves.
   - Otherwise, keep as one discordant group.

5. **Flag underpowering.**

   - n=35 discordant, likely ~5-10 events. Log-rank will struggle to reach significance.
   - Report the hazard ratio with 95% CI (Cox proportional hazards, univariate):
     ```python
     from lifelines import CoxPHFitter
     ```
   - Report the CI width as evidence of underpowering.

6. **Figures:**
   - `fig_04_4b_km_os.png`: KM curves for OS by group.
   - `fig_04_4b_km_dfs.png`: KM curves for DFS by group (if DFS data is sufficient).

7. **Report writing.**

   Lead with: "This analysis is hypothesis-generating. With n=35 discordant patients
   and limited events, we are severely underpowered to detect clinically meaningful
   survival differences."

   Report the HR and CI. If the point estimate is interesting (e.g., HR > 1.5 or < 0.7),
   note it as worth investigating in a larger Tempus cohort.

---

## Analysis #5d: Equivocal Demographics (PRIORITY 9)

**Script:** `scripts/04_biopharma_5d_equivocal_demographics.py`
**Report:** `reports/biopharma/5d_equivocal_demographics.md`

### Objective

Compare clinical/demographic features of equivocal-resolved-positive vs
equivocal-resolved-negative. Identify enrichments that would help target a
prospective study.

### Step-by-Step Instructions

1. **Define groups using RNA reclassification from Analysis #5a.**

   - Equivocal-resolved-positive: ensemble RNA probability >= 0.5
   - Equivocal-resolved-negative: ensemble RNA probability < 0.5

2. **Compare the same clinical variables as in Analysis #4a:**

   Age, stage, grade, ER status, PR status, histologic type, FGA.

3. **Table 1 for equivocal subgroups.**

   Same format as #4a but with equivocal-resolved-positive vs equivocal-resolved-negative.
   n=28 total, so expect ~10-15 per group. Flag as underpowered throughout.

4. **Report writing.**

   Brief. If any variable shows enrichment, note it as a candidate criterion for
   prospective study design. If nothing distinguishes the groups clinically, that
   itself is a finding: "RNA-based reclassification identifies a population that is
   clinically indistinguishable from the broader equivocal group, supporting the need
   for molecular rather than clinical selection criteria."

---

## Execution Order

The analyses have some shared data loading but are otherwise independent. Execute
in priority order:

1. **#5a** (equivocal concordance) -- fastest, highest impact
2. **#3** (discordant biology) -- deepest mechanistic insight
3. **#2** (prevalence) -- population sizing
4. **#5c** (T-DXd spectrum) -- strategic value
5. **#4a** (clinical correlates) -- clinical characterization
6. **#1** (test method) -- due diligence, quick
7. **#5b** (concordance tiers) -- adds depth
8. **#4b** (survival) -- hypothesis-generating
9. **#5d** (equivocal demographics) -- if time permits

Analyses #5a, #2, and #1 can run in parallel (no dependencies). Analyses #3 and
#4a share group definitions but are otherwise independent. Analysis #5b depends
on #5a's ensemble scoring. Analysis #5d depends on #5a's reclassification.
Analysis #4b is fully independent.

---

## Report Synthesis (Out of Scope)

After all individual reports are written to `reports/biopharma/`, a separate task
will synthesize them into a cohesive biopharma deliverable. That synthesis is outside
the scope of this plan.
