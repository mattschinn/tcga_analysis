# TSS Batch Effect Assessment: Methodology and Findings

**Date:** April 2026  
**Script:** `src/01s_tss_batch_assessment.py`  
**Outputs:** `outputs/01s_*.parquet`, `outputs/figures/fig_01s_*`

---

## Methodology

### Phase 1: Quantification

**Objective:** Measure how much transcriptomic variance is attributable to tissue source
site (TSS) versus biological variables (HER2, ER status).

**Approach:**
- Collapsed 40 TSS categories to 22 groups (19 rare sites with <5 samples merged into
  "Other").
- Ran PCA on standardized log2-normalized expression (17,637 genes, 1,093 tumor samples).
  Top 10 PCs capture 42.2% of total variance.
- Computed eta-squared (SS_between / SS_total) for each PC against three grouping
  variables: TSS (collapsed), HER2 composite label, and ER status by IHC.
- Ran vectorized one-way ANOVA across all 17,637 genes to identify genes most variable
  by TSS. Reported F-statistics, p-values, and per-gene eta-squared.

### Phase 2: Correction

**Objective:** Remove TSS-driven expression variation while preserving HER2 and ER
biological signal.

**Method:** Linear regression with protected covariates, solved as a single matrix
least-squares problem across all genes simultaneously.

For each gene, the model is:

```
expression = intercept + TSS_dummies + HER2_dummies + ER_binary + residual
```

Design matrix: 26 columns (1 intercept + 21 TSS dummies + 4 protected covariates:
3 HER2 levels + ER). Full rank confirmed (rank = 26).

Corrected expression = original - TSS_contribution, where TSS_contribution =
X_tss @ B_tss. This subtracts only the site-specific betas while retaining the
intercept, HER2, and ER contributions.

**Why not ComBat:** TSS is confounded with HER2 status (chi-squared = 177.9,
p = 3.73e-10). ComBat assumes batch is orthogonal to biology; violating this
assumption would attenuate the HER2 signal we are trying to detect.

### Phase 3: Validation

Re-ran PCA on corrected expression. Computed eta-squared before and after correction
for both TSS and HER2, confirming that TSS signal was removed without attenuating
HER2 signal.

---

## Key Findings

### 1. TSS is a substantial but non-dominant confounder

| PC  | TSS eta2 | HER2 eta2 | ER eta2  | Interpretation |
|-----|----------|-----------|----------|----------------|
| PC1 | **0.223** | 0.001    | 0.042    | TSS-dominated; not biologically informative |
| PC2 | 0.035    | 0.002     | **0.532** | ER-dominated (expected) |
| PC3 | **0.103** | 0.005    | 0.015    | TSS-dominated |
| PC4 | 0.086    | **0.076** | 0.007    | Mixed TSS/HER2 |
| PC5 | 0.076    | 0.006     | 0.010    | Moderate TSS |
| PC6 | 0.091    | 0.015     | 0.000    | TSS-dominated |
| PC7 | 0.042    | 0.006     | 0.012    | Low signal |
| PC8 | 0.023    | **0.053** | 0.002    | HER2-associated |
| PC9 | 0.053    | 0.030     | 0.014    | Mixed |
| PC10| 0.085    | 0.003     | 0.007    | TSS-dominated |

**TSS explains >10% of variance on PC1 and PC3.** These are the two largest TSS
effects. Importantly, neither PC1 nor PC3 loads heavily on HER2 (eta2 < 0.005),
meaning TSS and HER2 occupy largely different PCA axes. This is favorable for
correction: removing TSS signal is unlikely to distort HER2-related biology.

PC4 has the most HER2 signal (eta2 = 0.076) and moderate TSS (eta2 = 0.086). This
is the axis where confounding matters most.

### 2. ERBB2 expression is minimally affected by TSS

ERBB2 ranks 15,530th out of 17,637 genes by TSS F-statistic. Its TSS eta-squared
is 0.026 (2.6% of ERBB2 variance is TSS-attributable). By contrast, HER2 status
explains 40.3% of ERBB2 variance. **TSS is not a meaningful confounder for ERBB2
specifically.**

The top 5 most TSS-variable genes are: NACA2, SNRNP70, CCDC130, WASH7P, ZNF513.
None of these are HER2 pathway genes or known cancer drivers. This suggests TSS
variation is driven by housekeeping/technical genes rather than biology relevant to
this project.

### 3. Correction effectively removes TSS signal

| PC  | TSS eta2 Before | TSS eta2 After | Delta   |
|-----|-----------------|----------------|---------|
| PC1 | 0.2233          | 0.0144         | -0.209  |
| PC2 | 0.0347          | 0.0203         | -0.014  |
| PC3 | 0.1030          | 0.0109         | -0.092  |
| PC4 | 0.0862          | 0.0044         | -0.082  |
| PC5 | 0.0756          | 0.0010         | -0.075  |
| PC6 | 0.0907          | 0.0032         | -0.088  |
| PC7 | 0.0424          | 0.0010         | -0.041  |
| PC8 | 0.0229          | 0.0028         | -0.020  |
| PC9 | 0.0530          | 0.0034         | -0.050  |
| PC10| 0.0852          | 0.0006         | -0.085  |

**TSS eta-squared drops to <2% on all PCs after correction.** The largest residual
TSS signal is on PC2 (2.0%), which is the ER-dominated axis -- some TSS-ER confounding
remains, which is expected and acceptable.

### 4. HER2 signal is preserved through correction

| PC  | HER2 eta2 Before | HER2 eta2 After | Delta   |
|-----|------------------|-----------------|---------|
| PC4 | 0.0762           | 0.0862          | +0.010  |
| PC8 | 0.0531           | 0.0512          | -0.002  |
| PC9 | 0.0303           | 0.0407          | +0.010  |

**HER2 signal on its primary PCs (PC4, PC8, PC9) is preserved or slightly
strengthened after TSS correction.** This confirms the protected-covariate approach
works as intended: removing batch without attenuating the biology of interest.

### 5. Discordant cases are distributed across sites

Discordant cases (n=69) are not concentrated in a single site:
- A2: 10 cases
- OL: 6
- E9: 5
- E2: 5
- D8: 5

This spread across multiple TSS indicates that discordance is biological, not a
single-site artifact. **No single site dominates the discordant population.**

---

## Interpretation and Implications for Downstream Analyses

### Clustering (NB02/03)
TSS was the dominant signal on PC1 (22.3% of variance) before correction. Any
unsupervised clustering on uncorrected data risks producing clusters that partially
reflect sequencing center rather than tumor biology. **Recommendation:** Run
clustering on both corrected and uncorrected data and report concordance. If >90%
of samples maintain their cluster assignment, the result is robust.

### Supervised ML (NB03)
ERBB2 itself is minimally affected by TSS (eta2 = 0.026), and the top TSS-variable
genes are not cancer-relevant. For HER2 classification specifically, TSS correction
is unlikely to change AUC meaningfully. **Recommendation:** Compare SHAP top features
between corrected and uncorrected models. If a SHAP feature drops out after correction,
it was batch-driven.

### ERBB2 Expression Analysis (NB02)
With only 2.6% of ERBB2 variance attributable to TSS, the violin plots and ROC
analyses in NB02 are reliable on uncorrected data. No correction needed for
ERBB2-specific analyses.

### Discordant Cases (NB02/04)
Discordant cases are geographically distributed, not concentrated. This supports
their biological validity. However, the TSS x HER2 confounding (p = 3.73e-10)
means that HER2 label prevalence differs by site -- some sites may systematically
under-report HER2 positivity, creating apparent discordance. **Recommendation:**
Report per-site HER2 prevalence alongside discordant case tables.

---

## Figures

- `outputs/figures/fig_01s_tss_eta_heatmap.png` -- eta-squared heatmap showing
  TSS, HER2, and ER variance explained per PC, before and after correction.
  Key visual: TSS row goes from warm (high) to cool (low) after correction; ER
  and HER2 rows remain unchanged.

- `outputs/figures/fig_01s_pca_before_after.png` -- Four-panel PCA scatter. Top row
  colored by TSS (site-specific clustering diminishes after correction). Bottom row
  colored by HER2 (separation persists after correction).

---

## Limitations

1. **Protected-covariate approach depends on known confounders.** We protect HER2
   and ER, but other biology confounded with TSS (e.g., race, stage) may be partially
   removed. This is inherent to any regression-based batch correction.

2. **Rare sites collapsed to "Other."** 19 sites with <5 samples are lumped together.
   Their batch effects are not individually correctable but are absorbed into the
   "Other" group estimate.

3. **Label missingness bias is not addressed.** TSS drives HER2-label missingness
   (chi2 = 464.66, p ~ 10^-73). The labeled training set is not representative of
   all sites. This is a cohort composition issue, not an expression correction issue.

4. **Linear model assumes additive batch effects.** Non-linear site effects (e.g.,
   different dynamic ranges across sites) would not be captured. However, for
   log-transformed data, additive correction is standard and well-justified.
