# TSS Batch Effect: Strategy and Shared Context

**Status:** Executed (April 2026). See `tss_batch_findings.md` for full results.  
**Owner:** Mat Schinn  
**Supports:** Notebook 01 (QC), downstream clustering and ML  
**Script:** `src/01s_tss_batch_assessment.py`

---

## What We Know

From the NB00 exploratory draft and NB01 Section 3.7:

1. **TSS is associated with 8 of 10 top PCs** (PCs 1, 3, 4, 5, 6, 7, 9, 10 at p < 0.01).
   This is not a single-axis confound -- it leaks into nearly every dimension of the
   transcriptome.

2. **TSS is confounded with HER2 status** (chi2 = 177.9, p = 3.73e-10). This means TSS
   and HER2 are entangled -- you cannot correct for one without risking attenuating the
   other. This is the central complication.

3. **TSS drives HER2-label missingness** (chi2 = 464.66, p ~ 10^-73 from NB00). Some
   sites systematically lack IHC/FISH data, introducing selection bias into the labeled
   cohort.

4. **40 tissue source sites**, sample sizes ranging from 1 to 150. Many sites contribute
   fewer than 5 samples -- too few for reliable per-site estimation.

5. **NB01 currently flags the problem but does not address it.** The summary section
   does not mention TSS batch effects at all.

---

## Why This Matters

TSS batch effects are the most credible alternative explanation for any downstream finding.
If an "HER2-enriched cluster" is partly a "Roswell Park cluster," the biological claim
weakens. If SHAP features include genes that vary by sequencing center rather than by
biology, the ML model is learning batch, not disease.

The risk is asymmetric: **not addressing TSS is a reviewable weakness** in a take-home
for a genomics company. Tempus processes multi-site clinical sequencing data -- they will
notice if batch effects are flagged but left unaddressed.

---

## Strategy: Quantify First, Then Correct Minimally

### Phase 1 -- Quantify the Impact (must-do)

**Goal:** Produce concrete numbers that bound the magnitude of TSS bias so we can argue
intelligently about whether it matters for each downstream analysis.

Metrics to compute (implemented in `src/01s_tss_batch_assessment.py`):

| Metric | What It Tells Us | Method |
|--------|-----------------|--------|
| **Variance explained by TSS per PC** | How much of each PC's variance is TSS-attributable | eta2 (eta-squared) from one-way ANOVA on PC scores grouped by TSS |
| **Variance explained by TSS on ERBB2** | Whether the primary gene of interest varies by site | eta2 on ERBB2 expression ~ TSS |
| **TSS x HER2 confounding structure** | Which sites drive the confounding | Mosaic plot or enrichment table: per-site HER2 prevalence with Fisher p-values |
| **Silhouette contribution of TSS** | Whether removing TSS signal changes cluster structure | Compare silhouette scores on raw PCs vs TSS-residualized PCs |
| **Top-TSS-variable genes** | Which genes are most TSS-driven (and do they overlap SHAP top features?) | One-way ANOVA per gene, rank by F-statistic, cross-reference with SHAP |

**Key output:** A table showing eta2 for TSS across the top 10 PCs. If the largest eta2
is < 0.05, TSS is a minor nuisance. If eta2 > 0.10 on any PC that also loads on HER2
biology, correction is warranted.

### Phase 2 -- Correct via Linear Regression (should-do)

**Why regression, not ComBat:**
- ComBat (empirical Bayes batch correction) assumes batch is orthogonal to biology.
  TSS is confounded with HER2 status (chi2 = 177.9). ComBat would attenuate the HER2
  signal we are trying to detect. The revision plan already flags ComBat as unreliable
  here.
- Linear regression is transparent, tunable, and lets us protect specific covariates.

**Method: residualization with protected covariates.**

For each gene g across samples:

```
expression_g ~ B0 + B_tss x TSS_dummies + B_her2 x HER2_composite + B_er x ER_status + e
```

The corrected expression is:

```
corrected_g = B0 + B_her2 x HER2 + B_er x ER + e   (keep biology, remove TSS)
```

In practice this is:
```python
import statsmodels.api as sm

# Encode TSS as dummies (drop one reference level)
tss_dummies = pd.get_dummies(df['tss'], prefix='tss', drop_first=True)

# Protected covariates -- their signal is preserved, not regressed out
protected = pd.get_dummies(df['her2_composite'], prefix='her2', drop_first=True)
protected['ER_pos'] = (df['ER Status By IHC'] == 'Positive').astype(int)

# Design matrix: TSS dummies + protected covariates + intercept
X = pd.concat([tss_dummies, protected], axis=1)
X = sm.add_constant(X)

# For each gene, fit OLS, then reconstruct WITHOUT the TSS betas
corrected = gene_expression.copy()
for gene in gene_cols:
    model = sm.OLS(df[gene], X).fit()
    # Subtract only the TSS contribution
    tss_contribution = model.predict(X[['const'] + [c for c in tss_dummies.columns]].assign(const=0))
    # Actually: cleaner to just take residuals + predicted from protected only
    corrected[gene] = model.resid + model.params['const'] + (protected * model.params[protected.columns]).sum(axis=1)
```

**Implementation considerations:**
- Only apply to sites with >=5 samples (collapse rare sites into "Other" category).
- The protected-covariate approach ensures that HER2 and ER biological signal is
  explicitly preserved. Without this, regression would also remove the biological
  component correlated with TSS (because TSS and HER2 are confounded).
- Validate by re-running PCA on corrected data: TSS eta2 should drop, HER2/ER
  associations should be preserved or strengthened.

### Phase 3 -- Sensitivity Analysis (should-do, fast)

Run two versions of the key downstream analyses side-by-side:
1. **Uncorrected** (current pipeline)
2. **TSS-residualized**

Compare:
- Clustering: Do cluster assignments change? If >90% concordance, TSS correction is
  cosmetic. If <80%, it's substantive and must be reported.
- ML: Does AUC change? Do SHAP top features change? If a SHAP top feature drops after
  TSS correction, it was batch-driven.
- ERBB2 violin by HER2 status: Does the separation hold after correction?

**This is the most valuable deliverable.** Rather than arguing about whether to correct,
showing that results are robust (or not) to correction is the strongest possible claim.

---

## Decision Framework for Downstream Notebooks

| Downstream Analysis | TSS Risk | Recommended Action |
|--------------------|---------|--------------------|
| **Supervised ML (NB03)** | Moderate -- model may learn TSS proxies | Run with and without correction; compare SHAP features |
| **Unsupervised clustering (NB02/03)** | High -- TSS can create artificial clusters | Report corrected and uncorrected cluster concordance |
| **ERBB2 expression analysis (NB02)** | Low-moderate -- ERBB2 is strongly HER2-driven, but site-level variation exists | Report eta2 for ERBB2 ~ TSS; if < 0.05, proceed without correction |
| **Survival analysis (NB04)** | Low -- KM/Cox on clinical data, not expression | No correction needed on expression; check TSS isn't confounding survival directly |
| **Discordant case identification (NB02)** | Moderate -- a "discordant" case might just be from a batch-outlier site | Cross-tabulate discordant cases by TSS; flag if concentrated in few sites |

---

## What This Does NOT Solve

- **Label missingness bias.** TSS-driven missingness (some sites lack IHC/FISH data)
  introduces selection bias into the labeled training set. Regression correction on
  expression doesn't fix this -- it's a cohort composition issue. Acknowledge it as
  a limitation.
- **True biological site differences.** Some TSS variation is real (different patient
  populations at different centers). The regression removes all TSS variance, including
  real biology. The protected-covariate approach mitigates this for HER2/ER but not for
  other biology.
- **Single-sample sites.** Sites with 1-2 samples cannot be reliably corrected. Collapse
  them into "Other" or exclude from the batch assessment (but keep in the analysis).

---

## Analyst Checklist (for validating Phase 1 outputs)

- [ ] eta2 values are computed on the correct PC scores (post-normalization, not raw counts)
- [ ] Protected covariates include HER2 composite AND ER status (both are confounded with TSS)
- [ ] Rare-site handling is documented (collapsed or excluded, with threshold stated)
- [ ] ERBB2-specific eta2 is reported separately from genome-wide eta2
- [ ] Cross-reference of top TSS-variable genes vs SHAP top features is present
- [ ] Before/after PCA plots show TSS signal reduced without collapsing HER2 separation
