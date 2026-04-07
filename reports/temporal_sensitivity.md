# Temporal Sensitivity Analysis: Diagnosis Year as Potential Confounder

**Script:** `scripts/03_temporal_sensitivity.py`
**Outputs:** `outputs/03_temporal_sensitivity.parquet`, `outputs/03_temporal_crosstab.parquet`, `outputs/figures/fig_03s_temporal_sensitivity.png`

---

## Motivation

Notebook 01 (Section 2.3) identified that HER2 label availability correlates with
diagnosis year (chi2 = 11.5, p = 7.07e-04). In 2007, ASCO/CAP guidelines standardized
HER2 testing for breast cancer. Pre-2007 samples were tested under less uniform
protocols, raising the question: could diagnosis era act as a confounder for the
discordant and equivocal group findings in Notebooks 02--04?

This analysis addresses two questions:

1. Are pre-2007 samples overrepresented in the discordant or equivocal groups?
2. Does excluding pre-2007 samples change the concordant-only model's performance
   or its scoring of discordant/equivocal patients?

## Part 1: Diagnosis Year Distribution by HER2 Group

| Group                     | Post-2007 | Pre-2007 | Unknown | Total | % Pre-2007 |
|---------------------------|-----------|----------|---------|-------|------------|
| Concordant Positive       | 120       | 31       | 0       | 151   | 20.5%      |
| Concordant Negative       | 510       | 139      | 2       | 651   | 21.4%      |
| Discordant (IHC-/RNA-high)| 24        | 10       | 1       | 35    | 29.4%      |
| Equivocal                 | 25        | 3        | 0       | 28    | 10.7%      |
| Other                     | 42        | 59       | 0       | 101   | --         |

**Interpretation:** The discordant group has a modestly higher pre-2007 fraction
(29.4%) compared to the concordant baseline (~21%). However, 24/34 discordant
patients with known year are post-2007, so the group is not dominated by
pre-guideline cases. The equivocal group is actually under-represented in
pre-2007 samples (10.7%), consistent with IHC 2+ being a well-defined category
even before the 2007 guidelines.

## Part 2: Concordant-Only Model Comparison

The concordant-only Random Forest (same curated 44-gene panel, CN, ER/PR as
NB03 Section 4) was trained under two conditions:

| Model               | N (train) | Pos | Neg | CV AUC |
|---------------------|-----------|-----|-----|--------|
| All eras (baseline) | 621       | 107 | 514 | 1.000  |
| Post-2007 only      | 485       | 87  | 398 | 1.000  |

Both achieve perfect cross-validated AUC. The HER2 expression signal is strong
enough that removing 132 pre-2007 concordant training samples has no effect on
classification performance.

## Part 3: Discordant Patient Scoring Stability

The 35 IHC-/RNA-high discordant patients were scored by both models:

| Metric                           | Value   |
|----------------------------------|---------|
| Mean P(HER2+), all-era model    | 0.663   |
| Mean P(HER2+), post-2007 model  | 0.661   |
| Mean delta                       | -0.002  |
| Max absolute delta               | 0.062   |
| Pearson r (all-era vs post-2007) | 0.989   |

CN-stratified deltas are negligible:

| Subgroup   | N  | Mean delta | Std delta |
|------------|----|------------|-----------|
| CN >= 2    | 6  | +0.005     | 0.027     |
| CN < 2     | 29 | -0.003     | 0.027     |

10 of 34 discordant patients with known year are pre-2007, yet the scoring
is virtually unchanged when pre-2007 training data is removed.

## Part 4: Equivocal Patient Scoring Stability

The 28 equivocal patients were scored by both models:

| Metric                                | Value      |
|---------------------------------------|------------|
| Mean P(HER2+), all-era model         | 0.349      |
| Mean P(HER2+), post-2007 model       | 0.347      |
| Mean delta                            | -0.002     |
| Reclassification agreement (t = 0.5) | 26/28 (93%)|

The two reclassification disagreements reflect patients near the decision
boundary (probability close to 0.5), not systematic era-driven bias.

## Conclusion

**Diagnosis year does not confound the key findings.** The concordant-only model,
discordant patient scores, and equivocal reclassifications are stable when
pre-2007 samples are excluded from training. The modest overrepresentation of
pre-2007 samples in the discordant group (29% vs 21% baseline) does not
translate into measurable bias in model behavior. This is consistent with
the fact that the model relies on molecular features (gene expression, copy
number) rather than IHC labels, so era-dependent IHC testing practices do
not propagate into the classifier.
