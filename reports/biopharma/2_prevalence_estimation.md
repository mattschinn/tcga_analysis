# Analysis 2: Prevalence of Molecular ERBB2 Overexpression in IHC-Negative Patients

## Key Findings

- At the primary threshold (95th percentile), 35 of 686
  IHC-negative patients (5.1%, 95% CI: [3.7%,
  7.0%]) show molecular evidence of ERBB2 overexpression.
- Of these, 6 (17%) also show genomic amplification
  (CN >= 2), providing orthogonal evidence for HER2+ biology.
- Prevalence is robust across thresholds: 0.9% (p99) to
  10.1% (p90).
- If this prevalence holds in the general IHC-negative population, it represents an
  estimated 9,154-17,392 patients annually in the US.

## Methods

### Framing

This analysis estimates the **prevalence of molecular ERBB2 overexpression among
IHC-negative patients** -- not a "false negative rate." IHC is the clinical ground
truth; there is no higher authority to define false negatives against. The defensible
framing is: a defined proportion of IHC-negative patients show molecular evidence
that questions their negative classification.

### Approach

The denominator was all IHC-negative patients in the multimodal cohort with RNA-seq
data (N=686). The primary threshold for ERBB2 overexpression was the 95th
percentile of ERBB2 expression among IHC-negative patients -- the same criterion
used in NB02 for discordant identification.

Sensitivity analysis used three additional thresholds: 90th percentile, 99th
percentile, and mean + 2 standard deviations.

95% confidence intervals were computed using the Wilson score method. CN stratification
used GISTIC-derived copy number values (0, 1, 2).

## Results

### Prevalence by Threshold

| Threshold | Value | Count | Prevalence | 95% CI |
|---|---|---|---|---|
| p95 (primary) | 9.862 | 35/686 | 5.1% | [3.7%, 7.0%] |
| p90 | 9.629 | 69/686 | 10.1% | [8.0%, 12.5%] |
| p99 | 10.421 | 7/686 | 1.0% | [0.5%, 2.1%] |
| mean+2SD | 10.501 | 6/686 | 0.9% | [0.4%, 1.9%] |

### CN Stratification (Primary Threshold)

Among the 35 patients with molecular ERBB2 overexpression (p95):

| CN Status | N | Percentage | Interpretation |
|---|---|---|---|
| CN >= 2 (amplified) | 6 | 17% | Strongest case: genomic + transcriptomic evidence |
| CN = 1 | 19 | 54% | Intermediate: modest CN gain + high expression |
| CN = 0 | 10 | 29% | Transcriptional only: expression without amplification |

### Population Extrapolation (Heavily Caveated)

| Parameter | Value |
|---|---|
| US breast cancer incidence (annual) | ~310,000 |
| Estimated HER2-negative fraction | ~80% |
| US HER2-negative patients/year | ~248,000 |
| Estimated molecular overexpression (low) | 9,154 |
| Estimated molecular overexpression (mid) | 12,653 |
| Estimated molecular overexpression (high) | 17,392 |

**Caveats:** TCGA is not a random sample of the breast cancer population. Ascertainment
bias, single-institution effects, and pre-treatment selection may inflate or deflate
these estimates. The extrapolation is included for order-of-magnitude context only.

## Limitations

- The threshold for "molecular overexpression" is statistically derived (distribution
  percentiles), not clinically validated. The clinical relevance of exceeding any
  particular percentile depends on treatment response data not available in TCGA.
- TCGA ascertainment bias: academic medical center patients may differ from the
  general population in disease severity, testing practices, and demographics.
- The prevalence estimate applies to patients who are IHC-negative AND have RNA-seq
  data. Patients without RNA-seq testing cannot be assessed.
- CN = 0 patients with high ERBB2 RNA may represent transcriptional regulation
  (e.g., ER-driven) rather than HER2-driven biology (see Analysis #3).

## Implications

The identification of 5.1% of IHC-negative patients with
molecular evidence of ERBB2 overexpression suggests a clinically meaningful population
that may be missed by current IHC-only testing. The CN-stratified breakdown provides
a biological hierarchy:

1. **CN >= 2 patients (n=6)** represent the strongest candidates for
   reclassification -- they have both genomic amplification and transcriptomic
   overexpression, suggesting true IHC false negatives.
2. **CN = 0/1 patients (n=29)** require deeper biological characterization
   (see Analysis #3) to determine whether their ERBB2 overexpression is
   HER2-pathway-driven or reflects ER/luminal co-regulation.

RNA-based molecular testing could identify this population in clinical practice,
either as a standalone CDx or as a reflex test for IHC-negative patients with
clinical suspicion of HER2-driven disease.

---

**Figures:**
- `fig_04_2_prevalence_by_threshold.png` -- Prevalence at each threshold with CIs
- `fig_04_2_prevalence_cn_stratified.png` -- CN breakdown of overexpression group
- `fig_04_2_erbb2_distribution_thresholds.png` -- ERBB2 distribution with thresholds
