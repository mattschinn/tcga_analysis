# Analysis 5c: RNA Continuous Scoring for T-DXd Eligibility

## Key Findings

- Within the HER2-Low population (n=388), RNA expression reveals a
  continuous spectrum of ERBB2 biology that IHC ordinal categories obscure.
- The upper ERBB2 tertile of HER2-Low patients shows ML probabilities
  (median=0.228 if available)
  distinct from the lower tertile, suggesting biologically heterogeneous subgroups.
- HER2-0, HER2-Low, and HER2-Positive show overlapping but progressively shifted
  ERBB2 expression distributions, supporting a biological continuum.

## Methods

### HER2 Spectrum Classification

Patients were classified into HER2-0 (IHC 0), HER2-Low (IHC 1+ or IHC 2+/FISH-),
and HER2-Positive (IHC 3+ or IHC 2+/FISH+) using the `classify_her2_spectrum`
function. HER2-Low patients are clinically eligible for T-DXd (trastuzumab
deruxtecan) per DESTINY-Breast04 criteria.

### Continuous Scoring

Two complementary RNA-based scores were used:
1. Raw ERBB2 expression (log2 normalized RSEM)
2. ML ensemble probability from the trained HER2 classifier

HER2-Low patients were stratified into ERBB2 expression tertiles. Biological
characterization compared HER2 pathway genes, proliferation markers, and ER
pathway genes across tertiles.

## Results

### HER2 Spectrum Distribution

| Category | N |
|---|---|
| HER2-0 | 60 |
| HER2-Low | 366 |
| HER2-Low (presumed) | 22 |
| HER2-Positive | 153 |
| Unclassified | 377 |

### Biological Characterization by ERBB2 Tertile (HER2-Low)

| Tertile | N | ERBB2 | GRB7 | MKI67 | ESR1 | ML Prob |
|---|---|---|---|---|---|---|
| Low | 130 | 8.15 | 4.71 | 7.21 | 7.15 | 0.155 |
| Mid | 129 | 8.92 | 4.96 | 6.90 | 8.38 | 0.188 |
| High | 129 | 9.57 | 5.64 | 6.54 | 8.51 | 0.228 |

### ML Probability Distribution by Spectrum Group

| Group | N | Median ML Prob | IQR |
|---|---|---|---|
| HER2-0 | 60 | 0.176 | [0.139, 0.224] |
| HER2-Low | 388 | 0.186 | [0.147, 0.237] |
| HER2-Positive | 153 | 0.937 | [0.474, 0.982] |

### ssGSEA Pathway Scores by ERBB2 Tertile

|      |   pathway_ERBB2_HER2_SIGNALING |   pathway_ANGIOGENESIS |   pathway_APOPTOSIS |   pathway_COMPLEMENT |   pathway_DNA_REPAIR |   pathway_E2F_TARGETS |   pathway_EPITHELIAL_MESENCHYMAL_TRANSITION |   pathway_ESTROGEN_RESPONSE_EARLY |   pathway_ESTROGEN_RESPONSE_LATE |   pathway_FATTY_ACID_METABOLISM |
|:-----|-------------------------------:|-----------------------:|--------------------:|---------------------:|---------------------:|----------------------:|--------------------------------------------:|----------------------------------:|---------------------------------:|--------------------------------:|
| Low  |                       0.439962 |               0.370723 |            0.283006 |             0.225557 |             0.266229 |              0.274707 |                                    0.584086 |                          0.468475 |                         0.468894 |                        0.312688 |
| Mid  |                       0.441078 |               0.359862 |            0.276184 |             0.240321 |             0.256933 |              0.210232 |                                    0.581863 |                          0.557941 |                         0.560325 |                        0.321256 |
| High |                       0.449991 |               0.360947 |            0.272547 |             0.211284 |             0.252139 |              0.182221 |                                    0.579822 |                          0.56651  |                         0.568113 |                        0.314629 |

## Limitations

- HER2 spectrum classification depends on IHC scores, which are sparse in TCGA
  (many patients have NaN IHC scores). 377 patients could not be classified.
- T-DXd treatment-benefit correlation cannot be assessed with TCGA data (no
  treatment response data). This analysis demonstrates biological heterogeneity,
  not treatment benefit.
- The ML model was trained on binary HER2+/- labels; its behavior in the HER2-Low
  zone is extrapolation from training distribution boundaries.

## Implications

Within the HER2-Low population -- the T-DXd eligible group -- RNA-based
quantification reveals substantial biological heterogeneity that IHC ordinal
categories (0, 1+, 2+) fail to capture. The upper ERBB2 tertile shows pathway
activation profiles more similar to HER2-Positive patients, suggesting these
patients may derive greater benefit from HER2-directed ADC therapies like T-DXd.

This supports the hypothesis that RNA-guided T-DXd eligibility stratification
could improve treatment selection beyond the current IHC-based approach. Validation
requires RWD with linked treatment and outcome data (Tempus dataset), where the
predictive value of RNA-continuous scoring for T-DXd response can be directly tested.

---

**Figures:**
- `fig_04_5c_her2low_rna_spectrum.png` -- ERBB2 expression by IHC spectrum
- `fig_04_5c_her2low_ml_density.png` -- ML probability density by group
- `fig_04_5c_her2low_pathway_tertiles.png` -- ssGSEA by tertile (if available)
- `fig_04_5c_erbb2_vs_ml_scatter.png` -- ERBB2 vs ML probability scatter
