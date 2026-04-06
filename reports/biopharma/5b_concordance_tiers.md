# Analysis 5b: Multi-Modal Concordance Tiers for Equivocal Patients

## Key Findings

- **Tier 1: High confidence HER2+:** 4 patients (ERBB2 median=10.43, ML prob=0.876)
- **Tier 2: RNA-only HER2+:** 1 patients (ERBB2 median=9.29, ML prob=0.500)
- **Tier 3: Concordant HER2-:** 21 patients (ERBB2 median=9.16, ML prob=0.169)
- **Tier 4: Mixed signals:** 2 patients (ERBB2 median=9.63, ML prob=0.414)

## Methods

The 28 equivocal (IHC 2+) patients were classified into concordance tiers based on
agreement across three modalities:

- **RNA call:** Ensemble ML probability >= 0.5 -> Positive
- **CN call:** GISTIC copy number >= 2 -> Amplified
- **FISH call:** Definitive FISH result (Positive/Negative) or NA

Tier definitions:
- **Tier 1 (High confidence HER2+):** RNA+ AND (CN amplified OR FISH+)
- **Tier 2 (RNA-only HER2+):** RNA+ AND CN not amplified AND FISH not positive
- **Tier 3 (Concordant HER2-):** RNA- AND CN not amplified AND FISH not positive
- **Tier 4 (Mixed signals):** Any other combination

## Results

### Tier Distribution

| Tier | N | ERBB2 Median | GRB7 Median | ML Prob Median | ER+ Rate | FGA Median |
|---|---|---|---|---|---|---|
| Tier 1: High confidence HER2+ | 4 | 10.43 | 6.80 | 0.876 | 50% | 0.479 |
| Tier 2: RNA-only HER2+ | 1 | 9.29 | 5.05 | 0.500 | 100% | 0.436 |
| Tier 3: Concordant HER2- | 21 | 9.16 | 5.02 | 0.169 | 90% | 0.263 |
| Tier 4: Mixed signals | 2 | 9.63 | 6.29 | 0.414 | 100% | 0.297 |

### Patient-Level Table

| pid | RNA Call | CN Call | FISH Call | Tier | ERBB2 | ML Prob |
|---|---|---|---|---|---|---|
| TCGA-C8-A26W | Positive | Amplified | NA | Tier 1 | 9.96 | 0.922 |
| TCGA-C8-A12L | Positive | Amplified | NA | Tier 1 | 10.30 | 0.700 |
| TCGA-BH-A42T | Positive | Amplified | NA | Tier 1 | 10.55 | 0.830 |
| TCGA-B6-A1KF | Positive | Amplified | NA | Tier 1 | 11.74 | 0.938 |
| TCGA-C8-A1HN | Positive | Not amplified | NA | Tier 2 | 9.29 | 0.500 |
| TCGA-A2-A25E | Negative | Not amplified | NA | Tier 3 | 8.92 | 0.275 |
| TCGA-E2-A105 | Negative | Not amplified | NA | Tier 3 | 9.19 | 0.454 |
| TCGA-D8-A1XA | Negative | Not amplified | NA | Tier 3 | 9.16 | 0.163 |
| TCGA-C8-A1HE | Negative | Not amplified | NA | Tier 3 | 9.31 | 0.269 |
| TCGA-C8-A134 | Negative | Not amplified | NA | Tier 3 | 7.30 | 0.103 |
| TCGA-C8-A130 | Negative | Not amplified | NA | Tier 3 | 8.79 | 0.423 |
| TCGA-BH-A28Q | Negative | Not amplified | NA | Tier 3 | 9.54 | 0.103 |
| TCGA-E2-A1IH | Negative | Not amplified | NA | Tier 3 | 8.29 | 0.366 |
| TCGA-BH-A18H | Negative | Not amplified | NA | Tier 3 | 8.47 | 0.086 |
| TCGA-AR-A5QP | Negative | Not amplified | NA | Tier 3 | 8.82 | 0.122 |
| TCGA-AR-A5QM | Negative | Not amplified | NA | Tier 3 | 8.72 | 0.185 |
| TCGA-AN-A0XP | Negative | Not amplified | NA | Tier 3 | 9.59 | 0.166 |
| TCGA-AN-A0FZ | Negative | Not amplified | NA | Tier 3 | 10.08 | 0.169 |
| TCGA-AN-A0AK | Negative | Not amplified | NA | Tier 3 | 9.99 | 0.387 |
| TCGA-AN-A041 | Negative | Not amplified | NA | Tier 3 | 10.44 | 0.447 |
| TCGA-AC-A3TN | Negative | Not amplified | NA | Tier 3 | 10.17 | 0.228 |
| TCGA-AC-A3TM | Negative | Not amplified | NA | Tier 3 | 9.45 | 0.378 |
| TCGA-AC-A2FO | Negative | Not amplified | NA | Tier 3 | 9.69 | 0.138 |
| TCGA-A8-A07R | Negative | Not amplified | NA | Tier 3 | 8.87 | 0.048 |
| TCGA-BH-A0W5 | Negative | Not amplified | NA | Tier 3 | 8.34 | 0.088 |
| TCGA-E9-A3Q9 | Negative | Not amplified | NA | Tier 3 | 8.63 | 0.138 |
| TCGA-C8-A138 | Negative | Amplified | NA | Tier 4 | 9.32 | 0.361 |
| TCGA-C8-A1HL | Negative | Amplified | NA | Tier 4 | 9.95 | 0.466 |

## Limitations

- FISH data is absent for all 28 equivocal patients (all NA or Equivocal/Indeterminate),
  so FISH does not contribute to tier assignment in this dataset.
- Tier assignments rely primarily on RNA + CN agreement.
- Small sample size (n=28) limits statistical power for between-tier comparisons.

## Implications

Tier 1 patients (RNA+ and CN amplified) represent the highest-confidence candidates
for HER2 reclassification -- multiple independent modalities agree on HER2+ biology.
Tier 2 patients (RNA-only) may still be HER2+ but require additional validation.
Tier 3 patients are concordantly HER2- across available modalities.
Tier 4 patients show mixed signals and would be candidates for additional testing
in a clinical setting.

In a Tempus dataset with complete FISH data, tier assignment would gain a third
modality, further sharpening the confidence stratification.

---

**Figures:**
- `fig_04_5b_concordance_tiers.png` -- Biological characterization by tier
