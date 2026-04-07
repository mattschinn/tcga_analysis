# Analysis 5a Supplementary: Held-Out Concordance Validation

## Purpose

The primary Analysis 5a reported AUC = 0.994 for RNA-based prediction of FISH outcome
in IHC 2+ patients. However, those 154 patients were in the ML training set (their
FISH results determined their training labels). This supplementary analysis eliminates
that contamination by retraining models with all IHC 2+ FISH-resolved patients
excluded, then scoring them as a genuinely held-out population.

## Key Findings

- **Held-out AUC = 0.779** (vs. 0.994 in primary analysis).
- At threshold 0.5: sensitivity = 0.412, specificity = 0.983,
  kappa = 0.488.
- Optimal threshold (Youden's J): 0.314
  (sensitivity = 0.647, specificity = 0.917).
- Equivocal reclassification: 5 HER2+ (vs. 5 in primary),
  28/28 patients agree (100%).

## Methods

### Training Set Construction

The standard training set (837 labeled patients) was filtered to exclude all 154
IHC 2+ patients with definitive FISH results. The remaining 683 patients
consist of IHC 0, IHC 1+, and IHC 3+ patients (plus any IHC 2+ without FISH who
were labeled via other pathways).

| Set | N | Positive | Negative |
|---|---|---|---|
| Original training | 837 | 151 | 686 |
| Held-out training | 683 | 117 | 566 |
| Removed (IHC 2+ FISH-resolved) | 154 | 34 | 120 |

### Model Training

Three models (L1-LR, Random Forest, XGBoost) were trained on the reduced dataset
using the curated gene panel (~45 genes from 6 biological gene sets: HER2/17q12
amplicon, ERBB signaling, luminal/ER, basal, proliferation, EMT) plus copy number
and ER/PR status. This matches the feature reduction approach used in the
consolidated NB03 analysis. Models were fit on the full reduced training set and
applied to the held-out IHC 2+ population.

### Concordance Analysis

The held-out concordance table compares ensemble RNA probability (mean of 3 models)
against FISH ground truth. This is a genuinely out-of-sample evaluation: the models
have never seen any IHC 2+ patient during training.

## Results

### Model Performance

| Model | CV AUC (training) | Held-out AUC (IHC 2+) |
|---|---|---|
| L1-LR | 0.865 | 0.749 |
| Random Forest | 0.870 | 0.807 |
| Gradient Boosting | 0.852 | 0.758 |
| **Ensemble** | -- | **0.779** |


### Concordance Table (threshold = 0.5, N=154)

|  | RNA Positive | RNA Negative |
|---|---|---|
| FISH Positive | 14 | 20 |
| FISH Negative | 2 | 118 |

| Metric | Value (95% CI) |
|---|---|
| Sensitivity | 0.412 (0.264-0.578) |
| Specificity | 0.983 (0.941-0.995) |
| PPV | 0.875 (0.640-0.965) |
| NPV | 0.855 (0.787-0.904) |
| Accuracy | 0.857 (0.793-0.904) |
| Cohen's kappa | 0.488 |
| AUC | 0.779 |

**Optimal threshold (Youden's J):** 0.314
(sensitivity = 0.647, specificity = 0.917)

### Comparison with Primary Analysis

| Metric | Primary (5a) | Held-Out (this analysis) |
|---|---|---|
| AUC | 0.994 | 0.779 |
| Sensitivity (0.5) | 0.706 | 0.412 |
| Specificity (0.5) | 1.000 | 0.983 |
| Kappa (0.5) | 0.790 | 0.488 |
| Optimal threshold | 0.341 | 0.314 |
| Sens (optimal) | 0.941 | 0.647 |
| Spec (optimal) | 0.992 | 0.917 |

### Equivocal Reclassification Stability

| Metric | Original Models | Held-Out Models |
|---|---|---|
| HER2+ calls | 5 | 5 |
| HER2- calls | 23 | 23 |
| Agreement | 28/28 (100%) |  |

**Patient-level comparison:**

| pid | ERBB2 | CN | Prob (orig) | Prob (held-out) | Call (orig) | Call (held-out) |
|---|---|---|---|---|---|---|
| TCGA-B6-A1KF | 11.74 | 2 | 0.938 | 0.853 | Positive | Positive |
| TCGA-C8-A26W | 9.96 | 2 | 0.922 | 0.932 | Positive | Positive |
| TCGA-BH-A42T | 10.55 | 2 | 0.830 | 0.924 | Positive | Positive |
| TCGA-C8-A12L | 10.30 | 2 | 0.700 | 0.811 | Positive | Positive |
| TCGA-C8-A1HN | 9.29 | 1 | 0.500 | 0.601 | Positive | Positive |
| TCGA-C8-A1HL | 9.95 | 2 | 0.466 | 0.457 | Negative | Negative |
| TCGA-E2-A105 | 9.19 | 1 | 0.454 | 0.238 | Negative | Negative |
| TCGA-AN-A041 | 10.44 | 1 | 0.447 | 0.271 | Negative | Negative |
| TCGA-C8-A130 | 8.79 | 1 | 0.423 | 0.328 | Negative | Negative |
| TCGA-AN-A0AK | 9.99 | 0 | 0.387 | 0.373 | Negative | Negative |
| TCGA-AC-A3TM | 9.45 | 0 | 0.378 | 0.385 | Negative | Negative |
| TCGA-E2-A1IH | 8.29 | 1 | 0.366 | 0.175 | Negative | Negative |
| TCGA-C8-A138 | 9.32 | 2 | 0.361 | 0.292 | Negative | Negative |
| TCGA-A2-A25E | 8.92 | 1 | 0.275 | 0.171 | Negative | Negative |
| TCGA-C8-A1HE | 9.31 | 0 | 0.269 | 0.113 | Negative | Negative |
| TCGA-AC-A3TN | 10.17 | 0 | 0.228 | 0.321 | Negative | Negative |
| TCGA-AR-A5QM | 8.72 | 0 | 0.185 | 0.121 | Negative | Negative |
| TCGA-AN-A0FZ | 10.08 | 1 | 0.169 | 0.320 | Negative | Negative |
| TCGA-AN-A0XP | 9.59 | 0 | 0.166 | 0.216 | Negative | Negative |
| TCGA-D8-A1XA | 9.16 | 0 | 0.163 | 0.143 | Negative | Negative |
| TCGA-AC-A2FO | 9.69 | 0 | 0.138 | 0.061 | Negative | Negative |
| TCGA-E9-A3Q9 | 8.63 | 0 | 0.138 | 0.044 | Negative | Negative |
| TCGA-AR-A5QP | 8.82 | 0 | 0.122 | 0.062 | Negative | Negative |
| TCGA-BH-A28Q | 9.54 | 0 | 0.103 | 0.050 | Negative | Negative |
| TCGA-C8-A134 | 7.30 | -1 | 0.103 | 0.064 | Negative | Negative |
| TCGA-BH-A0W5 | 8.34 | 0 | 0.088 | 0.064 | Negative | Negative |
| TCGA-BH-A18H | 8.47 | -1 | 0.086 | 0.094 | Negative | Negative |
| TCGA-A8-A07R | 8.87 | 1 | 0.048 | 0.018 | Negative | Negative |

## Interpretation

The held-out AUC of 0.779 represents a substantial drop of 0.215
from the primary analysis (0.994). This suggests that the primary AUC was significantly
inflated by training-data overlap, and the model's ability to generalize into the
IHC 2+ zone from non-equivocal training data is limited. A CDx validation study would
likely need to include IHC 2+ patients in the training set (with proper cross-validation)
to achieve adequate performance.

## Limitations

- The held-out analysis removes 154 patients from training, including 34 Positives
  (22% of original Positive class). This reduces the model's exposure to borderline
  HER2+ biology, which may underestimate what a properly designed CDx training set
  would achieve.
- The held-out population is still retrospective TCGA data. A prospective concordance
  study remains the definitive validation.
- Feature set includes copy number and ER/PR status alongside RNA expression. The
  held-out concordance reflects multi-modal prediction, not RNA-only.

---

**Figures:**
- `fig_04_5a_supp_heldout_roc.png` -- ROC curve for held-out concordance
- `fig_04_5a_supp_heldout_scores_by_fish.png` -- Score distribution by FISH outcome
- `fig_04_5a_supp_equivocal_comparison.png` -- Equivocal scores: original vs held-out
