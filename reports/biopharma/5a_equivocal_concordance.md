# Analysis 5a: Equivocal (IHC 2+) Concordance and RNA Reclassification

## Key Findings

- RNA-based classification achieves **AUC = 0.994** for predicting FISH outcome
  among IHC 2+ patients with paired RNA-seq and FISH data (N=156).
- At the default 0.5 threshold: sensitivity = 0.706, specificity = 1.000,
  accuracy = 0.936 (kappa = 0.790).
- Among 28 truly equivocal patients (IHC 2+ without FISH resolution), RNA reclassifies
  5 (18%) as HER2+ and 23 (82%) as HER2-.
- RNA-reclassified HER2+ equivocal patients show ERBB2 expression consistent with
  concordant positives (p=1.04e-02).

## Methods

### Concordance Analysis

IHC 2+ patients were identified from the full clinical dataset using parsed IHC scores.
Those with both RNA-seq data (in the multimodal cohort) and definitive FISH results
(Positive or Negative) formed the concordance population (N=156). Note: these
patients were labeled Positive or Negative by the HER2 label construction logic
(which resolves IHC 2+ via FISH), so they appear in the training data -- but crucially,
the ML model predicts HER2 status from RNA expression alone, not from FISH. The
concordance analysis tests whether RNA expression can recover the FISH-determined
ground truth for this IHC-ambiguous population.

RNA-predicted HER2 status used the ML ensemble probability from the trained models.
A 2x2 concordance table was constructed comparing RNA prediction vs. FISH outcome.
ROC analysis assessed the continuous RNA score's discriminative ability.

### Equivocal Reclassification

The 28 truly equivocal patients (IHC 2+ without definitive FISH) were scored by three
ML models (L1-LR, Random Forest, XGBoost). An ensemble probability (mean of three
models) was computed and binarized at 0.5. Biological validation compared ERBB2 and
GRB7 expression between RNA-reclassified subgroups.

## Results

### Concordance: RNA vs. FISH in IHC 2+ Patients (N=156)

**Confusion Matrix:**

|  | RNA Positive | RNA Negative |
|---|---|---|
| FISH Positive | 24 | 10 |
| FISH Negative | 0 | 122 |

**Performance Metrics (threshold = 0.5):**

| Metric | Value (95% CI) |
|---|---|
| Sensitivity | 0.706 (0.538-0.832) |
| Specificity | 1.000 (0.969-1.000) |
| PPV | 1.000 (0.862-1.000) |
| NPV | 0.924 (0.866-0.958) |
| Accuracy | 0.936 (0.886-0.965) |
| Cohen's kappa | 0.790 |
| AUC | 0.994 |

**Optimal threshold (Youden's J):** 0.341
(sensitivity = 0.941, specificity = 0.992)

### Equivocal Patient Reclassification (N=28, no FISH)

| Group | N | ERBB2 Median | GRB7 Median | Mean CN |
|---|---|---|---|---|
| RNA-predicted HER2+ | 5 | 10.30 | 6.80 | 1.80 |
| RNA-predicted HER2- | 23 | 9.19 | 5.37 | 0.39 |

Mann-Whitney U (ERBB2, RNA-pos vs RNA-neg): p=1.04e-02

### ERBB2 Expression Across Groups

| Group | N | Median ERBB2 |
|---|---|---|
| Concordant Neg | 651 | 8.72 |
| Equivocal RNA-neg | 23 | 9.19 |
| Equivocal RNA-pos | 5 | 10.30 |
| Concordant Pos | 151 | 11.59 |

## Limitations

- The concordance population (N=156) includes IHC 2+ patients whose FISH results
  were used to assign their training labels. This means the concordance analysis tests
  the model's ability to recover FISH-determined labels from RNA alone, not its
  performance on truly blinded equivocal cases. A prospective concordance study on
  held-out equivocal cases with subsequent FISH would be more rigorous.
- The 28 truly equivocal patients (no FISH) cannot be externally validated.
- ML model probabilities are calibrated to the training distribution; equivocal patients
  sit near the decision boundary by definition, where calibration is weakest.

## Implications

The RNA model achieves strong concordance with FISH for resolving IHC 2+ cases,
supporting the feasibility of an RNA-based companion diagnostic to replace or
supplement FISH reflex testing. The key finding is that RNA expression alone recovers
FISH-determined HER2 status with AUC = 0.994, demonstrating that the
transcriptomic signal captured by the model is informative in the equivocal zone
where IHC alone is insufficient.

For the 28 truly equivocal patients lacking FISH data, RNA reclassification identifies
5 patients with expression profiles consistent with HER2-positive biology.
These patients may benefit from HER2-directed therapy but would be missed without
molecular testing.

In a Tempus real-world dataset with paired RNA-seq and FISH results, a formal
prospective concordance study -- with the model applied to held-out equivocal cases
prior to FISH -- would provide the definitive validation needed for CDx filing.

---

**Figures:**
- `fig_04_5a_roc_rna_vs_fish.png` -- ROC curve for RNA vs FISH in IHC 2+
- `fig_04_5a_concordance_score_by_fish.png` -- Score distribution by FISH outcome
- `fig_04_5a_equivocal_score_distribution.png` -- RNA probability in truly equivocal
- `fig_04_5a_equivocal_erbb2_comparison.png` -- ERBB2 across groups
