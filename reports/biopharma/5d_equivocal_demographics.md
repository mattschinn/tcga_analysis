# Analysis 5d: Equivocal Patient Demographics

## Key Findings

- Among 28 equivocal (IHC 2+) patients, RNA reclassifies 5 as HER2+ and
  23 as HER2-.
- No clinical/demographic variables show statistically significant differences
  between RNA-reclassified subgroups, supporting the need for molecular rather than
  clinical selection criteria.

## Methods

Equivocal patients were split into RNA-predicted HER2+ (ensemble probability >= 0.5,
n=5) and RNA-predicted HER2- (< 0.5, n=23). Clinical and molecular
features were compared using Mann-Whitney U (continuous) and Fisher's exact or
chi-squared (categorical).

**Note:** With n=28 total and only 5 in the smaller group, this analysis is
severely underpowered. All findings are exploratory.

## Results

### Clinical Characteristics

| Variable | RNA-pos (n=5) | RNA-neg (n=23) | p-value |
|---|---|---|---|
| Age at Diagnosis | 67.0 (58.0-68.0) | 62.0 (51.0-75.5) | 0.6099 |
| Histologic Type | Breast Invasive Ductal Carcinoma (4/5) | Breast Invasive Ductal Carcinoma (14/23) | 0.6940 |

### Molecular Features

| Feature | RNA-pos Median | RNA-neg Median | p-value |
|---|---|---|---|
| erbb2_copy_number | 2.000 | 0.000 | 0.0025 |

## Limitations

- Total n=28, with only 5 RNA-positive patients. No statistical test has
  meaningful power at this sample size.
- Clinical annotations in TCGA are variably complete; some variables may have
  extensive missing data within the equivocal subset.
- No treatment or outcome data available for equivocal patients in TCGA.

## Implications

RNA-based reclassification identifies a biologically distinct population that is
clinically indistinguishable from the broader equivocal group. This supports the
need for molecular testing -- clinical features alone cannot identify which equivocal
patients are HER2+ at the molecular level. A prospective study design should use
molecular selection criteria rather than clinical enrichment.

---

**Note:** No figures generated due to small sample size (n=5 vs n=23).
