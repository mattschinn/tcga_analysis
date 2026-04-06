# Analysis 1: HER2 Testing Method as Confounder

## Key Findings

- Testing method data was available for only 4/35
  (11%) of discordant (IHC-/RNA-high) patients,
  precluding a powered confounder analysis.
- Among the 90 patients with test method annotations (8%
  of full cohort), we describe the distribution below.
- This analysis would be substantially more informative in a Tempus dataset with
  standardized testing metadata.

## Methods

HER2 testing method (`her2_test_method`) was extracted from the cleaned clinical
dataset. Cross-tabulations compared method distribution across HER2 composite status
and discordant vs concordant groups. Statistical testing was limited by sparse data.

## Results

### Test Method Distribution (Full Cohort)

| Method | Count |
|---|---|
| IHC Result Description | 38 |
| Dako HercepTest | 27 |
| CAP 2010 Guidelines | 10 |
| Dextran Coated Charcoal | 9 |
| Ventana | 5 |
| CISH | 1 |
| Missing | 1018 |

### Test Method vs HER2 Status

| her2_test_method        |   Negative |   Positive |   All |
|:------------------------|-----------:|-----------:|------:|
| Dako HercepTest         |         19 |          8 |    27 |
| Dextran Coated Charcoal |          7 |          2 |     9 |
| IHC Result Description  |         32 |          4 |    36 |
| Ventana                 |          4 |          1 |     5 |
| All                     |         62 |         15 |    77 |

### Test Method vs Discordant Status

| her2_test_method        |   Concordant |   Discordant |   All |
|:------------------------|-------------:|-------------:|------:|
| CAP 2010 Guidelines     |           10 |            0 |    10 |
| CISH                    |            1 |            0 |     1 |
| Dako HercepTest         |           25 |            2 |    27 |
| Dextran Coated Charcoal |            9 |            0 |     9 |
| IHC Result Description  |           36 |            2 |    38 |
| Ventana                 |            5 |            0 |     5 |
| All                     |           86 |            4 |    90 |

### Discordant Patients with Test Method Data

| pid | Method | IHC Score | ERBB2 | CN | GRB7 |
|---|---|---|---|---|---|
| TCGA-AN-A049 | Dako HercepTest | 0.0 | 10.10 | 2 | 6.45 |
| TCGA-AO-A03M | IHC Result Description | 2.0 | 9.90 | 0 | 5.38 |
| TCGA-AO-A0JL | IHC Result Description | 1.0 | 10.20 | 2 | 7.19 |
| TCGA-D8-A1XL | Dako HercepTest | 2.0 | 10.25 | 1 | 6.77 |


## Limitations

- Testing method data was available for only 8% of the
  cohort and 11% of discordant patients.
- The sparse data precludes any meaningful statistical inference about testing method
  as a confounder.
- TCGA samples were processed across multiple institutions over several years; testing
  method variation likely reflects institutional and temporal heterogeneity rather than
  systematic bias.

## Implications

The near-complete absence of testing method data in TCGA highlights a critical gap:
method-level metadata is essential for investigating IHC performance variability.
In a Tempus dataset with standardized testing metadata (assay platform, antibody
clone, fixation protocol), this analysis could directly test whether specific testing
configurations are associated with higher discordance rates -- a finding that would
have immediate clinical quality implications.

---

**Note:** No figures generated for this analysis due to insufficient data for
meaningful visualization.
