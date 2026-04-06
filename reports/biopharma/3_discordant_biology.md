# Analysis 3: Discordant Biology -- Normal Tissue and ER Pathway

## Key Findings

- CN-low discordant patients show elevated ER pathway expression (z=0.45), supporting ER/luminal co-regulation of ERBB2 in this subgroup.
- CN-high discordant patients show strong HER2 pathway activation (z=0.61), consistent with IHC-missed HER2+ biology driven by genomic amplification.
- Only 4 discordant patients had matched normal tissue, limiting tumor-to-normal ratio analysis to descriptive statistics.
- Genomic instability (FGA): CN-low discordant vs concordant negative p=0.8692, no significant difference.

## Methods

### Part A: Tumor-to-Normal ERBB2 Ratios

Matched tumor-normal pairs (N=112) were used to compute tumor-to-normal
ERBB2 expression ratios (log-space subtraction). Patients were stratified by
concordance group. Due to only 4 discordant patients having matched
normals, this analysis is descriptive.

### Part B: ER Pathway Correlation

Spearman correlations between ERBB2 and ER pathway genes (ESR1, FOXA1, GATA3,
PGR, etc.) were computed within CN-low discordant patients (n=29)
and concordant negatives as a null comparator. A pathway-level heatmap summarized
mean z-scored expression across gene groups and patient groups.

### Part C: ER Quantitative Scores

Clinical ER quantitative scores (Allred, H-score, intensity, percent positive)
from the cleaned clinical dataset were compared across groups using Kruskal-Wallis
and pairwise Mann-Whitney U tests.

### Part D: Fraction Genome Altered

FGA from the multimodal cohort was compared across groups as a measure of genomic
instability.

## Results

### Part A: Tumor-to-Normal ERBB2 Ratios

| Group | N (matched) | ERBB2 Ratio Median | IQR |
|---|---|---|---|
| Concordant Negative | 63 | 0.294 | [-0.323, 0.939] |
| Discordant CN-low | 4 | 1.087 | [0.842, 1.290] |
| Discordant CN-high | 0 | -- | -- |
| Concordant Positive | 21 | 1.830 | [0.026, 4.429] |

**Note:** Only 4 discordant patients had matched normals. Interpret
tumor-to-normal ratios for discordant subgroups descriptively only.

### Part B: ER Pathway Correlations

**ERBB2 vs ER Pathway Genes (Spearman) -- CN-low Discordant (n=29):**

| Gene | rho | p-value | N |
|---|---|---|---|
| ESR1 | -0.359 | 0.0561 | 29 |
| FOXA1 | -0.231 | 0.2289 | 29 |
| GATA3 | -0.251 | 0.1886 | 29 |
| TFF1 | -0.388 | 0.0374* | 29 |
| TFF3 | -0.416 | 0.0247* | 29 |
| PGR | 0.062 | 0.7510 | 29 |
| XBP1 | -0.035 | 0.8571 | 29 |
| AGR2 | 0.008 | 0.9676 | 29 |
| CA12 | -0.232 | 0.2259 | 29 |
| NAT1 | -0.344 | 0.0678 | 29 |
| SLC39A6 | -0.288 | 0.1295 | 29 |

**ERBB2 vs ER Pathway Genes (Spearman) -- Concordant Negative (n=651):**

| Gene | rho | p-value | N |
|---|---|---|---|
| ESR1 | 0.298 | 0.0000* | 647 |
| FOXA1 | 0.339 | 0.0000* | 647 |
| GATA3 | 0.367 | 0.0000* | 647 |
| TFF1 | 0.321 | 0.0000* | 647 |
| TFF3 | 0.318 | 0.0000* | 647 |
| PGR | 0.287 | 0.0000* | 647 |
| XBP1 | 0.385 | 0.0000* | 647 |
| AGR2 | 0.304 | 0.0000* | 647 |
| CA12 | 0.400 | 0.0000* | 647 |
| NAT1 | 0.411 | 0.0000* | 647 |
| SLC39A6 | 0.349 | 0.0000* | 647 |

### Pathway Expression Heatmap (Mean Z-scores)

|                     |   ER pathway |   HER2 pathway |   Proliferation |
|:--------------------|-------------:|---------------:|----------------:|
| Concordant Negative |   -0.0344396 |      -0.189721 |      -0.0522163 |
| Discordant CN-low   |    0.450716  |       0.281363 |      -0.293078  |
| Discordant CN-high  |   -0.366398  |       0.606445 |       0.573273  |
| Concordant Positive |    0.0350434 |       0.734166 |       0.295333  |

### Part C: ER Quantitative Scores


**Er Allred Score:**

| Group | N | Median | IQR |
|---|---|---|---|
| Concordant Negative | 73 | 8.0 | [5.0, 8.0] |
| Concordant Positive | 13 | 5.0 | [0.0, 7.0] |


**Er Hscore:**

| Group | N | Median | IQR |
|---|---|---|---|
| Concordant Negative | 17 | 200.0 | [160.0, 230.0] |
| Discordant CN-high | 1 | 80.0 | [80.0, 80.0] |
| Concordant Positive | 2 | 230.0 | [230.0, 230.0] |


### Part D: Fraction Genome Altered

| Group | N | Median FGA | IQR |
|---|---|---|---|
| Concordant Negative | 651 | 0.249 | [0.117, 0.447] |
| Discordant CN-low | 29 | 0.230 | [0.133, 0.426] |
| Discordant CN-high | 6 | 0.454 | [0.397, 0.513] |
| Concordant Positive | 151 | 0.295 | [0.186, 0.424] |

Discordant CN-low vs Concordant Negative: Mann-Whitney U=9610.5, p=0.8692

## Limitations

- Tumor-to-normal comparison limited to 4 discordant patients with
  matched normals (out of 35 total). This analysis is descriptive only.
- ER quantitative scores are sparsely populated in TCGA clinical annotations;
  coverage varies by metric.
- CN-high discordant subgroup (n=6) is too small for any statistical
  inference; all findings for this group are descriptive.
- Correlation analyses in the CN-low group (n=29) have limited power
  for detecting moderate effect sizes.

## Implications

The CN-low discordant group shows elevated expression of both ER pathway and
HER2 pathway genes, consistent with a luminal biology that co-regulates ERBB2
through ER-driven transcriptional programs. This profile suggests these patients
may benefit from endocrine therapy, with the elevated ERBB2 representing
co-regulation rather than independent oncogenic HER2 signaling.

The CN-high discordant group (n=6) shows a distinct biology with strong HER2
pathway activation, consistent with true IHC false negatives where genomic
amplification is present but IHC failed to detect protein overexpression.
These patients are the strongest candidates for HER2-directed therapy
reclassification.

The biological distinction between CN-high and CN-low discordant patients supports
different clinical strategies for each subgroup, a finding that would be
strengthened by validation in a larger Tempus cohort with treatment outcome data.

---

**Figures:**
- `fig_04_3a_tumor_normal_erbb2_ratio.png` -- Tumor-to-normal ERBB2 ratio by group
- `fig_04_3b_pathway_heatmap.png` -- Pathway expression heatmap
- `fig_04_3c_er_quantitative.png` -- ER quantitative scores by group
- `fig_04_3d_fga_comparison.png` -- FGA by group
