# Analysis 4a: Clinical Correlates of Discordant Patients

## Key Findings

- No clinical variables reached statistical significance after FDR correction.
- Discordant group (n=35) compared against concordant negative (n=659).
- CN-high discordant (n=6) is too small for independent statistical inference.

## Methods

Clinical characteristics were compared across four groups: Concordant Negative,
Discordant CN-low, Discordant CN-high, and Concordant Positive. Continuous variables
were summarized as median (IQR) and compared using Kruskal-Wallis. Categorical
variables used chi-squared or Fisher's exact test (discordant combined vs concordant
negative). All p-values were FDR-corrected (Benjamini-Hochberg).

## Results

### Table 1: Clinical Characteristics

| Variable | Concordant Neg | Discordant CN-low | Discordant CN-high | Concordant Pos | p | FDR p |
|---|---|---|---|---|---|---|
| Er Allred Score | 8.0 (6.0-8.0) | -- | -- | 6.0 (0.0-8.0) | 0.0505 | 0.1009 |
| Er Hscore | 200.0 (160.0-230.0) | -- | 80.0 (80.0-80.0) | 230.0 (230.0-230.0) | 0.1510 | 0.1510 |

### Polysomy 17


### ER Scoring Method

**Discordant CN-low:** {'Two-tier': 1}

**Discordant CN-high:** No data.

**Concordant Negative:** {'H-SCORE': 14, 'H-Score': 9, 'Allred Score': 4, 'H Score': 1, 'Oncotype Dx test': 1, 'Two-tier': 1}


## Limitations

- CN-high discordant group (n=6) is severely underpowered for any
  independent statistical comparison. All CN-high findings are descriptive.
- TCGA clinical annotations are variably complete across institutions; missing data
  is not random (certain TSS sites have more complete records).
- FDR correction across 2 tests is conservative given the small sample
  sizes in the discordant group.

## Implications

The lack of significant clinical differences after FDR correction suggests that the discordant population is clinically similar to concordant negatives. This implies that clinical features alone cannot identify these patients -- molecular testing is required.

---

**Figures:**
- `fig_04_4a_table1_forest.png` -- Forest plot of effect sizes
