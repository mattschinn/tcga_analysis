# Column Cleaner Execution Plan: Tier 1 Cleaning + Column Fix

## Objective

Produce a cleaned clinical data CSV at `/data/brca_tcga_clinical_data_cleaned.csv` with
three changes applied to the raw export:

1. **Rename** the mislabeled `Brachytherapy first reference point administered total dose`
   column to `ER positivity scoring method`
2. **Clean** `HER2 positivity method text` (simple dedup)
3. **Clean** `ER positivity scale other` (complex column split)

The original raw file remains untouched. The cleaned file is the one downstream
notebooks should load.

---

## Why These Two Columns

### HER2 positivity method text (90 non-null, 20 unique -> ~5 categories)

**What it is.** Free-text describing the assay platform used for HER2 IHC/FISH testing.
20 spelling variants collapse to roughly 5 concepts: Dako HercepTest, CISH, CAP scoring
guidelines, Ventana, and result-format descriptors.

**Why it matters for the project.** Assay platform is a potential confounder -- and a
potential explanation -- for IHC/RNA discordance. If HER2-discordant patients cluster
on a specific test method (e.g., older dextran-coated charcoal vs. modern HercepTest),
that is a finding directly relevant to the T-DXd eligibility narrative. A clean
categorical column enables stratification and cross-tabulation in NB03/NB04.

**Pipeline complexity.** Low. Spelling-variant dedup into a single categorical column.
This is the pipeline's warmup -- validates end-to-end function before tackling the
harder column.

### ER positivity scale other (244 non-null, 65 unique -> multiple output columns)

**What it is.** Free-text containing ER positivity scoring from multiple incommensurable
measurement systems: Allred scores (0-8), H-scores (0-300), fmol/mg concentrations,
qualitative intensity (Weak/Moderate/Strong), and percentages.

**Why it matters for the project.** ER status is anti-correlated with HER2 in breast
cancer. The binary `ER Status By IHC` column already provides positive/negative, but
the granular scoring here enables:

- **Covariate in ML models**: ER scoring intensity/magnitude as a continuous feature
  rather than binary, which may improve HER2 prediction in the equivocal zone
- **Discordant subgroup characterization**: Do IHC-/RNA-high discordant patients have
  different ER scoring patterns than concordant patients? If so, ER scoring method or
  magnitude could be a clinical flag
- **Cross-tabulation with HER2 status**: ER score distribution across HER2+/HER2-/
  Equivocal groups, stratified by scoring system

The column cannot be harmonized into a single scale (Allred 7 is not comparable to
H-score 240). The right approach is to split into typed sub-columns, each with its
own scale, preserving the clinical semantics.

**Pipeline complexity.** High. Column-split case with 6 measurement systems. This is
the pipeline's hardest test case and highest downstream value.

---

## The Mislabeled Column

`Brachytherapy first reference point administered total dose` contains ER positivity
scoring *methods* (not dose values): "dextran coated charcoal", "IHC", "Allred (Biocare)",
"H-Score-190", "Image Analysis", etc. This is a cBioPortal export artifact where the
column header was mapped incorrectly.

**Evidence:**
- 218 non-null values, 54 unique -- all are ER scoring methods or method+result strings
- 164 patients have both this column and `ER positivity scale other` populated --
  the two are complementary (method vs. value)
- 173 of 218 populated rows have `ER Status By IHC = Positive` -- consistent with
  ER scoring metadata

**Action:** Rename to `ER positivity scoring method`. Do NOT run through the cleaning
pipeline -- the values are heterogeneous (some are pure method names, others are
method+result composites) and would need manual curation beyond the pipeline's scope.
The rename alone makes the column interpretable for downstream use.

---

## Execution Steps

### Step 1: Create the runner script

Write `agents/column_cleaner/run_tier1_cleaning.py` that:

1. Loads the raw CSV from `data/brca_tcga_clinical_data.csv`
2. Renames `Brachytherapy first reference point administered total dose` ->
   `ER positivity scoring method` (simple `df.rename`)
3. Runs `column_cleaner.clean_column()` on `HER2 positivity method text`:
   - Generates context via `generate_context()`
   - Calls `clean_column(series, context)` 
   - Receives `CleaningResult` with cleaned column(s) and report
4. Runs `column_cleaner.clean_column()` on `ER positivity scale other`:
   - Same flow, but expects multiple output columns (Allred, H-score, etc.)
5. Merges cleaned columns back into the DataFrame:
   - Drops original messy columns
   - Inserts cleaned replacement column(s) at the same position
6. Saves to `data/brca_tcga_clinical_data_cleaned.csv`
7. Saves reports and traces to `outputs/column_cleaner/`

### Step 2: Run the pipeline

Execute `run_tier1_cleaning.py`. The pipeline makes LLM calls (Claude Sonnet) via
LangChain, so requires `ANTHROPIC_API_KEY` in the environment.

**Expected outputs:**
- `data/brca_tcga_clinical_data_cleaned.csv` -- full dataset with cleaned columns
- `outputs/column_cleaner/report_her2_positivity_method_text.md`
- `outputs/column_cleaner/report_er_positivity_scale_other.md`
- `outputs/column_cleaner/cleaned_her2_positivity_method_text.csv`
- `outputs/column_cleaner/cleaned_er_positivity_scale_other.csv`
- `outputs/column_cleaner/trace_*.json` -- agent decision traces

### Step 3: Validate outputs

After the pipeline runs, verify:

1. **HER2 method text**: Should collapse 20 variants -> ~5 categories. Check that
   semantically identical entries (e.g., "Hecep Test TM DAKO" and "DAKOHercepTest TM")
   map to the same canonical value.
2. **ER positivity scale other**: Should produce multiple typed columns. Verify:
   - Allred scores are integers 0-8
   - H-scores are integers 0-300
   - fmol/mg values are numeric
   - No cross-contamination between sub-columns
3. **Brachytherapy rename**: Column header changed, values untouched.
4. **Row count**: Output CSV has same number of rows as input (1108).

### Step 4: Update downstream references

After producing the cleaned CSV:
- Downstream notebooks (NB02, NB03) should load `brca_tcga_clinical_data_cleaned.csv`
  instead of the raw file
- The cleaned ER sub-columns and HER2 method column become available as covariates

---

## Downstream Leverage

| Cleaned Column | Use In | How |
|----------------|--------|-----|
| `her2_method_cleaned` | NB03/NB04 discordant analysis | Stratify discordant patients by test platform; cross-tab with IHC/RNA concordance |
| `er_allred_score` | NB03 ML features | Ordinal ER covariate (0-8) for HER2 prediction models |
| `er_h_score` | NB03 ML features | Continuous ER covariate (0-300) for HER2 prediction models |
| `er_fmol_mg` | NB02 clinical characterization | Historical assay results; flag patients tested with older methods |
| `er_intensity` | NB03 ML features | Ordinal intensity (Weak/Moderate/Strong) as categorical feature |
| `ER positivity scoring method` | NB02/NB04 | Method metadata -- enables per-method stratification of ER scores |

The highest-value downstream application: if HER2-discordant patients (IHC-/RNA-high)
show systematically different ER scoring patterns or were tested with specific HER2
assay platforms, that connects clinical testing methodology to molecular discordance --
a finding with direct implications for IHC QC and T-DXd patient identification.
