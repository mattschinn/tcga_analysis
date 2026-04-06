# Column Cleaner Execution Plan

## Pipeline Fixes (Step 0)

1. Make `analysis_plan_path` and `qc_plan_path` optional in `generate_context()` -- the hardcoded context sections 1-5 are sufficient
2. Update `run_pipeline.py` paths to local project structure
3. Replace Unicode icons in `column_cleaner.py:678-679` with ASCII equivalents
4. Check prompt strings for non-ASCII characters

## Column Triage

### Data Anomaly

**`Brachytherapy first reference point administered total dose`** (218 non-null, 54 unique) -- column name is wrong. Values are ER positivity scoring methods/results ("Allred score 5+3 = 8", "H-SCORE 300", "dextran coated charcoal"). cBioPortal export artifact. Investigate mapping before cleaning.

### Tier 1: Run through the pipeline

| Column | Non-null | Unique | Why | Complexity |
|--------|----------|--------|-----|-----------|
| `ER positivity scale other` | 244 | 65 | Multiple incommensurable scales (Allred, H-score, fmol/mg, intensity, %). Key ER covariate for HER2 analysis. | High -- column split |
| `HER2 positivity method text` | 90 | 20 | Spelling variants of ~5 test methods. Assay platform may explain HER2 discordance. | Low -- dedup |

### Tier 2: Do if time permits

| Column | Non-null | Unique | Why | Complexity |
|--------|----------|--------|-----|-----------|
| `PR positivity scale other` | 223 | 66 | Same measurement-system-mix as ER. Weaker covariate but used in PAM50. | High -- column split |
| `HER2 fish method` | 50 | 30 | FISH protocol spelling variants. Low n. | Low -- dedup |

### Skip

| Column | Reason |
|--------|--------|
| `Tumor Other Histologic Subtype` | 30 values, all = "mixed ductal/lobular". One-liner fix. |
| `First surgical procedure other` | Downstream of diagnosis. Low HER2 value. |
| `Staging System.1` | 20 values. Low impact. |
| `First Pathologic Diagnosis Biospecimen Acquisition Other Method Type` | 64 values. Biopsy method irrelevant to HER2. |
| `Surgery for positive margins other` | 14 values. Post-treatment. |
| All `Nte *` columns | <15 non-null each. Insufficient data. |

## Execution Order

1. **Quick win: `HER2 positivity method text`** -- validates pipeline, context guidance exists
2. **High-value: `ER positivity scale other`** -- hardest case, highest downstream value
3. **Investigate Brachytherapy column** -- data provenance issue, not pipeline
4. **(If time) `PR positivity scale other`** -- mirror ER approach, add context guidance

## Downstream Value

Clean ER/PR scoring columns become covariates in discordant subgroup analysis. If IHC-negative/RNA-high patients have different ER scoring patterns or HER2 assay platforms, that is a finding connecting to T-DXd eligibility narrative.
