"""
Context Generator
=================
Auto-generates the CONTEXT document from project files for the column cleaning pipeline.

The CONTEXT provides agents with:
- Project objective (HER2 analysis in TCGA BRCA)
- Relevant biology (HER2/ERBB2, breast cancer subtypes, clinical testing)
- Clinical workflow context (IHC, FISH, Allred scoring, H-scoring)
- Dataset schema (what columns exist, how they relate)
- Column-specific guidance (what each column means, expected value types)
"""

import pandas as pd


def generate_context(
    analysis_plan_path: str,
    qc_plan_path: str,
    clinical_csv_path: str,
    column_name: str | None = None,
) -> str:
    """
    Generate a CONTEXT document for the column cleaning pipeline.

    Args:
        analysis_plan_path: Path to the analysis_plan.md file.
        qc_plan_path: Path to the qc_plan.md file.
        clinical_csv_path: Path to the clinical data CSV.
        column_name: If provided, include column-specific guidance.

    Returns:
        A markdown string with full context.
    """
    # Load project docs
    with open(analysis_plan_path) as f:
        analysis_plan = f.read()
    with open(qc_plan_path) as f:
        qc_plan = f.read()

    # Load clinical data schema
    df = pd.read_csv(clinical_csv_path, nrows=5)
    all_columns = df.columns.tolist()

    # Build the column-specific block
    column_block = ""
    if column_name:
        column_block = _get_column_guidance(column_name)

    context = f"""
# PROJECT CONTEXT FOR DATA CLEANING AGENTS

## 1. Project Objective

This is a Tempus Pharma R&D coding challenge analyzing HER2 (ERBB2) in the TCGA
breast cancer cohort. The goal is to identify HER2-positive patients using clinical
(IHC/FISH), genomic (copy number), and transcriptomic (RNA-Seq) data, and to discover
biologically distinct patient subsets via unsupervised learning and machine learning.

Downstream analyses include:
- Constructing a ground-truth HER2 label from IHC and FISH clinical data
- Comparing RNA vs DNA predictive power for clinical HER2 status
- Unsupervised clustering to find PAM50-like molecular subtypes
- ML classification with SHAP-based feature importance
- Pathway analysis of HER2-enriched subsets

**Data quality directly impacts every downstream analysis.** Messy clinical labels
mean noisy ground-truth for supervised learning. Messy categorical fields mean
unreliable stratification for subgroup analyses.

## 2. Relevant Biology

### HER2/ERBB2
- ERBB2 (HER2) is an oncogene on chromosome 17q12, commonly amplified in ~15-20%
  of breast cancers.
- HER2-positive tumors are aggressive but targetable (trastuzumab, pertuzumab, T-DXd).
- HER2 status is clinically determined by immunohistochemistry (IHC) and fluorescence
  in situ hybridization (FISH).

### Clinical HER2 Testing (ASCO/CAP Guidelines)
- **IHC scoring**: 0 (negative), 1+ (negative), 2+ (equivocal, reflex to FISH), 3+ (positive)
- **FISH**: Measures ERBB2 gene copy number relative to CEP17. Positive if ratio ≥ 2.0
  or absolute copies ≥ 6.
- Composite: IHC 3+ = positive regardless of FISH. IHC 2+ requires FISH confirmation.
- **HER2-low** (IHC 1+ or 2+/FISH-) is an emerging clinical category relevant for
  antibody-drug conjugates like trastuzumab deruxtecan.

### Estrogen Receptor (ER) Testing
- ER status is determined by IHC on tumor tissue.
- **Allred score**: Semi-quantitative, combines proportion score (0-5) + intensity
  score (0-3) = total 0-8. Score ≥ 3 is positive.
- **H-score**: Quantitative, ranges 0-300. Calculated as: 1×(% cells 1+) + 2×(% cells 2+)
  + 3×(% cells 3+). Positive threshold varies (often ≥ 1 or ≥ 10).
- **fmol/mg (femtomoles per milligram protein)**: Older biochemical ligand-binding
  assay (dextran-coated charcoal method). Largely replaced by IHC. Positive if ≥ 10 fmol/mg.
- **Qualitative intensity**: Weak/Moderate/Strong — refers to IHC staining intensity,
  which is one component of the Allred score.
- **Percentage positive**: Proportion of tumor cells staining positive, another
  component used in scoring.
- ER and HER2 are typically anti-correlated in breast cancer (most HER2+ tumors are
  ER-negative, most ER+ tumors are HER2-negative, though overlap exists).

### Breast Cancer Subtypes (PAM50)
- Luminal A: ER+/PR+/HER2-, low proliferation
- Luminal B: ER+/PR±/HER2±, higher proliferation
- HER2-enriched: ER-/HER2+
- Basal-like: ER-/PR-/HER2- (triple-negative)
- Normal-like: resembles normal breast tissue

## 3. Clinical Data Schema

The clinical dataset has {len(all_columns)} columns. Key column groups:

**HER2-related:**
- `IHC-HER2`: Pre-coded Positive/Negative/Equivocal/Indeterminate
- `HER2 ihc score`: 0, 1+, 2+, 3+
- `HER2 fish status`: Positive/Negative/Equivocal
- `HER2 positivity method text`: Free-text describing test method used
- `HER2 cent17 ratio`, `HER2 copy number`: Quantitative FISH results

**ER/PR-related:**
- `ER Status By IHC`: Positive/Negative
- `ER positivity scale other`: Free-text with various scoring systems (Allred, H-score, fmol/mg, etc.)
- `PR status by ihc`: Positive/Negative

**Histology:**
- `Tumor Other Histologic Subtype`: Free-text for non-standard histologic types

**Surgery:**
- `First surgical procedure other`: Free-text surgical procedure descriptions

**Other clinical:**
- Staging (AJCC stage, T/N/M), demographics, diagnosis year, tissue source site

All columns: {', '.join(all_columns[:50])}{'...' if len(all_columns) > 50 else ''}

## 4. Data Quality Context

This is real TCGA clinical data with known messiness:
- Free-text fields were entered by different institutions with no controlled vocabulary
- Multiple measurement systems may coexist in the same column
- Typos, casing inconsistencies, and abbreviation variants are expected
- The cohort spans 2001-2013, so testing practices and reporting standards evolved
- Missing data is extensive (~80% of rows lack HER2 annotations)

## 5. Downstream Use Requirements

Cleaned columns will be used for:
- **Stratification**: Grouping patients by clinical features for subgroup analysis
- **Covariates in ML models**: As features in XGBoost/logistic regression for HER2 prediction
- **Cross-tabulation**: Comparing distributions across HER2+/HER2- groups
- **Quality assessment**: Validating molecular labels against clinical annotations

Columns should be:
- Machine-readable (consistent categorical values or clean numerics)
- Biologically meaningful (preserve clinically relevant distinctions)
- Appropriately granular (not so coarse as to lose information, not so fine as to be sparse)

{column_block}
"""
    return context.strip()


def _get_column_guidance(column_name: str) -> str:
    """Return column-specific cleaning guidance."""
    guidance = {
        "ER positivity scale other": """
## 6. Column-Specific Guidance: ER positivity scale other

This column contains the ER positivity scoring details when a non-standard or
supplementary scale was used. It is a FREE-TEXT FIELD with MULTIPLE MEASUREMENT
SYSTEMS intermixed:

1. **Allred scores** (0-8): Look for "Allred score 7", "allred scrore = 8", etc.
   These are ordinal. Range 0-8. Positive if ≥ 3.
2. **H-scores** (0-300): Look for bare numbers like "120", "240 H", "H-Score", etc.
   Numbers > 8 and ≤ 300 are almost certainly H-scores. Positive if ≥ 1 (or ≥ 10).
3. **fmol/mg concentrations**: Look for "288 fmol/mg", "49 fmol/mg protein".
   Biochemical assay result. Positive if ≥ 10 fmol/mg.
4. **Qualitative intensity**: "Strong", "Moderate", "Weak", "Intensity=Strong".
   IHC staining intensity descriptor.
5. **Percentages**: "0%", ">10%", "10-75%", ">75%". Proportion of cells positive.
6. **Method references**: "Oncotype Dx test", "Two-tier", "dextran coated charcoal".
   These describe the method, not the result.

The RIGHT approach is to SPLIT this into separate columns by measurement system,
because these are incommensurable — you cannot meaningfully harmonize an Allred
score of 7 with an H-score of 240 into one scale. Each has different ranges,
different clinical thresholds, and different biological meaning.

For downstream analysis (ML, stratification), binary ER status (positive/negative)
can be derived from each system using its own threshold. The existing `ER Status By IHC`
column already provides this binary call for most patients, so the detailed scores
here are supplementary.
""",
        "HER2 positivity method text": """
## 6. Column-Specific Guidance: HER2 positivity method text

This column describes the method/kit used for HER2 positivity testing. It is a
FREE-TEXT FIELD with a SMALL number of distinct concepts but spelling/formatting variants:

Key concepts:
- **Dako HercepTest**: An FDA-approved IHC assay kit for HER2. Multiple spelling
  variants exist ("Dako Hercept Test", "DAKOHercepTest TM", "HercepTest TM Dako", etc.)
- **CISH**: Chromogenic in situ hybridization — an alternative to FISH for detecting
  HER2 gene amplification.
- **CAP scoring guidelines 2010**: Reference to College of American Pathologists
  scoring criteria.
- **Ventana**: Another IHC platform/kit for HER2 testing (note: "Venten" is a typo).
- Other: "+ or -" (describes result format), descriptive text about scoring criteria.

This should harmonize to a SINGLE categorical column with a controlled vocabulary
of test methods. The key distinction is the assay platform/method, not the specific
text formatting.
""",
        "First surgical procedure other": """
## 6. Column-Specific Guidance: First surgical procedure other

This column describes the first surgical procedure when it doesn't fit standard
TCGA categories. It is a FREE-TEXT FIELD with compound procedure descriptions.

Key procedure categories:
- **Mastectomy types**: Total, partial, segmental, modified radical, skin-sparing,
  nipple-sparing, bilateral. Laterality (left/right) is sometimes included.
- **Excision/biopsy types**: Excisional biopsy, wide local excision, needle-directed,
  wire-localized, fine needle aspiration.
- **Lymph node procedures**: Often appended to the primary procedure — sentinel node
  biopsy, axillary dissection, axillary node biopsy. These are clinically important.
- **Reconstruction**: Sometimes mentioned alongside mastectomy.
- **Compound entries**: Many entries describe procedure + lymph node work + sometimes
  reconstruction in one string.

For downstream analysis, consider splitting into:
1. Primary procedure category (mastectomy type vs. excision/biopsy type)
2. Whether lymph node procedure was performed (and what type)
Laterality (left/right) is generally not needed for this analysis.
""",
        "Tumor Other Histologic Subtype": """
## 6. Column-Specific Guidance: Tumor Other Histologic Subtype

This column captures histologic subtypes that don't fit standard TCGA categories.
In this dataset, ALL 30 non-null values are variants of "mixed ductal and lobular
carcinoma" — just with different phrasing.

This should harmonize to a SINGLE categorical value. All entries describe the same
histologic entity: a tumor with both ductal and lobular features (also called
"mixed infiltrating ductal and lobular carcinoma" in WHO classification).
""",
    }

    return guidance.get(column_name, f"""
## 6. Column-Specific Guidance: {column_name}

No specific guidance available for this column. The agents should analyze the
values and context to determine the appropriate cleaning strategy.
""")
