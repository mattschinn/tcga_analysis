# Tempus HER2 Coding Challenge — Revision & Execution Plan

**Author:** Mat Schinn, Ph.D.
**Date:** April 2026
**Status:** Pre-submission revision plan

---

## 1. Notebook Architecture & Refactoring

### 1.1 Structure

Refactor the monolithic notebook into five notebooks plus a shared utilities module.
Each notebook is self-contained: it loads upstream intermediates from `outputs/`,
runs its analysis, saves its own intermediates and figures, and can be executed
independently after Notebook 01 has run.

```
project/
├── data/                            # Raw data (unchanged)
│   ├── brca_tcga_clinical_data.csv
│   ├── tcga.brca.rsem.csv
│   └── brca_tcga_erbb2_copy_number.csv
├── outputs/                         # Intermediates and figures
│   ├── 01_clinical_qc.parquet       # Cleaned clinical with her2_composite, pid, flags
│   ├── 01_tumor_raw_filtered.parquet  # Gene-filtered tumor RSEM (not normalized)
│   ├── 01_normal_raw_filtered.parquet # Adjacent normal samples (gene-filtered)
│   ├── 01_tumor_norm.parquet        # Normalized tumor expression matrix
│   ├── 01_cn_qc.parquet             # Copy number with pid
│   ├── 01_gene_cols.json            # List of gene columns post-filtering
│   ├── 01_size_factors.parquet      # Per-sample size factors (for reference)
│   ├── 02_multimodal_cohort.parquet # Merged clinical + RNA + CN (cohort_c)
│   ├── 02_analysis_df.parquet       # Analysis-ready with ERBB2_expr, key genes
│   ├── 02_discordant_cases.parquet  # IHC/FISH/RNA/CN discordant patients
│   ├── 03_cluster_assignments.parquet  # Patient × cluster labels (k=2, k=4, k=5)
│   ├── 03_pca_embeddings.parquet    # PCA coordinates for all samples
│   ├── 03_umap_embeddings.parquet   # UMAP coordinates
│   ├── 04_ml_predictions.parquet    # Per-patient predicted probabilities (all models)
│   ├── 04_feature_importance.parquet # SHAP + gain-based importance
│   ├── 04_equivocal_scores.parquet  # ML scores for equivocal patients
│   └── figures/                     # Publication-ready figures (PDF + PNG)
├── src/
│   └── utils.py                     # Shared functions (normalization, label construction, plotting)
├── 01_QC_and_Normalization.ipynb
├── 02_HER2_Identification_and_Subsets.ipynb
├── 03_Unsupervised_Clustering.ipynb
├── 04_Machine_Learning.ipynb
├── 05_Deep_Dive_and_Clinical.ipynb
└── README.md                        # AI workflow description, methods summary
```

### 1.2 Shared Utilities (`src/utils.py`)

Extract these reusable functions from the current notebook:

- `construct_her2_label(row)` and `apply_her2_labels(df)` — HER2 composite label logic
- `upper_quartile_normalize(df, gene_cols)` — UQ normalization
- `deseq2_size_factors(df, gene_cols)` — median-of-ratios size factors
- `to_patient_id(id_str)` — 12-char TCGA barcode truncation
- `pca_libsize_analysis(...)` — PCA vs. library size diagnostic
- `classify_her2_spectrum(row)` — HER2-0 / HER2-low / HER2-positive
- Plotting helpers: consistent color palettes, figure export to `outputs/figures/`
- Data I/O wrappers: `save_intermediate(df, name)`, `load_intermediate(name)`

### 1.3 Intermediate Data Contracts

Each parquet file is the contract between notebooks. Key conventions:
- All patient-level data indexed or keyed by `pid` (12-char TCGA ID)
- Expression data stored wide (samples × genes), with `pid` as a column (not index)
- Metadata columns prefixed or clearly separated from gene columns
- JSON sidecar for gene column lists to avoid inferring them from data

---

## 2. Notebook 01: QC & Normalization

### 2.1 Clinical QC (keep what works)

The existing clinical QC is strong. Retain:
- Encoding harmonization and TCGA sentinel value mapping
- Missingness audit with visualization
- Structured missingness analysis (HER2 label availability vs. diagnosis year)
- HER2 composite label construction (tiered ASCO/CAP logic with contradiction flagging)
- Temporal context analysis (guideline drift)

**Add:**
- **TSS extraction.** Parse tissue source site from TCGA barcode (characters 6–7 of
  `bcr_patient_barcode`). Add as a column to the clinical dataframe. This is needed
  for batch effect assessment downstream.
- **Outcome field audit.** Before saving, check completeness of `Overall Survival (Months)`,
  `Overall Survival Status`, `Disease Free (Months)`, `Disease Free Status`. Report
  event counts. This determines whether Notebook 05 is feasible.

### 2.2 RNA-Seq QC

**Retain:**
- Library size computation and distribution analysis
- Gene filtering by zero-expression frequency

**Change: gene filtering threshold from 90% → 50%.**

Rationale: At 90%, genes expressed in as few as ~109/1093 samples are retained. This
adds noise to unsupervised clustering and inflates the feature space without contributing
reliable signal. At 50%, we retain ~17,600 genes (based on the sensitivity table showing
2,865 genes with >50% zeros). This is still comprehensive for ML, while substantially
reducing noise for clustering.

Implementation note: After filtering, explicitly verify that key HER2 pathway genes
survive the cut: ERBB2, GRB7, ESR1, PGR, MKI67, EGFR, ERBB3, TOP2A. If any are lost,
exempt them from the filter and document why.

**Change: normalization method.**

The current approach applies UQ normalization to data that appears pre-normalized
(size factors ≈ 1.0). This is effectively a no-op before the log transform. Two options:

**Option A (preferred if time permits): Use pyDESeq2 for VST.**
```python
from pydeseq2.dds import DeseqDataSet
from pydeseq2.preprocessing import deseq2_norm_transform
# ... fit on integer-rounded RSEM counts, extract VST-transformed matrix
```
VST (variance-stabilizing transform) is the gold standard for RNA-seq data entering
ML/clustering pipelines. It handles library size normalization and variance stabilization
in one step. Document that RSEM expected counts are rounded to integers for this purpose
(standard practice; see DESeq2 vignette).

**Option B (faster, defensible): Explicit log2(x+1) with documentation.**
If the data truly is pre-normalized (CV of library sizes = 7.3%), state this explicitly:
> "RSEM expected counts show minimal library-size variation (CV = 7.3%), indicating
> upstream normalization. We apply log2(x+1) for variance stabilization only.
> We verified that PC1–library size correlation (r = 0.46) is driven by biological
> composition (subtype effects), not technical depth artifacts, by showing that
> removing 17q12 amplicon genes does not attenuate the correlation."

Either way, the key is to **explicitly justify the choice** rather than applying UQ
normalization that demonstrably does nothing.

**Add: mean-variance diagnostic.**

Retain the mean-variance relationship plot (Cell 42). It correctly shows overdispersion
consistent with negative binomial. Add a sentence: "This overdispersion confirms that
Poisson-based methods are inappropriate and motivates NB-aware normalization (DESeq2/edgeR)
or distribution-agnostic approaches (UQ + log)."

### 2.3 Batch Effect Assessment

**Add: TSS-level batch effect check.**

After PCA, color PC1 vs. PC2 by tissue source site (extracted from barcode). Compute
ANOVA F-statistic for association between each top PC and TSS. If TSS explains significant
variance (F-test p < 0.01 on any of the top 5 PCs), note this as a caveat for clustering.

Decision framework:
- If TSS is associated with PCs but NOT confounded with HER2 status (χ² test, TSS × HER2),
  note as caveat but proceed. Tree-based ML models are relatively robust to batch effects
  that are orthogonal to the outcome.
- If TSS IS confounded with HER2 status, consider ComBat correction (from `combat` or
  `pycombat` packages) before clustering, with explicit documentation.
- For the supervised ML, batch effects are less concerning IF they don't create leakage.
  Cross-validate by patient, not by sample (already done).

### 2.4 Outputs

Save to `outputs/`:
- `01_clinical_qc.parquet`: Full clinical dataframe with `pid`, `her2_composite`,
  `her2_source`, `her2_flag`, `tss` (tissue source site), `dx_year`, cleaned fields.
- `01_tumor_raw_filtered.parquet`: Gene-filtered tumor RSEM counts (pre-normalization),
  with `pid` column. Used for potential re-normalization downstream.
- `01_normal_raw_filtered.parquet`: Adjacent normal samples, same gene set. For Notebook 05.
- `01_tumor_norm.parquet`: Normalized (VST or log2) expression matrix with `pid`.
- `01_cn_qc.parquet`: Copy number data with `pid`.
- `01_gene_cols.json`: List of gene column names after filtering.
- `01_size_factors.parquet`: Per-sample size factors (UQ and/or DESeq2) for documentation.

---

## 3. Notebook 02: HER2 Identification & Subsets

### 3.1 Dataset Merging

Load intermediates from Notebook 01. Merge clinical, RNA, and CN on `pid`. Document
patient counts at each merge step (existing logic is fine).

Define cohorts:
- **Cohort A:** Patients with resolved Positive/Negative HER2 label AND multimodal data
- **Cohort B:** Full clinical cohort (for clinical-only analyses)
- **Cohort C:** All patients with multimodal data (including Equivocal and unlabeled)

### 3.2 ERBB2-Specific Analysis (foregrounded)

This section directly addresses the assignment's central question and needs to be
the most prominent analysis in the notebook.

**3.2a. ERBB2 expression distribution by HER2 status.**

Violin + strip plot of normalized ERBB2 expression, stratified by `her2_composite`
(Positive / Negative / Equivocal). Annotate with Mann-Whitney U p-value for Positive
vs. Negative. Show individual points for Equivocal cases.

Key observations to make:
- Separation between Positive and Negative (expected: large)
- Where Equivocal cases fall (expected: intermediate, heterogeneous)
- Any overlap zone (defines the "gray area" where RNA alone is ambiguous)

**3.2b. ERBB2 CN vs. IHC concordance.**

Crosstab of GISTIC copy number (-2 to +2) × IHC-HER2 status. Report concordance rates:
- What fraction of GISTIC +2 (high amplification) are IHC-Positive?
- What fraction of IHC-Positive are GISTIC +2?
- Where do the discordances concentrate?

Visualize as heatmap or stacked bar chart.

**3.2c. RNA vs. DNA predictiveness (existing analysis, cleaned up).**

Retain the logistic regression comparison (RNA-only, CN-only, RNA+CN). This already
works well: RNA AUC 0.837 > CN AUC 0.808, combined 0.836.

**Add subgroup analysis:** Among IHC 2+ (equivocal) patients specifically, does RNA
expression better predict FISH outcome than CN alone? This addresses the known
literature finding that RNA adds sensitivity for equivocal cases.

```python
# Subset to IHC 2+ patients with FISH data
equivocal_with_fish = analysis_df[
    (analysis_df['HER2 ihc score'].isin(['2+', '2', 2.0])) &
    (analysis_df['HER2 fish status'].isin(['Positive', 'Negative']))
]
# Compare RNA vs CN for predicting FISH outcome in this subset
```

**3.2d. Discordant case characterization.**

Identify and tabulate discordant cases:

| Discordance Type | Definition | Expected Biology |
|---|---|---|
| IHC+/RNA-low | IHC Positive but ERBB2 expression below median of Negatives | Post-translational upregulation? Antibody artifact? |
| IHC-/RNA-high | IHC Negative but ERBB2 expression above median of Positives | Missed by IHC? Transcriptionally active but low protein? |
| IHC 3+/FISH- | IHC 3+ but FISH Negative | Polysomy 17 (centromere gain without ERBB2 amplification) |
| IHC 0-1+/FISH+ | IHC 0 or 1+ but FISH Positive | Amplification without overexpression (epigenetic silencing?) |
| CN-high/RNA-low | GISTIC +2 but low ERBB2 RNA | Gene amplification without transcription |
| CN-low/RNA-high | GISTIC ≤0 but high ERBB2 RNA | Transcriptional upregulation without amplification |

For each type, report N, examine GRB7 co-expression (as a 17q12 amplicon control),
and flag for downstream survival analysis.

Save `02_discordant_cases.parquet` with columns: `pid`, `discordance_type`, `ERBB2_expr`,
`erbb2_copy_number`, `her2_composite`, `HER2 ihc score`, `HER2 fish status`.

### 3.3 Outputs

- `02_multimodal_cohort.parquet`: Cohort C with all merged fields
- `02_analysis_df.parquet`: Analysis-ready dataframe with `ERBB2_expr`, key gene
  expression columns, CN, clinical labels
- `02_discordant_cases.parquet`: Discordant case table

---

## 4. Notebook 03: Unsupervised Clustering

### 4.1 Feature Selection

Use top variable genes by MAD (median absolute deviation). Retain the existing approach
but with the tighter gene filter from Notebook 01 (~17,600 genes after 50% zero filter).
Select top 2,000–3,000 by MAD for clustering.

### 4.2 Dimensionality Reduction

**PCA:** Compute on standardized expression. Determine number of PCs by cumulative
variance explained (retain PCs explaining ≥90% of variance, or cap at 20, whichever
is smaller). Report scree plot.

**UMAP:** 2D embedding on top PCs for visualization. Use consistent hyperparameters
(n_neighbors=15, min_dist=0.1, random_state=42).

### 4.3 Clustering

**K-means** with silhouette-guided k selection (k = 2–7).

**Add: consensus clustering or stability analysis.**
Run k-means 100 times with different random seeds. Compute co-association matrix
(fraction of times each pair of samples clusters together). This addresses the concern
that k-means with a single seed may produce unstable assignments. If consensus is high
(>0.8 for most pairs), the clusters are robust.

**Test both silhouette-optimal k and biology-motivated k values:**
- k=2: likely ER+/ER- split
- k=4: closest to PAM50 subtypes (Luminal A, Luminal B, HER2-enriched, Basal)
- k=5: PAM50 + Normal-like

### 4.4 Cluster Characterization

For each cluster, report:
- Size (N and %)
- HER2 enrichment (Fisher's exact test, odds ratio)
- ER/PR status distribution
- Mean ERBB2 expression
- Mean fraction genome altered (FGA) — a genomic instability proxy
- Attempt to assign PAM50-like labels based on marker expression:
  - High ESR1 + low MKI67 → Luminal A-like
  - High ESR1 + high MKI67 → Luminal B-like
  - High ERBB2 + low ESR1 → HER2-enriched-like
  - Low ESR1 + low ERBB2 + high EGFR → Basal-like

**Add: TSS distribution per cluster.** If any cluster is dominated by a single TSS,
flag as potentially batch-driven rather than biologically meaningful.

### 4.5 Outputs

- `03_cluster_assignments.parquet`: pid × cluster labels for each k
- `03_pca_embeddings.parquet`: pid × PC1–PC20
- `03_umap_embeddings.parquet`: pid × UMAP1–UMAP2

---

## 5. Notebook 04: Machine Learning

### 5.1 Feature Matrix Construction

Load multimodal cohort from Notebook 02. Build feature matrix:
- All normalized gene expression values (prefixed `expr_`)
- ERBB2 copy number
- ER status (binary)
- PR status (binary)

Target: `her2_composite` ∈ {Positive, Negative} → binary `y`.

### 5.2 Model Training

**Models:** L1-Logistic Regression, Random Forest, XGBoost. Existing setup is fine.

**Cross-validation:** Stratified 5-fold. Use `cross_val_predict` for out-of-fold
probabilities (already done).

**Add: nested CV for hyperparameter tuning (if time permits).**
Outer loop: 5-fold for evaluation. Inner loop: 3-fold for tuning. This avoids the
current setup where hyperparameters (C=1.0, max_depth=4, etc.) are fixed and
potentially suboptimal. If not done, explicitly state that hyperparameters were
chosen based on common defaults and not optimized.

### 5.3 Feature Importance

**Fix: install SHAP and use it.**

The current notebook falls back to gain-based importance because SHAP isn't installed.
This is a significant weakness — gain-based importance in XGBoost is biased toward
high-cardinality features and is not reliable for biological interpretation.

```bash
pip install shap
```

After SHAP analysis, address the biological plausibility of top features:
- If ERBB2 and GRB7 are in the top 5 (expected with SHAP), this validates the model.
- If unexpected genes dominate, investigate: are they on 17q12? Are they subtype markers
  being used as proxies? Are they noise from overfitting?
- Cross-reference top features against MSigDB HER2 pathway gene sets.

### 5.4 Calibration Analysis

Retain calibration curves and confusion matrix. Add:
- **Threshold sweep** with explicit sensitivity/specificity tradeoff table.
- **Clinical decision framing:** At what threshold does the model achieve 90% sensitivity
  (detecting 90% of HER2+ patients)? What is the corresponding false positive rate?
  Is this acceptable for a screening application?

### 5.5 Equivocal Sample Scoring

**New analysis (critical addition).**

After training on Positive/Negative patients, score the Equivocal patients:

```python
# Load equivocal patients
equivocal = cohort_c[cohort_c['her2_composite'] == 'Equivocal']

# Build feature matrix (same columns as training)
X_equiv = equivocal[feature_cols].values

# Score with best model (XGBoost, trained on full Pos/Neg data)
equiv_probs = xgb_full.predict_proba(X_equiv)[:, 1]

# Report distribution
print(f"Equivocal patients (n={len(equivocal)}):")
print(f"  Predicted P(HER2+) > 0.5: {(equiv_probs > 0.5).sum()}")
print(f"  Predicted P(HER2+) > 0.7: {(equiv_probs > 0.7).sum()}")
print(f"  Predicted P(HER2+) < 0.3: {(equiv_probs < 0.3).sum()}")
```

This directly demonstrates the clinical utility of RNA-based classification for
the ambiguous population — exactly what the challenge is asking for.

Save predicted probabilities for all patients (labeled + equivocal + unlabeled)
to `04_ml_predictions.parquet`.

### 5.6 Outputs

- `04_ml_predictions.parquet`: pid, y_true (or NaN), y_prob (all models), y_pred
- `04_feature_importance.parquet`: feature_name, shap_mean_abs, xgb_gain
- `04_equivocal_scores.parquet`: pid, predicted probability, feature values

---

## 6. Notebook 05: Deep Dive & Clinical Outcomes

This notebook addresses the "if time permits" section of the challenge plus the
clinical outcome validation plan.

### 6.1 Normal vs. Tumor Expression

**Fix the bug:** The current notebook loses ERBB2 from the gene list by Section 5.
This is a data flow issue. Load `01_tumor_norm.parquet` and `01_normal_raw_filtered.parquet`
fresh, normalize the normal samples with the same method, and compare.

**Analysis:**
- Paired comparison where possible (same patient has both tumor and normal)
- Focus on ERBB2, GRB7, ESR1, PGR, MKI67, EGFR
- Volcano plot or bar chart of log2 fold change (tumor/normal) with p-values
- Define an "overexpression threshold" as mean + 2SD of normal ERBB2 expression.
  How many HER2-negative patients exceed this threshold?

### 6.2 HER2-Low Exploration

**Fix:** The current `classify_her2_spectrum` function produces only 3 categories
because IHC score parsing fails for some formats. Debug and ensure proper mapping:
- IHC 0 → HER2-0
- IHC 1+ → HER2-Low
- IHC 2+ / FISH- → HER2-Low
- IHC 2+ / FISH+ or IHC 3+ → HER2-Positive

**Analysis:**
- Distribution of ERBB2 RNA expression across the HER2 spectrum (HER2-0, HER2-Low,
  HER2-Positive). Violin plot.
- Do HER2-Low patients form a distinct cluster, or are they distributed across
  luminal/basal clusters?
- ML model probability distribution for HER2-Low patients — does the model see
  them as "a little HER2-positive" or "basically negative"?

### 6.3 Discordant Case Biology

Using the discordant cases identified in Notebook 02:
- For IHC-/RNA-high patients: check GRB7 expression (if co-elevated, supports
  true HER2 biology rather than noise), check ERBB2 CN, check cluster membership
  (do they cluster with HER2-enriched group?)
- For IHC+/RNA-low patients: check for known HER2 pathway activation signatures
  (e.g., phospho-ERK, proliferation gene sets) — are they truly HER2-driven?

### 6.4 Contradiction Pattern Deep Dive

Expand on the 199 label contradictions from the HER2 label construction:

**IHC 3+ / FISH-:**
- Report N. Examine ERBB2 RNA expression — does it look like true HER2+ or intermediate?
- Check `Cent17 Copy Number` or `HER2 cent17 ratio` where available. Polysomy 17 would
  show elevated centromere signal, distinguishing from true amplification.
- Clinical implication: These patients may receive trastuzumab based on IHC alone despite
  lacking true amplification. RNA testing could prevent unnecessary treatment.

**IHC 0-1+ / FISH+:**
- Report N. Examine ERBB2 RNA — is gene amplified but not transcribed?
- If RNA is also high, this suggests an IHC antibody/fixation artifact.
- If RNA is low despite CN gain, this is genuine discordance (amplification without
  expression), potentially due to epigenetic silencing.
- Clinical implication: These patients might benefit from HER2-directed therapy
  but are missed by IHC-first testing algorithms.

### 6.5 Clinical Outcome Validation

**Prerequisites:** Check outcome data completeness from Notebook 01 audit. If DFS
has <50% completeness, use OS only. If total events <30, limit to KM curves (no Cox).

**6.5a. Cluster survival (KM curves).**

Stratify by k=4 clusters. Plot DFS KM curves with log-rank test. Expected:
HER2-enriched cluster shows worse DFS than luminal-A-like cluster. If this replicates,
it validates that the unsupervised clustering captures clinically meaningful biology.

**6.5b. Discordant case survival (the key translational analysis).**

Among clinically HER2-negative patients, stratify by molecular concordance:
- "Concordant HER2-negative": IHC-, RNA low, cluster not HER2-enriched
- "Molecularly discordant": IHC- but RNA high OR cluster = HER2-enriched OR
  ML-predicted P(HER2+) > 0.5

Compare DFS between these groups. If discordant patients have worse DFS, this is
direct evidence that IHC misses a clinically relevant population.

**6.5c. Cox regression (hypothesis-generating).**

If events are sufficient (≥30):
- Model 1: DFS ~ molecular_HER2_status (univariable)
- Model 2: DFS ~ molecular_HER2_status + AJCC_stage + ER_status (multivariable)

Report HR and 95% CI. Frame as hypothesis-generating, not definitive. State caveats
about treatment confounding (HER2+ patients received trastuzumab), sample size, and
TCGA not being a clinical trial.

**6.5d. HER2-low survival (if data permits).**

KM curves for HER2-0 vs. HER2-low vs. HER2-positive. Relevant to T-DXd
(trastuzumab deruxtecan), which showed benefit in HER2-low (DESTINY-Breast04).

### 6.6 Population Sizing and Actionability

**This section answers the challenge's call for "new opportunities for patient identification."**

For each identified patient subgroup, report:

**6.6a. Prevalence table.**

| Patient Group | N in Cohort | % of Denominator | Extrapolation (US annual) |
|---|---|---|---|
| Clinically HER2- but molecularly HER2-enriched | ? | ?% of HER2- | ~X,000 patients/year |
| Equivocal resolved to Positive by RNA | ? | ?% of equivocal | ~X,000 patients/year |
| Equivocal resolved to Negative by RNA | ? | ?% of equivocal | ~X,000 patients/year |
| HER2-Low (IHC 1+ or 2+/FISH-) | ? | ?% of cohort | ~X,000 patients/year |
| IHC 3+/FISH- (potential polysomy 17) | ? | ?% of HER2+ | ~X,000 patients/year |

US breast cancer incidence: ~310,000 new cases/year (2024 ACS estimate).
HER2+ prevalence: ~15-20%. HER2-low prevalence: ~50-60% of HER2-negative.

Caveat: TCGA is not population-representative (academic centers, particular demographics).
Prevalence estimates are illustrative, not definitive.

**6.6b. Multi-modal corroboration.**

For the most actionable group (clinically HER2- / molecularly HER2+), verify with
multiple lines of evidence:
- ERBB2 RNA expression elevated (> normal + 2SD threshold)
- GRB7 co-expression (supports 17q12 amplicon biology, not isolated gene upregulation)
- ERBB2 copy number gain (GISTIC ≥ 1)
- Cluster membership (in HER2-enriched cluster)
- ML-predicted P(HER2+) > 0.5

Patients with ≥3/5 concordant molecular signals are the strongest candidates for
reclassification. Report this "confidence tier" alongside the simple prevalence.

**6.6c. Clinical utility framing.**

One paragraph connecting to Tempus's mission: RNA-seq-based HER2 classification could
serve as a reflex test for IHC-equivocal cases (reducing time to treatment decision)
or as a complementary screen that identifies HER2-directed therapy candidates missed
by IHC alone. The population sizing shows this is not a niche finding — if 5-10% of
HER2-negative patients are molecularly HER2-enriched, that represents 20,000–40,000
patients/year in the US who might benefit from HER2-targeted agents but are currently
not identified.

---

## 7. Cross-Cutting Improvements

### 7.1 Clean Up Artifacts

- Remove "asdf" in Cell 67
- Convert Cell 28 from raw to code (or delete if redundant)
- Remove empty cells (89, 90, 108-110)
- Ensure all figure captions are complete (several say "Figure X")
- Number figures sequentially within each notebook

### 7.2 AI Workflow Description (README.md)

The challenge explicitly asks for this. Write a section covering:
- **Tools used:** Claude (via claude.ai and Claude Code), used in both assistant mode
  (iterating on analysis logic, debugging) and agentic mode (code generation for
  boilerplate like plotting functions and data wrangling).
- **Example prompt:** Include 1-2 actual prompts used during development.
- **Validation process:** How AI-generated code was reviewed — e.g., "All normalization
  code was manually verified against DESeq2 documentation. Statistical tests were
  cross-checked by inspecting intermediate values. Biological interpretations were
  validated against published TCGA BRCA analyses (Cancer Genome Atlas Network, Nature 2012)."
- **What AI accelerated:** Boilerplate (plotting, data munging, sklearn pipelines).
- **What required human judgment:** Normalization method choice, HER2 label logic
  (ASCO/CAP guidelines), biological interpretation of features, clinical framing.

### 7.3 Equivocal Sample Handling — Explicit Statement

In Notebook 04, add a dedicated markdown cell:
> "Equivocal samples (n=35) are excluded from model training and evaluation to avoid
> label noise. After training on resolved Positive/Negative cases, we score equivocal
> patients with the trained model to demonstrate RNA-seq's potential to resolve
> ambiguous IHC results. This is the primary clinical use case for molecular
> HER2 classification."

### 7.4 Contradiction Patterns — Separate Handling

In Notebook 02, after discordant case identification:
- Do NOT silently absorb contradictions into binary labels
- Create a `label_confidence` column: "High" (IHC and FISH concordant), "Moderate"
  (single-modality or minor discordance), "Low" (frank contradiction)
- In ML training, optionally test sensitivity to excluding Low-confidence labels

---

## 8. Execution Order and Time Estimates

| Task | Notebook | Estimated Time | Priority |
|---|---|---|---|
| Refactor into 5 notebooks + utils.py | All | 1.5 hours | High |
| Fix gene filtering (90% → 50%) | 01 | 15 min | High |
| Add normalization justification (or implement VST) | 01 | 30–45 min | High |
| Fix normal tissue bug, add ERBB2 normal-vs-tumor | 05 | 30 min | High |
| Add ERBB2 violin plot by HER2 status | 02 | 15 min | High |
| Score equivocal patients with trained model | 04 | 15 min | High |
| Add TSS batch effect check | 01 | 20 min | Medium |
| Install SHAP, rerun feature importance | 04 | 20 min | Medium |
| ERBB2 RNA vs CN subgroup analysis (IHC 2+ subset) | 02 | 30 min | Medium |
| Discordant case characterization table | 02 | 30 min | Medium |
| Contradiction pattern deep dive | 05 | 30 min | Medium |
| KM survival curves (clusters, discordant) | 05 | 45 min | Medium |
| Cox regression (if events sufficient) | 05 | 30 min | Medium |
| Population sizing table | 05 | 20 min | Medium |
| HER2-low exploration fix + analysis | 05 | 30 min | Medium |
| Consensus clustering / stability analysis | 03 | 30 min | Lower |
| Nested CV for hyperparameter tuning | 04 | 45 min | Lower |
| AI workflow write-up (README.md) | README | 20 min | High |
| Clean up artifacts, number figures | All | 20 min | High |

**Total estimated time: ~9–11 hours** for the full plan.

**Minimum viable revision (~4 hours):** Refactor, fix normalization justification,
fix gene filtering, fix normal tissue bug, add ERBB2 violin, score equivocal samples,
install SHAP, write AI workflow description, clean up artifacts.

---

## 9. Key Narrative Threads

The revised submission should tell a coherent story across all five notebooks:

1. **Clinical QC establishes the ground truth** — but reveals that the ground truth
   itself is imperfect (contradictions, guideline drift, missingness).

2. **RNA-seq normalization is handled appropriately** — with explicit justification
   for the method chosen, verified by diagnostics.

3. **ERBB2 RNA outperforms copy number** for predicting IHC-HER2 status, especially
   in the equivocal zone — consistent with published literature and providing the
   molecular basis for RNA-based patient identification.

4. **Unsupervised clustering recovers biologically meaningful subtypes** — validated
   by marker expression, HER2 enrichment, and (if feasible) clinical outcomes.

5. **ML identifies a multi-gene signature** beyond ERBB2 alone — with biologically
   interpretable features (SHAP) and a model that can resolve equivocal cases.

6. **Discordant and equivocal cases represent a clinically actionable population** —
   quantified by prevalence, corroborated by multiple modalities, and (if data permits)
   associated with differential outcomes.

7. **The translational opportunity is concrete:** ~5-10% of HER2-negative patients may
   be molecularly HER2-enriched and candidates for HER2-directed therapy. RNA-seq
   provides the resolution to identify them.
