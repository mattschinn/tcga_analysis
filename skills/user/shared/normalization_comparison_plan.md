# Normalization Comparison Plan: RSEM vs TPM vs TMM

**Objective:** Evaluate whether the current RSEM (upper-quartile, TSS-adjusted) normalization preserves HER2 biological signal better than more aggressive read-depth corrections (TPM, TMM). The deliverable is a standardized report for each normalization, enabling side-by-side comparison by an Analyst agent or human reviewer.

---

## Phase 0: Define the Report Template

Before any analysis, lock down the report structure so every normalization method produces identically formatted outputs. This ensures comparability.

### Report Schema

```
NORMALIZATION COMPARISON REPORT
================================
Method: <RSEM-UQ-TSS | TPM | TMM-edgeR>
Date: <ISO date>
Input: <description of input counts and normalization steps applied>

--- SECTION A: ERBB2 RNA vs Copy Number ---

A1. Pearson r (ERBB2 RNA vs CN, all samples):          ____
A2. Spearman rho (ERBB2 RNA vs CN, all samples):       ____
A3. Pearson r (ERBB2 RNA vs CN, HER2+ only):           ____
A4. Pearson r (ERBB2 RNA vs CN, HER2- only):           ____
A5. Effect size (Cohen's d): HER2+ vs HER2- ERBB2 RNA: ____
A6. Mann-Whitney U p-value: HER2+ vs HER2- ERBB2 RNA:  ____
A7. Fold-change (median ERBB2 RNA: HER2+ / HER2-):     ____

--- SECTION B: Logistic Regression (RNA/CN -> HER2 IHC) ---

B1. AUC-ROC, ERBB2 RNA only (5-fold CV):               ____
B2. AUC-PR,  ERBB2 RNA only (5-fold CV):               ____
B3. AUC-ROC, ERBB2 CN only  (5-fold CV):               ____
B4. AUC-PR,  ERBB2 CN only  (5-fold CV):               ____
B5. AUC-ROC, RNA + CN combined (5-fold CV):             ____
B6. AUC-PR,  RNA + CN combined (5-fold CV):             ____
B7. Delta AUC-ROC (RNA-only minus CN-only):             ____

--- SECTION C: Unsupervised Clustering ---

C1. Silhouette scores for k=2..7:                       [__, __, __, __, __, __]
C2. Best k (argmax silhouette):                         ____
C3. Silhouette at k=4:                                  ____
C4. ARI (k=4 clusters vs HER2 composite label):        ____
C5. ARI (k=4 clusters vs ER status):                   ____

--- SECTION D: Subtype Marker Separation ---

D1. Luminal score spread (IQR across k=4 clusters):    ____
D2. HER2 score spread (IQR across k=4 clusters):       ____
D3. Basal score spread (IQR across k=4 clusters):      ____
D4. Mean subtype-score gap (max cluster - min cluster): ____
D5. Fraction of HER2+ patients in HER2-enriched cluster: ____

--- SECTION E: Normalization Diagnostics ---

E1. CV of median gene expression across TSS sites:     ____
E2. Top-1 PC correlation with TSS (Kruskal-Wallis p):  ____
E3. Top-1 PC correlation with read depth proxy (r):    ____
E4. ERBB2 expression CV (within HER2+ group):          ____

--- PLOTS (saved to reports/norm_comparison/<method>/) ---
P1. ERBB2 RNA vs CN scatter (colored by HER2 status)
P2. ROC curves overlay (RNA, CN, combined)
P3. Silhouette profile across k
P4. UMAP/PCA colored by k=4 clusters
P5. Marker gene heatmap by cluster
```

### Why These Metrics

| Section | What it captures |
|---------|-----------------|
| A | Raw biological concordance between RNA and DNA for HER2 locus |
| B | Predictive power of RNA for clinical HER2 -- the primary use case |
| C | How well transcriptomic structure supports known subtypes |
| D | Whether marker genes separate cleanly across clusters |
| E | Whether technical artifacts (TSS batch, read depth) leak into the data |

**Key tension:** HER2 biology shifts transcriptome composition. Aggressive normalization (TPM/TMM) may reduce technical noise BUT also compress real biological signal in Sections A-D. Section E checks whether technical artifacts remain problematic under each method.

---

## Phase 1: Extract RSEM Baseline Report from Notebook 02a

**Goal:** Run Notebook 02a as-is and populate the report template for the RSEM-UQ-TSS method.

### Steps

1. **Locate project repository and data.**
   - Identify the project root containing `src/utils.py`, `intermediates/`, and `notebooks/`.
   - Confirm availability of: `01_clinical_qc`, `01_tumor_norm`, `01_cn_qc`, and `gene_cols`.
   - If the repo is not local, obtain it or reconstruct the data pipeline.

2. **Audit Notebook 02a for existing metrics.**
   - Many metrics (B1-B6, C1-C3, P1-P5) are already computed in Notebook 02a.
   - Catalog which report fields can be extracted directly vs. which need new code.
   - Expected gaps: A1-A7 (partial), C4-C5 (ARI not computed), D1-D5 (subtype score spreads not explicitly reported), E1-E4 (not computed).

3. **Write extraction script: `scripts/extract_rsem_report.py`.**
   - Import the same data as Notebook 02a (`load_intermediate`).
   - Re-run the exact same analysis pipeline (RNA vs CN, logistic regression, clustering).
   - Compute any missing metrics (ARI, subtype score spreads, normalization diagnostics).
   - Populate the report template and save to `reports/norm_comparison/rsem_uq_tss/report.md`.
   - Save plots to `reports/norm_comparison/rsem_uq_tss/`.

4. **Validate:** Confirm that metrics extracted match what Notebook 02a produces (e.g., AUC-ROC values, silhouette scores). Any discrepancy indicates a bug.

### Key Parameters to Preserve
- Top 3000 genes by MAD for clustering
- StandardScaler before PCA
- 10 PCs for clustering
- KMeans with `n_init=10, random_state=42`
- 5-fold stratified CV for logistic regression (`random_state=42`)
- Same marker gene lists for subtype scoring (luminal: ESR1/PGR/GATA3/FOXA1; HER2: ERBB2/GRB7/STARD3; basal: KRT5/KRT14/KRT17/EGFR)

---

## Phase 2: Implement Alternative Normalizations

### 2A: TPM Normalization

**What TPM does:** Transcripts Per Million normalizes each gene by its length, then scales all genes in a sample to sum to 1 million. This corrects for both gene length bias and sequencing depth.

**Why it might weaken HER2 signal:** TPM is compositional -- if ERBB2 is highly amplified, its high counts "eat into" the budget for other genes, compressing the apparent fold-change.

#### Steps

1. **Obtain raw counts and gene lengths.**
   - RSEM provides estimated counts (`*.genes.results` files contain `expected_count` and `effective_length`).
   - If raw RSEM output files are available: extract `expected_count` and `effective_length` columns.
   - If only the normalized matrix (`01_tumor_norm`) is available: check whether RSEM's `FPKM` or `TPM` columns were already computed. RSEM outputs TPM natively -- it may already exist in the data.
   - **Fallback:** If gene lengths are unavailable, use median transcript lengths from GENCODE (match by gene symbol). Document this approximation.

2. **Compute TPM matrix.**
   ```
   RPK_ig = count_ig / length_g (in kb)
   TPM_ig = RPK_ig / sum_g(RPK_ig) * 1e6
   ```
   - Apply log2(TPM + 1) transform for downstream analyses (matching the log-scale convention of RSEM).
   - Apply the same TSS-targeted normalization as the RSEM pipeline (combat or median-centering per TSS).

3. **Verify:** Check that per-sample sums are ~1e6 before log transform. Spot-check a few known housekeeping genes (ACTB, GAPDH) for reasonable values.

### 2B: TMM Normalization (edgeR)

**What TMM does:** Trimmed Mean of M-values computes a scaling factor per sample based on a reference sample, trimming extreme fold-changes. It is designed to handle compositional bias from highly expressed genes.

**Why it might weaken HER2 signal:** TMM explicitly adjusts for the scenario where a few genes dominate the library. ERBB2 amplification is exactly this scenario -- TMM may "correct away" the amplification signal.

#### Steps

1. **Obtain raw counts.**
   - Same raw count source as TPM (RSEM `expected_count`).
   - TMM requires integer-like counts. If RSEM expected counts are fractional, round to nearest integer.

2. **Compute TMM normalization factors via edgeR (in R, called from Python).**
   ```r
   library(edgeR)
   dge <- DGEList(counts = count_matrix)
   dge <- calcNormFactors(dge, method = "TMM")
   # Extract normalized log-CPM
   lcpm <- cpm(dge, log = TRUE, prior.count = 1)
   ```
   - Use `rpy2` to call R from Python, OR write a standalone R script that reads counts CSV and outputs normalized matrix CSV.
   - **Fallback if R/edgeR unavailable:** Implement TMM in pure Python following Robinson & Oshlack (2010). The algorithm is: (a) pick reference sample (whose upper quartile is closest to mean UQ), (b) for each sample compute M-values (log-ratios) and A-values (log-intensities) vs reference, (c) trim top/bottom 30% of M and 5% of A, (d) compute weighted mean of remaining M-values as log2 scaling factor.

3. **Apply the same TSS-targeted normalization** as the RSEM pipeline.

4. **Verify:** Check that TMM normalization factors are in a reasonable range (typically 0.9-1.1). Outlier factors suggest composition bias -- document any samples with extreme factors.

---

## Phase 3: Run Downstream Analyses on Each Alternative Normalization

For EACH alternative normalization (TPM, TMM), execute the identical analysis pipeline:

### 3.1 Data Preparation
- Replace `tumor_norm` with the alternatively normalized expression matrix.
- Keep `clinical` and `cn` data unchanged (these are normalization-independent).
- Merge into `cohort_c` using the same join logic as Notebook 02a (Cell 5).

### 3.2 ERBB2 RNA vs Copy Number (Report Section A)
- Extract ERBB2 expression from the new matrix.
- Compute Pearson/Spearman correlations (all, HER2+, HER2-).
- Compute Cohen's d, Mann-Whitney U, fold-change.
- Generate scatter plot P1.

### 3.3 Logistic Regression (Report Section B)
- Exact same pipeline: 5-fold stratified CV, LogisticRegression, 3 models (RNA, CN, combined).
- Record AUC-ROC, AUC-PR for each.
- Generate ROC plot P2.

### 3.4 Unsupervised Clustering (Report Section C)
- Top 3000 genes by MAD (recomputed on new matrix -- MAD ranking may change).
- StandardScaler -> PCA (20 components) -> KMeans on top 10 PCs.
- Silhouette scores for k=2..7.
- ARI vs HER2 composite and ER status at k=4.
- Generate silhouette plot P3, UMAP/PCA plot P4.

### 3.5 Subtype Marker Analysis (Report Section D)
- Subtype scoring using same gene lists at k=4.
- Marker gene heatmap P5.
- Compute score spreads and HER2-enriched cluster purity.

### 3.6 Normalization Diagnostics (Report Section E)
- CV of median expression across TSS sites.
- PC1 vs TSS (Kruskal-Wallis), PC1 vs read-depth proxy (correlation).
- ERBB2 CV within HER2+ group.

### 3.7 Output
- Populate report template, save to `reports/norm_comparison/<method>/report.md`.
- Save all plots to same directory.

---

## Phase 4: Organize Reports for Comparison

### Deliverables

```
reports/norm_comparison/
  rsem_uq_tss/
    report.md
    scatter_rna_vs_cn.png
    roc_curves.png
    silhouette_profile.png
    umap_clusters.png
    marker_heatmap.png
  tpm/
    report.md
    <same plots>
  tmm_edger/
    report.md
    <same plots>
  comparison_summary.md    <-- side-by-side table of all metrics
```

### Comparison Summary Table

The `comparison_summary.md` file should contain:

1. **Side-by-side metrics table** -- all numeric fields from the report, one column per method.
2. **Signal-vs-noise scorecard:**
   - "Signal" composite = mean of normalized ranks for A5, A7, B1, B7, C3, D4, D5 (higher = stronger HER2 biology).
   - "Noise" composite = mean of normalized ranks for E1, E2, E3 (lower = cleaner technical control).
   - Highlight the method with best signal-to-noise balance.
3. **Interpretive notes** for the Analyst:
   - If TPM/TMM metrics in Section A-D drop substantially vs RSEM, this confirms that read-depth normalization compresses real biological signal.
   - If Section E metrics improve substantially with TPM/TMM but Section A-D metrics hold, the current RSEM approach may be under-normalizing.
   - If all methods produce similar results, normalization choice is not a critical decision point for this dataset.

---

## Implementation Notes

### Code Organization
- All new code should go in `scripts/normalization_comparison/` with one module per normalization method.
- Shared analysis functions (logistic regression, clustering, subtype scoring, report generation) should be factored into a `scripts/normalization_comparison/analysis_pipeline.py` to guarantee identical methodology across methods.
- The pipeline function signature should be: `run_analysis(expr_matrix, clinical, cn, gene_cols, method_name, output_dir) -> report_dict`.

### Dependencies
- Existing: pandas, numpy, scipy, sklearn, matplotlib, seaborn
- New for TMM: either `rpy2` + R's `edgeR`, or a pure-Python TMM implementation
- Optional: `umap-learn` for UMAP visualization (fall back to PCA if unavailable)

### Reproducibility
- All random seeds fixed at 42.
- Gene filtering (MAD top 3000) is recomputed per normalization -- this is intentional, as normalization changes variance structure.
- Document any data-availability issues (e.g., missing gene lengths for TPM, missing raw counts).

### Risk: Raw Counts Availability
The largest risk to this plan is whether raw RSEM counts (pre-upper-quartile normalization) are available. If `01_tumor_norm` is already UQ-normalized and raw counts are not stored:
- **Check** whether Notebook 01 saves raw counts as a separate intermediate.
- **Check** whether RSEM `.genes.results` files are in the data directory.
- **If neither exists:** We cannot compute TPM or TMM from scratch. In this case, pivot to applying TPM/TMM on top of the existing UQ-normalized values (document this as a limitation) or attempt to reverse the UQ normalization if the scaling factors were saved.

---

## Sequence of Execution

| Step | Phase | Description | Depends On |
|------|-------|-------------|------------|
| 1 | 0 | Finalize report template (this document) | -- |
| 2 | 1.1 | Locate data, confirm raw counts availability | -- |
| 3 | 1.2 | Audit Notebook 02a, catalog existing metrics | Step 2 |
| 4 | 1.3 | Write + run RSEM extraction script | Steps 2, 3 |
| 5 | 1.4 | Validate RSEM report against notebook | Step 4 |
| 6 | 2A | Compute TPM matrix | Step 2 |
| 7 | 2B | Compute TMM matrix | Step 2 |
| 8 | 3 | Run analysis pipeline on TPM | Steps 4, 6 |
| 9 | 3 | Run analysis pipeline on TMM | Steps 4, 7 |
| 10 | 4 | Generate comparison summary | Steps 5, 8, 9 |

Steps 6 and 7 can run in parallel. Steps 8 and 9 can run in parallel.
