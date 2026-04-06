# Normalization Comparison: Interpretive Analysis

**Date:** 2026-04-05
**Analyst scope:** Interpret the comparison_summary.md results across three normalization
strategies (RSEM-UQ-TSS, TPM-TSS, TMM-edgeR) with respect to HER2 biological signal
preservation, technical artifact removal, and implications for downstream discordant
HER2 classification.

---

## 1. What Each Normalization Method Does (and Assumes)

### RSEM Upper Quartile (RSEM-UQ-TSS)

The established pipeline. Takes RSEM expected counts (fractional counts from the EM
algorithm), divides each sample by its 75th-percentile gene expression value, applies
log2(x+1), then regresses out TSS batch effects while protecting HER2 and ER
covariates.

**Key assumption:** The 75th-percentile gene is stably expressed across samples. This
is the standard "most genes are not differentially expressed" assumption, but applied
at a specific quantile rather than across the full distribution.

**Failure mode for HER2:** In HER2-amplified samples, the ERBB2 amplicon on 17q12
includes ~10-15 co-amplified genes (GRB7, STARD3, PGAP3, etc.) that are massively
over-expressed. These high-count genes push up the upper tail of the per-sample
expression distribution. Even though the 75th percentile gene itself may not be in the
amplicon, the overall ranking is shifted upward because amplicon genes displace other
genes. The result: HER2+ samples get a larger UQ divisor, which systematically
suppresses ALL gene expression values in these samples -- including ERBB2 itself. This
is a form of compositional over-correction specific to samples with focal
amplifications.

### TPM (Transcripts Per Million)

Divides raw counts by gene length (in kb) to obtain reads per kilobase (RPK), then
scales each sample so all RPK values sum to 1 million. Applies log2(x+1), then TSS
correction. Gene lengths were obtained from ENSEMBL BioMart (median transcript
length per gene, cached locally).

**Key assumption:** Gene lengths are constant across samples. This is true at the DNA
level but effective transcript lengths can vary with isoform usage. TPM is
compositional -- each gene's value depends on ALL other genes in the sample.

**Behavior for HER2:** TPM removes gene-length bias (important for cross-gene
comparisons) but introduces a sum-to-one-million constraint. When ERBB2 is highly
amplified, its large RPK consumes a disproportionate share of the 1M budget. However,
unlike UQ, the normalization denominator (total RPK sum) is not a single quantile that
can be disproportionately shifted by the amplicon. The net effect is milder compression
than UQ.

### TMM (Trimmed Mean of M-values)

Implements the Robinson & Oshlack (2010) algorithm. Selects a reference sample (whose
upper quartile is closest to the mean UQ across all samples). For each sample, computes
gene-level log fold-changes (M-values) and average log-intensities (A-values) versus
the reference. Trims the 30% most extreme M-values and the 5% most extreme A-values.
Computes a precision-weighted mean of the remaining M-values as the sample's log2
scaling factor. Factors are normalized so their geometric mean equals 1. Applied as
log2(CPM + 1) using effective library sizes (raw library size * TMM factor).

**Key assumption:** The majority of genes are NOT differentially expressed between any
pair of samples. By trimming extreme fold-changes, TMM explicitly ignores outlier genes
when computing scaling factors.

**Why this helps HER2:** ERBB2 and its co-amplified neighbors produce extreme M-values
in HER2+ samples. TMM trims these away. The scaling factor is computed from the
remaining ~70% of genes -- the "housekeeping majority" that truly should be comparable
across samples. This means the amplification signal is preserved in the final
normalized values rather than being divided out by an inflated normalization constant.

**TMM factor range:** The observed range of 0.57-1.33 is wider than the typical
0.9-1.1 seen in most RNA-seq datasets. This is expected in breast cancer data with
ERBB2 amplification driving library composition in a subset of samples. The wide range
means TMM is doing substantial work to correct for composition bias -- and doing so
without absorbing the ERBB2 signal.

---

## 2. The Central Finding: UQ Over-Corrects HER2+ Samples

The normalization comparison plan hypothesized that TPM and TMM would *compress* HER2
signal relative to RSEM-UQ, because more aggressive read-depth normalization should
reduce biological fold-changes. **The opposite occurred.** Both TPM and TMM
*strengthen* every key HER2 discrimination metric:

| Metric                  | RSEM-UQ | TPM    | TMM    | Direction |
|-------------------------|---------|--------|--------|-----------|
| Cohen's d (A5)          | 2.086   | 2.122  | 2.186  | TMM > TPM > RSEM |
| AUC-ROC RNA (B1)        | 0.837   | 0.856  | 0.863  | TMM > TPM > RSEM |
| Delta AUC RNA-CN (B7)   | +0.029  | +0.048 | +0.055 | TMM > TPM > RSEM |
| Fold-change median (A7) | 1.212   | 1.369  | 1.321  | TPM > TMM > RSEM |
| HER2 cluster purity (D5)| 0.579   | 0.550  | 0.591  | TMM > RSEM > TPM |

**Interpretation:** RSEM-UQ normalization inflates the 75th percentile in HER2+
samples because the amplicon genes shift the upper distribution. Dividing by this
inflated value systematically compresses expression in HER2+ samples, reducing the
apparent gap between HER2+ and HER2-. TMM avoids this by trimming the amplicon
genes when computing scaling factors. TPM avoids it by not using a single quantile as
its normalization reference.

This is not a subtle effect. The AUC-ROC improvement from RSEM to TMM (+0.026) is
clinically meaningful for a single-feature logistic regression on the same data. The
Cohen's d increase (+0.10) corresponds to approximately 5% of the pooled standard
deviation -- a non-trivial recovery of signal that UQ had suppressed.

---

## 3. Where the Methods Agree (Concordant Findings)

### 3a. Dominant transcriptomic structure is ER-driven, not HER2-driven

All three methods find best k=2 by silhouette (0.275-0.301), and at k=4 the
clustering is far more aligned with ER status (ARI 0.191-0.202) than HER2 status
(ARI 0.030-0.042). This is biologically expected: ER status drives a larger
fraction of transcriptomic variance in breast cancer than HER2 status. The ER+/ER-
split is the primary axis; HER2 enrichment is a secondary feature that cross-cuts
the ER axis (many HER2+ tumors are also ER+).

This finding is robust to normalization choice, which is reassuring -- it means the
dominant biological structure is not an artifact of any particular normalization.

### 3b. RNA outperforms CN for HER2 prediction

All methods show positive B7 (delta AUC-ROC: RNA minus CN), ranging from +0.029
(RSEM) to +0.055 (TMM). RNA captures information beyond copy number: transcriptional
regulation, epigenetic effects, and post-amplification expression modulation. This is
an important biological finding independent of normalization.

### 3c. Combined RNA+CN model does not improve over RNA alone

B5 is approximately equal to (or slightly less than) B1 across all methods. This means
CN adds no information beyond what RNA already provides for predicting IHC-defined
HER2 status. The CN signal is a subset of the RNA signal -- whatever genomic
amplification does, its downstream effect is fully captured by the expression level.

### 3d. TSS batch effects persist regardless of normalization

E2 (Kruskal-Wallis p for PC1 vs TSS) is significant (p < 0.05) for all three methods.
The regression-based TSS correction reduces but does not eliminate batch effects in the
leading principal component. This is expected: TSS effects are confounded with real
biology (TSS is associated with HER2 status, chi2=177.9), so the protected-covariate
regression correctly refuses to remove the portion of TSS variation that overlaps with
HER2/ER signal.

---

## 4. Where the Methods Disagree (Key Divergences)

### 4a. Read-depth confounding (E3) -- the sharpest divergence

| Method  | PC1 vs read-depth (r) |
|---------|-----------------------|
| RSEM-UQ | 0.310                 |
| TPM     | 0.234                 |
| TMM     | -0.015                |

TMM virtually eliminates read-depth confounding in the primary axis of variation.
RSEM-UQ leaves a moderate confound (r=0.31), meaning ~10% of PC1 variance correlates
with sequencing depth. TPM partially corrects this (r=0.23). TMM essentially zeroes
it out (r=-0.01, indistinguishable from no correlation).

**Why this matters:** Any downstream analysis that uses PCA-derived features
(clustering, dimensionality reduction, visualization) will carry read-depth bias under
RSEM-UQ. For a clinical application like HER2 classification, you do not want a
patient's predicted status to depend on how deeply their sample was sequenced.

### 4b. ERBB2 CV within HER2+ patients (E4)

| Method  | ERBB2 CV (HER2+) |
|---------|-------------------|
| RSEM-UQ | 0.133             |
| TMM     | 0.168             |
| TPM     | 0.190             |

RSEM-UQ produces the *tightest* ERBB2 distribution within HER2+ patients. At first
glance, low CV might seem desirable (less noise). But given that UQ systematically
compresses HER2+ expression (Section 2), this low CV is likely artifactual
over-compression rather than genuine biological homogeneity. TMM and TPM preserve more
within-group heterogeneity -- some of which is real biological variation (different
amplification levels, different transcriptional regulation, different co-amplification
partners). For identifying discordant cases within the HER2+ group, you want this
heterogeneity preserved.

### 4c. TSS-site variation (E1)

| Method  | CV of median expr across TSS |
|---------|------------------------------|
| TMM     | 0.010                        |
| RSEM-UQ | 0.014                        |
| TPM     | 0.049                        |

TPM actually *worsens* cross-TSS variability (3.5x higher CV than TMM). This is a
notable disadvantage. Gene-length normalization may introduce a new TSS-associated
artifact if different sites used library prep protocols that affect effective transcript
lengths (e.g., different fragmentation methods, different poly-A selection stringency).
TMM and RSEM-UQ both produce low cross-TSS CV, meaning site-to-site systematic
differences in median expression are well controlled.

### 4d. RNA-CN correlation structure (A1-A4)

| Metric               | RSEM-UQ | TPM    | TMM    |
|-----------------------|---------|--------|--------|
| Pearson r (all)       | 0.734   | 0.722  | 0.728  |
| Pearson r (HER2+)     | 0.827   | 0.814  | 0.805  |
| Pearson r (HER2-)     | 0.491   | 0.472  | 0.488  |

RSEM-UQ preserves the strongest RNA-CN correlation. TMM and TPM both reduce it
slightly. This is not contradictory to the finding that TMM/TPM improve discrimination
metrics (Cohen's d, AUC). The correlation measures *linearity* of the RNA-CN
relationship, while the discrimination metrics measure *separation* between groups.
UQ compression can maintain a tight linear relationship (high r) while reducing the
absolute difference between groups (lower d). Conceptually: UQ squeezes the HER2+ and
HER2- distributions closer together (lower d) but preserves their internal rank order
(high r).

### 4e. Clustering quality vs cluster-biology alignment

| Metric                    | RSEM-UQ | TPM    | TMM    |
|---------------------------|---------|--------|--------|
| Silhouette k=4 (C3)       | 0.197   | 0.180  | 0.191  |
| ARI k=4 vs HER2 (C4)      | 0.030   | 0.035  | 0.042  |
| ARI k=4 vs ER (C5)        | 0.191   | 0.198  | 0.202  |
| HER2 cluster purity (D5)  | 0.579   | 0.550  | 0.591  |

RSEM-UQ produces slightly tighter clusters (higher silhouette), but TMM's clusters
better correspond to known biology (higher ARI with both HER2 and ER labels, higher
HER2 cluster purity). The tighter clusters under RSEM-UQ may partly reflect the
compression of HER2+ expression values, which artificially reduces within-cluster
variance.

TMM achieves the best HER2-enriched cluster purity at 59.1% -- meaning 59% of
clinically HER2+ patients land in the cluster that the algorithm labels as
HER2-enriched. This is a modest but consistent advantage over RSEM (57.9%) and
especially TPM (55.0%).

---

## 5. Signal vs Noise Scorecard

| Composite        | RSEM-UQ | TPM   | TMM   |
|------------------|---------|-------|-------|
| Signal           | 0.357   | 0.357 | 0.786 |
| Noise            | 0.167   | 0.333 | 1.000 |
| Mean (S+N)/2     | 0.262   | 0.345 | 0.893 |

TMM dominates both dimensions. This is not a marginal result -- it is a clear
separation on both signal preservation and technical artifact removal. RSEM-UQ and
TPM are comparatively close to each other, while TMM is in a different tier.

**Why does TMM win signal?** Because UQ over-corrects HER2+ samples (Section 2). TMM
recovers signal that UQ had suppressed.

**Why does TMM win noise?** Primarily because of E3 (read-depth confound
near zero). TMM was designed to handle exactly this kind of composition-driven
library size variation.

---

## 6. Implications for Discordant HER2 Classification

The downstream goal is to identify "discordant" HER2 cases: patients who are IHC
negative but have high ERBB2 RNA expression (or vice versa). This is clinically
actionable because these patients may benefit from HER2-targeted therapies (e.g.,
T-DXd) despite negative IHC.

### 6a. Why normalization choice matters for discordant identification

Discordant patients live at the boundary between the HER2+ and HER2- distributions.
Any normalization that compresses this boundary makes discordant cases harder to
distinguish from true negatives. Conversely, a normalization that preserves the full
dynamic range of ERBB2 expression gives the classifier more room to identify patients
whose RNA expression is "surprisingly high" for their IHC status.

### 6b. TMM is the strongest choice for discordant prediction

Several lines of evidence converge:

1. **Largest effect size (A5=2.19):** A wider gap between HER2+ and HER2- means more
   room to identify patients in the borderline zone. The increase from RSEM's 2.09 to
   TMM's 2.19 corresponds to ~5% of the pooled standard deviation.

2. **Highest AUC-ROC for RNA prediction (B1=0.863):** A model trained on TMM-normalized
   ERBB2 expression is 2.6 percentage points better at predicting IHC-defined HER2
   status. The discordant cases are precisely the patients the model gets wrong. A
   better-calibrated model has higher confidence in its predictions, so the
   "surprising" predictions (discordant cases) carry more statistical weight.

3. **Largest delta AUC RNA vs CN (B7=+0.055):** TMM maximizes the informational
   advantage of RNA over CN. For discordant cases where CN and IHC disagree, RNA
   provides the most additional information under TMM normalization.

4. **Near-zero read-depth confound (E3=-0.015):** Under TMM, the primary axis of
   transcriptomic variation is almost entirely free of sequencing-depth artifacts.
   This means a patient classified as "IHC-negative but RNA-high" is unlikely to be a
   sequencing-depth artifact. Under RSEM-UQ (E3=0.31), you cannot rule out that some
   "discordant" calls are driven by high sequencing depth rather than genuine biology.

5. **Preserved within-group heterogeneity (E4=0.168):** TMM preserves more biological
   variation within the HER2+ group than RSEM-UQ (0.133). This heterogeneity is real:
   some HER2+ patients have much higher ERBB2 expression than others, reflecting
   different amplification levels and transcriptional programs. A normalization that
   preserves this gradient also preserves the low end of HER2+ expression -- exactly
   where discordant cases reside.

6. **Best HER2 cluster purity (D5=0.591):** In unsupervised clustering, TMM recovers
   the most HER2+ patients in the HER2-enriched cluster. This suggests TMM-normalized
   expression better reflects the underlying HER2 biology, giving supervised
   classifiers stronger features to work with.

### 6c. Practical recommendation

For the discordant-HER2 ML pipeline (Notebooks 03/04), TMM-normalized expression
should be used instead of RSEM-UQ-normalized expression. The expected impact:

- **Higher sensitivity** for identifying IHC-negative/RNA-high discordant patients,
  because the wider ERBB2 dynamic range separates the HER2+ distribution further from
  the HER2- distribution, making intermediate cases more identifiable.

- **Lower false-discovery rate** for discordant calls driven by sequencing-depth
  artifacts, because TMM eliminates the read-depth confound in PC1 that RSEM-UQ
  carries.

- **Better-calibrated probability estimates** from logistic regression or more complex
  models, because the underlying ERBB2 feature has stronger signal-to-noise ratio.

### 6d. Caveats

1. **CN-stratified discordant biology still matters.** Normalization affects the RNA
   side of the analysis, not the CN or IHC data. The known distinction between
   amplicon-driven (CN>=2) and transcription-driven (CN<=1) discordant cases remains
   independent of normalization choice. Both should be analyzed separately.

2. **The TMM implementation is pure Python, not edgeR.** The algorithm faithfully
   follows Robinson & Oshlack (2010), but minor numerical differences from the
   Bioconductor edgeR package are possible. For production use, cross-validation against
   edgeR output on a subset of samples would be prudent.

3. **The wide TMM factor range (0.57-1.33) deserves attention.** While expected for
   data with focal amplifications, factors this extreme mean TMM is making large
   per-sample adjustments. Any errors in the factor estimation (e.g., from a poorly
   chosen reference sample) would propagate into the normalized expression values.
   The reference sample selection (UQ closest to mean UQ) is the standard approach
   and should be robust for this dataset.

4. **TPM's increased TSS variability (E1=0.049) is a concern.** If TPM were selected
   for other reasons, the elevated cross-TSS CV would need investigation. The likely
   explanation is library-prep-dependent effective transcript lengths varying by site,
   which introduces a gene-length-mediated batch effect that the TSS regression does
   not fully correct.

---

## 7. Concordance and Disagreement: Section-by-Section Summary

| Section | Finding | Agreement level |
|---------|---------|-----------------|
| A (RNA vs CN) | TMM and TPM strengthen effect size; RSEM preserves tighter linear correlation | Partial -- both are "right" but measure different things (linearity vs separation) |
| B (Logistic regression) | TMM best AUC, RSEM worst; RNA > CN, combined = RNA alone | Full agreement on RNA > CN; quantitative disagreement on how much |
| C (Clustering) | k=2 dominant; ER > HER2 alignment at k=4; TMM best ARI | Full structural agreement; TMM clusters align best with known labels |
| D (Subtype markers) | RSEM/TMM similar marker spreads; TPM compresses markers; TMM best HER2 purity | Partial -- TPM's compositional constraint hurts marker separation |
| E (Diagnostics) | TMM cleanest on read-depth and TSS-CV; TPM worst on TSS-CV; RSEM moderate | Divergent -- this is where the methods differ most |

---

## 8. Mechanistic Summary: Why TMM Outperforms UQ for HER2 Biology

The core mechanism in one paragraph:

ERBB2 amplification creates a **compositional bias** in the transcriptome of HER2+
samples. The amplicon genes consume a disproportionate share of sequencing reads.
Upper-quartile normalization absorbs part of this compositional shift into the
normalization factor, artificially compressing ERBB2 expression in amplified samples.
TMM is explicitly designed to handle compositional bias by trimming extreme fold-changes
(which include the amplicon genes) before computing scaling factors. The result: TMM
normalizes for technical variation (library size, sequencing depth) while preserving the
biological signal from focal amplification. This is precisely the scenario Robinson &
Oshlack designed TMM for -- a small set of highly expressed genes creating
composition-dependent library size effects.

---

## 9. Recommendation

**Use TMM-edgeR normalization for all downstream HER2 analyses**, including discordant
patient identification, ML classification, and subtype characterization. TMM provides
the best signal-to-noise balance (scorecard: 0.893 vs 0.345 for TPM and 0.262 for
RSEM-UQ), eliminates read-depth confounding, and preserves biological heterogeneity
within HER2-status groups.

The RSEM-UQ pipeline should be retained as a sensitivity analysis to confirm that key
findings (e.g., number of discordant patients, their CN stratification) are robust to
normalization choice.

---

*Analysis performed as Phase 5 interpretation of the Phase 2-4 normalization comparison.
All three methods were run through an identical analysis pipeline
(scripts/normalization_comparison/analysis_pipeline.py) on the same 966-patient
multimodal cohort.*
