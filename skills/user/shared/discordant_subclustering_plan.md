# Tactical Plan: Discordant Subclustering & RNA-vs-DNA Narrative Bridge

**Status:** Ready for implementation
**Estimated effort:** 30-40 minutes total
**Placement:** NB03 Section 5 (new Section 5.4) + NB03 Section 2 intro edit

---

## Task 1: Discordant Group Subclustering (NB03, new Section 5.4)

### Goal

Perform unsupervised clustering on the 35 IHC-negative/RNA-high discordant patients
using the full filtered transcriptome. Validate (or challenge) the CN-stratified
two-population finding through an orthogonal analytical approach.

### Inputs

All inputs are already loaded in NB03:

- `tumor_norm` (TMM+TSS normalized expression, from `01_tumor_norm_tmm_tss`)
- `gene_cols` (filtered gene list, from `01_gene_cols.json`)
- `disc_rna_high` (discordant patient subset, defined in NB03 Cell 16 area)
- `disc_amplified` / `disc_non_amplified` (CN-stratified subsets, defined in Section 5)
- `cohort_c` (multimodal cohort for reference)

### Step-by-step

#### Step 1: New markdown cell (insert after current Section 5.3, before Section 6)

Title: `### 5.4 Unsupervised Validation: Does Transcriptomic Structure Confirm CN Stratification?`

Framing text (3-4 sentences):
- The CN-stratified analysis above was driven by a single variable (copy number).
- We now ask whether unsupervised structure in the full transcriptome independently
  recovers this split.
- If PCA + clustering on expression alone separates CN-high from CN-low discordant
  patients, the two-population finding is transcriptome-wide, not a single-variable
  artifact.

#### Step 2: Code cell -- PCA on discordant patients

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Extract expression for discordant patients only
disc_pids = disc_rna_high['pid'].values
disc_expr = tumor_norm[tumor_norm['pid'].isin(disc_pids)][['pid'] + gene_cols].copy()
disc_expr = disc_expr.set_index('pid')

# Standardize (gene-wise z-score)
scaler = StandardScaler()
disc_scaled = pd.DataFrame(
    scaler.fit_transform(disc_expr),
    index=disc_expr.index,
    columns=disc_expr.columns
)

# PCA
pca = PCA(n_components=min(10, len(disc_pids) - 1))
pcs = pca.fit_transform(disc_scaled)
pc_df = pd.DataFrame(
    pcs,
    index=disc_expr.index,
    columns=[f'PC{i+1}' for i in range(pcs.shape[1])]
)

print(f"Discordant patients in expression matrix: {len(disc_expr)}")
print(f"Genes used: {len(gene_cols)}")
print(f"Variance explained (PC1-5): {pca.explained_variance_ratio_[:5].round(3)}")
print(f"Cumulative (PC1-5): {pca.explained_variance_ratio_[:5].sum():.3f}")
```

**Check:** Print how many of the 35 discordant patients have expression data.
Some may be missing from the multimodal overlap. Report the actual n.

#### Step 3: Code cell -- k=2 clustering + CN overlay

```python
# k=2 clustering on PCA space (first 5 PCs)
km = KMeans(n_clusters=2, random_state=42, n_init=10)
pc_df['cluster'] = km.fit_predict(pc_df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']])

# Add CN status
cn_map = disc_rna_high.set_index('pid')['erbb2_copy_number']
pc_df['cn'] = pc_df.index.map(cn_map)
pc_df['cn_group'] = pc_df['cn'].apply(lambda x: 'CN-high (2)' if x == 2 else 'CN-low (<=1)')

# Cross-tabulate cluster vs CN
ct = pd.crosstab(pc_df['cluster'], pc_df['cn_group'])
print("Cluster x CN cross-tabulation:")
print(ct)
print()

# Fisher's exact test (or chi2 if counts allow)
from scipy.stats import fisher_exact
if ct.shape == (2, 2):
    odds, p = fisher_exact(ct.values)
    print(f"Fisher's exact: OR={odds:.2f}, p={p:.4f}")
```

**Decision point:** If Fisher's p < 0.05 and the clusters align with CN, the
unsupervised structure confirms CN stratification. If p > 0.05 or the alignment
is weak, report that honestly -- it means the two-population story is CN-driven,
not transcriptome-wide, which is also a valid finding.

#### Step 4: Code cell -- Visualization (2-panel figure)

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: PCA colored by unsupervised cluster
ax = axes[0]
for cl in [0, 1]:
    mask = pc_df['cluster'] == cl
    ax.scatter(pc_df.loc[mask, 'PC1'], pc_df.loc[mask, 'PC2'],
               label=f'Cluster {cl} (n={mask.sum()})', alpha=0.7, s=60)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
ax.set_title("Discordant Patients: Unsupervised Clusters")
ax.legend()

# Panel B: Same PCA colored by CN status
ax = axes[1]
cn_colors = {'CN-high (2)': '#e74c3c', 'CN-low (<=1)': '#3498db'}
for grp, color in cn_colors.items():
    mask = pc_df['cn_group'] == grp
    ax.scatter(pc_df.loc[mask, 'PC1'], pc_df.loc[mask, 'PC2'],
               c=color, label=f'{grp} (n={mask.sum()})', alpha=0.7, s=60)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
ax.set_title("Discordant Patients: CN Status Overlay")
ax.legend()

plt.tight_layout()
savefig(fig, 'fig_03_discordant_subclustering')
plt.show()
```

#### Step 5: Code cell -- Cluster characterization table

```python
# Characterize the two clusters
char_cols = ['expr_ERBB2', 'erbb2_copy_number', 'conc_model_prob']
# Add 17q12 amplicon genes if available
amplicon_genes = ['ERBB2', 'GRB7', 'STARD3', 'PGAP3', 'TCAP']
for g in amplicon_genes:
    col = f'expr_{g}' if f'expr_{g}' in disc_rna_high.columns else g
    if col in disc_rna_high.columns:
        char_cols.append(col)

# Merge characterization data
char_df = pc_df[['cluster', 'cn_group']].copy()
disc_indexed = disc_rna_high.set_index('pid')
for col in char_cols:
    if col in disc_indexed.columns:
        char_df[col] = char_df.index.map(disc_indexed[col])

# Summary by cluster
print("Cluster Characterization:")
print(char_df.groupby('cluster')[char_cols].agg(['mean', 'median', 'count']).round(3))
```

**Note:** Adapt column names to what's actually available in `disc_rna_high`.
The exact column names (e.g., `expr_ERBB2` vs bare gene name) depend on how
the multimodal DataFrame was constructed in NB03 Cells 3-6.

#### Step 6: Interpretive markdown cell

Write 3-5 sentences summarizing:
- Whether the clusters align with CN status (report the cross-tab and p-value)
- What this means for the two-population interpretation
- If alignment is strong: "Unsupervised transcriptomic structure independently
  confirms that the discordant group contains two biologically distinct
  populations, validating the CN-stratified analysis in Sections 5.1-5.3."
- If alignment is weak: "Transcriptomic structure does not cleanly separate
  CN-high from CN-low patients, suggesting the CN split captures one axis of
  variation but does not fully characterize discordant biology."

#### Step 7: Update Section 9 (Conclusions)

Add one bullet under the existing interpretation acknowledging the unsupervised
validation result. One sentence is sufficient.

### Outputs

- 1 figure: `fig_03_discordant_subclustering.png` (2-panel PCA)
- Cross-tabulation printed in notebook
- Characterization table printed in notebook
- No new intermediate files needed (this is a visualization/validation step)

---

## Task 2: RNA-vs-DNA Narrative Bridge (NB03 Section 2 intro)

### Goal

Add explicit framing connecting NB02's single-gene ERBB2 ROC to NB03's panel-based
ML as a modality comparison answering the prompt's "determine if RNA or DNA is more
predictive" question.

### Location

NB03 Cell 4 (the markdown cell introducing Section 2). Edit the existing text.

### Current text (paraphrased)

Section 2 intro talks about establishing a baseline with ML using the curated gene
panel. It mentions the 6 gene sets and feature construction but does not reference
NB02's ROC result or frame Section 2 as extending the RNA-vs-DNA comparison.

### Proposed addition

Insert after the first paragraph of Section 2 (Cell 4), before the feature
construction details:

> **Modality comparison context:** NB02 established that ERBB2 RNA expression alone
> outperforms ERBB2 copy number for predicting clinical HER2 status (AUC 0.84 vs
> 0.81, single-gene ROC). Here we extend that comparison to multi-gene panels and
> more complex models. If the curated RNA panel substantially outperforms CN-augmented
> models, it confirms that RNA's advantage is not limited to ERBB2 itself but extends
> to the broader transcriptomic context captured by biologically motivated gene sets.

### Also add

In Section 2.1.1 (Cell 12, the interpretation of the 3x3 comparison), add one
sentence connecting back:

> This confirms and extends the NB02 finding: RNA's advantage over CN holds at both
> the single-gene level (NB02, AUC 0.84 vs 0.81) and the multi-gene panel level,
> establishing RNA as the more informative modality for HER2 classification across
> analytical scales.

### Effort

Two markdown cell edits. No code changes.

---

## Execution Order

1. **Task 2 first** (5 min) -- it's a text edit, gets it out of the way
2. **Task 1** (25-30 min) -- insert new cells, run, interpret
3. **Review** -- re-read NB03 Sections 2 and 5 end-to-end to confirm narrative flow

## Risk Mitigation

- **n=35 is small for PCA.** With ~17K genes and 35 samples, PCA will produce at
  most 34 components. The first few PCs may be dominated by a single outlier. Check
  PC1 loadings -- if one patient drives >50% of PC1 variance, consider robust PCA
  or removing the outlier and noting it.
- **k=2 is prescribed, not discovered.** This is intentional -- we're testing a
  specific hypothesis (two populations). But mention in the markdown that k=2 was
  chosen a priori based on the CN stratification hypothesis, not from silhouette
  optimization.
- **Column name mismatches.** The code sketches above use assumed column names.
  The coder should verify actual column names in `disc_rna_high` and `tumor_norm`
  before implementing. Key variables to check: how ERBB2 expression is stored
  (as a gene column name or as `expr_ERBB2`), how CN is stored, how concordant
  model probabilities are attached.
