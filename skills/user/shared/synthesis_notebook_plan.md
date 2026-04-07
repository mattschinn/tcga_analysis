# Plan: Biopharma Synthesis Notebook (NB04)

**Goal:** A single, polished Jupyter notebook that tells a coherent two-part biopharma
deliverable story, bridging TCGA-based scientific findings to actionable Tempus
opportunities. This is the "executive deck in notebook form" -- every cell should earn
its place by advancing the argument.

---

## Narrative Architecture

The notebook follows a three-act structure:

1. **Act 1 -- The Equivocal CDx Story (Land Play)**
   Clinical problem -> evidence -> stability proof -> operational framework -> next steps

2. **Act 2 -- The Missed HER2 Story (Expand Play)**
   Population sizing -> biological characterization -> clinical invisibility -> Tempus opportunity

3. **Act 3 -- Strategic Synthesis**
   Gap table -> Tempus value proposition -> study designs -> timeline

Between acts: supporting evidence (negative clinical correlates, T-DXd opportunity,
testing method data gap) woven in where they strengthen the argument.

---

## Section-by-Section Plan

### 0. Executive Summary (Markdown only)
- 3-4 bullet headline findings
- Two value propositions in one sentence each
- "What TCGA proves" vs "what Tempus data would add"
- No code. This is the one-page version a VP would read.

### 1. Setup and Data Loading
- Load intermediates from NB01-03 + biopharma script outputs (04_* parquets)
- Load ssGSEA scores, ML predictions, discordant dossier, equivocal scores
- Minimal -- just imports, loads, and a cohort summary table
- **Figures:** None. Just a printed cohort summary (Cohort A/B/C sizes, key demographics).

### 2. Part I: RNA-Based Companion Diagnostic for IHC 2+ Equivocal Patients

#### 2.1 Clinical Context (Markdown)
- IHC 2+ requires FISH reflex testing: cost ($300-500), turnaround (5-7 days), access barriers
- ~15-20% of breast cancers are IHC 2+ -> large addressable population
- Opportunity: RNA-based test to replace/supplement FISH

#### 2.2 Primary Concordance Result
- **Key figure:** Side-by-side ROC curves (primary + held-out) -- **generated in-notebook**
  - Load primary + held-out prediction parquets, recompute ROC via sklearn
  - `plt.subplots(1,2)` with shared axes, annotated with AUC + key operating points
  - This is the headline figure. In-notebook generation shows analytical transparency.
- Performance table (recreated from report data): primary vs held-out metrics
- Key narrative: Lead with held-out specificity (0.983), not inflated AUC (0.994)
- Interpret the delta: sensitivity gap is training design, not signal ceiling

#### 2.3 Equivocal Reclassification Stability
- **Key figure:** `fig_04_5a_supp_equivocal_comparison.png` -- showing 28/28 agreement
- Also: `fig_04_5a_equivocal_erbb2_comparison.png` -- ERBB2 levels in reclassified patients
- Narrative: 5 patients called HER2+ are stable across model designs; driven by ERBB2 + CN
- This is the strongest single data point for clinical decision-making

#### 2.4 Multi-Modal Concordance Tiers (Operational Framework)
- **Key figure:** `fig_04_5b_concordance_tiers.png`
- Tier breakdown table: Tier 1 (RNA+CN, n=4), Tier 2 (RNA-only, n=1), Tier 3 (concordant neg, n=21), Tier 4 (mixed, n=2)
- Narrative: This is HOW you would operationalize multi-modal classification. Framework, not finding.

#### 2.5 Feature Reduction Validation (Brief)
- One-line result: 44-gene curated panel matches full-transcriptome (RF: 0.807 vs 0.810)
- Clinically translatable panel, not a research-grade 17K-feature model
- **Figure:** Optional -- could show SHAP importance from NB03 (`fig16_shap_importance.png` or `fig17_shap_importance.png`) to illustrate the compact feature set

#### 2.6 CDx Next Steps (Markdown)
- Predicate-device comparison pathway (RNA vs. FISH)
- Prospective concordance study design on Tempus data
- Training set design: include IHC 2+ with proper CV to close sensitivity gap

### 3. Part II: Identifying IHC-Missed HER2+ Patients

#### 3.1 Prevalence Estimation
- **Key figure:** `fig_04_2_erbb2_distribution_thresholds.png` -- ERBB2 distribution with threshold markers
- **Key figure:** `fig_04_2_prevalence_cn_stratified.png` -- CN-stratified breakdown (the money figure)
- Result: 5.1% (CI: 3.7-7.0%) at p95 threshold
- Extrapolation: ~9,000-17,000 US patients annually
- **Framing discipline:** NOT a false-negative rate. "Molecular evidence that questions negative classification."

#### 3.2 Two Populations, Two Clinical Strategies
- **Key figure:** `fig_04_3b_pathway_heatmap.png` -- pathway z-scores by CN stratum (central figure for this section)
- **Key figure:** `fig23_synthesis_discordant.png` or `fig24_heatmap_discordant_groups.png` -- the multi-panel synthesis
- Table: CN-high vs CN-low feature comparison (from synthesis report)
- CN-high (n=6): HER2-enriched biology -> HER2-directed therapy candidate
- CN-low (n=29): Luminal co-regulation with ERBB2 decoupling -> investigation needed
- Correlation reversal (ERBB2-ESR1): hypothesis, not finding (p=0.056, n=29)

#### 3.3 Clinical Invisibility (Negative Result as Evidence)
- Summary of Analysis 4a: no clinical differentiators after FDR correction
- **No figure needed** -- a printed statement or small table is sufficient
- Value: you CANNOT screen for these patients with demographics/stage/histology -> molecular testing required

#### 3.4 Supporting Biology from NB03
- **Key figure:** `fig22_de_volcano_key_genes.png` -- DE results in CN-low subgroup
- **Key figure:** `fig25_pathway_comparison_discordant.png` -- pathway comparison
- Brief: DE and GSEA confirm CN-stratified biology aligns with known HER2-enriched vs luminal signatures

### 4. Supporting Evidence and Emerging Opportunities

#### 4.1 Testing Method as Confounder (Data Gap)
- No figure -- print the sparse annotation summary (4/35 patients with data)
- Narrative: This is a demonstrated data gap, not a failed analysis. TCGA lacks testing
  method metadata (antibody clone, fixation protocol, assay platform). If specific testing
  configurations are associated with higher discordance rates, that has immediate clinical
  quality implications.
- Frame as Tempus differentiator: standardized testing metadata enables the analysis TCGA cannot.

#### 4.2 Survival Analysis (Supplementary, Appropriately Caveated)
- **Key figure:** `fig_04_4b_km_os.png` -- OS Kaplan-Meier
- Explicit caveat: n=35, 5 events, HR=1.20, CI spans 0.49-2.98. Not interpretable.
- Value: framework exists, needs Tempus data to power it

#### 4.3 T-DXd Stratification (Emerging Opportunity)
- **Key figure 1:** `fig_04_5c_her2low_rna_spectrum.png` -- RNA continuous spectrum in HER2-Low
  - Shows ERBB2 is not binary within HER2-Low; continuous distribution with clear heterogeneity
- **Key figure 2:** `fig_04_5c_her2low_pathway_tertiles.png` -- pathway profiles by ERBB2 tertile
  - The "so what": upper tertile has distinct pathway activation (higher HER2 pathway,
    higher estrogen response, lower proliferation) -- the biological rationale for why
    RNA scoring might predict T-DXd benefit
- Narrative paragraph 1: T-DXd approved for HER2-Low; IHC ordinal categories are crude;
  RNA reveals a continuous spectrum that IHC collapses
- Narrative paragraph 2: Pathway tertile differences suggest differential HER2 pathway
  dependence within HER2-Low. Hypothesis: RNA-guided selection could improve benefit-risk.
  Requires Tempus ADC treatment + response data to become actionable.
- **Framing discipline:** "We see biological heterogeneity" is defensible; "the upper
  tertile would benefit more from T-DXd" is speculation without outcome data.

### 5. Strategic Synthesis

#### 5.1 Evidence Tier Summary (Markdown table)
- Tier 1 (Actionable): 5a+supp, prevalence, discordant biology
- Tier 2 (Supporting): 5b, 5c, 4a
- Tier 3 (Data-walled): 1, 4b, 5d

#### 5.2 TCGA Proves vs. Tempus Enables (Gap Table)
- Recreate the gap table from the synthesis report
- Each row: what TCGA showed -> what TCGA cannot -> what Tempus adds

#### 5.3 Proposed Tempus Study Designs
- Study 1: Prospective RNA-FISH concordance in IHC 2+ (CDx validation)
- Study 2: Retrospective cohort with treatment outcomes for discordant patients
- Study 3: T-DXd RNA enrichment (if treatment data available)
- Each with: objective, cohort definition, primary endpoint, expected timeline

#### 5.4 Honest Assessment (Markdown)
- What we can say with confidence (bulleted)
- What we hypothesize but cannot prove (bulleted)
- What we cannot say (bulleted)

---

## Figure Inventory (What stays, what's new, what's cut)

### Must-Include Figures (earn their place in the narrative)
| Figure | Source | Section | Why it stays |
|--------|--------|---------|-------------|
| Side-by-side ROC (primary + held-out) | NEW composite from fig_04_5a_roc + fig_04_5a_supp_heldout_roc | 2.2 | Headline result, transparency |
| Equivocal comparison (held-out) | fig_04_5a_supp_equivocal_comparison | 2.3 | 28/28 stability proof |
| Concordance tiers | fig_04_5b_concordance_tiers | 2.4 | Operational framework |
| ERBB2 distribution + thresholds | fig_04_2_erbb2_distribution_thresholds | 3.1 | Population sizing |
| Prevalence CN-stratified | fig_04_2_prevalence_cn_stratified | 3.1 | Two-population reveal |
| Pathway heatmap by CN stratum | fig_04_3b_pathway_heatmap | 3.2 | Central biology figure |
| Synthesis discordant | fig23_synthesis_discordant | 3.2 | Multi-panel summary |

### Include If Space Allows (supporting)
| Figure | Source | Section | Role |
|--------|--------|---------|------|
| Equivocal ERBB2 comparison | fig_04_5a_equivocal_erbb2_comparison | 2.3 | ERBB2 levels in reclassified |
| SHAP importance | fig16_shap_importance or fig17_shap_importance | 2.5 | Feature panel validation |
| DE volcano | fig22_de_volcano_key_genes | 3.4 | DE evidence |
| Pathway comparison | fig25_pathway_comparison_discordant | 3.4 | Pathway evidence |
| KM OS | fig_04_4b_km_os | 4.1 | Supplementary, caveated |
| HER2-Low RNA spectrum | fig_04_5c_her2low_rna_spectrum | 4.3 | T-DXd spectrum |
| HER2-Low pathway tertiles | fig_04_5c_her2low_pathway_tertiles | 4.3 | T-DXd biological rationale |

### Cut (do not include in synthesis notebook)
- All NB01 QC figures (fig01-fig08) -- foundational but not biopharma-relevant
- Clustering figures (fig13, fig14) -- methodology, not deliverable
- fig_04_2_prevalence_by_threshold -- redundant with CN-stratified view
- fig_04_3a_tumor_normal_erbb2_ratio -- supporting detail, not story-advancing
- fig_04_3c_er_quantitative -- detail within discordant biology
- fig_04_3d_fga_comparison -- detail within discordant biology
- fig_04_5a_concordance_score_by_fish -- redundant with ROC
- fig_04_5a_equivocal_score_distribution -- redundant with comparison
- fig_04_5c_erbb2_vs_ml_scatter -- detail
- fig_04_5c_her2low_ml_density -- detail
- fig_04_4b_km_dfs -- DFS less relevant than OS for this narrative
- fig_04_5a_supp_heldout_scores_by_fish -- redundant with ROC

---

## Implementation Steps

### Step 1: Create notebook skeleton
- Create `notebooks/04_Biopharma_Synthesis.ipynb` with all markdown cells and section headers
- Include the executive summary and all narrative markdown
- No code cells yet

### Step 2: Data loading cell
- Single cell that loads all required intermediates
- Print cohort summary table

### Step 3: Section 2 (Equivocal CDx) code cells
- Cell 2.2: Load primary + held-out predictions; generate composite ROC figure; print metrics table
- Cell 2.3: Load equivocal scores; display stability comparison; print reclassification summary
- Cell 2.4: Load concordance tier data; display tier figure; print tier table
- Cell 2.5: Brief feature importance display (optional)

### Step 4: Section 3 (Missed HER2) code cells
- Cell 3.1: Load prevalence data; display distribution + CN-stratified figures; print prevalence table
- Cell 3.2: Load pathway scores; display pathway heatmap + synthesis figure; print comparison table
- Cell 3.3: Print clinical correlates summary (no figure)
- Cell 3.4: Display DE volcano and pathway comparison

### Step 5: Section 4 (Supporting/Emerging) code cells
- Cell 4.1: Print testing method annotation summary; markdown frames the data gap
- Cell 4.2: Recreate KM curve from intermediates with explicit caveats
- Cell 4.3a: Recreate HER2-Low RNA spectrum from intermediates
- Cell 4.3b: Recreate pathway tertile comparison from intermediates

### Step 6: Section 5 (Strategic Synthesis) -- all markdown
- Evidence tier table
- Gap table
- Study designs
- Honest assessment

### Step 7: Review and polish
- Ensure figure captions are precise and biopharma-appropriate
- Verify all caveats are present (especially survival, correlation reversal)
- Check that no overclaims survive (AUC 0.994 as headline, survival as finding, etc.)
- Ensure the notebook can execute end-to-end from saved intermediates

---

## Resolved Design Decisions

1. **Composite ROC figure:** Generate in-notebook from prediction parquets via sklearn.
   Transparent, reproducible, consistent with execution model.

2. **T-DXd section depth:** Slightly expanded -- 2 figures (RNA spectrum + pathway
   tertiles) + 2 paragraphs. Shows therapeutic area fluency without overclaiming.

3. **Testing method confounder:** Own subsection (4.1) -- frames the data gap as a
   Tempus differentiator.

4. **Execution model:** Re-run from intermediates. All figures generated in-notebook,
   not displayed from pre-generated PNGs. Reproducible and transparent.

5. **Audience:** Take-home deliverable. Content demonstrates ability to link scientific
   analysis to biopharma needs. Format stays faithful to the test's ask. Tone is
   analytical assessment (not sales pitch, not internal review).
