# Methodology and AI Usage Summary

**Author:** Mat Schinn, Ph.D.
**Project:** Tempus HER2 Coding Challenge -- TCGA BRCA Molecular Profiling
**Date:** April 2026

---

## Part 1: Quality Control Methodology

### Overview

Quality control in this project follows a principle of **QC as scientific argument**: each step states an assumption, describes a test, interprets the result, and decides how to proceed. The QC pipeline spans clinical data harmonization, RNA-seq normalization, and copy number validation across Notebook 01 (`01_QC_and_Normalization`).

### 1.1 Clinical Data QC

**Encoding harmonization.** TCGA clinical data contains inconsistent encodings -- mixed capitalization, trailing whitespace, and multiple representations of missing data. A systematic harmonization step maps all TCGA sentinel values (e.g., "[Not Available]", "[Not Applicable]", "[Discrepancy]") to NaN, normalizes casing, and extracts derived fields (patient ID, tissue source site, diagnosis year).

**Missingness audit.** For every column used downstream, missingness is computed and visualized. The audit distinguishes between truly empty fields and explicit sentinel strings. Key finding: HER2 label missingness is structured and correlates with diagnosis year (pre-2007 samples lack IHC scoring, consistent with the introduction of ASCO/CAP HER2 testing guidelines in 2007). This is expected and does not indicate data quality problems -- it reflects evolving clinical practice.

**HER2 composite label construction.** A tiered label following simplified ASCO/CAP guidelines:

| IHC Score | FISH Status | Composite Label |
|-----------|-------------|-----------------|
| 3+        | Any         | Positive        |
| 2+        | Positive    | Positive        |
| 2+        | Negative    | Negative        |
| 2+        | Equivocal/Missing | Equivocal |
| 0 or 1+  | Any         | Negative        |
| Missing   | Positive    | Positive (FISH-only) |
| Missing   | Negative    | Negative (FISH-only) |

A critical bug was discovered and fixed: float `3.0` stored in some rows failed string comparison against `"3+"`. The `_parse_ihc_score()` helper handles this type mismatch. Contradiction flagging identifies cases where IHC and FISH disagree (e.g., IHC 3+/FISH-negative), which are individually reviewed rather than silently resolved.

**Outcome data audit.** Survival endpoints (overall survival, disease-free survival) are audited for completeness before downstream clinical outcome analysis.

**Cohort definitions.** Three analyzable cohorts are defined:
- Cohort A: Positive/Negative HER2 label + multimodal data (~185 patients) -- ML training
- Cohort B: Full clinical cohort (~1,108 patients) -- clinical-only analyses
- Cohort C: All patients with multimodal data including Equivocal (~203 patients)

### 1.2 RNA-Seq QC and Normalization

**Library size assessment.** Total RSEM estimated counts per sample show a CV of ~7%, indicating the data may already be depth-normalized upstream (RSEM can output scaled estimates). This observation directly informs the normalization decision.

**Gene filtering.** Genes with >50% zero expression across samples are removed, reducing noise for unsupervised clustering. HER2 pathway genes (ERBB2, GRB7, STARD3, PGAP3, TCAP, and others in the 17q12 amplicon) are exempt from filtering to ensure the biological signal of interest is never inadvertently dropped.

**Mean-variance relationship.** The data is tested for the negative binomial mean-variance relationship (log-log slope ~2) to inform normalization method selection.

**Normalization method selection.** A systematic 25-metric comparison evaluated three approaches:
1. **Log2(x+1)** -- baseline variance stabilization
2. **Upper-quartile (UQ) + log2(x+1)** -- standard depth correction
3. **TMM (trimmed mean of M-values) + log2(x+1)** -- Robinson & Oshlack 2010

TMM was selected as the canonical pipeline based on:
- Stronger ERBB2 signal preservation (Cohen's d = 2.19 vs 2.09 for UQ; AUC-ROC = 0.863 vs 0.837)
- Best HER2 cluster purity in unsupervised clustering (0.591 vs 0.579)
- Modestly lower residual read-depth confound after TSS correction (r ~ 0.14 vs 0.18)
- TPM was explicitly excluded to avoid double length correction on top of RSEM

Note: TSS batch correction (regression with protected HER2/ER covariates) is the
primary driver of read-depth confound removal. TMM's advantage is concentrated in
HER2 signal preservation -- it avoids the 75th-percentile over-correction that UQ
applies to HER2-amplified samples with inflated upper-tail expression.

**Post-normalization verification.** Four checks confirm normalization quality:
1. Per-sample distribution alignment (boxplots, CV reduction)
2. PCA--library size decorrelation (Pearson r on top PCs)
3. PCA--clinical variable association heatmap (systematic testing of which clinical/technical variables associate with each PC using Pearson correlation for continuous and eta-squared for categorical variables)
4. Q-Q plots for normality assessment

Key finding: ER/PR status dominates PC2 while HER2 signal appears on PC4/PC8, confirming HER2 biology occupies a distinct variance axis from hormone receptor status.

### 1.3 Batch Effect Assessment and Correction

**Tissue source site (TSS)** is the most common source of batch effects in TCGA data. TSS is significantly associated with 8/10 top PCs and is confounded with HER2 status (chi2 = 177.9, p = 3.73e-10).

**Phase 1: Quantification.** TSS batch effects are quantified via eta-squared across principal components.

**Phase 2: Regression-based correction.** TSS effects are regressed out while explicitly preserving HER2 and ER signal via protected covariates. For each gene: `expression = intercept + TSS_dummies + HER2_dummy + ER_dummy + residual`. The corrected expression retains the biological signal while removing site-specific technical variation. ComBat was evaluated but rejected due to non-orthogonal covariates in this dataset.

**Phase 3: Validation.** PCA is re-run on corrected expression; TSS eta-squared drops while HER2 eta-squared is preserved or strengthened.

### 1.4 Copy Number QC

ERBB2 copy number data (GISTIC discrete calls, -2 to +2) is validated for distribution and completeness. This provides an independent genomic readout of ERBB2 amplification status for multi-modal HER2 classification.

### 1.5 17q12 Amplicon Gene Assessment

The HER2/ERBB2 amplicon on chromosome 17q12 contains co-amplified genes (GRB7, PPP1R1B, STARD3, TCAP, PNMT, TOP2A, PGAP3). These are assessed for disproportionate influence on PCA structure, confirming that the amplicon drives a coherent transcriptomic signal rather than an artifact.

---

## Part 2: LLM Usage in This Project

### 2.1 Overview of AI Integration

This project uses Claude (Anthropic) at two distinct levels:

1. **Claude Code as interactive analytical partner** -- the primary mode throughout the project, used for iterative analysis development, code generation, scientific reasoning, and biological interpretation.
2. **A custom LangGraph agentic pipeline** -- a purpose-built 7-agent system for automated clinical data column cleaning, demonstrating LLM-based data harmonization at the pipeline level.

Additionally, the Claude Code environment was configured with **three persona skills** (Analyst, Strategist, Coder) that provide domain-specialized reasoning templates.

### 2.2 Claude Code: Interactive Analytical Partner

**Mode of use.** Claude Code served as a co-scientist throughout the project. The human PI (Mat Schinn) made all scientific decisions -- normalization method choice, label construction logic, feature selection strategy, biological interpretation -- while Claude accelerated implementation and served as a sounding board for analytical reasoning.

**What AI accelerated:**
- Boilerplate code: data loading, pandas operations, sklearn pipeline setup, matplotlib/seaborn plotting
- Literature recall: HER2 pathway gene annotations, ASCO/CAP guideline details, PAM50 subtype biology
- Code debugging and refactoring
- Systematic evaluation of analytical alternatives (e.g., normalization method comparison)

**What required human judgment:**
- Normalization method choice (recognizing pre-normalized data, deciding against further depth correction, selecting TMM over UQ based on the 25-metric comparison)
- HER2 label construction logic (mapping clinical guidelines to code, identifying the float-to-string type mismatch)
- Biological interpretation of ML features and discordant cases
- Clinical framing (population sizing, T-DXd relevance, translational significance)
- Quality control decisions (gene filtering threshold, batch effect assessment strategy)

**Example interactions:**

*Normalization decision-making:*
> "The RSEM expected counts show library-size CV of only 7% and UQ size factors near unity. What does this mean for normalization? Should I still apply DESeq2, or is log2(x+1) sufficient?"

*Label construction validation:*
> "Review my HER2 composite label function following ASCO/CAP guidelines. Are there edge cases I'm missing? What about IHC 3+/FISH- discordances?"

*Biological interpretation:*
> "The XGBoost feature importance shows TSNAX and IRX2 above ERBB2. Is this biologically plausible or does it suggest overfitting?"

**Validation process:**
- Normalization code manually verified against DESeq2 documentation and RSEM output specifications
- HER2 label logic validated by cross-tabulating derived labels against pre-coded IHC-HER2 values
- ML models evaluated via cross-validation (not train-test split) to prevent overfitting on the small dataset
- Biological interpretations validated against published TCGA BRCA analyses (Cancer Genome Atlas Network, Nature 2012)

### 2.3 Persona Skills: Structured AI Reasoning

Three Claude Code persona skills were configured to provide domain-specialized reasoning:

| Persona | Purpose | Model Tier | Trigger |
|---------|---------|------------|---------|
| **Analyst** | Evaluate results, check biological plausibility, flag statistical issues | Opus | "Does this make sense?", "Check this result" |
| **Strategist** | Plan next steps, prioritize analyses, shape narrative arc | Opus | "What's next?", "What should I prioritize?" |
| **Coder** | Implement analyses as notebook cells, translate plans to Python | Sonnet | "Implement this", "Code this up" |

**Design rationale.** The Analyst and Strategist perform judgment-heavy reasoning (weighing biological plausibility, spotting circularity, prioritization under uncertainty) and use Opus-class models. The Coder translates already-decided analytical plans into Python following well-defined conventions -- structured, specification-following work where Sonnet is strong and additional reasoning capacity adds minimal value.

**Escalation protocol.** Personas follow a defined escalation chain:
- Analyst evaluates -> Strategist plans -> Coder implements
- If a bug resists two fix attempts, the Coder escalates to the Analyst (the float-to-string type mismatch in HER2 label construction is the canonical example)
- The Strategist defers to the Analyst when prerequisite results are uncertain

**Key features of each persona:**

*Analyst* -- convergent and critical. Evaluates along five dimensions: biological plausibility, statistical rigor, methodological coherence, assignment alignment, and figure/table quality. Contains project-specific priors (e.g., ERBB2 amplicon co-expression, RNA > CN for HER2 prediction, temporal confounding from ASCO/CAP guideline changes).

*Strategist* -- divergent and generative. Prioritizes by information value (must-do / should-do / nice-to-have / skip). Maintains awareness of the project's narrative arc and scope constraints (coding challenge, not a thesis). Makes ranked recommendations rather than flat menus of options.

*Coder* -- implementation-focused. Follows established conventions (cell organization, naming, data contracts). Explicitly instructed not to re-derive analytical logic. Uses defensive data shape checking and explicit NaN handling. ASCII-only output to avoid Windows encoding issues.

### 2.4 Agentic Column Cleaner: LLM-Based Data Harmonization Pipeline

A custom 7-agent pipeline built with LangGraph and Claude (Sonnet) for cleaning messy clinical data columns. This demonstrates LLM-based data harmonization as a systematic, auditable workflow rather than ad-hoc prompting.

**Architecture:**

```
A1: Format Inspector  <-->  A2: Format Reviewer   (max 2 loops)
         |
A3: Semantic Analyzer  ---- "no harmonization" --> A7: Report
         |
         | "needs harmonization"
         v
A4: Strategy Architect <-->  A5: Strategy Reviewer  (max 2 loops)
         |
A6: Executor          <-->  A7: Final QC + Report  (max 1 loop)
```

**Agent roles:**

| Agent | Role |
|-------|------|
| A1: Format Inspector | Fix whitespace, casing, typos, number formats |
| A2: Format Reviewer | Verify only formatting changed (no semantic merges) |
| A3: Semantic Analyzer | Decide if harmonization is needed (the fork) |
| A4: Strategy Architect | Design harmonization approach (split, merge, binarize) |
| A5: Strategy Reviewer | Verify strategy completeness and correctness |
| A6: Executor | Apply the strategy to every value |
| A7: Final QC + Report | Validate output and document everything |

**Key design decisions:**

1. **Review loops with caps.** Each stage has a proposer-reviewer pair with maximum iteration limits (format: 2, strategy: 2, final: 1), preventing infinite review cycles while allowing self-correction.

2. **Context-first architecture.** Every agent receives a rich CONTEXT document auto-generated from project files, including project objectives, relevant biology (HER2/ERBB2, ER scoring systems, ASCO/CAP guidelines), dataset schema, and column-specific guidance (value ranges, clinical thresholds, typical messiness patterns).

3. **Semantic fork.** Agent A3 decides whether a column needs harmonization or only formatting. Simple columns (spelling dedup) skip the strategy stage entirely. Complex columns (multiple measurement systems) proceed through full strategy design and review.

4. **Measurement system detection.** The pipeline recognizes when a column contains incommensurable scales (e.g., Allred scores + H-scores + fmol/mg in ER positivity data) and splits into separate typed columns rather than forcing lossy harmonization.

5. **Full auditability.** Every run produces a trace (agent name, decision, timing per step), a markdown report documenting the cleaning rationale, and the complete LangGraph state for debugging.

**Application to this project:**

The pipeline was applied to two clinical data columns:
- **HER2 positivity method text** (simpler case): 20 spelling variants of test method names deduplicated into 6 canonical categories (Dako HercepTest, CISH, Ventana, CAP 2010 Guidelines, Dextran Coated Charcoal, IHC Result Description).
- **ER positivity scale other** (complex case): Free-text field with 5+ incommensurable measurement systems (Allred scores 0-8, H-scores 0-300, fmol/mg concentrations, qualitative intensity, percentages, method references) split into 6 typed sub-columns.

**Hybrid execution.** The final production runner (`run_tier1_cleaning.py`) uses deterministic regex-based cleaners for these two columns (since the LLM-based executor was found to be inconsistent across runs for large value sets), while retaining the agentic pipeline as the design and validation framework that informed the deterministic rules. The LLM pipeline remains available for new columns via `run_pipeline.py`.

**Supporting infrastructure:**
- `context_generator.py`: Auto-generates column-specific CONTEXT documents with baked-in clinical knowledge
- `test_dry_run.py`: Mock-based tests validating the full graph topology and agent sequencing without API calls
- `run_tier1_cleaning.py`: Production runner that also handles a mislabeled column rename (Brachytherapy -> ER positivity scoring method)

### 2.5 Project Configuration for AI-Assisted Development

The project maintains several configuration files that structure the AI collaboration:

**CLAUDE.md** -- Project-level instructions for Claude Code, including:
- Data flow architecture and notebook dependencies
- Key conventions (ASCII-only, patient ID handling, normalization pipeline)
- Known pitfalls (type mismatches, circularity risks, small sample warnings)
- Feature reduction constraints (always use curated gene panels, never raw full-transcriptome)

**AGENTS.md** -- Documents the persona skill system, model selection rationale, invocation patterns, and escalation rules. Treats the skills as living documents updated as the project progresses.

**Shared knowledge base** (`skills/user/shared/`) -- Analysis plans, findings, and strategic documents that persist across conversations and inform both human and AI decision-making. Examples include TSS batch strategy, normalization comparison plans, ML consolidation plans, and biopharma analysis plans.
