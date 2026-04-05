# Tempus HER2 Coding Challenge

**Author:** Mat Schinn, Ph.D.  
**Date:** April 2026

---

## Project Structure

```
├── data/                          # Raw data files
│   ├── brca_tcga_clinical_data.csv
│   ├── tcga.brca.rsem.csv
│   └── brca_tcga_erbb2_copy_number.csv
├── outputs/                       # Intermediates and figures
│   ├── *.parquet                  # Intermediate data files
│   └── figures/                   # Publication-ready figures
├── src/
│   └── utils.py                   # Shared utilities
├── notebooks/
│   ├── 01_QC_and_Normalization    # Clinical QC, RNA-seq normalization, CN QC
│   ├── 02_HER2_Identification     # HER2 identification, discordant cases, clustering
│   ├── 03_Machine_Learning        # ML models, SHAP, equivocal scoring
│   └── 04_Deep_Dive_and_Clinical  # Normal vs tumor, HER2-low, survival, population sizing
└── README.md
```

Notebooks should be executed in order. Each notebook loads upstream intermediates from
`outputs/` and saves its own outputs there. After Notebook 01 runs, any downstream
notebook can be executed independently.

---

## Methodology Summary

### QC & Normalization
- Clinical data harmonized with TCGA sentinel value mapping and composite HER2 label 
  construction following ASCO/CAP guidelines (tiered: IHC score → FISH → pre-coded).
- RNA-seq gene filtering at ≤50% zero expression with pathway gene exemption.
- RSEM expected counts show minimal library-size variation (CV ~7%), indicating upstream 
  normalization. Applied log2(x+1) for variance stabilization. Verified by PCA–library 
  size decorrelation and size factor analysis.

### HER2 Identification
- ERBB2 RNA expression outperforms copy number (AUC 0.84 vs 0.81) for predicting 
  clinical IHC status. Combined model adds no benefit.
- Discordant cases identified across IHC/FISH/RNA/CN modalities with biological 
  interpretation (polysomy 17, epigenetic silencing, missed HER2+).

### Machine Learning
- XGBoost, Random Forest, and L1-Logistic Regression compared via 5-fold stratified CV.
- SHAP analysis for biologically interpretable feature importance.
- Equivocal IHC cases scored with trained model to demonstrate RNA-based resolution.

### Clinical Outcomes & Actionability
- Kaplan-Meier survival analysis by cluster and molecular concordance.
- Population sizing: molecularly reclassified patients extrapolated to US incidence.

---

## AI Workflow Description

### Tools Used
- **Claude** (claude.ai) — Used as both a coding assistant and analytical partner 
  throughout the project.
- **Mode:** Primarily interactive assistant mode with iterative refinement. 
  Claude generated initial code scaffolds for data wrangling, sklearn pipelines, and 
  plotting functions. All scientific decisions (normalization method, label construction 
  logic, feature selection, biological interpretation) were made by the PI with Claude 
  as a co-scientist.

### Example Prompts

**Normalization decision-making:**
> "The RSEM expected counts show library-size CV of only 7% and UQ size factors near 
> unity. What does this mean for normalization? Should I still apply DESeq2, or is 
> log2(x+1) sufficient? Walk me through the reasoning."

**Label construction validation:**
> "Here is my HER2 composite label function following ASCO/CAP guidelines. Review the 
> tiered logic. Are there edge cases I'm missing? What about IHC 3+/FISH- discordances?"

**Biological interpretation:**
> "The XGBoost feature importance shows TSNAX and IRX2 above ERBB2. Is this biologically 
> plausible or does it suggest overfitting? What would SHAP values show differently?"

### Validation Process
- **Normalization code** manually verified against DESeq2 documentation and RSEM output 
  specifications. Size factor distributions inspected to confirm expected behavior.
- **HER2 label logic** validated by cross-tabulating derived labels against pre-coded 
  IHC-HER2 values. Contradictions individually reviewed.
- **ML models** evaluated via cross-validation (not train-test split) to prevent 
  overfitting on the small dataset. SHAP used for feature importance to avoid 
  gain-based bias.
- **Biological interpretations** validated against published TCGA BRCA analyses 
  (Cancer Genome Atlas Network, Nature 2012) and ASCO/CAP HER2 testing guidelines.

### What AI Accelerated
- Boilerplate: data loading, plotting, sklearn pipeline setup, pandas operations.
- Literature recall: HER2 pathway gene annotations, ASCO/CAP guideline details.
- Code debugging and refactoring.

### What Required Human Judgment
- Normalization method choice (recognizing pre-normalized data, deciding against 
  further depth correction).
- HER2 label construction logic (mapping clinical guidelines to code).
- Biological interpretation of ML features and discordant cases.
- Clinical framing (population sizing, T-DXd relevance, translational significance).
- Quality control decisions (gene filtering threshold, batch effect assessment).
