# ML Exploration of HER2 Biology

**Author:** Matt Schinn, Ph.D.  
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
notebook can be executed independently. Note however, that due to file size `data/` and `outputs/` have not been included into this repo. 

---