"""
03_exploratory_multiclass_ml.py
================================
Exploratory multi-class subtype ML and pathway analysis extracted from Notebook 03b.

Contains:
- MSigDB Hallmark gene set definitions (curated subset for breast cancer)
- One-vs-rest GSEA by subtype
- HER2+ vs HER2- GSEA
- ssGSEA score computation and export
- Multi-class model comparison (3 models x feature sets)
- Per-subtype SHAP analysis
- GSEA-vs-SHAP biological cross-validation
- Kaplan-Meier survival overlay by subtype
- Equivocal scoring by subtype
- Clinical implications summary

This script is self-contained. The consolidated notebook loads pre-computed
ssGSEA scores from outputs/03_ssgsea_scores.parquet.

Outputs:
- outputs/03_ssgsea_scores.parquet
- outputs/03_subtype_gsea.parquet
- outputs/03_multiclass_results.parquet
- outputs/figures/fig_03s_gsea_*.png
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix,
                             f1_score)
from sklearn.preprocessing import StandardScaler, label_binarize

try:
    import xgboost as xgb
    has_xgb = True
except ImportError:
    has_xgb = False
    from sklearn.ensemble import GradientBoostingClassifier

try:
    import shap
    has_shap = True
except ImportError:
    has_shap = False

try:
    import gseapy as gp
    has_gseapy = True
except ImportError:
    has_gseapy = False
    print("gseapy not available. Install with: pip install gseapy")

from src.utils import (
    load_intermediate, load_gene_cols, save_intermediate, savefig,
    setup_plotting, HER2_PATHWAY_GENES
)

setup_plotting()
print(f"XGBoost: {'available' if has_xgb else 'NOT available'}")
print(f"SHAP: {'available' if has_shap else 'NOT available'}")
print(f"gseapy: {'available' if has_gseapy else 'NOT available'}")


# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n" + "=" * 70)
print("EXPLORATORY MULTI-CLASS ML & PATHWAY ANALYSIS")
print("=" * 70)

clinical = load_intermediate('01_clinical_qc')
tumor_norm = load_intermediate('01_tumor_norm_tmm_tss')
cn = load_intermediate('01_cn_qc')
gene_cols = load_gene_cols()
cohort_c = load_intermediate('02_multimodal_cohort')
analysis_df = load_intermediate('02_analysis_df')
subtype_df = load_intermediate('02_subtype_assignments')
cluster_df = load_intermediate('02_cluster_assignments')

print(f"Subtype assignments loaded: {len(subtype_df)} patients")
print(f"Subtypes: {subtype_df['provisional_subtype'].value_counts().to_dict()}")


# ============================================================================
# 2. MSigDB HALLMARK GENE SETS
# ============================================================================
HALLMARK_GENE_SETS = {
    'HALLMARK_PI3K_AKT_MTOR_SIGNALING': [
        'AKT1', 'AKT2', 'AKT3', 'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3CG',
        'PIK3R1', 'PIK3R2', 'PIK3R3', 'MTOR', 'RPTOR', 'RICTOR', 'RPS6KB1',
        'RPS6KB2', 'EIF4EBP1', 'PDPK1', 'TSC1', 'TSC2', 'RHEB', 'PTEN',
        'GSK3B', 'FOXO1', 'FOXO3', 'BAD', 'BCL2L11', 'CDKN1A', 'CDKN1B',
        'MDM2', 'RPS6', 'EIF4E', 'EIF4G1', 'INPP5D', 'INPP4B', 'SGK1',
        'CRTC1', 'IRS1', 'IRS2', 'GAB1', 'GAB2', 'GRB2', 'SOS1', 'HRAS',
        'NRAS', 'KRAS', 'BRAF', 'RAF1', 'MAP2K1', 'MAPK1', 'MAPK3',
        'PREX1', 'DEPTOR', 'MLST8', 'MAPKAP1', 'PRAS40', 'STRADA',
    ],
    'HALLMARK_MTORC1_SIGNALING': [
        'MTOR', 'RPTOR', 'MLST8', 'AKT1S1', 'DEPTOR', 'RPS6KB1', 'EIF4EBP1',
        'RPS6', 'EIF4E', 'EIF4G1', 'EEF2K', 'EEF2', 'DDIT4', 'HIF1A',
        'VEGFA', 'SLC2A1', 'LDHA', 'PKM', 'ENO1', 'HK2', 'PFKFB3',
        'ACACA', 'FASN', 'SCD', 'SREBF1', 'SREBF2', 'HMGCR', 'HMGCS1',
        'IDI1', 'MVD', 'MVK', 'SQLE', 'ACLY', 'PSMD1', 'PSMD2', 'PSMD3',
        'PSMD4', 'PSMD6', 'PSMD7', 'PSMD11', 'PSMD12', 'PSMD13', 'PSMD14',
        'PSMA1', 'PSMA2', 'PSMA3', 'PSMA4', 'PSMA5', 'PSMA6', 'PSMA7',
        'PSMB1', 'PSMB2', 'PSMB3', 'PSMB4', 'PSMB5', 'PSMB6', 'PSMB7',
        'CDK4', 'CCND1', 'CCND2', 'MYC', 'SRM', 'ODC1', 'AMD1', 'SMS',
    ],
    'HALLMARK_E2F_TARGETS': [
        'E2F1', 'E2F2', 'E2F3', 'RB1', 'RBL1', 'RBL2', 'CCNA2', 'CCNB1',
        'CCNB2', 'CCNE1', 'CCNE2', 'CDK1', 'CDK2', 'CDC6', 'CDC20',
        'CDC25A', 'CDC25C', 'CDC45', 'CDT1', 'MCM2', 'MCM3', 'MCM4',
        'MCM5', 'MCM6', 'MCM7', 'ORC1', 'ORC2', 'ORC3', 'ORC5', 'ORC6',
        'PCNA', 'POLA1', 'POLA2', 'POLD1', 'POLD2', 'POLE', 'POLE2',
        'RFC1', 'RFC2', 'RFC3', 'RFC4', 'RFC5', 'RPA1', 'RPA2', 'RPA3',
        'BRCA1', 'BRCA2', 'RAD51', 'RAD54L', 'BLM', 'CHEK1', 'CHEK2',
        'TOPBP1', 'CLSPN', 'TIMELESS', 'TIPIN', 'FANCD2',
        'TOP2A', 'SMC1A', 'SMC3', 'KIF11', 'AURKB', 'PLK1', 'BUB1',
        'BUB1B', 'MAD2L1', 'CENPE', 'CENPF', 'KIF2C', 'TPX2', 'BIRC5',
        'MKI67', 'FOXM1', 'MYBL2', 'HMGA1', 'DEK', 'NASP', 'ASF1B',
    ],
    'HALLMARK_G2M_CHECKPOINT': [
        'CDK1', 'CCNB1', 'CCNB2', 'CCNA2', 'CDC20', 'CDC25B', 'CDC25C',
        'PLK1', 'AURKB', 'AURKA', 'BUB1', 'BUB1B', 'MAD2L1', 'TTK',
        'CENPA', 'CENPE', 'CENPF', 'KIF11', 'KIF2C', 'KIF23', 'KIF4A',
        'TPX2', 'BIRC5', 'TOP2A', 'SMC2', 'SMC4', 'NCAPD2', 'NCAPG',
        'NCAPH', 'NUSAP1', 'PRC1', 'ECT2', 'ANLN', 'MKI67', 'FOXM1',
        'NEK2', 'DLGAP5', 'HMMR', 'TROAP', 'HJURP', 'CKAP2', 'CKAP5',
        'CDCA3', 'CDCA5', 'CDCA8', 'SKA1', 'SKA3', 'NDC80', 'NUF2',
        'SPC24', 'SPC25', 'SGOL1', 'TACC3', 'CHEK1', 'WEE1', 'MYT1',
    ],
    'HALLMARK_MYC_TARGETS_V1': [
        'MYC', 'MYCN', 'MAX', 'NCL', 'NPM1', 'NOP56', 'NOP58', 'FBL',
        'DKC1', 'GAR1', 'NHP2', 'NOP10', 'RRP9', 'UTP14A', 'UTP15',
        'UTP20', 'WDR12', 'BOP1', 'PES1', 'RRS1', 'DDX21', 'DDX18',
        'EIF4A1', 'EIF4E', 'EIF4G1', 'EIF3A', 'EIF3B', 'EIF3C',
        'RPS2', 'RPS3', 'RPS3A', 'RPS4X', 'RPS5', 'RPS6', 'RPS7', 'RPS8',
        'RPS9', 'RPS10', 'RPS11', 'RPS12', 'RPS13', 'RPS14', 'RPS15',
        'RPL3', 'RPL4', 'RPL5', 'RPL6', 'RPL7', 'RPL7A', 'RPL8', 'RPL9',
        'RPL10', 'RPL11', 'RPL12', 'RPL13', 'RPL13A', 'RPL14', 'RPL15',
        'LDHA', 'PKM', 'ENO1', 'HK2', 'SRM', 'ODC1', 'CDK4', 'CCND2',
    ],
    'HALLMARK_ESTROGEN_RESPONSE_EARLY': [
        'ESR1', 'GATA3', 'FOXA1', 'XBP1', 'TFF1', 'TFF3', 'CCND1',
        'MYB', 'PGR', 'PDZK1', 'CA12', 'STC2', 'IGFBP4', 'RARA',
        'NRIP1', 'TRIM25', 'SLC9A3R1', 'SIAH2', 'ABAT', 'MAPT',
        'AGR2', 'ANXA9', 'SLC39A6', 'NAT1', 'SCUBE2', 'MLPH',
        'AR', 'ERBB3', 'ERBB4', 'MUC1', 'ELF3', 'KRT18', 'KRT19',
        'KRT8', 'CLDN3', 'CLDN4', 'CLDN7', 'EHF', 'GRHL2', 'SPDEF',
        'CELSR1', 'GALNT6', 'SERPINA3', 'SERPINA5', 'SERPINA6',
        'CYP2B6', 'SLC7A2', 'PRSS8', 'BBOX1', 'KCNK5',
    ],
    'HALLMARK_ESTROGEN_RESPONSE_LATE': [
        'ESR1', 'PGR', 'GATA3', 'FOXA1', 'XBP1', 'TFF1', 'TFF3',
        'MYB', 'AR', 'CCND1', 'STC2', 'CA12', 'AGR2', 'NAT1',
        'MAPT', 'ABAT', 'SLC39A6', 'SERPINA3', 'KRT18', 'KRT19',
        'PDZK1', 'IGFBP4', 'ERBB4', 'MUC1', 'ELF3', 'SPDEF',
        'SLC44A4', 'GALNT6', 'PRSS8', 'CYP2B6', 'BBOX1',
    ],
    'HALLMARK_INTERFERON_GAMMA_RESPONSE': [
        'STAT1', 'IRF1', 'IRF9', 'IFIT1', 'IFIT2', 'IFIT3', 'IFITM1',
        'IFI35', 'OAS1', 'OAS2', 'OAS3', 'MX1', 'MX2', 'ISG15', 'ISG20',
        'GBP1', 'GBP2', 'GBP4', 'GBP5', 'TAP1', 'TAP2', 'PSMB8',
        'PSMB9', 'PSMB10', 'B2M', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-E',
        'CXCL9', 'CXCL10', 'CXCL11', 'CCL5', 'CD274', 'IDO1', 'WARS',
    ],
    'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION': [
        'VIM', 'FN1', 'CDH2', 'SNAI1', 'SNAI2', 'ZEB1', 'ZEB2',
        'TWIST1', 'COL1A1', 'COL1A2', 'COL3A1', 'COL5A1', 'COL5A2',
        'SPARC', 'THBS1', 'TNC', 'POSTN', 'LOX', 'ACTA2', 'TAGLN',
        'MMP2', 'MMP3', 'TIMP1', 'SERPINE1', 'TGFBI', 'BGN', 'DCN',
        'LUM', 'FAP', 'PDGFRB', 'CALD1', 'FSTL1', 'IGFBP3',
    ],
    'HALLMARK_P53_PATHWAY': [
        'TP53', 'MDM2', 'MDM4', 'CDKN1A', 'CDKN2A', 'BAX', 'BBC3',
        'PMAIP1', 'FAS', 'TNFRSF10B', 'GADD45A', 'GADD45B', 'GADD45G',
        'DDB2', 'XPC', 'SESN1', 'SESN2', 'TIGAR', 'SCO2', 'DRAM1',
        'PERP', 'PIDD1', 'EI24', 'PTEN', 'TSC2', 'IGFBP3', 'SERPINE1',
        'THBS1', 'RRM2B', 'SFN', 'STEAP3', 'FDXR', 'AEN', 'ZMAT3',
    ],
}


# ============================================================================
# 3. ONE-VS-REST GSEA BY SUBTYPE
# ============================================================================
print("\n" + "=" * 70)
print("GSEA: ONE-VS-REST PER SUBTYPE")
print("=" * 70)

tumor_expr = tumor_norm.set_index('pid')[gene_cols]
subtype_map_dict = dict(zip(subtype_df['pid'], subtype_df['provisional_subtype']))
expr_pids = [p for p in tumor_expr.index if p in subtype_map_dict]
expr_subtypes = pd.Series([subtype_map_dict[p] for p in expr_pids], index=expr_pids)

print(f"Patients with expression + subtype: {len(expr_pids)}")

subtype_de = {}
for subtype in sorted(expr_subtypes.unique()):
    in_pids = expr_subtypes[expr_subtypes == subtype].index
    out_pids = expr_subtypes[expr_subtypes != subtype].index
    t_stats = {}
    for gene in gene_cols:
        in_vals = tumor_expr.loc[in_pids, gene].dropna()
        out_vals = tumor_expr.loc[out_pids, gene].dropna()
        if len(in_vals) >= 3 and len(out_vals) >= 3:
            t_stat, _ = stats.ttest_ind(in_vals, out_vals, equal_var=False)
            t_stats[gene] = t_stat
    subtype_de[subtype] = pd.Series(t_stats).sort_values(ascending=False)
    print(f"  {subtype}: {len(t_stats)} genes tested")

subtype_gsea = {}
nes_matrix = pd.DataFrame()

if has_gseapy:
    available_genes = set(tumor_expr.columns)
    filtered_gs = {}
    for name, genes in HALLMARK_GENE_SETS.items():
        overlap = [g for g in genes if g in available_genes]
        if len(overlap) >= 10:
            filtered_gs[name] = overlap

    print(f"Gene sets with >= 10 genes: {len(filtered_gs)}")

    for subtype, rnk in subtype_de.items():
        rnk_clean = rnk.dropna()
        rnk_clean = rnk_clean[~np.isinf(rnk_clean)]
        if len(filtered_gs) >= 3 and len(rnk_clean) >= 100:
            try:
                result = gp.prerank(
                    rnk=rnk_clean, gene_sets=filtered_gs,
                    outdir=None, no_plot=True,
                    min_size=10, max_size=500,
                    permutation_num=1000, seed=42, verbose=False,
                )
                res_df = result.res2d.copy()
                res_df['NES'] = pd.to_numeric(res_df['NES'], errors='coerce')
                res_df['FDR q-val'] = pd.to_numeric(res_df['FDR q-val'], errors='coerce')
                subtype_gsea[subtype] = res_df
                n_sig = (res_df['FDR q-val'] < 0.05).sum()
                print(f"  {subtype}: {len(res_df)} pathways, {n_sig} significant (FDR < 0.05)")
            except Exception as e:
                print(f"  {subtype}: GSEA failed -- {e}")

    if subtype_gsea:
        all_pathways = set()
        for df in subtype_gsea.values():
            all_pathways.update(df['Term'].values)
        nes_matrix = pd.DataFrame(index=sorted(all_pathways))
        for subtype, df in subtype_gsea.items():
            nes_series = df.set_index('Term')['NES']
            nes_matrix[subtype] = nes_series
        nes_matrix = nes_matrix.fillna(0)

        # Save GSEA results
        gsea_combined = []
        for subtype, df in subtype_gsea.items():
            df = df.copy()
            df['subtype'] = subtype
            gsea_combined.append(df)
        if gsea_combined:
            save_intermediate(pd.concat(gsea_combined, ignore_index=True), '03_subtype_gsea')

    # Heatmap
    if len(nes_matrix) > 0:
        fig, ax = plt.subplots(figsize=(max(8, len(nes_matrix) * 0.35), 6))
        col_order = [s for s in ['Luminal A', 'Luminal B', 'HER2-enriched', 'Basal-like']
                     if s in nes_matrix.columns]
        plot_data = nes_matrix[col_order] if col_order else nes_matrix

        from scipy.cluster.hierarchy import linkage, leaves_list
        if len(plot_data) > 2:
            try:
                Z = linkage(plot_data.fillna(0).values, method='ward')
                row_order = leaves_list(Z)
                plot_data = plot_data.iloc[row_order]
            except Exception:
                pass

        sns.heatmap(plot_data.T, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
                    linewidths=0.5, ax=ax, cbar_kws={'label': 'NES'},
                    xticklabels=True, yticklabels=True)
        ax.set_title('Pathway Enrichment by Subtype (NES, one-vs-rest)',
                     fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        plt.tight_layout()
        savefig(fig, 'fig_03s_gsea_subtype_heatmap')
        plt.close()


# ============================================================================
# 4. HER2+ vs HER2- GSEA
# ============================================================================
print("\n" + "=" * 70)
print("GSEA: HER2+ vs HER2-")
print("=" * 70)

labeled = cohort_c[cohort_c['her2_composite'].isin(['Positive', 'Negative'])]
labeled_pids_gsea = labeled['pid'].values
common_pids = [p for p in labeled_pids_gsea if p in tumor_expr.index]
expr_labeled = tumor_expr.loc[common_pids]
labels = labeled.set_index('pid').loc[common_pids, 'her2_composite']

pos_pids = labels[labels == 'Positive'].index
neg_pids = labels[labels == 'Negative'].index
print(f"HER2+: {len(pos_pids)}, HER2-: {len(neg_pids)}")

t_stats_h = {}
for gene in gene_cols:
    pos_vals = expr_labeled.loc[pos_pids, gene].dropna()
    neg_vals = expr_labeled.loc[neg_pids, gene].dropna()
    if len(pos_vals) >= 3 and len(neg_vals) >= 3:
        t_stat, _ = stats.ttest_ind(pos_vals, neg_vals, equal_var=False)
        t_stats_h[gene] = t_stat

her2_de = pd.DataFrame({
    't_stat': pd.Series(t_stats_h),
}).dropna()
her2_de['rank_metric'] = her2_de['t_stat']
her2_de = her2_de.sort_values('rank_metric', ascending=False)

gsea_df = pd.DataFrame()
if has_gseapy:
    rnk = her2_de['rank_metric'].dropna()
    rnk = rnk[~np.isinf(rnk)]
    available_genes_h = set(rnk.index)
    filtered_gs_h = {n: [g for g in gs if g in available_genes_h]
                     for n, gs in HALLMARK_GENE_SETS.items()}
    filtered_gs_h = {n: gs for n, gs in filtered_gs_h.items() if len(gs) >= 10}

    if len(filtered_gs_h) >= 3:
        gsea_result = gp.prerank(
            rnk=rnk, gene_sets=filtered_gs_h,
            outdir=None, no_plot=True,
            min_size=10, max_size=500,
            permutation_num=1000, seed=42, verbose=False,
        )
        gsea_df = gsea_result.res2d.copy()
        gsea_df['NES'] = pd.to_numeric(gsea_df['NES'], errors='coerce')
        gsea_df['FDR q-val'] = pd.to_numeric(gsea_df['FDR q-val'], errors='coerce')
        gsea_df = gsea_df.sort_values('NES', ascending=False)

        print(f"\n{'Pathway':<50s}  {'NES':>6s}  {'FDR':>8s}")
        print("-" * 70)
        for _, row in gsea_df.iterrows():
            sig = '**' if row['FDR q-val'] < 0.05 else '*' if row['FDR q-val'] < 0.25 else ''
            name_short = row['Term'].replace('HALLMARK_', '')
            print(f"  {name_short:<48s}  {row['NES']:>+6.2f}  {row['FDR q-val']:>8.3f} {sig}")


# ============================================================================
# 5. ssGSEA COMPUTATION
# ============================================================================
print("\n" + "=" * 70)
print("ssGSEA SCORE COMPUTATION")
print("=" * 70)

ssgsea_scores = pd.DataFrame()
pathway_cols = []

if has_gseapy:
    tumor_expr_dict = tumor_norm.set_index('pid')
    all_pids = [p for p in cohort_c['pid'].values if p in tumor_expr_dict.index]
    expr_matrix = tumor_expr_dict.loc[all_pids, gene_cols].T

    available_genes_ss = set(expr_matrix.index)
    ssgsea_gene_sets = {}
    for name, genes in HALLMARK_GENE_SETS.items():
        overlap = [g for g in genes if g in available_genes_ss]
        if len(overlap) >= 10:
            ssgsea_gene_sets[name] = overlap

    print(f"Gene sets with >= 10 genes: {len(ssgsea_gene_sets)}")

    if len(ssgsea_gene_sets) >= 3:
        try:
            ss_result = gp.ssgsea(
                data=expr_matrix, gene_sets=ssgsea_gene_sets,
                outdir=None, no_plot=True,
                min_size=10, verbose=False,
            )
            ssgsea_scores = ss_result.res2d.pivot(index='Name', columns='Term', values='NES')
            ssgsea_scores.columns = ['pathway_' + c.replace('HALLMARK_', '')
                                     for c in ssgsea_scores.columns]
            ssgsea_scores = ssgsea_scores.astype(float)
            ssgsea_scores.index.name = 'pid'
            ssgsea_scores = ssgsea_scores.reset_index()
            pathway_cols = [c for c in ssgsea_scores.columns if c.startswith('pathway_')]

            save_intermediate(ssgsea_scores, '03_ssgsea_scores')
            print(f"ssGSEA scores saved: {len(ssgsea_scores)} patients x {len(pathway_cols)} pathways")
        except Exception as e:
            print(f"ssGSEA failed: {e}")
else:
    print("gseapy not available; skipping ssGSEA.")


# ============================================================================
# 6. MULTI-CLASS MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("MULTI-CLASS SUBTYPE CLASSIFICATION")
print("=" * 70)

ml_df = cohort_c.copy()
tumor_expr_dict2 = tumor_norm.set_index('pid')
for gene in gene_cols:
    if gene in tumor_expr_dict2.columns:
        ml_df[f'expr_{gene}'] = ml_df['pid'].map(tumor_expr_dict2[gene])

ml_df['er_positive'] = (ml_df['ER Status By IHC'] == 'Positive').astype(float)
ml_df['pr_positive'] = (ml_df['PR status by ihc'] == 'Positive').astype(float)
ml_df['er_positive'] = ml_df['er_positive'].fillna(ml_df['er_positive'].median())
ml_df['pr_positive'] = ml_df['pr_positive'].fillna(ml_df['pr_positive'].median())

ml_df = ml_df.merge(subtype_df[['pid', 'provisional_subtype']], on='pid', how='left')
ml_subtype = ml_df[ml_df['provisional_subtype'].notna()].copy()

subtype_names = sorted(ml_subtype['provisional_subtype'].unique())
subtype_to_int = {s: i for i, s in enumerate(subtype_names)}
ml_subtype['y_subtype'] = ml_subtype['provisional_subtype'].map(subtype_to_int)

print(f"Patients with subtype labels: {len(ml_subtype)}")
for s in subtype_names:
    n = (ml_subtype['provisional_subtype'] == s).sum()
    print(f"  {s}: {n}")

# Feature sets
marker_genes_a = ['ESR1', 'PGR', 'ERBB2', 'GRB7', 'KRT5', 'KRT14', 'EGFR', 'MKI67']
fs_a_cols = [f'expr_{g}' for g in marker_genes_a if f'expr_{g}' in ml_df.columns]
fs_a_cols.extend(['erbb2_copy_number', 'er_positive', 'pr_positive'])

panel_genes = marker_genes_a + [
    'GATA3', 'FOXA1', 'BCL2', 'STARD3', 'PGAP3', 'ERBB3', 'ERBB4',
    'KRT17', 'KRT6B', 'TOP2A', 'AURKA', 'CCNB1', 'CCNE1',
    'PIK3CA', 'AKT1', 'MTOR', 'MAPK1', 'VIM', 'CDH1', 'CDH2',
]
fs_b_cols = [f'expr_{g}' for g in panel_genes if f'expr_{g}' in ml_df.columns]
fs_b_cols.extend(['erbb2_copy_number', 'er_positive', 'pr_positive'])
fs_b_cols = list(dict.fromkeys(fs_b_cols))

# Merge ssGSEA if available
if len(ssgsea_scores) > 0 and len(pathway_cols) > 0:
    ml_df = ml_df.merge(ssgsea_scores, on='pid', how='left')
    ml_subtype = ml_df[ml_df['provisional_subtype'].notna()].copy()
    ml_subtype['y_subtype'] = ml_subtype['provisional_subtype'].map(subtype_to_int)
    fs_c_cols = pathway_cols + ['erbb2_copy_number', 'er_positive', 'pr_positive']
    has_pathway_features = True
else:
    fs_c_cols = []
    has_pathway_features = False

feature_sets = {'A: Subtype markers': fs_a_cols}
if len(fs_b_cols) > len(fs_a_cols):
    feature_sets['B: Expanded panel'] = fs_b_cols
if has_pathway_features and len(fs_c_cols) > 3:
    feature_sets['C: Pathway scores'] = fs_c_cols

n_classes = len(subtype_names)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

comparison_rows = []
all_results_mc = {}

for fs_name, fs_cols in feature_sets.items():
    print(f"\nFeature Set: {fs_name} ({len(fs_cols)} features)")
    ml_clean = ml_subtype.dropna(subset=fs_cols + ['y_subtype'])
    X_fs = ml_clean[fs_cols].values
    y_fs = ml_clean['y_subtype'].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fs)

    mc_models = {
        'L1-LR': LogisticRegressionCV(
            penalty='l1', solver='saga', max_iter=5000, random_state=42,
            class_weight='balanced', Cs=10, cv=3, scoring='roc_auc_ovr',
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42,
            class_weight='balanced', n_jobs=-1
        ),
    }
    if has_xgb:
        mc_models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric='mlogloss',
            use_label_encoder=False, num_class=n_classes,
            objective='multi:softprob'
        )

    for model_name, model in mc_models.items():
        X_use = X_scaled if 'LR' in model_name else X_fs
        y_prob = cross_val_predict(model, X_use, y_fs, cv=cv, method='predict_proba')
        y_pred = y_prob.argmax(axis=1)
        y_bin = label_binarize(y_fs, classes=list(range(n_classes)))

        try:
            macro_auc = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
        except Exception:
            macro_auc = np.nan

        macro_f1 = f1_score(y_fs, y_pred, average='macro')
        accuracy = (y_pred == y_fs).mean()

        per_class_auc = {}
        for cls_idx, cls_name in enumerate(subtype_names):
            try:
                per_class_auc[cls_name] = roc_auc_score(
                    (y_fs == cls_idx).astype(int), y_prob[:, cls_idx])
            except Exception:
                per_class_auc[cls_name] = np.nan

        all_results_mc[(model_name, fs_name)] = {
            'y': y_fs, 'y_pred': y_pred, 'y_prob': y_prob,
            'feature_cols': fs_cols, 'per_class_auc': per_class_auc,
        }
        comparison_rows.append({
            'Model': model_name, 'Feature Set': fs_name,
            'Macro AUC': macro_auc, 'Macro F1': macro_f1, 'Accuracy': accuracy,
        })
        print(f"  {model_name}: AUC={macro_auc:.3f}, F1={macro_f1:.3f}, Acc={accuracy:.3f}")

comparison_df = pd.DataFrame(comparison_rows)
save_intermediate(comparison_df, '03_multiclass_results')

# Best config
best_config_mc = comparison_df.loc[comparison_df['Macro AUC'].idxmax()]
print(f"\nBest: {best_config_mc['Model']}, {best_config_mc['Feature Set']} "
      f"(AUC={best_config_mc['Macro AUC']:.3f})")


# ============================================================================
# 7. PER-SUBTYPE SHAP (BRIEF)
# ============================================================================
if has_shap:
    print("\n" + "=" * 70)
    print("PER-SUBTYPE FEATURE IMPORTANCE (SHAP)")
    print("=" * 70)

    best_key_mc = (best_config_mc['Model'], best_config_mc['Feature Set'])
    res = all_results_mc[best_key_mc]
    fs_cols_imp = res['feature_cols']
    X_imp = ml_subtype.dropna(subset=fs_cols_imp)[fs_cols_imp].values
    y_imp = ml_subtype.dropna(subset=fs_cols_imp)['y_subtype'].values.astype(int)

    feature_names_imp = []
    for c in fs_cols_imp:
        if c.startswith('expr_'):
            feature_names_imp.append(c.replace('expr_', ''))
        elif c.startswith('pathway_'):
            feature_names_imp.append(c.replace('pathway_', '').replace('_', ' '))
        else:
            feature_names_imp.append(c)

    if has_xgb:
        shap_tree = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric='mlogloss',
            use_label_encoder=False, objective='multi:softprob'
        )
    else:
        shap_tree = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, class_weight='balanced'
        )
    shap_tree.fit(X_imp, y_imp)

    explainer = shap.TreeExplainer(shap_tree)
    shap_values = explainer.shap_values(X_imp)

    if isinstance(shap_values, list):
        n_classes_shap = len(shap_values)
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        n_classes_shap = shap_values.shape[2]
        shap_values = [shap_values[:, :, i] for i in range(n_classes_shap)]
    else:
        n_classes_shap = 1
        shap_values = [shap_values]

    # Global importance
    global_shap = np.zeros(X_imp.shape[1])
    for class_shap in shap_values:
        global_shap += np.abs(class_shap).mean(axis=0)
    global_shap /= len(shap_values)

    sorted_idx_shap = np.argsort(global_shap)[::-1]
    print(f"\nTop 15 features (global mean |SHAP|):")
    for i in range(min(15, len(feature_names_imp))):
        idx = sorted_idx_shap[i]
        print(f"  {i+1:2d}. {feature_names_imp[idx]:20s}  {global_shap[idx]:.4f}")

    # GSEA-vs-SHAP cross-validation
    if subtype_gsea:
        print("\n" + "=" * 70)
        print("BIOLOGICAL CROSS-VALIDATION: GSEA vs SHAP")
        print("=" * 70)
        for cls_idx, s_name in enumerate(subtype_names):
            if s_name in subtype_gsea and cls_idx < len(shap_values):
                cls_shap_vals = np.abs(shap_values[cls_idx]).mean(axis=0)
                top_shap = [feature_names_imp[i] for i in np.argsort(cls_shap_vals)[::-1][:5]]
                gsea_res = subtype_gsea[s_name]
                gsea_top = gsea_res.sort_values('NES', ascending=False).head(3)['Term'].str.replace('HALLMARK_', '').tolist()
                gsea_bot = gsea_res.sort_values('NES', ascending=True).head(3)['Term'].str.replace('HALLMARK_', '').tolist()
                print(f"\n  {s_name}:")
                print(f"    SHAP top 5:    {top_shap}")
                print(f"    GSEA enriched: {gsea_top}")
                print(f"    GSEA depleted: {gsea_bot}")


# ============================================================================
# 8. SURVIVAL OVERLAY
# ============================================================================
print("\n" + "=" * 70)
print("SURVIVAL ANALYSIS BY SUBTYPE")
print("=" * 70)

surv_cols = ['Overall Survival (Months)', 'Overall Survival Status']
clin_surv = clinical[['pid'] + [c for c in surv_cols if c in clinical.columns]].copy()
surv_df = subtype_df.merge(clin_surv, on='pid', how='left')

if 'Overall Survival Status' in surv_df.columns:
    surv_df['event'] = surv_df['Overall Survival Status'].apply(
        lambda x: 1 if str(x).startswith('1:') else 0 if str(x).startswith('0:') else np.nan
    )
    surv_df['time'] = pd.to_numeric(surv_df['Overall Survival (Months)'], errors='coerce')
    surv_df = surv_df.dropna(subset=['event', 'time'])
    surv_df = surv_df[surv_df['time'] > 0]

    print(f"Patients with survival data: {len(surv_df)}")
    print(f"Events: {surv_df['event'].sum():.0f}")

    for s in subtype_names:
        sub = surv_df[surv_df['provisional_subtype'] == s]
        print(f"  {s}: n={len(sub)}, events={sub['event'].sum():.0f}")

    # Manual KM
    def kaplan_meier(time, event):
        df = pd.DataFrame({'time': time, 'event': event}).sort_values('time')
        times = sorted(df['time'].unique())
        n_at_risk = len(df)
        surv = 1.0
        km_times, km_surv = [0], [1.0]
        for t in times:
            d = ((df['time'] == t) & (df['event'] == 1)).sum()
            c = ((df['time'] == t) & (df['event'] == 0)).sum()
            if n_at_risk > 0 and d > 0:
                surv *= (1 - d / n_at_risk)
            km_times.append(t)
            km_surv.append(surv)
            n_at_risk -= (d + c)
        return np.array(km_times), np.array(km_surv)

    subtype_colors = {
        'Luminal A': '#3498db', 'Luminal B': '#2ecc71',
        'HER2-enriched': '#e74c3c', 'Basal-like': '#9b59b6',
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    for s_name in subtype_names:
        sub = surv_df[surv_df['provisional_subtype'] == s_name]
        if len(sub) >= 5:
            t, s = kaplan_meier(sub['time'].values, sub['event'].values)
            ax.step(t, s, where='post', linewidth=2,
                    color=subtype_colors.get(s_name, 'gray'),
                    label=f'{s_name} (n={len(sub)})')
    ax.set_xlabel('Months')
    ax.set_ylabel('Overall Survival')
    ax.set_title('Overall Survival by Subtype', fontsize=13, fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    savefig(fig, 'fig_03s_km_by_subtype')
    plt.close()


print("\n" + "=" * 70)
print("EXPLORATORY MULTI-CLASS ANALYSIS COMPLETE")
print("=" * 70)
print("Outputs saved to outputs/")
