"""
Shared utilities for Tempus HER2 Coding Challenge.
Reusable functions for data loading, normalization, labeling, and visualization.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

CLINICAL_PATH = DATA_DIR / "brca_tcga_clinical_data.csv"
RSEM_PATH = DATA_DIR / "tcga.brca.rsem.csv"
CN_PATH = DATA_DIR / "brca_tcga_erbb2_copy_number.csv"

# TCGA sentinel missing values
TCGA_MISSING = [
    '[Not Available]', '[Not Evaluated]', '[Not Applicable]',
    '[Unknown]', '[Discrepancy]', '[Completed]', '[Not Assessed]',
    '', ' '
]


def ensure_dirs():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ── Data I/O ─────────────────────────────────────────────────────────────────

def save_intermediate(df, name, subdir=None):
    """Save a DataFrame as parquet to outputs/."""
    ensure_dirs()
    target = OUTPUT_DIR / f"{name}.parquet" if subdir is None else OUTPUT_DIR / subdir / f"{name}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    print(f"  Saved: {target.relative_to(PROJECT_ROOT)}  ({df.shape[0]} rows × {df.shape[1]} cols)")


def load_intermediate(name):
    """Load a parquet intermediate from outputs/."""
    path = OUTPUT_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Intermediate not found: {path}")
    df = pd.read_parquet(path)
    print(f"  Loaded: {path.relative_to(PROJECT_ROOT)}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    return df


def save_gene_cols(gene_cols, name="01_gene_cols"):
    """Save gene column list as JSON."""
    ensure_dirs()
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, 'w') as f:
        json.dump(gene_cols, f)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}  ({len(gene_cols)} genes)")


def load_gene_cols(name="01_gene_cols"):
    """Load gene column list from JSON."""
    path = OUTPUT_DIR / f"{name}.json"
    with open(path) as f:
        gene_cols = json.load(f)
    print(f"  Loaded: {path.relative_to(PROJECT_ROOT)}  ({len(gene_cols)} genes)")
    return gene_cols


def savefig(fig, name, dpi=150):
    """Save figure to outputs/figures/ as both PNG and PDF."""
    ensure_dirs()
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches='tight', facecolor='white')
    print(f"  Saved figure: {name}")


# ── Patient ID Harmonization ────────────────────────────────────────────────

def to_patient_id(id_str):
    """Extract 12-character TCGA patient ID from any barcode format.
    
    TCGA barcodes: TCGA-XX-XXXX-01A-11R-A089-07
    Patient ID:    TCGA-XX-XXXX (first 12 characters)
    """
    if pd.isna(id_str):
        return np.nan
    return str(id_str).strip()[:12]


def extract_tss(barcode):
    """Extract tissue source site from TCGA barcode (characters 5-6).
    
    Example: TCGA-3C-AAAU → '3C'
    """
    if pd.isna(barcode):
        return np.nan
    parts = str(barcode).split('-')
    if len(parts) >= 2:
        return parts[1]
    return np.nan


# ── Clinical Data Cleaning ───────────────────────────────────────────────────

def harmonize_clinical(clinical):
    """Standardize TCGA sentinel missing values and clean categorical fields.
    
    Returns the clinical DataFrame with:
    - Sentinel strings mapped to NaN
    - Categorical fields stripped and lowercased (selectively)
    - 'pid' column added (12-char patient ID)
    - 'tss' column added (tissue source site)
    - 'dx_year' parsed from Year Cancer Initial Diagnosis
    """
    df = clinical.copy()
    
    # Map sentinel strings to NaN
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(TCGA_MISSING, np.nan)
            df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x)
    
    # Add harmonized patient ID
    df['pid'] = df['Patient ID'].apply(to_patient_id)
    
    # Add tissue source site
    if 'Sample ID' in df.columns:
        df['tss'] = df['Sample ID'].apply(extract_tss)
    elif 'Patient ID' in df.columns:
        df['tss'] = df['Patient ID'].apply(extract_tss)
    
    # Parse diagnosis year
    if 'Year Cancer Initial Diagnosis' in df.columns:
        df['dx_year'] = pd.to_numeric(df['Year Cancer Initial Diagnosis'], errors='coerce')
    
    # Convert Cent17 Copy Number to numeric (it's sometimes stored as string)
    if 'Cent17 Copy Number' in df.columns:
        df['Cent17 Copy Number'] = pd.to_numeric(df['Cent17 Copy Number'], errors='coerce')
    
    return df


# ── HER2 Label Construction ─────────────────────────────────────────────────

def _clean_string(val):
    """Convert a value to a cleaned lowercase string, or None if missing."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().lower()
    if s in ('', 'nan', 'none', '[not available]', '[not evaluated]', '[not applicable]'):
        return None
    return s


def _parse_ihc_score(val):
    """Parse IHC score to a numeric value (0, 1, 2, or 3).
    
    Handles: 0, 1, 2, 3 (int/float), '0', '1+', '2+', '3+', '3.0', etc.
    Returns int or None.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        rounded = int(round(val))
        return rounded if rounded in (0, 1, 2, 3) else None
    s = str(val).strip().lower()
    if s in ('', 'nan', 'none', '[not available]', '[not evaluated]'):
        return None
    s = s.rstrip('+')
    try:
        rounded = int(round(float(s)))
        return rounded if rounded in (0, 1, 2, 3) else None
    except ValueError:
        return None


def construct_her2_label(row):
    """Derive composite HER2 status from IHC score, FISH status, and pre-coded IHC-HER2,
    following ASCO/CAP guidelines (simplified).
    
    Tiered logic:
      Tier 1: Derive from IHC score (+/- FISH)
      Tier 2: FISH only (no IHC score)
      Tier 3: Fallback to pre-coded IHC-HER2
      Tier 4: No data available
    
    Returns dict with:
        'label':  'Positive', 'Negative', 'Equivocal', or np.nan
        'source': which fields drove the call
        'flag':   description of any contradiction, or None
    """
    ihc_precoded = _clean_string(row.get('IHC-HER2'))
    ihc_score = _parse_ihc_score(row.get('HER2 ihc score'))
    fish = _clean_string(row.get('HER2 fish status'))
    flag = None

    # Tier 1: IHC score available
    if ihc_score is not None:
        if ihc_score == 3:
            label = 'Positive'
            source = 'IHC score 3+'
            if fish == 'negative':
                flag = f'IHC 3+ but FISH Negative (precoded: {ihc_precoded})'
        elif ihc_score == 2:
            if fish == 'positive':
                label, source = 'Positive', 'IHC score 2+ / FISH Positive'
            elif fish == 'negative':
                label, source = 'Negative', 'IHC score 2+ / FISH Negative'
            elif fish in ('equivocal', 'indeterminate'):
                label, source = 'Equivocal', f'IHC score 2+ / FISH {fish.title()}'
            else:
                label, source = 'Equivocal', 'IHC score 2+ / FISH unavailable'
        elif ihc_score <= 1:
            label = 'Negative'
            source = f'IHC score {int(ihc_score)}+'
            if fish == 'positive':
                flag = f'IHC {int(ihc_score)}+ but FISH Positive (precoded: {ihc_precoded})'
        else:
            label, source = np.nan, f'Unrecognized IHC score: {ihc_score}'

        if flag is None and ihc_precoded is not None and label is not np.nan:
            if ihc_precoded in ('positive', 'negative', 'equivocal') and ihc_precoded != label.lower():
                flag = (f'Derived {label} from score/FISH but precoded IHC-HER2 = '
                        f'{ihc_precoded.title()} (score={ihc_score}, FISH={fish})')
        return {'label': label, 'source': source, 'flag': flag}

    # Tier 2: FISH only
    if fish is not None:
        label_map = {'positive': 'Positive', 'negative': 'Negative',
                     'equivocal': 'Equivocal', 'indeterminate': 'Equivocal'}
        label = label_map.get(fish, np.nan)
        source = f'FISH {fish.title()} (no IHC score)'
        if ihc_precoded is not None and label is not np.nan:
            if ihc_precoded in ('positive', 'negative', 'equivocal') and ihc_precoded != label.lower():
                flag = f'FISH-only derived {label} but precoded IHC-HER2 = {ihc_precoded.title()}'
        return {'label': label, 'source': source, 'flag': flag}

    # Tier 3: Pre-coded IHC-HER2
    if ihc_precoded is not None:
        label_map = {'positive': 'Positive', 'negative': 'Negative',
                     'equivocal': 'Equivocal', 'indeterminate': 'Equivocal'}
        label = label_map.get(ihc_precoded, np.nan)
        return {'label': label, 'source': 'Pre-coded IHC-HER2 (no score or FISH)', 'flag': None}

    # Tier 4: Nothing
    return {'label': np.nan, 'source': 'No HER2 data', 'flag': None}


def apply_her2_labels(clinical_df):
    """Apply the composite HER2 label to a clinical dataframe.
    
    Adds columns: her2_composite, her2_source, her2_flag, label_confidence
    """
    df = clinical_df.copy()
    results = df.apply(construct_her2_label, axis=1, result_type='expand')
    df['her2_composite'] = results['label']
    df['her2_source'] = results['source']
    df['her2_flag'] = results['flag']
    
    # Add label confidence
    def _confidence(row):
        if pd.isna(row['her2_composite']):
            return np.nan
        if row['her2_flag'] is not None:
            return 'Low'
        if 'IHC score' in str(row['her2_source']) and 'FISH' in str(row['her2_source']):
            return 'High'
        if 'IHC score 3+' in str(row['her2_source']) or 'IHC score 0+' in str(row['her2_source']):
            return 'High'
        if 'IHC score 1+' in str(row['her2_source']):
            return 'High'
        if 'Pre-coded' in str(row['her2_source']):
            return 'Moderate'
        return 'Moderate'
    
    df['label_confidence'] = df.apply(_confidence, axis=1)
    
    # Reporting
    print("=" * 70)
    print("HER2 COMPOSITE LABEL CONSTRUCTION")
    print("=" * 70)
    
    source_counts = results['source'].value_counts()
    tier1 = source_counts[source_counts.index.str.contains('IHC score')].sum()
    tier2 = source_counts[source_counts.index.str.contains('FISH.*no IHC', regex=True)].sum()
    tier3 = source_counts[source_counts.index.str.contains('Pre-coded')].sum()
    tier4 = source_counts[source_counts.index.str.contains('No HER2')].sum()

    print(f"\nLabel sources:")
    print(f"  Tier 1 - Derived from IHC score (+/- FISH): {tier1}")
    print(f"  Tier 2 - Derived from FISH only:            {tier2}")
    print(f"  Tier 3 - Fallback to pre-coded IHC-HER2:    {tier3}")
    print(f"  Tier 4 - No HER2 data:                      {tier4}")
    print(f"  Total labeled:                               {tier1 + tier2 + tier3}")
    
    print(f"\nComposite HER2 label distribution:")
    print(df['her2_composite'].value_counts(dropna=False).to_string())
    
    flagged = results[results['flag'].notna()]
    print(f"\nContradictions flagged: {len(flagged)}")
    
    print(f"\nLabel confidence distribution:")
    print(df['label_confidence'].value_counts(dropna=False).to_string())
    
    return df


# ── HER2 Spectrum Classification ────────────────────────────────────────────

def classify_her2_spectrum(row):
    """Classify into HER2-0, HER2-low, HER2-positive spectrum.
    
    HER2-low (IHC 1+ or IHC 2+/FISH-) is clinically relevant for T-DXd eligibility.
    """
    ihc_score = _parse_ihc_score(row.get('HER2 ihc score'))
    fish = _clean_string(row.get('HER2 fish status'))
    composite = row.get('her2_composite', np.nan)
    
    if composite == 'Positive':
        return 'HER2-Positive'
    
    if ihc_score is not None:
        if ihc_score == 0:
            return 'HER2-0'
        elif ihc_score == 1:
            return 'HER2-Low'
        elif ihc_score == 2:
            if fish == 'negative':
                return 'HER2-Low'
            elif fish == 'positive':
                return 'HER2-Positive'
            else:
                return 'HER2-Low (presumed)'  # IHC 2+ without FISH → likely low
        elif ihc_score == 3:
            return 'HER2-Positive'
    
    return np.nan


# ── RNA-Seq Normalization ────────────────────────────────────────────────────

def upper_quartile_normalize(df, gene_cols):
    """Apply upper-quartile normalization followed by log2(x+1) transform.
    
    Steps:
    1. Compute 75th percentile of non-zero counts per sample.
    2. Divide all counts by that sample's 75th percentile.
    3. Multiply by a common scaling factor (median of all 75th percentiles).
    4. Apply log2(x + 1) transformation.
    
    Returns: (log_normalized DataFrame, size_factors Series, q75s Series)
    """
    gene_data = df[gene_cols].copy()
    
    def q75_nonzero(row):
        nonzero = row[row > 0]
        return np.percentile(nonzero, 75) if len(nonzero) > 0 else np.nan
    
    q75s = gene_data.apply(q75_nonzero, axis=1)
    median_q75 = q75s.median()
    size_factors = q75s / median_q75
    normalized = gene_data.div(size_factors, axis=0)
    log_normalized = np.log2(normalized + 1)
    
    return log_normalized, size_factors, q75s


def deseq2_size_factors(df, gene_cols):
    """Compute DESeq2-style median-of-ratios size factors.
    
    Steps:
    1. Compute geometric mean per gene across samples.
    2. Divide each sample's counts by these geometric means.
    3. Take the median of these ratios per sample = size factor.
    """
    gene_data = df[gene_cols].copy()
    gene_data_nz = gene_data.replace(0, np.nan)
    log_means = np.log(gene_data_nz).mean(axis=0)
    geo_means = np.exp(log_means)
    valid_genes = geo_means.dropna().index
    valid_genes = valid_genes[geo_means[valid_genes] > 0]
    
    if len(valid_genes) == 0:
        print("Warning: No genes with valid geometric means.")
        return pd.Series(np.nan, index=df.index)
    
    ratios = gene_data[valid_genes].div(geo_means[valid_genes], axis=1)
    sf = ratios.median(axis=1)
    return sf


def log2_normalize(df, gene_cols):
    """Simple log2(x+1) transform — for use when data is already depth-normalized.
    
    This is appropriate when RSEM expected counts show minimal library-size variation
    (CV < 10%), indicating upstream normalization.
    
    Returns: log2-transformed DataFrame (gene columns only)
    """
    return np.log2(df[gene_cols] + 1)


# ── Gene Filtering ───────────────────────────────────────────────────────────

# HER2 pathway genes that must survive filtering
HER2_PATHWAY_GENES = [
    'ERBB2', 'GRB7', 'ESR1', 'PGR', 'MKI67', 'EGFR', 'ERBB3',
    'TOP2A', 'PGAP3', 'STARD3', 'TCAP', 'PNMT', 'PPP1R1B',
    'PIK3CA', 'AKT1', 'CCND1', 'FOXA1'
]

def filter_genes(df, gene_cols, max_pct_zero=50, exempt_genes=None):
    """Filter genes by zero-expression frequency.
    
    Args:
        df: DataFrame with gene expression data
        gene_cols: list of gene column names
        max_pct_zero: maximum allowed % of zero-expression samples (default 50)
        exempt_genes: list of gene names to retain regardless of threshold
    
    Returns: (filtered gene_cols list, filtering stats dict)
    """
    if exempt_genes is None:
        exempt_genes = HER2_PATHWAY_GENES
    
    gene_data = df[gene_cols]
    pct_zero = (gene_data == 0).sum(axis=0) / len(gene_data) * 100
    
    # Genes passing the threshold
    genes_pass = pct_zero[pct_zero <= max_pct_zero].index.tolist()
    
    # Exempt genes that didn't pass
    exempt_rescued = [g for g in exempt_genes if g in gene_cols and g not in genes_pass]
    genes_keep = list(set(genes_pass + exempt_rescued))
    genes_keep = [g for g in gene_cols if g in genes_keep]  # preserve original order
    
    stats = {
        'total_before': len(gene_cols),
        'total_after': len(genes_keep),
        'dropped': len(gene_cols) - len(genes_keep),
        'threshold': max_pct_zero,
        'exempt_rescued': exempt_rescued,
    }
    
    print(f"Gene filtering (≤{max_pct_zero}% zeros):")
    print(f"  Before: {stats['total_before']}")
    print(f"  After:  {stats['total_after']}")
    print(f"  Dropped: {stats['dropped']}")
    if exempt_rescued:
        print(f"  Exempt genes rescued: {exempt_rescued}")
    
    return genes_keep, stats


# ── Plotting Helpers ─────────────────────────────────────────────────────────

# Consistent color palette
COLORS = {
    'Positive': '#e74c3c',
    'Negative': '#3498db',
    'Equivocal': '#f39c12',
    'HER2-Positive': '#e74c3c',
    'HER2-Low': '#f39c12',
    'HER2-0': '#3498db',
    'Unknown': '#95a5a6',
    np.nan: '#95a5a6',
}

def get_color(label):
    """Get consistent color for a label."""
    return COLORS.get(label, '#95a5a6')


def setup_plotting():
    """Set up consistent plotting defaults."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
    })
    
    import warnings
    warnings.filterwarnings('ignore')


# ── PCA Diagnostics ──────────────────────────────────────────────────────────

def pca_libsize_analysis(df, gene_cols, metadata_cols, color_by='her2_composite',
                         n_components=20, log_transform=False):
    """Run PCA and assess correlation with library size.
    
    Returns: (pca_components array, PCA object, pc_df DataFrame with PCs + metadata)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    X = df[gene_cols].fillna(0).values
    if log_transform:
        X = np.log2(X + 1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_comp = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(X_scaled)
    
    # Build PC dataframe
    pc_df = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(n_comp)], index=df.index)
    for col in metadata_cols:
        if col in df.columns:
            pc_df[col] = df[col].values
    
    # Library size correlation
    if 'library_size' in df.columns:
        log_lib = np.log10(df['library_size'].values)
        r, p = stats.pearsonr(pcs[:, 0], log_lib)
        print(f"PC1 vs log10(library_size): r = {r:.3f}, p = {p:.2e}")
    
    return pcs, pca, pc_df
