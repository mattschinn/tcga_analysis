"""
Phase 1 (re-run through shared pipeline): RSEM-UQ-TSS baseline.

Loads the UQ-normalized + TSS-corrected matrix (01_tumor_norm_uq_tss) and
runs it through the shared analysis pipeline, producing report.md and
report_metrics.json in reports/norm_comparison/rsem_uq_tss/ for use by
run_comparison.py.

NOTE: An earlier version incorrectly loaded 01_tumor_norm (UQ without TSS),
creating an apples-to-oranges comparison against TMM+TSS. Fixed 2026-04-07
to load 01_tumor_norm_uq_tss so both methods are compared with TSS applied.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_intermediate, load_gene_cols
from scripts.normalization_comparison.analysis_pipeline import run_analysis


def main():
    print("=" * 70)
    print("PHASE 1 (shared pipeline): RSEM-UQ-TSS BASELINE")
    print("=" * 70)

    # Load UQ + TSS-corrected expression (must match TMM+TSS for fair comparison)
    print("\nLoading intermediates...")
    tumor_norm = load_intermediate('01_tumor_norm_uq_tss')
    clinical = load_intermediate('01_clinical_qc')
    cn = load_intermediate('01_cn_qc')
    gene_cols = load_gene_cols()
    gene_cols = [g for g in gene_cols if g in tumor_norm.columns]

    print(f"  Tumor norm: {len(tumor_norm)} x {len(gene_cols)}")

    output_dir = PROJECT_ROOT / 'reports' / 'norm_comparison' / 'rsem_uq_tss'
    metrics = run_analysis(
        tumor_norm, clinical, cn, gene_cols,
        method_name='RSEM-UQ-TSS',
        output_dir=output_dir,
        gene_length_source='N/A (RSEM expected counts, UQ-normalized + log2(x+1) + TSS correction)'
    )

    print("\nPhase 1 (shared pipeline) complete.")
    return metrics


if __name__ == '__main__':
    main()
