"""
Phase 1 (re-run through shared pipeline): RSEM-UQ-TSS baseline.

Loads the existing TSS-corrected normalized matrix (01_tumor_norm, which is
UQ-normalized + log2(x+1) + TSS regression-corrected) and runs it through
the shared analysis pipeline, producing report.md and report_metrics.json
in reports/norm_comparison/rsem_uq_tss/ for use by run_comparison.py.

This replaces the standalone scripts/extract_rsem_report.py for the purpose
of Phase 4 comparison (identical pipeline, consistent JSON format).
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

    # Load the TSS-corrected normalized expression (this is the established pipeline output)
    print("\nLoading intermediates...")
    tumor_norm = load_intermediate('01_tumor_norm')
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
