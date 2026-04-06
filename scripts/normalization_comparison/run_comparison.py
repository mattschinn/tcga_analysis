"""
Phase 4: Generate comparison_summary.md from the three method reports.

Reads report_metrics.json from each method directory, builds a side-by-side
table, and produces a signal-vs-noise scorecard as specified in Phase 4 of
normalization_comparison_plan.md.

Usage:
    python scripts/normalization_comparison/run_comparison.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_ROOT = PROJECT_ROOT / 'reports' / 'norm_comparison'


def load_metrics(method_dir_name):
    path = REPORT_ROOT / method_dir_name / 'report_metrics.json'
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def rank_normalize(values):
    """Return normalized ranks in [0,1] for a list of floats (higher rank = higher value)."""
    import numpy as np
    arr = [v if v is not None and not (isinstance(v, float) and v != v) else float('nan')
           for v in values]
    valid = [(i, v) for i, v in enumerate(arr) if not (isinstance(v, float) and v != v)]
    n = len(valid)
    if n == 0:
        return [float('nan')] * len(values)
    sorted_valid = sorted(valid, key=lambda x: x[1])
    rank_map = {}
    for rank, (i, v) in enumerate(sorted_valid):
        rank_map[i] = rank / max(n - 1, 1)
    return [rank_map.get(i, float('nan')) for i in range(len(values))]


def compute_signal_noise(methods_data):
    """
    Signal composite = mean normalized rank for: A5, A7, B1, B7, C3, D4, D5.
    Noise composite  = mean normalized rank for: E1, E2, E3.
    Higher signal = better. Lower noise = better (so we invert noise ranks).

    Returns dict: method -> {'signal': float, 'noise': float}
    """
    import numpy as np

    signal_keys = ['A5', 'A7', 'B1', 'B7', 'C3', 'D4', 'D5']
    noise_keys = ['E1', 'E2', 'E3']  # lower = less technical confound

    method_names = list(methods_data.keys())

    def get_val(m, k):
        v = methods_data[m].get(k)
        if v is None:
            return float('nan')
        if isinstance(v, float) and v != v:
            return float('nan')
        # E2 is a p-value: higher p means less batch effect -- invert direction
        # We handle this separately in noise
        return float(v)

    scores = {m: {'signal': [], 'noise': []} for m in method_names}

    # Signal: higher is better
    for k in signal_keys:
        vals = [get_val(m, k) for m in method_names]
        ranks = rank_normalize(vals)
        for m, r in zip(method_names, ranks):
            scores[m]['signal'].append(r)

    # Noise: for E1, E3 lower is better (invert ranks)
    # For E2 (p-value), HIGHER is better (less TSS batch) so use raw rank
    for k in noise_keys:
        vals = [get_val(m, k) for m in method_names]
        ranks = rank_normalize(vals)
        if k == 'E2':
            # Higher p-value = less batch effect = better (keep as-is)
            for m, r in zip(method_names, ranks):
                scores[m]['noise'].append(r)
        else:
            # Lower value = better = invert
            for m, r in zip(method_names, ranks):
                scores[m]['noise'].append(1.0 - r if r == r else float('nan'))

    # Aggregate (skip nan)
    result = {}
    for m in method_names:
        s_vals = [v for v in scores[m]['signal'] if v == v]
        n_vals = [v for v in scores[m]['noise'] if v == v]
        result[m] = {
            'signal': float(np.mean(s_vals)) if s_vals else float('nan'),
            'noise': float(np.mean(n_vals)) if n_vals else float('nan'),
        }
    return result


def fmt(val, fmt_str='.4f'):
    if val is None:
        return 'N/A'
    if isinstance(val, float) and val != val:
        return 'N/A'
    try:
        return format(float(val), fmt_str)
    except (TypeError, ValueError):
        return str(val)


def main():
    print("=" * 70)
    print("PHASE 4: NORMALIZATION COMPARISON SUMMARY")
    print("=" * 70)

    # -- Load all three method metrics
    method_dirs = {
        'RSEM-UQ-TSS': 'rsem_uq_tss',
        'TPM-TSS': 'tpm',
        'TMM-edgeR': 'tmm_edger',
    }

    methods_data = {}
    for method_label, dir_name in method_dirs.items():
        try:
            methods_data[method_label] = load_metrics(dir_name)
            print(f"  Loaded: {method_label}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e} -- skipping {method_label}")

    if len(methods_data) < 2:
        print("ERROR: Need at least 2 method reports to compare.")
        sys.exit(1)

    # -- Compute signal-vs-noise scorecard
    sn_scores = compute_signal_noise(methods_data)

    # -- Identify best method
    best_signal = max(sn_scores, key=lambda m: sn_scores[m]['signal'])
    best_noise = max(sn_scores, key=lambda m: sn_scores[m]['noise'])

    # Best signal-to-noise: maximize (signal + noise) / 2 or use ratio
    best_sn = max(sn_scores, key=lambda m: (
        (sn_scores[m]['signal'] + sn_scores[m]['noise']) / 2
    ))

    method_list = list(methods_data.keys())
    col_header = ' | '.join(m for m in method_list)
    col_sep = ' | '.join(['---'] * len(method_list))

    def row(label, key, fmt_str='.4f'):
        vals = ' | '.join(fmt(methods_data[m].get(key), fmt_str) for m in method_list)
        return f"| {label} | {vals} |"

    def header_row():
        cols = ' | '.join(method_list)
        return f"| Metric | {cols} |"

    def sep_row():
        cols = ' | '.join(['---'] * len(method_list))
        return f"|--------|{cols}|"

    # Silhouette row: extract k=4 entry
    def sil_k_row(k=4):
        vals = []
        for m in method_list:
            c1 = methods_data[m].get('C1', {})
            v = c1.get(str(k)) or c1.get(k)
            vals.append(fmt(v, '.4f'))
        return f"| C1. Silhouette k={k} | {' | '.join(vals)} |"

    # -- Build comparison summary text
    lines = [
        "# Normalization Comparison Summary",
        "",
        "**Generated:** 2026-04-05",
        f"**Methods compared:** {', '.join(method_list)}",
        "",
        "---",
        "",
        "## Section A: ERBB2 RNA vs Copy Number",
        "",
        header_row(), sep_row(),
        row("A1. Pearson r (all)", "A1"),
        row("A2. Spearman rho (all)", "A2"),
        row("A3. Pearson r (HER2+)", "A3"),
        row("A4. Pearson r (HER2-)", "A4"),
        row("A5. Cohen's d", "A5"),
        row("A6. Mann-Whitney p", "A6", ".2e"),
        row("A7. Fold-change (median)", "A7"),
        "",
        "---",
        "",
        "## Section B: Logistic Regression (RNA/CN -> HER2 IHC)",
        "",
        header_row(), sep_row(),
        row("B1. AUC-ROC RNA only", "B1"),
        row("B2. AUC-PR RNA only", "B2"),
        row("B3. AUC-ROC CN only", "B3"),
        row("B4. AUC-PR CN only", "B4"),
        row("B5. AUC-ROC RNA+CN", "B5"),
        row("B6. AUC-PR RNA+CN", "B6"),
        row("B7. Delta AUC-ROC (RNA-CN)", "B7"),
        "",
        "---",
        "",
        "## Section C: Unsupervised Clustering",
        "",
        header_row(), sep_row(),
    ]
    # Silhouette for each k
    for k in range(2, 8):
        lines.append(sil_k_row(k))
    lines += [
        row("C2. Best k", "C2", '.0f'),
        row("C3. Silhouette at k=4", "C3"),
        row("C4. ARI (k=4 vs HER2)", "C4"),
        row("C5. ARI (k=4 vs ER)", "C5"),
        "",
        "---",
        "",
        "## Section D: Subtype Marker Separation",
        "",
        header_row(), sep_row(),
        row("D1. Luminal score spread", "D1"),
        row("D2. HER2 score spread", "D2"),
        row("D3. Basal score spread", "D3"),
        row("D4. Mean subtype-score gap", "D4"),
        row("D5. Frac HER2+ in HER2-enriched", "D5"),
        "",
        "---",
        "",
        "## Section E: Normalization Diagnostics",
        "",
        header_row(), sep_row(),
        row("E1. CV of median expr across TSS", "E1"),
        row("E2. PC1 vs TSS (Kruskal-Wallis p)", "E2", ".2e"),
        row("E3. PC1 vs read-depth (r)", "E3"),
        row("E4. ERBB2 CV within HER2+", "E4"),
        "",
        "---",
        "",
        "## Signal vs Noise Scorecard",
        "",
        "Normalized rank composites (0..1 scale; higher is better for both).",
        "",
        "**Signal** = mean rank across A5, A7, B1, B7, C3, D4, D5  (higher = stronger HER2 biology)",
        "**Noise**  = mean inverted rank across E1, E2, E3          (higher = cleaner technical control)",
        "",
        "| Composite | " + " | ".join(method_list) + " |",
        "|-----------|" + "|".join(["---"] * len(method_list)) + "|",
        "| Signal composite | " + " | ".join(
            f"{sn_scores[m]['signal']:.3f}" for m in method_list
        ) + " |",
        "| Noise composite  | " + " | ".join(
            f"{sn_scores[m]['noise']:.3f}" for m in method_list
        ) + " |",
        "| Mean (signal+noise)/2 | " + " | ".join(
            f"{(sn_scores[m]['signal'] + sn_scores[m]['noise']) / 2:.3f}"
            for m in method_list
        ) + " |",
        "",
        f"**Best signal:** {best_signal}",
        f"**Best noise control:** {best_noise}",
        f"**Best overall (signal+noise balance):** {best_sn}",
        "",
        "---",
        "",
        "## Interpretive Notes",
        "",
    ]

    # Interpretive analysis
    rsem = methods_data.get('RSEM-UQ-TSS', {})
    tpm = methods_data.get('TPM-TSS', {})
    tmm = methods_data.get('TMM-edgeR', {})

    def delta(alt_dict, key):
        base = rsem.get(key)
        alt = alt_dict.get(key) if alt_dict else None
        if base is None or alt is None:
            return float('nan')
        return float(alt) - float(base)

    notes = []

    # Check if alternative methods weaken HER2 signal (Section A-D drops)
    for method_label, m_data in [('TPM', tpm), ('TMM', tmm)]:
        if not m_data:
            continue
        d_a5 = delta(m_data, 'A5')
        d_b1 = delta(m_data, 'B1')
        d_d4 = delta(m_data, 'D4')

        if (d_a5 != d_a5 or d_b1 != d_b1):
            continue

        if d_a5 < -0.1 or d_b1 < -0.02:
            notes.append(
                f"- **{method_label} weakens HER2 signal vs RSEM:** "
                f"Cohen's d drops by {abs(d_a5):.2f} (delta_A5={d_a5:+.2f}), "
                f"AUC-ROC RNA drops by {abs(d_b1):.3f} (delta_B1={d_b1:+.3f}). "
                "This confirms that more aggressive read-depth normalization "
                "compresses real HER2 biology."
            )
        elif d_a5 > 0.05 or d_b1 > 0.01:
            notes.append(
                f"- **{method_label} strengthens HER2 signal vs RSEM:** "
                f"Cohen's d increases by {d_a5:+.2f}, AUC-ROC RNA by {d_b1:+.3f}. "
                "Review whether this reflects genuine signal recovery or artifact."
            )
        else:
            notes.append(
                f"- **{method_label} produces similar HER2 signal to RSEM:** "
                f"delta_A5={d_a5:+.2f}, delta_B1={d_b1:+.3f}. "
                "Normalization choice does not substantially affect signal for this metric."
            )

        d_e3 = delta(m_data, 'E3')
        d_e2 = delta(m_data, 'E2')
        if d_e3 != d_e3:
            continue
        if d_e3 < -0.05:
            notes.append(
                f"  - {method_label} reduces read-depth confound in PC1 "
                f"(E3 drops by {abs(d_e3):.3f})."
            )
        elif d_e3 > 0.05:
            notes.append(
                f"  - {method_label} INCREASES read-depth confound in PC1 "
                f"(E3 rises by {abs(d_e3):.3f}). Unexpected -- review normalization."
            )

    if not notes:
        notes.append(
            "- All methods produce similar signal-vs-noise profiles. "
            "Normalization choice is not a critical decision point for this dataset."
        )

    notes.append("")
    notes.append(
        "**Recommendation:** Use the method with the highest (signal+noise)/2 composite "
        "score above, confirming with the absolute AUC-ROC RNA (B1) as the primary clinical "
        "relevance metric and E3 (PC1 read-depth confound) as the primary technical risk metric."
    )

    lines += notes
    lines += ["", "---", "",
              "*Report generated by scripts/normalization_comparison/run_comparison.py -- Phase 4.*"]

    summary_text = '\n'.join(lines)

    out_path = REPORT_ROOT / 'comparison_summary.md'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f"\nComparison summary written to: {out_path}")
    print("\nSignal-vs-noise scorecard:")
    for m in method_list:
        s = sn_scores[m]
        print(f"  {m}: signal={s['signal']:.3f}, noise={s['noise']:.3f}, "
              f"mean={(s['signal']+s['noise'])/2:.3f}")
    print(f"\nBest overall: {best_sn}")


if __name__ == '__main__':
    main()
