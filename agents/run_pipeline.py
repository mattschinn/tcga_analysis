"""
Run the column cleaning pipeline on two test columns:
1. ER positivity scale other (hard — multiple measurement systems)
2. HER2 positivity method text (simpler — spelling variant dedup)
"""

import sys
import json
import pandas as pd

from context_generator import generate_context
from column_cleaner import clean_column

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MESSY_CSV = "/mnt/user-data/uploads/example_messy_data_260402.csv"
CLINICAL_CSV = "/mnt/project/_abbridged_brca_tcga_clinical_data.csv"
ANALYSIS_PLAN = "/mnt/project/analysis_plan.md"
QC_PLAN = "/mnt/project/qc_plan.md"

# Columns to test
TEST_COLUMNS = [
    "HER2 positivity method text",   # simpler one first (warmup)
    "ER positivity scale other",      # hard one (multiple measurement systems)
]


def main():
    # Load messy data
    print("Loading data...")
    df = pd.read_csv(MESSY_CSV, index_col=0)
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns\n")

    results = {}

    for col_name in TEST_COLUMNS:
        # Generate context specific to this column
        print(f"Generating context for '{col_name}'...")
        context = generate_context(
            analysis_plan_path=ANALYSIS_PLAN,
            qc_plan_path=QC_PLAN,
            clinical_csv_path=CLINICAL_CSV,
            column_name=col_name,
        )
        print(f"  Context: {len(context)} chars\n")

        # Run the pipeline
        series = df[col_name]
        result = clean_column(series, context=context, verbose=True)
        results[col_name] = result

        # Print the report
        print("\n" + "="*70)
        print(f"REPORT: {col_name}")
        print("="*70)
        print(result.report)
        print()

        # Show output columns
        for out_name, out_series in result.columns.items():
            non_null = out_series.notna().sum()
            unique = out_series.dropna().nunique()
            print(f"  Output column '{out_name}': {non_null} non-null, {unique} unique values")
            if unique <= 20:
                vc = out_series.dropna().value_counts()
                for val, count in vc.items():
                    print(f"    {val}: {count}")
        print()

        # Print trace
        print("Agent Trace:")
        for step in result.trace:
            print(f"  [{step['elapsed_sec']:5.1f}s] {step['agent']}: {step['summary']} → {step['decision']}")
        print()

    # Save results
    print("\nSaving outputs...")
    for col_name, result in results.items():
        # Save report
        safe_name = col_name.replace(" ", "_").lower()
        report_path = f"/home/claude/report_{safe_name}.md"
        with open(report_path, "w") as f:
            f.write(result.report)
        print(f"  Report → {report_path}")

        # Save output columns as CSV
        out_df = pd.DataFrame(result.columns)
        csv_path = f"/home/claude/cleaned_{safe_name}.csv"
        out_df.to_csv(csv_path)
        print(f"  Cleaned data → {csv_path}")

        # Save trace as JSON
        trace_path = f"/home/claude/trace_{safe_name}.json"
        with open(trace_path, "w") as f:
            json.dump(result.trace, f, indent=2)
        print(f"  Trace → {trace_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
