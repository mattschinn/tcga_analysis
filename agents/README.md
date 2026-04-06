# Agentic Column Cleaner Pipeline

A LangGraph-based 7-agent pipeline for cleaning messy clinical data columns using LLMs.

## Architecture

```
┌──────────┐     ┌──────────┐
│ A1:Format │◄───►│ A2:Review │  (max 2 loops)
└────┬─────┘     └──────────┘
     ▼
┌──────────┐
│A3:Analyze │──── "no harmonization" ──► A7: Report
└────┬─────┘
     │ "needs harmonization"
     ▼
┌──────────┐     ┌──────────┐
│A4:Strategy│◄───►│A5:StratRev│  (max 2 loops)
└────┬─────┘     └──────────┘
     ▼
┌──────────┐     ┌──────────┐
│A6:Execute │◄───►│ A7:QC+Rpt │  (max 1 loop)
└──────────┘     └──────────┘
```

### Agent Roles

| Agent | Role | Input | Output |
|-------|------|-------|--------|
| **A1: Format Inspector** | Fix whitespace, casing, typos, number formats | Raw column + CONTEXT | Format mapping |
| **A2: Format Reviewer** | Verify only formatting changed (no semantic merges) | A1's output | Pass/fail + feedback |
| **A3: Semantic Analyzer** | Decide if harmonization is needed (THE FORK) | Formatted column + CONTEXT | Decision + messiness type |
| **A4: Strategy Architect** | Design harmonization approach (split, merge, binarize) | A3's analysis + CONTEXT | Concrete strategy with mapping rules |
| **A5: Strategy Reviewer** | Verify strategy completeness and correctness | A4's strategy | Pass/fail + feedback |
| **A6: Executor** | Apply the strategy to every value | Strategy + formatted values | Transformed columns |
| **A7: Final QC + Report** | Validate output and document everything | All prior outputs | QC result + markdown report |

## Setup

```bash
pip install langgraph langchain-anthropic langchain-core pandas
export ANTHROPIC_API_KEY='sk-ant-...'
```

## Usage

### Quick run on test columns
```bash
python run_pipeline.py
```

### Programmatic usage
```python
import pandas as pd
from context_generator import generate_context
from column_cleaner import clean_column

df = pd.read_csv("your_clinical_data.csv")
context = generate_context(
    analysis_plan_path="analysis_plan.md",
    qc_plan_path="qc_plan.md",
    clinical_csv_path="your_clinical_data.csv",
    column_name="ER positivity scale other",
)

result = clean_column(df["ER positivity scale other"], context=context)

# Inspect results
print(result.report)                    # Markdown report
print(result.columns.keys())            # Output column names
print(result.trace)                     # Full agent execution trace
print(result.strategy_used)             # Strategy description
```

### Dry-run tests (no API key needed)
```bash
python test_dry_run.py
```

## Files

| File | Purpose |
|------|---------|
| `column_cleaner.py` | Core pipeline: LangGraph StateGraph, 7 agent nodes, routing logic |
| `context_generator.py` | Auto-generates CONTEXT from project docs with column-specific guidance |
| `run_pipeline.py` | Runner for two test columns (ER positivity + HER2 method text) |
| `test_dry_run.py` | Mock-based tests validating graph logic without API calls |

## Key Design Decisions

1. **Single stream, no double-invoke**: State is accumulated from `stream()` updates, avoiding redundant LLM calls.
2. **Max iteration caps**: Format loop (2), strategy loop (2), final loop (1) — prevents infinite review cycles.
3. **Context-first**: Every agent receives the full CONTEXT document including project objectives, relevant biology, clinical workflows, and column-specific guidance.
4. **Measurement system detection**: The strategy architect can recognize when a column contains incommensurable scales (e.g., Allred scores + H-scores + fmol/mg) and split into separate columns rather than forcing harmonization.
5. **Column-specific guidance**: The context generator includes baked-in knowledge about each clinical column (value ranges, clinical thresholds, typical messiness patterns).

## Output Structure

`CleaningResult` contains:
- `original`: Input pandas Series
- `columns`: Dict of output pandas Series (may be 1 or many)
- `report`: Markdown documentation of the cleaning process
- `trace`: List of dicts with agent name, summary, decision, timing
- `strategy_used`: Description of the harmonization strategy
- `full_state`: Complete LangGraph state for debugging
