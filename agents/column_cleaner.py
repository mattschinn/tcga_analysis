"""
Agentic Column Cleaner — LangGraph Implementation
==================================================

A 7-agent pipeline for cleaning messy clinical data columns using LLMs,
orchestrated as a LangGraph StateGraph with conditional edges and review loops.

Graph topology:
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
"""

from __future__ import annotations

import json
import time
import textwrap
from dataclasses import dataclass, field
from typing import Any, TypedDict

import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
MAX_FORMAT_LOOPS = 2
MAX_STRATEGY_LOOPS = 2
MAX_FINAL_LOOPS = 1
MAX_UNIQUE_VALUES_DISPLAY = 200

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


class PipelineState(TypedDict):
    """Full state passed through the LangGraph pipeline."""
    # Inputs (set once)
    column_name: str
    original_values: list
    original_index: list
    context: str

    # A1/A2: Format stage
    formatted_values: list
    a1_result: dict
    a1_feedback: str
    format_loop_count: int

    # A3: Semantic analysis
    a3_result: dict
    needs_harmonization: bool

    # A4/A5: Strategy stage
    a4_result: dict
    a4_feedback: str
    strategy_loop_count: int

    # A6/A7: Execution stage
    executed_columns: dict   # {col_name: {index: value, ...}, ...}
    a7_feedback: str
    final_loop_count: int

    # Output
    report: str
    trace: list

    # Control flow
    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_values(name: str, values: list, index: list) -> str:
    """Summarize a column for LLM consumption."""
    series = pd.Series(values, index=index, name=name)
    total = len(series)
    non_null = series.notna().sum()
    unique_vals = series.dropna().unique()
    n_unique = len(unique_vals)

    value_counts = series.dropna().value_counts()
    if n_unique <= MAX_UNIQUE_VALUES_DISPLAY:
        val_lines = [f"  [{val}] (n={count})" for val, count in value_counts.items()]
    else:
        top = value_counts.head(100)
        val_lines = [f"  [{val}] (n={count})" for val, count in top.items()]
        val_lines.append(f"  ... and {n_unique - 100} more unique values")

    return (
        f"Column name: {name}\n"
        f"Total rows: {total}\n"
        f"Non-null values: {non_null} ({100*non_null/total:.1f}%)\n"
        f"Null/missing: {total - non_null} ({100*(total-non_null)/total:.1f}%)\n"
        f"Unique non-null values: {n_unique}\n\n"
        f"All values (with counts):\n" + "\n".join(val_lines)
    )


def _call_llm(system: str, user: str, llm: ChatAnthropic) -> str:
    """Single LLM call, returns text."""
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    response = llm.invoke(messages)
    return response.content


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return json.loads(cleaned)


def _add_trace(state: PipelineState, agent: str, summary: str, decision: str, elapsed: float):
    """Return a new trace list with the entry appended."""
    entry = {
        "agent": agent,
        "summary": summary,
        "decision": decision,
        "elapsed_sec": round(elapsed, 2),
    }
    return state["trace"] + [entry]


def _apply_format_mapping(values: list, mapping: dict) -> list:
    """Apply value_mapping to a list of values."""
    result = []
    for v in values:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            result.append(v)
        else:
            str_v = str(v).strip()
            result.append(mapping.get(str_v, v))
    return result


def _get_llm():
    return ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)


# ---------------------------------------------------------------------------
# Agent node functions
# ---------------------------------------------------------------------------

def node_a1_format_inspector(state: PipelineState) -> dict:
    """A1: Proposes and applies format fixes."""
    t0 = time.time()
    llm = _get_llm()
    col_summary = _summarize_values(state["column_name"], state["original_values"], state["original_index"])

    feedback_block = ""
    if state.get("a1_feedback"):
        feedback_block = f"\nPrevious attempt REJECTED:\n{state['a1_feedback']}\nAddress the concerns.\n"

    system = textwrap.dedent("""\
        You are a data format inspector for clinical/biomedical datasets.
        Fix ONLY formatting issues:
        - Strip whitespace
        - Normalize casing of identical values ("STRONG" / "Strong" → one form)
        - Fix clear typos in otherwise identical values ("biospy" → "biopsy")
        - Standardize number formats ("9.2 " → "9.2")
        - Normalize missing-value sentinels to null

        Do NOT merge semantically different values. "Mastectomy" ≠ "Total Mastectomy".

        Respond ONLY with a JSON object (no markdown fences, no preamble):
        {
            "format_issues_found": ["list of issues"],
            "transformations_applied": ["list of transformations"],
            "value_mapping": {"original_value": "cleaned_value", ...},
            "notes": "observations"
        }
        Include ONLY values that actually change in value_mapping.
    """)

    raw = _call_llm(system, f"CONTEXT:\n{state['context']}\n\nCOLUMN:\n{col_summary}\n{feedback_block}", llm)
    elapsed = time.time() - t0

    try:
        result = _parse_json(raw)
    except json.JSONDecodeError:
        result = {"format_issues_found": ["parse failure"], "transformations_applied": [],
                  "value_mapping": {}, "notes": raw[:500]}

    formatted = _apply_format_mapping(state["original_values"], result.get("value_mapping", {}))

    return {
        "a1_result": result,
        "formatted_values": formatted,
        "trace": _add_trace(state, "A1: Format Inspector",
                            f"{len(result.get('value_mapping', {}))} values mapped", "proposed", elapsed),
        "status": "a1_done",
    }


def node_a2_format_reviewer(state: PipelineState) -> dict:
    """A2: Reviews A1 output for correctness."""
    t0 = time.time()
    llm = _get_llm()
    col_summary = _summarize_values(state["column_name"], state["original_values"], state["original_index"])

    system = textwrap.dedent("""\
        You are a QC reviewer for data formatting on clinical/biomedical datasets.
        Check that ONLY formatting changes were made (no semantic merges).

        Respond ONLY with JSON (no fences):
        {"passed": true/false, "issues": ["problems"], "feedback": "details if failed"}
    """)

    raw = _call_llm(system,
        f"CONTEXT:\n{state['context']}\n\nORIGINAL:\n{col_summary}\n\n"
        f"TRANSFORMS:\n{json.dumps(state['a1_result'], indent=2)}", llm)
    elapsed = time.time() - t0

    try:
        result = _parse_json(raw)
    except json.JSONDecodeError:
        result = {"passed": True, "issues": [], "feedback": ""}

    passed = result.get("passed", True)
    loop = state["format_loop_count"] + 1

    if passed or loop > MAX_FORMAT_LOOPS:
        status = "format_accepted"
        decision = "pass" if passed else "max_loops"
    else:
        status = "format_rejected"
        decision = "fail"

    return {
        "a1_feedback": result.get("feedback", ""),
        "format_loop_count": loop,
        "trace": _add_trace(state, "A2: Format Reviewer",
                            f"{'PASS' if passed else 'FAIL'} (loop {loop})", decision, elapsed),
        "status": status,
    }


def node_a3_semantic_analyzer(state: PipelineState) -> dict:
    """A3: Decides if harmonization is needed (the fork)."""
    t0 = time.time()
    llm = _get_llm()
    col_summary = _summarize_values(state["column_name"], state["formatted_values"], state["original_index"])

    system = textwrap.dedent("""\
        You are a clinical data scientist. Formatting is done. Determine whether
        the column needs SEMANTIC harmonization:
        - Merging values that mean the same thing
        - Splitting columns with multiple measurement systems
        - Categorizing free-text into controlled vocabulary

        Use the CONTEXT deeply — understand what the column is for, what downstream
        analyses need, and what clinical distinctions matter.

        Respond ONLY with JSON (no fences):
        {
            "needs_harmonization": true/false,
            "rationale": "why",
            "messiness_type": "casing_variants|measurement_system_mix|free_text_categorization|minor_dedup|none",
            "estimated_complexity": "low|medium|high",
            "recommended_approach": "brief description",
            "output_columns_expected": ["column_names"]
        }
    """)

    raw = _call_llm(system, f"CONTEXT:\n{state['context']}\n\nFORMATTED COLUMN:\n{col_summary}", llm)
    elapsed = time.time() - t0

    try:
        result = _parse_json(raw)
    except json.JSONDecodeError:
        result = {"needs_harmonization": False, "rationale": "parse failure",
                  "messiness_type": "none", "estimated_complexity": "low",
                  "recommended_approach": "", "output_columns_expected": [state["column_name"]]}

    needs = result.get("needs_harmonization", False)

    return {
        "a3_result": result,
        "needs_harmonization": needs,
        "trace": _add_trace(state, "A3: Semantic Analyzer",
                            f"{'HARMONIZE' if needs else 'SKIP'} ({result.get('messiness_type', '?')})",
                            "needs_harmonization" if needs else "no_harmonization", elapsed),
        "status": "needs_harmonization" if needs else "skip_to_report",
    }


def node_a4_strategy_architect(state: PipelineState) -> dict:
    """A4: Designs harmonization strategy."""
    t0 = time.time()
    llm = _get_llm()
    col_summary = _summarize_values(state["column_name"], state["formatted_values"], state["original_index"])

    feedback_block = ""
    if state.get("a4_feedback"):
        feedback_block = f"\nPrevious strategy REJECTED:\n{state['a4_feedback']}\nRevise.\n"

    system = textwrap.dedent("""\
        You are a clinical data harmonization strategist. Design a CONCRETE,
        EXECUTABLE strategy.

        For each output column specify:
        1. Column name  2. Data type  3. Allowed values/range
        4. EXACT mapping rules (input → output)  5. Null handling

        Respond ONLY with JSON (no fences):
        {
            "strategy_summary": "one paragraph",
            "output_columns": [
                {
                    "name": "col_name",
                    "dtype": "categorical|numeric|boolean",
                    "allowed_values": ["list"] or "range",
                    "description": "what it represents",
                    "mapping_rules": [
                        {"input_pattern": "pattern", "output_value": "result",
                         "rule_type": "exact|contains|regex|numeric_extract|default"}
                    ],
                    "null_handling": "description"
                }
            ],
            "rationale": "why this serves downstream",
            "caveats": ["limitations"]
        }
    """)

    raw = _call_llm(system,
        f"CONTEXT:\n{state['context']}\n\nCOLUMN:\n{col_summary}\n\n"
        f"SEMANTIC ANALYSIS:\n{json.dumps(state['a3_result'], indent=2)}\n{feedback_block}", llm)
    elapsed = time.time() - t0

    try:
        result = _parse_json(raw)
    except json.JSONDecodeError:
        result = {"strategy_summary": "parse failure", "output_columns": [],
                  "rationale": raw[:500], "caveats": ["parse failure"]}

    return {
        "a4_result": result,
        "trace": _add_trace(state, "A4: Strategy Architect",
                            f"{len(result.get('output_columns', []))} output cols",
                            "proposed", elapsed),
        "status": "a4_done",
    }


def node_a5_strategy_reviewer(state: PipelineState) -> dict:
    """A5: Reviews strategy for completeness and correctness."""
    t0 = time.time()
    llm = _get_llm()
    col_summary = _summarize_values(state["column_name"], state["formatted_values"], state["original_index"])

    system = textwrap.dedent("""\
        You are a senior clinical data scientist reviewing a harmonization strategy.
        Check: COMPLETENESS (all values covered?), CORRECTNESS (clinically right?),
        GRANULARITY (appropriate?), EXECUTABILITY (clear rules?), CONTEXT FIT.

        Respond ONLY with JSON (no fences):
        {"passed": true/false, "issues": ["problems"],
         "missing_values": ["uncovered values"], "feedback": "details if failed"}
    """)

    raw = _call_llm(system,
        f"CONTEXT:\n{state['context']}\n\nVALUES:\n{col_summary}\n\n"
        f"STRATEGY:\n{json.dumps(state['a4_result'], indent=2)}", llm)
    elapsed = time.time() - t0

    try:
        result = _parse_json(raw)
    except json.JSONDecodeError:
        result = {"passed": True, "issues": [], "missing_values": [], "feedback": ""}

    passed = result.get("passed", True)
    loop = state["strategy_loop_count"] + 1

    if passed or loop > MAX_STRATEGY_LOOPS:
        status = "strategy_accepted"
        decision = "pass" if passed else "max_loops"
    else:
        status = "strategy_rejected"
        decision = "fail"

    return {
        "a4_feedback": result.get("feedback", ""),
        "strategy_loop_count": loop,
        "trace": _add_trace(state, "A5: Strategy Reviewer",
                            f"{'PASS' if passed else 'FAIL'} (loop {loop})", decision, elapsed),
        "status": status,
    }


def node_a6_executor(state: PipelineState) -> dict:
    """A6: Implements the harmonization strategy."""
    t0 = time.time()
    llm = _get_llm()

    values_with_index = []
    for idx, val in zip(state["original_index"], state["formatted_values"]):
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            values_with_index.append({"index": idx, "value": str(val)})

    system = textwrap.dedent("""\
        You are a data transformation executor. Apply the strategy's mapping rules
        to every input value.

        Respond ONLY with JSON (no fences):
        {
            "columns": {
                "column_name": {
                    "values": [{"index": 0, "value": "mapped_or_null"}, ...]
                }
            },
            "unmapped_values": ["unmatched values"],
            "execution_notes": "edge cases"
        }
        Use JSON null for NaN. Process ALL values.
    """)

    raw = _call_llm(system,
        f"STRATEGY:\n{json.dumps(state['a4_result'], indent=2)}\n\n"
        f"INPUT ({len(values_with_index)} values):\n{json.dumps(values_with_index)}", llm)
    elapsed = time.time() - t0

    try:
        result = _parse_json(raw)
    except json.JSONDecodeError:
        result = {"columns": {}, "unmapped_values": [], "execution_notes": f"parse fail: {raw[:300]}"}

    columns = {}
    for col_name, col_data in result.get("columns", {}).items():
        mapping = {}
        for entry in col_data.get("values", []):
            idx = entry.get("index")
            val = entry.get("value")
            if idx is not None:
                mapping[idx] = val
        columns[col_name] = mapping

    return {
        "executed_columns": columns,
        "trace": _add_trace(state, "A6: Executor",
                            f"{len(columns)} cols produced, {len(result.get('unmapped_values', []))} unmapped",
                            "executed", elapsed),
        "status": "a6_done",
    }


def node_a7_final_qc_report(state: PipelineState) -> dict:
    """A7: Final QC + Report."""
    t0 = time.time()
    llm = _get_llm()

    col_summaries = {}
    if state.get("needs_harmonization") and state.get("executed_columns"):
        for col_name, idx_val_map in state["executed_columns"].items():
            vals = [v for v in idx_val_map.values() if v is not None]
            col_summaries[col_name] = {
                "non_null": len(vals),
                "unique_values": list(set(str(v) for v in vals))[:50],
                "n_unique": len(set(str(v) for v in vals)),
            }
    else:
        fvals = [v for v in state["formatted_values"]
                 if v is not None and not (isinstance(v, float) and pd.isna(v))]
        col_summaries[state["column_name"]] = {
            "non_null": len(fvals),
            "unique_values": list(set(str(v) for v in fvals))[:50],
            "n_unique": len(set(str(v) for v in fvals)),
        }

    orig_non_null = sum(1 for v in state["original_values"]
                        if v is not None and not (isinstance(v, float) and pd.isna(v)))

    system = textwrap.dedent("""\
        You are a senior clinical data scientist doing final QC and documentation.

        QC: All expected columns present? Non-null counts reasonable? Values match
        strategy? No suspicious patterns?

        REPORT (markdown): Column purpose, input characteristics, strategy + justification,
        output columns, confidence level, caveats, downstream recommendations.

        Respond ONLY with JSON (no fences):
        {"qc_passed": true/false, "qc_issues": ["issues"],
         "feedback": "if failed", "report": "full markdown report"}
    """)

    raw = _call_llm(system,
        f"CONTEXT:\n{state['context']}\n\n"
        f"ORIGINAL: {state['column_name']}, {orig_non_null} non-null\n\n"
        f"SEMANTIC ANALYSIS:\n{json.dumps(state.get('a3_result', {}), indent=2)}\n\n"
        f"STRATEGY:\n{json.dumps(state.get('a4_result', {}), indent=2)}\n\n"
        f"OUTPUT COLUMNS:\n{json.dumps(col_summaries, indent=2, default=str)}", llm)
    elapsed = time.time() - t0

    try:
        result = _parse_json(raw)
    except json.JSONDecodeError:
        result = {"qc_passed": True, "qc_issues": [], "feedback": "", "report": raw}

    passed = result.get("qc_passed", True)
    loop = state["final_loop_count"] + 1

    if passed or loop > MAX_FINAL_LOOPS:
        status = "complete"
        decision = "pass" if passed else "max_loops"
    else:
        status = "final_rejected"
        decision = "fail"

    return {
        "a7_feedback": result.get("feedback", ""),
        "final_loop_count": loop,
        "report": result.get("report", ""),
        "trace": _add_trace(state, "A7: Final QC + Report",
                            f"QC {'PASS' if passed else 'FAIL'} (loop {loop})", decision, elapsed),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_a2(state: PipelineState) -> str:
    return "a1_format_inspector" if state["status"] == "format_rejected" else "a3_semantic_analyzer"


def route_after_a3(state: PipelineState) -> str:
    return "a7_final_qc_report" if state["status"] == "skip_to_report" else "a4_strategy_architect"


def route_after_a5(state: PipelineState) -> str:
    return "a4_strategy_architect" if state["status"] == "strategy_rejected" else "a6_executor"


def route_after_a7(state: PipelineState) -> str:
    return "a6_executor" if state["status"] == "final_rejected" else END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    graph.add_node("a1_format_inspector", node_a1_format_inspector)
    graph.add_node("a2_format_reviewer", node_a2_format_reviewer)
    graph.add_node("a3_semantic_analyzer", node_a3_semantic_analyzer)
    graph.add_node("a4_strategy_architect", node_a4_strategy_architect)
    graph.add_node("a5_strategy_reviewer", node_a5_strategy_reviewer)
    graph.add_node("a6_executor", node_a6_executor)
    graph.add_node("a7_final_qc_report", node_a7_final_qc_report)

    graph.set_entry_point("a1_format_inspector")

    graph.add_edge("a1_format_inspector", "a2_format_reviewer")
    graph.add_conditional_edges("a2_format_reviewer", route_after_a2,
                                {"a1_format_inspector": "a1_format_inspector",
                                 "a3_semantic_analyzer": "a3_semantic_analyzer"})
    graph.add_conditional_edges("a3_semantic_analyzer", route_after_a3,
                                {"a7_final_qc_report": "a7_final_qc_report",
                                 "a4_strategy_architect": "a4_strategy_architect"})
    graph.add_edge("a4_strategy_architect", "a5_strategy_reviewer")
    graph.add_conditional_edges("a5_strategy_reviewer", route_after_a5,
                                {"a4_strategy_architect": "a4_strategy_architect",
                                 "a6_executor": "a6_executor"})
    graph.add_edge("a6_executor", "a7_final_qc_report")
    graph.add_conditional_edges("a7_final_qc_report", route_after_a7,
                                {"a6_executor": "a6_executor",
                                 END: END})

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class CleaningResult:
    """Final output of the pipeline."""
    original: pd.Series
    columns: dict[str, pd.Series]
    report: str
    trace: list[dict]
    strategy_used: str
    full_state: dict


def clean_column(
    series: pd.Series,
    context: str,
    verbose: bool = True,
) -> CleaningResult:
    """
    Run the full 7-agent cleaning pipeline on a single column.

    Args:
        series: The pandas Series to clean.
        context: Markdown string with project context.
        verbose: Print progress.

    Returns:
        CleaningResult with output columns, report, and trace.
    """
    values = series.tolist()
    index = series.index.tolist()

    initial_state: PipelineState = {
        "column_name": str(series.name),
        "original_values": values,
        "original_index": index,
        "context": context,
        "formatted_values": [],
        "a1_result": {},
        "a1_feedback": "",
        "format_loop_count": 0,
        "a3_result": {},
        "needs_harmonization": False,
        "a4_result": {},
        "a4_feedback": "",
        "strategy_loop_count": 0,
        "executed_columns": {},
        "a7_feedback": "",
        "final_loop_count": 0,
        "report": "",
        "trace": [],
        "status": "init",
    }

    app = build_graph()

    if verbose:
        print(f"\n{'='*70}")
        print(f"CLEANING COLUMN: {series.name}")
        print(f"{'='*70}")

    # Stream and accumulate state from updates (single run, no double-invoke)
    accumulated_state = dict(initial_state)
    for step_output in app.stream(initial_state, stream_mode="updates"):
        for node_name, state_update in step_output.items():
            # Merge each node's output into accumulated state
            accumulated_state.update(state_update)
            if verbose and "trace" in state_update and state_update["trace"]:
                latest = state_update["trace"][-1]
                icons = {"pass": "✓", "fail": "✗", "proposed": "→",
                         "executed": "⚡", "needs_harmonization": "⚙",
                         "no_harmonization": "○", "max_loops": "⚠"}
                icon = icons.get(latest["decision"], "·")
                print(f"  {icon} {latest['agent']}: {latest['summary']} ({latest['elapsed_sec']:.1f}s)")

    full_final = accumulated_state

    # Build output pandas Series
    output_columns = {}
    if full_final.get("needs_harmonization") and full_final.get("executed_columns"):
        for col_name, idx_val_map in full_final["executed_columns"].items():
            new_series = pd.Series(pd.NA, index=series.index, name=col_name, dtype=object)
            for idx, val in idx_val_map.items():
                if val is not None and val != "null":
                    new_series[idx] = val
            output_columns[col_name] = new_series
    else:
        output_columns[series.name] = pd.Series(
            full_final["formatted_values"], index=series.index, name=series.name
        )

    strategy = "formatting_only"
    if full_final.get("needs_harmonization"):
        strategy = full_final.get("a4_result", {}).get("strategy_summary", "harmonized")

    if verbose:
        total_time = sum(t["elapsed_sec"] for t in full_final.get("trace", []))
        print(f"\n{'='*70}")
        print(f"DONE: {series.name}")
        print(f"  Output columns: {list(output_columns.keys())}")
        print(f"  Agent calls: {len(full_final.get('trace', []))}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"{'='*70}\n")

    return CleaningResult(
        original=series,
        columns=output_columns,
        report=full_final.get("report", ""),
        trace=full_final.get("trace", []),
        strategy_used=strategy,
        full_state=full_final,
    )
