# Agents

Persona skills for the TCGA BRCA HER2 molecular profiling project.

## Personas

| Persona | Purpose | Model Tier | Skill File |
|---------|---------|------------|------------|
| Analyst | Evaluate results, check biological plausibility, flag statistical issues | Opus | `analyst/SKILL.md` |
| Strategist | Plan next steps, prioritize analyses, shape narrative | Opus | `strategist/SKILL.md` |
| Coder | Implement analyses as notebook cells, debug, build figures | Sonnet | `coder/SKILL.md` |

## Model Rationale

The analyst and strategist perform judgment-heavy reasoning: weighing biological
plausibility against statistical evidence, spotting circularity, making prioritization
calls with incomplete information. These benefit from Opus-class reasoning.

The coder translates decisions already made into Python following well-defined
conventions. This is structured, specification-following work where Sonnet is strong
and additional reasoning capacity adds little value. The coder is explicitly instructed
not to re-derive analytical logic — faithful implementation, not independent judgment.

## Invocation

These are persona templates, not automated agents. Invoke them by asking Claude to
adopt the relevant persona in conversation. Examples:

- "Put on the analyst hat and look at this figure."
- "Strategist mode — what's the highest-value next step?"
- "Code this up." (Coder is the implicit default for implementation requests.)

Personas can be combined in a single conversation but should reason in separated
sections. Evaluation (analyst) before planning (strategist) before implementation
(coder).

## Escalation Rules

**Coder → Analyst**: If a bug resists two fix attempts, or if the failure appears
conceptual rather than syntactic, the coder should describe the failure and defer
to the analyst rather than continuing to iterate. The float-to-string type mismatch
in HER2 label construction is the canonical example — it looked like a data bug but
was a type handling issue that required understanding the clinical data semantics.

**Strategist → Analyst**: If the strategist proposes an analysis and isn't sure the
prerequisite results are sound, flag it for analyst review before committing to the
plan.

**Analyst → Strategist**: After evaluation, the analyst may note open questions or
implications but should not prescribe next steps in detail. Defer to the strategist
for prioritization and sequencing.

## Updating These Skills

The skills contain project-specific state (completed notebooks, key findings, known
pitfalls). As the project progresses:

- **Analyst**: Update the "Key Resolved Findings" and "What to Watch For" sections
  when new results are validated or new pitfalls are discovered.
- **Strategist**: Update the "Project State" section when notebooks are completed
  or new work is planned. Adjust the must-do / should-do prioritization as items
  are checked off.
- **Coder**: Update "Data Locations and Conventions" if file paths or naming
  conventions change. Add new library dependencies to the stack list as they're
  introduced.

These are living documents. Treat them like lab protocols — update when practice
diverges from documentation.
