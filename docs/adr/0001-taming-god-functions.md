# ADR 0001: containing the oversized functions without a big-bang rewrite

Status: accepted (2026-11-22)

## Context

Eight functions exceed 300 lines; `run_tool_loop` is 1809 lines with 31 nested
closures, cyclomatic complexity 490 and nesting depth 12. They carry the
qualified behaviour of the product. A wholesale split would touch state shared
through closures and has no behavioural safety net beyond long, model-dependent
qualification runs, so the risk of a silent regression outweighs the
maintainability gain.

## Decision

Reduce risk per function by *kind*, never by a single rewrite:

1. **Freeze, in two dimensions.** `tests/test_architecture.py` freezes line count
   (`FROZEN_LONG_FUNCTIONS`) and, new, cyclomatic complexity and nesting depth
   (`FROZEN_COMPLEXITY`). Neither may grow, no new offender may appear, and an
   improvement must shrink the table (ratchet).
2. **Flat, order-dependent chains become data/matchers** (mechanical, provable).
   `detect_native_model_profile` is a sequence of independent
   "identity check, return profile" pairs. It is split into ordered matcher
   functions behind a characterization (golden) test captured from the original
   code first. Behaviour is identical by construction and by test.
3. **Stateful monoliths are strangled, not split.** For `run_tool_loop` and
   `_run_autonomous_locked`, new behaviour goes into new modules and is called
   from the loop through a narrow seam; code is extracted from the monolith only
   when a change already requires touching it, with the helper keeping the
   caller's guards (see `AnalysisRuntime.step`, 537 to 516 lines). The ratchets
   make "add logic inline" fail the build.
4. **Extraction needs evidence.** A change that extracts from a frozen function
   must show the targeted tests plus `scripts/check_core.py`, and lower the
   frozen number in the same commit.

## Consequences

* Debt cannot grow silently; every reduction is permanent.
* The riskiest code is changed only when a feature needs it, with a test.
* Fully dismantling `run_tool_loop` remains out of scope until a deterministic
  replay harness exists for it.
