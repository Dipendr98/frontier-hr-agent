# Agent Trajectories

One JSON file per representative case, covering **every agent in every mode**.

Regenerate the whole set:

```bash
python trajectories/generate.py          # Mode 1 only — no API key needed
python trajectories/generate.py --llm    # adds Mode 2 and Mode 3
```

The generator *searches* the cohort for each outcome rather than hardcoding
employee IDs. Hand-curated trajectories go stale the moment a threshold moves,
and a stale trajectory is worse than none — it documents a system that no
longer exists.

## How to read one

**Mode 1** files are the trajectory array: a `config` step recording the
provider (`none` = deterministic, so the run reproduces exactly), then every
agent step in order, then a `tool_calls` step listing every invocation with its
arguments and result. Revision attempts are numbered, and `strictness` rises
per attempt — a retry must clear a *higher* evidence bar, not repeat itself.

**Mode 2 / Mode 3** files wrap the trace with the final decision, the
verification payload and the usage accounting, because for those modes the
feedback that shaped the next step *is* the verification result.

Every decision traces to a rule in [`../AGENTS.md`](../AGENTS.md).

## What each file shows

### Mode 1 — deterministic pipeline

| File | Shows |
|---|---|
| `MODE1_APPROVED_FOR_REVIEW_EMP1180_challenging.json` | **The challenging case.** 89.5% risk on an employee with 4/4 job satisfaction, who actually stayed. Four drivers confirmed, split into two the system can act on (overtime, work-life balance) and two it cannot honestly move (commute, business travel). Cohort memory notes that work-life balance is confirmed for only 5% of the cohort, so it is specific to this person rather than environmental. See CHANGELOG § The challenging case — and note that the *previous* version of this file was the bug in Iteration 13 looking like good judgement. |
| `MODE1_APPROVED_FOR_REVIEW_EMP4.json` | The happy path: quality gate → risk → confirmed drivers → targeted intervention → Critic approves → human gate with `requires_human_decision: true`. |
| `MODE1_ESCALATED_EMP1234.json` | High risk, drivers confirmed, but **every one of them is contextual** — business travel and job level have no honest lever. The planner refuses to improvise, the Critic raises `high_risk_no_action`, and the case reaches a human with the objection attached. Refusing to act is a valid outcome; silently closing a High case is not. |
| `MODE1_NO_ACTION_EMP11.json`, `MODE1_NO_ACTION_EMP18.json` | Low risk → the proportionality gate stops any intervention. Risk decides *whether*; drivers decide *what*. |
| `MODE1_HALTED_DATA_QUALITY_EMP_MALFORMED.json` | `JobSatisfaction=99` is caught before scoring. A bad record never becomes a confident risk number. |

### Modes 2 and 3 — genuine LLM tool-calling agent

Generated only with `--llm` and a provider configured. Filenames carry the
mode, the terminal status and the employee.

| Pattern | Shows |
|---|---|
| `MODE2_RAW_*` | The model chooses its own tools, order and stopping point, and its `finalize` call is **accepted as written**. Compare the tool sequence across employees — the count ranges 8–12 here and the *order* differs case to case. That variability is the evidence a model rather than a script is deciding. |
| `MODE3_VERIFIED_*` | The same agent, with every claim re-derived from the tools before it reaches a reviewer. Read the `verification` step: `verified_drivers` are recomputed statements, not the model's wording. |

**Start with `MODE3_VERIFIED_APPROVED_FOR_REVIEW_EMP10.json`** — it is the
clearest example of the loop the brief asks about. The model finalizes claiming
five drivers; verification rejects two of them with the measurement that
contradicts each:

```
attempt 1: REJECTED
  x unverified_driver: Years with current manager contributes only +0.32
    to the logit, below the 0.4 floor required to call it a driver.
  x unverified_driver: Work-life balance contributes only +0.23
    to the logit, below the 0.4 floor required to call it a driver.
attempt 2: VERIFIED
```

That objection text is sent back as the tool result, the model re-finalizes with
only the claims it can support, and the case reaches a human with four verified
drivers instead of six asserted ones. `MODE3_VERIFIED_APPROVED_FOR_REVIEW_EMP7.json`
shows the same loop catching a single 0.36-against-0.40 claim.

Measured across 20 held-out cases, **25% of unsupervised Mode 2 finalize calls
assert something the tools do not support**, and every one of the seven driver
failures was this same near-miss-on-the-floor shape. The per-case breakdown is
in `../evidence/llm_agent_report.json`, and what verification does *and does not*
fix is reported honestly in `../CHANGELOG.md` § What verification is worth.

## Human checkpoints

No trajectory in this directory ends in an executed action. The terminal states
are `APPROVED_FOR_REVIEW`, `ESCALATED`, `NO_ACTION` and
`HALTED_DATA_QUALITY`. Approved and escalated cases carry
`requires_human_decision: true`, which is a state transition rather than a line
of UI text — `tests/test_pipeline.py::test_no_case_reaches_an_executed_action`
asserts it over the cohort.
