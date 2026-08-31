# Agent Instructions

The brief asks for "the instructions that shape each agent". In this system
most of those instructions are not prompts — the Mode 1 agents are
deterministic Python, and their behaviour is shaped by explicit rules,
thresholds and contracts. This file states each agent's instruction set in one
place, with the enforcing code linked so every rule can be checked against its
implementation.

Where a real prompt exists — the tool-calling agent in Modes 2 and 3 — it is
quoted here in full, because that prompt *is* the instruction set.

---

## The three modes, and which agents run in each

| | Mode 1 · deterministic | Mode 2 · LLM raw | Mode 3 · LLM verified |
|---|---|---|---|
| Who picks the next step | Python, fixed order | the model | the model |
| Needs an API key | no | yes | yes |
| Agents involved | 1–6 | tool layer only | tool layer + agent 7 |
| Model may state a conclusion unchecked | n/a | **yes** | no |
| Terminal states | 4 | `COMPLETED` | same 4 as Mode 1 |
| Used for the headline numbers | **yes** | no | no |

Mode 1 carries the reported metrics because it reproduces exactly from a clean
environment with no credentials. Modes 2 and 3 exist to answer a different
question — what happens when a model genuinely drives — and they are measured
against each other in `evidence/llm_agent_report.json`.

---

# Mode 1 — the deterministic pipeline

Sequence, fixed in `advanced/orchestration/workflow.py`:

```
data_quality → risk_analysis → (root_cause → intervention → critic) ×≤3
             → human_gate → [cohort_memory]
```

When an LLM provider is configured, it receives the task instructions quoted in
the "LLM task" rows below, and **nothing it returns can change any decision on
this page**. The prose layer is off by default in batch runs (`LLM_NARRATE`).

## 1. Data Quality Agent — `advanced/agents/data_quality.py`

**Instruction:** Refuse to let an untrustworthy record be scored.

| Rule | Detail |
|---|---|
| Required fields | All 13 model features must be present and numeric |
| Range checks | Survey scales within bounds (JobSatisfaction 1–4, TrainingTimesLastYear 0–6, OverTime_flag 0/1) per `FEATURE_BOUNDS` |
| On failure | `FAIL` with the specific issues; the orchestrator halts the case (`HALTED_DATA_QUALITY`). A bad record is never scored |

## 2. Risk Analysis Agent — `advanced/agents/risk_analysis.py`

**Instruction:** Score and classify. Do not explain, do not recommend.

| Rule | Detail |
|---|---|
| Scoring | Call `predict_risk` on the trained artifact only — never retrain, never adjust |
| Bands | High ≥ 0.60, Medium ≥ 0.35, else Low |
| Scope | Explanation belongs to Root Cause; action belongs to Intervention. This separation is what lets the Critic check one against the other |

## 3. Root Cause Agent — `advanced/agents/root_cause.py`

**Instruction:** Explain the score using only the model's own arithmetic, and
say "unexplained" rather than invent a driver.

| Rule | Detail |
|---|---|
| Method | Exact logit decomposition: `contribution = coef × standardised deviation` per feature — an identity for a linear model, not a story |
| Confirmation floor 1 | Contribution ≥ `MIN_CONTRIBUTION` (0.40; set by the sweep in `evidence/threshold_sweep.json`) |
| Confirmation floor 2 | The employee must sit in the worst 35% of the cohort on that feature, measured on `severity_percentile` — **direction-aware**, so 0 is the worst value whether the feature is bad-when-low (satisfaction) or bad-when-high (overtime, commute). Comparing the raw percentile instead made every bad-when-high feature unconfirmable; see Iteration 13 |
| Both floors required | Evidence that something is true is not evidence that it matters — see the EMP38 bug in CHANGELOG.md |
| Actionable vs contextual | A confirmed driver with no honest lever (commute, business travel, job level, previous employers) is reported to the reviewer as **contextual** and is never actioned. "We found something" must not be mistaken for "we can do something" |
| If nothing clears both floors | Status `UNEXPLAINED`, stated plainly |
| LLM task (optional) | "Summarise the root cause in 2 sentences. Use ONLY the numbers provided. Do not speculate about causes not listed. If the evidence is weak, say so." |

## 4. Intervention Planner — `advanced/agents/intervention.py`

**Instruction:** Propose only actions that have a mechanism on a confirmed
driver, and only when the risk warrants acting at all.

| Rule | Detail |
|---|---|
| Proportionality gate | Risk band Low → `NO_ACTION_PROPOSED`, always. Risk decides *whether*; drivers decide *what* |
| Candidate set | Only interventions with a lever on a *confirmed, actionable* driver (`FEATURE_LEVERS`) |
| Ranking | Simulated what-if via `simulate_intervention`; most negative delta wins |
| No candidates | `NO_ACTION_PROPOSED` with the reason — escalation, not improvisation |
| Labelling | Every simulated delta carries: "SIMULATED model what-if on correlational features; NOT a causal effect." |
| LLM task (optional) | "Justify choosing this intervention over the alternatives. Reference only the drivers and simulated deltas given. Do not claim the intervention will cause the reduction." |

## 5. Critic — `advanced/agents/critic.py`

**Instruction:** Five decidable checks. No prose verdicts, nothing to persuade.

| Check | Rejects when |
|---|---|
| C1 unsupported_recommendation | Risk elevated, root cause `UNEXPLAINED`, yet an action is proposed |
| C2 driver_mismatch | Proposed action targets features outside the confirmed actionable set |
| C3 no_simulated_benefit | Chosen intervention's simulated delta ≥ 0 |
| C4 high_risk_no_action | High risk case closing with nothing proposed |
| C5 over_intervention | Low risk case receiving a formal intervention |

On `REVISE`, the orchestrator re-runs Root Cause with the evidence floor raised
×1.4 per attempt (max 2 revisions) — a retry must clear a *higher* bar, not
repeat itself.

## 6. Human Approval Gate — `advanced/agents/human_gate.py`

**Instruction:** Nothing is ever executed. Package for a human, or refuse to.

| Rule | Detail |
|---|---|
| Critic not cleared after revisions | `ESCALATED`, with the unresolved objections attached to the case |
| High risk + no proposable action | `ESCALATED` — refusing to act is a valid outcome; silently closing a High case is not |
| Approved path | `APPROVED_FOR_REVIEW` with `requires_human_decision: True` — a state transition, not UI text |
| Terminal states | `APPROVED_FOR_REVIEW` · `ESCALATED` · `NO_ACTION` · `HALTED_DATA_QUALITY`. No state sends a message, changes compensation, or takes any HR action (`test_no_case_reaches_an_executed_action`) |

## Cohort memory — `advanced/memory.py`

**Instruction:** Carry forward only what a single case cannot see. Annotate;
never decide.

| Rule | Detail |
|---|---|
| Driver prevalence | A driver confirmed for ≥40% of the cohort is reported as **environmental**; ≤10% as **distinguishing**. The same fact means different things at different frequencies |
| Systemic findings | One intervention indicated for ≥25% of the **actioned** employees is raised as a structural finding addressed to the org, not as N individual recommendations. The denominator excludes people receiving no intervention — including them dilutes the signal to nothing |
| Silence below sample | No prevalence claim at all below 20 recorded cases |
| Precedent | Comparable cases (same band, same driver set) with differing outcomes are surfaced as a consistency flag |
| Hard boundary | Memory cannot change `status`. `test_memory_never_changes_a_decision` runs the cohort with memory on and off and asserts identical terminal states |

---

# Modes 2 and 3 — the genuine tool-calling agent

`advanced/orchestration/llm_agent.py`. The model is handed a tool schema and
decides itself which tool to call, with what arguments, in what order, and when
it has enough evidence to call `finalize` and stop. **There is no rule-based
fallback in this file** — it raises `RuntimeError` without a key rather than
quietly running the deterministic pipeline and calling that "the agent decided".

```bash
export LLM_PRESET=nvidia && export LLM_API_KEY=<your key>
python -m advanced.doctor                                  # check the provider first
python advanced/orchestration/llm_agent.py EMP1180         # Mode 3 (verified)
python advanced/orchestration/llm_agent.py EMP1180 --raw   # Mode 2 (unchecked)
```

## Tools offered

`get_employee_profile` · `predict_risk` · `get_risk_drivers` ·
`get_cohort_percentile` · `simulate_intervention` · `finalize`

Budget: **12 tool executions**, counted in actual tool calls rather than model
turns. Counting turns is why an earlier run advertised as "max 8" executed 17 —
a budget that does not count the thing it limits is not a budget. Every bad
argument comes back as a structured `error` the model can correct from, because
a model guessing a feature name is normal behaviour, not a reason to lose the
case.

## System prompt

The model is told the operational definitions it is judged against — the risk
thresholds and the full feature→lever map. This is not politeness. The first
measured Mode 2 run failed verification on 83% of cases, and most of those were
the model applying its own idea of what "Medium risk" means (it called 10.4%
Medium) and its own idea of which intervention fits which problem. Neither rule
was in the prompt. Measuring a model against rules it was never given reports a
prompt bug as a model failure; with the definitions supplied, the error rate
fell to 30%. Same discipline as BASELINE+ — beat the strongest version of the
opponent.

> You are an HR onboarding-risk investigator. You have tools, not a fixed
> script — decide yourself which to call, in what order, and when you have
> enough evidence to stop.
>
> Definitions you are held to. These are the system's, not yours:
>   Risk bands: High >= 0.60, Medium >= 0.35, otherwise Low. Apply these
>   numerically to the probability predict_risk returns. Do not substitute your
>   own sense of what "medium risk" feels like.
>   Which intervention can move which driver: *(full lever map inserted here)*
>   An intervention that does not appear against a driver cannot address it.
>
> Rules you must follow:
> 1. Never state a risk level before calling predict_risk.
> 2. Never claim a driver is real without calling get_risk_drivers AND
>    get_cohort_percentile on it. A driver must be BOTH a material contributor
>    (logit_contribution >= 0.4) AND unusual for this employee versus the cohort
>    (severity_percentile <= 35). Use severity_percentile, not the raw
>    percentile: it already accounts for whether high or low is the bad
>    direction. A feature nobody is unusual on is not a driver just because its
>    contribution is positive.
> 3. Before recommending an intervention, call simulate_intervention on your top
>    1-2 candidates and prefer the one with the larger negative risk delta. Only
>    recommend an intervention that can actually move a driver you verified.
> 4. If risk is Low, call finalize immediately with an empty confirmed_drivers
>    list and no intervention — do not spend tool calls investigating someone
>    who is not at risk.
> 5. You have at most 12 tool calls. Call finalize as soon as you have enough
>    evidence — do not call tools you do not need.
> 6. finalize is mandatory. You must call it to end the investigation.
> 7. Recommending nothing is a legitimate answer. If no driver survives, say so
>    and finalize with an empty list rather than reaching for a plausible
>    action. Nothing you decide is executed; a human reviews every case.

## 7. Claim Verifier — `advanced/agents/claim_verifier.py` *(Mode 3 only)*

**Instruction:** Treat the model's conclusion as an allegation. Re-derive all of
it from the tools. Believe the numbers, not the sentence.

| Check | Rejects when |
|---|---|
| V1 risk_level_mismatch | Claimed band ≠ band recomputed from `predict_risk` |
| V2 unverified_driver | A claimed driver fails the contribution floor or the cohort-severity floor when recomputed |
| V3 unknown_intervention | Recommended key is not in the catalogue |
| V4 no_mechanism | Recommended intervention has no lever on any **verified** driver |
| V5 no_simulated_benefit | Re-simulating the recommendation gives delta ≥ 0 |
| V6 action_without_evidence | An intervention is recommended with no driver surviving verification |
| V7 unverifiable_rationale | The free-text rationale states a number about a feature that the employee's record contradicts |

V7 exists because V1–V6 check the structured payload and the reviewer does not
read the structured payload — they read the rationale. An agent whose fields are
all correct and whose sentence says "job satisfaction of 2" about someone who
scored 4 has still misinformed the person making the decision.

On `REJECTED` the specific measurements are sent back to the model
(`objection_prompt`), up to **2 corrections**. A case that still cannot be
verified is `ESCALATED` with the objections attached — never approved.

**Why re-derive rather than ask the model to check itself.** A model asked to
review its own output tends to agree with it, and the resulting loop looks like
verification while functioning as agreement. That is the exact failure the
Critic was built to avoid, one level up.

**Deliberately conservative on V7.** It fires only on an explicit
"`<feature> is <number>`" construction and stays silent when the number is
qualified by a statistic word. A verifier that false-positives sends the model
back to correct a claim that was fine, which burns budget and trains it away
from correct answers. Under-reporting is recoverable; over-reporting is not.

---

## Where the LLM sits — and what it is not allowed to do

| Decided by deterministic Python | Left to the LLM |
|---|---|
| whether a record is valid | *(Mode 1)* the prose of an explanation, after the decision exists |
| the risk band, in every mode | *(Modes 2/3)* which tool to call next, and when to stop |
| which drivers count as confirmed | *(Mode 2 only)* the conclusion itself, unchecked |
| which interventions are permissible | |
| whether the Critic or Verifier approves | |
| whether a case reaches a human | |

The pattern across all three modes: **the model may choose the path; it may not
certify the destination.** In Mode 1 it cannot even choose the path. In Mode 3
it chooses freely and every claim is re-derived. Mode 2 exists precisely because
the difference has to be measurable — see `evidence/llm_agent_report.json`.
