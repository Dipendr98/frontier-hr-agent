# Frontier HR Agent

An agentic workflow that turns an onboarding attrition score into a case an HR
reviewer can actually act on — with the evidence attached, the weak
recommendations filtered out, and a human making every decision that touches a
person.

```bash
git clone <repo> && cd frontier-hr-agent
./run_all.sh          # ~60 seconds, no API keys, $0
```

---

## The user and the bottleneck

**Who has the problem.** HR business partners and people-ops reviewers who own
onboarding follow-up. In a mid-size company one reviewer covers a rolling cohort
of new hires and gets, at best, a ranked risk list from an analytics tool.

**The bottleneck.** A risk score is not a decision. When the reviewer opens a
flagged case they still have to work out *why* this person is flagged, *which*
of the signals is worth acting on, and *what* intervention has any mechanism to
change it. That investigation is the actual work, and it is done manually, per
case, inconsistently between reviewers.

The failure mode is specific and measurable: the reviewer gets a list of names
with no case behind them. On our held-out set, the standard score-plus-rules
approach flags 22 of the 26 employees who actually left — and attaches zero
case-specific evidence to any of them. The reviewer is exactly where they
started, just with a shorter list.

**Why solving it is valuable.** Every flagged case that arrives without evidence
either consumes reviewer time to reconstruct, or gets actioned with a generic
intervention that may have no bearing on the person's situation. Both waste the
scarce resource, which is manager attention, not model accuracy.

---

## Result

Three systems, the **same 86 held-out cases**, the **same rubric**, the same
trained model, features and risk thresholds. The only difference is what happens
after the score.

**BASELINE** — score + static risk-band rules.
**BASELINE+** — the same, plus the employee's largest cohort deviation cited as
a reason **and the intervention that has a lever on it**. This exists so the
agent cannot win by construction: a metric that rewards evidence would beat an
opponent that returns none, so we built the strongest cheap alternative and ran
it under the identical rubric. It was under-powered on the first attempt and we
found that by auditing it — see [CHANGELOG.md](CHANGELOG.md) § It. 21.
**AGENT** — the full workflow.

| Metric | BASELINE | BASELINE+ | AGENT |
|---|---:|---:|---:|
| **Reviewer case quality** (primary) | 42.8% | 68.4% | **93.0%** |
| mean score (0–5) | 2.14 | 3.42 | **4.65** |
| — d1 correct triage | 69.8% | 69.8% | 72.1% |
| — d2 evidence present | 0.0% | 51.2% | 98.8% |
| — d3 evidence verifiable | 0.0% | 51.2% | 98.8% |
| — d4 action has a mechanism | 44.2% | 69.8% | 95.3% |
| — d5 proportionate | 100% | 100% | 100% |
| **Wasted intervention rate** (counter) | 36.7% | 36.7% | 33.3% |
| Catch rate w/ verified evidence | 0.0% | 84.6% | 84.6% |
| Raw flag rate on leavers | 84.6% | 84.6% | 84.6% |
| Unverifiable evidence claims | 0 | 0 | **0** |
| Interventions proposed | 44 | 44 | 42 |
| Avg tool calls per case | 0 | 0 | 5.8 |
| Wall clock, 86 cases | 0.05s | 0.05s | 0.25s |

**The headline is +24.6pp over BASELINE+**, not the larger gap over the naive
baseline. We report the harder comparison because it is the honest one.

Note the counter-metric moved slightly *against* us (31.7% → 33.3%) in the
second build pass. That is discussed rather than buried — see
[CHANGELOG.md](CHANGELOG.md) § An honest note on the counter-metric.

### In the brief's summary format

| Metric | Simple baseline | Agent solution | Change |
|---|---:|---:|---|
| **Primary outcome** — reviewer case quality | 42.8% | **93.0%** | +50.2pp |
| — versus the strengthened baseline | 68.4% | **93.0%** | **+24.6pp** |
| Verifiable evidence per case | 0.00 | **2.56** | 0 → 220 statements |
| Cases with unverifiable claims | 0 | **0** | unchanged |
| Machine time per case | 0.6 ms | 2.7 ms | +2.1 ms |
| **Cost per case** | $0.00 | **$0.00** | unchanged |
| Human time per case | *not measured* | *not measured* | — |

**On cost.** The reported configuration uses no API calls at all, so the honest
figure is zero and it is zero for both columns. With an LLM provider enabled the
optional prose layer adds two short completions per case — under $0.01 on every
provider listed in `.env.example`, and $0 on the free NVIDIA tier we used. The
meter is real: every run reports its own token count and estimated cost.

**On human time — the row we cannot fill.** The brief asks for it and we do not
have it, so it is blank rather than estimated. Measuring it means timing a real
reviewer reconstructing a case from the raw record versus reading the generated
one, and no HR reviewer has used this system. Guessing "saves 20 minutes per
case" would be the single least defensible number in this report, and the
project deletes numbers like that elsewhere (see the removed ROI estimates in
[CHANGELOG.md](CHANGELOG.md)). What we can measure — whether the case contains
verifiable, case-specific, actionable evidence — is what the primary metric
scores.

**Disclosed resource difference.** All three systems share the same trained
model, the same feature set, the same risk thresholds and the same 86 frozen
holdout cases. The agent may additionally call tools the baselines do not: logit
decomposition, cohort percentile, and intervention simulation. Those tools *are*
the intervention under evaluation, so the difference is the point rather than a
confound — but it is stated here so a reader can weigh it.

### The rubric

**Reviewer Case Quality Score**, 0–5 per case, five dimensions any of the three
systems can score on:

1. **Correct triage** — flagged if and only if the person actually left
2. **Evidence present** — at least one case-specific fact cited
3. **Evidence verifiable** — every cited fact is **independently recomputed by
   the rubric** from the employee record and must match
4. **Action has a mechanism** — the proposed action can move a cited signal
5. **Proportionate** — action intensity matches the risk band

Dimension 3 is what stops "attach a plausible sentence" from scoring: the rubric
recomputes the numbers itself and rejects any claim it cannot confirm. All three
systems produce **zero unverifiable claims**, so none is winning by fabricating
citations.

**Counter-metric — wasted intervention rate**, frozen in advance so that case
quality cannot be bought by simply flagging more people.

The rubric lives in [`evaluation/rubric.py`](evaluation/rubric.py) and imports
**no agent logic** — the solution does not grade its own exam.
`test_rubric_does_not_import_agent_logic` asserts this.

### Where the improvement comes from

Triage is nearly identical across all three (69.8% / 69.8% / 72.1%) — the model
is the same, so the flags are the same. The gain is concentrated in d2/d3/d4:
the agent produces evidence that survives independent re-checking, and actions
that have a mechanism on the cited signal. That is a precise statement of what
the agentic layer contributes: **not better detection, better cases.**

---

## What the reviewer actually receives

Running one employee at a time is how you demo an agent. It is not how anyone
does this job.

```bash
python -m advanced.orchestration.cohort
```

342 employees in **2.0 seconds** (~6 ms/case, no API key), producing:

- **`evidence/cohort_briefing.md`** — the reviewer-facing document: what needs a
  decision, in priority order, with evidence attached and caveats next to the
  numbers they qualify.
- **`evidence/cohort_worklist.csv`** — the same as a ranked worklist.

Escalations sort first. Approved cases carry their evidence, their proposed
action, its simulated delta, and — where it applies — a note that a confirmed
driver has **no honest lever** and therefore will not be actioned by this
system, only reported.

### Does it hold up on a real headcount?

342 employees is a small company. Measured on synthetic cohorts resampled from
the real data (`python evaluation/benchmark_scale.py`):

| Cohort | Wall clock | ms/case |
|---:|---:|---:|
| 342 | 2.3 s | 6.7 |
| 5,000 | 27 s | 5.4 |
| 10,000 | 55 s | 5.5 |
| 20,000 | 113 s | 5.7 |
| 40,000 | 212 s | 5.3 |
| 80,000 | 431 s | 5.4 |

**ms/case is flat**, so the cost is linear in headcount and the table
extrapolates: ~9 minutes for 100k, ~1.5 hours for a million. It is CPU-bound
with no API calls, so it parallelises across employees if that is ever needed.

It was *not* linear until we measured it. A 342-row dataset is too small for
anything to be visibly slow, and three separate O(n)-per-employee scans were
hiding behind that — cohort memory recomputing driver prevalence on every case
(96 million `Counter.update` calls at n=8,000), precedent lookup scanning every
stored case, and the cohort percentile scanning a full column per call. At
20,000 employees the run took **417 seconds**; the same run now takes **113**.
Fixed by caching the memory rollup with write invalidation, indexing precedent
by (risk band, driver set), and replacing the percentile scan with a binary
search over a presorted column. Identical outputs —
`test_percentile_matches_the_naive_scan` checks the optimised percentile against
the original scan for every feature at every value in the cohort.

**What does not scale** is the briefing document itself: it writes one section
per case, so at 80,000 employees it is a 40 MB markdown file no one will read.
The worklist CSV and the cohort-level findings are the parts that stay useful at
that size. A real deployment would page the briefing by team or manager, which
is not built.

### The finding no single case contains

The briefing also carries what only the cohort shows:

> **Review workload and reduce overtime** — 83 of the 172 employees receiving an
> intervention (48%)
>
> At this concentration the constraint is more likely to be structural —
> staffing, planning or manager load — than 83 separate personal situations.
> Running it as 83 individual conversations spends the reviewer's whole week
> treating one cause 83 times.

A reviewer handed 83 separate cards saying "review this person's workload" has
learned one thing 83 times. The fact that matters is in none of those cards,
because each case only ever sees one employee. That is what the memory layer in
[`advanced/memory.py`](advanced/memory.py) is for, and it is the whole reason it
exists — it is not a cache.

Memory **never changes a decision**. It annotates. Every terminal status is
still produced by the deterministic agents from the employee's own record, so a
stale or poisoned store cannot change who gets escalated — only what context is
attached. `test_memory_never_changes_a_decision` runs the cohort with memory on
and off and asserts identical outcomes.

---

## How the agent works

```
employee record
      │
      ▼
┌─────────────────┐   invalid record halts here rather than being scored
│  Data Quality   │
└────────┬────────┘
         ▼
┌─────────────────┐   trained model → probability → risk band
│ Risk Analysis   │
└────────┬────────┘
         ▼
┌─────────────────┐   exact logit decomposition + direction-aware cohort
│   Root Cause    │   severity; may return UNEXPLAINED rather than invent
└────────┬────────┘   a driver; splits actionable from contextual
         ▼
┌─────────────────┐   options restricted to levers on confirmed drivers,
│  Intervention   │   ranked by simulated what-if
└────────┬────────┘
         ▼
┌─────────────────┐   5 decidable checks; REVISE loops back with the
│     Critic      │   objection and a raised evidence bar
└────────┬────────┘
         ▼
┌─────────────────┐   APPROVED_FOR_REVIEW / ESCALATED / NO_ACTION
│   Human Gate    │   nothing is ever executed
└────────┬────────┘
         ▼
┌─────────────────┐   prevalence, systemic findings, precedent
│  Cohort Memory  │   annotation only — cannot change the status above
└─────────────────┘
```

**Why these agents and not more.** Each one exists because it closes a specific
failure the previous version actually exhibited, and each is separable enough
that the Critic can check one against another. A single agent that both decides
and justifies its own decision cannot be audited that way — the separation is
what makes the verification meaningful rather than decorative.

Full instruction set for every agent: [AGENTS.md](AGENTS.md).

### Grounding

Every factual claim about an employee traces to a number produced in
[`advanced/tools.py`](advanced/tools.py). Root Cause explains the score by
decomposing the model's own logit (`contribution = coef × standardised
deviation`), which is an exact identity for a linear model, not a post-hoc
narrative. `tests/test_pipeline.py` asserts the contributions sum to the model's
logit.

---

## Three modes

| | Mode 1 · deterministic | Mode 2 · LLM raw | Mode 3 · LLM verified |
|---|---|---|---|
| Who picks the next step | Python, fixed order | the model | the model |
| Needs an API key | **no** | yes | yes |
| Model may state a conclusion unchecked | n/a | **yes** | no |
| Carries the headline numbers | **yes** | no | no |

```bash
python advanced/orchestration/workflow.py EMP1180          # Mode 1
python advanced/orchestration/llm_agent.py EMP1180 --raw   # Mode 2
python advanced/orchestration/llm_agent.py EMP1180         # Mode 3
```

**Mode 1** carries every reported metric because it reproduces exactly from a
clean environment with no credentials. When a provider *is* configured it writes
explanation prose only, after the structured decision already exists, and its
output is never read back into control flow. The same run with and without a
provider produces **identical decisions**.

**Mode 2** hands the model a tool schema and lets it decide which tool to call,
in what order, and when it has enough evidence to call `finalize` and stop.
There is no rule-based fallback in that file — it raises rather than quietly
running the deterministic pipeline and calling that "the agent decided". Its
`finalize` call is accepted as written.

**Mode 3** is the same agent with the conclusion treated as an *allegation*.
[`advanced/agents/claim_verifier.py`](advanced/agents/claim_verifier.py)
re-derives every claim from the tools — recomputing the risk band, re-checking
each driver against both evidence floors, re-simulating the recommendation, and
scanning the free-text rationale for numbers the record contradicts. A rejected
claim goes back to the model with the specific measurement that contradicts it,
up to two corrections. A case that still cannot be verified is **escalated, never
approved**.

### What verification is worth

Both modes, same cases, same model, same prompt — scored only on the cases both
completed, so a rate limit in one mode cannot masquerade as a quality
difference. Mode 2 is *also* checked passively — observed, never corrected — to
get an unbiased error rate.

| | MODE 2 raw | MODE 3 verified |
|---|---:|---:|
| **Finalize claims failing verification** | **25%** (5 of 20) | 0% by construction |
| Reviewer case quality | 79.0% | 80.0% |
| d4 action has a mechanism | 90.0% | 95.0% |
| Unverifiable claims in the **recomputed evidence** | — | **0** (vs 12 in the prose) |
| Corrections sent back to the model | — | 4 |
| Avg tool calls per case | 6.8 | 7.3 |

**A quarter of unsupervised finalize calls assert something the tools do not
support.** Not invented employees; subtler things, and strikingly consistent
ones. **All seven** driver failures were the same shape: citing a contribution
of 0.38, 0.36, 0.32 or 0.11 as a confirmed driver against a floor of 0.40 that
the prompt states explicitly. The rest: an intervention with no lever on the
driver it just cited, and one case scored at 89.4% where the model stated no
risk level at all.

The model does not ignore the rule; it rounds toward its own judgement at the
boundary. That is a much harder failure to prompt away than a hallucination, and
it is exactly what a deterministic re-check catches for free.

**What we will not claim:** verification does not reliably lift the composite
case-quality score. 79.0% vs 80.0% is a tie, and across three runs the composite
moved by several points in both directions — the model is non-deterministic and
n is 20. Reporting that as the win would be the same error as Iteration 9.

What *does* move is the thing verification actually does. Mode 3's recomputed
evidence contains **zero** unverifiable claims. Its own prose still contains
twelve — and that is the sharp finding here: **if you verify the structured
payload and ship the unverified sentence next to it, you have verified the part
nobody reads.** V7 exists for that reason and is deliberately conservative, so
prose that is merely unconfirmable (rather than contradicted) still gets through.
Both scorings are published in `evidence/llm_agent_report.json`; neither is
chosen for being kinder.

An earlier measurement put the error rate at 83%. Most of it was ours: the risk
thresholds and the lever map were never in the system prompt, so the model was
being judged against rules it had not been given. Adding them cut it to ~25–30%,
and that is the number we report.

---

## The challenging case

**EMP1180** — job satisfaction 4/4, yet scored 89.5% (High); the employee
actually stayed. The agent confirms four drivers, each in the worst tail of the
cohort, and **splits them by whether anything can be done**: overtime and
work-life balance (worst in the entire cohort) have levers; a 26-mile commute
and frequent business travel do not, and are reported as context rather than
dressed up as actionable. One targeted intervention follows, and cohort memory
adds that work-life balance is confirmed for only 5% of the cohort, so it is
specific to this person.

The previous version of this README presented the same employee as a success
story for the opposite reason — the agent "rejecting three plausible drivers"
looked like discipline. It was a bug. See [CHANGELOG.md](CHANGELOG.md) § The bug
that taught us more.

Trajectory:
[`trajectories/MODE1_APPROVED_FOR_REVIEW_EMP1180_challenging.json`](trajectories/MODE1_APPROVED_FOR_REVIEW_EMP1180_challenging.json).

---

## Reproduction

Python 3.11+. **Total runtime ~60 seconds from an empty virtualenv. No API keys required, $0 cost.**

```bash
git clone <repo> && cd frontier-hr-agent
./run_all.sh          # creates .venv, installs pinned deps, runs everything
```

Or step by step, inside the virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export LLM_NARRATE=0                          # batch runs set this themselves

python data/prepare_data.py                   # ~1s  → data/onboarding_data.csv (342 rows)
python baseline/train.py                      # ~2s  → baseline/attrition_model.joblib
python baseline/fairness_cost.py              # ~3s  → evidence/fairness_cost.json
python evaluation/evaluate.py                 # ~1s  → the three-way table above
python evaluation/threshold_sweep.py          # ~3s  → evidence/threshold_sweep.json
FRONTIER_SKIP_LIVE=1 python -m pytest tests/ -q  # ~6s → 91 passed, 7 skipped
python -m advanced.orchestration.cohort       # ~2s  → briefing + worklist
python evaluation/benchmark_scale.py          # ~2min → scaling up to 20k employees
python advanced/orchestration/workflow.py EMP4  # one case + trajectory
python trajectories/generate.py               # ~5s  → representative trajectories
python -m advanced.doctor                     # provider self-test
```

**Expected output.** Seeds are fixed (`random_state=42`) and the split is
stratified, so every number above is deterministic. The suite reports **91
passed, 7 skipped** in about six seconds.

The 7 skips are the live-agent tests. They need a real provider, and they are
skipped by `FRONTIER_SKIP_LIVE=1` — which `run_all.sh` also sets — so that the
reproduction takes the same ~60 seconds whether or not you have a key
configured. Without that flag and *with* a key, those 7 each drive a full agent
loop and the suite takes 5–10 minutes. Run them deliberately:

```bash
python -m pytest tests/test_llm_agent.py -q    # 5-10 min, needs a key
```

Some of those 7 may still skip individually if the provider rate-limits — a run
the provider never completed says nothing about the agent, so it skips rather
than failing.

**Versions are pinned exactly** (`scikit-learn==1.9.0`, `pandas==3.0.2`,
`numpy==2.4.4`, `joblib==1.5.3`). This matters: the `.joblib` artifact is written
by these versions, and loading it under a different scikit-learn raises
`InconsistentVersionWarning` and can change predictions across major versions.
`run_all.sh` builds a virtual environment so the pins are what actually get used
— a bare `pip install` fails outright on PEP 668 distributions, which we hit and
fixed.

`baseline/train.py` prints and stores an environment + data fingerprint
(`evidence/baseline_metrics.json`), so a reviewer can confirm they are looking at
the artifact the report describes rather than discovering a drift silently.

### Optional: enable Modes 2 and 3

Copy `.env.example` to `.env` and set a provider:

```bash
LLM_PRESET=nvidia          # or gemini | groq | openrouter | openai | anthropic | ollama
LLM_API_KEY=<your key>
```

Then **check it before you rely on it**:

```bash
python -m advanced.doctor
```

```
[  ok  ] Mode 1 (deterministic) reaches a terminal state
         EMP4 -> APPROVED_FOR_REVIEW  (7 tool calls, no key required)
[  ok  ] Provider reachable and authenticated (0.4s)
[  ok  ] Model emits tool calls (Modes 2 and 3 available)
[  ok  ] Latency is workable for the agent loop
         1.4s per round trip x ~8 turns = ~11s per Mode 2 case
```

Those are three different questions and a model can pass the first while failing
the others silently. Our original default model authenticated perfectly and took
**575 seconds** for one case; the current one takes ~23s. `doctor` reports
latency for exactly that reason, and it exits non-zero in CI if the configured
model cannot do tool calling.

`.env` is loaded by `advanced/__init__.py`, so the CLI agents, pytest and the
dashboard all see the same config. **Cost with a provider:** Mode 3 is ~8–10
completions per case; on the free NVIDIA tier that is $0, elsewhere a fraction of
a cent. The meter is real — every run reports its own token count and an
estimated cost.

**Rate limits.** Agent loops are many short sequential calls, which is exactly
the traffic shape that trips free tiers. The client self-throttles
(`LLM_MAX_CONCURRENCY`, `LLM_MIN_INTERVAL`) and honours `Retry-After` rather than
discovering the limit by hitting it.

---

## Dashboard

```bash
streamlit run app.py
```

Three pages: **Case review** (one employee, any of the three modes, with the
live trajectory and — in Mode 3 — the verifier's objections as they are sent
back), **Cohort triage** (the worklist, systemic findings, and a downloadable
briefing), and **Evidence** (every table in this README, read straight from
`evidence/`, with the command to regenerate each one).

The pages degrade honestly: with no provider, Modes 2 and 3 are disabled with an
explanation rather than failing at runtime, and the Evidence page tells you which
command to run rather than showing a traceback.
`tests/test_dashboard.py` exercises all three pages headlessly.

### Deployment

`Dockerfile` and `railway.json` are included for Railway, Cloud Run or ECS. Set
`LLM_PRESET`, `LLM_API_KEY` and `LLM_TIMEOUT` as environment variables — never in
the image.

---

## Data

**Source.** IBM HR Analytics Employee Attrition & Performance, distributed via
Kaggle. This is a **fictional dataset created by IBM data scientists** — no real
employee records appear anywhere in this project. Included at `data/raw/` for
reproducibility.

**Preparation** ([`data/prepare_data.py`](data/prepare_data.py)), three
deliberate decisions:

1. **Tenure filter.** The raw file spans the full employee lifecycle; this
   project is about onboarding, so it keeps `YearsAtCompany <= 2` → 342 rows,
   29.8% attrition.

2. **Protected attributes dropped.** Gender, Age and MaritalStatus are excluded
   from the feature set. We measured what that costs: CV AUC 0.8095 → 0.8377 if
   they are added back, a difference of **+0.028 — smaller than the
   cross-validation standard deviation of ±0.047**. The ethical choice is also
   not a measurable accuracy loss. Reproduce with `python baseline/fairness_cost.py`.

3. **No fabricated productivity target.** The dataset has no time-to-productivity
   column. Rather than invent one, the regression head was removed and the system
   predicts attrition risk only. An earlier version reported "MAE 2.92 days"
   against a randomly generated target; that number was meaningless and is gone.

### Model performance (secondary)

The model is a supporting component, not the contribution.

| | Value |
|---|---:|
| ROC-AUC (frozen test set) | 0.798 |
| PR-AUC | 0.694 |
| Precision / Recall / F1 | 0.563 / 0.692 / 0.621 |
| 5-fold CV ROC-AUC | 0.826 (±0.084) |
| Reference: majority-class accuracy | 0.698 |

---

## Safety and human control

- **Nothing is executed.** The terminal states are `APPROVED_FOR_REVIEW`,
  `ESCALATED`, `NO_ACTION` and `HALTED_DATA_QUALITY`. No state sends a message,
  changes compensation, or takes any HR action.
  `test_no_case_reaches_an_executed_action` asserts this over the cohort, in
  every mode.
- **Human approval is a state transition, not UI text.** An approved case
  carries `requires_human_decision: True` and stops.
- **Unresolved objections travel with the case.** If the Critic (Mode 1) or the
  Verifier (Mode 3) never clears a plan, the case is `ESCALATED` with the
  objections attached — the system does not quietly approve what it could not
  verify.
- **Drivers with no honest lever are reported, never actioned.** A 26-mile
  commute is a real fact about a person and not something an onboarding
  intervention can claim to fix. The system says both parts.
- **Simulations are labelled.** Every `simulate_intervention` result carries
  `"SIMULATED model what-if on correlational features; NOT a causal effect."`
- **Memory annotates, never decides.** Asserted by test, not by policy.
- **No business-impact theatre.** No dollar savings or ROI figures are claimed.
- **No credentials in the repository.** Provider config is environment-only;
  `.env` is gitignored and `advanced/doctor.py` reports only whether a key is
  present and how long it is.

---

## What this is, and what it is not

The bottleneck is real. Risk scores arriving without cases is a genuine, boring,
widely-experienced problem, and "build the case, don't just rank the name" is a
real answer to it. The parts of this that would survive contact with a real
deployment are the *architecture* — an evidence gate that can say "unexplained",
verification that re-derives rather than re-asks, actions restricted to levers
that exist, a human gate that is a state transition — and none of that is
specific to HR.

What this is **not** is a system anyone should point at real employees tomorrow.
Stated plainly, because the ground rules ask for claims tied to evidence and
these are the places where there is none:

- **No real data.** The IBM dataset is fictional, by construction. Nothing here
  has been validated against real HR records, and real ones would be messier,
  more missing, and differently distributed.
- **No real user.** No HR business partner has used this. The claim that these
  cases are more useful than a bare score is argued from a rubric we wrote, not
  from a reviewer who tried both. That is the single biggest gap.
- **The interventions are hand-authored.** `FEATURE_LEVERS` and the catalogue
  encode our assumptions about which actions plausibly address which drivers.
  They are not derived from evidence about what actually retains people.
- **Simulated deltas are not effect estimates.** "−37.6 pp" is what the model
  would predict under changed inputs. It is not what would happen if HR did the
  thing. Every such number is labelled, and no aggregate benefit is claimed from
  them anywhere.
- **The model is small and imprecise.** 342 rows, precision 0.563 — roughly two
  in five High flags will not materialise. That is why nothing is executed.
- **Attrition prediction is ethically loaded** independently of accuracy. A
  system that labels new hires as flight risks can become self-fulfilling if the
  label reaches a manager as a verdict rather than a case. This design pushes
  against that — evidence attached, no execution, human decides, protected
  attributes excluded — but design intent is not a guarantee, and a real
  deployment would need consent, an appeals path, and monitoring for
  disparate impact that this project does not have.

The honest summary: **a credible prototype of a real solution to a real problem,
evaluated rigorously against a fair baseline on fictional data.** The evaluation
is the part we would defend hardest; the deployment readiness is the part we
would not defend at all.

---

## What existed before the hackathon

Required disclosure. Before the competition there was a Zerve notebook
prototype: synthetic data generation, an engagement score, HRIS export formats
and matplotlib dashboards.

**None of it is in this submission.** The prototype's model used hardcoded
weights (`weight = -1.2`, `bias = 0.8`) that were never fitted, its attrition
labels were `np.random.choice`, its "5-fold cross-validation" scored every fold
with the same fixed parameters, and its validation report printed PASS for checks
it had not performed. Everything in this repository — the training pipeline, the
tool layer, all seven agents, both orchestrators, the memory layer, the
evaluation harness and the tests — was written during the hackathon. The
prototype's value was showing us exactly which failure modes to design against.

---

## The main failure mode, and the hot take

**A verifier that can be argued with is not a verifier — and a verifier can only
see what reaches it.**

We had three independent layers of checking: a Critic with five decidable
checks, an evaluation rubric that imports no agent logic, and a test suite that
asserts each README claim. All three passed while a third of the feature set was
being silently discarded before any of them saw it, because all three sat
*downstream* of the evidence filter. They were answering "is what arrived here
well-formed?" when the question that mattered was "what never arrived?"

Verification confirms the quality of what your pipeline produces. It is
structurally blind to what your pipeline never produces — and those failures
present as *absence*, which does not trip an assertion. Worse, absence looks like
rigour: a system that declines to cite evidence reads as disciplined, and "it
refused to" is indistinguishable from "it could not" unless someone checks the
denominator. We wrote up that exact incapacity as good judgement in the first
version of this README.

**What we would do differently, concretely:** for every gate, log the
distribution of what it rejects and assert that no category is rejected 100% of
the time. That single check would have caught this in the first hour. And when
two subsystems disagree about the same number — as the LLM agent and the
deterministic pipeline did about a 96.5th percentile — treat the disagreement as
the most valuable signal available. It is worth more than either system's
confidence in itself.

Full story: [CHANGELOG.md](CHANGELOG.md).

---

## Repository

```
README.md                  this file
AGENTS.md                  the instructions that shape each agent, all 3 modes
CHANGELOG.md               the improvement story, incl. removed experiments
run_all.sh                 clean-environment reproduction, ~60s, $0
requirements.txt           exact pins (see Reproduction)
.env.example               provider configuration, fully commented
app.py                     dashboard entry point (st.navigation)
app_pages/                 case_review · cohort · evidence
app_shared.py              cached loaders shared by the pages
Dockerfile · railway.json  container + deployment config
data/
  README.md                source, license (DbCL v1.0), preparation decisions
  raw/                     IBM dataset as distributed
  prepare_data.py          tenure filter, protected-attribute exclusion
baseline/
  train.py                 real .fit(), frozen holdout, true CV, fingerprints
  fairness_cost.py         measures the cost of the protected-attr exclusion
  pipeline.py              BASELINE: score + static rules
  pipeline_plus.py         BASELINE+: score + top cohort deviation as reason
advanced/
  llm.py                   multi-provider client: retries, throttle, cost meter
  doctor.py                provider self-test — key, tool calling, latency
  tools.py                 the only path to data and models; logs every call
  memory.py                cross-case memory; annotates, never decides
  agents/                  data_quality · risk_analysis · root_cause ·
                           intervention · critic · human_gate · claim_verifier
  orchestration/
    workflow.py            Mode 1: bounded loop, revision, trajectory logging
    llm_agent.py           Modes 2/3: tool-calling agent, verification loop
    cohort.py              batch triage → worklist + reviewer briefing
evaluation/
  rubric.py                independent 5-dimension rubric — no agent imports
  evaluate.py              BASELINE / BASELINE+ / AGENT on identical cases
  evaluate_llm_agent.py    Mode 2 vs Mode 3, paired scoring
  threshold_sweep.py       the experiment behind MIN_CONTRIBUTION
  benchmark_scale.py       cohort throughput from 342 to 80,000 employees
evidence/                  metrics, sweep, per-case results, briefing, memory
trajectories/
  generate.py              regenerates the representative set
  README.md                what each trajectory demonstrates
tests/                     91 offline + 7 live-agent tests
```
