# Improvement Changelog

Every entry records what was tried, why, the evidence, and what we decided.
Experiments that were removed stay in the log with what they taught us.

Unless stated otherwise, "quality" is the Reviewer Case Quality Score (primary,
0–5 rubric) and "wasted" is the wasted intervention rate, both on the same
86-case frozen holdout. Iterations 1–8 were measured under an earlier primary
metric (actionable catch rate); It. 9 replaced it and later numbers are stated
under the current rubric.

Iterations 13–17 are the second build pass. They are listed separately below
because two of them changed numbers that appear throughout this document, and
one of them found that a case this changelog previously held up as the system
working well was in fact a bug.

---

## First pass — building the deterministic pipeline

| Stage | What was tried and why | Evidence | Decision / learning |
|---|---|---|---|
| **Baseline** | Trained logistic regression + static risk-band rules — the standard analytics approach | catch **0.0%**, wasted **36.7%**, 22 leavers flagged with no evidence | Starting point. The flags are mostly right; the cases behind them are empty. |
| **It. 1** | Replace the prototype's hardcoded weights with a real fitted model | ROC-AUC **0.798** holdout, CV **0.826 ±0.084** vs majority-class 0.698 | Kept. The prototype's "AUC 1.0" was an artifact of never training anything. |
| **It. 2** | Swap randomly-generated labels for the real IBM dataset | 342 onboarding-window rows, 29.8% attrition | Kept. Any metric on random labels measures nothing. |
| **It. 3** | Drop Gender / Age / MaritalStatus from features | CV AUC 0.8377 → **0.8095**, forgone **0.028**, inside the ±0.047 CV std | Kept. The ethical choice cost less than the noise floor. |
| **It. 4** | Add Root Cause via exact logit decomposition, so every flag arrives with evidence | leavers flagged with no evidence **22 → 0**, catch **0% → 80.8%** | Kept. Largest single contribution. |
| **It. 5** | Restrict interventions to levers that can move a *confirmed* driver | interventions retargeted; `test_recommendation_targets_only_confirmed_drivers` | Kept. Stops "assign a mentor" firing at someone whose issue is workload. |
| **It. 6** | Add the Critic with 5 decidable checks + revision loop | Critic flagged 64 escalations, 161 revisions on the full cohort | Kept — and it immediately found a real bug (It. 7). |
| **It. 7** | **Bug found by the Critic**: planner proposed formal interventions for employees at ~1% risk | escalations **64 → 6**, revisions **161 → 7** after adding a proportionality gate | Fixed. See "the bug that taught us the most". |
| **It. 8** | Sweep the evidence threshold `MIN_CONTRIBUTION` | At the time: 0.25 → quality 92.8%, wasted 36.7% · 0.40 → quality 90.9%, wasted 31.7% · 0.60 → catch collapses to 46.2% | Set to **0.40**, accepting a 1.9pp quality cost. Superseded by It. 13 — the trade-off no longer exists. |
| **Removed** | LLM-authored root-cause explanations as the *primary* explanation path | With no key the pipeline still had to run; with a key, output varied run to run | Demoted to a presentation layer over fixed structured output. |
| **Removed** | Time-to-productivity regressor | No such target exists in the data; the prototype's "MAE 2.92 days" scored against random numbers | Deleted rather than faked. |
| **Removed** | Business-impact estimates ($650K savings, 450% ROI) | Derived entirely from assumed constants, presented as measured results | Deleted. We have no causal evidence for any of it. |
| **It. 9** | **Replaced the primary metric.** The old one gave the baseline 0% by construction — it returned no evidence, so any evidence-rewarding metric won automatically | New 5-dimension Reviewer Case Quality rubric, incl. independent re-verification of every cited fact | Kept. A metric an opponent cannot score on is not a comparison. |
| **It. 10** | **Built BASELINE+**, a stronger opponent: score + largest cohort deviation + template, the approach a competent engineer would reach for without agents | BASELINE 42.8% → **BASELINE+ 63.3%** | Kept. The honest headline is the gap over this, not over the naive baseline. |
| **It. 11** | **Bug in our own rubric.** It matched only raw column names, so it marked all 165 agent citations unverifiable | agent d3 **0.0% → 94.2%** after matching human-readable labels too | Fixed. The measurement was wrong, not the system — and it would have silently decided the comparison. |
| **It. 12** | Pinned exact dependency versions after an `InconsistentVersionWarning` (artifact written by sklearn 1.9.0, loaded under 1.8.0) | clean-venv run reproduces the table; fingerprints in `evidence/baseline_metrics.json` | Kept. `run_all.sh` also had to build a venv — a bare `pip install` fails on PEP 668 distributions. |

**End of first pass: agent 90.9%** vs BASELINE+ 63.3%, wasted 31.7%.

---

## Second pass — the direction bug, real verification, and memory

| Stage | What was tried and why | Evidence | Decision / learning |
|---|---|---|---|
| **It. 13** | **The evidence gate was direction-blind.** Auditing which features ever got confirmed across all 342 employees, to check the gate did what `AGENTS.md` said it did | Bad-when-high features confirmed **0 times**, rejected as `not_unusual_vs_cohort` **251 times**. Overtime alone: **104 rejections**, with a lever sitting unused | Fixed. quality **90.9% → 93.0%**, d2/d3 94.2% → **98.8%**, d4 93.0% → **95.3%**, wasted 31.7% → 33.3%. See below. |
| **It. 14** | Re-ran the threshold sweep after It. 13, because every row in It. 8 was measured through the bug | 0.10, 0.25 and 0.40 now all score **93.0%**; 0.40 wastes fewer interventions (**33.3%** vs 36.7%) | Kept 0.40 — but it now **dominates** rather than trading off. The unresolved trade-off we published in the first pass has disappeared; the entry stays so the reversal is visible. |
| **It. 15** | Built **Mode 3**: the genuine tool-calling agent with every claim re-derived from the tools (`claim_verifier.py`), and measured it against unverified Mode 2 | First run: **83%** of Mode 2 finalize calls failed verification | Kept — but see It. 16. The 83% was mostly our fault. |
| **It. 16** | **Most of that 83% was a prompt bug, not a model failure.** The risk-band thresholds and the feature→lever map were never in the system prompt; the model was being judged on rules it had not been given | After adding both to the prompt: **83% → 25–30%** unsupervised claim error rate, stable across three runs of 20 held-out cases | Fixed, and the weaker number is the one we report. Same discipline as BASELINE+: beat the strongest version of the opponent. |
| **It. 17** | Added **V7**: verify the free-text rationale, not just the structured payload | Mode 3's recomputed evidence: **0** unverifiable claims. Its prose: **12** | Kept, and it is not enough. V7 rejects only what the record contradicts, so unconfirmable prose still ships. Verifying the struct alone verifies the part nobody reads. |
| **It. 18** | Added **cross-case memory** — driver prevalence, systemic concentration, decision precedent | Systemic finding on the full cohort: `workload_review` indicated for **83 of the 172** employees receiving an intervention (**48%**) | Kept, as annotation only. `test_memory_never_changes_a_decision` asserts the boundary. |
| **It. 19** | Fixed the memory denominator. Measured over everyone reviewed, the largest intervention sat at 24% and looked unremarkable | Over the *actioned* employees it is **48%** — the finding, not noise | Kept. Including the people you are not acting on dilutes exactly the signal you are looking for. |
| **It. 20** | Cohort triage (`advanced/orchestration/cohort.py`) — the artifact a reviewer actually receives | 342 employees in **2.0s** (~6 ms/case) → ranked worklist + briefing | Kept. Running one employee at a time is how you demo an agent, not how anyone does the job. |
| **Removed** | Counting the Mode 2/3 tool budget in model turns | A run advertised as "max 8" executed **17** tools, because a model can emit four tool calls in one turn | Replaced with a count of actual executions. A budget that does not count the thing it limits is not a budget. |
| **Removed** | The original NVIDIA default model (`deepseek-v4-pro`) | One EMP1180 Mode 2 investigation took **575 seconds**. The replacement takes **~23s** | Swapped. It was a good model and an unusable one — inside an agent loop those are compatible. |
| **Removed** | Unpaired Mode 2 vs Mode 3 scoring | A rate-limited run completed 20 Mode 2 cases and 15 Mode 3 cases, and the table compared them directly | Replaced with paired scoring over cases both modes completed. A provider outage was about to be reported as an effect of verification. |
| **It. 21** | **Audited our own baseline and found it under-powered.** BASELINE+ cited a driver, then proposed an unrelated action from the risk-band table and declared `targets_drivers: []` — honest about itself, and worth zero on "action has a mechanism" for every flagged case | Mapping the cited feature to its lever is one dictionary lookup: BASELINE+ **63.3% → 68.4%**, d4 44.2% → **69.8%**. Agent margin **+29.8pp → +24.6pp** | Kept the stronger opponent and the smaller margin. An opponent you have quietly handicapped is not a baseline. |
| **It. 22** | Checked whether the rubric's evidence test is too lenient — it passes a statement if ANY number in it matches the record, and the agent writes five numbers per statement to the baseline's three | Re-scored all 264 statements under a strict "the stated value is the actual value" parse: **agent 220/220, baseline+ 44/44, zero gap** | No change. The leniency does no work, so the headline is not an artifact of a loose measurement. Worth knowing rather than assuming. |
| **It. 23** | Actually built and ran the container, having previously shipped an edited Dockerfile as untested | Build clean; container reproduces **42.8/68.4/93.0** and 107 tests on **Python 3.11** where development was 3.14, same data hash. Found `docker stop` taking the full 10s grace period — shell-form `CMD` left `/bin/sh` as PID 1, so SIGTERM never reached Streamlit | Fixed with `exec` (10s → 0s) and added a HEALTHCHECK. The cross-version match is now reported as evidence the pins hold, not just as a deploy target. |
| **It. 24** | Read the reviewer briefing as the person who has to hold the meeting, rather than as its author — a heuristic review, explicitly not user validation | Six defects. The worst: a 98.9%-risk case proposing a **−0.9 pp** action with nothing flagging it as marginal; the same environmental note on **126 of 172** cards; and no guidance at all on using a flag in a conversation with the person it describes | Fixed all six. The conversation-safety gap is the one that only surfaces from that angle — the metrics were happy, and a document telling a manager to open a difficult conversation with no guidance on doing it safely is not a finished artifact. |
| **Final** | Seven bounded agents across three modes, direction-aware evidence gate, claim verification on a real LLM agent, cohort memory, independent rubric, strengthened baseline | quality **68.4% → 93.0%** vs strongest baseline, **0** unverifiable claims, unsupervised LLM claim error **25-30% → 0%**, 106 tests pass offline | Main contribution: It. 4 + It. 7, corrected by It. 13, honestly measured by It. 9–11 and It. 16. |

---

## The bug that taught us the most (first pass)

The Critic was added to catch weak recommendations. On its first full-cohort
run it escalated 64 of 342 cases — far more than expected. Reading the
escalations found the reason, and it was not the Critic misfiring.

EMP38 sat at **1.1% attrition risk** and the planner had still proposed a formal
intervention, which the Critic correctly flagged as disproportionate, escalating
the case to a human. A 1% risk employee was being routed to manager review.

The root cause was an assumption we had not noticed making. The planner required
a *confirmed driver* before proposing an action, and that felt like a sufficient
gate. It is not: **every employee is below the cohort mean on something**. Driver
evidence is almost always available, so requiring it filters almost nothing.
Evidence that something is true is not evidence that it matters.

The fix was a proportionality gate — risk band decides *whether* to act, drivers
decide *what* to do. Escalations dropped 64 → 6 and revisions 161 → 7.

What generalises: **a verification agent's value is not only the plans it
rejects, but the design flaw its rejection pattern reveals.** We would have
shipped that gate error without the Critic, and no accuracy metric would have
shown it — the model was fine. The bug lived in the policy layer, where model
metrics do not look.

---

## The bug that taught us more (Iteration 13)

`AGENTS.md` said a driver must put the employee in "the worst 35% of the cohort
on that feature". The code compared the raw cohort percentile against 35.

Those are the same rule only when *low* is the concerning direction. For
overtime, commute distance, business travel and previous-employer count, the bad
end is the **high** percentile — so a genuinely alarming value sat at the 70th or
96th percentile and was rejected as "not unusual for this cohort". The gate was
not too strict. It was inverted, on a third of the feature set.

Measured across all 342 employees before the fix:

| | Confirmed as a driver | Rejected as `not_unusual_vs_cohort` |
|---|---:|---:|
| bad-when-low features (satisfaction, balance, training...) | 642 | 0 |
| bad-when-high features (overtime, commute, travel, employers) | **0** | **251** |

Not "rarely". Zero, for every employee, for the life of the system. Overtime —
one of the strongest attrition signals in this dataset, and one of the few with
an honest intervention attached — was rejected **104 times** while
`workload_review` sat in the catalogue unreachable through it.

**Why nothing caught it.** Every test passed. The model was fine — AUC did not
move, because the model always used overtime; only the *explanation* layer could
not cite it. The rubric was fine, and gave the system 90.9%. The Critic was fine:
its job is to check that a recommendation targets a confirmed driver, and every
recommendation did. A driver that is never confirmed never reaches the Critic at
all. **Every verification layer we had was downstream of the filter, so a filter
that silently dropped evidence was invisible to all of them.**

What actually found it was reading a Mode 2 trajectory. The LLM agent, given the
same tools and no percentile gate, called `get_cohort_percentile` on
`DistanceFromHome`, got the 96.5th percentile, and confirmed it as a driver. The
deterministic pipeline had rejected the identical fact from the identical tool
call. Two systems disagreeing about the same number is a much louder signal than
either system alone, and neither one's tests could have produced it.

**And the case we were proudest of was the bug.** The previous version of this
changelog held up EMP1180 as the system at its best: "Root Cause checked four
candidate drivers and rejected three of them as `not_unusual_vs_cohort`,
confirming exactly one." That reads like discipline. It was not discipline. The
three it rejected were overtime, business travel and commute — all bad-when-high,
all structurally unconfirmable. The system was not exercising judgement; it was
incapable of the alternative, and we wrote that incapacity up as rigour.

The generalisable lesson is the uncomfortable one: **a system that declines to
make claims looks rigorous from the outside, and "it refused to cite that" is
indistinguishable from "it could not cite that" unless you go and check the
denominator.** Restraint is the easiest quality to fake, including from yourself.
We now assert it directly — `test_bad_when_high_features_can_now_be_confirmed`
fails if any direction becomes unconfirmable again.

---

## The challenging case

**EMP1180.** Job satisfaction 4/4, environment satisfaction fine — yet the model
scores this employee at **89.5% attrition risk (High)**. The employee actually
stayed. Conflicting signals, a High flag, and a ground truth that disagrees with
it: the case that tests whether the agent investigates or just relays a number.

What the agent produces now (`trajectories/MODE1_APPROVED_FOR_REVIEW_EMP1180_challenging.json`):

- **Four confirmed drivers**, each an exact contribution to the model's own
  logit, each in the worst tail of the cohort: overtime (worst 30%), business
  travel (worst 19%), commute 26 miles (worst 4%), and work-life balance 1/4 —
  **the worst value in the entire cohort**.
- **Split by whether anything can be done.** Overtime and work-life balance have
  honest levers. Business travel and a 26-mile commute do not, and are reported
  as *contextual*: true, relevant, and not something an onboarding intervention
  can claim to change. The reviewer is told the whole picture and lied to about
  none of it.
- **One targeted action**: review workload and reduce overtime, the largest
  simulated reduction (−37.6 pp) among the options with a mechanism.
- **Cohort memory adds what the case cannot see**: work-life balance is confirmed
  for only 5% of the cohort, so this is specific to this person rather than an
  environmental effect. Job level, by contrast, is confirmed for 63% of the
  cohort and is annotated as environmental wherever it appears.

That the employee stayed is not a system failure — precision 0.563 means roughly
two in five High flags will not materialise. The case still hands the reviewer
four checkable facts and one action, instead of an 89.5%.

Mode 3 reaches the same four drivers on this case by a different route: the
model chooses its own tool sequence, and the verifier independently re-derives
every claim before it is shown to anyone.

---

## What verification is worth, measured

`evaluation/evaluate_llm_agent.py` runs the same tool-calling agent twice over
the same held-out cases — once accepting its conclusion (Mode 2), once
re-deriving every claim (Mode 3). Mode 2 runs are *also* passed through the
verifier passively: checked, never corrected, never told. That gives an unbiased
error rate without the measurement changing the behaviour.

Scored only on cases **both** modes completed, so a rate limit in one mode
cannot masquerade as a quality difference.

| | MODE 2 raw | MODE 3 verified |
|---|---:|---:|
| Finalize claims failing verification | **25%** (5 of 20) | 0% by construction |
| Reviewer case quality | 79.0% | 80.0% |
| d4 action has a mechanism | 90.0% | 95.0% |
| Unverifiable claims — recomputed evidence | — | **0** |
| Unverifiable claims — including the model's prose | 12 | 12 |
| Corrections sent back | — | 4 |

**The stable finding: about a quarter of unsupervised finalize calls assert
something the tools do not support.** Measured at 30%, 30% and 25% across three
runs. Not hallucinated employees or invented features — subtler things, and
strikingly consistent. **All seven** driver failures were the same shape:
citing a contribution of 0.38, 0.36, 0.32 or 0.11 as a confirmed driver against
a floor of 0.40 that the prompt states explicitly. Once, the model gave no risk
level at all on an 89.4% case.

The model is not ignoring the rule. It is rounding toward its own judgement at
the boundary — 0.38 "is basically 0.4" — which is a far harder failure to prompt
away than a hallucination, and exactly what a deterministic re-check catches for
nothing.

**What we will not claim.** Verification does *not* reliably improve the
composite case-quality score. 79.0% vs 80.0% is a tie, and across runs it moves
several points in both directions. Reporting a noisy composite as the win would
be the same error as It. 9.

**What it does do, and the finding that came out of measuring it honestly.**
Mode 3's *recomputed evidence* carries zero unverifiable claims. Its *prose*
still carries twelve — the same twelve as Mode 2, because V7 only rejects
statements the record contradicts, not statements it cannot confirm. So verifying
the structured payload while shipping the sentence beside it means **you have
verified the part nobody reads.** The dashboard now separates them explicitly
("Verified evidence" vs "The model's rationale") and both scorings are published,
because the choice of artifact moves the number by six points and picking the
flattering one is exactly the move this document criticises elsewhere.

---

## An honest note on the counter-metric

Iteration 13 improved case quality (90.9% → 93.0%) and made the wasted
intervention rate slightly **worse** (31.7% → 33.3%). Confirming more genuine
drivers means more employees have an actionable one, so more of them receive a
proposal — including some who would have stayed anyway.

We kept the change. The added interventions are supported by real, verifiable,
cohort-unusual evidence that the system was previously unable to state, and
`d4 action has mechanism` rose to 95.3%. But the counter-metric moved against us
and it is reported here rather than left in the JSON.

---

## Hot take

**A verifier that can be argued with is not a verifier — and a verifier can only
see what reaches it.**

The first half we learned in the first pass. Our original Critic returned prose.
It read well and it was useless, because when the planner "revised", nothing
measurable changed — the two agents produced compatible-sounding text and the
loop terminated satisfied. It looked like verification and functioned as
agreement. The Critic that works returns five decidable checks over structured
fields. It cannot be persuaded, because there is nothing in it to persuade.

The second half we learned the hard way in Iteration 13, and it is the one we
would carry forward. We had built three independent layers of checking — the
Critic, an independent rubric, a full test suite — and every one of them sat
*downstream* of the evidence filter. So when the filter silently discarded a
third of the feature space, all three agreed the system was working. They were
not wrong; they were answering "is what arrived here well-formed?" when the
question that mattered was "what never arrived?"

Verification confirms the quality of what your pipeline produces. It is
structurally blind to what your pipeline never produces. Those failures show up
as *absence* — a driver never cited, an intervention never proposed, a check that
never fires — and absence does not trip an assertion. The bug that survived
longest here was not a wrong answer. It was a missing one, and it presented as
good judgement.

So: **audit your filters by their rejection rate, not their acceptance rate, and
be suspicious of a component that never fires.** Concretely, in the next thing we
build: for every gate, log the distribution of what it rejects and assert that no
category is rejected 100% of the time. That single check would have caught this
in the first hour. And when two subsystems disagree about the same number — as
Mode 2 and Mode 1 did about a 96.5th percentile — that disagreement is the most
valuable signal available, and worth more than either system's own confidence.
