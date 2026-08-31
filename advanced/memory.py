"""
Cross-case memory.

The point of memory here is NOT to make a single case faster. It is that some
conclusions are invisible from inside one case and only exist across the cohort.

A reviewer handed forty separate cards that each say "review this person's
workload" has learned one thing forty times, and will act on it forty times, at
forty individual meetings. The fact that matters — *this is a staffing problem,
not forty personal problems* — is nowhere in any individual case, because each
case only ever sees one employee. That is the gap this module fills.

Three things are carried forward, and each one changes an output:

  1. Driver prevalence — how common a confirmed driver is across the cohort.
     A driver that 4% of people have is a fact about this person. The same
     driver at 55% is a fact about the workplace, and the case says so.

  2. Intervention concentration — when one intervention is indicated for more
     than SYSTEMIC_SHARE of everyone reviewed, the cohort report raises a
     systemic finding addressed to whoever owns the org, not the individual.

  3. Decision precedent — a materially similar employee who was treated
     differently is surfaced as a consistency flag. Two comparable new hires
     getting opposite outcomes is a fairness problem, and it is not detectable
     from either case alone.

Memory NEVER overrides a decision. It annotates. Every terminal status is still
produced by the deterministic agents from the current employee's own record, so
the pipeline stays reproducible and a poisoned or stale store cannot change who
gets escalated — only what context is attached. That boundary is deliberate:
memory that can silently alter decisions is memory you cannot audit.
"""
import json
import os
import threading
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STORE = os.path.join(BASE_DIR, "evidence", "cohort_memory.json")

# One intervention indicated for more than this share of reviewed employees is
# reported as a systemic finding rather than N individual recommendations.
SYSTEMIC_SHARE = 0.25
# A driver confirmed for more than this share of the cohort is environmental
# context, not a distinguishing fact about the individual.
COMMON_DRIVER_SHARE = 0.40
# Below this share, a confirmed driver is genuinely unusual and worth leading with.
RARE_DRIVER_SHARE = 0.10
# Minimum cases before prevalence claims are made at all. Below this the shares
# are noise and stating them would be worse than staying quiet.
MIN_CASES_FOR_PREVALENCE = 20


class CohortMemory:
    """
    An append-only record of what the agents concluded, per employee.

    Keyed by employee_id so re-running a case replaces its entry rather than
    double-counting it — otherwise prevalence would drift upward every rerun and
    the systemic threshold would fire on nothing.
    """

    def __init__(self, path: str = DEFAULT_STORE, autoload: bool = True):
        self.path = path
        self._lock = threading.Lock()
        self.cases = {}
        self._stats = None
        self._frozen = False
        self._annotate = True
        if autoload:
            self.load()

    # -- read/write modes --------------------------------------------------
    #
    # The two-pass cohort run has passes with opposite jobs, and saying so in
    # code is what keeps it linear:
    #
    #   pass 1  WRITES every case, and its annotations are discarded
    #   pass 2  READS the finished store, and writes nothing new
    #
    # Left undeclared, each pass does both. Pass 1 computes prevalence and
    # precedent per employee and throws the answers away; pass 2 re-records
    # data already present, invalidating the statistics cache once per
    # employee. Each of those is an O(n) scan per case, so the cohort run went
    # quadratic twice over for work that was never needed.

    def collect_only(self):
        """Pass 1: accept writes, skip the annotations nobody reads."""
        self._annotate = False
        return self

    def freeze(self):
        """Pass 2: serve reads, ignore writes."""
        self._frozen = True
        self._annotate = True
        return self

    def unfreeze(self):
        self._frozen = False
        self._annotate = True
        return self

    # -- derived statistics, computed once per write ----------------------

    def _invalidate(self):
        self._stats = None

    def _compute_stats(self) -> dict:
        """
        Roll up prevalence and concentration in ONE pass over the store.

        This is cached and invalidated on write because the alternative — which
        is what this class originally did — recomputes both from scratch on
        every case. That is invisible on 342 employees and quadratic beyond it:
        profiling a 8,000-employee run showed 96 million Counter.update calls
        and 36% of total runtime spent recounting facts that had not changed.
        Prevalence is a property of the store, so it belongs to the store, not
        to each read of it.
        """
        n = len(self.cases)
        drivers, interventions, actioned = Counter(), Counter(), 0
        # Comparable cases are indexed by their (band, driver-set) signature so
        # precedent lookup is a dict hit rather than a scan of every case. The
        # scan version was a second O(n) per case, i.e. quadratic again.
        by_signature = {}
        for c in self.cases.values():
            confirmed = set(c.get("confirmed_drivers") or [])
            drivers.update(confirmed)
            key = c.get("intervention_key")
            if key:
                interventions[key] += 1
                actioned += 1
            by_signature.setdefault(
                (c.get("risk_level"), frozenset(confirmed)), []).append(c)
        return {
            "n": n,
            "n_actioned": actioned,
            "driver_prevalence": {f: round(k / n, 4) for f, k in drivers.items()} if n else {},
            "intervention_concentration": (
                {k: round(v / actioned, 4) for k, v in interventions.items()}
                if actioned else {}),
            "by_signature": by_signature,
        }

    def _get_stats(self) -> dict:
        if self._stats is None:
            self._stats = self._compute_stats()
        return self._stats

    # -- persistence ------------------------------------------------------

    def load(self):
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.cases = json.load(f).get("cases", {})
            except (json.JSONDecodeError, OSError):
                # A corrupt store must not take the pipeline down with it.
                self.cases = {}
        self._invalidate()
        return self

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"schema": 1, "cases": self.cases}, f, indent=2, default=str)

    def clear(self):
        with self._lock:
            self.cases = {}
            self._invalidate()
        return self

    # -- writing ----------------------------------------------------------

    def record(self, employee_id: str, risk: dict, root_cause: dict,
               intervention: dict, status: str):
        """Store one completed case. Idempotent per employee_id; no-op if frozen."""
        if self._frozen:
            return
        rec = intervention.get("recommendation") or {}
        with self._lock:
            self.cases[str(employee_id)] = {
                "employee_id": str(employee_id),
                "risk_level": risk.get("risk_level"),
                "attrition_probability": risk.get("attrition_probability"),
                "confirmed_drivers": [c["feature"] for c in root_cause.get("confirmed_drivers", [])],
                "actionable_features": list(root_cause.get("actionable_features", [])),
                "intervention_key": rec.get("key"),
                "status": status,
            }
            self._invalidate()

    # -- reading ----------------------------------------------------------

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    def driver_prevalence(self) -> dict:
        """Share of recorded cases in which each feature was a confirmed driver."""
        return self._get_stats()["driver_prevalence"]

    @property
    def n_actioned(self) -> int:
        """Cases that received a proposed intervention."""
        return self._get_stats()["n_actioned"]

    def intervention_concentration(self) -> dict:
        """
        Share of ACTIONED cases for which each intervention was chosen.

        The denominator is deliberately the acted-upon, not everyone reviewed.
        The question this answers is "are we giving the same answer to everyone
        we are answering?" — and roughly half this cohort is Low risk and gets
        no intervention by design, so including them just dilutes the signal
        toward zero and hides the concentration we are looking for. Measured
        over the whole cohort, this project's largest intervention sits at 24%
        and looks unremarkable; measured over the employees actually receiving
        one, it is 46% and is the finding.
        """
        return self._get_stats()["intervention_concentration"]

    def contextualise_drivers(self, confirmed: list) -> list:
        """
        Annotate this case's confirmed drivers with how common they are.

        Returns [] until MIN_CASES_FOR_PREVALENCE cases are known — a "rare
        driver" claim computed from four employees is not a finding.
        """
        if not self._annotate or self.n_cases < MIN_CASES_FOR_PREVALENCE:
            return []
        prev = self.driver_prevalence()
        notes = []
        for c in confirmed:
            feat = c["feature"]
            share = prev.get(feat)
            if share is None:
                continue
            if share >= COMMON_DRIVER_SHARE:
                kind, note = "environmental", (
                    f"{c['label']} is a confirmed driver for {share:.0%} of the "
                    f"cohort reviewed so far, so it describes the environment more "
                    f"than this individual."
                )
            elif share <= RARE_DRIVER_SHARE:
                kind, note = "distinguishing", (
                    f"{c['label']} is confirmed for only {share:.0%} of the cohort "
                    f"reviewed so far — this is specific to this employee."
                )
            else:
                continue
            notes.append({"feature": feat, "cohort_share": share,
                          "kind": kind, "note": note})
        return notes

    def systemic_findings(self) -> list:
        """
        Interventions concentrated enough to be an org-level issue.

        Addressed to whoever owns the team, not to the individual's reviewer —
        the whole point is that the individual is not the unit of the problem.
        """
        if self.n_cases < MIN_CASES_FOR_PREVALENCE:
            return []
        from advanced.tools import INTERVENTION_CATALOG
        out = []
        for key, share in sorted(self.intervention_concentration().items(),
                                 key=lambda kv: -kv[1]):
            if share < SYSTEMIC_SHARE:
                continue
            label = INTERVENTION_CATALOG.get(key, {}).get("label", key)
            affected = [c["employee_id"] for c in self.cases.values()
                        if c.get("intervention_key") == key]
            out.append({
                "intervention_key": key,
                "label": label,
                "share_of_actioned": share,
                "affected_count": len(affected),
                "n_actioned": self.n_actioned,
                "n_reviewed": self.n_cases,
                "affected_sample": sorted(affected)[:8],
                "finding": (
                    f"'{label}' is the recommendation for {len(affected)} of the "
                    f"{self.n_actioned} employees receiving one ({share:.0%}), "
                    f"out of {self.n_cases} reviewed. At this concentration the "
                    f"constraint is more likely to be structural — staffing, "
                    f"planning or manager load — than {len(affected)} separate "
                    f"personal situations. Running it as {len(affected)} "
                    f"individual conversations spends the reviewer's whole week "
                    f"treating one cause {len(affected)} times."
                ),
                "addressed_to": "org/people-ops lead, not the individual reviewer",
            })
        return out

    def precedent_for(self, employee_id: str, risk_level: str,
                      confirmed_drivers: list) -> dict:
        """
        Find previously decided cases with the same risk band and driver set.

        A consistency check, not a decision input: if two comparable employees
        received different outcomes, a human should know before signing either.
        """
        if not self._annotate:
            return {}
        signature = (risk_level, frozenset(confirmed_drivers))
        eid = str(employee_id)
        matches = [c for c in self._get_stats()["by_signature"].get(signature, [])
                   if c["employee_id"] != eid]
        if not matches:
            return {}
        outcomes = Counter(c["status"] for c in matches)
        return {
            "comparable_cases": len(matches),
            "prior_outcomes": dict(outcomes),
            "inconsistent": len(outcomes) > 1,
            "sample": sorted(c["employee_id"] for c in matches)[:8],
        }
