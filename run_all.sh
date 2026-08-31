#!/usr/bin/env bash
# Full reproduction from a clean environment. ~60 seconds, no API keys, no cost.
#
# Everything below runs Mode 1, the deterministic pipeline, which needs no
# provider. Modes 2 and 3 require a key and are NOT part of this script — see
# the README section "Modes 2 and 3" to run those.
#
# Uses a virtual environment because many modern Linux distributions ship
# PEP 668 "externally managed" Python, where a bare `pip install` fails.
set -e

PY=python3
VENV=.venv

if [ ! -d "$VENV" ]; then
  echo "==> creating virtual environment ($VENV)"
  $PY -m venv $VENV
fi
# shellcheck disable=SC1091
source $VENV/bin/activate

echo "==> installing dependencies"
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Off for the whole script: the prose layer costs one completion per agent per
# case and changes no decision, so the reported numbers must not depend on it.
export LLM_NARRATE=0

echo ""
echo "==> preparing data (IBM HR Analytics, onboarding window)"
python data/prepare_data.py

echo ""
echo "==> training the attrition model"
python baseline/train.py

echo ""
echo "==> measuring the cost of excluding protected attributes"
python baseline/fairness_cost.py

echo ""
echo "==> BASELINE vs BASELINE+ vs AGENT on identical held-out cases"
python evaluation/evaluate.py

echo ""
echo "==> evidence threshold experiment"
python evaluation/threshold_sweep.py

echo ""
echo "==> tests (live-agent tests excluded — see below)"
# Excluded on purpose. This script is the no-key reproduction path and claims
# ~60 seconds; the seven live tests each drive a full agent loop against a real
# provider, so on a machine that happens to have a key in .env the same script
# takes ten minutes and looks like a hang. No reported number depends on a
# provider, so this script must not either. Run them deliberately:
#   python -m pytest tests/test_llm_agent.py -q
FRONTIER_SKIP_LIVE=1 python -m pytest tests/ -q

echo ""
echo "==> cohort triage: the artifact a reviewer actually receives"
python -m advanced.orchestration.cohort

echo ""
echo "==> one agent case, printed step by step"
python advanced/orchestration/workflow.py EMP4

# After the single-case demo, not before: the demo writes its own
# trajectories/EMP4.json, and generate.py clears the directory first so the
# committed set is exactly the representative one rather than that plus
# whatever was last run by hand.
echo ""
echo "==> representative trajectories (add --llm for Modes 2 and 3)"
python trajectories/generate.py

echo ""
echo "==> provider self-test (reports Modes 2/3 as unavailable with no key)"
python -m advanced.doctor || true

echo ""
echo "Done."
echo "  evidence/evaluation_report.json  the headline comparison"
echo "  evidence/cohort_briefing.md      the reviewer-facing briefing"
echo "  trajectories/                    per-agent traces"
