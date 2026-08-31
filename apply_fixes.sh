#!/usr/bin/env bash
# Applies the six pending .new files, reruns the full reproduction, and REFUSES
# to continue if the result does not match what README.md claims.
#
# Why this exists: README.md and CHANGELOG.md were updated to the audited
# numbers (BASELINE+ 68.4%, margin +24.6pp) and pushed, but the code and
# evidence that produce those numbers were left as .new files. The repo
# currently claims one thing and computes another, which is the single worst
# state to submit in.
#
#   bash apply_fixes.sh
set -e

cd "$(dirname "$0")"

echo "==> applying pending fixes"
applied=0
for f in data/prepare_data.py \
         baseline/train.py \
         baseline/pipeline_plus.py \
         evidence/evaluation_report.json \
         evidence/baseline_plus_case_results.csv \
         evidence/baseline_plus_case_scores.csv; do
  if [ -f "$f.new" ]; then
    mv "$f.new" "$f"
    echo "    $f"
    applied=$((applied + 1))
  fi
done
echo "    ($applied file(s) applied)"

echo ""
echo "==> checking the fix is actually in place"
if ! grep -q "FEATURE_LEVERS" baseline/pipeline_plus.py; then
  echo "FAILED: baseline/pipeline_plus.py is still the old version." >&2
  exit 1
fi
if ! grep -q "FEATURES contains excluded column" data/prepare_data.py; then
  echo "FAILED: data/prepare_data.py is still the old version." >&2
  exit 1
fi
echo "    ok"

echo ""
echo "==> full reproduction from scratch"
./run_all.sh > /tmp/apply_fixes_run.log 2>&1 || {
  echo "FAILED: run_all.sh exited non-zero. Full log: /tmp/apply_fixes_run.log" >&2
  tail -30 /tmp/apply_fixes_run.log >&2
  exit 1
}

echo ""
echo "==> verifying the numbers match what README.md claims"
fail=0
check() {  # check <label> <pattern> <file>
  if grep -q "$2" "$3"; then
    echo "    ok    $1"
  else
    echo "    WRONG $1  (expected to find: $2)" >&2
    fail=1
  fi
}
check "BASELINE+ 68.4% in the run output" "68.4%" /tmp/apply_fixes_run.log
check "AGENT 93.0% in the run output"     "93.0%" /tmp/apply_fixes_run.log
check "tests passed"                      "passed" /tmp/apply_fixes_run.log
check "README claims 68.4%"               "68.4%" README.md
check "README claims +24.6pp"             "24.6pp" README.md

if [ "$fail" -ne 0 ]; then
  echo "" >&2
  echo "The code and the README still disagree. Do NOT push." >&2
  echo "Send the output above and /tmp/apply_fixes_run.log for diagnosis." >&2
  exit 1
fi

echo ""
grep -A14 "THREE SYSTEMS" /tmp/apply_fixes_run.log | head -16
grep -E "passed" /tmp/apply_fixes_run.log | tail -1

echo ""
echo "==> checking nothing secret is about to be committed"
if git status --short 2>/dev/null | grep -qE '(^|/)\.env$'; then
  echo "STOP: .env is staged or untracked-and-visible to git. Do not push." >&2
  exit 1
fi
echo "    ok — .env is not tracked"

echo ""
echo "Everything agrees. Push with:"
echo ""
echo "  git add -A && git commit -m 'Strengthen BASELINE+ after audit; margin +29.8pp -> +24.6pp' && git push"
echo ""
echo "Then delete this script: rm apply_fixes.sh"
