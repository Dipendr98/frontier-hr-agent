"""
Does this hold up on a large cohort?

    python evaluation/benchmark_scale.py                 # up to 20k, ~2 min
    python evaluation/benchmark_scale.py --max 80000     # ~13 min

Synthesises larger cohorts by sampling the real prepared data with replacement,
so the feature distributions stay realistic, and times a full two-pass triage at
each size. Prints ms/case, which is the number that matters: if it is flat the
pipeline is linear, and if it climbs with n something inside is scanning the
whole cohort once per employee.

That is not a hypothetical failure mode — it is what this file was written to
catch, after a profile of an 8,000-employee run showed 96 million
`Counter.update` calls. Three separate O(n)-per-case scans were hiding behind a
342-row dataset where nothing is slow enough to notice:

  1. cohort memory recomputing driver prevalence on every case
  2. precedent lookup scanning every stored case on every case
  3. `get_cohort_percentile` scanning the full column on every call

Fixed by caching the rollup with write invalidation, indexing precedent by
(risk band, driver set), and replacing the percentile scan with a binary search
over a presorted column. Measured on an M-series laptop, Python 3.14:

    cohort     before      after
       20000     417 s      103 s
       40000     655 s      212 s
       80000       —        431 s   (5.4 ms/case, flat)

No API keys are used. Results are deterministic apart from wall clock.
"""
import argparse
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pandas as pd  # noqa: E402

from advanced import llm  # noqa: E402
from advanced.orchestration.cohort import run_cohort  # noqa: E402

EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
SEED = 1


def synth(base: pd.DataFrame, n: int, path: str) -> str:
    """
    Sample with replacement to size n, with fresh employee ids.

    Resampling rather than generating keeps every marginal distribution and
    every feature correlation exactly as the real data has them, so the timing
    reflects work the pipeline would actually do. Duplicated rows are fine here:
    this measures throughput, not model quality.
    """
    df = base.sample(n=n, replace=True, random_state=SEED).reset_index(drop=True)
    df["employee_id"] = [f"E{i}" for i in range(n)]
    df.to_csv(path, index=False)
    return path


def main(max_n: int, keep: bool):
    llm.set_narration(False)          # the prose layer is not what we are timing
    base = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))

    sizes = [n for n in (len(base), 5000, 10000, 20000, 40000, 80000, 160000)
             if n <= max_n]
    print(f"Two-pass cohort triage, no API keys. Base cohort: {len(base)} rows.\n")
    print(f"{'cohort':>9} {'seconds':>9} {'ms/case':>9} {'escalated':>10}")

    rows = []
    for n in sizes:
        path = os.path.join("/tmp", f"frontier_bench_{n}.csv")
        synth(base, n, path)
        t0 = time.time()
        report = run_cohort(data_path=path, use_memory=True, save=False)
        elapsed = time.time() - t0
        per = elapsed / n * 1000
        rows.append({"n": n, "seconds": round(elapsed, 1), "ms_per_case": round(per, 2),
                     "status_counts": report["status_counts"]})
        print(f"{n:>9} {elapsed:>9.1f} {per:>9.2f} "
              f"{report['status_counts'].get('ESCALATED', 0):>10}")
        if not keep:
            os.remove(path)

    # Flat ms/case is the pass condition. Compare the largest run against the
    # smallest multi-thousand one, ignoring the base cohort where fixed startup
    # cost dominates and inflates the per-case figure.
    scaled = [r for r in rows if r["n"] >= 5000]
    verdict = None
    if len(scaled) >= 2:
        ratio = scaled[-1]["ms_per_case"] / scaled[0]["ms_per_case"]
        verdict = {
            "smallest": scaled[0]["n"], "largest": scaled[-1]["n"],
            "ms_per_case_ratio": round(ratio, 2),
            "linear": ratio < 1.5,
        }
        print()
        if verdict["linear"]:
            print(f"LINEAR: ms/case changed {ratio:.2f}x while the cohort grew "
                  f"{scaled[-1]['n'] // scaled[0]['n']}x.")
        else:
            print(f"SUPERLINEAR: ms/case grew {ratio:.2f}x. Something is "
                  f"scanning the cohort once per employee — profile it.")

    out = {"base_rows": len(base), "seed": SEED, "results": rows, "verdict": verdict}
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(os.path.join(EVIDENCE_DIR, "scale_benchmark.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nevidence/scale_benchmark.json written")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=20000, dest="max_n",
                    help="largest cohort to test (default 20000, ~2 min)")
    ap.add_argument("--keep", action="store_true", help="keep the synthetic CSVs")
    args = ap.parse_args()
    main(args.max_n, args.keep)
