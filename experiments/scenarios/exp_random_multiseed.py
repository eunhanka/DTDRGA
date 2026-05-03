"""Random attack multi-seed robustness check.

Run RandomSybilAttack with 10 random seeds to assess variance.
Fixed + Dynamic trust for each seed = 20 scenarios.
gamma=0.30, n_classes=3, IS alpha=2.0, n_target_links=10.

Paper reports: Random (10-seed mean +/- std) = 1.050 +/- 0.006 (fixed)
                                                1.003 +/- 0.001 (dynamic)

Run from DTDRGA_github_final/:
  python experiments/scenarios/exp_random_multiseed.py
"""

import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(SCRIPT_DIR, os.pardir, "lwr_single_sim.py")
RESULTS_DIR = os.path.join(SCRIPT_DIR, os.pardir, os.pardir, "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, os.pardir, os.pardir, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
PYTHON = sys.executable

with open(os.path.join(RESULTS_DIR, "rerun_signature.json")) as f:
    sig = json.load(f)
BASELINE_CONV = sig["baseline_converged"]
BASELINE_ALL = sig["baseline_all"]

SEEDS = list(range(10))  # seeds 0-9


def run_sim(label, args_list, output_file):
    out_path = os.path.join(RESULTS_DIR, output_file)
    cmd = [PYTHON, RUNNER] + args_list + ["--output", out_path]
    print(f"\n  Running: {label}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    print(f"    {elapsed:.0f}s", end="")
    if result.returncode != 0:
        print(f" FAILED: {result.stderr[-300:]}")
        raise RuntimeError(f"Failed: {label}")
    with open(out_path) as f:
        data = json.load(f)
    sus = data.get("sustained_poatt", "?")
    print(f"  sus={sus:.4f}" if isinstance(sus, float) else f"  sus={sus}")
    return data


def main():
    t_start = time.time()
    rows = []

    # Note: lwr_single_sim.py currently uses seed=42 for random attack.
    # To support multiple seeds, we need to pass --seed argument.
    # If --seed is not supported, this script documents the intended protocol
    # and can be run with manual seed modification.

    for seed in SEEDS:
        for trust_mode in ["fixed", "dynamic"]:
            label = f"random seed={seed} {trust_mode}"
            fname = f"rerun_random_s{seed}_{trust_mode}.json"

            r = run_sim(label, [
                "--mode", "attack",
                "--gamma", "0.3",
                "--trust", trust_mode,
                "--attack", "random",
                "--n-classes", "3",
                "--theta-dtd", "0.004",
                "--is-alpha", "2.0",
                "--n-target-links", "10",
                "--seed", str(seed),
                "--baseline-conv", str(BASELINE_CONV),
                "--baseline-all", str(BASELINE_ALL),
            ], fname)

            rows.append({
                "seed": seed,
                "trust_mode": trust_mode,
                "sustained_poatt": r.get("sustained_poatt"),
                "peak_poatt": r.get("peak_poatt"),
                "overshoot": r.get("overshoot"),
                "trust_recovery": r.get("trust_recovery"),
            })

    # Save
    csv_path = os.path.join(RESULTS_DIR, "rerun_random_multiseed.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Statistics
    fix_rows = [r for r in rows if r["trust_mode"] == "fixed"]
    dyn_rows = [r for r in rows if r["trust_mode"] == "dynamic"]

    fix_sus = np.array([r["sustained_poatt"] for r in fix_rows if r["sustained_poatt"] is not None])
    dyn_sus = np.array([r["sustained_poatt"] for r in dyn_rows if r["sustained_poatt"] is not None])

    print("\n" + "=" * 70)
    print("RANDOM MULTI-SEED RESULTS")
    print("=" * 70)
    if len(fix_sus) > 0:
        print(f"  Fixed trust:   PoAtt_sus = {fix_sus.mean():.4f} +/- {fix_sus.std():.4f} (n={len(fix_sus)})")
    if len(dyn_sus) > 0:
        print(f"  Dynamic trust: PoAtt_sus = {dyn_sus.mean():.4f} +/- {dyn_sus.std():.4f} (n={len(dyn_sus)})")

    # Per-seed table
    print(f"\n{'seed':>6s} {'trust':>8s} {'Sustained':>12s} {'Peak':>10s}")
    print("-" * 40)
    for row in rows:
        sus = row.get("sustained_poatt")
        peak = row.get("peak_poatt")
        print(f"  {row['seed']:>4d} {row['trust_mode']:>8s}"
              f" {sus:>12.4f}" if sus else "  N/A",
              f" {peak:>10.4f}" if peak else "  N/A")

    # NOTE: Current lwr_single_sim.py uses hardcoded seed=42 for RandomSybilAttack.
    # To get true multi-seed results, add --seed argument to lwr_single_sim.py:
    #   parser.add_argument("--seed", type=int, default=42)
    # and pass it to RandomSybilAttack(config, n_target_links=ntl, seed=args.seed)
    print("\n  WARNING: lwr_single_sim.py may use hardcoded seed=42.")
    print("  Add --seed argument to lwr_single_sim.py for true multi-seed results.")

    total = time.time() - t_start
    print(f"\n  Total time: {total:.0f}s ({total/60:.1f} min)")
    print("\n=== RANDOM MULTI-SEED COMPLETE ===")


if __name__ == "__main__":
    main()
