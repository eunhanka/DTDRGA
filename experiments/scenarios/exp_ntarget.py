"""N-target sweep: vary number of targeted links.

n_target = [3, 5, 10, 15, 20, 30]
For each: fixed + dynamic trust = 12 scenarios.
gamma=0.30, n_classes=3, IS alpha=2.0.

Run from DTDRGA_github_final/:
  python experiments/scenarios/exp_ntarget.py
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

N_TARGETS = [3, 5, 10, 15, 20, 30]


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

    for nt in N_TARGETS:
        for trust_mode in ["fixed", "dynamic"]:
            label = f"n_target={nt} {trust_mode}"
            fname = f"rerun_ntarget_{nt}_{trust_mode}.json"

            r = run_sim(label, [
                "--mode", "attack",
                "--gamma", "0.3",
                "--trust", trust_mode,
                "--attack", "bc_deflation",
                "--n-classes", "3",
                "--theta-dtd", "0.004",
                "--is-alpha", "2.0",
                "--n-target-links", str(nt),
                "--baseline-conv", str(BASELINE_CONV),
                "--baseline-all", str(BASELINE_ALL),
            ], fname)

            rows.append({
                "n_target": nt,
                "trust_mode": trust_mode,
                "sustained_poatt": r.get("sustained_poatt"),
                "peak_poatt": r.get("peak_poatt"),
                "overshoot": r.get("overshoot"),
                "trust_recovery": r.get("trust_recovery"),
            })

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "rerun_ntarget_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {csv_path}")

    # Print table
    print("\n" + "=" * 80)
    print("N-TARGET SWEEP RESULTS")
    print("=" * 80)
    print(f"{'n_target':>10s} {'trust':>8s} {'Sustained':>12s} {'Peak':>10s} {'Overshoot':>10s}")
    print("-" * 55)
    for row in rows:
        print(f"  {row['n_target']:>8d} {row['trust_mode']:>8s}"
              f" {row['sustained_poatt']:>12.4f} {row['peak_poatt']:>10.4f}"
              f" {row['overshoot']:>10.4f}")

    # Saturation analysis
    fix_rows = [r for r in rows if r["trust_mode"] == "fixed"]
    dyn_rows = [r for r in rows if r["trust_mode"] == "dynamic"]

    print("\n  Saturation analysis (fixed trust):")
    for i in range(1, len(fix_rows)):
        delta = fix_rows[i]["sustained_poatt"] - fix_rows[i-1]["sustained_poatt"]
        print(f"    {fix_rows[i-1]['n_target']}→{fix_rows[i]['n_target']}: delta={delta:+.4f}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        nts_f = [r["n_target"] for r in fix_rows]
        nts_d = [r["n_target"] for r in dyn_rows]
        ax.plot(nts_f, [r["sustained_poatt"] for r in fix_rows],
                "o-", color="#e74c3c", label="Fixed trust")
        ax.plot(nts_d, [r["sustained_poatt"] for r in dyn_rows],
                "s-", color="#3498db", label="Dynamic trust")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Number of target links (N_att)")
        ax.set_ylabel("Sustained PoAtt")
        ax.set_title("Attack Budget Sweep: PoAtt vs N_att")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "rerun_ntarget_sweep.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: figures/rerun_ntarget_sweep.png")
    except Exception as e:
        print(f"  Figure failed: {e}")

    total = time.time() - t_start
    print(f"\n  Total time: {total:.0f}s ({total/60:.1f} min)")
    print("\n=== N-TARGET SWEEP COMPLETE ===")


if __name__ == "__main__":
    main()
