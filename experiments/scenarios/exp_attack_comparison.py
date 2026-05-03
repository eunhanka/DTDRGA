"""STEP 8: Attack Strategy Comparison (7 scenarios).

3 attack types x 2 trust conditions + 1 baseline (reuse) = 7 scenarios.
gamma=0.30, n_classes=3, IS alpha=2.0.

A: SybilInflation (dynamic top-10 congested links, re-selects daily)
B: StrategicSybil (fixed structural vulnerability targets)
C: RandomSybil (random 10 links, seed=42)

Run from dtd_attack/:
  python experiments/master_rerun_step8.py
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

ATTACKS = ["bc_deflation", "strategic", "random"]
TRUST_MODES = ["fixed", "dynamic"]

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

    for attack in ATTACKS:
        for trust_mode in TRUST_MODES:
            label = f"{attack} {trust_mode}"
            fname = f"rerun_s8_{attack}_{trust_mode}.json"

            r = run_sim(label, [
                "--mode", "attack",
                "--gamma", "0.3",
                "--trust", trust_mode,
                "--attack", attack,
                "--n-classes", "3",
                "--theta-dtd", "0.004",
                "--is-alpha", "2.0",
                "--baseline-conv", str(BASELINE_CONV),
                "--baseline-all", str(BASELINE_ALL),
            ], fname)

            rows.append({
                "attack": attack,
                "trust_mode": trust_mode,
                "sustained_poatt": r.get("sustained_poatt"),
                "peak_poatt": r.get("peak_poatt"),
                "overshoot": r.get("overshoot"),
                "trust_recovery": r.get("trust_recovery"),
                "trust_end_attack": r.get("trust_end_attack"),
            })

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "rerun_attack_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Print
    print("\n" + "=" * 70)
    print("STEP 8 ATTACK COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Attack':>12s} {'Trust':>8s} {'Sustained':>12s} {'Peak':>10s} {'Overshoot':>10s} {'TrustRec':>10s}")
    print("-" * 65)
    for row in rows:
        print(f"  {row['attack']:>10s} {row['trust_mode']:>8s}"
              f" {row['sustained_poatt']:>12.4f} {row['peak_poatt']:>10.4f}"
              f" {row['overshoot']:>10.4f} {str(row['trust_recovery']):>10s}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(ATTACKS))
        width = 0.35

        # Sustained PoAtt
        ax = axes[0]
        fix_vals = [r["sustained_poatt"] for r in rows if r["trust_mode"] == "fixed"]
        dyn_vals = [r["sustained_poatt"] for r in rows if r["trust_mode"] == "dynamic"]
        ax.bar(x - width/2, fix_vals, width, label="Fixed trust", color="#e74c3c", alpha=0.8)
        ax.bar(x + width/2, dyn_vals, width, label="Dynamic trust", color="#3498db", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([a.capitalize() for a in ATTACKS])
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Sustained PoAtt")
        ax.set_title("Sustained PoAtt by Attack Type")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Peak PoAtt
        ax = axes[1]
        fix_vals = [r["peak_poatt"] for r in rows if r["trust_mode"] == "fixed"]
        dyn_vals = [r["peak_poatt"] for r in rows if r["trust_mode"] == "dynamic"]
        ax.bar(x - width/2, fix_vals, width, label="Fixed trust", color="#e74c3c", alpha=0.8)
        ax.bar(x + width/2, dyn_vals, width, label="Dynamic trust", color="#3498db", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([a.capitalize() for a in ATTACKS])
        ax.set_ylabel("Peak PoAtt")
        ax.set_title("Peak PoAtt by Attack Type")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "rerun_attack_comparison.png"), dpi=200)
        plt.close(fig)
        print(f"\n  Saved: figures/rerun_attack_comparison.png")
    except Exception as e:
        print(f"  Figure failed: {e}")

    total = time.time() - t_start
    print(f"\n  Total Step 8 time: {total:.0f}s ({total/60:.1f} min)")
    print("\n=== STEP 8 COMPLETE ===")


if __name__ == "__main__":
    main()
