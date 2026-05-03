"""STEP 3: BR Sweep -- BOTH Fixed AND Dynamic Trust (12 scenarios).

delta = [0, 100, 200, 300, 400, 500]
For EACH delta: Fixed trust + Dynamic trust.
n_classes=3 for ALL scenarios.
When sweeping delta, override indifference_band for ALL classes to same delta.

Run from dtd_attack/:
  python experiments/master_rerun_step3.py
"""

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

DELTAS = [0, 100, 200, 300, 400, 500]

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

    for delta in DELTAS:
        for trust_mode in ["fixed", "dynamic"]:
            label = f"delta={delta} {trust_mode}"
            fname = f"rerun_s3_d{delta}_{trust_mode}.json"

            r = run_sim(label, [
                "--mode", "attack",
                "--gamma", "0.3",
                "--trust", trust_mode,
                "--attack", "bc_deflation",
                "--n-classes", "3",
                "--theta-dtd", "0.004",
                "--is-alpha", "2.0",
                "--delta-all", str(delta),
                "--baseline-conv", str(BASELINE_CONV),
                "--baseline-all", str(BASELINE_ALL),
            ], fname)

            rows.append({
                "delta": delta,
                "trust_mode": trust_mode,
                "sustained_poatt": r.get("sustained_poatt"),
                "peak_poatt": r.get("peak_poatt"),
                "overshoot": r.get("overshoot"),
                "trust_recovery": r.get("trust_recovery"),
            })

    # Save
    with open(os.path.join(RESULTS_DIR, "rerun_br_sweep.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # Print results
    print("\n" + "=" * 70)
    print("STEP 3 BR SWEEP RESULTS")
    print("=" * 70)
    print(f"{'delta':>8s} {'trust':>8s} {'Sustained':>12s} {'Peak':>10s} {'Overshoot':>10s} {'TrustRec':>10s}")
    print("-" * 60)
    for row in rows:
        print(f"  {row['delta']:>6d} {row['trust_mode']:>8s}"
              f" {row['sustained_poatt']:>12.4f} {row['peak_poatt']:>10.4f}"
              f" {row['overshoot']:>10.4f} {str(row['trust_recovery']):>10s}")

    # Crossing points
    fix_rows = [r for r in rows if r["trust_mode"] == "fixed"]
    dyn_rows = [r for r in rows if r["trust_mode"] == "dynamic"]
    for label, rr in [("Fixed", fix_rows), ("Dynamic", dyn_rows)]:
        for i in range(len(rr) - 1):
            if (rr[i]["sustained_poatt"] - 1.0) * (rr[i+1]["sustained_poatt"] - 1.0) < 0:
                d1, d2 = rr[i]["delta"], rr[i+1]["delta"]
                p1, p2 = rr[i]["sustained_poatt"], rr[i+1]["sustained_poatt"]
                cross = d1 + (1.0 - p1) / (p2 - p1) * (d2 - d1)
                print(f"\n  {label} trust PoAtt crosses 1.0 at delta ~ {cross:.0f}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot([r["delta"] for r in fix_rows], [r["sustained_poatt"] for r in fix_rows],
                "o-", color="#e74c3c", label="Fixed trust")
        ax.plot([r["delta"] for r in dyn_rows], [r["sustained_poatt"] for r in dyn_rows],
                "s-", color="#3498db", label="Dynamic trust")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Indifference Band (delta) [s]")
        ax.set_ylabel("Sustained PoAtt")
        ax.set_title("BR Sweep: Sustained PoAtt vs delta")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "rerun_br_sweep.png"), dpi=200)
        plt.close(fig)
        print(f"\n  Saved: figures/rerun_br_sweep.png")
    except Exception as e:
        print(f"  Figure failed: {e}")

    total = time.time() - t_start
    print(f"\n  Total Step 3 time: {total:.0f}s ({total/60:.1f} min)")
    print("\n=== STEP 3 COMPLETE ===")


if __name__ == "__main__":
    main()
