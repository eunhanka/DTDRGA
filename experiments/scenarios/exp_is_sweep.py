"""STEP 5: IS Sweep -- BOTH Fixed AND Dynamic Trust (8 scenarios).

alpha = [0, 1, 2, 3]
For EACH alpha: Fixed trust + Dynamic trust.
n_classes=3 for all.

Run from dtd_attack/:
  python experiments/master_rerun_step5.py
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

ALPHAS = [0, 1, 2, 3]

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

    for alpha in ALPHAS:
        for trust_mode in ["fixed", "dynamic"]:
            label = f"alpha={alpha} {trust_mode}"
            fname = f"rerun_s5_a{alpha}_{trust_mode}.json"

            r = run_sim(label, [
                "--mode", "attack",
                "--gamma", "0.3",
                "--trust", trust_mode,
                "--attack", "bc_deflation",
                "--n-classes", "3",
                "--theta-dtd", "0.004",
                "--is-alpha", str(alpha),
                "--baseline-conv", str(BASELINE_CONV),
                "--baseline-all", str(BASELINE_ALL),
            ], fname)

            rows.append({
                "alpha": alpha,
                "trust_mode": trust_mode,
                "sustained_poatt": r.get("sustained_poatt"),
                "peak_poatt": r.get("peak_poatt"),
                "overshoot": r.get("overshoot"),
                "trust_recovery": r.get("trust_recovery"),
            })

    # Save
    with open(os.path.join(RESULTS_DIR, "rerun_is_sweep.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # Print
    print("\n" + "=" * 70)
    print("STEP 5 IS SWEEP RESULTS")
    print("=" * 70)
    print(f"{'alpha':>8s} {'trust':>8s} {'Sustained':>12s} {'Peak':>10s} {'Overshoot':>10s} {'TrustRec':>10s}")
    print("-" * 60)
    for row in rows:
        print(f"  {row['alpha']:>6d} {row['trust_mode']:>8s}"
              f" {row['sustained_poatt']:>12.4f} {row['peak_poatt']:>10.4f}"
              f" {row['overshoot']:>10.4f} {str(row['trust_recovery']):>10s}")

    # Defense redundancy
    fix_rows = [r for r in rows if r["trust_mode"] == "fixed"]
    dyn_rows = [r for r in rows if r["trust_mode"] == "dynamic"]

    delta_IS_fixed = fix_rows[0]["sustained_poatt"] - fix_rows[-1]["sustained_poatt"]
    delta_IS_dynamic = dyn_rows[0]["sustained_poatt"] - dyn_rows[-1]["sustained_poatt"]
    bound = (12.0 / 50.0) * delta_IS_fixed  # n*=12, D=50

    print(f"\n  Defense redundancy:")
    print(f"    Delta_IS_fixed  = {delta_IS_fixed:.4f}")
    print(f"    Delta_IS_dynamic = {delta_IS_dynamic:.4f}")
    print(f"    Bound (n*/D * Delta_IS_fixed) = {bound:.4f}")
    print(f"    Check: {delta_IS_dynamic:.4f} <= {bound:.4f} ? {'YES' if delta_IS_dynamic <= bound + 0.001 else 'NO'}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot([r["alpha"] for r in fix_rows], [r["sustained_poatt"] for r in fix_rows],
                "o-", color="#e74c3c", label="Fixed trust")
        ax.plot([r["alpha"] for r in dyn_rows], [r["sustained_poatt"] for r in dyn_rows],
                "s-", color="#3498db", label="Dynamic trust")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("IS alpha")
        ax.set_ylabel("Sustained PoAtt")
        ax.set_title("IS Sweep: Sustained PoAtt vs alpha")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "rerun_is_sweep.png"), dpi=200)
        plt.close(fig)
        print(f"\n  Saved: figures/rerun_is_sweep.png")
    except Exception as e:
        print(f"  Figure failed: {e}")

    total = time.time() - t_start
    print(f"\n  Total Step 5 time: {total:.0f}s ({total/60:.1f} min)")
    print("\n=== STEP 5 COMPLETE ===")


if __name__ == "__main__":
    main()
