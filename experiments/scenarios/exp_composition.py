"""STEP 6: Composition Sweep -- Dynamic Trust (6 scenarios).

CAV penetration: [0%, 10%, 25%, 50%, 75%, 100%]
Dynamic trust only. n_classes=3.

Run from dtd_attack/:
  python experiments/master_rerun_step6.py
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

CAV_FRACS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]

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

    for cav in CAV_FRACS:
        label = f"CAV={cav*100:.0f}%"
        fname = f"rerun_s6_cav{cav:.2f}.json"

        r = run_sim(label, [
            "--mode", "attack",
            "--gamma", "0.3",
            "--trust", "dynamic",
            "--attack", "bc_deflation",
            "--n-classes", "3",
            "--theta-dtd", "0.004",
            "--is-alpha", "2.0",
            "--cav-frac", str(cav),
            "--baseline-conv", str(BASELINE_CONV),
            "--baseline-all", str(BASELINE_ALL),
        ], fname)

        # Compute chi(pi)
        if cav >= 1.0:
            fracs = np.array([1.0, 0.0, 0.0])
        elif cav <= 0.0:
            fracs = np.array([0.0, 0.70, 0.30])
        else:
            fracs = np.array([cav, (1-cav)*0.70, (1-cav)*0.30])
        fracs = fracs / fracs.sum()

        theta_k = 0.004  # common θ across all classes (paper §3.4)
        lambda_max = np.array([0.90, 0.70, 0.30])
        chi = float(np.sum(fracs * theta_k * lambda_max))

        rows.append({
            "cav_frac": cav,
            "chi": chi,
            "sustained_poatt": r.get("sustained_poatt"),
            "peak_poatt": r.get("peak_poatt"),
            "overshoot": r.get("overshoot"),
            "trust_recovery": r.get("trust_recovery"),
        })

    # Save
    with open(os.path.join(RESULTS_DIR, "rerun_composition.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # Print
    print("\n" + "=" * 70)
    print("STEP 6 COMPOSITION SWEEP RESULTS")
    print("=" * 70)
    print(f"{'CAV%':>8s} {'chi':>8s} {'Sustained':>12s} {'Peak':>10s} {'Overshoot':>10s} {'TrustRec':>10s}")
    print("-" * 60)
    for row in rows:
        print(f"  {row['cav_frac']*100:>5.0f}% {row['chi']:>8.4f}"
              f" {row['sustained_poatt']:>12.4f} {row['peak_poatt']:>10.4f}"
              f" {row['overshoot']:>10.4f} {str(row['trust_recovery']):>10s}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot([r["cav_frac"]*100 for r in rows], [r["sustained_poatt"] for r in rows],
                "o-", color="#8e44ad", lw=2)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("CAV Penetration (%)")
        ax.set_ylabel("Sustained PoAtt")
        ax.set_title("PoAtt vs CAV%")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot([r["chi"] for r in rows], [r["sustained_poatt"] for r in rows],
                "o-", color="#e67e22", lw=2)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("chi(pi) = sum(pi_k * theta_k * lambda_max_k)")
        ax.set_ylabel("Sustained PoAtt")
        ax.set_title("PoAtt vs chi (susceptibility)")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "rerun_composition.png"), dpi=200)
        plt.close(fig)
        print(f"\n  Saved: figures/rerun_composition.png")
    except Exception as e:
        print(f"  Figure failed: {e}")

    total = time.time() - t_start
    print(f"\n  Total Step 6 time: {total:.0f}s ({total/60:.1f} min)")
    print("\n=== STEP 6 COMPLETE ===")


if __name__ == "__main__":
    main()
