"""STEP 10: Smooth Model Gamma Sweep (8 scenarios).

Same gamma values as Step 2: [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
smooth_mode=True, eta=10.0. Dynamic trust only.

KNOWN ISSUE: With eta=10.0, mean|guidance_error| ~ 1500s during attack,
so xi = exp(-10 * 1500) = exp(-15000) ~ 0. This makes smooth model
identical to threshold model.

Run from dtd_attack/:
  python experiments/master_rerun_step10.py
"""

import json
import os
import subprocess
import sys
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(SCRIPT_DIR, "lwr_single_sim.py")
RESULTS_DIR = os.path.join(SCRIPT_DIR, os.pardir, "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, os.pardir, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
PYTHON = sys.executable

with open(os.path.join(RESULTS_DIR, "rerun_signature.json")) as f:
    sig = json.load(f)
BASELINE_CONV = sig["baseline_converged"]
BASELINE_ALL = sig["baseline_all"]

GAMMAS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

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
    smooth_rows = []

    for gamma in GAMMAS:
        label = f"smooth g={gamma}"
        fname = f"rerun_s10_smooth_g{gamma}.json"

        r = run_sim(label, [
            "--mode", "attack",
            "--gamma", str(gamma),
            "--trust", "dynamic",
            "--attack", "sybil",
            "--n-classes", "3",
            "--theta-dtd", "0.004",
            "--is-alpha", "2.0",
            "--smooth-mode",
            "--eta", "10.0",
            "--baseline-conv", str(BASELINE_CONV),
            "--baseline-all", str(BASELINE_ALL),
        ], fname)

        smooth_rows.append({
            "gamma": gamma,
            "sustained_poatt": r.get("sustained_poatt"),
            "peak_poatt": r.get("peak_poatt"),
            "overshoot": r.get("overshoot"),
            "trust_recovery": r.get("trust_recovery"),
            "trust_end_attack": r.get("trust_end_attack"),
        })

    # Save
    with open(os.path.join(RESULTS_DIR, "rerun_gamma_smooth.json"), "w") as f:
        json.dump(smooth_rows, f, indent=2)

    # Load threshold results for comparison
    try:
        import csv
        threshold_rows = []
        with open(os.path.join(RESULTS_DIR, "rerun_gamma_sweep.csv")) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["trust_mode"] == "dynamic":
                    threshold_rows.append({
                        "gamma": float(row["gamma"]),
                        "sustained_poatt": float(row["sustained_poatt"]),
                        "trust_recovery": int(row["trust_recovery"]) if row["trust_recovery"] else None,
                    })
    except Exception:
        threshold_rows = []

    # Print
    print("\n" + "=" * 70)
    print("STEP 10 SMOOTH MODEL RESULTS")
    print("=" * 70)
    print(f"{'gamma':>8s} {'Smooth Sus':>12s} {'Thresh Sus':>12s} {'Smooth TRec':>12s} {'Thresh TRec':>12s} {'Same?':>8s}")
    print("-" * 70)

    all_same = True
    for sr in smooth_rows:
        g = sr["gamma"]
        tr = next((t for t in threshold_rows if abs(t["gamma"] - g) < 0.01), None)
        tr_sus = f"{tr['sustained_poatt']:.4f}" if tr else "N/A"
        tr_rec = str(tr.get("trust_recovery", "N/A")) if tr else "N/A"
        same = "YES" if tr and abs(sr["sustained_poatt"] - tr["sustained_poatt"]) < 0.001 else "NO"
        if same != "YES":
            all_same = False
        print(f"  {g:>6.2f} {sr['sustained_poatt']:>12.4f} {tr_sus:>12s}"
              f" {str(sr['trust_recovery']):>12s} {tr_rec:>12s} {same:>8s}")

    if all_same:
        print("\n  NOTE: eta=10 causes saturation; meaningful smooth behavior requires eta~0.001")
        print("  With eta=10, exp(-10*|error|) -> 0 during attack, making smooth = threshold.")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        gammas_s = [r["gamma"] for r in smooth_rows]
        ax.plot(gammas_s, [r["sustained_poatt"] for r in smooth_rows],
                "s-", color="#e74c3c", lw=2, label="Smooth (eta=10)")
        if threshold_rows:
            gammas_t = [r["gamma"] for r in threshold_rows]
            ax.plot(gammas_t, [r["sustained_poatt"] for r in threshold_rows],
                    "o--", color="#3498db", lw=2, label="Threshold")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("gamma")
        ax.set_ylabel("Sustained PoAtt")
        ax.set_title("Smooth vs Threshold Model (eta=10)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "rerun_gamma_smooth_vs_threshold.png"), dpi=200)
        plt.close(fig)
        print(f"\n  Saved: figures/rerun_gamma_smooth_vs_threshold.png")
    except Exception as e:
        print(f"  Figure failed: {e}")

    total = time.time() - t_start
    print(f"\n  Total Step 10 time: {total:.0f}s ({total/60:.1f} min)")
    print("\n=== STEP 10 COMPLETE ===")


if __name__ == "__main__":
    main()
