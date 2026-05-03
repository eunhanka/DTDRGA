"""STEP 9: Anaheim (3 scenarios).

Anaheim network, 200 days, 3 classes, gamma=0.30.
Uses load_anaheim() and LWR via subprocess.

3 scenarios: baseline, attack+fixed trust, attack+dynamic trust.

Run from dtd_attack/:
  python experiments/master_rerun_step9.py
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

# Load Sioux Falls baseline for cross-network comparison
with open(os.path.join(RESULTS_DIR, "rerun_signature.json")) as f:
    sig = json.load(f)
SF_BASELINE_CONV = sig["baseline_converged"]

def run_sim(label, args_list, output_file):
    out_path = os.path.join(RESULTS_DIR, output_file)
    cmd = [PYTHON, RUNNER] + args_list + ["--output", out_path]
    print(f"\n  Running: {label}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    print(f"    {elapsed:.0f}s ({elapsed/60:.1f} min)", end="")
    if result.returncode != 0:
        print(f" FAILED: {result.stderr[-500:]}")
        raise RuntimeError(f"Failed: {label}")
    with open(out_path) as f:
        data = json.load(f)
    sus = data.get("sustained_poatt", "?")
    print(f"  sus={sus:.4f}" if isinstance(sus, float) else f"  sus={sus}")
    return data


def main():
    t_start = time.time()
    print("Anaheim LWR: dt=180s, all links discretized to FFT_mod=180s")
    print("(99.3% of Anaheim links have FFT < 180s)")

    # Scenario 1: Baseline
    r1 = run_sim("Anaheim Baseline", [
        "--mode", "baseline",
        "--network", "anaheim",
        "--n-classes", "3",
        "--theta-dtd", "0.004",
        "--is-alpha", "2.0",
    ], "rerun_s9_anaheim_baseline.json")

    ana_baseline = float(np.mean(r1["tstt"][29:50]))
    ana_baseline_all = float(np.mean(r1["tstt"][:50]))
    print(f"\n  Anaheim baseline_converged: {ana_baseline:.2f}")

    # Scenario 2: Attack + Fixed trust
    r2 = run_sim("Anaheim Sybil Fixed", [
        "--mode", "attack",
        "--network", "anaheim",
        "--gamma", "0.3",
        "--trust", "fixed",
        "--attack", "bc_deflation",
        "--n-classes", "3",
        "--theta-dtd", "0.004",
        "--is-alpha", "2.0",
        "--n-target-links", "10",
        "--baseline-conv", str(ana_baseline),
        "--baseline-all", str(ana_baseline_all),
    ], "rerun_s9_anaheim_fixed.json")

    # Scenario 3: Attack + Dynamic trust
    r3 = run_sim("Anaheim Sybil Dynamic", [
        "--mode", "attack",
        "--network", "anaheim",
        "--gamma", "0.3",
        "--trust", "dynamic",
        "--attack", "bc_deflation",
        "--n-classes", "3",
        "--theta-dtd", "0.004",
        "--is-alpha", "2.0",
        "--n-target-links", "10",
        "--baseline-conv", str(ana_baseline),
        "--baseline-all", str(ana_baseline_all),
    ], "rerun_s9_anaheim_dynamic.json")

    # Save combined results
    result = {
        "anaheim_baseline_converged": ana_baseline,
        "sf_baseline_converged": SF_BASELINE_CONV,
        "fixed": {
            "sustained_poatt": r2.get("sustained_poatt"),
            "peak_poatt": r2.get("peak_poatt"),
            "overshoot": r2.get("overshoot"),
            "trust_recovery": r2.get("trust_recovery"),
        },
        "dynamic": {
            "sustained_poatt": r3.get("sustained_poatt"),
            "peak_poatt": r3.get("peak_poatt"),
            "overshoot": r3.get("overshoot"),
            "trust_recovery": r3.get("trust_recovery"),
        },
    }
    with open(os.path.join(RESULTS_DIR, "rerun_anaheim.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Get SF results for comparison
    sf_fixed_path = os.path.join(RESULTS_DIR, "rerun_s1_sybil_fixed.json")
    sf_dynamic_path = os.path.join(RESULTS_DIR, "rerun_s1_sybil_dynamic.json")
    with open(sf_fixed_path) as f:
        sf_fixed = json.load(f)
    with open(sf_dynamic_path) as f:
        sf_dynamic = json.load(f)

    # Print comparison
    print("\n" + "=" * 70)
    print("STEP 9 CROSS-NETWORK COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25s} {'Sioux Falls':>15s} {'Anaheim':>15s}")
    print("-" * 55)
    print(f"  {'Baseline TSTT':<23s} {SF_BASELINE_CONV:>15.2f} {ana_baseline:>15.2f}")
    print(f"  {'PoAtt (Fixed)':<23s} {sf_fixed.get('sustained_poatt', 0):>15.4f} {r2.get('sustained_poatt', 0):>15.4f}")
    print(f"  {'PoAtt (Dynamic)':<23s} {sf_dynamic.get('sustained_poatt', 0):>15.4f} {r3.get('sustained_poatt', 0):>15.4f}")
    print(f"  {'Peak (Fixed)':<23s} {sf_fixed.get('peak_poatt', 0):>15.4f} {r2.get('peak_poatt', 0):>15.4f}")
    print(f"  {'Peak (Dynamic)':<23s} {sf_dynamic.get('peak_poatt', 0):>15.4f} {r3.get('peak_poatt', 0):>15.4f}")
    print(f"  {'Trust Recovery':<23s} {sf_dynamic.get('trust_recovery', 'N/A'):>15s} {str(r3.get('trust_recovery', 'N/A')):>15s}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = np.arange(1, 201)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(days, r1["tstt"], "k--", lw=1, alpha=0.5, label="Baseline")
        ax.plot(days, r2["tstt"], color="#e67e22", lw=1.2, label="Fixed trust")
        ax.plot(days, r3["tstt"], color="#8e44ad", lw=1.2, label="Dynamic trust")
        ax.axvspan(51, 100, alpha=0.08, color="red")
        ax.set_title("(a) Anaheim TSTT (LWR)")
        ax.set_ylabel("TSTT [veh-hours]")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        trust3 = np.array(r3["trust"])
        colors = ["#e74c3c", "#3498db", "#2ecc71"]
        names = ["CAV", "App", "Exp"]
        for k in range(3):
            ax.plot(days, trust3[:, k], color=colors[k], lw=1.2, label=names[k])
        ax.axvspan(51, 100, alpha=0.08, color="red")
        ax.set_title("(b) Trust Evolution (Dynamic)")
        ax.set_ylabel("Expected Trust")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        ax = axes[1, 0]
        x = np.arange(2)
        width = 0.35
        sf_vals = [sf_fixed.get("sustained_poatt", 1), sf_dynamic.get("sustained_poatt", 1)]
        ana_vals = [r2.get("sustained_poatt", 1), r3.get("sustained_poatt", 1)]
        ax.bar(x - width/2, sf_vals, width, label="Sioux Falls", color="#3498db", alpha=0.8)
        ax.bar(x + width/2, ana_vals, width, label="Anaheim", color="#e74c3c", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Fixed trust", "Dynamic trust"])
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Sustained PoAtt")
        ax.set_title("(c) Cross-network comparison")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1, 1]
        ax.axis("off")
        table_data = [
            ["Baseline TSTT", f"{SF_BASELINE_CONV:.1f}", f"{ana_baseline:.1f}"],
            ["PoAtt (Fixed)", f"{sf_fixed.get('sustained_poatt', 0):.4f}", f"{r2.get('sustained_poatt', 0):.4f}"],
            ["PoAtt (Dynamic)", f"{sf_dynamic.get('sustained_poatt', 0):.4f}", f"{r3.get('sustained_poatt', 0):.4f}"],
            ["Trust Recovery", str(sf_dynamic.get("trust_recovery")), str(r3.get("trust_recovery"))],
        ]
        table = ax.table(cellText=table_data,
                         colLabels=["Metric", "Sioux Falls", "Anaheim"],
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        ax.set_title("(d) Metrics", fontsize=12, pad=20)

        fig.suptitle("Anaheim Network (LWR, g=0.3)", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "rerun_anaheim.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: figures/rerun_anaheim.png")
    except Exception as e:
        print(f"  Figure failed: {e}")

    total = time.time() - t_start
    print(f"\n  Total Step 9 time: {total:.0f}s ({total/60:.1f} min)")
    print("\n=== STEP 9 COMPLETE ===")


if __name__ == "__main__":
    main()
