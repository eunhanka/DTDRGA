"""STEP 1: Baseline + Signature Result (5 scenarios).

Sioux Falls, LWRDNL, 200 days each.
Uses subprocess runner pattern (lwr_single_sim.py).

Scenarios:
1. No attack baseline
2. Physical disruption (link 68 degraded, days 51-100)
3. Sybil gamma=0.30, Fixed trust, days 51-100, top-10 links
4. Sybil gamma=0.30, Dynamic trust
5. Sybil gamma=1.0, Dynamic trust

Run from dtd_attack/:
  python experiments/master_rerun_step1.py
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

def run_sim(label, args_list, output_file):
    """Run a simulation via subprocess and return parsed JSON."""
    out_path = os.path.join(RESULTS_DIR, output_file)
    cmd = [PYTHON, RUNNER] + args_list + ["--output", out_path]
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"  cmd: {' '.join(cmd[-8:])}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"Simulation failed: {label}")
    with open(out_path) as f:
        data = json.load(f)
    return data


def main():
    t_start = time.time()
    results = {}

    # ============================================================
    # Scenario 1: Baseline (no attack)
    # ============================================================
    r1 = run_sim("Scenario 1: No-attack baseline", [
        "--mode", "baseline",
        "--n-classes", "3",
        "--theta-dtd", "0.004",
        "--is-alpha", "2.0",
    ], "rerun_s1_baseline.json")

    baseline_converged = float(np.mean(r1["tstt"][29:50]))
    baseline_all = float(np.mean(r1["tstt"][:50]))
    print(f"\n  baseline_converged (mean TSTT[30:50]): {baseline_converged:.2f}")
    print(f"  baseline_all (mean TSTT[:50]): {baseline_all:.2f}")
    results["baseline"] = {
        "baseline_converged": baseline_converged,
        "baseline_all": baseline_all,
        "tstt": r1["tstt"],
    }

    bc_str = f"{baseline_converged}"
    ba_str = f"{baseline_all}"

    # ============================================================
    # Scenario 2: Physical disruption
    # ============================================================
    r2 = run_sim("Scenario 2: Physical disruption (link 68)", [
        "--mode", "disruption",
        "--theta-dtd", "0.004",
        "--baseline-conv", bc_str,
        "--baseline-all", ba_str,
    ], "rerun_s1_disruption.json")
    results["disruption"] = r2

    # ============================================================
    # Scenario 3: Sybil gamma=0.30, Fixed trust
    # ============================================================
    r3 = run_sim("Scenario 3: Sybil g=0.30 Fixed trust", [
        "--mode", "attack",
        "--gamma", "0.30",
        "--trust", "fixed",
        "--attack", "bc_deflation",
        "--n-classes", "3",
        "--theta-dtd", "0.004",
        "--is-alpha", "2.0",
        "--baseline-conv", bc_str,
        "--baseline-all", ba_str,
    ], "rerun_s1_sybil_fixed.json")
    results["sybil_fixed"] = r3

    # ============================================================
    # Scenario 4: Sybil gamma=0.30, Dynamic trust
    # ============================================================
    r4 = run_sim("Scenario 4: Sybil g=0.30 Dynamic trust", [
        "--mode", "attack",
        "--gamma", "0.30",
        "--trust", "dynamic",
        "--attack", "bc_deflation",
        "--n-classes", "3",
        "--theta-dtd", "0.004",
        "--is-alpha", "2.0",
        "--baseline-conv", bc_str,
        "--baseline-all", ba_str,
    ], "rerun_s1_sybil_dynamic.json")
    results["sybil_dynamic"] = r4

    # ============================================================
    # Scenario 5: Sybil gamma=1.0, Dynamic trust
    # ============================================================
    r5 = run_sim("Scenario 5: Sybil g=1.0 Dynamic trust", [
        "--mode", "attack",
        "--gamma", "1.0",
        "--trust", "dynamic",
        "--attack", "bc_deflation",
        "--n-classes", "3",
        "--theta-dtd", "0.004",
        "--is-alpha", "2.0",
        "--baseline-conv", bc_str,
        "--baseline-all", ba_str,
    ], "rerun_s1_sybil_dynamic_g1.json")
    results["sybil_dynamic_g1"] = r5

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1 RESULTS TABLE")
    print("=" * 70)
    print(f"  Baseline converged TSTT: {baseline_converged:.2f} veh-hours")
    print()
    header = f"{'Scenario':<30s} {'Sustained':>10s} {'Peak':>10s} {'Overshoot':>10s} {'TrustRec':>10s}"
    print(header)
    print("-" * len(header))

    scenarios = [
        ("Disruption", results["disruption"]),
        ("Sybil g=0.3 Fixed", results["sybil_fixed"]),
        ("Sybil g=0.3 Dynamic", results["sybil_dynamic"]),
        ("Sybil g=1.0 Dynamic", results["sybil_dynamic_g1"]),
    ]
    for name, r in scenarios:
        sus = r.get("sustained_poatt", "N/A")
        peak = r.get("peak_poatt", "N/A")
        over = r.get("overshoot", "N/A")
        trec = r.get("trust_recovery", "N/A")
        sus_s = f"{sus:.4f}" if isinstance(sus, float) else str(sus)
        peak_s = f"{peak:.4f}" if isinstance(peak, float) else str(peak)
        over_s = f"{over:.4f}" if isinstance(over, float) else str(over)
        trec_s = f"{trec}" if not isinstance(trec, float) else f"{trec:.0f}"
        print(f"  {name:<28s} {sus_s:>10s} {peak_s:>10s} {over_s:>10s} {trec_s:>10s}")

    # Save combined results
    out_path = os.path.join(RESULTS_DIR, "rerun_signature.json")
    with open(out_path, "w") as f:
        json.dump({
            "baseline_converged": baseline_converged,
            "baseline_all": baseline_all,
            "scenarios": {k: {key: v[key] for key in v if key != "tstt"} if isinstance(v, dict) and "tstt" in v else v
                          for k, v in results.items()},
        }, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    # ============================================================
    # Figure: 4-panel
    # ============================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = np.arange(1, 201)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) TSTT trajectories
        ax = axes[0, 0]
        ax.plot(days, r1["tstt"], "k--", lw=1, alpha=0.5, label="Baseline")
        ax.plot(days, r2["tstt"], color="#27ae60", lw=1.2, label="Disruption")
        ax.plot(days, r3["tstt"], color="#e67e22", lw=1.2, label="Sybil Fixed g=0.3")
        ax.plot(days, r4["tstt"], color="#8e44ad", lw=1.2, label="Sybil Dynamic g=0.3")
        ax.plot(days, r5["tstt"], color="#e74c3c", lw=1.2, label="Sybil Dynamic g=1.0")
        ax.axvspan(51, 100, alpha=0.08, color="red")
        ax.set_title("(a) TSTT Trajectories", fontsize=12)
        ax.set_ylabel("TSTT [veh-hours]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (b) Trust evolution
        ax = axes[0, 1]
        trust4 = np.array(r4["trust"])
        trust5 = np.array(r5["trust"])
        colors = ["#e74c3c", "#3498db", "#2ecc71"]
        names = ["CAV", "App", "Exp"]
        for k in range(3):
            ax.plot(days, trust4[:, k], color=colors[k], lw=1.2,
                    label=f"{names[k]} g=0.3", linestyle="-")
            ax.plot(days, trust5[:, k], color=colors[k], lw=1.2,
                    label=f"{names[k]} g=1.0", linestyle="--")
        ax.axvspan(51, 100, alpha=0.08, color="red")
        ax.set_title("(b) Trust Evolution", fontsize=12)
        ax.set_ylabel("Expected Trust")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # (c) Info weight (lambda)
        ax = axes[1, 0]
        iw4 = np.array(r4["info_weight"])
        iw5 = np.array(r5["info_weight"])
        for k in range(3):
            ax.plot(days, iw4[:, k], color=colors[k], lw=1.2,
                    label=f"{names[k]} g=0.3", linestyle="-")
            ax.plot(days, iw5[:, k], color=colors[k], lw=1.2,
                    label=f"{names[k]} g=1.0", linestyle="--")
        ax.axvspan(51, 100, alpha=0.08, color="red")
        ax.set_title("(c) Information Weight", fontsize=12)
        ax.set_ylabel("Lambda")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # (d) Metrics table
        ax = axes[1, 1]
        ax.axis("off")
        table_data = []
        for name, r in scenarios:
            sus = r.get("sustained_poatt", "-")
            peak = r.get("peak_poatt", "-")
            over = r.get("overshoot", "-")
            trec = r.get("trust_recovery", "-")
            table_data.append([
                name,
                f"{sus:.4f}" if isinstance(sus, float) else str(sus),
                f"{peak:.4f}" if isinstance(peak, float) else str(peak),
                f"{over:.4f}" if isinstance(over, float) else str(over),
                str(trec),
            ])
        table = ax.table(cellText=table_data,
                         colLabels=["Scenario", "Sustained", "Peak", "Overshoot", "Trust Rec"],
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        ax.set_title("(d) Metrics", fontsize=12, pad=20)

        fig.suptitle(f"Signature Result (LWR, baseline={baseline_converged:.1f})", fontsize=14, y=1.01)
        fig.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "rerun_signature.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fig_path}")
    except Exception as e:
        print(f"  Figure generation failed: {e}")

    total = time.time() - t_start
    print(f"\n  Total Step 1 time: {total:.0f}s ({total/60:.1f} min)")
    print("\n=== STEP 1 COMPLETE ===")


if __name__ == "__main__":
    main()
