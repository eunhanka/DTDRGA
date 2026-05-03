"""STEP 7: Trust Asymmetry Sweep + Recovery (7 scenarios).

w_f/w_s ratios: [1, 3, 5, 7, 10, 15]
Dynamic trust, gamma=0.30, n_classes=3.
Vary w_f for ALL classes proportionally while keeping w_s fixed.

Also: recovery decomposition analysis.

Run from dtd_attack/:
  python experiments/master_rerun_step7.py
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

RATIOS = [1, 3, 5, 7, 10, 15]

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

    for ratio in RATIOS:
        label = f"wf/ws={ratio}"
        fname = f"rerun_s7_ratio{ratio}.json"

        r = run_sim(label, [
            "--mode", "attack",
            "--gamma", "0.3",
            "--trust", "dynamic",
            "--attack", "bc_deflation",
            "--n-classes", "3",
            "--theta-dtd", "0.004",
            "--is-alpha", "2.0",
            "--wf-ratio", str(ratio),
            "--baseline-conv", str(BASELINE_CONV),
            "--baseline-all", str(BASELINE_ALL),
        ], fname)

        rows.append({
            "wf_ws_ratio": ratio,
            "sustained_poatt": r.get("sustained_poatt"),
            "peak_poatt": r.get("peak_poatt"),
            "overshoot": r.get("overshoot"),
            "trust_recovery": r.get("trust_recovery"),
        })

    # Save
    with open(os.path.join(RESULTS_DIR, "rerun_trust_sensitivity.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # Print
    print("\n" + "=" * 70)
    print("STEP 7 TRUST ASYMMETRY RESULTS")
    print("=" * 70)
    print(f"{'ratio':>8s} {'Sustained':>12s} {'Peak':>10s} {'Overshoot':>10s} {'TrustRec':>10s}")
    print("-" * 55)
    for row in rows:
        print(f"  {row['wf_ws_ratio']:>6d}"
              f" {row['sustained_poatt']:>12.4f} {row['peak_poatt']:>10.4f}"
              f" {row['overshoot']:>10.4f} {str(row['trust_recovery']):>10s}")

    # Power-law fit: log(recovery_time) = a * log(wf/ws) + b
    valid = [(r["wf_ws_ratio"], r["trust_recovery"]) for r in rows
             if r["trust_recovery"] is not None and r["trust_recovery"] > 0]
    if len(valid) >= 2:
        x = np.log([v[0] for v in valid])
        y = np.log([v[1] for v in valid])
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"\n  Power-law fit: slope={slope:.3f}, R^2={r_sq:.3f}")
    else:
        slope, r_sq = None, None
        print("\n  Power-law fit: insufficient data")

    # Recovery decomposition from default dynamic trust scenario (Step 1)
    print("\n  Recovery decomposition (from Step 1 g=0.3 dynamic):")
    with open(os.path.join(RESULTS_DIR, "rerun_s1_sybil_dynamic.json")) as f:
        r_default = json.load(f)

    tstt_arr = np.array(r_default["tstt"])
    trust_arr = np.array(r_default["trust"])
    attack_end = 100

    # TSTT recovery day (within 1% of baseline)
    tstt_target = BASELINE_CONV * 1.01
    tstt_rec_day = None
    for i in range(attack_end, len(tstt_arr)):
        if tstt_arr[i] <= tstt_target:
            tstt_rec_day = i - attack_end
            break
    if tstt_rec_day is None:
        tstt_rec_day = len(tstt_arr) - attack_end

    # Trust recovery day (mean trust reaches 95% of pre-attack)
    pre_attack_trust = trust_arr[49].mean()
    trust_target = 0.95 * pre_attack_trust
    trust_rec_day = None
    for i in range(attack_end, len(trust_arr)):
        if trust_arr[i].mean() >= trust_target:
            trust_rec_day = i - attack_end
            break
    if trust_rec_day is None:
        trust_rec_day = len(trust_arr) - attack_end

    vulnerability_window = trust_rec_day - tstt_rec_day

    print(f"    TSTT recovery day: {tstt_rec_day} days after attack")
    print(f"    Trust recovery day: {trust_rec_day} days after attack")
    print(f"    Hidden vulnerability window: {vulnerability_window} days")

    # Figure: trust sensitivity
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot([r["wf_ws_ratio"] for r in rows], [r["sustained_poatt"] for r in rows],
                "o-", color="#8e44ad", lw=2)
        ax.set_xlabel("w_f / w_s ratio")
        ax.set_ylabel("Sustained PoAtt")
        ax.set_title("PoAtt vs Trust Asymmetry")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        rec_vals = [r["trust_recovery"] for r in rows]
        ax.plot([r["wf_ws_ratio"] for r in rows], rec_vals, "o-", color="#e74c3c", lw=2)
        ax.set_xlabel("w_f / w_s ratio")
        ax.set_ylabel("Trust Recovery Delay (days)")
        ax.set_title(f"Recovery vs Asymmetry (slope={slope:.2f})" if slope else "Recovery vs Asymmetry")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "rerun_trust_sensitivity.png"), dpi=200)
        plt.close(fig)
        print(f"\n  Saved: figures/rerun_trust_sensitivity.png")

        # Recovery decomposition figure
        fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
        days = np.arange(1, 201)
        ax2 = ax.twinx()
        ax.plot(days, tstt_arr, color="#e74c3c", lw=1.5, label="TSTT")
        ax.axhline(BASELINE_CONV, color="gray", linestyle="--", alpha=0.5, label="Baseline TSTT")
        ax2.plot(days, trust_arr.mean(axis=1), color="#3498db", lw=1.5, label="Mean Trust")
        ax.axvspan(51, 100, alpha=0.08, color="red")
        if tstt_rec_day is not None:
            ax.axvline(100 + tstt_rec_day, color="#e74c3c", linestyle=":", alpha=0.7, label=f"TSTT recovered (d={tstt_rec_day})")
        if trust_rec_day is not None:
            ax.axvline(100 + trust_rec_day, color="#3498db", linestyle=":", alpha=0.7, label=f"Trust recovered (d={trust_rec_day})")
        ax.set_xlabel("Day")
        ax.set_ylabel("TSTT [veh-hours]", color="#e74c3c")
        ax2.set_ylabel("Mean Trust", color="#3498db")
        ax.set_title(f"Recovery Decomposition (vulnerability window = {vulnerability_window} days)")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
        fig2.tight_layout()
        fig2.savefig(os.path.join(FIGURES_DIR, "rerun_recovery_decomposition.png"), dpi=200)
        plt.close(fig2)
        print(f"  Saved: figures/rerun_recovery_decomposition.png")
    except Exception as e:
        print(f"  Figure failed: {e}")

    total = time.time() - t_start
    print(f"\n  Total Step 7 time: {total:.0f}s ({total/60:.1f} min)")
    print("\n=== STEP 7 COMPLETE ===")


if __name__ == "__main__":
    main()
