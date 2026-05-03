"""STEP 2: Gamma Sweep (16 scenarios + reuse Step 1 baseline).

gamma values: [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
For EACH gamma: Fixed trust + Dynamic trust = 16 scenarios.
Baseline reused from Step 1.

Run from dtd_attack/:
  python experiments/master_rerun_step2.py
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

# Load baseline from Step 1
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
    rows = []

    for gamma in GAMMAS:
        g_str = f"{gamma}"
        bc_str = f"{BASELINE_CONV}"
        ba_str = f"{BASELINE_ALL}"

        for trust_mode in ["fixed", "dynamic"]:
            label = f"g={gamma} {trust_mode}"
            fname = f"rerun_s2_g{gamma}_{trust_mode}.json"

            r = run_sim(label, [
                "--mode", "attack",
                "--gamma", g_str,
                "--trust", trust_mode,
                "--attack", "bc_deflation",
                "--n-classes", "3",
                "--theta-dtd", "0.004",
                "--is-alpha", "2.0",
                "--baseline-conv", bc_str,
                "--baseline-all", ba_str,
            ], fname)

            row = {
                "gamma": gamma,
                "trust_mode": trust_mode,
                "sustained_poatt": r.get("sustained_poatt"),
                "peak_poatt": r.get("peak_poatt"),
                "overshoot": r.get("overshoot"),
                "trust_recovery": r.get("trust_recovery"),
                "trust_end_attack": r.get("trust_end_attack"),
                "mean_attack_tstt": r.get("mean_attack_tstt"),
            }
            rows.append(row)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "rerun_gamma_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {csv_path}")

    # Print table
    print("\n" + "=" * 80)
    print("STEP 2 GAMMA SWEEP RESULTS")
    print("=" * 80)
    print(f"{'gamma':>8s} {'trust':>8s} {'Sustained':>12s} {'Peak':>10s} {'Overshoot':>10s} {'TrustRec':>10s} {'TrustEnd':>10s}")
    print("-" * 70)
    for row in rows:
        trec = row["trust_recovery"]
        tend = row["trust_end_attack"]
        print(f"  {row['gamma']:>6.2f} {row['trust_mode']:>8s}"
              f" {row['sustained_poatt']:>12.4f} {row['peak_poatt']:>10.4f}"
              f" {row['overshoot']:>10.4f}"
              f" {str(trec):>10s}"
              f" {f'{tend:.4f}' if tend else 'N/A':>10s}")

    # Find gamma* (dynamic trust sustained PoAtt crosses 1.0)
    dyn_rows = [r for r in rows if r["trust_mode"] == "dynamic"]
    fix_rows = [r for r in rows if r["trust_mode"] == "fixed"]

    gamma_star = None
    for i in range(len(dyn_rows) - 1):
        if dyn_rows[i]["sustained_poatt"] < 1.0 and dyn_rows[i+1]["sustained_poatt"] >= 1.0:
            g1, g2 = dyn_rows[i]["gamma"], dyn_rows[i+1]["gamma"]
            p1, p2 = dyn_rows[i]["sustained_poatt"], dyn_rows[i+1]["sustained_poatt"]
            gamma_star = g1 + (1.0 - p1) / (p2 - p1) * (g2 - g1)
            break

    if gamma_star is None:
        # Check if all are above or below
        if all(r["sustained_poatt"] >= 1.0 for r in dyn_rows):
            gamma_star = f"< {GAMMAS[0]}"
        else:
            gamma_star = f"> {GAMMAS[-1]}"

    print(f"\n  gamma* (dynamic trust crossing 1.0): {gamma_star}")

    # Defense effectiveness
    print("\n  Defense effectiveness (1 - eta) * 100%:")
    for g in [0.3, 1.0, 2.0]:
        fix = [r for r in fix_rows if r["gamma"] == g]
        dyn = [r for r in dyn_rows if r["gamma"] == g]
        if fix and dyn:
            eta = (dyn[0]["sustained_poatt"] - 1.0) / (fix[0]["sustained_poatt"] - 1.0) if fix[0]["sustained_poatt"] != 1.0 else 0
            eff = (1 - eta) * 100
            print(f"    g={g}: eta={eta:.4f}, defense_eff={eff:.1f}%")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        gammas_f = [r["gamma"] for r in fix_rows]
        gammas_d = [r["gamma"] for r in dyn_rows]

        # (a) Sustained PoAtt
        ax = axes[0, 0]
        ax.plot(gammas_f, [r["sustained_poatt"] for r in fix_rows], "o-", color="#e74c3c", label="Fixed trust")
        ax.plot(gammas_d, [r["sustained_poatt"] for r in dyn_rows], "s-", color="#3498db", label="Dynamic trust")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("gamma")
        ax.set_ylabel("Sustained PoAtt")
        ax.set_title("(a) Sustained PoAtt vs gamma")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (b) Peak PoAtt
        ax = axes[0, 1]
        ax.plot(gammas_f, [r["peak_poatt"] for r in fix_rows], "o-", color="#e74c3c", label="Fixed trust")
        ax.plot(gammas_d, [r["peak_poatt"] for r in dyn_rows], "s-", color="#3498db", label="Dynamic trust")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("gamma")
        ax.set_ylabel("Peak PoAtt")
        ax.set_title("(b) Peak PoAtt vs gamma")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (c) Overshoot ratio
        ax = axes[1, 0]
        ax.plot(gammas_f, [r["overshoot"] for r in fix_rows], "o-", color="#e74c3c", label="Fixed trust")
        ax.plot(gammas_d, [r["overshoot"] for r in dyn_rows], "s-", color="#3498db", label="Dynamic trust")
        ax.set_xlabel("gamma")
        ax.set_ylabel("Overshoot ratio")
        ax.set_title("(c) Overshoot vs gamma")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (d) Trust end-of-attack
        ax = axes[1, 1]
        tend_vals = [r.get("trust_end_attack") for r in dyn_rows]
        if any(v is not None for v in tend_vals):
            ax.plot(gammas_d, [v if v else 0 for v in tend_vals], "s-", color="#3498db")
        ax.set_xlabel("gamma")
        ax.set_ylabel("Trust (end of attack)")
        ax.set_title("(d) Trust at attack end (dynamic)")
        ax.grid(True, alpha=0.3)

        fig.suptitle("Gamma Sweep (LWR, 3-class)", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "rerun_gamma_sweep.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: figures/rerun_gamma_sweep.png")
    except Exception as e:
        print(f"  Figure failed: {e}")

    total = time.time() - t_start
    print(f"\n  Total Step 2 time: {total:.0f}s ({total/60:.1f} min)")
    print("\n=== STEP 2 COMPLETE ===")


if __name__ == "__main__":
    main()
