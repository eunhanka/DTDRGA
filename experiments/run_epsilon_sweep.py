"""Epsilon sensitivity sweep for the trust-activation threshold gamma_hat.

Coarse fallback grid:
    epsilon_h = [0.05, 0.10, 0.15]
    gamma     = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

Reuses existing default-epsilon (0.1 h = 360 s) JSON results from results/exp_g_*_*.json
when available; only the new (epsilon, gamma) cells require a new simulation.
Fixed trust is epsilon-independent: one run per gamma.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RES_DIR = os.path.join(BASE_DIR, os.pardir, "results")
OUT_DIR = os.path.join(RES_DIR, "epsilon_sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)
RUNNER = os.path.join(BASE_DIR, "lwr_single_sim.py")

# --- Grid (coarse fallback selected in Step 2.5) ---
EPSILON_H = [0.05, 0.10, 0.15]                     # hours
EPSILON_S = [eh * 3600.0 for eh in EPSILON_H]      # seconds
GAMMAS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
DEFAULT_EPS_H = 0.10                               # already in main results
SEED = 42


def baseline_values():
    bp = os.path.join(RES_DIR, "exp_s1_baseline.json")
    with open(bp) as f:
        b = json.load(f)
    bc = float(np.mean(b["tstt"][29:50]))
    ba = float(np.mean(b["tstt"][:50]))
    return bc, ba


def run_sim(args_list, label, timeout=2400):
    outfile = args_list[args_list.index("--output") + 1]
    cmd = [sys.executable, RUNNER] + args_list
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=BASE_DIR)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"  [FAIL] {label} rc={proc.returncode} ({elapsed:.0f}s)")
            if proc.stderr:
                for ln in proc.stderr.strip().split("\n")[-4:]:
                    print(f"    {ln}")
            return None, elapsed
        with open(outfile) as f:
            return json.load(f), elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] {label} ({elapsed:.0f}s)")
        return None, elapsed
    except Exception as e:
        print(f"  [ERROR] {label}: {e}")
        return None, time.time() - t0


def fixed_args(g, bc, ba, output):
    return [
        "--mode", "attack", "--attack", "bc_deflation",
        "--trust", "fixed", "--gamma", str(g),
        "--n-target-links", "10",
        "--network", "siouxfalls",
        "--theta-dtd", "0.004", "--is-alpha", "2.0", "--n-classes", "3",
        "--baseline-conv", str(bc), "--baseline-all", str(ba),
        "--seed", str(SEED),
        "--output", output,
    ]


def dynamic_args(g, eps_s, bc, ba, output, save_errors=False):
    a = [
        "--mode", "attack", "--attack", "bc_deflation",
        "--trust", "dynamic", "--gamma", str(g),
        "--n-target-links", "10",
        "--network", "siouxfalls",
        "--theta-dtd", "0.004", "--is-alpha", "2.0", "--n-classes", "3",
        "--baseline-conv", str(bc), "--baseline-all", str(ba),
        "--seed", str(SEED),
        "--epsilon-sec", str(eps_s),
        "--output", output,
    ]
    if save_errors:
        a.append("--save-errors")
    return a


def main():
    t_global = time.time()
    bc, ba = baseline_values()
    print(f"Baseline converged TSTT: {bc:.2f}; baseline all: {ba:.2f}")
    print(f"Grid: epsilon_h = {EPSILON_H}, gamma = {GAMMAS}")

    expected_total_new = 12 + 6  # 12 new dynamic + 6 (one per gamma) for error capture
    completed = 0
    failed = 0

    # ==========================================================
    # 1) Fixed runs (epsilon-independent): reuse if present, else run.
    # ==========================================================
    fixed_results = {}
    for g in GAMMAS:
        legacy = os.path.join(RES_DIR, f"exp_g_{g}_fixed.json")
        new = os.path.join(OUT_DIR, f"fixed_g{g}.json")
        if os.path.exists(legacy):
            print(f"  [reuse] fixed gamma={g} from {os.path.basename(legacy)}")
            with open(legacy) as f:
                fixed_results[g] = json.load(f)
        else:
            print(f"  [run] fixed gamma={g}")
            res, el = run_sim(fixed_args(g, bc, ba, new), f"fixed g={g}")
            if res is None:
                failed += 1
                fixed_results[g] = None
            else:
                fixed_results[g] = res
            completed += 1
            print(f"    elapsed {el:.0f}s, completed {completed}/{expected_total_new}")

    # ==========================================================
    # 2) Day-51 guidance error per gamma (for gamma_hat_error analytic).
    #    Pre-attack dynamics are eps-independent, so day-51 error is eps-
    #    independent. Run one fixed-trust simulation per gamma with
    #    --save-errors. (We use fixed because trust update doesn't matter
    #    for error on day 51 — same flows whichever trust mode.)
    # ==========================================================
    error_per_gamma = {}
    print("\n[error capture] running 1 sim per gamma to record day-51 error...")
    for g in GAMMAS:
        out = os.path.join(OUT_DIR, f"errors_g{g}.json")
        if os.path.exists(out):
            with open(out) as f:
                d = json.load(f)
            if "guidance_error" in d:
                error_per_gamma[g] = d["guidance_error"]
                print(f"  [reuse] errors gamma={g}")
                continue
        # Use dynamic with eps=0.1 (default) so trust trajectory matches existing
        # eps=0.1 dynamic JSON, allowing reuse.
        args = dynamic_args(g, 360.0, bc, ba, out, save_errors=True)
        res, el = run_sim(args, f"err g={g}")
        if res is None:
            failed += 1
            error_per_gamma[g] = None
        else:
            error_per_gamma[g] = res.get("guidance_error")
            # Also store as the eps=0.10 dynamic result (cross-check)
            d_path = os.path.join(OUT_DIR, f"dynamic_eps0.10_g{g}.json")
            if not os.path.exists(d_path):
                with open(d_path, "w") as f:
                    json.dump(res, f)
        completed += 1
        print(f"    gamma={g} elapsed {el:.0f}s err51={error_per_gamma[g][50] if error_per_gamma[g] else 'NA'}; completed {completed}/{expected_total_new}")

    # ==========================================================
    # 3) Dynamic runs: for each (epsilon, gamma).
    #    - eps=0.10 row: reuse from step 2 (saved to dynamic_eps0.10_g{g}.json)
    #    - eps=0.05 and eps=0.15 rows: run fresh
    # ==========================================================
    dynamic_results = {eh: {} for eh in EPSILON_H}
    for eh, es in zip(EPSILON_H, EPSILON_S):
        for g in GAMMAS:
            out = os.path.join(OUT_DIR, f"dynamic_eps{eh:.2f}_g{g}.json")
            if os.path.exists(out):
                with open(out) as f:
                    dynamic_results[eh][g] = json.load(f)
                print(f"  [reuse] dynamic eps={eh} gamma={g}")
                continue
            print(f"  [run] dynamic eps={eh} gamma={g}")
            args = dynamic_args(g, es, bc, ba, out, save_errors=False)
            res, el = run_sim(args, f"dyn eps={eh} g={g}")
            if res is None:
                failed += 1
                dynamic_results[eh][g] = None
            else:
                dynamic_results[eh][g] = res
            completed += 1
            elap_total = time.time() - t_global
            eta = (elap_total / max(completed, 1)) * (expected_total_new - completed)
            print(f"    elapsed {el:.0f}s; total {elap_total:.0f}s, ETA {eta:.0f}s; completed {completed}/{expected_total_new}, failed {failed}")
            if failed > 0.2 * expected_total_new:
                print("ABORT: more than 20% of simulations failed")
                sys.exit(1)

    # ==========================================================
    # Persist combined index for downstream metrics computation
    # ==========================================================
    index = {
        "epsilon_h": EPSILON_H,
        "gammas": GAMMAS,
        "baseline_converged": bc,
        "baseline_all": ba,
        "fixed": {str(g): fixed_results[g] for g in GAMMAS},
        "dynamic": {f"{eh}": {str(g): dynamic_results[eh][g] for g in GAMMAS}
                    for eh in EPSILON_H},
        "errors_day51": {str(g): (error_per_gamma[g][50] if error_per_gamma[g] else None) for g in GAMMAS},
    }
    with open(os.path.join(OUT_DIR, "_sweep_index.json"), "w") as f:
        json.dump(index, f)

    total = time.time() - t_global
    print(f"\nSweep done in {total:.0f}s ({total/60:.1f} min). Failures: {failed}/{expected_total_new}")


if __name__ == "__main__":
    main()
