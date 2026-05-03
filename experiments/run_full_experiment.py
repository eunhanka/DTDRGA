"""Complete Experiment: BC-Deflation Attack + Full Sensitivity Analysis.

Steps 1-10: Run all scenarios using subprocess runner pattern.
Each step saves results to dtd_attack/results/exp_*.json
"""

import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RES_DIR = os.path.join(BASE_DIR, os.pardir, "results")
os.makedirs(RES_DIR, exist_ok=True)
RUNNER = os.path.join(BASE_DIR, "lwr_single_sim.py")

CAPTURE_DAYS_SF = "39,54,64,74,84,94,104,124,149,174"
CAPTURE_DAYS_ANA = "39,54,74,94,124,174"

N_DAYS = 200
ATTACK_START = 50
ATTACK_END = 100


def run_sim(args_list, label="", timeout=7200):
    """Run a single simulation via subprocess. Returns parsed JSON or None."""
    outfile = args_list[args_list.index("--output") + 1]
    cmd = [sys.executable, RUNNER] + args_list
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=BASE_DIR)
        elapsed = time.time() - t0
        if proc.stdout:
            for line in proc.stdout.strip().split("\n"):
                if "Sustained" in line or "Peak" in line or "Done:" in line or "BCSybil" in line:
                    print(f"    {line.strip()}")
        if proc.returncode != 0:
            print(f"  FAILED ({elapsed:.0f}s, rc={proc.returncode})")
            if proc.stderr:
                for line in proc.stderr.strip().split("\n")[-5:]:
                    print(f"    {line}")
            return None
        with open(outfile) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT ({time.time()-t0:.0f}s)")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def make_args(mode="attack", attack="bc_deflation", trust="fixed", gamma=0.3,
              n_targets=10, network="siouxfalls", output="tmp.json",
              baseline_conv=None, baseline_all=None,
              capture_days=None, theta_dtd=0.004, is_alpha=2.0,
              n_classes=3, delta_all=None, wf_ratio=None, cav_frac=None):
    """Build argument list for lwr_single_sim.py."""
    args = ["--mode", mode, "--output", os.path.join(RES_DIR, output)]
    if mode != "baseline" and mode != "disruption":
        args += ["--attack", attack, "--trust", trust, "--gamma", str(gamma)]
        args += ["--n-target-links", str(n_targets)]
    args += ["--network", network, "--theta-dtd", str(theta_dtd)]
    args += ["--is-alpha", str(is_alpha), "--n-classes", str(n_classes)]
    if baseline_conv is not None:
        args += ["--baseline-conv", str(baseline_conv)]
    if baseline_all is not None:
        args += ["--baseline-all", str(baseline_all)]
    if capture_days:
        args += ["--capture-days", capture_days]
    if delta_all is not None:
        args += ["--delta-all", str(delta_all)]
    if wf_ratio is not None:
        args += ["--wf-ratio", str(wf_ratio)]
    if cav_frac is not None:
        args += ["--cav-frac", str(cav_frac)]
    return args


# ====================================================================
# STEP 1: Signature (6 scenarios)
# ====================================================================
def step1(total_t0):
    print("\n" + "=" * 70)
    print("STEP 1: BASELINE + SIGNATURE (6 scenarios)")
    print("=" * 70)

    results = {}

    # 1. Baseline (reuse if already exists)
    baseline_path = os.path.join(RES_DIR, "exp_s1_baseline.json")
    if os.path.exists(baseline_path):
        print("\n  [1/6] Baseline... (reusing existing)")
        with open(baseline_path) as f:
            r = json.load(f)
    else:
        print("\n  [1/6] Baseline...")
        r = run_sim(make_args(mode="baseline", output="exp_s1_baseline.json",
                              capture_days=CAPTURE_DAYS_SF))
    if r is None:
        print("  FATAL: baseline failed")
        return None
    results["baseline"] = r
    bc = float(np.mean(r["tstt"][29:50]))
    ba = float(np.mean(r["tstt"][:50]))
    print(f"    baseline_converged={bc:.2f}")

    # 2. Physical disruption (reuse if already exists)
    disruption_path = os.path.join(RES_DIR, "exp_s1_disruption.json")
    if os.path.exists(disruption_path):
        print("  [2/6] Physical disruption... (reusing existing)")
        with open(disruption_path) as f:
            r2 = json.load(f)
    else:
        print("  [2/6] Physical disruption...")
        r2 = run_sim(["--mode", "disruption", "--output",
                       os.path.join(RES_DIR, "exp_s1_disruption.json"),
                       "--baseline-conv", str(bc), "--baseline-all", str(ba)])
    results["disruption"] = r2
    if r2:
        print(f"    PoAtt={r2.get('sustained_poatt','?')}")

    # 3. BC-Deflation Fixed gamma=0.3
    print("  [3/6] BC-Deflation Fixed gamma=0.3...")
    r3 = run_sim(make_args(attack="bc_deflation", trust="fixed", gamma=0.3,
                            output="exp_s1_defl_fixed.json",
                            baseline_conv=bc, baseline_all=ba,
                            capture_days=CAPTURE_DAYS_SF))
    results["bc_defl_fixed_03"] = r3
    if r3:
        print(f"    Sustained={r3.get('sustained_poatt','?'):.4f}")

    # 4. BC-Deflation Dynamic gamma=0.3
    print("  [4/6] BC-Deflation Dynamic gamma=0.3...")
    r4 = run_sim(make_args(attack="bc_deflation", trust="dynamic", gamma=0.3,
                            output="exp_s1_defl_dyn_03.json",
                            baseline_conv=bc, baseline_all=ba,
                            capture_days=CAPTURE_DAYS_SF))
    results["bc_defl_dynamic_03"] = r4
    if r4:
        print(f"    Sustained={r4.get('sustained_poatt','?'):.4f}")

    # 5. BC-Deflation Dynamic gamma=1.0
    print("  [5/6] BC-Deflation Dynamic gamma=1.0...")
    r5 = run_sim(make_args(attack="bc_deflation", trust="dynamic", gamma=1.0,
                            output="exp_s1_defl_dyn_10.json",
                            baseline_conv=bc, baseline_all=ba))
    results["bc_defl_dynamic_10"] = r5

    # 6. BC-Inflation Fixed gamma=0.3 (direction comparison)
    print("  [6/6] BC-Inflation Fixed gamma=0.3...")
    r6 = run_sim(make_args(attack="bc_inflation", trust="fixed", gamma=0.3,
                            output="exp_s1_infl_fixed.json",
                            baseline_conv=bc, baseline_all=ba))
    results["bc_infl_fixed_03"] = r6

    # Assemble output
    output = {"baseline_converged": bc, "baseline_all": ba, "scenarios": {}}
    for k, r in results.items():
        if r is not None:
            output["scenarios"][k] = r

    with open(os.path.join(RES_DIR, "exp_signature.json"), "w") as f:
        json.dump(output, f)
    print(f"\n  Saved exp_signature.json")
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")
    print("=== STEP 1 COMPLETE ===")
    return bc, ba


# ====================================================================
# STEP 2: Gamma Sweep (20 scenarios)
# ====================================================================
def step2(bc, ba, total_t0):
    print("\n" + "=" * 70)
    print("STEP 2: GAMMA SWEEP (20 scenarios)")
    print("=" * 70)

    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}
    n = 0

    for g in gammas:
        for trust in ["fixed", "dynamic"]:
            n += 1
            key = f"g{g}_{trust}"
            print(f"  [{n}/20] gamma={g}, {trust}...")
            r = run_sim(make_args(attack="bc_deflation", trust=trust, gamma=g,
                                   output=f"exp_g_{g}_{trust}.json",
                                   baseline_conv=bc, baseline_all=ba))
            if r:
                results[key] = {
                    "gamma": g, "trust": trust,
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                    "overshoot": r.get("overshoot"),
                    "trust_recovery": r.get("trust_recovery"),
                    "tstt": r["tstt"],
                }
                if "trust" in r:
                    results[key]["trust_traj"] = r["trust"]
                if "info_weight" in r:
                    results[key]["info_weight"] = r["info_weight"]
                print(f"    Sus={r.get('sustained_poatt','?'):.4f} Peak={r.get('peak_poatt','?'):.4f}")

    output = {"baseline_converged": bc, "gammas": gammas, "results": results}
    with open(os.path.join(RES_DIR, "exp_gamma_sweep.json"), "w") as f:
        json.dump(output, f)

    # Print table
    print(f"\n{'g':>5} {'Fixed Sus':>10} {'Fix Peak':>10} {'Dyn Sus':>10} {'Dyn Peak':>10} {'Def%':>8}")
    for g in gammas:
        kf, kd = f"g{g}_fixed", f"g{g}_dynamic"
        sf = results.get(kf, {}).get("sustained_poatt")
        pf = results.get(kf, {}).get("peak_poatt")
        sd = results.get(kd, {}).get("sustained_poatt")
        pd_ = results.get(kd, {}).get("peak_poatt")
        de = f"{(1-(sd-1)/(sf-1))*100:.0f}%" if sf and sd and sf > 1 else "N/A"
        print(f"{g:>5.1f} {sf or 0:>10.4f} {pf or 0:>10.4f} {sd or 0:>10.4f} {pd_ or 0:>10.4f} {de:>8}")

    print(f"  Elapsed: {time.time()-total_t0:.0f}s")
    print("=== STEP 2 COMPLETE ===")


# ====================================================================
# STEP 3: n_target Sweep (12 scenarios)
# ====================================================================
def step3(bc, ba, total_t0):
    print("\n" + "=" * 70)
    print("STEP 3: n_target SWEEP (12 scenarios)")
    print("=" * 70)

    n_targets = [3, 5, 10, 15, 20, 30]
    results = {}
    n = 0

    for nt in n_targets:
        for trust in ["fixed", "dynamic"]:
            n += 1
            key = f"nt{nt}_{trust}"
            print(f"  [{n}/12] n_target={nt}, {trust}...")
            r = run_sim(make_args(attack="bc_deflation", trust=trust, gamma=0.3,
                                   n_targets=nt, output=f"exp_nt_{nt}_{trust}.json",
                                   baseline_conv=bc, baseline_all=ba))
            if r:
                results[key] = {
                    "n_target": nt, "trust": trust,
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                    "tstt": r["tstt"],
                }
                print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    output = {"baseline_converged": bc, "n_targets": n_targets, "results": results}
    with open(os.path.join(RES_DIR, "exp_ntarget_sweep.json"), "w") as f:
        json.dump(output, f)
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")
    print("=== STEP 3 COMPLETE ===")


# ====================================================================
# STEP 4: BR delta Sweep (12 scenarios)
# ====================================================================
def step4(bc, ba, total_t0):
    print("\n" + "=" * 70)
    print("STEP 4: BR DELTA SWEEP (12 scenarios)")
    print("=" * 70)

    deltas = [0, 100, 200, 300, 400, 500]
    results = {}
    n = 0

    for d in deltas:
        for trust in ["fixed", "dynamic"]:
            n += 1
            key = f"d{d}_{trust}"
            print(f"  [{n}/12] delta={d}, {trust}...")
            r = run_sim(make_args(attack="bc_deflation", trust=trust, gamma=0.3,
                                   delta_all=d, output=f"exp_br_{d}_{trust}.json",
                                   baseline_conv=bc, baseline_all=ba))
            if r:
                results[key] = {
                    "delta": d, "trust": trust,
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                    "tstt": r["tstt"],
                }
                print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    output = {"baseline_converged": bc, "deltas": deltas, "results": results}
    with open(os.path.join(RES_DIR, "exp_br_sweep.json"), "w") as f:
        json.dump(output, f)
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")
    print("=== STEP 4 COMPLETE ===")


# ====================================================================
# STEP 5: theta Sweep (10 scenarios)
# ====================================================================
def step5(bc, ba, total_t0):
    print("\n" + "=" * 70)
    print("STEP 5: THETA SWEEP (10 scenarios)")
    print("=" * 70)

    thetas = [0.002, 0.004, 0.006, 0.008, 0.010]
    results = {}
    n = 0

    for th in thetas:
        for trust in ["fixed", "dynamic"]:
            n += 1
            key = f"th{th}_{trust}"
            print(f"  [{n}/10] theta={th}, {trust}...")
            r = run_sim(make_args(attack="bc_deflation", trust=trust, gamma=0.3,
                                   theta_dtd=th, output=f"exp_th_{th}_{trust}.json",
                                   baseline_conv=bc, baseline_all=ba))
            if r:
                results[key] = {
                    "theta": th, "trust": trust,
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                    "overshoot": r.get("overshoot"),
                    "tstt": r["tstt"],
                }
                print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    output = {"baseline_converged": bc, "thetas": thetas, "results": results}
    with open(os.path.join(RES_DIR, "exp_theta_sweep.json"), "w") as f:
        json.dump(output, f)
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")
    print("=== STEP 5 COMPLETE ===")


# ====================================================================
# STEP 6: IS alpha Sweep (8 scenarios)
# ====================================================================
def step6(bc, ba, total_t0):
    print("\n" + "=" * 70)
    print("STEP 6: IS ALPHA SWEEP (8 scenarios)")
    print("=" * 70)

    alphas = [0, 1, 2, 3]
    results = {}
    n = 0

    for a in alphas:
        for trust in ["fixed", "dynamic"]:
            n += 1
            key = f"a{a}_{trust}"
            print(f"  [{n}/8] alpha={a}, {trust}...")
            r = run_sim(make_args(attack="bc_deflation", trust=trust, gamma=0.3,
                                   is_alpha=float(a), output=f"exp_is_{a}_{trust}.json",
                                   baseline_conv=bc, baseline_all=ba))
            if r:
                results[key] = {
                    "alpha": a, "trust": trust,
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                    "tstt": r["tstt"],
                }
                print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    output = {"baseline_converged": bc, "alphas": alphas, "results": results}
    with open(os.path.join(RES_DIR, "exp_is_sweep.json"), "w") as f:
        json.dump(output, f)
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")
    print("=== STEP 6 COMPLETE ===")


# ====================================================================
# STEP 7: Composition Sweep (12 scenarios: 6 CAV × 2 trust)
# ====================================================================
def step7(bc, ba, total_t0):
    print("\n" + "=" * 70)
    print("STEP 7: COMPOSITION SWEEP (12 scenarios)")
    print("=" * 70)

    cavs = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    results = {}
    n = 0

    for cav in cavs:
        # Compute fractions
        if cav >= 1.0:
            fracs = [1.0, 0.0, 0.0]
        elif cav <= 0.0:
            fracs = [0.0, 0.70, 0.30]
        else:
            rem = 1.0 - cav
            fracs = [cav, rem * 0.70, rem * 0.30]
        # chi(pi) = sum(pi_k * theta_trust_k * lambda_max_k)
        theta_t = [2.0, 1.0, 0.5]
        lam_max = [0.9, 0.7, 0.3]
        chi = sum(fracs[k] * theta_t[k] * lam_max[k] for k in range(3))

        for trust in ["fixed", "dynamic"]:
            n += 1
            key = f"cav{cav}_{trust}"
            print(f"  [{n}/12] CAV={cav*100:.0f}% {trust}...")
            r = run_sim(make_args(attack="bc_deflation", trust=trust, gamma=0.3,
                                   cav_frac=cav, output=f"exp_comp_{cav}_{trust}.json",
                                   baseline_conv=bc, baseline_all=ba))
            if r:
                results[key] = {
                    "cav_frac": cav, "fractions": fracs, "chi": chi,
                    "trust": trust,
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                    "trust_recovery": r.get("trust_recovery"),
                    "tstt": r["tstt"],
                }
                print(f"    Sus={r.get('sustained_poatt','?'):.4f}, chi={chi:.3f}")

    output = {"baseline_converged": bc, "cavs": cavs, "results": results}
    with open(os.path.join(RES_DIR, "exp_composition.json"), "w") as f:
        json.dump(output, f)
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")
    print("=== STEP 7 COMPLETE ===")


# ====================================================================
# STEP 8: Trust Asymmetry Sweep (6 scenarios)
# ====================================================================
def step8(bc, ba, total_t0):
    print("\n" + "=" * 70)
    print("STEP 8: TRUST ASYMMETRY SWEEP (6 scenarios)")
    print("=" * 70)

    ratios = [1, 3, 5, 7, 10, 15]
    results = {}
    n = 0

    for ratio in ratios:
        n += 1
        key = f"r{ratio}"
        print(f"  [{n}/6] w_f/w_s={ratio}...")
        r = run_sim(make_args(attack="bc_deflation", trust="dynamic", gamma=0.3,
                               wf_ratio=float(ratio),
                               output=f"exp_trust_{ratio}.json",
                               baseline_conv=bc, baseline_all=ba))
        if r:
            results[key] = {
                "ratio": ratio,
                "sustained_poatt": r.get("sustained_poatt"),
                "peak_poatt": r.get("peak_poatt"),
                "trust_recovery": r.get("trust_recovery"),
                "tstt": r["tstt"],
            }
            if "trust" in r:
                results[key]["trust_traj"] = r["trust"]
            if "info_weight" in r:
                results[key]["info_weight"] = r["info_weight"]
            print(f"    Sus={r.get('sustained_poatt','?'):.4f} TrustRec={r.get('trust_recovery','?')}")

    output = {"baseline_converged": bc, "ratios": ratios, "results": results}
    with open(os.path.join(RES_DIR, "exp_trust_sensitivity.json"), "w") as f:
        json.dump(output, f)

    # Save full trajectory for default (ratio=5) as recovery_data
    if "r5" in results and "trust_traj" in results["r5"]:
        rec_data = {
            "baseline_converged": bc,
            "tstt": results["r5"]["tstt"],
            "trust": results["r5"]["trust_traj"],
            "info_weight": results["r5"].get("info_weight"),
            "sustained_poatt": results["r5"]["sustained_poatt"],
            "trust_recovery": results["r5"]["trust_recovery"],
        }
        with open(os.path.join(RES_DIR, "exp_recovery_data.json"), "w") as f:
            json.dump(rec_data, f)

    # Power-law fit
    valid_ratios = []
    valid_rec = []
    for ratio in ratios:
        k = f"r{ratio}"
        tr = results.get(k, {}).get("trust_recovery")
        if tr is not None and tr > 0:
            valid_ratios.append(ratio)
            valid_rec.append(tr)
    if len(valid_ratios) >= 2:
        lr = np.log(valid_ratios)
        lt = np.log(valid_rec)
        slope, intercept = np.polyfit(lr, lt, 1)
        print(f"  Power-law fit: log(recovery) = {slope:.3f} * log(ratio) + {intercept:.3f}")

    print(f"  Elapsed: {time.time()-total_t0:.0f}s")
    print("=== STEP 8 COMPLETE ===")


# ====================================================================
# STEP 9: Anaheim (3 scenarios)
# ====================================================================
def step9(total_t0):
    print("\n" + "=" * 70)
    print("STEP 9: ANAHEIM (3 scenarios)")
    print("=" * 70)

    results = {}

    # 1. Baseline
    print("  [1/3] Anaheim baseline...")
    r1 = run_sim(make_args(mode="baseline", network="anaheim",
                            output="exp_ana_baseline.json",
                            capture_days=CAPTURE_DAYS_ANA), timeout=14400)
    if r1 is None:
        print("  Anaheim baseline failed")
        return
    results["baseline"] = r1
    bc_a = float(np.mean(r1["tstt"][29:50]))
    ba_a = float(np.mean(r1["tstt"][:50]))
    print(f"    baseline_converged={bc_a:.2f}")

    # 2. BC-Deflation Fixed
    print("  [2/3] Anaheim BC-Deflation Fixed...")
    r2 = run_sim(make_args(attack="bc_deflation", trust="fixed", gamma=0.3,
                            network="anaheim", output="exp_ana_defl_fixed.json",
                            baseline_conv=bc_a, baseline_all=ba_a,
                            capture_days=CAPTURE_DAYS_ANA), timeout=14400)
    results["bc_defl_fixed"] = r2

    # 3. BC-Deflation Dynamic
    print("  [3/3] Anaheim BC-Deflation Dynamic...")
    r3 = run_sim(make_args(attack="bc_deflation", trust="dynamic", gamma=0.3,
                            network="anaheim", output="exp_ana_defl_dyn.json",
                            baseline_conv=bc_a, baseline_all=ba_a,
                            capture_days=CAPTURE_DAYS_ANA), timeout=14400)
    results["bc_defl_dynamic"] = r3

    output = {"baseline_converged": bc_a, "baseline_all": ba_a, "scenarios": {}}
    for k, r in results.items():
        if r is not None:
            output["scenarios"][k] = r

    with open(os.path.join(RES_DIR, "exp_anaheim.json"), "w") as f:
        json.dump(output, f)
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")
    print("=== STEP 9 COMPLETE ===")


# ====================================================================
# STEP 10: Attack Comparison (4 scenarios) + Random Multiseed (20)
# ====================================================================
def step10(bc, ba, total_t0):
    print("\n" + "=" * 70)
    print("STEP 10: ATTACK COMPARISON + RANDOM MULTISEED")
    print("=" * 70)

    # --- Part A: Topo-BC vs Path-BC (4 scenarios) ---
    attacks = [
        ("bc_deflation", "Topo-BC"),
        ("path_bc", "Path-BC"),
    ]
    results = {}
    n = 0
    for atk_type, atk_label in attacks:
        for trust in ["fixed", "dynamic"]:
            n += 1
            key = f"{atk_type}_{trust}"
            print(f"  [A-{n}/4] {atk_label} {trust}...")
            r = run_sim(make_args(attack=atk_type, trust=trust, gamma=0.3,
                                   output=f"exp_cmp_{atk_type}_{trust}.json",
                                   baseline_conv=bc, baseline_all=ba))
            if r:
                results[key] = {
                    "attack": atk_type, "label": atk_label, "trust": trust,
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                    "overshoot": r.get("overshoot"),
                    "trust_recovery": r.get("trust_recovery"),
                    "tstt": r["tstt"],
                }
                print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    output = {"baseline_converged": bc, "results": results}
    with open(os.path.join(RES_DIR, "exp_attack_comparison.json"), "w") as f:
        json.dump(output, f)

    # --- Part B: Random Multiseed (10 seeds × 2 trust = 20 scenarios) ---
    print(f"\n  Random multiseed (10 seeds × 2 trust)...")
    seeds = list(range(10))
    rand_results = {}
    for seed in seeds:
        for trust in ["fixed", "dynamic"]:
            n += 1
            key = f"seed{seed}_{trust}"
            print(f"  [B-{n}/24] Random seed={seed} {trust}...")
            args = make_args(attack="random", trust=trust, gamma=0.3,
                             output=f"exp_rand_s{seed}_{trust}.json",
                             baseline_conv=bc, baseline_all=ba)
            # Add seed argument
            args += ["--seed", str(seed)]
            r = run_sim(args, label=f"rand s{seed} {trust}")
            if r:
                rand_results[key] = {
                    "seed": seed, "trust": trust,
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                }

    # Compute summary
    summary = {}
    for trust in ["fixed", "dynamic"]:
        vals = [rand_results[f"seed{s}_{trust}"]["sustained_poatt"]
                for s in seeds
                if f"seed{s}_{trust}" in rand_results
                and rand_results[f"seed{s}_{trust}"].get("sustained_poatt")]
        if vals:
            summary[f"{trust}_mean"] = float(np.mean(vals))
            summary[f"{trust}_std"] = float(np.std(vals))
            print(f"    Random {trust}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    rand_output = {"baseline_converged": bc, "seeds": seeds,
                   "results": rand_results, "summary": summary}
    with open(os.path.join(RES_DIR, "exp_random_multiseed.json"), "w") as f:
        json.dump(rand_output, f)

    print(f"  Elapsed: {time.time()-total_t0:.0f}s")
    print("=== STEP 10 COMPLETE ===")


# ====================================================================
# MAIN
# ====================================================================
def main():
    total_t0 = time.time()

    # Step 1
    result = step1(total_t0)
    if result is None:
        print("FATAL: Step 1 failed")
        return
    bc, ba = result

    # Steps 2-8 (all Sioux Falls)
    step2(bc, ba, total_t0)
    step3(bc, ba, total_t0)
    step4(bc, ba, total_t0)
    step5(bc, ba, total_t0)
    step6(bc, ba, total_t0)
    step7(bc, ba, total_t0)
    step8(bc, ba, total_t0)

    # Step 9 (Anaheim)
    step9(total_t0)

    # Step 10 (Attack comparison)
    step10(bc, ba, total_t0)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"ALL SIMULATION STEPS COMPLETE")
    print(f"Total elapsed: {total_elapsed:.0f}s ({total_elapsed/3600:.1f} hours)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
