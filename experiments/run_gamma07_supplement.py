"""Supplement: γ=0.7 experiments for detectable-regime analysis.

Runs all sweep experiments at γ=0.7 (above detection threshold γ̂≈0.5)
to enable meaningful fixed-vs-dynamic trust comparison.

Reuses the same runner infrastructure as run_full_experiment.py.
Saves results as exp_*_g07.json alongside existing γ=0.3 results.

Usage:
    python experiments/run_gamma07_supplement.py
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
GAMMA = 0.7


def run_sim(args_list, label="", timeout=7200):
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


def make_args(mode="attack", attack="bc_deflation", trust="fixed", gamma=GAMMA,
              n_targets=10, network="siouxfalls", output="tmp.json",
              baseline_conv=None, baseline_all=None,
              capture_days=None, theta_dtd=0.004, is_alpha=2.0,
              n_classes=3, delta_all=None, wf_ratio=None, cav_frac=None):
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


def main():
    total_t0 = time.time()
    print("=" * 70)
    print(f"SUPPLEMENT: All sweep experiments at gamma={GAMMA}")
    print(f"(above detection threshold gamma_hat ~ 0.5)")
    print("=" * 70)

    # Load baseline from existing γ=0.3 run
    sig_path = os.path.join(RES_DIR, "exp_signature.json")
    if not os.path.exists(sig_path):
        print("ERROR: exp_signature.json not found. Run run_full_experiment.py first.")
        return
    with open(sig_path) as f:
        sig = json.load(f)
    bc = sig["baseline_converged"]
    ba = sig["baseline_all"]
    print(f"  Baseline TSTT (from γ=0.3 run): {bc:.2f}")
    print(f"  Baseline all-days mean: {ba:.2f}")

    # ================================================================
    # S1: Signature at γ=0.7 (4 scenarios)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"S1: SIGNATURE at gamma={GAMMA} (4 scenarios)")
    print("=" * 70)

    sig_results = {}

    # Fixed trust γ=0.7
    print(f"  [1/4] Fixed trust gamma={GAMMA}...")
    r = run_sim(make_args(trust="fixed", gamma=GAMMA,
                          output=f"exp_g07_sig_fixed.json",
                          baseline_conv=bc, baseline_all=ba,
                          capture_days=CAPTURE_DAYS_SF))
    if r:
        sig_results["fixed"] = {
            "sustained_poatt": r.get("sustained_poatt"),
            "peak_poatt": r.get("peak_poatt"),
            "trust_recovery": r.get("trust_recovery"),
            "tstt": r["tstt"],
            "trust": r.get("trust"),
        }
        print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    # Dynamic trust γ=0.7
    print(f"  [2/4] Dynamic trust gamma={GAMMA}...")
    r = run_sim(make_args(trust="dynamic", gamma=GAMMA,
                          output=f"exp_g07_sig_dynamic.json",
                          baseline_conv=bc, baseline_all=ba,
                          capture_days=CAPTURE_DAYS_SF))
    if r:
        sig_results["dynamic"] = {
            "sustained_poatt": r.get("sustained_poatt"),
            "peak_poatt": r.get("peak_poatt"),
            "trust_recovery": r.get("trust_recovery"),
            "tstt": r["tstt"],
            "trust": r.get("trust"),
        }
        print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    # Dynamic trust γ=1.0 (for comparison)
    print(f"  [3/4] Dynamic trust gamma=1.0...")
    r = run_sim(make_args(trust="dynamic", gamma=1.0,
                          output=f"exp_g07_sig_dynamic_10.json",
                          baseline_conv=bc, baseline_all=ba))
    if r:
        sig_results["dynamic_10"] = {
            "sustained_poatt": r.get("sustained_poatt"),
            "peak_poatt": r.get("peak_poatt"),
            "trust_recovery": r.get("trust_recovery"),
            "tstt": r["tstt"],
            "trust": r.get("trust"),
        }
        print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    # Fixed trust γ=1.0 (for full comparison)
    print(f"  [4/4] Fixed trust gamma=1.0...")
    r = run_sim(make_args(trust="fixed", gamma=1.0,
                          output=f"exp_g07_sig_fixed_10.json",
                          baseline_conv=bc, baseline_all=ba))
    if r:
        sig_results["fixed_10"] = {
            "sustained_poatt": r.get("sustained_poatt"),
            "peak_poatt": r.get("peak_poatt"),
            "tstt": r["tstt"],
        }
        print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    output = {"baseline_converged": bc, "baseline_all": ba,
              "gamma": GAMMA, "scenarios": sig_results}
    with open(os.path.join(RES_DIR, "exp_signature_g07.json"), "w") as f:
        json.dump(output, f)
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")

    # ================================================================
    # S2: N-target sweep at γ=0.7 (12 scenarios)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"S2: N-TARGET SWEEP at gamma={GAMMA} (12 scenarios)")
    print("=" * 70)

    n_targets = [3, 5, 10, 15, 20, 30]
    nt_results = {}
    n = 0
    for nt in n_targets:
        for trust in ["fixed", "dynamic"]:
            n += 1
            key = f"nt{nt}_{trust}"
            print(f"  [{n}/12] n_target={nt}, {trust}...")
            r = run_sim(make_args(trust=trust, gamma=GAMMA,
                                   n_targets=nt,
                                   output=f"exp_g07_nt_{nt}_{trust}.json",
                                   baseline_conv=bc, baseline_all=ba))
            if r:
                nt_results[key] = {
                    "n_target": nt, "trust": trust,
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                }
                print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    with open(os.path.join(RES_DIR, "exp_ntarget_sweep_g07.json"), "w") as f:
        json.dump({"baseline_converged": bc, "gamma": GAMMA,
                   "n_targets": n_targets, "results": nt_results}, f)
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")

    # ================================================================
    # S3: BR sweep at γ=0.7 (12 scenarios)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"S3: BR SWEEP at gamma={GAMMA} (12 scenarios)")
    print("=" * 70)

    deltas = [0, 100, 200, 300, 400, 500]
    br_results = {}
    n = 0
    for d in deltas:
        for trust in ["fixed", "dynamic"]:
            n += 1
            key = f"d{d}_{trust}"
            print(f"  [{n}/12] delta={d}, {trust}...")
            r = run_sim(make_args(trust=trust, gamma=GAMMA,
                                   delta_all=d,
                                   output=f"exp_g07_br_{d}_{trust}.json",
                                   baseline_conv=bc, baseline_all=ba))
            if r:
                br_results[key] = {
                    "delta": d, "trust": trust,
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                }
                print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    with open(os.path.join(RES_DIR, "exp_br_sweep_g07.json"), "w") as f:
        json.dump({"baseline_converged": bc, "gamma": GAMMA,
                   "deltas": deltas, "results": br_results}, f)
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")

    # ================================================================
    # S4: Theta sweep at γ=0.7 (10 scenarios)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"S4: THETA SWEEP at gamma={GAMMA} (10 scenarios)")
    print("=" * 70)

    thetas = [0.002, 0.004, 0.006, 0.008, 0.01]
    th_results = {}
    n = 0
    for th in thetas:
        for trust in ["fixed", "dynamic"]:
            n += 1
            key = f"th{th}_{trust}"
            print(f"  [{n}/10] theta={th}, {trust}...")
            r = run_sim(make_args(trust=trust, gamma=GAMMA,
                                   theta_dtd=th,
                                   output=f"exp_g07_th_{th}_{trust}.json",
                                   baseline_conv=bc, baseline_all=ba))
            if r:
                th_results[key] = {
                    "theta": th, "trust": trust,
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                }
                print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    with open(os.path.join(RES_DIR, "exp_theta_sweep_g07.json"), "w") as f:
        json.dump({"baseline_converged": bc, "gamma": GAMMA,
                   "thetas": thetas, "results": th_results}, f)
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")

    # ================================================================
    # S5: IS sweep at γ=0.7 (8 scenarios)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"S5: IS SWEEP at gamma={GAMMA} (8 scenarios)")
    print("=" * 70)

    alphas = [0, 1, 2, 3]
    is_results = {}
    n = 0
    for a in alphas:
        for trust in ["fixed", "dynamic"]:
            n += 1
            key = f"a{a}_{trust}"
            print(f"  [{n}/8] alpha={a}, {trust}...")
            r = run_sim(make_args(trust=trust, gamma=GAMMA,
                                   is_alpha=float(a),
                                   output=f"exp_g07_is_{a}_{trust}.json",
                                   baseline_conv=bc, baseline_all=ba))
            if r:
                is_results[key] = {
                    "alpha": a, "trust": trust,
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                }
                print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    with open(os.path.join(RES_DIR, "exp_is_sweep_g07.json"), "w") as f:
        json.dump({"baseline_converged": bc, "gamma": GAMMA,
                   "alphas": alphas, "results": is_results}, f)
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")

    # ================================================================
    # S6: Composition sweep at γ=0.7 (12 scenarios)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"S6: COMPOSITION SWEEP at gamma={GAMMA} (12 scenarios)")
    print("=" * 70)

    cavs = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    comp_results = {}
    n = 0
    for cav in cavs:
        if cav >= 1.0:
            fracs = [1.0, 0.0, 0.0]
        elif cav <= 0.0:
            fracs = [0.0, 0.70, 0.30]
        else:
            rem = 1.0 - cav
            fracs = [cav, rem * 0.70, rem * 0.30]
        theta_t = [2.0, 1.0, 0.5]
        lam_max = [0.9, 0.7, 0.3]
        chi = sum(fracs[k] * theta_t[k] * lam_max[k] for k in range(3))

        for trust in ["fixed", "dynamic"]:
            n += 1
            key = f"cav{cav}_{trust}"
            print(f"  [{n}/12] CAV={cav*100:.0f}% {trust}...")
            r = run_sim(make_args(trust=trust, gamma=GAMMA,
                                   cav_frac=cav,
                                   output=f"exp_g07_comp_{cav}_{trust}.json",
                                   baseline_conv=bc, baseline_all=ba))
            if r:
                comp_results[key] = {
                    "cav_frac": cav, "fractions": fracs, "chi": chi,
                    "trust": trust,
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                }
                print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    with open(os.path.join(RES_DIR, "exp_composition_g07.json"), "w") as f:
        json.dump({"baseline_converged": bc, "gamma": GAMMA,
                   "cavs": cavs, "results": comp_results}, f)
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")

    # ================================================================
    # S7: Trust asymmetry at γ=0.7 (6 scenarios)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"S7: TRUST ASYMMETRY at gamma={GAMMA} (6 scenarios)")
    print("=" * 70)

    ratios = [1, 3, 5, 7, 10, 15]
    ta_results = {}
    n = 0
    for ratio in ratios:
        n += 1
        key = f"r{ratio}"
        print(f"  [{n}/6] w_f/w_s={ratio}...")
        r = run_sim(make_args(trust="dynamic", gamma=GAMMA,
                               wf_ratio=ratio,
                               output=f"exp_g07_trust_{ratio}.json",
                               baseline_conv=bc, baseline_all=ba))
        if r:
            ta_results[key] = {
                "ratio": ratio,
                "sustained_poatt": r.get("sustained_poatt"),
                "peak_poatt": r.get("peak_poatt"),
                "trust_recovery": r.get("trust_recovery"),
                "tstt": r["tstt"],
                "trust": r.get("trust"),
            }
            print(f"    Sus={r.get('sustained_poatt','?'):.4f}, "
                  f"TrustRec={r.get('trust_recovery')}")

    with open(os.path.join(RES_DIR, "exp_trust_sensitivity_g07.json"), "w") as f:
        json.dump({"baseline_converged": bc, "gamma": GAMMA,
                   "ratios": ratios, "results": ta_results}, f)
    print(f"  Elapsed: {time.time()-total_t0:.0f}s")

    # ================================================================
    # S8: Anaheim at γ=0.7 (2 scenarios)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"S8: ANAHEIM at gamma={GAMMA} (2 scenarios)")
    print("=" * 70)

    # Load Anaheim baseline from existing run
    ana_path = os.path.join(RES_DIR, "exp_anaheim.json")
    if os.path.exists(ana_path):
        with open(ana_path) as f:
            ana_existing = json.load(f)
        ana_bc = ana_existing["baseline_converged"]
        ana_ba = ana_existing["baseline_all"]
    else:
        print("  WARNING: exp_anaheim.json not found, running Anaheim baseline...")
        r = run_sim(make_args(mode="baseline", network="anaheim",
                               output="exp_g07_ana_baseline.json"),
                    timeout=14400)
        if r:
            ana_bc = float(np.mean(r["tstt"][29:50]))
            ana_ba = float(np.mean(r["tstt"][:50]))
        else:
            print("  SKIP Anaheim")
            ana_bc = None

    ana_results = {}
    if ana_bc:
        for trust in ["fixed", "dynamic"]:
            print(f"  Anaheim {trust} gamma={GAMMA}...")
            r = run_sim(make_args(trust=trust, gamma=GAMMA,
                                   network="anaheim",
                                   output=f"exp_g07_ana_{trust}.json",
                                   baseline_conv=ana_bc, baseline_all=ana_ba),
                        timeout=14400)
            if r:
                ana_results[f"bc_defl_{trust}"] = {
                    "sustained_poatt": r.get("sustained_poatt"),
                    "peak_poatt": r.get("peak_poatt"),
                    "trust_recovery": r.get("trust_recovery"),
                }
                print(f"    Sus={r.get('sustained_poatt','?'):.4f}")

    with open(os.path.join(RES_DIR, "exp_anaheim_g07.json"), "w") as f:
        json.dump({"baseline_converged": ana_bc, "gamma": GAMMA,
                   "scenarios": ana_results}, f)

    # ================================================================
    # SUMMARY
    # ================================================================
    total = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"SUPPLEMENT COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total:.0f}s ({total/3600:.1f}h)")
    print(f"  Gamma: {GAMMA}")

    # Count output files
    g07_files = [f for f in os.listdir(RES_DIR) if "g07" in f and f.endswith(".json")]
    print(f"  Output files: {len(g07_files)} g07 JSONs")

    # Print key results
    sig_g07 = os.path.join(RES_DIR, "exp_signature_g07.json")
    if os.path.exists(sig_g07):
        with open(sig_g07) as f:
            s = json.load(f)
        sc = s.get("scenarios", {})
        fx = sc.get("fixed", {}).get("sustained_poatt")
        dy = sc.get("dynamic", {}).get("sustained_poatt")
        if fx and dy and fx > 1:
            tia = (1 - (dy - 1) / (fx - 1)) * 100
            print(f"\n  γ=0.7 Signature:")
            print(f"    Fixed: {fx:.4f}, Dynamic: {dy:.4f}")
            print(f"    TIA: {tia:.0f}%")

    ta_g07 = os.path.join(RES_DIR, "exp_trust_sensitivity_g07.json")
    if os.path.exists(ta_g07):
        with open(ta_g07) as f:
            t = json.load(f)
        print(f"\n  γ=0.7 Trust Asymmetry:")
        for r in t.get("ratios", []):
            d = t["results"].get(f"r{r}", {})
            print(f"    w_f/w_s={r}: PoAtt={d.get('sustained_poatt','?')}, "
                  f"TrustRec={d.get('trust_recovery','?')}")

    print(f"\n=== ALL SUPPLEMENT STEPS COMPLETE ===")


if __name__ == "__main__":
    main()
