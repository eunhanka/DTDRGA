"""Single LWR simulation runner — called as subprocess to avoid memory fragmentation.

Usage:
  python lwr_single_sim.py --mode baseline --output result.json
  python lwr_single_sim.py --mode attack --gamma 0.3 --trust fixed --output result.json
  python lwr_single_sim.py --mode attack --gamma 0.3 --trust dynamic --output result.json
  python lwr_single_sim.py --mode attack --gamma 0.3 --trust dynamic --attack sybil --output result.json

  # Step 3 experiments:
  python lwr_single_sim.py --mode attack --gamma 0.3 --trust fixed --n-classes 1 --delta 200 --output r.json
  python lwr_single_sim.py --mode attack --gamma 0.3 --trust dynamic --n-classes 1 --wf 0.5 --ws 0.1 --output r.json
  python lwr_single_sim.py --mode attack --gamma 0.3 --trust fixed --theta-dtd 0.01 --output r.json
  python lwr_single_sim.py --mode attack --gamma 0.3 --trust fixed --is-alpha 0 --output r.json
  python lwr_single_sim.py --mode attack --gamma 0.3 --trust dynamic --cav-frac 0.5 --output r.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from network.network import load_sioux_falls
import dataclasses
from dnl.lwr_dnl import LWRDNL
from dtd.dtd_simulator import DTDParams, DTDSimulator
from dtd.dtd_trust_simulator import DTDTrustSimulator
from trust.trust_dynamics import TrustParams
from attack.attack_models import (
    AttackConfig, NoAttack,
    SybilInflationAttack, StrategicSybilAttack, RandomSybilAttack,
    BCSybilInflationAttack, BCSybilDeflationAttack,
    BCLinkAdditiveDeflationAttack, PathBCDeflationAttack,
)
from utils.metrics import ResilienceAnalyzer

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "sioux_falls")

N_DAYS = 200
ATTACK_START = 50
ATTACK_END = 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["baseline", "attack", "disruption"])
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--trust", default="dynamic", choices=["fixed", "dynamic"])
    parser.add_argument("--attack", default="bc_deflation", choices=["sybil", "strategic", "random", "bc_inflation", "bc_deflation", "link_additive", "path_bc"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-conv", type=float, default=None)
    parser.add_argument("--baseline-all", type=float, default=None)
    parser.add_argument("--capture-flows", action="store_true")
    parser.add_argument("--capture-days", type=str, default=None, help="Comma-separated list of 0-indexed days to capture link flows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for RandomSybilAttack")
    # Step 3 experiment parameters
    parser.add_argument("--n-classes", type=int, default=3, choices=[1, 3])
    parser.add_argument("--delta", type=float, default=None, help="Indifference band (single-class)")
    parser.add_argument("--wf", type=float, default=None, help="w_f (single-class)")
    parser.add_argument("--ws", type=float, default=None, help="w_s (single-class)")
    # 3-class overrides
    parser.add_argument("--delta-all", type=float, default=None, help="Override indifference band for ALL 3 classes")
    parser.add_argument("--wf-ratio", type=float, default=None, help="w_f/w_s ratio for ALL 3 classes (w_f = ratio * w_s)")
    parser.add_argument("--network", default="siouxfalls", choices=["siouxfalls", "anaheim"], help="Network to use")
    parser.add_argument("--n-target-links", type=int, default=10, help="Number of target links for attack")
    parser.add_argument("--theta-dtd", type=float, default=0.004, help="DTD theta")
    parser.add_argument("--is-alpha", type=float, default=2.0, help="IS alpha (0=disabled)")
    parser.add_argument("--cav-frac", type=float, default=None, help="CAV penetration fraction")
    # Smooth trust model (supplementary experiments)
    parser.add_argument("--smooth-mode", action="store_true", help="Use smooth trust model")
    parser.add_argument("--eta", type=float, default=10.0, help="Smooth trust eta parameter")
    # Trust accuracy tolerance (epsilon) override; default leaves TrustParams default (360 s = 0.1 h)
    parser.add_argument("--epsilon-sec", type=float, default=None,
                        help="Trust accuracy tolerance in seconds (default: TrustParams default = 360 s)")
    parser.add_argument("--save-errors", action="store_true",
                        help="Include daily flow-weighted guidance error in output JSON")
    args = parser.parse_args()

    if args.network == "anaheim":
        from exp_anaheim import load_anaheim
        ANA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "anaheim")
        net, demand, sim_params = load_anaheim(ANA_DIR)
    else:
        net, demand, sim_params = load_sioux_falls(DATA_DIR)
    dtd_p = DTDParams(n_days=N_DAYS, theta=args.theta_dtd, memory_days=6, memory_weight=0.7)

    use_is = args.is_alpha > 0
    is_alpha = args.is_alpha

    # Build trust params based on experiment type
    if args.n_classes == 1:
        # Single-class experiments (BR sweep, theta, IS, trust sensitivity)
        delta = args.delta if args.delta is not None else 0.0
        if args.trust == "fixed":
            tp_use = TrustParams(
                n_classes=1, class_fractions=np.array([1.0]),
                w_s=np.array([1e-10]), w_f=np.array([1e-10]),
                lambda_forget=0.9999999,
                lambda_max=np.array([0.7]),
                indifference_band=np.array([delta]),
                alpha_init=np.array([100.0]),
                beta_init=np.array([0.01]),
            )
        else:
            wf = args.wf if args.wf is not None else 0.5
            ws = args.ws if args.ws is not None else 0.1
            tp_use = TrustParams(
                n_classes=1, class_fractions=np.array([1.0]),
                w_s=np.array([ws]), w_f=np.array([wf]),
                lambda_max=np.array([0.7]),
                indifference_band=np.array([delta]),
            )
    elif args.cav_frac is not None:
        # Composition sweep (3 classes with custom fractions)
        cav_f = args.cav_frac
        remaining = 1.0 - cav_f
        if cav_f >= 1.0:
            fracs = np.array([1.0, 0.0, 0.0])
        elif cav_f <= 0.0:
            fracs = np.array([0.0, 0.70, 0.30])
        else:
            fracs = np.array([cav_f, remaining * 0.70, remaining * 0.30])
        fracs = fracs / fracs.sum()
        if args.trust == "fixed":
            tp_use = TrustParams(
                n_classes=3, class_fractions=fracs,
                w_s=np.array([1e-10, 1e-10, 1e-10]),
                w_f=np.array([1e-10, 1e-10, 1e-10]),
                lambda_forget=0.9999999,
                lambda_max=np.array([0.90, 0.70, 0.30]),
                indifference_band=np.array([0.0, 200.0, 400.0]),
                alpha_init=np.full(3, 100.0),
                beta_init=np.full(3, 0.01),
            )
        else:
            tp_use = TrustParams(
                n_classes=3, class_fractions=fracs,
                w_s=np.array([0.16, 0.10, 0.06]),
                w_f=np.array([0.80, 0.50, 0.30]),
                lambda_max=np.array([0.90, 0.70, 0.30]),
                indifference_band=np.array([0.0, 200.0, 400.0]),
            )
    else:
        # Standard 3-class
        ws_default = np.array([0.16, 0.10, 0.06])
        wf_default = np.array([0.80, 0.50, 0.30])
        ib_default = np.array([0.0, 200.0, 400.0])

        # Apply overrides
        if args.wf_ratio is not None:
            wf_default = ws_default * args.wf_ratio
        if args.delta_all is not None:
            ib_default = np.full(3, args.delta_all)

        if args.trust == "dynamic":
            tp_use = TrustParams(
                n_classes=3,
                class_fractions=np.array([0.10, 0.60, 0.30]),
                w_s=ws_default,
                w_f=wf_default,
                lambda_max=np.array([0.90, 0.70, 0.30]),
                indifference_band=ib_default,
            )
        else:
            tp_use = TrustParams(
                n_classes=3,
                class_fractions=np.array([0.10, 0.60, 0.30]),
                w_s=np.array([1e-10, 1e-10, 1e-10]),
                w_f=np.array([1e-10, 1e-10, 1e-10]),
                lambda_forget=0.9999999,
                lambda_max=np.array([0.90, 0.70, 0.30]),
                indifference_band=ib_default,
                alpha_init=np.full(3, 100.0),
                beta_init=np.full(3, 0.01),
            )

    # Apply smooth trust mode if requested
    if args.smooth_mode:
        tp_use.smooth_mode = True
        tp_use.eta = args.eta

    # Override accuracy tolerance epsilon if requested (default unchanged)
    if args.epsilon_sec is not None:
        tp_use.epsilon = float(args.epsilon_sec)

    # Handle disruption mode separately
    if args.mode == "disruption":
        # Paper §5.2.1: "capacity is reduced by one-third on the same
        # top-10 topological BC links" — match attack target selection
        import networkx as nx
        G = nx.DiGraph()
        for i in range(net.n_links):
            u = int(net.adjacency_list[i, 0])
            v = int(net.adjacency_list[i, 1])
            G.add_edge(u, v, weight=max(float(net.link_fft[i]), 0.001),
                       link_idx=i)
        topo_bc = nx.edge_betweenness_centrality(G, weight='weight',
                                                  normalized=False)
        link_bc = np.zeros(net.n_links)
        for (u, v), bc_val in topo_bc.items():
            link_bc[G[u][v]['link_idx']] = bc_val
        target_links = np.argsort(link_bc)[::-1][:args.n_target_links]

        disrupted_net = dataclasses.replace(net, link_capacity=net.link_capacity.copy())
        for idx in target_links:
            disrupted_net.link_capacity[idx] *= 2.0 / 3.0
        print(f"  Disruption: {len(target_links)} top-BC links degraded to 2/3 capacity")
        print(f"  Target links (0-idx): {sorted(target_links.tolist())}")

        dtd_phys = DTDParams(
            n_days=N_DAYS, theta=args.theta_dtd, memory_days=6, memory_weight=0.7,
            disruption_days=(ATTACK_START + 1, ATTACK_END),
        )
        t0 = time.time()
        sim = DTDSimulator(net, demand, LWRDNL(verbose=False), dtd_phys, sim_params,
                           disrupted_network=disrupted_net)
        day_results = sim.run()
        tstt_arr = np.array([dr.tstt for dr in day_results])
        trust_arr = np.ones((N_DAYS, 3))
        elapsed = time.time() - t0

        result = {
            "elapsed": elapsed,
            "tstt": tstt_arr.tolist(),
            "trust": trust_arr.tolist(),
            "info_weight": np.zeros((N_DAYS, 3)).tolist(),
            "baseline_converged": float(tstt_arr[29:50].mean()),
            "baseline_all": float(tstt_arr[:ATTACK_START].mean()),
        }
        if args.baseline_conv is not None:
            ra = ResilienceAnalyzer(
                tstt_arr, trust_arr, ATTACK_START, ATTACK_END,
                baseline_tstt=args.baseline_all,
                baseline_converged=args.baseline_conv,
            )
            result["sustained_poatt"] = ra.sustained_poatt()
            result["peak_poatt"] = ra.peak_poatt()
            result["overshoot"] = ra.transient_overshoot_ratio()
            result["triangle"] = ra.resilience_triangle_area()
            result["trust_recovery"] = ra.trust_recovery_delay()

        with open(args.output, "w") as f:
            json.dump(result, f)
        print(f"Done: {elapsed:.0f}s")
        if "sustained_poatt" in result:
            print(f"  Sustained PoAtt: {result['sustained_poatt']:.4f}")
            print(f"  Peak PoAtt: {result['peak_poatt']:.4f}")
        return

    if args.mode == "baseline":
        atk = NoAttack()
    else:
        atk_cfg = AttackConfig(start_day=ATTACK_START, end_day=ATTACK_END, bias_fraction=args.gamma)
        ntl = args.n_target_links
        if args.attack == "sybil":
            atk = SybilInflationAttack(atk_cfg, n_target_links=ntl)
        elif args.attack == "strategic":
            atk = StrategicSybilAttack(atk_cfg, n_target_links=ntl)
        elif args.attack == "bc_inflation":
            atk = BCSybilInflationAttack(atk_cfg, n_target_links=ntl)
        elif args.attack == "bc_deflation":
            atk = BCSybilDeflationAttack(atk_cfg, n_target_links=ntl)
        elif args.attack == "link_additive":
            atk = BCLinkAdditiveDeflationAttack(atk_cfg, n_target_links=ntl)
        elif args.attack == "path_bc":
            atk = PathBCDeflationAttack(atk_cfg, n_target_links=ntl)
        else:
            atk = RandomSybilAttack(atk_cfg, n_target_links=ntl, seed=args.seed)

    if args.capture_days:
        capture_days = [int(x.strip()) for x in args.capture_days.split(",")]
    elif args.capture_flows:
        capture_days = [39, 54, 74, 94, 124, 174]
    else:
        capture_days = None

    t0 = time.time()
    sim = DTDTrustSimulator(
        net, demand, LWRDNL(verbose=False), dtd_p, sim_params,
        trust_params=tp_use, attack_policy=atk,
        use_info_sharing=use_is, is_alpha=is_alpha,
    )
    r = sim.run(capture_link_flow_days=capture_days)
    elapsed = time.time() - t0

    result = {
        "elapsed": elapsed,
        "tstt": r.tstt.tolist(),
        "trust": r.trust.tolist(),
        "info_weight": r.info_weight.tolist(),
        "baseline_converged": float(r.tstt[29:50].mean()),
        "baseline_all": float(r.tstt[:ATTACK_START].mean()),
    }

    # Compute metrics if baseline values provided
    if args.baseline_conv is not None:
        ra = ResilienceAnalyzer(
            r.tstt, r.trust, ATTACK_START, ATTACK_END,
            baseline_tstt=args.baseline_all,
            baseline_converged=args.baseline_conv,
        )
        result["sustained_poatt"] = ra.sustained_poatt()
        result["peak_poatt"] = ra.peak_poatt()
        result["overshoot"] = ra.transient_overshoot_ratio()
        result["triangle"] = ra.resilience_triangle_area()
        result["trust_recovery"] = ra.trust_recovery_delay()
        result["mean_attack_tstt"] = float(r.tstt[ATTACK_START:ATTACK_END].mean())
        result["trust_end_attack"] = float(r.trust[ATTACK_END - 1].mean())
        result["trust_class"] = r.trust[ATTACK_END - 1].tolist()

    if (args.capture_flows or args.capture_days) and r.link_flows_at:
        result["link_flows"] = {str(k): v.tolist() for k, v in r.link_flows_at.items()}

    if args.save_errors and r.guidance_error is not None:
        result["guidance_error"] = r.guidance_error.tolist()

    with open(args.output, "w") as f:
        json.dump(result, f)

    print(f"Done: {elapsed:.0f}s")
    if "sustained_poatt" in result:
        print(f"  Sustained PoAtt: {result['sustained_poatt']:.4f}")
        print(f"  Peak PoAtt: {result['peak_poatt']:.4f}")

if __name__ == "__main__":
    main()
