"""Step 6: Anaheim Network Setup and Baseline.

Load Anaheim network data and run 3 scenarios with BPRDNL.
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from network.network import (
    NetworkData, DemandData, SimulationParams,
    _squeeze_scalar, _flatten_col, _extract_cell_list,
    _extract_path_link_junctions,
)
from dnl.dnl import BPRDNL
from dtd.dtd_simulator import DTDParams
from dtd.dtd_trust_simulator import DTDTrustSimulator
from trust.trust_dynamics import TrustParams
from attack.attack_models import AttackConfig, SybilInflationAttack, NoAttack
from utils.metrics import ResilienceAnalyzer

ANAHEIM_DIR = os.path.join(os.path.dirname(__file__), os.pardir,
                            "data", "anaheim")
SF_DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "sioux_falls")
FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_anaheim(data_dir):
    """Load Anaheim network from DTA reference data.

    Returns (NetworkData, DemandData, SimulationParams) or raises.
    """
    net_file = os.path.join(data_dir, "Anaheim30719_pp.mat")
    od_file = os.path.join(data_dir, "OD_info.mat")
    param_file = os.path.join(data_dir, "Network_planning_parameters.mat")
    due_file = os.path.join(data_dir, "DUE_out.mat")

    for f in [net_file, od_file, param_file, due_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing: {f}")

    # Load network
    nf = scipy.io.loadmat(net_file, squeeze_me=False)

    link_s = nf["link"][0, 0]
    node_s = nf["node"][0, 0]
    source_s = nf["source"][0, 0]
    sink_s = nf["sink"][0, 0]
    path_s = nf["path"][0, 0]

    n_links = int(_squeeze_scalar(link_s["count"]))
    n_nodes = int(_squeeze_scalar(node_s["count"]))
    n_sources = int(_squeeze_scalar(source_s["count"]))
    n_sinks = int(_squeeze_scalar(sink_s["count"]))
    n_paths = int(_squeeze_scalar(path_s["count"]))

    link_capacity = _flatten_col(link_s["capacity"]).astype(np.float64)
    link_fft = _flatten_col(link_s["FFT"]).astype(np.float64)
    link_length = _flatten_col(link_s["length"]).astype(np.float64)
    link_index = _flatten_col(link_s["index"]).astype(np.intp) - 1

    tail = _flatten_col(link_s["tailNode"]).astype(np.intp) - 1
    head = _flatten_col(link_s["headNode"]).astype(np.intp) - 1
    adjacency_list = np.column_stack([tail, head])

    path_link_raw = np.asarray(nf["pathList"])
    path_link_matrix = path_link_raw.astype(np.intp) - 1  # 0-indexed, pad=-1
    path_link_count = _flatten_col(path_s["linkCount"]).astype(np.intp)
    path_source_node = _flatten_col(path_s["sourceNode"]).astype(np.intp) - 1
    path_sink_node = _flatten_col(path_s["sinkNode"]).astype(np.intp) - 1
    path_source_index = _flatten_col(path_s["sourceIndex"]).astype(np.intp) - 1
    path_sink_index = _flatten_col(path_s["sinkIndex"]).astype(np.intp) - 1

    links_in = _extract_cell_list(nf["linksIn"], to_0index=True)
    links_out = _extract_cell_list(nf["linksOut"], to_0index=True)
    num_links_in = _flatten_col(nf["numLinksIn"]).astype(np.intp)
    num_links_out = _flatten_col(nf["numLinksOut"]).astype(np.intp)

    source_nodes = _flatten_col(source_s["nodes"]).astype(np.intp) - 1
    sink_nodes = _flatten_col(sink_s["nodes"]).astype(np.intp) - 1
    source_index = _flatten_col(source_s["index"]).astype(np.intp) - 1
    sink_index = _flatten_col(sink_s["index"]).astype(np.intp) - 1

    # Node coordinates (may not exist for Anaheim)
    if "X" in node_s.dtype.names:
        node_x = _flatten_col(node_s["X"]).astype(np.float64)
        node_y = _flatten_col(node_s["Y"]).astype(np.float64)
    else:
        node_x = np.zeros(n_nodes)
        node_y = np.zeros(n_nodes)

    sp_cell = node_s["signalPriorities"]
    signal_priorities = [
        _flatten_col(sp_cell[i, 0]).astype(np.float64) for i in range(n_nodes)
    ]

    path_links_in = _extract_path_link_junctions(nf["pathLinksIn"], to_0index=True)
    path_links_out = _extract_path_link_junctions(nf["pathLinksOut"], to_0index=True)
    path_source_link_idx = _flatten_col(nf["pathSourceLinkIdx"]).astype(np.intp) - 1
    path_sink_link_idx_raw = nf.get("pathSinkLinkIdx", None)
    if path_sink_link_idx_raw is not None:
        path_sink_link_idx = _flatten_col(path_sink_link_idx_raw).astype(np.intp) - 1
    else:
        path_sink_link_idx = _flatten_col(nf["sinkOutIdx"]).astype(np.intp) - 1

    # Load OD info
    od_raw = scipy.io.loadmat(od_file, squeeze_me=False)
    od_set_raw = np.asarray(od_raw["OD_set"])
    od_set = od_set_raw.astype(np.intp) - 1
    od_path_set = _extract_cell_list(od_raw["ODpath_set"], to_0index=True)
    n_od = od_set.shape[0]

    # Load demand
    np_raw = scipy.io.loadmat(param_file, squeeze_me=False)
    od_demand = _flatten_col(np_raw["OD_demand"]).astype(np.float64)
    target_arrival = _flatten_col(np_raw["T_A"]).astype(np.float64) * 3600.0

    # Load DUE solution for initial departures
    due = scipy.io.loadmat(due_file, squeeze_me=False)
    h_final = np.asarray(due["h_final"], dtype=np.float64)  # [30719, 100]
    dt_val = float(_squeeze_scalar(due["dt"]))

    # Simulation params
    time_horizon = (0.0, dt_val * h_final.shape[1])
    n_steps = h_final.shape[1]
    departure_window = 900.0
    steps_per_window = max(1, int(departure_window / dt_val))
    n_windows = max(1, n_steps // steps_per_window)

    network = NetworkData(
        n_nodes=n_nodes, n_links=n_links, n_paths=n_paths, n_od=n_od,
        link_capacity=link_capacity, link_fft=link_fft, link_length=link_length,
        adjacency_list=adjacency_list, path_link_matrix=path_link_matrix,
        path_link_count=path_link_count, path_source_node=path_source_node,
        path_sink_node=path_sink_node, links_in=links_in, links_out=links_out,
        num_links_in=num_links_in, num_links_out=num_links_out,
        source_nodes=source_nodes, sink_nodes=sink_nodes,
        source_index=source_index, sink_index=sink_index,
        n_sources=n_sources, n_sinks=n_sinks,
        node_x=node_x, node_y=node_y, signal_priorities=signal_priorities,
        path_links_in=path_links_in, path_links_out=path_links_out,
        path_source_link_idx=path_source_link_idx,
        path_sink_link_idx=path_sink_link_idx,
        path_source_index=path_source_index, path_sink_index=path_sink_index,
        link_index=link_index,
    )

    demand_data = DemandData(
        od_set=od_set, od_path_set=od_path_set,
        od_demand=od_demand, target_arrival=target_arrival,
    )

    params = SimulationParams(
        dt=dt_val, departure_window=departure_window,
        n_steps=n_steps, n_windows=n_windows,
        steps_per_window=steps_per_window,
        time_horizon=time_horizon,
        initial_path_departures=h_final,
    )

    return network, demand_data, params


# ======================================================================
# Main
# ======================================================================
print("Checking Anaheim data availability...")
try:
    net_a, demand_a, sim_params_a = load_anaheim(ANAHEIM_DIR)
    print(f"Anaheim loaded: {net_a.n_nodes} nodes, {net_a.n_links} links, "
          f"{net_a.n_paths} paths, {net_a.n_od} OD pairs")
    print(f"  dt={sim_params_a.dt}s, n_steps={sim_params_a.n_steps}, "
          f"n_windows={sim_params_a.n_windows}, steps_per_window={sim_params_a.steps_per_window}")
    print(f"  Initial departures shape: {sim_params_a.initial_path_departures.shape}")
    anaheim_available = True
except Exception as e:
    print(f"Anaheim loading failed: {e}")
    anaheim_available = False

if anaheim_available:
    dnl = BPRDNL()
    N_DAYS = 200
    ATTACK_START = 50
    ATTACK_END = 100

    dtd_base = DTDParams(
        n_days=N_DAYS, theta=0.02, memory_days=3, memory_weight=0.7,
    )

    attack_config = AttackConfig(
        start_day=ATTACK_START, end_day=ATTACK_END, bias_fraction=0.3,
    )

    tp_default = TrustParams()

    # ======================================================================
    # Scenario 1: Baseline (no attack)
    # ======================================================================
    print("\nScenario 1: Anaheim baseline (no attack)...")
    t0 = time.time()
    sim1 = DTDTrustSimulator(
        net_a, demand_a, dnl, dtd_base, sim_params_a,
        trust_params=tp_default, attack_policy=NoAttack(),
        use_info_sharing=True, is_alpha=2.0,
    )
    r1 = sim1.run(callbacks=[
        lambda d, tstt, ts: print(f"  Day {d+1:3d}  TSTT = {tstt:.1f}")
        if (d + 1) % 25 == 0 else None
    ])
    t1 = time.time() - t0
    baseline_tstt = r1.tstt[:ATTACK_START].mean()
    print(f"  Done: {t1:.0f}s ({t1/60:.1f} min), baseline TSTT={baseline_tstt:.1f}")

    # ======================================================================
    # Scenario 2: SybilInflation + fixed trust
    # ======================================================================
    print("\nScenario 2: Anaheim SybilInflation + fixed trust...")
    tp_fixed = TrustParams(
        w_s=np.array([1e-10, 1e-10, 1e-10]),
        w_f=np.array([1e-10, 1e-10, 1e-10]),
        lambda_forget=0.9999999,
    )
    t0 = time.time()
    atk2 = SybilInflationAttack(attack_config, n_target_links=20)
    sim2 = DTDTrustSimulator(
        net_a, demand_a, dnl, dtd_base, sim_params_a,
        trust_params=tp_fixed, attack_policy=atk2,
        use_info_sharing=True, is_alpha=2.0,
    )
    r2 = sim2.run(callbacks=[
        lambda d, tstt, ts: print(f"  Day {d+1:3d}  TSTT = {tstt:.1f}")
        if (d + 1) % 25 == 0 else None
    ])
    t2 = time.time() - t0
    print(f"  Done: {t2:.0f}s ({t2/60:.1f} min)")

    # ======================================================================
    # Scenario 3: SybilInflation + dynamic trust (3 classes)
    # ======================================================================
    print("\nScenario 3: Anaheim SybilInflation + dynamic trust...")
    t0 = time.time()
    atk3 = SybilInflationAttack(attack_config, n_target_links=20)
    sim3 = DTDTrustSimulator(
        net_a, demand_a, dnl, dtd_base, sim_params_a,
        trust_params=tp_default, attack_policy=atk3,
        use_info_sharing=True, is_alpha=2.0,
    )
    r3 = sim3.run(callbacks=[
        lambda d, tstt, ts: print(f"  Day {d+1:3d}  TSTT = {tstt:.1f}")
        if (d + 1) % 25 == 0 else None
    ])
    t3 = time.time() - t0
    print(f"  Done: {t3:.0f}s ({t3/60:.1f} min)")

    # Metrics
    ra2 = ResilienceAnalyzer(r2.tstt, r2.trust, ATTACK_START, ATTACK_END,
                              baseline_tstt=baseline_tstt)
    ra3 = ResilienceAnalyzer(r3.tstt, r3.trust, ATTACK_START, ATTACK_END,
                              baseline_tstt=baseline_tstt)

    poatt_fixed = ra2.price_of_attack()
    poatt_dyn = ra3.price_of_attack()
    tri_fixed = ra2.resilience_triangle_area()
    tri_dyn = ra3.resilience_triangle_area()
    trust_delay = ra3.trust_recovery_delay()

    # Also load Sioux Falls results for comparison
    from network.network import load_sioux_falls
    net_sf, demand_sf, sim_params_sf = load_sioux_falls(SF_DATA_DIR)
    dnl_sf = BPRDNL()  # fresh DNL instance for SF (different incidence matrix)

    dtd_sf = DTDParams(n_days=N_DAYS, theta=0.02, memory_days=3, memory_weight=0.7)

    sim_sf_base = DTDTrustSimulator(
        net_sf, demand_sf, dnl_sf, dtd_sf, sim_params_sf,
        trust_params=tp_default, attack_policy=NoAttack(),
        use_info_sharing=True, is_alpha=2.0,
    )
    r_sf_base = sim_sf_base.run()
    sf_baseline = r_sf_base.tstt[:ATTACK_START].mean()

    atk_sf = SybilInflationAttack(attack_config, n_target_links=10)
    sim_sf_dyn = DTDTrustSimulator(
        net_sf, demand_sf, dnl_sf, dtd_sf, sim_params_sf,
        trust_params=tp_default, attack_policy=atk_sf,
        use_info_sharing=True, is_alpha=2.0,
    )
    r_sf_dyn = sim_sf_dyn.run()
    ra_sf = ResilienceAnalyzer(r_sf_dyn.tstt, r_sf_dyn.trust, ATTACK_START, ATTACK_END,
                                baseline_tstt=sf_baseline)
    sf_poatt = ra_sf.price_of_attack()
    sf_tri = ra_sf.resilience_triangle_area()
    sf_trust_delay = ra_sf.trust_recovery_delay()

    # ======================================================================
    # Figure: 2x2
    # ======================================================================
    days = np.arange(1, N_DAYS + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) TSTT trajectories
    ax = axes[0, 0]
    ax.plot(days, r1.tstt, "k--", lw=1, alpha=0.5, label="Baseline")
    ax.plot(days, r2.tstt, color="#e67e22", lw=1.5, label="Sybil + Fixed Trust")
    ax.plot(days, r3.tstt, color="#8e44ad", lw=1.5, label="Sybil + Dynamic Trust")
    ax.axvspan(ATTACK_START + 1, ATTACK_END, alpha=0.08, color="red")
    ax.set_title("(a) Anaheim TSTT Trajectories", fontsize=12)
    ax.set_ylabel("TSTT [veh-hours]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, N_DAYS)

    # (b) Trust evolution (Scenario 3)
    ax = axes[0, 1]
    class_colors = ["#e74c3c", "#3498db", "#2ecc71"]
    class_names = ["CAV (w_f=0.8)", "App (w_f=0.5)", "Exp (w_f=0.3)"]
    for k in range(3):
        ax.plot(days, r3.trust[:, k], color=class_colors[k], lw=1.5,
                label=class_names[k])
    ax.axvspan(ATTACK_START + 1, ATTACK_END, alpha=0.08, color="red")
    ax.set_title("(b) Anaheim Trust Evolution", fontsize=12)
    ax.set_ylabel("Expected Trust")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(1, N_DAYS)

    # (c) PoAtt comparison: Sioux Falls vs Anaheim
    ax = axes[1, 0]
    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width/2, [sf_poatt, sf_tri], width,
                   label="Sioux Falls", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width/2, [poatt_dyn, tri_dyn], width,
                   label="Anaheim", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["PoAtt", "Res. Triangle"])
    ax.set_title("(c) Sioux Falls vs Anaheim (Dynamic Trust)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    # (d) Metrics table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = [
        ["PoAtt (Fixed)", f"{poatt_fixed:.4f}", f"{sf_poatt:.4f}"],
        ["PoAtt (Dynamic)", f"{poatt_dyn:.4f}", f"{sf_poatt:.4f}"],
        ["Res. Triangle", f"{tri_dyn:.2f}", f"{sf_tri:.2f}"],
        ["Trust Recovery", f"{trust_delay}", f"{sf_trust_delay}"],
    ]
    table = ax.table(cellText=table_data,
                     colLabels=["Metric", "Anaheim", "Sioux Falls"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax.set_title("(d) Resilience Metrics", fontsize=12, pad=20)

    fig.suptitle("Experiment: Anaheim Network (SybilInflation, BPR)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "exp_anaheim.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: figures/exp_anaheim.png")

    # Print summary
    print("\n" + "=" * 60)
    print("ANAHEIM RESULTS")
    print("=" * 60)
    print(f"  Anaheim: {net_a.n_nodes} nodes, {net_a.n_links} links, "
          f"{net_a.n_paths} paths, {net_a.n_od} OD pairs")
    print(f"  Baseline TSTT: {baseline_tstt:.1f}")
    print(f"  PoAtt (fixed trust): {poatt_fixed:.4f}")
    print(f"  PoAtt (dynamic trust): {poatt_dyn:.4f}")
    print(f"  Resilience Triangle: {tri_dyn:.2f}")
    print(f"  Trust Recovery Delay: {trust_delay}")
    print(f"\n  Sioux Falls comparison:")
    print(f"  SF PoAtt: {sf_poatt:.4f}, SF Triangle: {sf_tri:.2f}, SF Trust Del: {sf_trust_delay}")
    print(f"  Runtime: Sc1={t1:.0f}s, Sc2={t2:.0f}s, Sc3={t3:.0f}s")

else:
    # Anaheim not available - create placeholder
    print("\nAnaheim path enumeration data not available in DTA repo.")
    print("The DTA repo contains DUE solver data (link/node level)")
    print("but pre-enumerated paths are needed for DTD simulation.")
    print("Anaheim experiments require path generation as future work.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5,
            "Anaheim Network Experiment\n\n"
            "Data Status: Path enumeration available (30,719 paths)\n"
            "but loading failed.\n\n"
            "Network: 416 nodes, 914 links, 1,406 OD pairs\n"
            "Paths: 30,719 (max 59 links per path)\n\n"
            "See error above for details.",
            transform=ax.transAxes, fontsize=12, ha="center", va="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(os.path.join(FIG_DIR, "exp_anaheim_placeholder.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/exp_anaheim_placeholder.png")

print("\n=== STEP 6 COMPLETE ===")
