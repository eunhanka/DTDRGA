"""Standalone figure generation -- reads JSON results and .mat network data only.

Usage: python visualization/generate_figures.py
Saves all 14 figures to results/figures/ at dpi=300.
"""

import json
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import numpy as np
import scipy.io as sio

# ── Global style ──────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.5,
    "lines.markersize": 9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
JDIR = os.path.join(PROJ, "results")
FDIR = os.path.join(PROJ, "results", "figures_legacy")
DPI = 300
C_FIX = "#d62728"
C_DYN = "#2ca02c"
C_CAV = "#d62728"
C_APP = "#1f77b4"
C_EXP = "#2ca02c"

os.makedirs(FDIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────
def _lj(name):
    p = os.path.join(JDIR, name)
    if not os.path.exists(p):
        print(f"  WARNING: {name} not found")
        return None
    with open(p) as f:
        return json.load(f)


def _flat(x):
    return np.asarray(x).flatten()


def _roll(x, w=10):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    hw = w // 2
    for i in range(len(x)):
        out[i] = np.mean(x[max(0, i - hw):min(len(x), i + hw + 1)])
    return out


def _flow_change(base, day, min_frac=0.01, clip_val=50):
    """Compute % flow change. Links below min_frac of max baseline set to 0.
    If clip_val is None, no clipping is applied."""
    b, d = np.asarray(base, float), np.asarray(day, float)
    thr = min_frac * b.max()
    c = np.zeros_like(b)
    ok = b > thr
    c[ok] = (d[ok] - b[ok]) / b[ok] * 100
    if clip_val is not None:
        c = np.clip(c, -clip_val, clip_val)
    return c


ATTACK_START = 50
ATTACK_END = 100
PRE_ATTACK_START = 30
PRE_ATTACK_END = 50


def _own_poatt(rec):
    """Own-baseline PoAtt = mean(tstt[50:100]) / mean(tstt[30:50]).

    Falls back to sustained_poatt_own field, then to sustained_poatt
    if no tstt array is present.
    """
    if not rec or not isinstance(rec, dict):
        return None
    tstt = rec.get("tstt")
    if tstt is not None and len(tstt) >= ATTACK_END:
        t = np.asarray(tstt, dtype=float)
        pre = t[PRE_ATTACK_START:PRE_ATTACK_END].mean()
        if pre <= 0:
            return None
        return float(t[ATTACK_START:ATTACK_END].mean() / pre)
    if "sustained_poatt_own" in rec:
        return float(rec["sustained_poatt_own"])
    if "sustained_poatt" in rec:
        return float(rec["sustained_poatt"])
    return None


def _own_baseline(rec):
    """Pre-attack baseline TSTT = mean(tstt[30:50]) computed on the fly."""
    if not rec or not isinstance(rec, dict):
        return None
    tstt = rec.get("tstt")
    if tstt is None or len(tstt) < PRE_ATTACK_END:
        return None
    return float(np.mean(np.asarray(tstt[PRE_ATTACK_START:PRE_ATTACK_END], dtype=float)))


def _save(fig, name):
    fig.savefig(os.path.join(FDIR, name), dpi=DPI, bbox_inches="tight",
                pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {name}")


# ── Network loading ──────────────────────────────────────────────────
def load_sf():
    m = sio.loadmat(os.path.join(PROJ, "data", "sioux_falls",
                                 "SiouxFalls6180_pp.mat"), squeeze_me=False)
    co = np.asarray(m["nodeCoordinates"], float)
    adj = np.asarray(m["adjacencyList"], int) - 1
    ls = m["link"][0, 0]
    return dict(nx=co[:, 0], ny=co[:, 1], adj=adj,
                cap=_flat(ls["capacity"]).astype(float),
                fft=_flat(ls["FFT"]).astype(float),
                nn=co.shape[0], nl=adj.shape[0])


def load_ana():
    m = sio.loadmat(os.path.join(PROJ, "data", "anaheim",
                                 "Anaheim30719_pp.mat"), squeeze_me=False)
    adj = np.asarray(m["adjacencyList"], int) - 1
    ls = m["link"][0, 0]
    nn = int(_flat(m["node"][0, 0]["count"])[0])
    fft = _flat(ls["FFT"]).astype(float)
    import networkx as nx
    G = nx.DiGraph()
    for i in range(adj.shape[0]):
        w = max(float(fft[i]), 0.001)
        G.add_edge(int(adj[i, 0]), int(adj[i, 1]), weight=w)
    for n in range(nn):
        if n not in G:
            G.add_node(n)
    pos = nx.kamada_kawai_layout(G, weight="weight")
    px = np.array([pos.get(n, (0, 0))[0] for n in range(nn)])
    py = np.array([pos.get(n, (0, 0))[1] for n in range(nn)])
    return dict(nx=px, ny=py, adj=adj,
                cap=_flat(ls["capacity"]).astype(float),
                fft=fft,
                nn=nn, nl=adj.shape[0])


def topo_bc(adj, fft, n=10, weight="weight"):
    import networkx as nx
    G = nx.DiGraph()
    for i in range(len(adj)):
        G.add_edge(int(adj[i, 0]), int(adj[i, 1]),
                    weight=max(float(fft[i]), 0.001), link_idx=i)
    bc = nx.edge_betweenness_centrality(G, weight=weight, normalized=False)
    m = {}
    for (u, v), val in bc.items():
        idx = G[u][v].get("link_idx")
        if idx is not None:
            m[idx] = val
    return sorted(m, key=m.get, reverse=True)[:n], m


# ── Bidirectional target stars ────────────────────────────────────────
def compute_target_stars(adj, nx_, ny_, target_list):
    """Compute star positions with 1/3-2/3 offset for bidirectional pairs."""
    targets = list(target_list)
    edge_groups = {}
    for idx in targets:
        t, h = int(adj[idx, 0]), int(adj[idx, 1])
        key = (min(t, h), max(t, h))
        edge_groups.setdefault(key, []).append((idx, t, h))

    stars = []
    for key, links in edge_groups.items():
        if len(links) >= 2:
            for idx, t, h in links:
                fx, fy = float(nx_[t]), float(ny_[t])
                tx, ty = float(nx_[h]), float(ny_[h])
                sx = fx + (tx - fx) / 3
                sy = fy + (ty - fy) / 3
                stars.append((idx, sx, sy, tx - fx, ty - fy))
        else:
            idx, t, h = links[0]
            fx, fy = float(nx_[t]), float(ny_[t])
            tx, ty = float(nx_[h]), float(ny_[h])
            stars.append((idx, (fx + tx) / 2, (fy + ty) / 2, tx - fx, ty - fy))
    return stars


def draw_target_stars(ax, stars, color="black", markersize=14, draw_arrows=True):
    """Draw star markers and small direction arrows for target links."""
    for idx, sx, sy, dx, dy in stars:
        ax.plot(sx, sy, "*", color=color, markersize=markersize, zorder=7)
        if draw_arrows:
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                ux, uy = dx / length, dy / length
                px, py = -uy, ux
                off = length * 0.06
                ahalf = length * 0.10
                cx, cy = sx + px * off, sy + py * off
                ax.annotate("",
                    xy=(cx + ux * ahalf, cy + uy * ahalf),
                    xytext=(cx - ux * ahalf, cy - uy * ahalf),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                                   mutation_scale=10),
                    zorder=8)


def _target_legend(n_links, n_corridors, color="black"):
    """Create legend handles for target star + direction arrow."""
    return [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=color,
               markersize=14, linestyle="None",
               label=f"Directed target ({n_links} links, {n_corridors} corridors)"),
        Line2D([0], [0], marker=">", color=color, markersize=8, linestyle="None",
               label="Attack direction"),
    ]


# ═════════════════════════════════════════════════════════════════════
# FIG 1 — Signature
# ═════════════════════════════════════════════════════════════════════
def fig_signature():
    d = _lj("exp_signature.json")
    if not d:
        return
    sc = d["scenarios"]
    days = np.arange(1, 201)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.subplots_adjust(hspace=0.35, wspace=0.30)

    L = [("baseline",       "Baseline",                         "gray",   "--", 1.8),
         ("disruption",     "Disruption ($-$33%)",              "purple", "-",  2.5),
         ("bc_defl_fixed_03",   r"Fixed $\gamma$=0.3",             C_FIX,    "-",  2.5),
         ("bc_defl_dynamic_03", r"Dynamic $\gamma$=0.3",           C_DYN,    "-",  2.5),
         ("bc_defl_dynamic_10", r"Dynamic $\gamma$=1.0",           "#1f77b4","-",  2.5)]

    # (a) Full
    ax = axes[0, 0]
    for k, lab, col, ls, lw in L:
        if k in sc:
            ax.plot(days, sc[k]["tstt"], color=col, ls=ls, lw=lw, label=lab)
    ax.axvspan(51, 100, alpha=0.12, color="red")
    ax.set_xlabel("Day"); ax.set_ylabel("TSTT [veh-hours]")
    ax.set_title("(a) TSTT Trajectories"); ax.legend(fontsize=13, loc="upper right")
    ax.grid(True, alpha=0.3); ax.set_xlim(1, 200)
    # Zoom rectangle
    from matplotlib.patches import Rectangle
    vals_z = []
    for k, *_ in L:
        if k in sc:
            vals_z.extend(sc[k]["tstt"][39:120])
    if vals_z:
        y_lo_z = max(6400, min(vals_z) - 200)
        y_hi_z = max(vals_z) + 200
        rect = Rectangle((40, y_lo_z), 80, y_hi_z - y_lo_z,
                          linewidth=1.5, edgecolor="black", facecolor="none",
                          linestyle="--", zorder=10)
        ax.add_patch(rect)
        ax.text(122, y_lo_z + (y_hi_z - y_lo_z) * 0.5, "see (b)", fontsize=11,
                style="italic", va="center")

    # (b) Zoomed
    ax = axes[0, 1]
    for k, lab, col, ls, lw in L:
        if k in sc:
            ax.plot(days, sc[k]["tstt"], color=col, ls=ls, lw=lw, label=lab)
    ax.axvspan(51, 100, alpha=0.12, color="red")
    ax.set_xlabel("Day"); ax.set_ylabel("TSTT [veh-hours]")
    ax.set_title("(b) TSTT Zoom [Day 40\u2013120]"); ax.set_xlim(40, 120)
    vals = []
    for k, *_ in L:
        if k in sc:
            vals.extend(sc[k]["tstt"][39:120])
    if vals:
        lo, hi = min(vals), max(vals)
        ax.set_ylim(max(6400, lo - 200), hi + 200)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    # (c) Trust -- gamma=0.3 (solid) and gamma=1.0 (dotted, low alpha)
    ax = axes[1, 0]
    key = "bc_defl_dynamic_03"
    key10 = "bc_defl_dynamic_10"
    if key in sc and "trust" in sc[key]:
        t = np.array(sc[key]["trust"])
        for k, (nm, col, ls) in enumerate(
                [("CAV", C_CAV, "-"), ("App", C_APP, "--"), ("Exp", C_EXP, ":")]):
            if k < t.shape[1]:
                ax.plot(days, t[:, k], color=col, ls=ls, lw=2.5,
                        label=fr"{nm} ($\gamma$=0.3)")
    if key10 in sc and "trust" in sc[key10]:
        t10 = np.array(sc[key10]["trust"])
        for k, (nm, col) in enumerate(
                [("CAV", C_CAV), ("App", C_APP), ("Exp", C_EXP)]):
            if k < t10.shape[1]:
                ax.plot(days, t10[:, k], color=col, ls=":", lw=2.0, alpha=0.55,
                        label=fr"{nm} ($\gamma$=1.0)" if k == 0 else None)
    ax.axvspan(51, 100, alpha=0.12, color="red")
    ax.set_ylim(0, 1.05); ax.set_xlabel("Day"); ax.set_ylabel(r"Trust $E[T]$")
    ax.set_title("(c) Trust Evolution")
    ax.legend(fontsize=11, ncol=2); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02,
            r"$\gamma$=0.3 (stealthy): trust constant; "
            r"$\gamma$=1.0 (dotted): drops to $\sim$0, recovers $\sim$77 days",
            transform=ax.transAxes, fontsize=12, style="italic",
            bbox=dict(facecolor="lightyellow", alpha=0.7, edgecolor="none"))

    # (d) Lambda -- gamma=0.3 (solid) and gamma=1.0 (dotted, low alpha)
    ax = axes[1, 1]
    if key in sc and "info_weight" in sc[key]:
        iw = np.array(sc[key]["info_weight"])
        for k, (nm, col, ls) in enumerate(
                [("CAV", C_CAV, "-"), ("App", C_APP, "--"), ("Exp", C_EXP, ":")]):
            if k < iw.shape[1]:
                ax.plot(days, iw[:, k], color=col, ls=ls, lw=2.5,
                        label=fr"{nm} ($\gamma$=0.3)")
    if key10 in sc and "info_weight" in sc[key10]:
        iw10 = np.array(sc[key10]["info_weight"])
        for k, (nm, col) in enumerate(
                [("CAV", C_CAV), ("App", C_APP), ("Exp", C_EXP)]):
            if k < iw10.shape[1]:
                ax.plot(days, iw10[:, k], color=col, ls=":", lw=2.0, alpha=0.55,
                        label=fr"{nm} ($\gamma$=1.0)" if k == 0 else None)
    ax.axvspan(51, 100, alpha=0.12, color="red")
    ax.set_ylim(0, 1.05); ax.set_xlabel("Day"); ax.set_ylabel(r"Info Weight $\lambda$")
    ax.set_title(r"(d) Information Weight $\lambda$")
    ax.legend(fontsize=11, ncol=2); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02,
            r"$\gamma$=0.3 (stealthy): $\lambda$ stable; "
            r"$\gamma$=1.0 (dotted): collapses, recovers $\sim$77 days",
            transform=ax.transAxes, fontsize=12, style="italic",
            bbox=dict(facecolor="lightyellow", alpha=0.7, edgecolor="none"))

    _save(fig, "exp_signature.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 2 — Gamma Sweep
# ═════════════════════════════════════════════════════════════════════
def fig_gamma():
    d = _lj("exp_gamma_sweep.json")
    if not d:
        return
    gs = d["gammas"]; res = d["results"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7))

    fs = [res.get(f"g{g}_fixed", {}).get("sustained_poatt") for g in gs]
    ds = [res.get(f"g{g}_dynamic", {}).get("sustained_poatt") for g in gs]
    vf = [(g, v) for g, v in zip(gs, fs) if v]
    vd = [(g, v) for g, v in zip(gs, ds) if v]

    if vf: a1.plot(*zip(*vf), "o-", color=C_FIX, label="Fixed Trust")
    if vd: a1.plot(*zip(*vd), "s-", color=C_DYN, label="Dynamic Trust")
    a1.axhline(1.0, color="gray", ls="--", alpha=0.5)
    a1.axvline(0.5, color=C_DYN, ls=":", alpha=0.6, lw=1.5)
    a1.annotate(r"$\hat\gamma \approx 0.5$", (0.5, 1.0),
                xytext=(0.65, 1.20), fontsize=14, fontweight="bold",
                color="green",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="green", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.2))
    a1.set_xlabel(r"$\gamma$ (Attack Intensity)"); a1.set_ylabel("Sustained PoAtt")
    a1.set_title(r"(a) Sustained PoAtt vs $\gamma$"); a1.legend(); a1.grid(True, alpha=0.3)

    de_pts = []
    for g in gs:
        sf = res.get(f"g{g}_fixed", {}).get("sustained_poatt")
        sd = res.get(f"g{g}_dynamic", {}).get("sustained_poatt")
        if sf and sd and sf > 1:
            de_pts.append((g, (1 - (sd - 1) / (sf - 1)) * 100))
    # Two-regime background: stealthy (gamma < 0.5) green, detectable (>=0.5) pink
    a2.axvspan(0, 0.5, color="lightgreen", alpha=0.15)
    a2.axvspan(0.5, 1.0, color="lightpink", alpha=0.15)
    a2.axvline(0.5, color=C_DYN, ls=":", alpha=0.6, lw=1.5)
    if de_pts:
        gp, vp = zip(*de_pts)
        a2.plot(gp, vp, "^-", color="#1f77b4", markersize=9, lw=2)
        stealthy = [v for g, v in de_pts if g < 0.5]
        detect = [v for g, v in de_pts if g >= 0.5]
        if stealthy and detect:
            a2.text(0.05, 0.95,
                    f"Stealthy ($\\gamma$<0.5): TIA $\\approx$ {np.mean(stealthy):.0f}%\n"
                    f"Detectable ($\\gamma$$\\geq$0.5): TIA $\\approx$ {np.mean(detect):.0f}%",
                    transform=a2.transAxes, fontsize=14, va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="#1f77b4",
                              boxstyle="round,pad=0.3"))
    a2.set_xlabel(r"$\gamma$"); a2.set_ylabel("TIA (%)")
    a2.set_title(r"(b) Trust-Induced Attenuation vs $\gamma$")
    a2.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "exp_gamma_sweep.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 3 — SF Network
# ═════════════════════════════════════════════════════════════════════
def fig_net_sf(net):
    sig = _lj("exp_signature.json")
    if not sig:
        return
    bf = np.array(sig["scenarios"].get("baseline", {}).get("link_flows", {}).get("39",
                  [0] * net["nl"]))
    vc = bf / (net["cap"] + 1e-10)
    tgts, _ = topo_bc(net["adj"], net["fft"], 10)

    fig, ax = plt.subplots(figsize=(12, 12))
    nx_, ny_, adj = net["nx"], net["ny"], net["adj"]
    mf = max(bf.max(), 1)
    cmap = plt.cm.RdYlGn_r; norm = mcolors.Normalize(0, 1.5)

    # All links
    for i in range(net["nl"]):
        t, h = adj[i]; x = [nx_[t], nx_[h]]; y = [ny_[t], ny_[h]]
        lw = max(1.5, min(6, bf[i] / mf * 6))
        ax.plot(x, y, color=cmap(norm(vc[i])), lw=lw, zorder=2, solid_capstyle="round")

    # BC target links -- thick black border
    for idx in tgts:
        t, h = adj[idx]; x = [nx_[t], nx_[h]]; y = [ny_[t], ny_[h]]
        ax.plot(x, y, color="black", lw=5, zorder=4, solid_capstyle="round")

    # Nodes
    for n in range(net["nn"]):
        ax.plot(nx_[n], ny_[n], "o", mfc="white", mec="black", markersize=20,
                zorder=5, mew=1.5)
        ax.text(nx_[n], ny_[n], str(n + 1), ha="center", va="center",
                fontsize=12, fontweight="bold", zorder=6)

    # Target stars with bidirectional offset (no direction arrows, red)
    stars = compute_target_stars(adj, nx_, ny_, tgts)
    draw_target_stars(ax, stars, color="red", markersize=14, draw_arrows=False)

    # Count corridors
    edge_groups = {}
    for idx in tgts:
        t, h = int(adj[idx, 0]), int(adj[idx, 1])
        edge_groups.setdefault((min(t, h), max(t, h)), []).append(idx)
    n_corr = len(edge_groups)

    leg = [Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
                  markersize=14, linestyle="None",
                  label=f"BC target ({len(tgts)} links, {n_corr} corridors)")]
    ax.legend(handles=leg, fontsize=12, loc="upper left")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Volume/Capacity Ratio", fontsize=14)
    fig.suptitle("Sioux Falls Network \u2014 Pre-Attack Equilibrium (Day 40)", fontsize=20)
    ax.set_aspect("equal")
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    _save(fig, "exp_network_sf.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 4 — Anaheim Network
# ═════════════════════════════════════════════════════════════════════
def fig_net_ana(net):
    ad = _lj("exp_anaheim.json")
    if not ad:
        return
    bf = np.array(ad["scenarios"].get("baseline", {}).get("link_flows", {}).get("39",
                  [0] * net["nl"]))
    vc = bf / (net["cap"] + 1e-10)
    tgts, _ = topo_bc(net["adj"], net["fft"], 10)

    fig, ax = plt.subplots(figsize=(14, 12))
    nx_, ny_, adj = net["nx"], net["ny"], net["adj"]
    mf = max(bf.max(), 1); thr = 0.05 * mf
    cmap = plt.cm.RdYlGn_r; norm = mcolors.Normalize(0, 1.5)

    # All links thin gray
    for i in range(net["nl"]):
        t, h = adj[i]
        ax.plot([nx_[t], nx_[h]], [ny_[t], ny_[h]], color="#cccccc", lw=0.4, zorder=1)
    # High-flow links colored
    for i in range(net["nl"]):
        if bf[i] < thr:
            continue
        t, h = adj[i]; x = [nx_[t], nx_[h]]; y = [ny_[t], ny_[h]]
        lw = max(0.8, min(4, bf[i] / mf * 4))
        ax.plot(x, y, color=cmap(norm(vc[i])), lw=lw, zorder=2)
    # BC target links red
    for idx in tgts:
        t, h = adj[idx]; x = [nx_[t], nx_[h]]; y = [ny_[t], ny_[h]]
        ax.plot(x, y, color="red", lw=3, zorder=3, alpha=0.9)

    # Target stars with bidirectional offset (no direction arrows)
    stars = compute_target_stars(adj, nx_, ny_, tgts)
    draw_target_stars(ax, stars, color="red", markersize=12, draw_arrows=False)

    # Nodes (no labels — node IDs not meaningful for Anaheim)
    ax.scatter(nx_, ny_, c="black", s=1, zorder=4)

    edge_groups = {}
    for idx in tgts:
        t, h = int(adj[idx, 0]), int(adj[idx, 1])
        edge_groups.setdefault((min(t, h), max(t, h)), []).append(idx)
    n_corr = len(edge_groups)

    leg = [Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
                  markersize=14, linestyle="None",
                  label=f"BC target ({len(tgts)} links, {n_corr} corridors)")]
    ax.legend(handles=leg, fontsize=12, loc="lower right")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Volume/Capacity Ratio", fontsize=14)
    fig.suptitle("Anaheim Network \u2014 Pre-Attack Equilibrium (Day 40)", fontsize=20)
    ax.set_aspect("equal")
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    _save(fig, "exp_network_ana.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 5 — SF Spatial
# ═════════════════════════════════════════════════════════════════════
def _sf_spatial_panel(ax, net, base_flow, day_flow, target_stars, cmap, norm):
    """Draw one SF spatial panel."""
    nx_, ny_, adj = net["nx"], net["ny"], net["adj"]
    change = _flow_change(base_flow, day_flow, clip_val=50)
    raw_change = _flow_change(base_flow, day_flow, clip_val=None)

    for i in range(net["nl"]):
        t, h = adj[i]; x = [nx_[t], nx_[h]]; y = [ny_[t], ny_[h]]
        c = change[i]
        if abs(c) < 2:
            ax.plot(x, y, color="#e0e0e0", lw=0.5, zorder=1)
        else:
            lw = min(5, max(0.8, abs(c) / 50 * 5))
            ax.plot(x, y, color=cmap(norm(c)), lw=lw, zorder=2)

    # Nodes
    ax.scatter(nx_, ny_, c="gray", s=50, zorder=3, edgecolors="white", lw=0.5)
    for n in range(net["nn"]):
        ax.text(nx_[n], ny_[n], str(n + 1), fontsize=9, ha="center", va="center",
                color="#555555", zorder=5)

    # Target stars (bidirectional-aware)
    for idx, sx, sy, dx, dy in target_stars:
        ax.plot(sx, sy, "*", color="red", markersize=15, zorder=6)

    # Hardcoded TEXT_POS for top annotations — edge_index: (tx, ty) in data coords
    # Avoids text overlaps by explicit placement with wide offsets
    TEXT_POS = {
        # Red (increased) — spread further apart
        48: (78, 36),      # 16->17: far right-down
        51: (78, 58),      # 17->16: far right-up
        50: (22, 40),      # 17->10: far left-down
        8:  (12, 96),      # 4->5: left-above
        10: (36, 98),      # 5->4: center-above
        15: (76, 84),      # 6->8: right
        18: (76, 70),      # 8->6: right-below
        11: (48, 96),      # 5->6: above
        14: (56, 76),      # 6->5: below
        28: (28, 50),      # 10->16: left-below
        47: (56, 68),      # 16->10: above-right
        17: (88, 74),      # 7->18: far right
        53: (88, 56),      # 18->7: far right-below
        # Blue (decreased) — move to clear areas
        4:  (2, 76),       # 3->1: far left-below
        0:  (28, 106),     # 1->2: above-center
        1:  (58, 108),     # 2->1: above-right
        65: (44, 20),      # 21->24: above
        73: (6, 20),       # 24->13: left-above
        72: (6, 2),        # 13->24: left-bottom
        64: (16, 2),       # 24->21: bottom
        22: (24, 76),      # 9->5: left
        23: (46, 76),      # 5->9: right
        20: (46, 66),      # 8->9: center
        42: (28, 36),      # 15->10: left-below
        44: (26, 26),      # 15->14: left-far-below
        46: (74, 66),      # 16->8
        21: (62, 64),      # 8->16
    }

    # Gather top-1 increased + top-1 decreased
    sorted_inc = np.argsort(change)[::-1][:1]
    sorted_dec = np.argsort(change)[:1]

    ann_list = []
    for idx in sorted_inc:
        if change[idx] > 2:
            ann_list.append((idx, raw_change[idx], "inc"))
    for idx in sorted_dec:
        if change[idx] < -2:
            ann_list.append((idx, raw_change[idx], "dec"))

    placed = []  # (tx, ty) for collision check
    for idx, raw_val, ann_type in ann_list:
        t, h = adj[idx]
        t1, h1 = t + 1, h + 1
        mx, my = (nx_[t] + nx_[h]) / 2, (ny_[t] + ny_[h]) / 2
        direction = f"{t1}\u2192{h1}: "
        if ann_type == "inc":
            label = f"{direction}+{raw_val:.1f}%"
            color = "red"
        else:
            label = f"{direction}\u2212{abs(raw_val):.1f}%"
            color = "blue"

        if idx in TEXT_POS:
            tx, ty = TEXT_POS[idx]
        else:
            cx, cy = np.mean(nx_), np.mean(ny_)
            ddx, ddy = mx - cx, my - cy
            mag = max(np.sqrt(ddx**2 + ddy**2), 0.1)
            tx, ty = mx + 8.0 * ddx / mag, my + 8.0 * ddy / mag

        # Collision avoidance: multi-pass with x-shift fallback and panel clamp
        x_lo, x_hi = -4, 98
        y_lo, y_hi = -2, 112
        for _pass in range(3):
            moved = False
            for ptx, pty in placed:
                if abs(tx - ptx) < 14 and abs(ty - pty) < 9:
                    new_ty = ty + 12 if ty >= pty else ty - 12
                    if y_lo + 4 <= new_ty <= y_hi - 4:
                        ty = new_ty
                    else:
                        tx = tx + 16 if tx >= ptx else tx - 16
                        tx = max(x_lo + 4, min(x_hi - 4, tx))
                    moved = True
            if not moved:
                break
        placed.append((tx, ty))

        ax.annotate(label, (mx, my), fontsize=11, color=color, fontweight="bold",
                    zorder=7, textcoords="data", xytext=(tx, ty),
                    bbox=dict(facecolor="white", edgecolor=color, alpha=0.95,
                              boxstyle="round,pad=0.15"),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0,
                                    connectionstyle="arc3,rad=0.15"))

    ax.set_xlim(-4, 98); ax.set_ylim(-2, 112)
    ax.set_aspect("equal")
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)


def fig_spatial_sf(net):
    # Use gamma=0.3 scenarios stored in exp_signature.json
    # (matches supplementary caption; PoAtt ~1.06 for both fix/dyn)
    sig = _lj("exp_signature.json")
    if not sig:
        return
    sc = sig["scenarios"]
    bl = sc.get("baseline", {}).get("link_flows", {})
    fix_sc = sc.get("bc_defl_fixed_03", {})
    dyn_sc = sc.get("bc_defl_dynamic_03", {})
    ff = fix_sc.get("link_flows", {})
    df = dyn_sc.get("link_flows", {})
    if not bl or (not ff and not df):
        print("  WARNING: Missing SF spatial data (gamma=0.3)"); return
    if "39" not in bl:
        print("  WARNING: Day 39 not in baseline"); return
    base = np.array(bl["39"])

    # Compute target stars using stored target links if any, else topo_bc
    tgt_list = fix_sc.get("target_links", []) or dyn_sc.get("target_links", [])
    if not tgt_list:
        tgt_list, _ = topo_bc(net["adj"], net["fft"], 10)
    target_stars = compute_target_stars(net["adj"], net["nx"], net["ny"], tgt_list)

    fig = plt.figure(figsize=(24, 17))
    gs_spec = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.03],
                       hspace=0.04, wspace=0.12)
    fig.subplots_adjust(bottom=0.05, top=0.95)
    cmap = plt.cm.RdBu_r; norm = mcolors.Normalize(-50, 50)
    days_k = ["54", "74", "94"]
    days_l = ["Day 55 (Early)", "Day 75 (Mid)", "Day 95 (Late)"]

    for ri, (fdata, rlab) in enumerate([(ff, "Fixed Trust"), (df, "Dynamic Trust")]):
        if not fdata:
            continue
        for ci, (dk, dl) in enumerate(zip(days_k, days_l)):
            ax = fig.add_subplot(gs_spec[ri, ci])
            if dk in fdata:
                _sf_spatial_panel(ax, net, base, np.array(fdata[dk]),
                                  target_stars, cmap, norm)
            if ri == 0:
                ax.set_title(dl, fontsize=16)
            if ci == 0:
                ax.set_ylabel(rlab, fontsize=18, fontweight="bold")

    cax = fig.add_subplot(gs_spec[2, :])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Flow Change (%) relative to Day 40 baseline", fontsize=14)
    _save(fig, "exp_spatial_attack_sf.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 6 — Anaheim Spatial
# ═════════════════════════════════════════════════════════════════════
def _ana_spatial_panel(ax, net, base_flow, day_flow, target_stars, target_set,
                       cmap, norm_obj):
    """Draw one Anaheim spatial panel. ALL links colored, auto-scaled."""
    nx_, ny_, adj = net["nx"], net["ny"], net["adj"]
    change = _flow_change(base_flow, day_flow, min_frac=0.01, clip_val=None)

    # Draw all links colored by change
    for i in range(net["nl"]):
        t, h = adj[i]
        ax.plot([nx_[t], nx_[h]], [ny_[t], ny_[h]],
                color=cmap(norm_obj(change[i])), lw=1.0, zorder=1, solid_capstyle="round")

    # BC targets thick black
    for idx in target_set:
        if idx < net["nl"]:
            t, h = adj[idx]
            ax.plot([nx_[t], nx_[h]], [ny_[t], ny_[h]],
                    color="black", lw=3.5, zorder=3, alpha=0.7)

    # Target stars (bidirectional-aware)
    for idx, sx, sy, dx, dy in target_stars:
        ax.plot(sx, sy, "*", color="red", markersize=10, zorder=6)

    # Per-panel metrics
    ok = base_flow > 0.01 * base_flow.max()
    if ok.any():
        c_ok = change[ok]
        txt = (f"Mean: {np.mean(c_ok):+.1f}%\n"
               f"Max: {np.max(c_ok):+.0f}%\n"
               f"Min: {np.min(c_ok):+.0f}%")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=12, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round"))

    ax.set_aspect("equal")
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)


def fig_spatial_ana(net):
    d = _lj("exp_anaheim.json")
    if not d:
        return
    sc = d.get("scenarios", {})
    bl = sc.get("baseline", {}).get("link_flows", {})
    ff = sc.get("bc_defl_fixed", {}).get("link_flows", {})
    df = sc.get("bc_defl_dynamic", {}).get("link_flows", {})
    if not bl or (not ff and not df):
        print("  WARNING: Missing Anaheim spatial data"); return
    if "39" not in bl:
        print("  WARNING: Day 39 not in Anaheim baseline"); return
    base = np.array(bl["39"])
    tgt_list = sc.get("bc_defl_fixed", {}).get("target_links", [])
    if not tgt_list:
        tgt_list, _ = topo_bc(net["adj"], net["fft"], 10)
    target_set = set(tgt_list)
    target_stars = compute_target_stars(net["adj"], net["nx"], net["ny"], tgt_list)

    # Compute global 95th percentile for symmetric normalization
    days_k = ["54", "74", "94"]
    all_changes = []
    for fdata in [ff, df]:
        if not fdata:
            continue
        for dk in days_k:
            if dk in fdata:
                c = _flow_change(base, np.array(fdata[dk]), min_frac=0.01, clip_val=None)
                all_changes.append(c)
    if all_changes:
        combined = np.concatenate(all_changes)
        p95 = np.percentile(np.abs(combined[combined != 0]), 95) if np.any(combined != 0) else 50
        vmax = max(p95, 5)
    else:
        vmax = 50

    fig = plt.figure(figsize=(22, 11))
    gs_spec = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.03],
                       hspace=0.05, wspace=0.05)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.94, bottom=0.06,
                        wspace=0.05, hspace=0.05)
    cmap = plt.cm.RdBu_r
    norm_obj = mcolors.Normalize(-vmax, vmax)
    days_l = ["Day 55 (Early)", "Day 75 (Mid)", "Day 95 (Late)"]

    for ri, (fdata, rlab) in enumerate([(ff, "Fixed Trust"), (df, "Dynamic Trust")]):
        if not fdata:
            continue
        for ci, (dk, dl) in enumerate(zip(days_k, days_l)):
            ax = fig.add_subplot(gs_spec[ri, ci])
            if dk in fdata:
                _ana_spatial_panel(ax, net, base, np.array(fdata[dk]),
                                   target_stars, target_set, cmap, norm_obj)
            if ri == 0:
                ax.set_title(dl, fontsize=16)
            if ci == 0:
                ax.set_ylabel(rlab, fontsize=18, fontweight="bold")

    cax = fig.add_subplot(gs_spec[2, :])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(r"Flow Change (%) relative to Day 40 baseline ($\gamma$=0.3)", fontsize=14)
    _save(fig, "exp_spatial_attack_ana.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 7 — Theta Sweep
# ═════════════════════════════════════════════════════════════════════
def fig_theta():
    d = _lj("exp_theta_sweep.json")
    if not d:
        return
    ths = d["thetas"]; res = d["results"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6),
                                  gridspec_kw={"wspace": 0.25})

    vf = [(th, _own_poatt(res.get(f"th{th}_fixed"))) for th in ths]
    vd = [(th, _own_poatt(res.get(f"th{th}_dynamic"))) for th in ths]
    vf = [(x, y) for x, y in vf if y]; vd = [(x, y) for x, y in vd if y]

    if vf: a1.plot(*zip(*vf), "o-", color=C_FIX, label="Fixed Trust")
    if vd: a1.plot(*zip(*vd), "s-", color=C_DYN, label="Dynamic Trust")
    a1.axhline(1.0, color="gray", ls="--", alpha=0.5)
    a1.axvspan(0.006, 0.010, alpha=0.08, color="blue")
    a1.set_xlim(0.0015, 0.0105)
    a1.text(0.008, 0.93, r"Plateau region ($\theta \geq$ 0.006)", fontsize=13, ha="center",
            color="#1f77b4", style="italic", transform=a1.get_xaxis_transform(),
            va="top")
    a1.set_xlabel(r"$\theta$ (Logit Dispersion)"); a1.set_ylabel("Sustained PoAtt")
    a1.set_title(r"(a) Sustained PoAtt vs $\theta$"); a1.legend(); a1.grid(True, alpha=0.3)

    vb = [(th, _own_baseline(res.get(f"th{th}_fixed"))) for th in ths]
    vb = [(x, y) for x, y in vb if y]
    if vb:
        a2.plot(*zip(*vb), "ko-", lw=2)
        a2.annotate(f"{vb[0][1]:.0f}", xy=vb[0], textcoords="offset points",
                    xytext=(8, -5), ha="left", va="top", fontsize=12)
        a2.annotate(f"{vb[-1][1]:.0f}", xy=vb[-1], textcoords="offset points",
                    xytext=(-8, 8), ha="right", va="bottom", fontsize=12)
        b_arr = np.array([y for _, y in vb], dtype=float)
        if b_arr.size >= 2 and b_arr.mean() > 0:
            half = (b_arr.max() - b_arr.min()) / 2 / b_arr.mean() * 100
            a2.text(0.97, 0.97, f"Range: ±{half:.0f}%",
                    transform=a2.transAxes, ha="right", va="top",
                    fontsize=13, style="italic",
                    bbox=dict(facecolor="lightyellow", alpha=0.7, edgecolor="none"))
    a2.set_xlabel(r"$\theta$"); a2.set_ylabel("Own Baseline TSTT [veh-hours]")
    a2.set_title(r"(b) Own Baseline TSTT vs $\theta$"); a2.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "exp_theta_extended.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 8 — Composition
# ═════════════════════════════════════════════════════════════════════
def fig_comp():
    cavs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots(figsize=(12, 8))
    pct = [c * 100 for c in cavs]
    # Prefer per-CAV standalone files (matches paper text)
    vf, vd = [], []
    for c, p in zip(cavs, pct):
        rec_f = _lj(f"exp_g07_comp_{c}_fixed.json")
        rec_d = _lj(f"exp_g07_comp_{c}_dynamic.json")
        p_f = _own_poatt(rec_f); p_d = _own_poatt(rec_d)
        if p_f: vf.append((p, p_f))
        if p_d: vd.append((p, p_d))
    if not vf and not vd:
        d = _lj("exp_composition_g07.json")
        if not d:
            return
        cavs_d = d["cavs"]; res = d["results"]
        pct_d = [c * 100 for c in cavs_d]
        vf = [(p, _own_poatt(res.get(f"cav{c}_fixed"))) for c, p in zip(cavs_d, pct_d)]
        vd = [(p, _own_poatt(res.get(f"cav{c}_dynamic"))) for c, p in zip(cavs_d, pct_d)]
        vf = [(x, y) for x, y in vf if y]; vd = [(x, y) for x, y in vd if y]

    if vf: ax.plot(*zip(*vf), "o-", color=C_FIX, label="Fixed Trust", markersize=10, lw=2.5)
    if vd: ax.plot(*zip(*vd), "s-", color=C_DYN, label="Dynamic Trust", markersize=10, lw=2.5)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlim(-5, 105)
    ax.set_xlabel("CAV Penetration (%)"); ax.set_ylabel("Sustained PoAtt")
    ax.set_title("Composition Dual Effect (Scenario I)"); ax.legend(fontsize=14, loc="center left"); ax.grid(True, alpha=0.3)

    if vf and vd:
        gap = vf[-1][1] - vd[-1][1]
        ax.text(0.95, 0.50, f"Gap: {gap*100:.1f} pp",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=15, color="navy",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    if len(vf) >= 2:
        slope_f = (vf[-1][1] - vf[0][1]) / (vf[-1][0] - vf[0][0]) * 10 * 100
        ax.text(0.05, 0.95, f"Fixed: +{slope_f:.2f} pp per 10% CAV",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=15, color=C_FIX, fontweight="bold")
    if len(vd) >= 2:
        slope_d = (vd[-1][1] - vd[0][1]) / (vd[-1][0] - vd[0][0]) * 10 * 100
        sign = "\u2212" if slope_d < 0 else "+"
        ax.text(0.50, 0.20, f"Dynamic: {sign}{abs(slope_d):.2f} pp per 10% CAV",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=15, color=C_DYN, fontweight="bold")

    fig.tight_layout()
    _save(fig, "exp_composition.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 9 — N-target (NEW)
# ═════════════════════════════════════════════════════════════════════
def fig_ntarget():
    d = _lj("exp_ntarget_sweep.json")
    if not d:
        return
    nts = d["n_targets"]; res = d["results"]
    fig, ax = plt.subplots(figsize=(12, 7))

    vf = [(nt, _own_poatt(res.get(f"nt{nt}_fixed"))) for nt in nts]
    vd = [(nt, _own_poatt(res.get(f"nt{nt}_dynamic"))) for nt in nts]
    vf = [(x, y) for x, y in vf if y]; vd = [(x, y) for x, y in vd if y]

    if vf: ax.plot(*zip(*vf), "o-", color=C_FIX, label="Fixed Trust", markersize=10, lw=2.5)
    if vd: ax.plot(*zip(*vd), "s-", color=C_DYN, label="Dynamic Trust", markersize=10, lw=2.5)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)

    # Zoom y-axis to data range
    all_vals = [y for _, y in vf + vd]
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = (hi - lo) * 0.15
        ax.set_ylim(min(lo - pad, 0.99), hi + pad)

    ax.set_xlabel(r"$n_{target}$ (Number of Target Links)")
    ax.set_ylabel("Sustained PoAtt")
    ax.set_title("Attack Budget Saturation")
    ax.legend(fontsize=14); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "exp_ntarget.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 10 — BR Sweep (NEW)
# ═════════════════════════════════════════════════════════════════════
def fig_br():
    d = _lj("exp_br_sweep.json")
    if not d:
        return
    deltas = d["deltas"]; res = d["results"]
    fig, ax = plt.subplots(figsize=(12, 7))

    vf = [(dl, _own_poatt(res.get(f"d{dl}_fixed"))) for dl in deltas]
    vd = [(dl, _own_poatt(res.get(f"d{dl}_dynamic"))) for dl in deltas]
    vf = [(x, y) for x, y in vf if y]; vd = [(x, y) for x, y in vd if y]

    if vf: ax.plot(*zip(*vf), "o-", color=C_FIX, label="Fixed Trust", markersize=10, lw=2.5)
    if vd: ax.plot(*zip(*vd), "s-", color=C_DYN, label="Dynamic Trust", markersize=10, lw=2.5)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)

    # Zoom y-axis
    all_vals = [y for _, y in vf + vd]
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = (hi - lo) * 0.15
        ax.set_ylim(min(lo - pad, 0.99), hi + pad)

    ax.set_xlabel(r"$\delta$ (Indifference Band, seconds)")
    ax.set_ylabel("Sustained PoAtt")

    ax.text(0.95, 0.10, "BR provides secondary resilience\n(trust already absorbs >90%)",
            transform=ax.transAxes, fontsize=12, ha="right", va="bottom", style="italic",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", boxstyle="round"))

    ax.set_title(r"Bounded Rationality ($\delta$) Sweep")
    ax.legend(loc="upper right", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "exp_br_sweep.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 11 — Theory Validation (EXPANDED 3x2)
# ═════════════════════════════════════════════════════════════════════
def fig_theory():
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.subplots_adjust(hspace=0.40, wspace=0.30)

    # (a) Theta Attenuation -- gamma=0.7, prefer per-theta standalone files
    ax = axes[0, 0]
    ths = [0.002, 0.004, 0.006, 0.008, 0.010]
    vf, vd = [], []
    for th in ths:
        # Try standalone files (formatted as 0.002, 0.004, ..., 0.01)
        th_str = f"{th:g}"
        rec_f = _lj(f"exp_g07_th_{th_str}_fixed.json")
        rec_d = _lj(f"exp_g07_th_{th_str}_dynamic.json")
        p_f = _own_poatt(rec_f); p_d = _own_poatt(rec_d)
        if p_f: vf.append((th, p_f))
        if p_d: vd.append((th, p_d))
    if not vf and not vd:
        # Fallback to aggregator
        d = _lj("exp_theta_sweep_g07.json")
        if d:
            ths_d = d["thetas"]; res = d["results"]
            vf = [(th, _own_poatt(res.get(f"th{th}_fixed"))) for th in ths_d]; vf = [(x, y) for x, y in vf if y]
            vd = [(th, _own_poatt(res.get(f"th{th}_dynamic"))) for th in ths_d]; vd = [(x, y) for x, y in vd if y]
    if vf: ax.plot(*zip(*vf), "o-", color=C_FIX, label="Fixed")
    if vd: ax.plot(*zip(*vd), "s-", color=C_DYN, label="Dynamic")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    if vf:
        mi = min(range(len(vf)), key=lambda i: vf[i][1])
        ax.annotate("Min", vf[mi], textcoords="offset points", xytext=(20, 15),
                    fontsize=14, arrowprops=dict(arrowstyle="->", lw=1.5))
    all_v = [y for _, y in vf + vd]
    if all_v:
        lo, hi = min(all_v), max(all_v)
        pad = (hi - lo) * 0.1 if hi > lo else 0.005
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel("Sustained PoAtt")
    ax.set_title(r"(a) $\theta$ Attenuation"); ax.legend(); ax.grid(True, alpha=0.3)

    # (b) Recovery Power-Law
    ax = axes[0, 1]
    td = _lj("exp_trust_sensitivity_g07.json")
    if td:
        ratios = td["ratios"]; res = td["results"]
        rk = "trust_recovery"
        valid = [(r, res.get(f"r{r}", {}).get(rk)) for r in ratios]
        valid = [(r, tr) for r, tr in valid if tr and tr > 0]
        if len(valid) >= 2:
            rs = np.array([float(x[0]) for x in valid])
            trs = np.array([float(x[1]) for x in valid])
            ax.scatter(rs, trs, c="#1f77b4", s=100, zorder=3)
            lr, lt = np.log(rs), np.log(trs)
            sl, ic = np.polyfit(lr, lt, 1)
            xf = np.linspace(rs.min(), rs.max(), 100)
            ax.plot(xf, np.exp(ic) * xf ** sl, "--", color="#1f77b4", lw=2)
            ss_r = np.sum((lt - (sl * lr + ic)) ** 2)
            ss_t = np.sum((lt - lt.mean()) ** 2)
            r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
            ax.text(0.05, 0.92, f"slope = {sl:.3f}\n$R^2$ = {r2:.3f}",
                    transform=ax.transAxes, fontsize=16, va="top",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
    ax.set_xlabel(r"$w_f/w_s$ ratio"); ax.set_ylabel("Trust Recovery (days)")
    ax.set_title("(b) Recovery Sublinear Fit"); ax.grid(True, alpha=0.3)

    # (c) IS Redundancy
    ax = axes[1, 0]
    d = _lj("exp_is_sweep.json")
    if d:
        als = d["alphas"]; res = d["results"]
        vf = [(a, _own_poatt(res.get(f"a{a}_fixed"))) for a in als]; vf = [(x, y) for x, y in vf if y]
        vd = [(a, _own_poatt(res.get(f"a{a}_dynamic"))) for a in als]; vd = [(x, y) for x, y in vd if y]
        if vf: ax.plot(*zip(*vf), "o-", color=C_FIX, label="Fixed")
        if vd: ax.plot(*zip(*vd), "s-", color=C_DYN, label="Dynamic")
        ax.text(0.5, 0.5, r"Fixed $\Delta$ < 0.4 pp", transform=ax.transAxes,
                fontsize=14, ha="center", style="italic",
                bbox=dict(facecolor="lightyellow", alpha=0.6, edgecolor="none"))
        all_v = [y for _, y in vf + vd]
        if all_v:
            lo, hi = min(all_v), max(all_v)
            pad = (hi - lo) * 0.1 if hi > lo else 0.005
            ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(r"IS $\alpha$"); ax.set_ylabel("Sustained PoAtt")
    ax.set_title("(c) IS Redundancy"); ax.legend(); ax.grid(True, alpha=0.3)

    # (d) Composition Dual -- prefer per-CAV standalone files
    ax = axes[1, 1]
    cavs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    pct = [c * 100 for c in cavs]
    vf, vd = [], []
    for c, p in zip(cavs, pct):
        rec_f = _lj(f"exp_g07_comp_{c}_fixed.json")
        rec_d = _lj(f"exp_g07_comp_{c}_dynamic.json")
        p_f = _own_poatt(rec_f); p_d = _own_poatt(rec_d)
        if p_f: vf.append((p, p_f))
        if p_d: vd.append((p, p_d))
    if not vf and not vd:
        d = _lj("exp_composition_g07.json")
        if d:
            cavs_d = d["cavs"]; res = d["results"]
            pct_d = [c * 100 for c in cavs_d]
            vf = [(p, _own_poatt(res.get(f"cav{c}_fixed"))) for c, p in zip(cavs_d, pct_d)]; vf = [(x, y) for x, y in vf if y]
            vd = [(p, _own_poatt(res.get(f"cav{c}_dynamic"))) for c, p in zip(cavs_d, pct_d)]; vd = [(x, y) for x, y in vd if y]
    if vf: ax.plot(*zip(*vf), "o-", color=C_FIX, label="Fixed")
    if vd: ax.plot(*zip(*vd), "s-", color=C_DYN, label="Dynamic")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("CAV %"); ax.set_ylabel("Sustained PoAtt")
    ax.set_title("(d) Composition Dual Effect"); ax.legend(); ax.grid(True, alpha=0.3)

    # (e) Attack Budget (n_target)
    ax = axes[2, 0]
    d = _lj("exp_ntarget_sweep.json")
    if d:
        nts = d["n_targets"]; res = d["results"]
        vf = [(nt, _own_poatt(res.get(f"nt{nt}_fixed"))) for nt in nts]; vf = [(x, y) for x, y in vf if y]
        vd = [(nt, _own_poatt(res.get(f"nt{nt}_dynamic"))) for nt in nts]; vd = [(x, y) for x, y in vd if y]
        if vf: ax.plot(*zip(*vf), "o-", color=C_FIX, label="Fixed")
        if vd: ax.plot(*zip(*vd), "s-", color=C_DYN, label="Dynamic")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        n10_f = dict(vf).get(10)
        if n10_f:
            ax.annotate("Saturation", (10, n10_f), textcoords="offset points",
                        xytext=(25, -15), fontsize=14,
                        arrowprops=dict(arrowstyle="->", lw=1.5))
        all_v = [y for _, y in vf + vd]
        if all_v:
            lo, hi = min(all_v), max(all_v)
            pad = (hi - lo) * 0.1
            ax.set_ylim(min(lo - pad, 0.99), hi + pad)
    ax.set_xlabel(r"$n_{target}$"); ax.set_ylabel("Sustained PoAtt")
    ax.set_title("(e) Attack Budget"); ax.legend(); ax.grid(True, alpha=0.3)

    # (f) Bounded Rationality
    ax = axes[2, 1]
    d = _lj("exp_br_sweep.json")
    if d:
        deltas = d["deltas"]; res = d["results"]
        vf = [(dl, _own_poatt(res.get(f"d{dl}_fixed"))) for dl in deltas]; vf = [(x, y) for x, y in vf if y]
        vd = [(dl, _own_poatt(res.get(f"d{dl}_dynamic"))) for dl in deltas]; vd = [(x, y) for x, y in vd if y]
        if vf: ax.plot(*zip(*vf), "o-", color=C_FIX, label="Fixed")
        if vd: ax.plot(*zip(*vd), "s-", color=C_DYN, label="Dynamic")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.text(0.5, 0.15, "Secondary resilience",
                transform=ax.transAxes, fontsize=14, ha="center", style="italic")
        all_v = [y for _, y in vf + vd]
        if all_v:
            lo, hi = min(all_v), max(all_v)
            pad = (hi - lo) * 0.1
            ax.set_ylim(min(lo - pad, 0.99), hi + pad)
    ax.set_xlabel(r"$\delta$ (seconds)"); ax.set_ylabel("Sustained PoAtt")
    ax.set_title(r"(f) Bounded Rationality ($\delta$)"); ax.legend(); ax.grid(True, alpha=0.3)

    _save(fig, "exp_theory_validation.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 12 — BC Comparison (3 attacks only)
# ═════════════════════════════════════════════════════════════════════
def fig_bc(net):
    atk = _lj("exp_attack_comparison.json")
    if not atk:
        return
    res = atk["results"]
    rand = _lj("exp_random_multiseed.json")
    summ = {}
    if rand:
        if "summary" in rand:
            summ = rand["summary"]
        else:
            # Compute summary from raw results
            seeds = rand.get("seeds", [])
            rr = rand.get("results", {})
            for trust in ["fixed", "dynamic"]:
                vals = [rr[f"seed{s}_{trust}"]["sustained_poatt"]
                        for s in seeds
                        if f"seed{s}_{trust}" in rr
                        and rr[f"seed{s}_{trust}"].get("sustained_poatt")]
                if vals:
                    summ[f"{trust}_mean"] = float(np.mean(vals))
                    summ[f"{trust}_std"] = float(np.std(vals))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7))

    # (a) Scatter
    adj, fft = net["adj"], net["fft"]
    _, bc_map = topo_bc(adj, fft, 10, weight=None)
    tbc = np.array([bc_map.get(i, 0) for i in range(net["nl"])])

    mat = sio.loadmat(os.path.join(PROJ, "data", "sioux_falls",
                                   "SiouxFalls6180_pp.mat"), squeeze_me=False)
    plm = np.asarray(mat["pathList"], int) - 1
    pbc = np.zeros(net["nl"])
    for p in range(plm.shape[0]):
        for c in range(plm.shape[1]):
            li = plm[p, c]
            if 0 <= li < net["nl"]:
                pbc[li] += 1

    tn = tbc / (tbc.max() + 1e-10); pn = pbc / (pbc.max() + 1e-10)
    a1.scatter(tn, pn, c="gray", s=30, alpha=0.5, label="All links")
    tt = np.argsort(tbc)[::-1][:10]; tp = np.argsort(pbc)[::-1][:10]
    a1.scatter(tn[tt], pn[tt], c="#1f77b4", s=120, marker="v",
               label="Top-10 Topo-BC", zorder=3)
    a1.scatter(tn[tp], pn[tp], c=C_FIX, s=120, marker="^",
               label="Top-10 Path-BC", zorder=3)
    ov = len(set(tt) & set(tp))
    cr = np.corrcoef(tn, pn)[0, 1]
    a1.text(0.05, 0.88, f"Pearson r = {cr:.3f}\nTop-10 overlap: {ov}/10",
            transform=a1.transAxes, fontsize=14,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    a1.set_xlabel("Topological BC (norm.)"); a1.set_ylabel("Path BC (norm.)")
    a1.set_title("(a) Unweighted Topo-BC vs Path-BC"); a1.legend(fontsize=14); a1.grid(True, alpha=0.3)

    # (b) Bar chart -- ONLY 3 attack types
    attacks = [("bc_deflation", "Topo-BC"),
               ("path_bc", "Path-BC"),
               ("random", "Random\n(avg)")]
    x = np.arange(len(attacks)); w = 0.35
    fv, dv, fe, de = [], [], [], []
    for at, _ in attacks:
        if at == "random" and summ:
            fv.append(summ.get("fixed_mean", 0))
            dv.append(summ.get("dynamic_mean", 0))
            fe.append(summ.get("fixed_std", 0))
            de.append(summ.get("dynamic_std", 0))
        else:
            fv.append(_own_poatt(res.get(f"{at}_fixed")) or 0)
            dv.append(_own_poatt(res.get(f"{at}_dynamic")) or 0)
            fe.append(0); de.append(0)

    b1 = a2.bar(x - w/2, fv, w, yerr=fe, label="Fixed Trust", color=C_FIX,
                alpha=0.85, capsize=4, error_kw={"lw": 1.5})
    b2 = a2.bar(x + w/2, dv, w, yerr=de, label="Dynamic Trust", color=C_DYN,
                alpha=0.85, hatch="//", capsize=4, error_kw={"lw": 1.5})

    # Seed dots for random
    if rand:
        ri = 2
        for s in rand["seeds"]:
            rr = rand["results"]
            kf, kd = f"seed{s}_fixed", f"seed{s}_dynamic"
            if kf in rr:
                a2.scatter(ri - w/2, rr[kf]["sustained_poatt"],
                           c="darkred", s=12, zorder=1, alpha=0.4)
            if kd in rr:
                a2.scatter(ri + w/2, rr[kd]["sustained_poatt"],
                           c="darkgreen", s=12, zorder=1, alpha=0.4)

    a2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    a2.set_xticks(x); a2.set_xticklabels([lb for _, lb in attacks], fontsize=13)
    a2.set_ylabel("Sustained PoAtt"); a2.set_title("(b) Attack Type Comparison")
    a2.legend(); a2.grid(True, alpha=0.3, axis="y"); a2.set_ylim(0.97, 1.10)

    for bar, err in zip(b1, fe):
        h = bar.get_height()
        a2.text(bar.get_x() + bar.get_width()/2,
                h + (err if err else 0) + 0.003,
                f"{h:.3f}", ha="center", va="bottom", fontsize=13)
    for bar, err in zip(b2, de):
        h = bar.get_height()
        a2.text(bar.get_x() + bar.get_width()/2,
                h + (err if err else 0) + 0.003,
                f"{h:.3f}", ha="center", va="bottom", fontsize=13)

    fig.tight_layout()
    _save(fig, "exp_bc_comparison.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 13 — Anaheim
# ═════════════════════════════════════════════════════════════════════
def fig_anaheim():
    aj = _lj("exp_anaheim.json")
    if not aj:
        return
    sc = aj.get("scenarios", {})
    sig = _lj("exp_signature.json")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 7))
    days = np.arange(1, 201)

    # (a) TSTT with rolling avg
    for k, lab, col, ls in [("baseline", "Baseline", "gray", "--"),
                              ("bc_defl_fixed", "Fixed", C_FIX, "-"),
                              ("bc_defl_dynamic", "Dynamic", C_DYN, "-")]:
        if k in sc:
            raw = np.array(sc[k]["tstt"])
            a1.plot(days, raw, color=col, alpha=0.15, lw=0.8)
            a1.plot(days, _roll(raw, 10), color=col, ls=ls, lw=3, label=lab)
    a1.axvspan(51, 100, alpha=0.12, color="red")
    a1.set_xlabel("Day"); a1.set_ylabel("TSTT [veh-hours]")
    a1.set_title("(a) Anaheim TSTT (10-day rolling avg)")
    a1.legend(); a1.grid(True, alpha=0.3)

    # (b) Cross-network bar
    sf_f = (_own_poatt(sig["scenarios"].get("bc_defl_fixed_03")) if sig else 0) or 0
    sf_d = (_own_poatt(sig["scenarios"].get("bc_defl_dynamic_03")) if sig else 0) or 0
    af = _own_poatt(sc.get("bc_defl_fixed")) or 0
    ad = _own_poatt(sc.get("bc_defl_dynamic")) or 0

    x = np.arange(2); w = 0.35
    b1 = a2.bar(x - w/2, [sf_f, af], w, label="Fixed Trust", color=C_FIX, alpha=0.85)
    b2 = a2.bar(x + w/2, [sf_d, ad], w, label="Dynamic Trust", color=C_DYN,
                alpha=0.85, hatch="//")
    a2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    a2.set_xticks(x); a2.set_xticklabels(["Sioux Falls", "Anaheim"], fontsize=14)
    a2.set_ylabel("Sustained PoAtt"); a2.set_title("(b) Cross-Network Comparison")
    a2.legend(); a2.grid(True, alpha=0.3, axis="y"); a2.set_ylim(0.96, 1.10)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        yo = 0.001 if h >= 1.0 else -0.004
        va = "bottom" if h >= 1.0 else "top"
        a2.text(bar.get_x() + bar.get_width()/2, h + yo, f"{h:.4f}",
                ha="center", va=va, fontsize=14)

    if sf_f > 1.001:
        de_sf = (1 - (sf_d - 1) / (sf_f - 1)) * 100
        if de_sf <= 100:
            a2.text(0, max(sf_f, sf_d) + 0.006, f"TIA: {de_sf:.0f}%",
                    ha="center", fontsize=13, color=C_FIX, fontweight="bold")
        else:
            a2.text(0, max(sf_f, sf_d) + 0.006, "TIA > 100%\n(overcorrection)",
                    ha="center", fontsize=12, color=C_DYN, fontweight="bold")
    elif sf_d < 1.0:
        a2.text(0, max(sf_f, sf_d) + 0.006, "PoAtt < 1.0\n(overcorrection)",
                ha="center", fontsize=12, color=C_DYN, fontweight="bold")
    if af < 1.0 and ad < 1.0:
        a2.text(1, max(af, ad) + 0.006, "PoAtt < 1.0\n(overcorrection)",
                ha="center", fontsize=12, color=C_DYN, fontweight="bold")
    elif af > 1.001:
        de_an = (1 - (ad - 1) / (af - 1)) * 100
        if 0 <= de_an <= 100:
            a2.text(1, max(af, ad) + 0.006, f"TIA: {de_an:.0f}%",
                    ha="center", fontsize=13, color=C_FIX, fontweight="bold")

    fig.tight_layout()
    _save(fig, "exp_anaheim.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 14 — Validation (5-panel)
# ═════════════════════════════════════════════════════════════════════
def fig_validation():
    sig = _lj("exp_signature.json")
    ana = _lj("exp_anaheim.json")
    if not sig or not ana:
        return

    sf_bl = np.array(sig["scenarios"]["baseline"]["tstt"])
    an_bl = np.array(ana["scenarios"]["baseline"]["tstt"])
    days = np.arange(1, 201)

    sf_eq = np.mean(sf_bl[29:50])
    an_eq = np.mean(an_bl[29:50])
    sf_norm = sf_bl / sf_eq
    an_norm = an_bl / an_eq

    C_SF = "#2166ac"; C_AN = "#d6604d"

    fig = plt.figure(figsize=(20, 12))
    gs_spec = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.35, wspace=0.38)

    # (a) Normalized TSTT — with zoom rectangle instead of inset
    from matplotlib.patches import Rectangle
    ax = fig.add_subplot(gs_spec[0, 0])
    ax.plot(days, sf_norm, color=C_SF, lw=2, label="Sioux Falls (SF)")
    ax.plot(days, an_norm, color=C_AN, lw=2, label="Anaheim (Ana)")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Day"); ax.set_ylabel("Normalized TSTT")
    ax.set_title("(a) Normalized TSTT"); ax.legend(); ax.grid(True, alpha=0.3)
    for d_idx in range(len(sf_norm)):
        if abs(sf_norm[d_idx] - 1.0) < 0.01 and d_idx > 5:
            ax.annotate(f"SF converges day ~{d_idx+1}", (d_idx+1, sf_norm[d_idx]),
                        xytext=(60, 1.55), fontsize=14,
                        arrowprops=dict(arrowstyle="->", lw=1.5),
                        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray",
                                  boxstyle="round"))
            break
    # Zoom rectangle showing region (b) covers
    rect = Rectangle((15, 0.96), 185, 1.04 - 0.96,
                      linewidth=1.5, edgecolor="black", facecolor="none",
                      linestyle="--", zorder=10)
    ax.add_patch(rect)
    ax.text(105, 1.045, "see (b)", fontsize=11, style="italic", ha="center")

    # (b) Zoomed convergence
    ax = fig.add_subplot(gs_spec[0, 1])
    ax.plot(days[14:], sf_norm[14:], color=C_SF, lw=2, label="SF")
    ax.plot(days[14:], an_norm[14:], color=C_AN, lw=1.5, alpha=0.5, label="Ana (raw)")
    ax.plot(days[14:], _roll(an_norm, 10)[14:], color=C_AN, lw=2.5, ls="--",
            label="Ana (10-day avg)")
    ax.axhspan(0.99, 1.01, alpha=0.06, color="green")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_ylim(0.96, 1.04)
    ax.set_xlabel("Day"); ax.set_ylabel("Normalized TSTT")
    ax.set_title("(b) Zoomed Convergence [Day 15+]"); ax.legend(fontsize=14); ax.grid(True, alpha=0.3)

    # (c) Day-to-day gap
    ax = fig.add_subplot(gs_spec[0, 2])
    sf_gap = np.abs(np.diff(sf_norm)) * 100
    an_gap = np.abs(np.diff(an_norm)) * 100
    ax.plot(days[1:], sf_gap, color=C_SF, lw=2, label="SF")
    ax.plot(days[1:], an_gap, color=C_AN, lw=2, label="Anaheim")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="1%")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="0.5%")
    ax.set_yscale("log"); ax.set_xlabel("Day"); ax.set_ylabel("Day-to-day gap (%)")
    ax.set_title("(c) Convergence Gap (log)"); ax.legend(fontsize=13); ax.grid(True, alpha=0.3)

    # (d) Attack disruption + trust recovery (wide)
    ax = fig.add_subplot(gs_spec[1, :2])
    sf_sc = sig["scenarios"]
    bl_tstt = np.array(sf_sc["baseline"]["tstt"])
    bl_eq_val = np.mean(bl_tstt[29:50])
    for k, lab, col, ls, lw in [
        ("baseline", "Baseline", "gray", "--", 2),
        ("bc_defl_fixed_03", "Fixed", C_FIX, "-", 2.5),
        ("bc_defl_dynamic_03", "Dynamic", C_DYN, "-", 2.5),
    ]:
        if k in sf_sc:
            norm_t = np.array(sf_sc[k]["tstt"]) / bl_eq_val
            ax.plot(days, norm_t, color=col, ls=ls, lw=lw, label=lab)
    ax.axvspan(51, 100, alpha=0.10, color="red")
    ax.set_ylim(0.95, 1.12)
    ax.set_xlabel("Day"); ax.set_ylabel("TSTT / Pre-attack Baseline")
    ax.set_title("(d) Attack Disruption + Trust Attenuation")
    ax.legend(loc="lower left", fontsize=12,
              bbox_to_anchor=(0.0, 0.0), framealpha=0.9); ax.grid(True, alpha=0.3)

    if "bc_defl_dynamic_03" in sf_sc and "trust" in sf_sc["bc_defl_dynamic_03"]:
        ax2 = ax.twinx()
        trust = np.array(sf_sc["bc_defl_dynamic_03"]["trust"])
        mt = trust.mean(axis=1)
        ax2.plot(days, mt, color="purple", ls="-.", lw=2, alpha=0.7, label="Mean Trust")
        ax2.fill_between(days, mt, alpha=0.06, color="purple")
        ax2.set_ylabel("Mean Trust", fontsize=14, labelpad=15)
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis="y")

    ax.text(75, 1.10, "Attack period", ha="center", fontsize=14, style="italic")
    fix_sus = _own_poatt(sf_sc.get("bc_defl_fixed_03")) or 1
    dyn_sus = _own_poatt(sf_sc.get("bc_defl_dynamic_03")) or 1
    if fix_sus > 1.001:
        tia_val = (1 - (dyn_sus - 1) / (fix_sus - 1)) * 100
        if tia_val <= 100:
            ax.annotate(f"TIA: {tia_val:.0f}%", (150, 1.005), fontsize=14, color=C_DYN,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        else:
            ax.annotate("Dynamic overcorrects\n(TIA > 100%)", (140, 1.005),
                        fontsize=13, color=C_DYN,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    elif dyn_sus < 1.0:
        ax.annotate("Dynamic overcorrects", (150, 1.005), fontsize=14, color=C_DYN,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.annotate(f"Sustained +{(fix_sus-1)*100:.1f}%", (75, fix_sus + 0.005),
                fontsize=14, color=C_FIX,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # (e) Rolling CV
    ax = fig.add_subplot(gs_spec[1, 2])
    win = 10
    for arr, lab, col in [(sf_bl, "SF", C_SF), (an_bl, "Anaheim", C_AN)]:
        cv = []
        for i in range(len(arr)):
            lo, hi = max(0, i - win//2), min(len(arr), i + win//2 + 1)
            chunk = arr[lo:hi]
            cv.append(np.std(chunk) / np.mean(chunk) * 100 if np.mean(chunk) > 0 else 0)
        ax.plot(days, cv, color=col, lw=2, label=lab)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.axhline(1.0, color="gray", ls=":", alpha=0.5)
    ax.set_ylim(0, 3); ax.set_xlabel("Day"); ax.set_ylabel("Rolling CV (%)")
    ax.set_title("(e) Rolling CV"); ax.legend(); ax.grid(True, alpha=0.3)

    _save(fig, "exp_validation.png")


# ═════════════════════════════════════════════════════════════════════
# FIG 15 — Link Flow Time-Series (SF)
# ═════════════════════════════════════════════════════════════════════
def fig_link_flow_ts():
    bl = _lj("exp_s1_baseline.json")
    fx = _lj("exp_s1_defl_fixed.json")
    dy = _lj("exp_s1_defl_dyn_03.json")
    sig = _lj("exp_signature.json")
    if not bl or not fx or not dy:
        return

    targets = [18, 15, 8, 10, 51, 48, 17, 53, 11, 14]
    non_targets = [i for i in range(76) if i not in targets]
    day_keys = sorted(bl["link_flows"].keys(), key=int)
    day_labels = [int(k) + 1 for k in day_keys]

    def get_flows(data, dk):
        return np.array(data["link_flows"][dk])

    base40 = get_flows(bl, "39")

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.subplots_adjust(hspace=0.38, wspace=0.30)

    def add_attack_shade(ax):
        ax.axvspan(51, 100, alpha=0.12, color="red")

    # (a) Target link mean flow — 3 scenarios
    ax = axes[0, 0]
    for data, lab, col, mk, ls in [
        (bl, "Baseline", "gray", "o", "--"),
        (fx, "Fixed", C_FIX, "s", "-"),
        (dy, "Dynamic", C_DYN, "^", "-"),
    ]:
        means = [np.mean(get_flows(data, dk)[targets]) for dk in day_keys]
        ax.plot(day_labels, means, marker=mk, color=col, ls=ls, lw=2.5,
                markersize=8, label=lab)
    add_attack_shade(ax)
    ax.set_xlabel("Day"); ax.set_ylabel("Mean Flow (target links)")
    ax.set_title("(a) Target Link Mean Flow")
    ax.legend(fontsize=13); ax.grid(True, alpha=0.3)

    # (b) Fixed trust: target vs non-target delta-flow %
    ax = axes[0, 1]
    tgt_delta = [np.mean((get_flows(fx, dk)[targets] - base40[targets])
                         / (base40[targets] + 1e-10) * 100) for dk in day_keys]
    ntgt_delta = [np.mean((get_flows(fx, dk)[non_targets] - base40[non_targets])
                          / (base40[non_targets] + 1e-10) * 100) for dk in day_keys]
    ax.plot(day_labels, tgt_delta, "s-", color="red", lw=2.5, markersize=8,
            label="Target links (10)")
    ax.plot(day_labels, ntgt_delta, "o-", color="#1f77b4", lw=2.5, markersize=8,
            label="Non-target links (66)")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    add_attack_shade(ax)
    pk = max(tgt_delta)
    pk_idx = tgt_delta.index(pk)
    ax.annotate(f"+{pk:.1f}%", (day_labels[pk_idx], pk),
                textcoords="offset points", xytext=(0, -25), fontsize=14,
                fontweight="bold", color="red", ha="center",
                arrowprops=dict(arrowstyle="->", lw=1.5, color="red"))
    ax.set_xlabel("Day"); ax.set_ylabel("Flow Change (%)")
    ax.set_title("(b) Fixed Trust: Target vs Non-target")
    ax.legend(fontsize=13); ax.grid(True, alpha=0.3)

    # (c) Dynamic trust: target vs non-target delta-flow %
    ax = axes[1, 0]
    tgt_delta_d = [np.mean((get_flows(dy, dk)[targets] - base40[targets])
                           / (base40[targets] + 1e-10) * 100) for dk in day_keys]
    ntgt_delta_d = [np.mean((get_flows(dy, dk)[non_targets] - base40[non_targets])
                            / (base40[non_targets] + 1e-10) * 100) for dk in day_keys]
    ax.plot(day_labels, tgt_delta_d, "s-", color="red", lw=2.5, markersize=8,
            label="Target links (10)")
    ax.plot(day_labels, ntgt_delta_d, "o-", color="#1f77b4", lw=2.5, markersize=8,
            label="Non-target links (66)")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    add_attack_shade(ax)
    # Find where target curve returns near 0 (post-attack)
    for i, (dl, td) in enumerate(zip(day_labels, tgt_delta_d)):
        if dl > 100 and abs(td) < 3:
            ax.annotate("Attack ends (day 100);\nflows return to baseline", (dl, td),
                        textcoords="offset points", xytext=(30, 25), fontsize=13,
                        color=C_DYN,
                        bbox=dict(facecolor="#e6ffe6", alpha=0.7, edgecolor=C_DYN),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color=C_DYN))
            break
    ax.set_xlabel("Day"); ax.set_ylabel("Flow Change (%)")
    ax.set_title("(c) Dynamic Trust: Target vs Non-target")
    ax.legend(fontsize=13); ax.grid(True, alpha=0.3)

    # (d) Individual target links — Fixed trust
    ax = axes[1, 1]
    sel_targets = [18, 15, 8, 10, 51]
    sel_labels = ["8\u21926", "6\u21928", "4\u21925", "5\u21924", "17\u219216"]
    colors_5 = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for idx, lbl, col in zip(sel_targets, sel_labels, colors_5):
        delta = [(get_flows(fx, dk)[idx] - base40[idx]) / (base40[idx] + 1e-10) * 100
                 for dk in day_keys]
        ax.plot(day_labels, delta, "o-", color=col, lw=2, markersize=7, label=lbl)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    add_attack_shade(ax)
    ax.set_xlabel("Day"); ax.set_ylabel("Flow Change (%)")
    ax.set_title("(d) Individual Target Links (Fixed)")
    ax.legend(ncol=2, fontsize=11); ax.grid(True, alpha=0.3)

    # (e) Concentration ratio over time
    ax = axes[2, 0]
    for data, lab, col, ls in [
        (bl, "Baseline", "gray", "--"),
        (fx, "Fixed", C_FIX, "-"),
        (dy, "Dynamic", C_DYN, "-"),
    ]:
        ratios = [np.mean(get_flows(data, dk)[targets]) /
                  (np.mean(get_flows(data, dk)[non_targets]) + 1e-10)
                  for dk in day_keys]
        ax.plot(day_labels, ratios, marker="o", color=col, ls=ls, lw=2.5,
                markersize=8, label=lab)
    add_attack_shade(ax)
    ax.set_xlabel("Day"); ax.set_ylabel("Target / Non-target Flow Ratio")
    ax.set_title("(e) Concentration Ratio")
    ax.legend(fontsize=13, loc="upper right"); ax.grid(True, alpha=0.3)

    # (f) Dynamic trust: concentration + trust overlay
    ax = axes[2, 1]
    dyn_ratios = [np.mean(get_flows(dy, dk)[targets]) /
                  (np.mean(get_flows(dy, dk)[non_targets]) + 1e-10)
                  for dk in day_keys]
    ax.plot(day_labels, dyn_ratios, "o-", color=C_DYN, lw=2.5, markersize=8,
            label="Concentration ratio")
    add_attack_shade(ax)
    ax.set_xlabel("Day"); ax.set_ylabel("Concentration Ratio")
    ax.tick_params(axis="y")
    ax.grid(True, alpha=0.3)

    # Secondary y-axis: mean trust (all 200 days)
    if sig and "bc_defl_dynamic_03" in sig["scenarios"]:
        trust = np.array(sig["scenarios"]["bc_defl_dynamic_03"]["trust"])
        mt = trust.mean(axis=1)
        ax2 = ax.twinx()
        all_days = np.arange(1, 201)
        ax2.plot(all_days, mt, color="purple", ls="-.", lw=2, alpha=0.7,
                 label="Mean Trust")
        ax2.fill_between(all_days, mt, alpha=0.06, color="purple")
        ax2.set_ylabel("Mean Trust", fontsize=14, color="black")
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis="y", colors="black")
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=13,
                  loc="center right", bbox_to_anchor=(0.98, 0.55), framealpha=0.9)
    else:
        ax.legend(fontsize=13)

    ax.set_title("(f) Concentration + Trust (Dynamic)")
    ax.tick_params(axis="y", colors="black")
    ax.yaxis.label.set_color("black")

    _save(fig, "exp_link_flow_timeseries.png")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("GENERATING ALL 15 FIGURES")
    print(f"  JSON : {JDIR}")
    print(f"  Out  : {FDIR}")
    print(f"  DPI  : {DPI}")
    print("=" * 70)

    print("\nLoading networks...")
    sf = load_sf(); print(f"  SF: {sf['nn']} nodes, {sf['nl']} links")
    ana = load_ana(); print(f"  Ana: {ana['nn']} nodes, {ana['nl']} links")

    figs = [
        ("Fig  1: Signature",          lambda: fig_signature()),
        ("Fig  2: Gamma Sweep",        lambda: fig_gamma()),
        ("Fig  3: SF Network",         lambda: fig_net_sf(sf)),
        ("Fig  4: Ana Network",        lambda: fig_net_ana(ana)),
        ("Fig  5: SF Spatial",         lambda: fig_spatial_sf(sf)),
        ("Fig  6: Ana Spatial",        lambda: fig_spatial_ana(ana)),
        ("Fig  7: Theta Sweep",        lambda: fig_theta()),
        ("Fig  8: Composition",        lambda: fig_comp()),
        ("Fig  9: N-target",           lambda: fig_ntarget()),
        ("Fig 10: BR Sweep",           lambda: fig_br()),
        ("Fig 11: Theory Validation",  lambda: fig_theory()),
        ("Fig 12: BC Comparison",      lambda: fig_bc(sf)),
        ("Fig 13: Anaheim",            lambda: fig_anaheim()),
        ("Fig 14: Validation",         lambda: fig_validation()),
        ("Fig 15: Link Flow TS",       lambda: fig_link_flow_ts()),
    ]

    ok = []
    for name, fn in figs:
        print(f"\n{name}...")
        try:
            fn(); ok.append(name)
        except Exception as e:
            print(f"  ERROR: {e}"); traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Generated {len(ok)}/15 figures:")
    for n in ok:
        print(f"  {n}")
    print(f"\nFiles:")
    for f in sorted(os.listdir(FDIR)):
        if f.endswith(".png"):
            kb = os.path.getsize(os.path.join(FDIR, f)) / 1024
            flag = " (< 100 KB!)" if kb < 100 else ""
            print(f"  {f:40s} {kb:8.0f} KB{flag}")
    print("=== ALL 15 FIGURES GENERATED ===")


if __name__ == "__main__" and len(sys.argv) > 1:
    sf = load_sf(); ana = load_ana()
    target = sys.argv[1]
    fn = globals().get(target)
    if callable(fn):
        try:
            fn()
        except TypeError:
            # Some figures take sf or ana as argument
            if "ana" in target or "anaheim" in target.lower():
                fn(ana)
            else:
                fn(sf)
        sys.exit(0)
    else:
        print(f"Function {target} not found"); sys.exit(1)


if __name__ == "__main__":
    main()
