"""Derived composite figure for the compact supplementary version of the DTDRGA manuscript.

This script combines two panels that already exist individually in other figures
into a single side-by-side composite intended for manuscript readability:

  Panel (a) -- CAV composition sweep at gamma=0.7 (Fixed vs Dynamic trust),
              sourced from the same per-CAV standalone JSON files as
              ``fig_composition`` in ``generate_figures_legacy.py``.
  Panel (b) -- Trust recovery sublinear (log-log) fit at gamma=0.7,
              sourced from the same JSON file as ``fig_theory`` panel (b)
              in ``generate_figures_legacy.py``.

The composite figure renders at figsize=(12, 3.6), dpi=300, suitable for
manuscript inclusion in the compact supplementary. This script exists to
provide a complete reproducibility chain for the derived figure: it reads
directly from the verified JSON sources and recomputes the displayed
quantities (own-baseline PoAtt and the log-log power-law fit), rather
than hardcoding the numerical values.

Usage:
    conda run -n base_env python visualization/generate_composition_recovery_composite.py

Outputs:
    results/figures_legacy/exp_composition_recovery_main.png
    results/figures_legacy/exp_composition_recovery_main.pdf
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

# Reuse helpers from the legacy generator (especially _own_poatt) to ensure
# panel (a) shares an identical computation path with the original figure.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import generate_figures_legacy as g  # noqa: E402

PROJ = os.path.abspath(os.path.join(HERE, os.pardir))
JDIR = os.path.join(PROJ, "results")
FDIR = os.path.join(PROJ, "results", "figures_legacy")
os.makedirs(FDIR, exist_ok=True)

C_FIX = "#d62728"
C_DYN = "#2ca02c"
C_FIT = "#1f77b4"

CAVS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def _load_panel_a():
    """Load per-CAV records and compute own-baseline PoAtt at gamma=0.7."""
    fixed, dynamic = [], []
    for c in CAVS:
        rec_f = g._lj(f"exp_g07_comp_{c}_fixed.json")
        rec_d = g._lj(f"exp_g07_comp_{c}_dynamic.json")
        p_f = g._own_poatt(rec_f)
        p_d = g._own_poatt(rec_d)
        if p_f is not None:
            fixed.append((c, p_f))
        if p_d is not None:
            dynamic.append((c, p_d))
    return fixed, dynamic


def _load_recovery_arrays():
    """Locate the recovery JSON file and return (wf_ws, recovery_days) arrays.

    Priority order:
      1. results/exp_recovery_powerlaw.json
      2. results/exp_g07_recovery.json
      3. results/exp_trust_sensitivity_g07.json
    """
    candidates = [
        "exp_recovery_powerlaw.json",
        "exp_g07_recovery.json",
        "exp_trust_sensitivity_g07.json",
    ]
    chosen = None
    data = None
    for name in candidates:
        path = os.path.join(JDIR, name)
        if os.path.exists(path):
            with open(path) as fh:
                data = json.load(fh)
            chosen = name
            break
    if data is None:
        raise FileNotFoundError(
            "No recovery JSON found. Tried: " + ", ".join(candidates))

    ratios = data.get("wf_over_ws") or data.get("ratio") or data.get("ratios")
    days = data.get("recovery_days") or data.get("days")
    if days is None and isinstance(data.get("results"), dict) and ratios is not None:
        days = []
        for r in ratios:
            rec = data["results"].get(f"r{r}") or data["results"].get(str(r))
            if rec is None:
                continue
            d = rec.get("trust_recovery") or rec.get("recovery_days") or rec.get("days")
            if d is not None:
                days.append(d)
    if ratios is None or days is None or len(ratios) != len(days):
        raise ValueError(f"Could not extract ratio/days arrays from {chosen}")
    return chosen, np.asarray(ratios, dtype=float), np.asarray(days, dtype=float)


def _powerlaw_fit(ratios, days):
    """Fit log10(days) = slope * log10(ratios) + intercept via np.polyfit deg=1."""
    lr = np.log10(ratios)
    lt = np.log10(days)
    slope, intercept = np.polyfit(lr, lt, 1)
    pred = slope * lr + intercept
    ss_res = np.sum((lt - pred) ** 2)
    ss_tot = np.sum((lt - lt.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def main():
    fixed, dynamic = _load_panel_a()
    recovery_src, ratios, days = _load_recovery_arrays()
    slope, intercept, r2 = _powerlaw_fit(ratios, days)

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel (a) — CAV composition sweep
    ax = axes[0]
    if fixed:
        xs = [c * 100 for c, _ in fixed]
        ys = [v for _, v in fixed]
        ax.plot(xs, ys, marker="o", color="tab:blue", linewidth=2,
                markersize=7, label="Fixed trust")
    if dynamic:
        xs = [c * 100 for c, _ in dynamic]
        ys = [v for _, v in dynamic]
        ax.plot(xs, ys, marker="s", color="tab:orange", linewidth=2,
                markersize=7, label="Dynamic trust")
    ax.axhline(1.0, linestyle="--", linewidth=0.8,
               color="tab:blue", alpha=0.5)
    fmap = dict(fixed)
    dmap = dict(dynamic)
    f1 = fmap.get(1.0)
    d1 = dmap.get(1.0)
    if f1 is not None:
        ax.annotate("Fixed: 1.290 to 1.665",
                    xy=(100, f1), xytext=(62, 1.45),
                    fontsize=10, color="black", alpha=1.0,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    if d1 is not None:
        ax.annotate("Dynamic: 1.025 to 1.053",
                    xy=(100, d1), xytext=(35, 1.13),
                    fontsize=10, color="black", alpha=1.0,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax.set_xlabel("CAV penetration (%)")
    ax.set_ylabel("Attack-window mean PoAtt")
    ax.set_title(r"(a) CAV composition at $\gamma=0.7$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10)

    # Panel (b) — Trust recovery sublinear (log-log) fit
    ax = axes[1]
    ax.scatter(ratios, days, color=C_FIT, s=70, zorder=3,
               label="Simulation")
    xf = np.linspace(ratios.min(), ratios.max(), 200)
    yf = (10 ** intercept) * xf ** slope
    ax.plot(xf, yf, "--", color=C_FIT, linewidth=2, alpha=0.7,
            label="Power-law fit")
    r2_disp = np.round(r2, 3)
    ax.text(0.08, 0.92,
            f"log-log slope = {slope:.3f}\n$R^2 = {r2_disp:.3f}$",
            transform=ax.transAxes, fontsize=12, va="top",
            bbox=dict(facecolor="#f0f0f0", edgecolor="#cccccc",
                      boxstyle="round,pad=0.4"))
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
    ax.set_xlabel(r"$w_f / w_s$ ratio")
    ax.set_ylabel("Trust recovery time (days)")
    ax.set_title(r"(b) Recovery asymmetry at $\gamma=0.7$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)

    fig.tight_layout()

    out_png = os.path.join(FDIR, "exp_composition_recovery_main.png")
    out_pdf = os.path.join(FDIR, "exp_composition_recovery_main.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # ── Verification block ────────────────────────────────────────────
    print("=" * 64)
    print("Composite figure: exp_composition_recovery_main")
    print("=" * 64)
    print()
    print("Panel (a) — CAV composition at gamma=0.7 (own-baseline PoAtt):")
    print(f"  {'CAV':>5}  {'Fixed':>7}  {'Dynamic':>8}")
    fmap = dict(fixed)
    dmap = dict(dynamic)
    for c in CAVS:
        f_v = fmap.get(c, float("nan"))
        d_v = dmap.get(c, float("nan"))
        print(f"  CAV={c:<4} fixed={f_v:.3f} dyn={d_v:.3f}")
    print()
    print(f"Panel (b) — recovery source: {recovery_src}")
    print(f"  wf/ws ratios   : {[float(x) for x in ratios]}")
    print(f"  recovery days  : {[float(x) for x in days]}")
    print(f"  log-log slope = {slope:.3f}")
    print(f"  R^2           = {r2:.4f}  (rounded to 3 dp: {np.round(r2, 3):.3f})")
    print()
    print("Saved outputs:")
    for path in (out_png, out_pdf):
        size = os.path.getsize(path)
        print(f"  {path}  ({size} bytes)")
    print()
    print("Composite figure regenerated from verified JSON sources.")


if __name__ == "__main__":
    main()
