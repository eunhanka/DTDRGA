"""Step 4: Corrected Visualization — Standalone Script.

Reads corrected_exp_*.json and new experiment results to generate updated figures.
Can be re-run independently: python experiments/generate_corrected_figures.py

NO simulation code — only reads saved JSON and generates plots.
"""

import json
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- Global Style (dpi=200, serif, readable) ----
matplotlib.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 20,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
})

DPI = 200

COLORS = {
    'fixed': '#d62728',      # red
    'dynamic': '#2ca02c',    # green
    'baseline': '#7f7f7f',   # gray
}

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RES_DIR = os.path.join(BASE_DIR, os.pardir, "results")
FIG_DIR = os.path.join(BASE_DIR, os.pardir, "results", "figures_current")
os.makedirs(FIG_DIR, exist_ok=True)

ATTACK_START = 50
ATTACK_END = 100
N_DAYS = 200


def load(filename):
    path = os.path.join(RES_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def shade_attack(ax):
    ax.axvspan(ATTACK_START, ATTACK_END, alpha=0.10, color='red', label='Attack window')


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    sz = os.path.getsize(path) // 1024
    print(f"  Saved: {name} ({sz} KB)")


# ============================================================
# Figure 1: Corrected Theta Sweep
# ============================================================
def figure_theta_sweep():
    """Corrected theta sweep: monotonic decrease, no inversion."""
    data = load("corrected_exp_theta_sweep.json")
    ext = load("exp_theta_extended.json")
    if not data:
        print("  [SKIP] corrected_exp_theta_sweep.json not found")
        return

    results = data.get("results", {})
    if ext:
        results.update(ext.get("results", {}))

    # Collect data points
    thetas_fix, poatt_fix = [], []
    thetas_dyn, poatt_dyn = [], []
    converged_fix = []

    all_thetas = [0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.030]
    for theta in all_thetas:
        fk = f"th{theta}_fixed"
        dk = f"th{theta}_dynamic"
        if fk in results:
            thetas_fix.append(theta)
            poatt_fix.append(results[fk]["sustained_poatt"])
            cv = results[fk].get("baseline_cv_pct", 0)
            converged_fix.append(cv < 2.0)
        if dk in results:
            thetas_dyn.append(theta)
            poatt_dyn.append(results[dk]["sustained_poatt"])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Plot fixed
    tf = np.array(thetas_fix)
    pf = np.array(poatt_fix)
    cf = np.array(converged_fix)

    # Converged points
    if cf.any():
        ax.plot(tf[cf], pf[cf], 'o-', color=COLORS['fixed'], label='Fixed trust', zorder=3)
    # Non-converged points
    if (~cf).any():
        ax.plot(tf[~cf], pf[~cf], 'x', color=COLORS['fixed'], markersize=12,
                markeredgewidth=2, label='Fixed (not converged)', zorder=3)

    # Plot dynamic
    ax.plot(thetas_dyn, poatt_dyn, 's-', color=COLORS['dynamic'], label='Dynamic trust', zorder=3)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No effect')
    ax.set_xlabel(r'Logit parameter $\theta$')
    ax.set_ylabel('Sustained PoAtt')
    ax.set_title(r'Corrected $\theta$ Sensitivity (own baseline)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, "corrected_theta_sweep.png")


# ============================================================
# Figure 2: Corrected BR Sweep
# ============================================================
def figure_br_sweep():
    """Corrected BR sweep: dynamic stays above 1.0."""
    data = load("corrected_exp_br_sweep.json")
    if not data:
        print("  [SKIP] corrected_exp_br_sweep.json not found")
        return

    results = data.get("results", {})
    deltas = [0, 100, 200, 300, 400, 500]

    fix_vals = []
    dyn_vals = []
    for d in deltas:
        fk = f"d{d}_fixed"
        dk = f"d{d}_dynamic"
        fix_vals.append(results.get(fk, {}).get("sustained_poatt", None))
        dyn_vals.append(results.get(dk, {}).get("sustained_poatt", None))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(deltas, fix_vals, 'o-', color=COLORS['fixed'], label='Fixed trust')
    ax.plot(deltas, dyn_vals, 's-', color=COLORS['dynamic'], label='Dynamic trust')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No effect')
    ax.set_xlabel(r'Indifference band $\delta$ (s)')
    ax.set_ylabel('Sustained PoAtt')
    ax.set_title(r'Corrected BR Sensitivity (own baseline)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, "corrected_br_sweep.png")


# ============================================================
# Figure 3: Composition Dual (Fixed vs Dynamic)
# ============================================================
def figure_composition_dual():
    """Dual composition: Fixed (increasing) vs Dynamic (decreasing)."""
    fix_data = load("exp_comp_fixed.json")
    dyn_data = load("corrected_exp_composition.json")

    if not fix_data and not dyn_data:
        print("  [SKIP] No composition data found")
        return

    cavs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    cav_pct = [c * 100 for c in cavs]

    fix_vals = []
    dyn_vals = []
    for cav in cavs:
        label = f"cav{cav}"
        if fix_data and label in fix_data.get("results", {}):
            fix_vals.append(fix_data["results"][label]["sustained_poatt"])
        else:
            fix_vals.append(None)
        if dyn_data and label in dyn_data.get("results", {}):
            dyn_vals.append(dyn_data["results"][label]["sustained_poatt"])
        else:
            dyn_vals.append(None)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if any(v is not None for v in fix_vals):
        valid_x = [x for x, v in zip(cav_pct, fix_vals) if v is not None]
        valid_y = [v for v in fix_vals if v is not None]
        ax.plot(valid_x, valid_y, 'o-', color=COLORS['fixed'], label='Fixed trust')
    if any(v is not None for v in dyn_vals):
        valid_x = [x for x, v in zip(cav_pct, dyn_vals) if v is not None]
        valid_y = [v for v in dyn_vals if v is not None]
        ax.plot(valid_x, valid_y, 's-', color=COLORS['dynamic'], label='Dynamic trust')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('CAV Penetration (%)')
    ax.set_ylabel('Sustained PoAtt')
    ax.set_title('Composition Vulnerability: Fixed vs Dynamic Trust')
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, "corrected_composition.png")


# ============================================================
# Figure 4: BC Comparison (path-based vs topological)
# ============================================================
def figure_bc_comparison():
    """Scatter of path BC vs topo BC + bar chart of PoAtt."""
    data = load("exp_topo_bc.json")
    if not data:
        print("  [SKIP] exp_topo_bc.json not found")
        return

    path_bc = np.array(data["path_bc_vector"])
    topo_bc = np.array(data["topo_bc_vector"])
    path_top10 = np.array(data["path_bc_top10"]) - 1  # back to 0-indexed
    topo_top10 = np.array(data["topo_bc_top10"]) - 1

    results = data["results"]

    # Load path-based reference from corrected signature
    sig = load("corrected_exp_signature.json")
    path_fix = sig["scenarios"]["bc_defl_fixed_03"]["sustained_poatt"] if sig else None
    path_dyn = sig["scenarios"]["bc_defl_dynamic_03"]["sustained_poatt"] if sig else None
    topo_fix = results.get("topo_bc_fixed", {}).get("sustained_poatt")
    topo_dyn = results.get("topo_bc_dynamic", {}).get("sustained_poatt")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Scatter plot
    ax1.scatter(path_bc, topo_bc, alpha=0.5, color='#7f7f7f', s=30, label='All links')
    ax1.scatter(path_bc[path_top10], topo_bc[path_top10], color=COLORS['fixed'],
                s=80, marker='^', zorder=3, label='Path-BC top-10')
    ax1.scatter(path_bc[topo_top10], topo_bc[topo_top10], color='#1f77b4',
                s=80, marker='v', zorder=3, label='Topo-BC top-10')
    ax1.set_xlabel('Path-based BC (normalized)')
    ax1.set_ylabel('Topological BC (normalized)')
    ax1.set_title(f'BC Correlation (r={data["pearson_r"]:.3f})')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # (b) Bar chart
    methods = ['PathBC\nFixed', 'PathBC\nDynamic', 'TopoBC\nFixed', 'TopoBC\nDynamic']
    vals = [path_fix or 0, path_dyn or 0, topo_fix or 0, topo_dyn or 0]
    colors = [COLORS['fixed'], COLORS['dynamic'], '#c44e52', '#55a868']
    bars = ax2.bar(methods, vals, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Sustained PoAtt')
    ax2.set_title('BC Target Method Comparison')
    for bar, val in zip(bars, vals):
        if val:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    savefig(fig, "exp_bc_comparison.png")


# ============================================================
# Figure 5: Extended Theta (full range with dual effect check)
# ============================================================
def figure_theta_extended():
    """Full theta range [0.002-0.030] with convergence markers."""
    data = load("corrected_exp_theta_sweep.json")
    ext = load("exp_theta_extended.json")
    if not data:
        print("  [SKIP] No theta data found")
        return

    results = data.get("results", {})
    if ext:
        results.update(ext.get("results", {}))

    all_thetas = [0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.030]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Fixed PoAtt
    thetas, poatts, conv = [], [], []
    for theta in all_thetas:
        fk = f"th{theta}_fixed"
        if fk in results:
            thetas.append(theta)
            poatts.append(results[fk]["sustained_poatt"])
            cv = results[fk].get("baseline_cv_pct", 0)
            conv.append(cv < 2.0)

    thetas = np.array(thetas)
    poatts = np.array(poatts)
    conv = np.array(conv)

    if conv.any():
        ax1.plot(thetas[conv], poatts[conv], 'o-', color=COLORS['fixed'], label='Converged')
    if (~conv).any():
        ax1.plot(thetas[~conv], poatts[~conv], 'x', color=COLORS['fixed'],
                 markersize=12, markeredgewidth=2, label='Not converged')
        # Connect all with dashed line
        ax1.plot(thetas, poatts, '--', color=COLORS['fixed'], alpha=0.3)

    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel(r'Logit parameter $\theta$')
    ax1.set_ylabel('Sustained PoAtt (Fixed)')
    ax1.set_title(r'Fixed Trust: $\theta$ Extended Range')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Check for upturn
    for i in range(1, len(poatts)):
        if poatts[i] > poatts[i-1]:
            ax1.annotate(f'Upturn at\n$\\theta$={thetas[i]:.3f}',
                         xy=(thetas[i], poatts[i]), xytext=(thetas[i]+0.003, poatts[i]+0.005),
                         arrowprops=dict(arrowstyle='->', color='black'),
                         fontsize=12, ha='left')
            break

    # Right: Own baseline vs theta
    baselines = []
    for theta in all_thetas:
        fk = f"th{theta}_fixed"
        if fk in results:
            baselines.append(results[fk].get("own_baseline_converged", 0))
        else:
            baselines.append(0)

    ax2.plot(all_thetas[:len(baselines)], baselines, 'o-', color='#1f77b4')
    ax2.set_xlabel(r'Logit parameter $\theta$')
    ax2.set_ylabel('Own Baseline TSTT')
    ax2.set_title(r'Baseline TSTT vs $\theta$')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, "corrected_theta_extended.png")


# ============================================================
# Figure 6: Corrected Theory Validation (4-panel)
# ============================================================
def figure_theory_validation():
    """4-panel theory validation with corrected PoAtt values."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Theta attenuation
    ax = axes[0, 0]
    data = load("corrected_exp_theta_sweep.json")
    ext = load("exp_theta_extended.json")
    if data:
        results = data.get("results", {})
        if ext:
            results.update(ext.get("results", {}))
        all_thetas = [0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.030]
        fix_th, fix_p = [], []
        dyn_th, dyn_p = [], []
        for theta in all_thetas:
            fk = f"th{theta}_fixed"
            dk = f"th{theta}_dynamic"
            if fk in results:
                fix_th.append(theta)
                fix_p.append(results[fk]["sustained_poatt"])
            if dk in results:
                dyn_th.append(theta)
                dyn_p.append(results[dk]["sustained_poatt"])
        ax.plot(fix_th, fix_p, 'o-', color=COLORS['fixed'], label='Fixed')
        ax.plot(dyn_th, dyn_p, 's-', color=COLORS['dynamic'], label='Dynamic')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel('Sustained PoAtt')
        ax.set_title(r'(a) $\theta$ Attenuation')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    # (b) Recovery power-law (from trust sensitivity)
    ax = axes[0, 1]
    ts_data = load("corrected_exp_trust_sensitivity.json")
    if ts_data:
        results = ts_data.get("results", {})
        ratios = [1, 3, 5, 7, 10, 15]
        recoveries = []
        for r in ratios:
            rk = f"r{r}"
            if rk in results and "trust_recovery" in results[rk]:
                recoveries.append(results[rk]["trust_recovery"])
            else:
                recoveries.append(None)
        valid = [(r, rec) for r, rec in zip(ratios, recoveries) if rec is not None and rec > 0]
        if len(valid) >= 2:
            lr = np.array([np.log(r) for r, _ in valid])
            lrec = np.array([np.log(rec) for _, rec in valid])
            slope, intercept = np.polyfit(lr, lrec, 1)
            ax.scatter([r for r, _ in valid], [rec for _, rec in valid],
                       color='#1f77b4', s=80, zorder=3)
            r_fit = np.linspace(min(ratios), max(ratios), 50)
            rec_fit = np.exp(intercept) * r_fit**slope
            ax.plot(r_fit, rec_fit, '--', color='#1f77b4', alpha=0.7,
                    label=f'slope={slope:.3f}')
            ax.set_xlabel(r'$w_f / w_s$ ratio')
            ax.set_ylabel('Recovery time (days)')
            ax.set_title(f'(b) Recovery Power-Law (slope={slope:.3f})')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

    # (c) IS redundancy
    ax = axes[1, 0]
    is_data = load("corrected_exp_is_sweep.json")
    if is_data:
        results = is_data.get("results", {})
        alphas = [0, 1, 2, 3]
        fix_p, dyn_p = [], []
        for a in alphas:
            fk = f"a{a}_fixed"
            dk = f"a{a}_dynamic"
            fix_p.append(results.get(fk, {}).get("sustained_poatt", None))
            dyn_p.append(results.get(dk, {}).get("sustained_poatt", None))
        ax.plot(alphas, fix_p, 'o-', color=COLORS['fixed'], label='Fixed')
        ax.plot(alphas, dyn_p, 's-', color=COLORS['dynamic'], label='Dynamic')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(r'IS $\alpha$')
        ax.set_ylabel('Sustained PoAtt')
        ax.set_title(r'(c) IS Redundancy')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    # (d) Composition chi(pi) with dual effect
    ax = axes[1, 1]
    fix_comp = load("exp_comp_fixed.json")
    dyn_comp = load("corrected_exp_composition.json")
    if fix_comp or dyn_comp:
        cavs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        cav_pct = [c * 100 for c in cavs]
        if fix_comp:
            fix_p = [fix_comp["results"].get(f"cav{c}", {}).get("sustained_poatt") for c in cavs]
            valid = [(x, y) for x, y in zip(cav_pct, fix_p) if y is not None]
            if valid:
                ax.plot([x for x, _ in valid], [y for _, y in valid],
                        'o-', color=COLORS['fixed'], label='Fixed')
        if dyn_comp:
            dyn_p = [dyn_comp["results"].get(f"cav{c}", {}).get("sustained_poatt") for c in cavs]
            valid = [(x, y) for x, y in zip(cav_pct, dyn_p) if y is not None]
            if valid:
                ax.plot([x for x, _ in valid], [y for _, y in valid],
                        's-', color=COLORS['dynamic'], label='Dynamic')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('CAV Penetration (%)')
        ax.set_ylabel('Sustained PoAtt')
        ax.set_title(r'(d) Composition $\chi(\pi)$: Dual Effect')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, "corrected_theory_validation.png")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("STEP 4: GENERATE CORRECTED FIGURES")
    print("=" * 70)

    figures = [
        ("Corrected Theta Sweep", figure_theta_sweep),
        ("Corrected BR Sweep", figure_br_sweep),
        ("Composition Dual (Fixed vs Dynamic)", figure_composition_dual),
        ("BC Comparison (Path vs Topo)", figure_bc_comparison),
        ("Theta Extended Range", figure_theta_extended),
        ("Corrected Theory Validation", figure_theory_validation),
    ]

    generated = 0
    for i, (name, func) in enumerate(figures, 1):
        print(f"\n[{i}/{len(figures)}] {name}")
        try:
            func()
            generated += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\nGenerated {generated}/{len(figures)} figures")
    print("\n=== STEP 4 COMPLETE ===")


if __name__ == "__main__":
    main()
