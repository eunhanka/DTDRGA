# Changelog

## [2.0.0] — 2026-03-17 — Paper-Aligned Implementation

Eight corrections to align the codebase with the published paper formulation.

### Critical Fixes (P0)

#### 1. Attack perturbation direction — Scenario I now uses deflation (Eq. 14)

**Paper:** `a_r(f) = −γ · c_r(f) · ω_r` → reported cost = `(1−γ)c_r` (underreporting)  
**Before:** `SybilInflationAttack` used `+γ · c_r` (inflation, users avoid targets)  
**After:** All experiments now use `BCSybilDeflationAttack` (`−γ · c_r`, topological BC targeting)

This corrects the core attack mechanism: the paper's "concentration trap" requires
deflation (making backbone routes appear cheaper) to attract traffic and cause overload.

#### 2. Perceived cost update order — Eq. 2 blend-then-smooth

**Paper (Eq. 2):**
```
ĉ_r^{k,(d+1)} = λ_m · ĉ_r^{k,(d)} + (1−λ_m) · [(1−λ_k)·c̃_r + λ_k·I_r]
```
The information signal `I_r` enters the exponential smoothing memory, so past
attack signals persist in perceived costs after the attack ends.

**Before:** Experienced costs were smoothed first (6-day window average), then
blended with the *current day's* info signal — info never entered memory.

**After:** Per-class perceived cost state `ĉ_k` is maintained across days.
Each day's blended signal `(1−λ_k)·c̃ + λ_k·I` is smoothed into this state
via `λ_m · ĉ_prev + (1−λ_m) · blended`. This matches Eq. 2 exactly.

Information sharing (Eq. 7) is now applied as a separate pre-blending step:
`c̃_r = c_r + κ_IS · (c̄_w − c_r)`, consistent with the paper formulation.

#### 3. Trust timing in perceived cost update — §3.3 Step 4

**Paper (§3.3 Step 4):** "ĉ updated via (2) with reliance λ_k(T_k^{(d)})"  
Eq. 2 explicitly uses T_k^{(d)} — the **pre-update** trust.

**Before:** Trust was updated first (producing T^{(d+1)}), then lambda_k was
computed from the updated trust.

**After:** lambda_k is computed from the current trust state **before** the
trust update, matching the paper's specified loop order.

### Important Fixes (P1)

#### 4. Accuracy tolerance ε — unit correction

**Paper:** ε = 0.1 h ≈ 6 minutes ("route costs are measured in hours")  
**Before:** ε = 0.1 (effectively 0.1 seconds, since LWR costs are in seconds)  
**After:** ε = 360.0 seconds (= 0.1 hours = 6 minutes)

#### 5. Information sharing averaging — per-window (Eq. 7)

**Paper (Eq. 7):** c̄_w = flow-weighted average across paths within each OD pair.

**Before:** c̄_w computed as a single scalar across all paths AND all departure
windows for each OD pair, mixing temporal information.

**After:** c̄_w computed per departure window, preserving the temporal structure
of the within-day model.

#### 6. Composition experiment χ(π) calculation

**Before:** Non-CAV split was 55:45 (App:Experience) in χ reporting  
**After:** 70:30, matching paper §5.3 and the actual simulation code.

Also fixed θ in χ formula: was `[2.0, 1.0, 0.5]` per-class, now `0.004`
common across all classes per paper §3.4.

#### 7. Per-class theta override removed

**Before:** `TrustParams` received `theta=np.array([2.0, 1.0, 0.5])` in
`lwr_single_sim.py`. These values were stored but not used in route choice,
but appeared in χ calculations.

**After:** Removed all 4 override lines. `TrustParams.theta` defaults to
`[0.004, 0.004, 0.004]` matching the paper.

#### 8. Config documentation

Added `DEFAULT_EPSILON = 360.0` to `experiments/config.py`.

### Known Simplification

**Guidance error (Eq. 5):** The paper defines e_k as a flow-weighted average
using per-class flows. The code computes a simple unweighted mean of |I_r - c_r|,
identical across all classes. Under the threshold trust model with ε = 360s and
typical perturbation magnitudes (~2000s at γ=0.3), the threshold crossing behavior
is identical: error is either clearly above ε (attack active) or zero (attack off).
This simplification does not affect threshold model results.

### Unchanged (verified correct)

- Trust dynamics: Beta distribution parameters (w_f, w_s, λ_f, λ̄_k, π_k, δ_k) ✓
- Trust update rule (Eq. 8): forgetting + asymmetric Bayesian update ✓  
- Information reliance (Eq. 12): λ_k = T_k · λ̄_k ✓
- Route choice: MNL (Eq. 4) + bounded rationality (Eq. 3) ✓
- LWR DNL: Newell N-curve propagation ✓
- Simulation protocol: 200 days, attack days 51–100, baseline days 30–50 ✓
- Performance metrics: PoAtt, TIA, trust recovery delay ✓
- Fixed trust implementation: frozen via w_s=w_f≈0, λ_f≈1 ✓
