# Accuracy-Tolerance Sensitivity Analysis

## Parameter grid

The sweep uses Sioux Falls under Scenario I (route-level BC-deflation), topo-BC targeting with N_att = 10, a 200-day horizon, attack window days 51 to 100, and seed 42. The accuracy tolerance epsilon takes values 0.05, 0.10, and 0.15 hours (3, 6, and 9 minutes). The attack intensity gamma takes values 0.1, 0.3, 0.5, 0.7, 0.9, 1.0. The grid was selected as the coarse fallback in Step 2.5 because the preferred and minimum grids exceeded the 3 hour compute cap under measured per-simulation runtime.

## Reproduction gate

Reproduction values match the existing default-epsilon (0.1 h = 360 s) JSONs.

| gamma | fixed PoAtt (meas / exp) | dynamic PoAtt (meas / exp) | TIA (meas / exp) |
|-------|--------------------------|------------------------------|-------------------|
| 0.3   | 1.0618 / 1.062 | 1.0604 / 1.060 | 2.3% / 2% |
| 0.7   | 1.3155 / 1.316 | 1.0283 / 1.028 | 91.0% / 91% |
| 1.0   | 1.6534 / 1.653 | 1.0559 / 1.056 | not specified |

All measured values lie within 0.5 percentage points of the expected values, so the reproduction gate passes.

## Runtime

Mean per-simulation runtime measured from existing default-epsilon JSONs is 486 seconds. The new sweep ran 18 fresh simulations in 5439 seconds (90.7 minutes), including 6 day-51 error-capture runs that double as the eps = 0.10 dynamic row. The 6 fixed-trust JSONs and the eps = 0.10 dynamic row were reused, so the total reported wall-clock is the new compute only.

## Main numerical findings

### Empirical and predicted thresholds

| epsilon (h) | epsilon (min) | gamma_hat_error | gamma_hat_TIA | predicted (aggregate) |
|---|---|---|---|---|
| 0.050 | 3.0 | 0.500 | 0.300 | 0.336 |
| 0.100 | 6.0 | 0.700 | 0.500 | 0.672 |
| 0.150 | 9.0 | nan   | 0.700 | 1.007 |

The day-51 flow-weighted guidance error scales linearly with gamma: e(d=51) = 53.6, 160.8, 268.0, 375.2, 482.4, 536.0 seconds for gamma = 0.1, 0.3, 0.5, 0.7, 0.9, 1.0. Linearity gives D = 536 seconds, derived from the smallest gamma. The empirical gamma_hat_error is the smallest grid value at which the day-51 error exceeds epsilon, so it is rounded up to the next grid point relative to the predicted threshold. This explains the systematic empirical-minus-predicted gap of one grid step.

For epsilon = 0.15 h, the maximum day-51 error in the sweep grid (gamma = 1.0, error = 536.0 s) lies just below 540 s. The empirical gamma_hat_error is therefore not defined within this grid, while the predicted threshold is 1.007. The TIA-based threshold gamma_hat_TIA = 0.7 indicates that attenuation does emerge at gamma = 0.7, because errors accumulate over the 50-day attack window even when day-51 error alone is below epsilon.

### Linearity of gamma_hat versus epsilon

A linear fit of the two valid empirical gamma_hat_error values gives slope 4.000 per hour, intercept 0.300, R squared 1.000. With only two valid points the fit is degenerate, so the high R squared is not informative. The analytical prediction gamma_hat = epsilon / D is a line through the origin with slope 1/D = 6.72 per hour. The empirical slope is below the predicted slope because the empirical thresholds are coarse-grid-rounded upward by one gamma step.

### Two-regime persistence

The two-regime structure is preserved across all tested epsilon values. For each epsilon, dynamic-trust attack-window mean PoAtt remains within a few percent of one for sweep gammas below the threshold, and stays substantially below the fixed-trust curve for sweep gammas above it. Smaller epsilon shifts the regime boundary toward smaller gamma; larger epsilon shifts it toward larger gamma. The qualitative shape, low-PoAtt stealthy regime followed by a transition to high-TIA attenuation, is identical in all three epsilon settings.

Post-threshold TIA at gamma = 0.7:

| epsilon (h) | TIA at gamma=0.7 |
|---|---|
| 0.05 | 0.910 |
| 0.10 | 0.910 |
| 0.15 | 0.746 |

For epsilon = 0.05 and 0.10 the post-threshold attenuation at gamma = 0.7 is 91 percent, matching the default-epsilon result in the main paper. For epsilon = 0.15 the attenuation is 75 percent, lower because gamma = 0.7 sits near the regime boundary rather than well inside the high-TIA regime.

### Recovery and hidden window

At gamma = 0.7, trust recovery time and hidden vulnerability window are 72 days for epsilon = 0.05 and 0.10, indicating that trust does not return to 95 percent of its pre-attack mean within the 100 post-attack days for either case (the value 72 reflects the day on which trust crosses 95 percent of pre-attack mean). For epsilon = 0.15 both quantities drop to about 40 days, because the attack at gamma = 0.7 induces a smaller trust drop when the tolerance is large. TSTT recovery time is at most one day in all cases, so the hidden vulnerability window is dominated by trust-recovery slowness.

## Deviations and warnings

Trust dynamics are evaluated with the threshold (binary accurate / inaccurate) model. The smooth-trust mode (eta) is not exercised. Hidden-window-days uses the existing trust recovery criterion (95 percent of pre-attack mean) and the TSTT 5 percent criterion. The grid is coarse, so all empirical gamma_hat_error values are quantized to the nearest sweep gamma; the analytical prediction (using D estimated from the smallest stealthy-regime gamma) provides a finer estimate. The eps = 0.15 empirical gamma_hat_error is undefined inside the grid because day-51 error never exceeds 540 s for gamma in [0.1, 1.0]; the predicted value of 1.007 lies just outside the grid.
