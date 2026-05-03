# Derived Composite Figure for Compact Supplementary

## Overview

The figure `exp_composition_recovery_main.png` is a derived composite
generated for the compact version of the supplementary manuscript. It
combines two panels that exist individually in other figures:

- Panel (a) CAV composition sweep — same data as
  `exp_composition.png` (Fig S6 in the full supplementary).
- Panel (b) Trust recovery sublinear fit — same data as
  `exp_theory_validation.png` panel (b) (Fig S7 in the full
  supplementary).

This composite is NOT one of the 15 original figures produced by
`generate_figures_legacy.py`. It was created later for manuscript
readability in the compact supplementary version.

## Data sources

Panel (a): per-CAV standalone JSON files at gamma=0.7 in `results/`,
matching the pattern `exp_g07_comp_{cav}_{trust}.json`. Own-baseline
PoAtt is computed via the helper `_own_poatt` in
`generate_figures_legacy.py`.

Panel (b): trust sensitivity sweep JSON, one of (in priority order)
`exp_recovery_powerlaw.json`, `exp_g07_recovery.json`, or
`exp_trust_sensitivity_g07.json`. The script uses the first one that
exists. Recovery days {38, 57, 67, 73, 80, 88} for
wf/ws ratios {1, 3, 5, 7, 10, 15}.

## Reproduction

  conda run -n base_env python visualization/generate_composition_recovery_composite.py

Outputs:
- results/figures_legacy/exp_composition_recovery_main.png
- results/figures_legacy/exp_composition_recovery_main.pdf

## Verified expected values

Panel (a) at gamma=0.7:
- Fixed PoAtt: 1.290 (0% CAV) to 1.665 (100% CAV)
- Dynamic PoAtt: 1.025 (0% CAV) to 1.053 (100% CAV)

Panel (b) at gamma=0.7:
- Power-law slope: 0.311
- Power-law R^2: 0.9898 (rounds to 0.990 at 3 decimals)
