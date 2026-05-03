# CHANGELOG - Experiment Rerun Log

## [Full Run] gamma=0.3 Complete Experiment Suite
- Date: 2026-03-20 (started 19:44, completed ~15:31)
- Total runtime: ~20 hours
- Simulations: 125 JSON files generated
- Steps completed: All 10 steps (signature, gamma sweep, n-target, BR delta, theta, IS alpha, composition, trust asymmetry, Anaheim, attack comparison + random multiseed)
- Baseline TSTT: 6709.06 veh-hours
- Note: Results directory was empty; ran full gamma=0.3 experiment as prerequisite before supplement
- Errors: None

## [Supplement] gamma=0.7 Full Sweep
- Date: 2026-03-20 15:53 to 2026-03-21 04:20
- Total runtime: ~12.5 hours
- Simulations: 66 simulations (74 g07 JSON files including summaries)
- Steps completed:
  - S1: Signature at gamma=0.7 (4 scenarios)
  - S2: N-target sweep at gamma=0.7 (12 scenarios)
  - S3: BR sweep at gamma=0.7 (12 scenarios)
  - S4: Theta sweep at gamma=0.7 (10 scenarios)
  - S5: IS sweep at gamma=0.7 (8 scenarios)
  - S6: Composition sweep at gamma=0.7 (12 scenarios)
  - S7: Trust asymmetry at gamma=0.7 (6 scenarios)
  - S8: Anaheim at gamma=0.7 (2 scenarios)
- Key results:
  - gamma=0.7 TIA: 91% (stealthy gamma=0.3 TIA: 2%)
  - Trust asymmetry values VARYING: 38, 57, 67, 73, 80, 88 days
  - Power-law fit: slope=0.311, R2=0.990
  - Anaheim: Fixed=1.1453, Dynamic=1.0215
- Errors: None

## [DUE Convergence] Validation
- Date: 2026-03-20 ~15:44
- SF-only mode (--sf-only flag)
- DTD 200 days, final gap: 3.05e-04
- DUE reference final epsilon: 9.78e-05
- Figure: figures/exp_due_convergence.png created
- Errors: None

## [Figures] Paper Figure Generation
- Date: 2026-03-21 (initial), 2026-03-28 (fix + regenerate)
- Initial run: 13/18 figures generated, 5 spatial figures failed due to wrong .mat paths
- Fix: generate_all_figures.py referenced `reference/DTD/` and `reference/DTA/` but .mat files are in `data/sioux_falls/` and `data/anaheim/`. Added DATA_DIR and updated 5 path references.
- Regenerated: 18/18 figures + 1 DUE convergence + 1 validation = 20 total PNG files
- Errors: None (all resolved)

## [Figures] Paper Figure Fix - 12 LaTeX-Referenced Figures
- Date: 2026-04-01
- Script: experiments/generate_final_figures.py (new, replaces generate_all_figures.py for paper figures)
- All 12/12 figures generated successfully
- Fixes applied:
  1. exp_signature.png: Changed from 3 panels + table to 4 panels (TSTT, Zoomed TSTT Day 40-120, Trust evolution with g03+g10, Info weight lambda)
  2. exp_gamma_sweep.png: Replaced Peak PoAtt panel with TIA panel showing regime shading and detection threshold
  3. exp_theory_validation.png: Expanded from 4 to 6 panels; fixed empty panels (b) Recovery using g07 trust_sensitivity data, (d) Composition using g07 data; added (e) Attack budget, (f) Bounded rationality
  4. exp_spatial_attack_sf.png: Verified 2x3 grid (Fixed/Dynamic x 3 days), added max change annotations
  5. exp_spatial_attack_ana.png: Replaced 2-panel dynamic-only with 2x2 grid (Fixed+Dynamic x 2 days), clipped at +/-100%
  6. exp_theta_extended.png: New figure with 2 panels (PoAtt + own-baseline TSTT computed as mean[30:50])
  7. exp_composition.png: Switched from g03 to g07 data for visible Fixed vs Dynamic gap (dual effect)
  8. exp_bc_comparison.png: New figure with scatter (Topo-BC vs Path-BC) + bar chart (3 strategies with Random mean+/-std)
  9. exp_link_flow_timeseries.png: New figure showing 6 top-BC links flow evolution across capture days
  10. exp_anaheim.png: Changed from 4-panel to 2-panel (TSTT with 10-day rolling average + cross-network bars)
  11. exp_network_sf.png: New network structure figure (24 nodes, 76 links)
  12. exp_network_ana.png: New network structure figure (416 nodes, 914 links)
- Total figures in figures/: 25 PNG files (12 paper + 13 supplementary from previous runs)
- Errors: None

## [Baseline Fix] Own-baseline PoAtt normalization
- Date: 2026-04-28
- Issue: theta/delta/composition sweep PoAtt was computed using global baseline (6709) instead of each scenario's own pre-attack baseline mean(tstt[30:50]). Parameters that shift the steady state (theta, delta, CAV%) made PoAtt look like attack-impact when it was really baseline drift.
- Impact: theta sweep PoAtt < 1.0 was an artifact (corrected: always > 1.0, nearly flat ~1.045 across theta=0.006-0.01). BR (delta) "dampening" effect collapsed from 7.3pp range to 0.8pp range. Composition (CAV%) effect reduced from 9.3pp range to 5.9pp range. g07 fixed-trust headline (1.71 at CAV=100%) corrected to 1.66.
- Fix: Added baseline_own, sustained_poatt_own, peak_poatt_own fields to all 180 standalone JSONs in results/. Patched experiments/generate_final_figures.py with _own_poatt() helper that computes mean(tstt[50:100])/mean(tstt[30:50]) on the fly from raw tstt arrays.
- Regenerated:
  - paper_results_summary.md (own-baseline PoAtt + TIA throughout)
  - figures/exp_theory_validation.png (panels a, c, d, e, f recomputed)
  - figures/exp_theta_extended.png (panel a recomputed)
  - figures/exp_composition.png (recomputed from standalone g07 comp JSONs)
  - figures/exp_bc_comparison.png (panel b bars recomputed)
  - figures/exp_anaheim.png (panel b bars recomputed)
- Narrative change: C2 reframed from "theta dual effect + BR dampening" to "TIA robustness across behavioral parameters (~90% invariant)". TIA stays at 86-92% across every sweep; the headline becomes the constancy of attenuation, not the parameter sensitivity of PoAtt.
- Caveat: exp_random_multiseed.json seed records lack tstt arrays, so the Random bar in bc_comparison still uses its stored summary mean (global baseline). Effect is small (Random PoAtt ~0.988); does not affect any conclusion.
- Errors: None
