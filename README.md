# DTDRGA — Day-to-Day Traffic Dynamics under Route-Guidance Attacks

Coupled day-to-day traffic dynamics and trust evolution framework for studying
network resilience under route-guidance misinformation.

**Paper**: Ka, E. & Ukkusuri, S.V. (2026). *Day-to-Day Traffic Network Modeling
under Route-Guidance Misinformation: Endogenous Trust and Resilience in CAV
Environments*. Under review (IEEE Trans. on Intelligent Transportation Systems).

## Quick Start

```bash
pip install numpy scipy matplotlib networkx

# 1. Validate convergence against DUE reference
python experiments/validate_due_convergence.py --sf-only     # ~7 min
python experiments/validate_due_convergence.py               # ~45 min (+ Anaheim)

# 2. Run all experiments
python experiments/run_full_experiment.py                     # ~8-12 hours

# 3. Generate all paper figures
python experiments/generate_all_figures.py
```

## Validation

The DTD relative gap ε(d) = ||h(d)−h(d−1)||²/||h(d−1)||² is compared against
the DUE fixed-point solver (Han et al. 2019). Both converge to ~10⁻⁴ order.

## Project Structure

```
src/                  Core library (DNL, DTD, Trust, Attack, Metrics)
experiments/          Simulation scripts and figure generators
data/sioux_falls/     Sioux Falls network (24 nodes, 76 links, 6180 paths)
data/anaheim/         Anaheim network (416 nodes, 914 links, 30719 paths)
data/due_reference/   DUE solver solutions for validation
```

## License

MIT


## Reproduction

### Environment setup

```bash
conda env create -f environment.yml
conda activate dtd-rga
```

### Reproduce all paper figures (8–14 hours)

```bash
python experiments/run_full_experiment.py
python experiments/run_gamma07_supplement.py
python experiments/run_epsilon_sweep.py
python experiments/eps_postprocess.py
python visualization/generate_figures_legacy.py
python visualization/generate_composition_recovery_composite.py
```

### Use bundled results without re-simulating (5 minutes)

The repository ships pre-computed JSON outputs in `results/`, so figure
regeneration alone takes only a few minutes:

```bash
python visualization/generate_figures_legacy.py
python visualization/generate_composition_recovery_composite.py
python experiments/eps_postprocess.py
```

The seven figures used in the manuscript will appear in
`results/figures_legacy/` and `results/epsilon_sensitivity/`.

### Reproducibility verification

This release has been verified end-to-end by a clean-install rerun completed
on 2026-05-02. With `numpy=1.26.4` (pinned via `environment.yml`), the
reproduction commands above regenerate every figure-input JSON to bit-exact
equality with the bundled outputs (224 092 scalar values compared, max
absolute difference 0.0). All numerical claims in the IEEE TITS manuscript
(PoAtt values, TIA percentages, recovery slope, R^2, composition coefficients,
and figure content) are reproducible from the bundled `results/*.json` files
using the figure scripts in `visualization/`.

The regenerated PNG figures contain plotted content identical to the
manuscript figures. Pixel-level rasterization may differ slightly across
machines because matplotlib font and line rendering depends on the user's
freetype and Pillow versions; this does not affect the underlying data or
any numerical conclusion.

### Coverage of bundled results

The 199 JSON files at `results/` and the 25 files at
`results/epsilon_sensitivity/` cover the full set of experiments from the
original study design. The seven figures in the IEEE TITS manuscript (six
main + one supplementary) draw on a subset of these. The remaining JSON
files include sensitivity sweeps over n_target, bounded rationality (delta),
information-sharing exponent (alpha), and route-choice sensitivity (theta),
as well as gamma=0.3 trust-asymmetry variants. They are kept for
reproducibility of supplementary analyses and follow-up research; they are
not referenced from the manuscript body.

## Data attribution

The Sioux Falls and Anaheim network data in `data/` were downloaded from the
DrKeHan/DTA repository accompanying:

> Han, K., Eve, G., Friesz, T. L. (2019). Computing dynamic user equilibria on
> large-scale networks with software implementation. *Networks and Spatial
> Economics*, 19(3), 869–902. DOI: [10.1007/s11067-018-9433-y](https://doi.org/10.1007/s11067-018-9433-y).

Source repository: <https://github.com/DrKeHan/DTA>

If you use these network data in your own work, please cite Han et al. (2019).
See [`data/README.md`](data/README.md) for full attribution and usage notice.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{ka2026dtd,
  title   = {Day-to-Day Traffic Network Modeling under Route-Guidance
             Misinformation: Endogenous Trust and Resilience in CAV
             Environments},
  author  = {Ka, Eunhan and Ukkusuri, Satish V.},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
  note    = {to appear}
}
```

Update this entry with volume, issue, pages, and DOI when the paper is
published.
