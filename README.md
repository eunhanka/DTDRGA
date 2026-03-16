# Day-to-Day Traffic Dynamics under Route-Guidance Misinformation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and data for the paper:

> **E. Ka and S. V. Ukkusuri**, "Day-to-Day Traffic Dynamics under Route-Guidance
> Misinformation: Trust Evolution, Behavioral Heterogeneity, and Network Resilience,"
> *IEEE Transactions on Intelligent Transportation Systems*, 2026.

## Overview

This repository provides a coupled day-to-day (DTD) traffic assignment and trust
evolution simulator for studying network resilience under adversarial route-guidance
misinformation. The framework models trust as an endogenous state variable via a
Beta reputation mechanism, supporting multi-class behavioral heterogeneity and
LWR kinematic wave network loading.

## Key Features

- **Coupled DTD-Trust dynamics**: Beta-distributed trust evolving with traffic flows
- **Multi-class heterogeneity**: CAV, App-dependent, and Experience-based drivers
- **Dual attack scenarios**: Recommendation-layer compromise (Scenario I) and
  data-plane Sybil poisoning (Scenario II)
- **LWR network loading**: Newell's simplified N-curve propagation
- **Theoretical analysis**: Equilibrium existence, stability bounds, recovery dynamics

## Quick Start

```bash
# Clone repository
git clone https://github.com/EunhanKa/DTD-Trust-Resilience.git
cd DTD-Trust-Resilience

# Create conda environment
conda env create -f environment.yml
conda activate dtd

# Run all experiments (~24h on 8-core machine)
python experiments/run_all.py

# Or generate figures from pre-computed results
python figures/generate_all.py
```

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `src/` | Core simulation engine (DTD simulator, LWR DNL, trust dynamics, attack models) |
| `experiments/` | Experiment runners and configurations |
| `data/` | Sioux Falls and Anaheim network data (.mat format) |
| `results/` | Pre-computed JSON results (Scenario I + II) |
| `figures/` | Figure generation scripts |
| `paper/` | LaTeX manuscript |
| `tests/` | Verification tests |

## Source Code (`src/`)

| Module | Description |
|--------|-------------|
| `src/dtd/dtd_trust_simulator.py` | Main DTD-Trust integrated simulator |
| `src/dtd/dtd_simulator.py` | Base DTD simulator (no trust) |
| `src/dnl/lwr_dnl.py` | LWR kinematic wave network loading |
| `src/dnl/dnl.py` | Base DNL interface and utilities |
| `src/trust/trust_dynamics.py` | Beta trust model (threshold + smooth) |
| `src/attack/attack_models.py` | Attack policies (TopoBCRGA, PathBCRGA, Random, Sybil, etc.) |
| `src/network/network.py` | Network data structures and loaders |
| `src/utils/metrics.py` | PoAtt, TIA, trust recovery computation |
| `src/utils/network_loader.py` | Unified network loader |

## Reproducing Paper Results

### From pre-computed results (fast, ~5 min)
```bash
python figures/generate_all.py
# Figures saved to figures/output/
```

### From scratch (full reproduction, ~24h)
```bash
python experiments/run_all.py --output results/
python figures/generate_all.py --input results/
```

### Verify numbers match paper
```bash
python tests/test_audit.py
# Checks all 20 key numerical claims against JSON data
```

## Key Results

| Metric | Value |
|--------|-------|
| Scenario I Fixed PoAtt (gamma=0.3) | 1.046 |
| Trust-Induced Attenuation | 91-94% (gamma in [0.3, 1.0]) |
| Hidden vulnerability window | 73 days |
| CAV dual effect (fixed) | +0.40 pp per 10% CAV |
| CAV dual effect (dynamic) | -0.07 pp per 10% CAV |
| Recovery fit | slope=0.289, R^2=0.989 |

## Networks

- **Sioux Falls**: 24 nodes, 76 links, 528 OD pairs, 6180 paths (primary)
- **Anaheim**: 416 nodes, 914 links, 1406 OD pairs, 30719 paths (validation)

## Citation

```bibtex
@article{ka2026dtd,
  author  = {Ka, Eunhan and Ukkusuri, Satish V.},
  title   = {Day-to-Day Traffic Dynamics under Route-Guidance Misinformation:
             Trust Evolution, Behavioral Heterogeneity, and Network Resilience},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
  note    = {Under review}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
