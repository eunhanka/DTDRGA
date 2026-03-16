# Day-to-Day Traffic Dynamics under Route-Guidance Misinformation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and data for the paper:

> **E. Ka and S. V. Ukkusuri**, "Day-to-Day Traffic Dynamics under Route-Guidance
> Misinformation: Trust Evolution, Behavioral Heterogeneity, and Network Resilience,"
> *IEEE Transactions on Intelligent Transportation Systems*, 2026. (Under review)

## Overview

This repository provides a coupled day-to-day (DTD) traffic assignment and trust
evolution simulator for studying network resilience under adversarial route-guidance
misinformation. The framework models trust as an endogenous state variable via a
Beta reputation mechanism, supporting multi-class behavioral heterogeneity and
LWR kinematic wave network loading.

## Key Features

- **Coupled DTD-Trust dynamics**: Beta-distributed trust evolving with traffic flows
- **Multi-class heterogeneity**: CAV, App-dependent, and Experience-based driver classes
- **Dual attack scenarios**: Recommendation-layer compromise (Scenario I) and data-plane Sybil poisoning (Scenario II)
- **LWR network loading**: Newell's simplified N-curve propagation
- **Theoretical analysis**: Equilibrium existence, stability bounds, recovery dynamics (C1–C4)

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

## Quick Start

```bash
# Clone repository
git clone https://github.com/EunhanKa/DTDRGA.git
cd DTDRGA

# Create conda environment
conda env create -f environment.yml
conda activate dtd

# Run all experiments
python experiments/run_all.py

# Or generate figures from pre-computed results
python figures/generate_all.py
```

## Networks

- **Sioux Falls**: 24 nodes, 76 links, 528 OD pairs (primary)
- **Anaheim**: 416 nodes, 914 links, 1,406 OD pairs (validation)

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

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
