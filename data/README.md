# Network Data Attribution

The Sioux Falls and Anaheim network datasets in this directory were obtained
directly from the **DrKeHan/DTA** repository, which accompanies the following
publication:

> Han, K., Eve, G., Friesz, T. L. (2019). Computing dynamic user equilibria
> on large-scale networks with software implementation. *Networks and Spatial
> Economics*, 19(3), 869–902.
> DOI: [10.1007/s11067-018-9433-y](https://doi.org/10.1007/s11067-018-9433-y) (open access).

Source repository: <https://github.com/DrKeHan/DTA>

These benchmark instances are used here for the experiments reported in:

> Ka, E., Ukkusuri, S. V. (2026). Day-to-Day Traffic Network Modeling under
> Route-Guidance Misinformation: Endogenous Trust and Resilience in CAV
> Environments. *IEEE Transactions on Intelligent Transportation Systems*.

## Usage notice

If you use these network data in your own research, please cite the original
Han et al. (2019) publication in addition to any citation of the present
work.

## Files

### `sioux_falls/` — Sioux Falls network (24 nodes, 76 links, 528 OD pairs)

| File | Size | Description |
|---|---|---|
| `SiouxFalls6180_pp.mat`        | 246 KB | Pre-processed network with 6 180 enumerated paths |
| `SiouxFalls6180_pp_68.mat`     | 246 KB | Variant with 68-link parameter set |
| `Path_flow_data.mat`           | 733 KB | Path-level flow data |
| `Network_planning_parameters.mat` |   4 KB | Topology and capacity parameters |
| `OD_info.mat`                  |  13 KB | Origin–destination pair definitions |

### `anaheim/` — Anaheim network (416 nodes, 914 links, 1 406 OD pairs)

| File | Size | Description |
|---|---|---|
| `Anaheim30719_pp.mat`          | 3.4 MB | Pre-processed network with 30 719 enumerated paths |
| `Anaheim30k_paths.mat`         | 436 KB | Path enumeration data (30 k paths) |
| `Anaheim_dat.mat`              |  6 KB | Network attributes |
| `DUE_out.mat`                  | 13.2 MB | Reference DUE-solver output (Han et al. 2019 convergence baseline) |
| `Network_planning_parameters.mat` |  13 KB | Topology and capacity parameters |
| `OD_info.mat`                  |  60 KB | Origin–destination pair definitions |

### `due_reference/` — DUE-solver reference solutions for validation

| File | Size | Description |
|---|---|---|
| `DUE_out_sf.mat`               | 3.8 MB | DUE fixed-point solution for Sioux Falls (validates DTD relative-gap convergence ε(d) ~ 10⁻⁴) |
| `DUE_out_ana.mat`              | 13.2 MB | DUE fixed-point solution for Anaheim (same purpose) |

File formats and pre-enumerated path sets follow the conventions of the
source repository.
