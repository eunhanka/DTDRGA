# DTD Framework Architecture (MATLAB Reference)

Reference: Yu, Y., Han, K., Ochieng, W.Y., 2020. Day-to-Day Dynamic Traffic Assignment
with Imperfect Information, Bounded Rationality and Information Sharing.
*Transportation Research Part C*, 114, 59-83.

DNL Module: Han, K., Eve, G., Friesz, T.L., 2019. Computing Dynamic User Equilibria
on Large-Scale Networks. *Networks and Spatial Economics*, 19(3), 869-902.

---

## 1. Model Variants Overview

| Script | Choice Model | Dep. Time | Route | Extensions |
|---|---|---|---|---|
| `Base_Model_I.m` | Joint MNL | Logit (joint) | Logit (joint) | None |
| `Base_Model_I_BR.m` | Joint MNL + BR | Logit (joint) | Logit (joint) | Bounded rationality (indifference band) |
| `Base_Model_I_IS.m` | Joint MNL + IS | Logit (joint) | Logit (joint) | Information sharing (flow-weighted costs) |
| `Base_Model_II.m` | Sequential NL | MNL (mean cost) | Path-size nested logit | Path-size correction |
| `Base_Model_II_IS.m` | Sequential NL + IS | MNL (mean cost) | Path-size nested logit | Info sharing + path-size |

---

## 2. Main Loop Structure (Common to All Models)

All `Base_Model_*.m` scripts share the same high-level loop:

```
LOAD data files (Path_flow_data.mat, OD_info.mat, Network_planning_parameters.mat)
SET parameters (N, lambda, theta, Num_days, DT, factor, ...)
INITIALIZE arrays (aggPath_flow, aggE, PC)

FOR T = 1 to Num_days:
    1. DYNAMIC NETWORK LOADING
       - Call DYNAMIC_NETWORK_LOADING(pathDepartures, nt, dt, networkFile)
       - Returns: delay [n_paths x nt] matrix of path travel times
       - Link disruption scenario: days 51-100 use '_68.mat' (degraded network)

    2. ARRIVAL PENALTY & TRAVEL COST
       - Compute arrival_time = time_grid + delay
       - Early penalty: gamma_early * max(0, T_A - arrival_time)
       - Late penalty:  gamma_late  * max(0, arrival_time - T_A)
       - Effective delay: E = delay + AP

    3. AGGREGATE COSTS BY DEPARTURE WINDOW
       - aggE(:,i,T) = mean of E over time steps within window i

    4. PERCEIVED COST (exponential memory model)
       - PC = weighted average of aggE over past N days
       - Weight: lambda^(T-i) for day i, normalized by (1-lambda)/(1-lambda^N)

    5. CHOICE MODEL (varies by model variant)
       - Compute new aggPath_flow(:,:,T+1) from PC

    6. DISAGGREGATE path flows back to fine time steps for next day's DNL
END
```

### Model-Specific Choice Logic

#### Model I (Joint MNL)
```
For each OD pair i:
    P(path j, window k) = exp(-theta * PC(j,k)) / Sum_all(j',k') exp(-theta * PC(j',k'))
    aggPath_flow(j,k) = OD_demand(i) * P(j,k) / DT
```

#### Model I + BR (Bounded Rationality)
```
Adds indifference band delta (=400s):
For each OD pair i:
    Not-switching probability for (j,k):
        aggPath_flow(j,k,T) * exp(-theta*(PC(j,k)-delta)) / (Den - exp(-theta*PC(j,k)) + exp(-theta*(PC(j,k)-delta)))
    Switching-from-others:
        Sum over (jj,kk) != (j,k): aggPath_flow(jj,kk,T) * exp(-theta*PC(j,k)) / (Den - exp(-theta*PC(jj,kk)) + exp(-theta*(PC(jj,kk)-delta)))
```

#### Model I + IS (Information Sharing)
```
Perceived cost uses flow-weighted memory:
    weight(path,window,day_i) = Flow_split(path,window,i)^alpha * lambda^(T-i)
    PC = Sum(weight * aggE) / Sum(weight)
Where:
    Flow_split = aggPath_flow / sum(aggPath_flow) per OD pair
    alpha = 2 (information sharing power)
```

#### Model II (Sequential Nested Logit)
```
Two-stage choice:
1. Departure time: Prob_T(k) = exp(-theta_T * mean(PC(:,k))) / Sum_k' exp(...)
2. Route (path-size nested logit):
   P(j|k) = exp(theta*(-PC(j,k) + beta_PS*log(PS(j)))) / Sum_j' exp(...)
   aggPath_flow(j,k) = OD_demand / DT * Prob_T(k) * P(j|k)

Path-size attribute PS:
    PS(path) = Sum_links_in_path [ link_length/path_length / num_OD_paths_using_link ]
```

#### Model II + IS
```
Combines Model II's sequential choice with Model I+IS's flow-weighted perceived cost.
```

---

## 3. DYNAMIC_NETWORK_LOADING.m Logic

**Signature:** `[delay, Nup, Ndn] = DYNAMIC_NETWORK_LOADING(pathDepartures, nt, dt, processedDataFile)`

### Algorithm (Variational Inequality / LWR-based DNL)

```
LOAD network data from processedDataFile (SiouxFalls6180_pp.mat)

PREPROCESS:
    - Round free-flow times (FFT) to multiples of dt
    - Compute jam density: N_jam = 4 * capacity * FFT_mod
    - tnk = FFT_mod / dt  (free-flow time steps)
    - tnw = 3 * FFT_mod / dt  (shockwave time steps, i.e., backward wave speed = v_f/3)

INITIALIZE:
    - Nup, Ndn = cumulative vehicle counts (upstream/downstream)
    - Qin, Qout = flow rates
    - Departure rates from path departures aggregated by source node

FOR tn = 1 to nt:  (time step loop)

    === LINK MODEL (Newell's simplified kinematic wave) ===
    Demand (sending flow):
        If Nup(a, t-tnk) - Ndn(a, t) > 0:  D = capacity
        Else:  D = Qin(a, t-tnk)

    Supply (receiving flow):
        If Nup(a, t) - Ndn(a, t-tnw) - N_jam < 0:  S = capacity
        Else:  S = Qout(a, t-tnw)

    Source demand:
        If queue exists: D = inf
        Else: D = departure_rate

    === JUNCTION MODEL (node-based flow distribution) ===
    For each node:
        Compute turning fractions gamma from path-disaggregated flows
        Compute effective supply = min(capacity, supply/turning_fraction)
        Outflow = min(demand, priority * effective_supply)
        Distribute flows to outgoing links using turning fractions

    UPDATE cumulative counts:
        Nup(:,tn+1) = Nup(:,tn) + Qin(:,tn) * dt
        Ndn(:,tn+1) = Ndn(:,tn) + Qout(:,tn) * dt
        Update source queues

END time loop

COMPUTE PATH TRAVEL TIMES:
    For each link: build timeFunction(a, t_in) = exit time
        Based on: find t_out where Ndn(a, t_out) >= Nup(a, t_in)
        Enforce minimum = free-flow time
    For each path: chain link travel times
        delay(r, tn) = total path travel time for departure at tn
```

### Key Data Structures in DNL

| Variable | Type | Description |
|---|---|---|
| `link` | struct | `.tailNode`, `.headNode`, `.count`, `.capacity`, `.length`, `.FFT`, `.index` |
| `node` | struct | `.count`, `.X`, `.Y`, `.signalPriorities` |
| `source` | struct | `.nodes`, `.count`, `.index` |
| `sink` | struct | `.nodes`, `.count`, `.index` |
| `path` | struct | `.linkCount`, `.sourceNode`, `.sinkNode`, `.count`, `.sourceIndex`, `.sinkIndex` |
| `pathList` | matrix [6180x19] | Link sequence for each path (zero-padded) |
| `linksIn/linksOut` | cell [24x1] | Incoming/outgoing link indices per node |
| `pathLinksIn/Out` | cell [24x1] | Path-link indices at each node junction |
| `Nup` | matrix | Cumulative upstream vehicle count per link per time step |
| `Ndn` | matrix | Cumulative downstream vehicle count per link per time step |

---

## 4. .mat File Variable Structures

### Path_flow_data.mat
| Variable | Shape | Dtype | Description |
|---|---|---|---|
| `dt` | (1,1) | uint8 | Simulation time step = 180 seconds |
| `pathDepartures` | (6180, 100) | float64 | Initial path departure rates [veh/s], 6180 paths x 100 time steps |

### OD_info.mat
| Variable | Shape | Dtype | Description |
|---|---|---|---|
| `OD_set` | (528, 3) | uint8 | OD pair definitions [origin, destination, ?] |
| `ODpath_set` | (528, 1) | cell/object | Each cell contains array of path indices for that OD pair |

### Network_planning_parameters.mat
| Variable | Shape | Dtype | Description |
|---|---|---|---|
| `OD_demand` | (1, 528) | float64 | Total demand per OD pair [vehicles], all = 56.82 |
| `T_A` | (1, 528) | float64 | Target arrival time per OD pair [hours], range [1.0, 4.49] |

### SiouxFalls6180_pp.mat (Network Data)
| Variable | Shape | Description |
|---|---|---|
| `adjacencyList` | (76, 2) | Edge list [tail, head] |
| `adjacencyMatrix` | (24, 24) | Node adjacency matrix |
| `link` | struct | Link properties (76 links): capacity, length, FFT, etc. |
| `node` | struct | Node properties (24 nodes): coordinates, signal priorities |
| `source` | struct | Source node indices and counts |
| `sink` | struct | Sink node indices and counts |
| `path` | struct | Path metadata (6180 paths) |
| `pathList` | (6180, 19) | Link sequence per path (max 19 links, zero-padded) |
| `linksIn` | (24, 1) cell | Incoming links per node |
| `linksOut` | (24, 1) cell | Outgoing links per node |
| `numLinksIn` | (24, 1) | Count of incoming links per node |
| `numLinksOut` | (24, 1) | Count of outgoing links per node |
| `pathLinksIn` | (24, 1) cell | Path-link junctions (incoming) |
| `pathLinksOut` | (24, 1) cell | Path-link junctions (outgoing) |
| `pathSourceLinkIdx` | (1, 6180) | Virtual source link index per path |
| `pathSinkLinkIdx` | (1, 6180) | Virtual sink link index per path |
| `nodeCoordinates` | (24, 2) | (X, Y) positions of nodes |
| `numTotalPathLinks` | (1, 1) | Total path-link entries |

### SiouxFalls6180_pp_68.mat
Same structure as above but with link 68 degraded (disruption scenario).

---

## 5. Helper Functions

### dispstat.m
Console progress display utility. Overwrites previous line for clean progress reporting.
**Python equivalent:** `tqdm` or simple `\r` print.

### flowProfile.m
Visualization: plots cumulative counts (Nup, Ndn) and flows (Qin, Qout) for a selected link/source/sink.
**Python equivalent:** `matplotlib` plotting function.

---

## 6. Variable Naming Map: MATLAB to Python

| MATLAB Variable | Python Name | Type | Description |
|---|---|---|---|
| `pathDepartures` | `path_departures` | `np.ndarray [P, T_steps]` | Path departure rates |
| `n_paths` | `n_paths` | `int` | Number of paths (6180) |
| `NumOD` | `n_od` | `int` | Number of OD pairs (528) |
| `OD_set` | `od_set` | `np.ndarray [528, 3]` | OD pair definitions |
| `ODpath_set` | `od_path_set` | `list[np.ndarray]` | Path indices per OD pair |
| `OD_demand` | `od_demand` | `np.ndarray [528]` | Demand per OD pair |
| `T_A` | `target_arrival` | `np.ndarray [528]` | Target arrival times (seconds) |
| `time_horizon` | `time_horizon` | `tuple(float, float)` | (0, 18000) seconds |
| `dt` | `dt` | `float` | Time step (180s) |
| `DT` | `departure_window` | `float` | Departure window (900s) |
| `TSPW` | `steps_per_window` | `int` | Time steps per window (5) |
| `NT` | `n_windows` | `int` | Number of departure windows (20) |
| `nt` | `n_steps` | `int` | Number of DNL time steps (100) |
| `Num_days` | `n_days` | `int` | Total simulation days |
| `N` | `memory_days` | `int` | Memory window (3 or 6) |
| `lambda` | `memory_weight` | `float` | Exponential decay (0.7) |
| `theta` | `theta` | `float` | Logit dispersion parameter |
| `theta_T` | `theta_dep` | `float` | Departure time logit param (Model II) |
| `factor` | `demand_factor` | `float` | Demand scaling factor |
| `gamma_early` | `gamma_early` | `float` | Early arrival penalty coeff (0.8) |
| `gamma_late` | `gamma_late` | `float` | Late arrival penalty coeff (1.8) |
| `delta` | `indifference_band` | `float` | BR band in seconds (400) |
| `alpha` | `is_alpha` | `float` | Info sharing power (2) |
| `beta_PS` | `beta_path_size` | `float` | Path-size coefficient (400) |
| `aggPath_flow` | `agg_path_flow` | `np.ndarray [P, W, D]` | Aggregated path flows |
| `aggE` | `agg_cost` | `np.ndarray [P, W, D]` | Aggregated effective delay |
| `PC` | `perceived_cost` | `np.ndarray [P, W, D]` | Perceived travel cost |
| `E` | `effective_delay` | `np.ndarray [P, T_steps]` | Effective delay (delay + penalty) |
| `AP` | `arrival_penalty` | `np.ndarray [P, T_steps]` | Schedule delay penalty |
| `delay` | `travel_time` | `np.ndarray [P, T_steps]` | Raw path travel times |
| `Flow_split` | `flow_split` | `np.ndarray [P, W, D]` | Flow proportions (IS models) |
| `PS` | `path_size` | `np.ndarray [P]` | Path-size attribute (Model II) |
| `plength` | `path_length` | `np.ndarray [P]` | Physical path length |
| `Nup` | `n_up` | `np.ndarray [L+, T_steps]` | Cumulative upstream count |
| `Ndn` | `n_dn` | `np.ndarray [L+, T_steps]` | Cumulative downstream count |
| `Qin` | `q_in` | `np.ndarray [L+, T_steps]` | Upstream flow rate |
| `Qout` | `q_out` | `np.ndarray [L+, T_steps]` | Downstream flow rate |
| `link` | `network.links` | `dataclass/dict` | Link properties |
| `node` | `network.nodes` | `dataclass/dict` | Node properties |
| `source` | `network.sources` | `dataclass/dict` | Source properties |
| `sink` | `network.sinks` | `dataclass/dict` | Sink properties |
| `path` | `network.paths` | `dataclass/dict` | Path metadata |
| `pathList` | `path_link_matrix` | `np.ndarray [P, max_links]` | Link sequence per path |

### Dimension Legend
- `P` = number of paths (6180)
- `W` = number of departure windows (20)
- `D` = number of simulation days
- `T_steps` = number of DNL time steps (100)
- `L+` = number of links + sources + sinks

---

## 7. Network Summary: Sioux Falls

- **24 nodes**, **76 directed links**, **528 OD pairs** (24x23 - some excluded)
- **6180 paths** (pre-enumerated, max 19 links per path)
- **Time horizon:** 5 hours (0-18000s)
- **Time step (dt):** 180s -> 100 time steps
- **Departure window (DT):** 900s -> 20 windows, 5 steps each
- **Uniform demand:** 56.82 vehicles per OD pair
- **Disruption scenario:** Link 68 degraded on days 51-100
