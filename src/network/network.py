"""
Network data layer for the DTD framework.

Loads MATLAB .mat files from the reference DTD implementation and
exposes them as typed Python dataclasses with 0-indexed arrays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.io


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NetworkData:
    """Static network topology and link properties (from SiouxFalls6180_pp.mat)."""

    # Counts
    n_nodes: int
    n_links: int
    n_paths: int
    n_od: int

    # Link properties – all shape [n_links]
    link_capacity: np.ndarray
    link_fft: np.ndarray
    link_length: np.ndarray
    adjacency_list: np.ndarray       # [n_links, 2] tail-head (0-indexed)

    # Path info
    path_link_matrix: np.ndarray     # [n_paths, max_links] 0-indexed, -1 = pad
    path_link_count: np.ndarray      # [n_paths] number of links per path
    path_source_node: np.ndarray     # [n_paths] 0-indexed source node per path
    path_sink_node: np.ndarray       # [n_paths] 0-indexed sink node per path

    # Node topology – lists of 0-indexed link indices per node
    links_in: list[np.ndarray]       # len = n_nodes
    links_out: list[np.ndarray]      # len = n_nodes
    num_links_in: np.ndarray         # [n_nodes]
    num_links_out: np.ndarray        # [n_nodes]

    # Source / sink
    source_nodes: np.ndarray         # 0-indexed node ids
    sink_nodes: np.ndarray           # 0-indexed node ids
    source_index: np.ndarray         # 0-indexed indices into Nup/Ndn arrays
    sink_index: np.ndarray           # 0-indexed indices into Nup array
    n_sources: int = 0
    n_sinks: int = 0

    # Node coordinates
    node_x: Optional[np.ndarray] = None   # [n_nodes]
    node_y: Optional[np.ndarray] = None   # [n_nodes]

    # Signal priorities per node – list of arrays
    signal_priorities: Optional[list[np.ndarray]] = None

    # DNL junction bookkeeping (path-link turn indices)
    path_links_in: Optional[list] = None    # [n_nodes] of (nIn, nOut) object arrays
    path_links_out: Optional[list] = None   # [n_nodes] of (nIn, nOut) object arrays
    path_source_link_idx: Optional[np.ndarray] = None  # [n_paths] 0-indexed
    path_sink_link_idx: Optional[np.ndarray] = None    # [n_paths] 0-indexed

    # Path struct source/sink virtual-link indices (0-indexed into Nup/Ndn)
    path_source_index: Optional[np.ndarray] = None  # [n_paths]
    path_sink_index: Optional[np.ndarray] = None     # [n_paths]

    # Link index mapping (0-indexed indices into the combined Nup/Ndn arrays)
    link_index: Optional[np.ndarray] = None  # [n_links]


@dataclass
class DemandData:
    """OD demand and path-set information."""

    od_set: np.ndarray               # [n_od, 3]  (origin, dest, ?) 0-indexed nodes
    od_path_set: list[np.ndarray]    # len = n_od, each entry is 0-indexed path indices
    od_demand: np.ndarray            # [n_od] total demand per OD pair
    target_arrival: np.ndarray       # [n_od] target arrival time in seconds


@dataclass
class SimulationParams:
    """Temporal discretisation parameters."""

    dt: float = 180.0                # DNL time step [s]
    departure_window: float = 900.0  # departure time window [s]
    n_steps: int = 100               # number of DNL time steps
    n_windows: int = 20              # number of departure windows
    steps_per_window: int = 5        # time steps per window
    time_horizon: tuple[float, float] = (0.0, 18000.0)

    # Initial path departure rates [n_paths, n_steps]
    initial_path_departures: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _squeeze_scalar(arr: np.ndarray):
    """Extract a Python scalar from a MATLAB (1,1) array."""
    return arr.flat[0]


def _flatten_col(arr: np.ndarray) -> np.ndarray:
    """Flatten a MATLAB column vector (N,1) to 1-D."""
    return np.asarray(arr).flatten()


def _extract_cell_list(cell_array: np.ndarray, to_0index: bool = False) -> list[np.ndarray]:
    """Convert a MATLAB cell column vector {N,1} to a Python list of 1-D arrays.

    Parameters
    ----------
    cell_array : (N, 1) object ndarray from loadmat
    to_0index  : if True, subtract 1 from every element (MATLAB 1-index → 0-index)
    """
    result = []
    for i in range(cell_array.shape[0]):
        inner = np.asarray(cell_array[i, 0]).flatten()
        if to_0index and inner.size > 0:
            inner = inner.astype(np.intp) - 1
        result.append(inner)
    return result


def _extract_path_link_junctions(
    cell_array: np.ndarray, to_0index: bool = False
) -> list[list[list[np.ndarray]]]:
    """Convert pathLinksIn / pathLinksOut.

    Structure: cell{n_nodes, 1}, where each element is an object array
    of shape (nLinksIn_i, nLinksOut_i), and each inner element is an
    index array (or empty).

    Returns
    -------
    list of length n_nodes.  Each element is a list-of-lists (nIn x nOut)
    where each leaf is a 1-D int64 array (0-indexed if requested).
    """
    n_nodes = cell_array.shape[0]
    result: list[list[list[np.ndarray]]] = []
    for n in range(n_nodes):
        mat = cell_array[n, 0]  # (nIn, nOut) object array
        rows: list[list[np.ndarray]] = []
        for i in range(mat.shape[0]):
            cols: list[np.ndarray] = []
            for j in range(mat.shape[1]):
                elem = np.asarray(mat[i, j]).flatten()
                if to_0index and elem.size > 0:
                    elem = elem.astype(np.intp) - 1
                cols.append(elem)
            rows.append(cols)
        result.append(rows)
    return result


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_sioux_falls(
    data_dir: str,
    network_file: str = "SiouxFalls6180_pp.mat",
) -> tuple[NetworkData, DemandData, SimulationParams]:
    """Load all .mat files from *data_dir* and return structured data.

    Parameters
    ----------
    data_dir : path to the directory containing the four .mat files
        (Path_flow_data.mat, OD_info.mat, Network_planning_parameters.mat,
         and the network file).
    network_file : which preprocessed network file to load (default or _68).

    Returns
    -------
    (NetworkData, DemandData, SimulationParams)
    """

    def _path(name: str) -> str:
        return os.path.join(data_dir, name)

    # ------------------------------------------------------------------
    # 1. Path_flow_data.mat  →  dt, pathDepartures
    # ------------------------------------------------------------------
    pf = scipy.io.loadmat(_path("Path_flow_data.mat"), squeeze_me=False)
    dt = float(_squeeze_scalar(pf["dt"]))             # 180
    path_departures = np.asarray(pf["pathDepartures"], dtype=np.float64)  # [P, T]

    # ------------------------------------------------------------------
    # 2. OD_info.mat  →  OD_set, ODpath_set
    # ------------------------------------------------------------------
    od_raw = scipy.io.loadmat(_path("OD_info.mat"), squeeze_me=False)
    od_set_raw = np.asarray(od_raw["OD_set"])          # [528, 3] 1-indexed nodes
    od_set = od_set_raw.astype(np.intp) - 1            # → 0-indexed
    od_path_set = _extract_cell_list(od_raw["ODpath_set"], to_0index=True)
    n_od = od_set.shape[0]

    # ------------------------------------------------------------------
    # 3. Network_planning_parameters.mat  →  OD_demand, T_A
    # ------------------------------------------------------------------
    np_raw = scipy.io.loadmat(
        _path("Network_planning_parameters.mat"), squeeze_me=False
    )
    od_demand = _flatten_col(np_raw["OD_demand"]).astype(np.float64)      # [528]
    # T_A is stored in hours in the .mat; the MATLAB code multiplies by 3600
    target_arrival = _flatten_col(np_raw["T_A"]).astype(np.float64) * 3600.0  # → seconds

    # ------------------------------------------------------------------
    # 4. Network file (SiouxFalls6180_pp.mat)
    # ------------------------------------------------------------------
    nf = scipy.io.loadmat(_path(network_file), squeeze_me=False)

    # -- structs --------------------------------------------------------
    link_s = nf["link"][0, 0]
    node_s = nf["node"][0, 0]
    source_s = nf["source"][0, 0]
    sink_s = nf["sink"][0, 0]
    path_s = nf["path"][0, 0]

    # -- scalar counts --------------------------------------------------
    n_links = int(_squeeze_scalar(link_s["count"]))
    n_nodes = int(_squeeze_scalar(node_s["count"]))
    n_sources = int(_squeeze_scalar(source_s["count"]))
    n_sinks = int(_squeeze_scalar(sink_s["count"]))
    n_paths = int(_squeeze_scalar(path_s["count"]))

    # -- link properties (flatten from (N,1) columns) -------------------
    link_capacity = _flatten_col(link_s["capacity"]).astype(np.float64)
    link_fft = _flatten_col(link_s["FFT"]).astype(np.float64)
    link_length = _flatten_col(link_s["length"]).astype(np.float64)
    link_index = _flatten_col(link_s["index"]).astype(np.intp) - 1

    # adjacency_list from explicit tail/head (0-indexed)
    tail = _flatten_col(link_s["tailNode"]).astype(np.intp) - 1
    head = _flatten_col(link_s["headNode"]).astype(np.intp) - 1
    adjacency_list = np.column_stack([tail, head])

    # -- path properties ------------------------------------------------
    path_link_raw = np.asarray(nf["pathList"])  # [n_paths, max_links] 1-indexed, 0=pad
    path_link_matrix = path_link_raw.astype(np.intp) - 1  # 0-indexed; pad becomes -1
    path_link_count = _flatten_col(path_s["linkCount"]).astype(np.intp)
    path_source_node = _flatten_col(path_s["sourceNode"]).astype(np.intp) - 1
    path_sink_node = _flatten_col(path_s["sinkNode"]).astype(np.intp) - 1
    path_source_index = _flatten_col(path_s["sourceIndex"]).astype(np.intp) - 1
    path_sink_index = _flatten_col(path_s["sinkIndex"]).astype(np.intp) - 1

    # -- node topology --------------------------------------------------
    links_in = _extract_cell_list(nf["linksIn"], to_0index=True)
    links_out = _extract_cell_list(nf["linksOut"], to_0index=True)
    num_links_in = _flatten_col(nf["numLinksIn"]).astype(np.intp)
    num_links_out = _flatten_col(nf["numLinksOut"]).astype(np.intp)

    # -- source / sink --------------------------------------------------
    source_nodes = _flatten_col(source_s["nodes"]).astype(np.intp) - 1
    sink_nodes = _flatten_col(sink_s["nodes"]).astype(np.intp) - 1
    source_index = _flatten_col(source_s["index"]).astype(np.intp) - 1
    sink_index = _flatten_col(sink_s["index"]).astype(np.intp) - 1

    # -- node coordinates -----------------------------------------------
    node_x = _flatten_col(node_s["X"]).astype(np.float64)
    node_y = _flatten_col(node_s["Y"]).astype(np.float64)

    # -- signal priorities (cell array of column vectors) ----------------
    sp_cell = node_s["signalPriorities"]  # (n_nodes, 1) object array
    signal_priorities = [
        _flatten_col(sp_cell[i, 0]).astype(np.float64) for i in range(n_nodes)
    ]

    # -- path-link junction bookkeeping ---------------------------------
    path_links_in = _extract_path_link_junctions(nf["pathLinksIn"], to_0index=True)
    path_links_out = _extract_path_link_junctions(nf["pathLinksOut"], to_0index=True)
    path_source_link_idx = _flatten_col(nf["pathSourceLinkIdx"]).astype(np.intp) - 1
    path_sink_link_idx = _flatten_col(nf["pathSinkLinkIdx"]).astype(np.intp) if "pathSinkLinkIdx" in nf else None
    if path_sink_link_idx is not None:
        path_sink_link_idx = path_sink_link_idx - 1

    # ------------------------------------------------------------------
    # Assemble dataclasses
    # ------------------------------------------------------------------
    time_horizon = (0.0, 5.0 * 3600.0)
    n_steps = int((time_horizon[1] - time_horizon[0]) / dt)
    departure_window = 900.0
    steps_per_window = int(departure_window / dt)
    n_windows = int((time_horizon[1] - time_horizon[0]) / departure_window)

    network = NetworkData(
        n_nodes=n_nodes,
        n_links=n_links,
        n_paths=n_paths,
        n_od=n_od,
        link_capacity=link_capacity,
        link_fft=link_fft,
        link_length=link_length,
        adjacency_list=adjacency_list,
        path_link_matrix=path_link_matrix,
        path_link_count=path_link_count,
        path_source_node=path_source_node,
        path_sink_node=path_sink_node,
        links_in=links_in,
        links_out=links_out,
        num_links_in=num_links_in,
        num_links_out=num_links_out,
        source_nodes=source_nodes,
        sink_nodes=sink_nodes,
        source_index=source_index,
        sink_index=sink_index,
        n_sources=n_sources,
        n_sinks=n_sinks,
        node_x=node_x,
        node_y=node_y,
        signal_priorities=signal_priorities,
        path_links_in=path_links_in,
        path_links_out=path_links_out,
        path_source_link_idx=path_source_link_idx,
        path_sink_link_idx=path_sink_link_idx,
        path_source_index=path_source_index,
        path_sink_index=path_sink_index,
        link_index=link_index,
    )

    demand = DemandData(
        od_set=od_set,
        od_path_set=od_path_set,
        od_demand=od_demand,
        target_arrival=target_arrival,
    )

    params = SimulationParams(
        dt=dt,
        departure_window=departure_window,
        n_steps=n_steps,
        n_windows=n_windows,
        steps_per_window=steps_per_window,
        time_horizon=time_horizon,
        initial_path_departures=path_departures,
    )

    return network, demand, params
