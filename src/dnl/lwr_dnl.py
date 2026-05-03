"""LWR Newell kinematic-wave Dynamic Network Loading.

Port of DYNAMIC_NETWORK_LOADING.m (Han, Eve, Friesz 2018).

Uses the simplified Newell kinematic-wave model with a triangular
fundamental diagram:
  - Free-flow speed: v_f = length / FFT
  - Backward wave speed: w = v_f / 3
  - Jam density: k_jam = 4 * capacity * FFT_mod / length_mod

State variables are cumulative vehicle counts (Nup, Ndn) and flow rates
(Qin, Qout) on every link (physical + virtual source/sink links).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dnl.dnl import BaseDNL
from network.network import NetworkData, SimulationParams


class LWRDNL(BaseDNL):
    """LWR Newell kinematic-wave DNL.

    Parameters
    ----------
    verbose : bool
        Print progress messages during simulation.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public interface (BaseDNL)
    # ------------------------------------------------------------------

    def compute_costs(
        self,
        path_departures: np.ndarray,
        network: NetworkData,
        params: SimulationParams,
    ) -> np.ndarray:
        """Run LWR DNL and return path travel times [n_paths, n_steps]."""
        dt = params.dt
        nt = params.n_steps

        # Pad path_departures to nt columns if needed
        n_paths, nt_dep = path_departures.shape
        if nt < nt_dep:
            nt = nt_dep
        pd = np.zeros((n_paths, nt), dtype=np.float64)
        pd[:, :nt_dep] = path_departures

        # Preprocess link parameters
        fft_mod, length_mod, N_jam, tnk, tnw = self._preprocess(network, dt)

        n_links = network.n_links
        n_sources = network.n_sources
        n_sinks = network.n_sinks
        n_total_up = n_links + n_sources + n_sinks
        n_total_dn = n_links + n_sources

        # Count total path-links for Qin_pathLinks
        num_total_path_links = 0
        for node_junc in network.path_links_in:
            for row in node_junc:
                for arr in row:
                    if arr.size > 0:
                        num_total_path_links = max(
                            num_total_path_links, int(arr.max()) + 1
                        )
        for node_junc in network.path_links_out:
            for row in node_junc:
                for arr in row:
                    if arr.size > 0:
                        num_total_path_links = max(
                            num_total_path_links, int(arr.max()) + 1
                        )
        # Also check pathSourceLinkIdx
        if network.path_source_link_idx is not None:
            num_total_path_links = max(
                num_total_path_links,
                int(network.path_source_link_idx.max()) + 1,
            )

        # Allocate state arrays
        Nup = np.zeros((n_total_up, nt + 1), dtype=np.float64)
        Ndn = np.zeros((n_total_dn, nt + 1), dtype=np.float64)
        Qin = np.zeros((n_total_up, nt), dtype=np.float64)
        Qout = np.zeros((n_total_dn, nt), dtype=np.float64)
        Qin_pathLinks = np.zeros((num_total_path_links, nt), dtype=np.float64)
        N_source = np.zeros(n_sources, dtype=np.float64)

        # Index arrays (0-indexed)
        link_idx = network.link_index
        source_idx = network.source_index
        sink_idx = network.sink_index

        # Source departure rates
        source_dep = np.zeros((n_sources, nt), dtype=np.float64)
        for n in range(n_sources):
            snode = network.source_nodes[n]
            # Sum path departure rates for paths originating at this source
            mask = network.path_source_node == snode
            source_dep[n, :] = pd[mask, :].sum(axis=0)

        Qin[source_idx, :] = source_dep
        Qin_pathLinks[network.path_source_link_idx, :] = pd

        # Capacity arrays for combined indices
        capacity = np.full(n_total_up, np.inf, dtype=np.float64)
        capacity[link_idx] = network.link_capacity

        link_cap = network.link_capacity.copy()
        eps = 1e-9

        # ----------------------------------------------------------
        # Main time-stepping loop
        # ----------------------------------------------------------
        for tn in range(nt):
            # --- Link demand ------------------------------------------
            D_link = np.zeros(n_links, dtype=np.float64)
            tkappa = tn - tnk  # t - FFT/dt for each link
            valid_k = np.where(tkappa >= 0)[0]
            if valid_k.size > 0:
                nup_vals = Nup[link_idx[valid_k], tkappa[valid_k]]
                ndn_vals = Ndn[link_idx[valid_k], tn]
                cond = nup_vals - ndn_vals > eps
                D_link[valid_k[cond]] = link_cap[valid_k[cond]]
                D_link[valid_k[~cond]] = Qin[
                    link_idx[valid_k[~cond]], tkappa[valid_k[~cond]]
                ]

            # --- Link supply ------------------------------------------
            S_link = link_cap.copy()
            tomega = tn - tnw  # t - 3*FFT/dt
            valid_w = np.where(tomega >= 0)[0]
            if valid_w.size > 0:
                nup_w = Nup[link_idx[valid_w], tn]
                ndn_w = Ndn[link_idx[valid_w], tomega[valid_w]]
                scond = nup_w - ndn_w - N_jam[valid_w] < -eps
                # Where supply condition is NOT met, supply = Qout at tomega
                not_scond = ~scond
                S_link[valid_w[not_scond]] = Qout[
                    link_idx[valid_w[not_scond]], tomega[valid_w[not_scond]]
                ]

            # --- Source demand ----------------------------------------
            D_source = np.zeros(n_sources, dtype=np.float64)
            is_queue = N_source > eps
            D_source[is_queue] = np.inf
            Q_source = Qin[source_idx, tn]
            D_source[~is_queue] = Q_source[~is_queue]

            # --- Assemble demand/supply for combined array ------------
            demand = np.zeros(n_total_up, dtype=np.float64)
            demand[link_idx] = D_link
            demand[source_idx] = D_source

            supply = np.full(n_total_up, np.inf, dtype=np.float64)
            supply[link_idx] = S_link

            # --- Junction model ---------------------------------------
            self._junction_model(
                tn, network, Nup, Ndn, Qin, Qout, Qin_pathLinks,
                demand, supply, capacity, link_idx, source_idx, sink_idx,
            )

            # --- Update cumulative counts -----------------------------
            Nup[:, tn + 1] = Nup[:, tn] + Qin[:, tn] * dt
            Ndn[:, tn + 1] = Ndn[:, tn] + Qout[:, tn] * dt

            # --- Update source queues ---------------------------------
            N_source = np.maximum(
                0.0,
                N_source + dt * (Qin[source_idx, tn] - Qout[source_idx, tn]),
            )

            if self.verbose and (tn + 1) % 20 == 0:
                pct = (tn + 1) / nt * 100
                print(f"  LWR DNL: {pct:5.1f}% complete")

        # Remove the extra column (keep nt columns)
        Nup = Nup[:, :nt]
        Ndn = Ndn[:, :nt]

        # --- Path travel times ----------------------------------------
        delay = self._compute_path_travel_times(
            Nup, Ndn, network, fft_mod, dt, nt, link_idx, source_idx,
        )

        return delay

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(
        network: NetworkData, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute modified FFT, jam density, and wave time-step indices.

        Returns (fft_mod, length_mod, N_jam, tnk, tnw).
        """
        fft = network.link_fft
        length = network.link_length

        fft_mod = np.round(fft / dt) * dt
        fft_mod[fft_mod == 0] = dt  # minimum one time step

        length_mod = fft_mod * length / fft
        N_jam = 4.0 * network.link_capacity * fft_mod

        tnk = np.round(fft_mod / dt).astype(np.intp)
        tnw = np.round(3.0 * fft_mod / dt).astype(np.intp)

        return fft_mod, length_mod, N_jam, tnk, tnw

    # ------------------------------------------------------------------
    # Junction model
    # ------------------------------------------------------------------

    @staticmethod
    def _junction_model(
        tn: int,
        network: NetworkData,
        Nup: np.ndarray,
        Ndn: np.ndarray,
        Qin: np.ndarray,
        Qout: np.ndarray,
        Qin_pathLinks: np.ndarray,
        demand: np.ndarray,
        supply: np.ndarray,
        capacity: np.ndarray,
        link_idx: np.ndarray,
        source_idx: np.ndarray,
        sink_idx: np.ndarray,
    ) -> None:
        """Resolve flows at all junctions (nodes) for time step tn.

        Modifies Qin, Qout, Qin_pathLinks in place.
        """
        for i in range(network.n_nodes):
            n_lin = network.num_links_in[i]
            n_lout = network.num_links_out[i]
            Lin = network.links_in[i]   # 0-indexed combined indices
            Lout = network.links_out[i]

            # --- Compute turning fractions gamma and split ratios alpha
            gamma = [[None] * n_lout for _ in range(n_lin)]
            alpha = np.zeros((n_lin, n_lout), dtype=np.float64)

            for ik in range(n_lin):
                Lik = Lin[ik]
                qin_hist = Qin[Lik, :tn + 1]
                # Replicate MATLAB tau logic exactly:
                #   qin = Qin(Lik, 1:tn);
                #   tau = find(qin(Nup(Lik,1:tn) <= Ndn(Lik,tn)), 1, 'last');
                # MATLAB find() returns index INTO the compressed array,
                # which is then used as a direct time index for Qin(Lik, tau).
                cond = Nup[Lik, :tn + 1] <= Ndn[Lik, tn]
                compressed = qin_hist[cond]
                nz = np.flatnonzero(compressed)
                if nz.size > 0:
                    tau = int(nz[-1])
                else:
                    tau = 0

                qi = Qin[Lik, tau]

                for jk in range(n_lout):
                    pli = network.path_links_in[i][ik][jk]
                    if pli.size == 0 or qi <= 1e-10:
                        gamma[ik][jk] = np.zeros(max(pli.size, 1))
                    else:
                        qijr = Qin_pathLinks[pli, tau]
                        gamma[ik][jk] = np.maximum(qijr / qi, 0.0)
                        alpha[ik, jk] = gamma[ik][jk].sum()

            # --- Effective supply and outflow
            eta = network.signal_priorities[i]

            # effectiveSupply = min(capacity(Lin), supply(Lout)' ./ alpha)
            # For each incoming link ik, find min across outgoing links
            eff_supply = np.full(n_lin, np.inf, dtype=np.float64)
            for ik in range(n_lin):
                eff_supply[ik] = capacity[Lin[ik]]
                for jk in range(n_lout):
                    if alpha[ik, jk] > 1e-15:
                        eff_supply[ik] = min(
                            eff_supply[ik],
                            supply[Lout[jk]] / alpha[ik, jk],
                        )

            # eta * eff_supply: guard against 0 * inf = NaN
            eta_es = eta * eff_supply
            eta_es = np.where(np.isfinite(eta_es), eta_es,
                              np.where(eta > 0, eff_supply, 0.0))
            qout = np.minimum(demand[Lin], eta_es)
            # Replace any residual NaN with 0
            np.nan_to_num(qout, copy=False, nan=0.0, posinf=0.0)

            # Qin(Lout) = alpha' * qout  (matrix multiply)
            Qin[Lout, tn] = alpha.T @ qout
            Qout[Lin, tn] = qout

            # Split flow back into individual paths
            for ik in range(n_lin):
                qi_out = qout[ik]
                if qi_out == 0.0:
                    continue
                for jk in range(n_lout):
                    plo = network.path_links_out[i][ik][jk]
                    if plo.size > 0:
                        Qin_pathLinks[plo, tn] = gamma[ik][jk] * qi_out

    # ------------------------------------------------------------------
    # Path travel times
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_path_travel_times(
        Nup: np.ndarray,
        Ndn: np.ndarray,
        network: NetworkData,
        fft_mod: np.ndarray,
        dt: float,
        nt: int,
        link_idx: np.ndarray,
        source_idx: np.ndarray,
    ) -> np.ndarray:
        """Compute path travel times from cumulative counts.

        For each link, builds a timeFunction[a, t_in] = exit time step,
        then chains across path links.

        Returns [n_paths, nt] travel times in seconds.
        """
        n_dn = Ndn.shape[0]

        # Build FFT_mod for combined array (links + sources)
        FFT_combined = np.zeros(n_dn, dtype=np.float64)
        FFT_combined[link_idx] = fft_mod
        # Sources have zero FFT

        fft_steps_all = np.round(FFT_combined / dt).astype(np.intp)
        # Physical links have FFT_mod >= dt, so fft_steps >= 1.
        # Source virtual links have FFT = 0, so fft_steps = 0 (immediate pass-through),
        # matching the MATLAB reference where sources add zero traversal time.

        # timeFunction[a, t_in] = exit time step index
        # Exact replication of MATLAB:
        #   [~,ltt] = min(abs(Ndn(a,t_in:nt) - Nup(a,t_in)));
        #   if ltt <= fft: timeFunction(a,t_in) = t_in + fft;
        #   else:          timeFunction(a,t_in) = t_in + ltt - 1;
        timeFunction = np.zeros((n_dn, nt), dtype=np.intp)

        for a in range(n_dn):
            fft_s = fft_steps_all[a]
            nup_a = Nup[a, :nt]
            ndn_a = Ndn[a, :nt]

            for t_in in range(nt):
                # Subarray from t_in to end
                sub = ndn_a[t_in:nt]
                if sub.size == 0:
                    timeFunction[a, t_in] = t_in + fft_s
                    continue
                abs_diff = np.abs(sub - nup_a[t_in])
                ltt = np.argmin(abs_diff)  # 0-indexed position in subarray
                # ltt=0 means exit at t_in itself
                if ltt < fft_s:
                    timeFunction[a, t_in] = t_in + fft_s
                else:
                    timeFunction[a, t_in] = t_in + ltt

        # Chain across path links to get path travel times
        path_tt = np.zeros((network.n_paths, nt), dtype=np.float64)

        for r in range(network.n_paths):
            # Build link sequence: source virtual link + physical links
            source_vlink = network.path_source_index[r]
            n_plinks = network.path_link_count[r]
            phys_links = network.path_link_matrix[r, :n_plinks]
            # Convert physical link indices to combined indices
            link_seq = [source_vlink]
            for li in range(n_plinks):
                link_seq.append(link_idx[phys_links[li]])

            for tn in range(nt):
                t_in = tn
                incomplete = False
                for a in link_seq:
                    if t_in >= nt:
                        incomplete = True
                        break
                    t_in = timeFunction[a, t_in]
                    if t_in >= nt:
                        incomplete = True
                        break

                if incomplete:
                    tt_val = (t_in - tn) * dt
                    path_tt[r, tn:] = tt_val
                    break
                else:
                    path_tt[r, tn] = (t_in - tn) * dt

        return path_tt