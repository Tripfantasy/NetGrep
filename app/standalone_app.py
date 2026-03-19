import dearpygui.dearpygui as dpg
import networkx as nx
import polars as pl
import numpy as np
import pickle
import sys
import types
import json
import threading   # NEW
import csv
import os

W, H      = 1440, 920
SIDEBAR_W = 380
CANVAS_W  = W - SIDEBAR_W - 4
CANVAS_H  = H

MAX_PKL_EDGES = 999_999_999  

class NetGrep:
    def __init__(self, file_path, pkl_path=None):
        self.clr_bg      = [10, 10, 14]
        self.clr_accent  = [90, 210, 200]
        self.clr_node    = [80, 150, 155, 220]
        self.clr_edge    = [200, 80, 60, 55]
        self.clr_hl_tf   = [255, 210, 0, 255]
        self.clr_hl_tgt  = [255, 135, 0, 210]
        self.clr_hl_edge = [255, 210, 0, 150]

        self.node_scale     = 5.0
        self.edge_thickness = 1.0
        self.user_scale     = 1.0   

        self.view_ox = float(CANVAS_W // 2)
        self.view_oy = float(CANVAS_H // 2)
        self.view_scale = 300.0  

        self.min_weight = 0.0
        self.max_edges  = MAX_PKL_EDGES
        self.pos        = {}
        self._base_pos  = {}   # raw layout positions (never modified by spread)
        self._dirty     = True
        self.prevent_overlap = False
        self._layout_running = False   
        self.hub_separation  = 5.0   
        self.cluster_spread  = 1.0   

        self.csv_path   = file_path
        self.pkl_path   = pkl_path
        self.regulons   = {}
        self.regulons_json = {}
        self.viz_source = "CSV"

        # set of TF node names in PKL graph
        self.pkl_tfs     = set()

        self.highlighted_tf      = None
        self.highlighted_targets = set()

        self._subgraph_mode  = False   # True when viewing a single regulon subgraph
        self._full_G_pkl     = None    # backup of full G_pkl while in subgraph mode

        self.G_csv = self._load_csv(file_path)
        if pkl_path:
            self.load_pkl(pkl_path)
            self.convert_pkl_to_json()
        self.G_pkl = self._build_graph_from_json()

        self.G = self.G_csv
        self.degrees = dict(self.G.degree())
        self._compute_layout("ForceAtlas2")
        self._rebuild_weights()

        self.init_gui()
        self._subnetgrep_gene = ""   # last queried gene

    def _load_csv(self, path):
        try:
            df = pl.read_csv(path, has_header=False, new_columns=["s", "t", "w"])
            df = df.with_columns(pl.col("w").cast(pl.Float64))
            return nx.from_pandas_edgelist(df.to_pandas(), 's', 't', edge_attr='w')
        except:
            return nx.Graph()

    def _build_graph_from_json(self):
        """Build a DiGraph from the JSON-like regulons data."""
        all_edges = []
        self.pkl_tfs = set()
        for tf, data in self.regulons_json.items():
            for gene, weight in data["gene_weights"].items():
                all_edges.append((tf, gene, weight))
            self.pkl_tfs.add(tf)

        all_edges.sort(key=lambda x: x[2], reverse=True)
        # only apply cap if explicitly set below the total
        if self.max_edges < len(all_edges):
            all_edges = all_edges[:self.max_edges]

        G = nx.DiGraph()
        for tf, gene, weight in all_edges:
            G.add_edge(tf, gene, w=weight)
        return G

    def _tf_name(self, reg, fallback):
        # Keep the full name including (+) / (-) suffix
        raw = getattr(reg, 'transcription_factor', None) \
              or getattr(reg, 'name', None) \
              or fallback
        return raw if raw else fallback

    def _gene_weights(self, reg):
        g2w = getattr(reg, 'gene2weight', None)
        if g2w and hasattr(g2w, 'items'):
            return list(g2w.items())
        return [(g, 1.0) for g in self.get_regulon_targets(reg)]

    def _rebuild_weights(self):
        self.weights = [float(d.get('w', 0)) for u, v, d in self.G.edges(data=True)]
        if self.weights:
            self.hist_data, self.bin_edges = np.histogram(self.weights, bins=20)
        else:
            self.hist_data = np.zeros(20)
            self.bin_edges = np.linspace(0, 1, 21)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.min_weight  = 0.0
        self._dirty = True

    def load_pkl(self, path):
        class Regulon:
            def __setstate__(self, state): self.__dict__.update(state)
        for m in ['pkg_resources', 'pkg_resources.extern',
                  'pyscenic', 'pyscenic.genesig', 'ctxcore', 'ctxcore.genesig']:
            if m not in sys.modules:
                sys.modules[m] = types.ModuleType(m)
        sys.modules['ctxcore.genesig'].Regulon  = Regulon
        sys.modules['pyscenic.genesig'].Regulon = Regulon
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, list):
                for reg in data:
                    name = getattr(reg, 'name', None) or reg.__dict__.get('name', str(reg))
                    self.regulons[name] = reg
            elif isinstance(data, dict):
                self.regulons = data
        except Exception as e:
            print(f"[pkl error] {e}")

    def get_regulon_targets(self, reg):
        for attr in ('gene2weight', 'genes', '_gene2weight'):
            v = getattr(reg, attr, None)
            if v is not None:
                return set(v.keys()) if hasattr(v, 'keys') else set(v)
        return set()
    
    def convert_pkl_to_json(self):
        """Convert PKL data to a lightweight JSON-like structure and store it in memory."""
        self.regulons_json = {}
        for name, reg in self.regulons.items():
            tf = self._tf_name(reg, name)
            targets = self.get_regulon_targets(reg)
            gene_weights = {gene: float(weight) for gene, weight in self._gene_weights(reg)}
            self.regulons_json[tf] = {
                "name": name,
                "targets": list(targets),
                "gene_weights": gene_weights
            }
        print("PKL data converted to JSON-like structure and stored in memory.")

    def _enter_subgraph_mode(self):
        """Isolate the highlighted regulon as a subgraph and visualise it alone."""
        name = self.highlighted_tf
        if not name or name not in self.regulons_json:
            return

        targets = set(self.regulons_json[name]["targets"])
        nodes   = {name} | targets

        sub = nx.DiGraph()
        for u, v, d in self.G_pkl.edges(data=True):
            if u in nodes and v in nodes:
                sub.add_edge(u, v, **d)
        for n in nodes:
            if n not in sub:
                sub.add_node(n)

        self._full_G_pkl    = self.G_pkl
        self._subgraph_mode = True
        self.G_pkl          = sub
        self.G              = sub

        if hasattr(self, '_subgraph_btn'):
            dpg.configure_item(self._subgraph_btn, label="Exit Subgraph",
                               callback=self._exit_subgraph_mode)

        # Weight-driven radial layout Collect (target, weight) pairs
        gene_weights = []
        for gene in sorted(targets):
            d = sub.get_edge_data(name, gene)
            w = float(d.get('w', 0)) if d else 0.0
            gene_weights.append((gene, w))

        # higher weight → smaller radius
        weights_arr = np.array([w for _, w in gene_weights], dtype=float)
        w_min, w_max = weights_arr.min(), weights_arr.max()
        w_range = w_max - w_min if w_max > w_min else 1.0

        n_tgt = max(len(targets), 1)
        R_max = max(2.5, n_tgt * 0.20)   
        R_min = R_max * 0.20 # highest-weight targets

        pos = {name: np.array([0.0, 0.0])}

        # sort by weight descending so highest-weight get smaller radii
        gene_weights_sorted = sorted(gene_weights, key=lambda x: x[1], reverse=True)

        for k, (gene, w) in enumerate(gene_weights_sorted):
            # normalised weight: 1.0 = highest weight, 0.0 = lowest
            t      = (w - w_min) / w_range
            radius = R_min + (1.0 - t) * (R_max - R_min)
            # evenly distribute angles 
            angle  = 2 * np.pi * k / n_tgt
            pos[gene] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        self._base_pos = {n: np.array(p, dtype=float) for n, p in pos.items()}
        self.pos       = dict(self._base_pos)
        self._reset_view()
        self._dirty = True

    def _exit_subgraph_mode(self):
        """Return to the full PKL graph."""
        if not self._subgraph_mode or self._full_G_pkl is None:
            return
        self._subgraph_mode = False
        self.G_pkl          = self._full_G_pkl
        self._full_G_pkl    = None
        self.G              = self.G_pkl

        if hasattr(self, '_subgraph_btn'):
            dpg.configure_item(self._subgraph_btn, label="View Subgraph",
                               callback=self._enter_subgraph_mode)

        preset = dpg.get_value(self.layout_combo)
        self._compute_layout(preset)
        self._dirty = True

    def _layout_grn_clusters(self):
        """Spiral layout — TFs on a circle, genes fanned outward."""
        G    = self.G
        tfs  = [n for n in G.nodes() if n in self.pkl_tfs]
        N_tf = len(tfs)
        if N_tf == 0:
            rng = np.random.default_rng(42)
            return {n: rng.uniform(-1, 1, 2) for n in G.nodes()}

        tfs_sorted = sorted(tfs, key=lambda t: G.out_degree(t), reverse=True)
        reordered, lo, hi, toggle = [], 0, len(tfs_sorted) - 1, True
        while lo <= hi:
            if toggle: reordered.append(tfs_sorted[lo]);  lo += 1
            else:      reordered.append(tfs_sorted[hi]);  hi -= 1
            toggle = not toggle

        # hub_separation slider (1–20) scales the ring radius
        R_tf = max(3.0, N_tf * 0.75) * (self.hub_separation / 5.0)
        tf_pos, tf_angle = {}, {}
        for k, tf in enumerate(reordered):
            angle = 2 * np.pi * k / N_tf
            tf_pos[tf]   = np.array([R_tf * np.cos(angle), R_tf * np.sin(angle)])
            tf_angle[tf] = angle

        return self._place_genes(tf_pos, tf_angle, R_tf)

    def _layout_grn_overlap(self):
        """GRN Cluster — TF positions driven by Jaccard MDS on target-set overlap."""
        G    = self.G
        tfs  = [n for n in G.nodes() if n in self.pkl_tfs]
        N_tf = len(tfs)
        if N_tf == 0:
            rng = np.random.default_rng(42)
            return {n: rng.uniform(-1, 1, 2) for n in G.nodes()}

        tf_targets = {tf: set(G.successors(tf)) for tf in tfs}

        D = np.zeros((N_tf, N_tf))
        for i in range(N_tf):
            for j in range(i + 1, N_tf):
                a, b  = tf_targets[tfs[i]], tf_targets[tfs[j]]
                inter = len(a & b)
                union = len(a | b)
                jac   = 1.0 - (inter / union if union > 0 else 0.0)
                D[i, j] = D[j, i] = jac

        n   = N_tf
        D2  = D ** 2
        H   = np.eye(n) - np.ones((n, n)) / n
        B   = -0.5 * H @ D2 @ H
        B   = (B + B.T) / 2
        eigvals, eigvecs = np.linalg.eigh(B)
        idx2   = np.argsort(eigvals)[::-1][:2]
        lam    = np.maximum(eigvals[idx2], 0.0)
        coords = eigvecs[:, idx2] * np.sqrt(lam)[None, :]

        # hub_separation slider (1–20) scales overall spread
        R_tf   = max(3.0, N_tf * 0.75) * (self.hub_separation / 5.0)
        spread = np.std(coords) + 1e-8
        coords = coords / spread * R_tf * 0.85

        # enforce minimum TF–TF distance
        min_tf_dist = R_tf * 0.55
        for _ in range(80):
            moved = False
            for i in range(N_tf):
                for j in range(i + 1, N_tf):
                    d    = coords[i] - coords[j]
                    dist = float(np.linalg.norm(d)) + 1e-12
                    if dist < min_tf_dist:
                        push = (min_tf_dist - dist) / 2 * d / dist
                        coords[i] += push
                        coords[j] -= push
                        moved = True
            if not moved:
                break

        tf_pos   = {tfs[k]: coords[k] for k in range(N_tf)}
        centroid = coords.mean(axis=0)
        tf_angle = {tfs[k]: float(np.arctan2(coords[k][1] - centroid[1],
                                              coords[k][0] - centroid[0]))
                    for k in range(N_tf)}

        return self._place_genes(tf_pos, tf_angle, R_tf)

    def _place_genes(self, tf_pos, tf_angle, R_tf):
        G     = self.G
        tfs   = set(tf_pos.keys())
        genes = [n for n in G.nodes() if n not in self.pkl_tfs]
        rng   = np.random.default_rng(42)

        gene_tfs     = {g: [p for p in G.predecessors(g) if p in tfs] for g in genes}
        unique_genes = {g: gene_tfs[g][0] for g in genes if len(gene_tfs[g]) == 1}
        shared_genes = {g: gene_tfs[g]    for g in genes if len(gene_tfs[g]) >  1}
        orphan_genes = [g for g in genes  if len(gene_tfs[g]) == 0]

        pos = dict(tf_pos)

        for gene, tfl in shared_genes.items():
            centre    = np.mean([tf_pos[t] for t in tfl], axis=0)
            dist      = np.linalg.norm(centre)
            direction = centre / dist if dist > 1e-6 else np.array([1.0, 0.0])
            pos[gene] = direction * (dist + R_tf * 0.18)

        tf_unique = {tf: [] for tf in tfs}
        for gene, tf in unique_genes.items():
            tf_unique[tf].append(gene)

        for tf, gene_list in tf_unique.items():
            if not gene_list:
                continue
            n_g      = len(gene_list)
            base_ang = tf_angle[tf]
            arc      = min(np.pi * 0.85, 0.22 * np.sqrt(n_g))
            r_min    = R_tf * 0.55
            r_max    = R_tf * 1.1 + 0.10 * np.sqrt(n_g)

            def edge_w(g, _tf=tf):
                d = G.get_edge_data(_tf, g)
                return float(d.get('w', 0)) if d else 0.0
            gene_list_sorted = sorted(gene_list, key=edge_w, reverse=True)

            for j, gene in enumerate(gene_list_sorted):
                a = base_ang if n_g == 1 else base_ang - arc/2 + arc * j / (n_g - 1)
                r = r_min + (r_max - r_min) * (j / max(n_g - 1, 1))
                r += rng.uniform(-r_min * 0.1, r_min * 0.1)
                pos[gene] = tf_pos[tf] + np.array([r * np.cos(a), r * np.sin(a)])

        for k, gene in enumerate(orphan_genes):
            angle     = 2 * np.pi * k / max(len(orphan_genes), 1)
            pos[gene] = np.array([np.cos(angle), np.sin(angle)]) * (R_tf * 1.6)

        pos = self._remove_gene_overlap(pos, tfs, min_dist=R_tf * 0.12)
        return pos

    def _remove_gene_overlap(self, pos, tf_set, min_dist=0.55, iterations=40):
        """Push gene nodes apart. Capped at 5000 genes for performance."""
        genes = [n for n in pos if n not in tf_set]
        if len(genes) < 2:
            return pos
        # avoid O(N²) stall on huge graphs (TESTING)
        if len(genes) > 5000:
            return pos

        P = np.array([pos[g] for g in genes], dtype=float)
        for _ in range(iterations):
            diff  = P[:, None] - P[None, :]
            dist  = np.sqrt((diff**2).sum(axis=2) + 1e-12)
            np.fill_diagonal(dist, np.inf)
            close = dist < min_dist
            if not close.any():
                break
            overlap = np.where(close, (min_dist - dist) / 2, 0.0)
            unit    = diff / dist[:, :, None]
            P      += (overlap[:, :, None] * unit).sum(axis=1)

        for i, g in enumerate(genes):
            pos[g] = P[i]
        return pos

    def _layout_forceatlas2(self, iterations=150, kr=0.05, kg=1.0, ks=0.1):
        """FA2 mimic for CSV / undirected graphs only."""
        G     = self.G
        nodes = list(G.nodes())
        N     = len(nodes)
        if N == 0:
            return {}
        idx   = {n: i for i, n in enumerate(nodes)}
        rng   = np.random.default_rng(42)
        P     = rng.uniform(-1, 1, (N, 2)).astype(float)
        deg   = np.array([G.degree(n) + 1 for n in nodes], dtype=float)
        edges = [(idx[u], idx[v]) for u, v in G.edges() if u in idx and v in idx]

        do_full_rep = N <= 2000
        for it in range(iterations):
            t = 1.0 - it / iterations
            F = np.zeros((N, 2))
            if do_full_rep:
                diff    = P[:, None] - P[None, :]
                dist2   = (diff ** 2).sum(axis=2) + 1e-6
                np.fill_diagonal(dist2, np.inf)
                dist    = np.sqrt(dist2)
                rep_mag = kr * deg[:, None] * deg[None, :] / dist2
                F      += (rep_mag[:, :, None] * diff / dist[:, :, None]).sum(axis=1)
            else:
                centre = P.mean(axis=0)
                diff_c = P - centre
                d2     = (diff_c ** 2).sum(axis=1, keepdims=True) + 1e-6
                F     += kr * 50 * diff_c / d2
            if edges:
                ei = np.array([e[0] for e in edges])
                ej = np.array([e[1] for e in edges])
                d  = P[ej] - P[ei]
                np.add.at(F, ei,  d * ks)
                np.add.at(F, ej, -d * ks)
            F -= kg * P * (deg[:, None] / deg.mean())
            mag = np.linalg.norm(F, axis=1, keepdims=True)
            F   = F / np.maximum(mag / (0.05 * (1 + t * 2)), 1.0)
            P  += F
            self._layout_progress = (it + 1) / iterations

        return {n: P[i] for i, n in enumerate(nodes)}

    def _apply_cluster_spread(self):
        """Re-run the current PKL layout with updated hub_separation, then refit."""
        if not self._base_pos:
            return
        if not self.pkl_tfs or self.viz_source != "PKL":
            self.pos = dict(self._base_pos)
            self.center_to_fit()
            return
        # recompute layout so spread is baked into positions, then refit
        preset = dpg.get_value(self.layout_combo) if hasattr(self, 'layout_combo') else "GRN Cluster"
        self._recompute_pkl_layout(preset)

    def _recompute_pkl_layout(self, preset):
        """Run PKL layout in background thread with a status indicator."""
        if self._layout_running:
            return
        self._layout_running  = True
        self._layout_progress = 0.0
        self._start_layout_ui()

        def _worker():
            if preset == "GRN Cluster":
                raw = self._layout_grn_overlap()
            else:
                raw = self._layout_grn_clusters()
            self._layout_progress = 1.0
            self._base_pos = {n: np.array(p, dtype=float) for n, p in raw.items()}
            self.pos = dict(self._base_pos)
            self._reset_view()
            self._dirty          = True
            self._layout_running = False

        threading.Thread(target=_worker, daemon=True).start()

    def _compute_layout(self, preset):
        G = self.G
        if G.number_of_nodes() == 0:
            self.pos = {}; self._base_pos = {}; self._dirty = True; return
        N  = G.number_of_nodes()
        Gu = G.to_undirected() if G.is_directed() else G

        # PKL layouts always go through _recompute_pkl_layout (shows progress)
        if preset in ("GRN Cluster", "Spiral"):
            self._recompute_pkl_layout(preset)
            return

        # CSV instant layouts
        instant = {"Circular", "Spectral", "Random"}
        if preset in instant:
            raw = self._run_instant_layout(preset, Gu, N)
            self._base_pos = {n: np.array(p, dtype=float) for n, p in raw.items()}
            self.pos = dict(self._base_pos)
            self._reset_view()
            self._dirty = True
            self._finish_layout_ui()
            return

        # CSV slow layouts → background thread
        if self._layout_running:
            return
        self._layout_running  = True
        self._layout_progress = 0.0
        self._start_layout_ui()

        def _worker():
            if preset == "ForceAtlas2":
                iters = max(80, min(150, 6000 // max(N, 1)))
                raw = self._layout_forceatlas2(iterations=iters)
            elif preset == "Spring":
                raw = nx.spring_layout(Gu, k=2.5 / max(1, N**0.5), seed=42,
                                       iterations=max(30, min(80, 3000 // max(N, 1))))
                self._layout_progress = 1.0
            else:
                raw = {n: np.zeros(2) for n in G.nodes()}
            self._base_pos = {n: np.array(p, dtype=float) for n, p in raw.items()}
            self.pos = dict(self._base_pos)
            self._reset_view()
            self._dirty          = True
            self._layout_running = False

        threading.Thread(target=_worker, daemon=True).start()

    def _run_instant_layout(self, preset, Gu, N):
        G   = self.G
        rng = np.random.default_rng(42)
        if preset == "Circular":
            return nx.circular_layout(Gu)
        elif preset == "Spectral":
            try:    return nx.spectral_layout(Gu)
            except: return nx.spring_layout(Gu, k=2.5/max(1,N**0.5), seed=42, iterations=50)
        elif preset == "Hierarchical":
            return self._layout_hierarchical()
        elif preset == "Radial TF":
            r = self._layout_radial_tf()
            return r if r else nx.circular_layout(Gu)
        else:
            return {n: rng.uniform(-1, 1, 2) for n in G.nodes()}

    def _start_layout_ui(self):
        if hasattr(self, 'layout_progress_bar'):
            dpg.configure_item(self.layout_progress_bar, show=True, default_value=0.0)
            dpg.configure_item(self.layout_apply_btn, enabled=False)

    def _finish_layout_ui(self):
        if hasattr(self, 'layout_progress_bar'):
            dpg.configure_item(self.layout_progress_bar, show=False, default_value=0.0)
            dpg.configure_item(self.layout_apply_btn, enabled=True)

    def _apply_cluster_spread(self):
        """
        Post-layout spread: scale TF positions from centroid by cluster_spread,
        move gene clouds rigidly. Then refit view.
        """
        if not self._base_pos:
            return
        if not self.pkl_tfs or self.viz_source != "PKL":
            self.pos = dict(self._base_pos)
            self.center_to_fit()
            return

        s  = self.cluster_spread
        G  = self.G
        bp = self._base_pos

        tf_base = {n: bp[n] for n in bp if n in self.pkl_tfs}
        if not tf_base:
            self.pos = dict(bp)
            self.center_to_fit()
            return

        centroid = np.mean(list(tf_base.values()), axis=0)
        new_tf   = {tf: centroid + (bp[tf] - centroid) * s for tf in tf_base}

        new_pos = dict(new_tf)
        for node in bp:
            if node in self.pkl_tfs:
                continue
            preds = [p for p in G.predecessors(node) if p in self.pkl_tfs and p in bp]
            if preds:
                primary       = max(preds, key=lambda p: G.out_degree(p))
                offset        = bp[node] - bp[primary]
                new_pos[node] = new_tf[primary] + offset
            else:
                # scale stray nodes 
                new_pos[node] = centroid + (bp[node] - centroid) * s

        self.pos = new_pos
        self.center_to_fit()

    def render_frame(self):
        # update progress bar from main thread 
        if self._layout_running and hasattr(self, 'layout_progress_bar'):
            dpg.set_value(self.layout_progress_bar, self._layout_progress)
        elif not self._layout_running and hasattr(self, 'layout_progress_bar'):
            if dpg.get_item_configuration(self.layout_progress_bar).get('show', False):
                self._finish_layout_ui()

        if not self._dirty:
            return

        dpg.delete_item("graph_layer", children_only=True)
        if not self.pos:
            return

        ox  = self.view_ox
        oy  = self.view_oy
        sc  = self.view_scale
        pad = self.node_scale + 8
        cw  = CANVAS_W + pad
        ch  = CANVAS_H + pad

        nodes    = list(self.pos.keys())
        node_idx = {n: i for i, n in enumerate(nodes)}
        P        = np.array([self.pos[n] for n in nodes])
        Sx       = P[:, 0] * sc + ox
        Sy       = P[:, 1] * sc + oy
        vis      = (Sx > -pad) & (Sx < cw) & (Sy > -pad) & (Sy < ch)

        min_w        = self.min_weight
        hl_tf        = self.highlighted_tf
        hl_tgt       = self.highlighted_targets
        is_pkl       = self.viz_source == "PKL"
        pkl_tfs      = self.pkl_tfs
        et           = self.edge_thickness
        active_nodes = set()

        n_edges   = self.G.number_of_edges()
        n_nodes   = self.G.number_of_nodes()

        # LOD thresholds (TESTING)
        EDGE_LOD_SKIP  = 15_000   
        NODE_LABEL_LOD = 500      
        large_graph    = n_edges > EDGE_LOD_SKIP

        sng_active = getattr(self, '_subnetgrep_active', False)
        sng_roles  = getattr(self, '_subnetgrep_roles', {})

        # role → fill colour
        SNG_COLORS = {
            "gene":        [50,  200,  80, 255],   # green
            "direct_tf":   [220,  60,  60, 255],   # red
            "adjacent":    [60,  130, 220, 255],   # blue
            "indirect_tf": [160,  80, 200, 255],   # purple
        }

        for u, v, d in self.G.edges(data=True):
            if float(d.get('w', 0)) < min_w:
                continue
            i = node_idx.get(u)
            j = node_idx.get(v)
            if i is None or j is None:
                continue

            
            if not (vis[i] and vis[j]):
                continue

            is_hl_edge = hl_tf and ((u == hl_tf and v in hl_tgt) or
                                     (v == hl_tf and u in hl_tgt))

            # LOD: on large graphs skip non-highlighted edges entirely (TESTING)
            if large_graph and not is_hl_edge:
                active_nodes.add(u)
                active_nodes.add(v)
                continue

            active_nodes.add(u)
            active_nodes.add(v)

            if is_hl_edge:
                clr = self.clr_hl_edge
                th  = et * 2.5
            else:
                clr = self.clr_edge
                th  = et

            p1 = [float(Sx[i]), float(Sy[i])]
            p2 = [float(Sx[j]), float(Sy[j])]   

            undirected_edge = d.get('undirected', False)
            if sng_active or (is_pkl and not undirected_edge):
                dx, dy = p2[0]-p1[0], p2[1]-p1[1]
                ln = max((dx**2+dy**2)**0.5, 1e-6)
                shrink = self.node_scale + 4
                p2s = [p2[0] - dx/ln*shrink, p2[1] - dy/ln*shrink]
                if undirected_edge:
                    dpg.draw_line(p1=p1, p2=p2s, color=clr,
                                  thickness=th, parent="graph_layer")
                else:
                    self._draw_arrow(p1, p2s, clr, th)
            else:
                dpg.draw_line(p1=p1, p2=p2, color=clr, thickness=th,
                              parent="graph_layer")

        r = self.node_scale
        for node in active_nodes:
            i = node_idx.get(node)
            if i is None or not vis[i]:
                continue
            cx, cy = float(Sx[i]), float(Sy[i])

            if sng_active and node in sng_roles:
                role   = sng_roles[node]
                fill   = SNG_COLORS.get(role, self.clr_node)
                node_r = r * 2.2 if role in ("gene", "direct_tf") else r
            elif is_pkl:
                if node == hl_tf:
                    fill   = self.clr_hl_tf
                    node_r = r * 2.2          # highlighted TF: larger
                    dpg.draw_circle(center=[cx, cy], radius=node_r + 5,
                                    color=[*self.clr_hl_tf[:3], 50], thickness=2,
                                    parent="graph_layer")
                elif node in pkl_tfs:
                    fill   = self.clr_hl_tf
                    node_r = r * 2.2          # TF hubs always larger
                elif node in hl_tgt:
                    fill   = self.clr_hl_edge
                    node_r = r
                else:
                    fill   = self.clr_hl_tgt
                    node_r = r
            else:
                node_r = r
                if node == hl_tf:
                    fill = self.clr_hl_tf
                    dpg.draw_circle(center=[cx, cy], radius=r + 5,
                                    color=[*self.clr_hl_tf[:3], 50], thickness=2,
                                    parent="graph_layer")
                elif node in hl_tgt:
                    fill = self.clr_hl_tgt
                else:
                    fill = self.clr_node

            dpg.draw_circle(center=[cx, cy], radius=node_r, fill=fill,
                            color=[0, 0, 0, 0], parent="graph_layer")

            show_label = (sng_active or node == hl_tf or node in hl_tgt or
                          (is_pkl and node in pkl_tfs and n_nodes < NODE_LABEL_LOD))
            if show_label:
                dpg.draw_text(pos=[cx + node_r + 3, cy - 6], text=str(node),
                              size=12, color=[225, 225, 225, 220],
                              parent="graph_layer")

        self._dirty = False

    def _slider(self, label, default, lo, hi, attr, fmt="%.2f", cb=None):
        dpg.add_text(label, color=[140, 145, 155])
        def _cb(s, a):
            setattr(self, attr, a)
            self._dirty = True
            if cb: cb(a)
        dpg.add_slider_float(width=-1, default_value=default,
                             min_value=lo, max_value=hi, format=fmt, callback=_cb)
        dpg.add_spacer(height=2)

    def _on_prevent_overlap_toggle(self, sender, app_data):
        self.prevent_overlap = app_data
        if app_data:
            # apply spread to current positions without full recompute
            self._spread_overlaps_fast()
            self._reset_view()
            self._dirty = True
        else:
            # recompute layout without spread
            preset = dpg.get_value(self.layout_combo) if hasattr(self, 'layout_combo') else "ForceAtlas2"
            self._compute_layout(preset)

    def init_gui(self):
        dpg.create_context()

        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,         self.clr_bg)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,          [14, 14, 18])
                dpg.add_theme_color(dpg.mvThemeCol_Border,           [38, 40, 52])
                dpg.add_theme_color(dpg.mvThemeCol_Text,             [180, 182, 192])
                dpg.add_theme_color(dpg.mvThemeCol_Button,           [26, 28, 36])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,    [50, 160, 152])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,     [35, 120, 112])
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,       self.clr_accent)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, [60, 220, 210])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,          [20, 21, 27])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,   [28, 30, 38])
                dpg.add_theme_color(dpg.mvThemeCol_Header,           [20, 60, 58])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,    [30, 85, 80])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,     [40, 110, 105])
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg,          [16, 16, 22])
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,   0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,    4)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,     4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,      8, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,     6, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,    10, 8)
        dpg.bind_theme(global_theme)

        with dpg.file_dialog(tag="csv_dialog", directory_selector=False, show=False,
                             callback=self.on_csv_selected, width=640, height=440):
            dpg.add_file_extension(".csv", color=[100, 220, 100, 255])
        with dpg.file_dialog(tag="pkl_dialog", directory_selector=False, show=False,
                             callback=self.on_pkl_selected, width=640, height=440):
            dpg.add_file_extension(".pkl", color=[100, 180, 255, 255])
        with dpg.file_dialog(tag="export_dialog", directory_selector=False, show=False,
                             callback=self._on_export_selected, width=640, height=440,
                             default_filename="regulon_targets.csv"):
            dpg.add_file_extension(".csv", color=[100, 220, 100, 255])

        with dpg.window(tag="main_win", no_scrollbar=True, no_move=True,
                        no_resize=True, no_title_bar=True):

            with dpg.group(horizontal=True):

                with dpg.child_window(width=SIDEBAR_W, height=-1,
                                      border=True, no_scrollbar=False):

                    dpg.add_text("  NetGrep", color=self.clr_accent)
                    dpg.add_separator()
                    dpg.add_spacer(height=2)

                    with dpg.collapsing_header(label="  FILES", default_open=True):
                        dpg.add_spacer(height=2)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Load CSV", width=175,
                                           callback=lambda: dpg.show_item("csv_dialog"))
                            dpg.add_button(label="Load PKL", width=-1,
                                           callback=lambda: dpg.show_item("pkl_dialog"))
                        dpg.add_spacer(height=4)
                        dpg.add_text("Visualize from:", color=[140, 145, 155])
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="CSV Graph", width=175,
                                           callback=lambda: self._switch_source("CSV"))
                            dpg.add_button(label="PKL Graph", width=-1,
                                           callback=lambda: self._switch_source("PKL"))
                        dpg.add_spacer(height=4)
                        dpg.add_text("Max edges (PKL)  [0 = all]", color=[140, 145, 155])
                        dpg.add_slider_int(width=-1, default_value=0,
                                           min_value=0, max_value=1000000,
                                           callback=lambda s, a: self._apply_max_edges(a))
                        dpg.add_spacer(height=4)

                    with dpg.collapsing_header(label="  FILTER", default_open=False):
                        dpg.add_spacer(height=2)
                        dpg.add_text("Min Edge Weight", color=[140, 145, 155])
                        wmax = float(max(self.weights)) if self.weights else 10.0
                        dpg.add_slider_float(
                            tag="weight_slider", width=-1,
                            min_value=0.0, max_value=wmax, default_value=0.0,
                            format="%.2f",
                            callback=lambda s, a: [setattr(self, 'min_weight', a),
                                                   setattr(self, '_dirty', True)])
                        dpg.add_spacer(height=3)
                        with dpg.plot(height=80, width=-1, no_title=True,
                                      no_mouse_pos=True, no_box_select=True, no_menus=True):
                            dpg.add_plot_axis(dpg.mvXAxis, tag="hist_x",
                                              no_gridlines=True, no_tick_marks=True)
                            dpg.set_axis_limits("hist_x", 0.0, wmax)
                            with dpg.plot_axis(dpg.mvYAxis, tag="hist_y",
                                               no_tick_labels=True, no_gridlines=True,
                                               no_tick_marks=True):
                                dpg.set_axis_limits("hist_y", 0.0,
                                                    float(self.hist_data.max()) * 1.15 + 1)
                                bw = float(self.bin_centers[1] - self.bin_centers[0]) \
                                     if len(self.bin_centers) > 1 else 1.0
                                self.hist_bars_tag = dpg.add_bar_series(
                                    self.bin_centers.tolist(),
                                    self.hist_data.tolist(), weight=bw * 0.9)
                        dpg.add_spacer(height=4)

                    with dpg.collapsing_header(label="  LAYOUT", default_open=True):
                        dpg.add_spacer(height=2)
                        self.layout_combo = dpg.add_combo(
                            items=["ForceAtlas2", "Spring", "Spectral", "Circular", "Random"],
                            default_value="ForceAtlas2", width=-1)
                        dpg.add_spacer(height=3)
                        self.layout_apply_btn = dpg.add_button(
                            label="Apply Layout", width=-1,
                            callback=lambda: self.apply_layout())
                        dpg.add_spacer(height=3)
                        self.layout_progress_bar = dpg.add_progress_bar(
                            width=-1, default_value=0.0, show=False,
                            overlay="Computing layout...")
                        dpg.add_spacer(height=4)
                        dpg.add_text("Hub Separation", color=[140, 145, 155])
                        dpg.add_slider_float(
                            tag="hub_sep_slider", width=-1,
                            default_value=self.hub_separation,
                            min_value=1.0, max_value=20.0, format="%.1f",
                            callback=lambda s, a: [
                                setattr(self, 'hub_separation', a),
                                self._apply_cluster_spread()])
                        dpg.add_spacer(height=3)
                        dpg.add_text("Cluster Spread", color=[140, 145, 155])
                        dpg.add_slider_float(
                            tag="cluster_spread_slider", width=-1,
                            default_value=self.cluster_spread,
                            min_value=0.5, max_value=5.0, format="%.2f",
                            callback=lambda s, a: [
                                setattr(self, 'cluster_spread', a),
                                self._apply_cluster_spread()])
                        dpg.add_spacer(height=6)
                        dpg.add_button(label="Center to Fit", width=-1,
                                       callback=lambda: self.center_to_fit())
                        dpg.add_spacer(height=4)

                    with dpg.collapsing_header(label="  APPEARANCE", default_open=False):
                        dpg.add_spacer(height=2)
                        self._slider("Node Size",  self.node_scale,     1.0, 20.0, 'node_scale',     "%.1f")
                        self._slider("Edge Width", self.edge_thickness, 0.3,  5.0, 'edge_thickness', "%.2f")
                        dpg.add_spacer(height=4)
                        with dpg.table(header_row=False, borders_innerV=False,
                                       borders_outerH=False, borders_outerV=False):
                            for _ in range(4):
                                dpg.add_table_column(width_fixed=True, init_width_or_weight=84)
                            with dpg.table_row():
                                for lbl in ("Node", "Edge", "TF", "Target"):
                                    dpg.add_text(lbl, color=[140, 145, 155])
                            with dpg.table_row():
                                dpg.add_color_edit(default_value=self.clr_node,
                                                   no_inputs=True, callback=self._clr_cb('clr_node'))
                                dpg.add_color_edit(default_value=self.clr_edge,
                                                   no_inputs=True, callback=self._clr_cb('clr_edge'))
                                dpg.add_color_edit(default_value=self.clr_hl_tf,
                                                   no_inputs=True, callback=self._clr_cb('clr_hl_tf'))
                                dpg.add_color_edit(default_value=self.clr_hl_tgt,
                                                   no_inputs=True, callback=self._clr_cb('clr_hl_tgt'))
                        dpg.add_spacer(height=4)

                    with dpg.collapsing_header(label="  REGULONS", default_open=False):
                        dpg.add_spacer(height=2)
                        self.regulon_status = dpg.add_text(
                            f"{len(self.regulons)} regulons loaded" if self.regulons
                            else "No .pkl loaded", color=[120, 125, 138])
                        dpg.add_spacer(height=3)
                        dpg.add_input_text(hint="Search regulons", width=-1,
                                           callback=self._filter_regulons)
                        dpg.add_spacer(height=3)
                        self.regulon_listbox = dpg.add_listbox(
                            items=list(self.regulons.keys()),
                            width=-1, num_items=7,
                            callback=self.select_regulon)
                        dpg.add_spacer(height=3)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Clear Selection", width=175,
                                           callback=self._clear_regulon)
                            self._subgraph_btn = dpg.add_button(
                                label="View Subgraph", width=-1,
                                callback=self._enter_subgraph_mode)
                        dpg.add_spacer(height=3)
                        dpg.add_button(label="Export Targets as CSV", width=-1,
                                       callback=self._export_regulon_targets)
                        dpg.add_spacer(height=4)

                    with dpg.collapsing_header(label="  SUBNETGREP", default_open=False):
                        dpg.add_spacer(height=2)
                        dpg.add_text("Gene of interest", color=[140, 145, 155])
                        self._subnetgrep_input = dpg.add_input_text(
                            hint="e.g. Sox2", width=-1)
                        dpg.add_spacer(height=3)
                        # legend
                        with dpg.group(horizontal=True):
                            dpg.draw_circle(center=[8,8], radius=7,
                                            fill=[50,200,80,255],  color=[0,0,0,0],
                                            parent=dpg.add_drawlist(width=18, height=18))
                            dpg.add_text("Gene of interest", color=[140,145,155])
                        with dpg.group(horizontal=True):
                            dpg.draw_circle(center=[8,8], radius=7,
                                            fill=[220,60,60,255],  color=[0,0,0,0],
                                            parent=dpg.add_drawlist(width=18, height=18))
                            dpg.add_text("Direct TF regulators", color=[140,145,155])
                        with dpg.group(horizontal=True):
                            dpg.draw_circle(center=[8,8], radius=7,
                                            fill=[60,130,220,255], color=[0,0,0,0],
                                            parent=dpg.add_drawlist(width=18, height=18))
                            dpg.add_text("Shared adjacencies",   color=[140,145,155])
                        with dpg.group(horizontal=True):
                            dpg.draw_circle(center=[8,8], radius=7,
                                            fill=[160,80,200,255], color=[0,0,0,0],
                                            parent=dpg.add_drawlist(width=18, height=18))
                            dpg.add_text("Indirect TF regulators", color=[140,145,155])
                        dpg.add_spacer(height=4)
                        self._subnetgrep_run_btn = dpg.add_button(
                            label="Run SubNetGrep", width=-1,
                            callback=self._run_subnetgrep)
                        self._subnetgrep_exit_btn = dpg.add_button(
                            label="Exit SubNetGrep", width=-1,
                            callback=self._exit_subnetgrep, show=False)
                        dpg.add_spacer(height=4)

                with dpg.child_window(tag="canvas_window", width=-1, height=-1,
                                      border=False, no_scrollbar=True):
                    with dpg.drawlist(width=CANVAS_W, height=CANVAS_H, tag="canvas"):
                        with dpg.draw_layer(tag="graph_layer"):
                            pass

        dpg.create_viewport(title="NetGrep", width=W, height=H, resizable=False)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_win", True)

        while dpg.is_dearpygui_running():
            self.render_frame()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def _on_scale_slider(self, val):
        self.user_scale = float(val)
        self._apply_user_scale()

    def _filter_regulons(self, sender, app_data):
        q = app_data.lower()
        filtered = [n for n in self.regulons_json if q in n.lower()]
        dpg.configure_item(self.regulon_listbox, items=filtered)

    def _layout_radial_tf(self):
        G = self.G
        if not self.pkl_tfs or self.viz_source != "PKL":
            return None
        tfs   = [n for n in G.nodes() if n in self.pkl_tfs]
        genes = [n for n in G.nodes() if n not in self.pkl_tfs]
        N_tf  = len(tfs)
        pos   = {}
        for i, tf in enumerate(tfs):
            angle = 2 * np.pi * i / max(N_tf, 1)
            pos[tf] = np.array([np.cos(angle), np.sin(angle)]) * 1.0
        rng = np.random.default_rng(42)
        for gene in genes:
            preds = [p for p in G.predecessors(gene) if p in pos]
            if preds:
                centre = np.mean([pos[p] for p in preds], axis=0)
                pos[gene] = centre * 0.55 + rng.uniform(-0.12, 0.12, 2)
            else:
                pos[gene] = rng.uniform(-0.4, 0.4, 2)
        return pos

    def _layout_hierarchical(self):
        G  = self.G
        Gu = G.to_undirected() if G.is_directed() else G
        try:
            layers = {}
            Gd = G if G.is_directed() else G.to_directed()
            for n in nx.topological_sort(Gd):
                preds = list(Gd.predecessors(n))
                layers[n] = (max(layers[p] for p in preds) + 1) if preds else 0
        except nx.NetworkXUnfeasible:
            root = max(G.nodes(), key=lambda n: G.degree(n))
            layers = nx.single_source_shortest_path_length(Gu, root)
        max_layer = max(layers.values()) if layers else 1
        by_layer  = {}
        for n, l in layers.items():
            by_layer.setdefault(l, []).append(n)
        rng = np.random.default_rng(42)
        pos = {}
        for l, nds in by_layer.items():
            y = l / max(max_layer, 1)
            for j, n in enumerate(nds):
                x = (j + 0.5) / len(nds)
                pos[n] = np.array([x + rng.uniform(-0.02, 0.02), y])
        return pos

    def _spread_overlaps_fast(self, iterations=25):
        if len(self.pos) < 2:
            return
        nodes = list(self.pos.keys())
        P     = np.array([self.pos[n] for n in nodes], dtype=float)
        deg   = np.array([self.G.degree(n) + 1 for n in nodes], dtype=float)
        sep   = 0.04 * (deg ** 0.4)
        for _ in range(iterations):
            diff  = P[:, None, :] - P[None, :, :]
            dist  = np.sqrt((diff ** 2).sum(axis=2) + 1e-12)
            np.fill_diagonal(dist, np.inf)
            min_d = sep[:, None] + sep[None, :]
            close = dist < min_d
            if not close.any():
                break
            overlap = np.where(close, (min_d - dist) / 2, 0.0)
            unit    = diff / dist[:, :, None]
            P      += (overlap[:, :, None] * unit).sum(axis=1)
        for i, n in enumerate(nodes):
            self.pos[n] = P[i]

    def _reset_view(self):
        """Fit the current layout to canvas. Called once after layout completes."""
        self.user_scale = 1.0
        if not self.pos:
            self.view_scale = 300.0
            self.view_ox = CANVAS_W / 2
            self.view_oy = CANVAS_H / 2
            return
        pts = np.array(list(self.pos.values()))
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        span = np.where((mx - mn) < 1e-6, 1.0, mx - mn)
        self._base_scale = float(min(CANVAS_W * 0.88 / span[0],
                                     CANVAS_H * 0.88 / span[1]))
        mid = (mn + mx) / 2
        self._base_ox = CANVAS_W / 2 - mid[0] * self._base_scale
        self._base_oy = CANVAS_H / 2 - mid[1] * self._base_scale
        self._apply_user_scale()

    def _apply_user_scale(self):
        s = self._base_scale * self.user_scale
        mid_screen_x = CANVAS_W / 2
        mid_screen_y = CANVAS_H / 2
        gx = (mid_screen_x - self._base_ox) / self._base_scale
        gy = (mid_screen_y - self._base_oy) / self._base_scale
        self.view_scale = s
        self.view_ox = mid_screen_x - gx * s
        self.view_oy = mid_screen_y - gy * s
        self._dirty = True

    def _switch_source(self, source):
        self.viz_source = source
        self.G       = self.G_csv if source == "CSV" else self.G_pkl
        self.degrees = dict(self.G.degree())
        # update layout combo items based on source
        self._update_layout_combo()
        preset = dpg.get_value(self.layout_combo) if hasattr(self, 'layout_combo') else "ForceAtlas2"
        self._compute_layout(preset)
        self._rebuild_weights()
        self._update_histogram_ui()

    def _update_layout_combo(self):
        if not hasattr(self, 'layout_combo'):
            return
        if self.viz_source == "PKL":
            items   = ["GRN Cluster", "Spiral"]
            default = "GRN Cluster"
        else:
            items   = ["ForceAtlas2", "Spring", "Spectral", "Circular", "Random"]
            default = "ForceAtlas2"
        dpg.configure_item(self.layout_combo, items=items)
        dpg.set_value(self.layout_combo, default)

    def apply_layout(self):
        self._compute_layout(dpg.get_value(self.layout_combo))

    def center_to_fit(self):
        """Zoom/pan to fit current self.pos without modifying layout."""
        if not self.pos:
            return
        pts  = np.array(list(self.pos.values()))
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        span = np.where((mx - mn) < 1e-6, 1.0, mx - mn)
        fit_scale = float(min(CANVAS_W * 0.88 / span[0],
                              CANVAS_H * 0.88 / span[1]))
        mid = (mn + mx) / 2
        self.view_scale = fit_scale
        self.view_ox    = CANVAS_W / 2 - mid[0] * fit_scale
        self.view_oy    = CANVAS_H / 2 - mid[1] * fit_scale
        self._dirty     = True

    def _apply_max_edges(self, val):
        self.max_edges = int(val) if val > 0 else 999_999_999
        if self.regulons_json:
            self.G_pkl = self._build_graph_from_json()
            if self.viz_source == "PKL":
                self.G = self.G_pkl
                self.degrees = dict(self.G.degree())
                self._compute_layout(dpg.get_value(self.layout_combo))
                self._rebuild_weights()
                self._update_histogram_ui()
                n, e = self.G_pkl.number_of_nodes(), self.G_pkl.number_of_edges()
                dpg.set_value(self.regulon_status,
                              f"{len(self.regulons_json)} regulons  |  {n}n  {e}e")

    def _update_histogram_ui(self):
        if not hasattr(self, 'hist_bars_tag'):
            return
        wmax = float(max(self.weights)) if self.weights else 1.0
        dpg.set_value(self.hist_bars_tag,
                      [self.bin_centers.tolist(), self.hist_data.tolist()])
        dpg.set_axis_limits("hist_x", 0.0, wmax)
        dpg.set_axis_limits("hist_y", 0.0, float(self.hist_data.max()) * 1.15 + 1)
        dpg.configure_item("weight_slider", max_value=wmax, default_value=0.0)
        self.min_weight = 0.0
        self._dirty = True

    @staticmethod
    def _to_rgba(v):
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            if all(isinstance(x, float) and x <= 1.0 for x in v):
                return [int(x * 255) for x in v]
        return list(v)

    def _clr_cb(self, attr):
        def cb(s, a):
            setattr(self, attr, self._to_rgba(a))
            self._dirty = True
        return cb

    def on_csv_selected(self, sender, app_data):
        path = app_data.get('file_path_name', '')
        if not path:
            return
        self.G_csv = self._load_csv(path)
        if self.viz_source == "CSV":
            self.G = self.G_csv
            self.degrees = dict(self.G.degree())
            self._compute_layout(dpg.get_value(self.layout_combo))
            self._rebuild_weights()
            self._update_histogram_ui()

    def on_pkl_selected(self, sender, app_data):
        path = app_data.get('file_path_name', '')
        if not path:
            return
        self.regulons = {}
        self.regulons_json = {}
        self.load_pkl(path)
        self.convert_pkl_to_json()
        self.G_pkl = self._build_graph_from_json()
        names = list(self.regulons_json.keys())
        dpg.configure_item(self.regulon_listbox, items=names)
        n, e = self.G_pkl.number_of_nodes(), self.G_pkl.number_of_edges()
        dpg.set_value(self.regulon_status, f"{len(names)} regulons  |  {n}n  {e}e")
        if self.viz_source == "PKL":
            self.G = self.G_pkl
            self.degrees = dict(self.G.degree())
            self._compute_layout(dpg.get_value(self.layout_combo))
            self._rebuild_weights()
            self._update_histogram_ui()

    def select_regulon(self, sender, app_data):
        name = dpg.get_value(self.regulon_listbox)
        if not name or name not in self.regulons_json:
            self._clear_regulon()
            return
        # toggle off if already selected
        if name == self.highlighted_tf:
            self._clear_regulon()
            return
        self.highlighted_tf      = name
        self.highlighted_targets = set(self.regulons_json[name]["targets"])
        self._dirty = True

    def _clear_regulon(self):
        self.highlighted_tf      = None
        self.highlighted_targets = set()
        if hasattr(self, 'regulon_listbox'):
            dpg.set_value(self.regulon_listbox, "")
        self._dirty = True

    def _draw_arrow(self, p1, p2, color, thickness):
        dpg.draw_line(p1=p1, p2=p2, color=color, thickness=thickness,
                      parent="graph_layer")
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = max((dx**2 + dy**2)**0.5, 1e-6)
        ux, uy = dx / length, dy / length
        arr = 8.0 * thickness ** 0.5
        ax = p2[0] - ux * arr
        ay = p2[1] - uy * arr
        px, py = -uy * arr * 0.4, ux * arr * 0.4
        dpg.draw_triangle(
            p1=p2,
            p2=[ax + px, ay + py],
            p3=[ax - px, ay - py],
            color=color, fill=color,
            parent="graph_layer")

    def _export_regulon_targets(self):
        """Export the current regulon's targets and weights to a CSV file."""
        name = self.highlighted_tf
        if not name or name not in self.regulons_json:
            return
        safe_name = name.replace('/', '_').replace('\\', '_')
        dpg.configure_item("export_dialog",
                           default_filename=f"{safe_name}_targets")
        dpg.show_item("export_dialog")

    def _on_export_selected(self, sender, app_data):
        path = app_data.get('file_path_name', '')
        if not path:
            return
        # ensure .csv extension
        if not path.lower().endswith('.csv'):
            path += '.csv'
        name = self.highlighted_tf
        if not name or name not in self.regulons_json:
            return
        gene_weights = self.regulons_json[name]["gene_weights"]
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["gene", "weight"])
                for gene, weight in sorted(gene_weights.items(),
                                           key=lambda x: x[1], reverse=True):
                    writer.writerow([gene, weight])
            print(f"[export] {len(gene_weights)} targets written to {path}")
        except Exception as e:
            print(f"[export error] {e}")

    # SubNetGrep

    def _run_subnetgrep(self):
        gene = dpg.get_value(self._subnetgrep_input).strip()
        if not gene:
            return
        if not self.regulons_json:
            return
        self._subnetgrep_gene = gene

        # Direct TFs (red): regulons that target gene 
        direct_tfs = {tf for tf, data in self.regulons_json.items()
                      if gene in data["gene_weights"]}

        # CSV adjacencies (blue): neighbours of gene in CSV graph that share at least one direct TF
        csv_neighbours = set()
        if gene in self.G_csv:
            csv_neighbours = set(self.G_csv.neighbors(gene))

        blue_nodes = set()
        for nb in csv_neighbours:
            nb_tfs = {tf for tf, data in self.regulons_json.items()
                      if nb in data["gene_weights"]}
            if nb_tfs & direct_tfs:          # shares at least one red TF
                blue_nodes.add(nb)

        # Indirect TFs (purple): regulate blue nodes but not green 
        indirect_tfs = set()
        for nb in blue_nodes:
            for tf, data in self.regulons_json.items():
                if nb in data["gene_weights"] and tf not in direct_tfs:
                    indirect_tfs.add(tf)

        # Build subgraph
        sub = nx.DiGraph()
        sub.add_node(gene,        role="gene")
        for tf in direct_tfs:    sub.add_node(tf,  role="direct_tf")
        for nb in blue_nodes:    sub.add_node(nb,  role="adjacent")
        for tf in indirect_tfs:  sub.add_node(tf,  role="indirect_tf")

        # directed: TF → gene
        for tf in direct_tfs:
            w = self.regulons_json[tf]["gene_weights"].get(gene, 0.0)
            sub.add_edge(tf, gene, w=w)

        # undirected-style: gene — blue 
        for nb in blue_nodes:
            d = self.G_csv.get_edge_data(gene, nb) or {}
            w = float(d.get('w', d.get('weight', 1.0)))
            sub.add_edge(gene, nb, w=w, undirected=True)
            sub.add_edge(nb, gene, w=w, undirected=True)

        # directed: TF → blue
        for tf in direct_tfs:
            for nb in blue_nodes:
                if nb in self.regulons_json.get(tf, {}).get("gene_weights", {}):
                    w = self.regulons_json[tf]["gene_weights"][nb]
                    sub.add_edge(tf, nb, w=w)

        # directed: indirect TF → blue
        for tf in indirect_tfs:
            for nb in blue_nodes:
                if nb in self.regulons_json.get(tf, {}).get("gene_weights", {}):
                    w = self.regulons_json[tf]["gene_weights"][nb]
                    sub.add_edge(tf, nb, w=w)

        pos = self._layout_subnetgrep(gene, direct_tfs, blue_nodes,
                                      indirect_tfs, sub)

        self._subnetgrep_graph    = sub
        self._subnetgrep_pos      = pos
        self._subnetgrep_roles    = {
            gene:  "gene",
            **{tf: "direct_tf"   for tf in direct_tfs},
            **{nb: "adjacent"    for nb in blue_nodes},
            **{tf: "indirect_tf" for tf in indirect_tfs},
        }
        self._subnetgrep_active   = True
        self._pre_subnetgrep_G    = self.G
        self._pre_subnetgrep_pos  = dict(self.pos)
        self.G   = sub
        self.pos = pos
        self._base_pos = dict(pos)
        self._reset_view()
        self._dirty = True
        dpg.configure_item(self._subnetgrep_run_btn,  show=False)
        dpg.configure_item(self._subnetgrep_exit_btn, show=True)

    def _layout_subnetgrep(self, gene, direct_tfs, blue_nodes,
                           indirect_tfs, sub):
        """
        Concentric ring layout:
          centre      → gene of interest
          ring 1      → direct TFs (red)
          ring 2      → blue adjacency nodes
          ring 3      → indirect TFs (purple)
        """
        pos = {gene: np.array([0.0, 0.0])}

        def _ring(nodes, radius):
            nodes = sorted(nodes)
            n = max(len(nodes), 1)
            return {nd: np.array([radius * np.cos(2*np.pi*k/n),
                                  radius * np.sin(2*np.pi*k/n)])
                    for k, nd in enumerate(nodes)}

        n_d  = max(len(direct_tfs),   1)
        n_b  = max(len(blue_nodes),   1)
        n_i  = max(len(indirect_tfs), 1)

        R1 = max(1.5, n_d * 0.35)
        R2 = R1 + max(1.5, n_b * 0.25)
        R3 = R2 + max(1.5, n_i * 0.25)

        pos.update(_ring(direct_tfs,   R1))
        pos.update(_ring(blue_nodes,   R2))
        pos.update(_ring(indirect_tfs, R3))
        return pos

    def _exit_subnetgrep(self):
        if not getattr(self, '_subnetgrep_active', False):
            return
        self._subnetgrep_active = False
        self.G   = self._pre_subnetgrep_G
        self.pos = self._pre_subnetgrep_pos
        self._base_pos = dict(self.pos)
        self._reset_view()
        self._dirty = True
        dpg.configure_item(self._subnetgrep_run_btn,  show=True)
        dpg.configure_item(self._subnetgrep_exit_btn, show=False)

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    pkl_path = sys.argv[2] if len(sys.argv) > 2 else None
    app = NetGrep(csv_path, pkl_path)