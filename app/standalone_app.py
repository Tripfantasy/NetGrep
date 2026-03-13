import dearpygui.dearpygui as dpg
import networkx as nx
import polars as pl
import numpy as np
import pickle
import sys
import types
import json

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
        self._dirty     = True

        self.csv_path   = file_path
        self.pkl_path   = pkl_path
        self.regulons   = {}
        self.regulons_json = {}
        self.viz_source = "CSV"

        # set of TF node names in PKL graph
        self.pkl_tfs     = set()

        self.highlighted_tf      = None
        self.highlighted_targets = set()

        self.G_csv = self._load_csv(file_path)
        if pkl_path:
            self.load_pkl(pkl_path)
            self.convert_pkl_to_json()
        self.G_pkl = self._build_graph_from_json()

        self.G = self.G_csv
        self.degrees = dict(self.G.degree())
        self._compute_layout("Spring")
        self._rebuild_weights()

        self.init_gui()


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
        raw = getattr(reg, 'transcription_factor', fallback)
        return raw.split('(')[0] if '(' in raw else raw

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

    def _compute_layout(self, preset):
        G = self.G
        if G.number_of_nodes() == 0:
            self.pos = {}; self._dirty = True; return
        N = G.number_of_nodes()
        # use undirected copy for layout
        Gu = G.to_undirected() if G.is_directed() else G
        if preset == "Spring":
            raw = nx.spring_layout(Gu, k=2.0 / max(1, N**0.5), seed=42,
                                   iterations=max(30, min(100, 5000 // max(N, 1))))
        elif preset == "Circular":
            raw = nx.circular_layout(Gu)
        elif preset == "Kamada-Kawai":
            try:
                if N > 500: raise ValueError
                raw = nx.kamada_kawai_layout(Gu)
            except:
                raw = nx.spring_layout(Gu, k=2.0 / max(1, N**0.5), seed=42, iterations=50)
        elif preset == "Spectral":
            try:    raw = nx.spectral_layout(Gu)
            except: raw = nx.spring_layout(Gu, k=2.0 / max(1, N**0.5), seed=42, iterations=50)
        else:
            rng = np.random.default_rng(42)
            raw = {n: rng.uniform(-1, 1, 2) for n in G.nodes()}
        self.pos = {n: np.array(p, dtype=float) for n, p in raw.items()}
        self._spread_overlaps_fast()
        self._reset_view()
        self._dirty = True

    def _reset_view(self):
        """Fit layout into canvas, reset user scale to 1."""
        self.user_scale = 1.0
        if hasattr(self, 'scale_slider'):
            dpg.set_value(self.scale_slider, 1.0)
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
        # re-centre after scaling
        mid_screen_x = CANVAS_W / 2
        mid_screen_y = CANVAS_H / 2
        # graph-space centre
        gx = (mid_screen_x - self._base_ox) / self._base_scale
        gy = (mid_screen_y - self._base_oy) / self._base_scale
        self.view_scale = s
        self.view_ox = mid_screen_x - gx * s
        self.view_oy = mid_screen_y - gy * s
        self._dirty = True

    def _spread_overlaps_fast(self, iterations=15):
        if len(self.pos) < 2:
            return
        nodes = list(self.pos.keys())
        P = np.array([self.pos[n] for n in nodes], dtype=float)
        min_sep = 0.035
        for _ in range(iterations):
            diff  = P[:, None, :] - P[None, :, :]
            dist  = np.sqrt((diff**2).sum(axis=2) + 1e-12)
            np.fill_diagonal(dist, np.inf)
            close = dist < min_sep
            if not close.any():
                break
            overlap = np.where(close, (min_sep - dist) / 2, 0.0)
            unit    = diff / dist[:, :, None]
            P      += (overlap[:, :, None] * unit).sum(axis=1)
        for i, n in enumerate(nodes):
            self.pos[n] = P[i]

    def _switch_source(self, source):
        self.viz_source = source
        self.G       = self.G_csv if source == "CSV" else self.G_pkl
        self.degrees = dict(self.G.degree())
        preset = dpg.get_value(self.layout_combo) if hasattr(self, 'layout_combo') else "Spring"
        self._compute_layout(preset)
        self._rebuild_weights()
        self._update_histogram_ui()

    def apply_layout(self):
        self._compute_layout(dpg.get_value(self.layout_combo))

    def center_to_fit(self):
        self._reset_view()

    def _apply_max_edges(self, val):
        # 0 means unlimited
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
        if name and name in self.regulons_json:
            self.highlighted_tf      = name
            self.highlighted_targets = set(self.regulons_json[name]["targets"])
        else:
            self._clear_regulon()
        self._dirty = True

    def _clear_regulon(self):
        self.highlighted_tf      = None
        self.highlighted_targets = set()
        if hasattr(self, 'regulon_listbox'):
            dpg.set_value(self.regulon_listbox, "")
        self._dirty = True


    def _draw_arrow(self, p1, p2, color, thickness):
        """Draw a line with a small arrowhead at p2."""
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

    def render_frame(self):
        if not self._dirty:
            return
        self._dirty = False

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

        # LOD thresholds
        EDGE_LOD_SKIP  = 15_000   
        NODE_LABEL_LOD = 500      
        large_graph    = n_edges > EDGE_LOD_SKIP

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

            # LOD: on large graphs skip non-highlighted edges entirely
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

            if is_pkl:
                dx, dy = p2[0]-p1[0], p2[1]-p1[1]
                ln = max((dx**2+dy**2)**0.5, 1e-6)
                shrink = self.node_scale + 4
                p2s = [p2[0] - dx/ln*shrink, p2[1] - dy/ln*shrink]
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

            if is_pkl:
                if node == hl_tf:
                    fill = self.clr_hl_tf
                    dpg.draw_circle(center=[cx, cy], radius=r + 5,
                                    color=[*self.clr_hl_tf[:3], 50], thickness=2,
                                    parent="graph_layer")
                elif node in hl_tgt:
                    fill = self.clr_hl_edge
                elif node in pkl_tfs:
                    fill = self.clr_hl_tf
                else:
                    fill = self.clr_hl_tgt
            else:
                if node == hl_tf:
                    fill = self.clr_hl_tf
                    dpg.draw_circle(center=[cx, cy], radius=r + 5,
                                    color=[*self.clr_hl_tf[:3], 50], thickness=2,
                                    parent="graph_layer")
                elif node in hl_tgt:
                    fill = self.clr_hl_tgt
                else:
                    fill = self.clr_node

            dpg.draw_circle(center=[cx, cy], radius=r, fill=fill,
                            color=[0, 0, 0, 0], parent="graph_layer")

            # LOD: on large graphs only label TFs and highlighted nodes
            show_label = node == hl_tf or node in hl_tgt or \
                         (is_pkl and node in pkl_tfs and n_nodes < NODE_LABEL_LOD)
            if show_label:
                dpg.draw_text(pos=[cx + r + 3, cy - 6], text=str(node),
                              size=12, color=[225, 225, 225, 220],
                              parent="graph_layer")


    def _slider(self, label, default, lo, hi, attr, fmt="%.2f", cb=None):
        dpg.add_text(label, color=[140, 145, 155])
        def _cb(s, a):
            setattr(self, attr, a)
            self._dirty = True
            if cb: cb(a)
        dpg.add_slider_float(width=-1, default_value=default,
                             min_value=lo, max_value=hi, format=fmt, callback=_cb)
        dpg.add_spacer(height=2)

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
                            items=["Spring", "Circular", "Kamada-Kawai", "Spectral", "Random"],
                            default_value="Spring", width=-1)
                        dpg.add_spacer(height=3)
                        dpg.add_button(label="Apply Layout", width=-1,
                                       callback=lambda: self.apply_layout())
                        dpg.add_spacer(height=6)
                        dpg.add_text("Scale Factor", color=[140, 145, 155])
                        self.scale_slider = dpg.add_slider_float(
                            width=-1, default_value=1.0,
                            min_value=0.1, max_value=5.0, format="%.2f",
                            callback=lambda s, a: self._on_scale_slider(a))
                        dpg.add_spacer(height=3)
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
                        dpg.add_input_text(hint="Search regulons…", width=-1,
                                           callback=self._filter_regulons)
                        dpg.add_spacer(height=3)
                        self.regulon_listbox = dpg.add_listbox(
                            items=list(self.regulons.keys()),
                            width=-1, num_items=7,
                            callback=self.select_regulon)
                        dpg.add_spacer(height=3)
                        dpg.add_button(label="Clear Selection", width=-1,
                                       callback=self._clear_regulon)
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


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    pkl_path = sys.argv[2] if len(sys.argv) > 2 else None
    app = NetGrep(csv_path, pkl_path)