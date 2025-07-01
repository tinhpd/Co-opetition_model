import sys
import pandas as pd
import numpy as np
import torch
import networkx as nx
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QTableView, QAction,
    QMessageBox, QInputDialog
)
from PyQt5.QtCore import Qt, QAbstractTableModel, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import pyqtgraph as pg
from model import inside_competitive_dynamics_model

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_parameter(file = None):
    if file == None:
        return {'num_agent': 2,
                'alpha': torch.tensor([0.8219, 0.6502], device=device, dtype=torch.float64),
                'gamma': torch.tensor([0.9891, 0.9287], device=device, dtype=torch.float64),
                'lamda': torch.tensor([0.5496, 0.3652], device=device, dtype=torch.float64),
                'W_att': torch.tensor([[0.3882, 0.4271],[0.4271, 0.8785]], device=device, dtype=torch.float64),
                }
    else:
        import json
        from typing import Any, Dict, List

        def _convert_lists_to_tensors(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _convert_lists_to_tensors(v) for k, v in obj.items()}

            elif isinstance(obj, list):
                # Thử convert toàn bộ list thành tensor
                try:
                    # Nếu list chỉ chứa int/float hoặc nested list số, PyTorch sẽ tạo tensor thành công
                    return torch.tensor(obj, device = device, dtype = torch.float64)
                except Exception:
                    # Nếu không thể (vd. list chứa dict, string, hoặc hỗn hợp), đệ quy xuống phần tử
                    return [_convert_lists_to_tensors(x) for x in obj]

            else:
                # Nếu không phải dict hay list (ví dụ int, float, str, bool), giữ nguyên
                return obj


        def read_jsonl_and_list_to_tensor(path_jsonl: str) -> List[Dict]:
            converted_records: List[Dict] = []
            with open(path_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw_obj = json.loads(line)                    # Parse JSON thành dict
                    new_obj = _convert_lists_to_tensors(raw_obj)   # Đệ quy chuyển list→tensor
                    converted_records.append(new_obj)

            return converted_records
        records = read_jsonl_and_list_to_tensor(file)

        max_ = 0
        pr = 0
        for idx, rec in enumerate(records):
            if max_ < float(rec['score']):
                max_ = float(rec["score"])
                pr = rec
        return pr

class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame()):
        super().__init__()
        self._df = df
    def rowCount(self, parent=None): return len(self._df)
    def columnCount(self, parent=None): return len(self._df.columns)
    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            return str(self._df.iat[index.row(), index.column()])
        return None
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self._df.columns[section] if orientation==Qt.Horizontal else section
        return None

class TrainingThread(QThread):
    delta_signal = pyqtSignal(float)
    finished_signal = pyqtSignal(pd.DataFrame)
    def __init__(self, model):
        super().__init__()
        self.model = model
    def run(self):
        def cb(delta):
            if self.isInterruptionRequested():
                return
            self.delta_signal.emit(delta)
        self.model.train(callback=cb)
        if not self.isInterruptionRequested():
            self.finished_signal.emit(self.model.result())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Network Visualization App")
        self.df = pd.DataFrame(); self.result_df = None
        self.delta_x = []; self.delta_y = []
        self.thread = None
        self.setup_ui()
        self.b_param = load_parameter()

    def set_parameter(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "*.jsonl")
        if not path: return
        self.b_param = load_parameter(path)
        self.set_param_default_action.setChecked(False) 
        
    def set_default_parameter(self):
        self.b_param = load_parameter()
        self.set_param_default_action.setChecked(True)
    
    def setup_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        load_action = QAction("Load network from file", self)
        load_action.triggered.connect(self.load_file)
        file_menu.addAction(load_action)
        params_menu = menubar.addMenu("Parameters")
        self.set_param_default_action = QAction("Default", self, checkable=True)
        self.set_param_default_action.setChecked(True)
        self.set_param_default_action.triggered.connect(self.set_default_parameter)
        params_menu.addAction(self.set_param_default_action)
        set_param_from_file_action = QAction("Load parameter from file", self)
        set_param_from_file_action.triggered.connect(self.set_parameter)
        params_menu.addAction(set_param_from_file_action)
        view_menu = menubar.addMenu("View")
        self.show_graph_action = QAction("Show Graph", self, checkable=True)
        self.show_graph_action.setChecked(True)
        self.show_graph_action.triggered.connect(self.toggle_show_graph)
        view_menu.addAction(self.show_graph_action)
        self.show_graph = True

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left panel
        left = QVBoxLayout()
        layout.addLayout(left, 1)
        self.table = QTableView(); left.addWidget(self.table)
        cb_layout = QHBoxLayout(); left.addLayout(cb_layout)
        for name in ["Start","End","Weight","Direct","Network Type"]:
            combo = QComboBox(); setattr(self, f"cb_{name.lower().replace(' ','_')}", combo)
            cb_layout.addWidget(QLabel(name)); cb_layout.addWidget(combo)
        self.cb_network_type.addItems(["Directed","Undirected","Mixed"])
        self.btn_load = QPushButton("Load Network"); self.btn_load.clicked.connect(self.load_network)
        left.addWidget(self.btn_load)
        self.btn_run = QPushButton("Execute"); self.btn_run.clicked.connect(self.run_analysis); self.btn_run.setVisible(False)
        left.addWidget(self.btn_run)
        self.btn_save = QPushButton("Save Result"); self.btn_save.clicked.connect(self.save_result); self.btn_save.setVisible(False)
        left.addWidget(self.btn_save)
        self.result_table = QTableView(); left.addWidget(self.result_table)

        # Right panel
        right = QVBoxLayout(); layout.addLayout(right, 2)
        self.net_view = pg.GraphicsLayoutWidget(); self.net_view.setBackground('w'); right.addWidget(self.net_view)
        self.net_plot = self.net_view.addPlot(); self.net_plot.showGrid(x=True, y=True); self.net_plot.setAspectLocked(True)
        self.graph_item = pg.GraphItem(); self.net_plot.addItem(self.graph_item)

        # Stop button and delta label controls
        ctrl_layout = QHBoxLayout(); right.addLayout(ctrl_layout)
        self.btn_stop = QPushButton("Stop Training"); self.btn_stop.clicked.connect(self.stop_training)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addStretch()

        # Delta plot
        self.delta_view = pg.PlotWidget(title="Delta per Iteration"); self.delta_view.setBackground('w'); right.addWidget(self.delta_view)
        self.delta_curve = self.delta_view.plot(self.delta_x, self.delta_y, pen=pg.mkPen('r', width=2), symbol='o', symbolSize=6)
        self.delta_label = pg.TextItem('', anchor=(1, 0)); self.delta_view.addItem(self.delta_label)

    def toggle_show_graph(self, checked):
        self.show_graph = checked

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "*.csv *.tsv *.txt")
        if not path: return
        sep = '\t' if path.endswith(('.tsv', '.txt')) else ','
        try: self.df = pd.read_csv(path, sep=sep)
        except Exception as e: QMessageBox.critical(self, "Error", str(e)); return
        self.table.setModel(PandasModel(self.df))
        cols = list(self.df.columns)
        for cb, allow_none in [(self.cb_start, False), (self.cb_end, False), (self.cb_weight, True), (self.cb_direct, True)]:
            cb.clear();
            if allow_none: cb.addItem("None")
            cb.addItems(cols)
        self.cb_network_type.setCurrentText("Mixed"); self.btn_run.setVisible(False); self.btn_save.setVisible(False)
        self.setWindowTitle(f"Network Visualization App - {path}")

    def load_network(self):
        if self.df.empty:
            QMessageBox.warning(self, "Warning", "Load a network file first.")
            return
        # Clear previous graph and labels
        self.net_plot.clear()
        self.graph_item = pg.GraphItem()
        self.net_plot.addItem(self.graph_item)
        
        start, end, wt, direct, nt = (
            cb.currentText() for cb in [
                self.cb_start, self.cb_end, self.cb_weight, self.cb_direct, self.cb_network_type
            ]
        )
        if not start or not end or not nt:
            QMessageBox.warning(self, "Warning", "Select Start, End and Network Type.")
            return
        if nt == 'Mixed' and direct == 'None':
            QMessageBox.warning(self, "Missing Information", "For Mixed network type, please select a Direct column.")
            return
        if self.show_graph:
            edges = []
            genes = set()
            for _, r in self.df.iterrows():
                u = r[start]
                v = r[end]
                w = 1 if wt == 'None' else r[wt]
                genes.update([u, v])
                edges.append((u, v, w, r.get(direct, 1)))
            nodes = list(genes)
            id_map = {n: i for i, n in enumerate(nodes)}
            N = len(nodes)

            # Build graph for layout
            G = nx.DiGraph() if nt != 'Undirected' else nx.Graph()
            for u, v, w, d in edges:
                G.add_edge(u, v, weight=w)

            # Force-directed layout
            pos = nx.spring_layout(G, k=1/np.sqrt(N) * 1.2, iterations=100)

            # Prepare data for GraphItem
            pts = np.array([pos[n] for n in nodes])
            adj = np.array([[id_map[u], id_map[v]] for u, v, _, _ in edges])
            node_px = 30  # pixel size base d on N
        
            self.graph_item.setData(
                pos=pts,
                adj=adj,
                size=node_px,
                symbol='o',
                pxMode=True,
                pen=pg.mkPen(100, 100, 100, 200),
                brush=pg.mkBrush(80, 160, 255),
                edgePen=pg.mkPen(180, 180, 180, 100)
            )

            # Add labels inside nodes
            for n, (x, y) in zip(nodes, pts):
                label = pg.TextItem(
                    text=str(n),
                    anchor=(0.5, 0.5),  # center inside node
                    color=(0, 0, 0)  # white text contrasting node color
                )
                font_size = max(8, min(12, int(4000 / N)))
                label.setFont(QFont('Arial', font_size))
                label.setPos(x, y)
                self.net_plot.addItem(label)

        self.btn_run.setVisible(True)

        # Build node_id_to_gene and gene_to_node_id
        genes = set()
        for _, r in self.df.iterrows():
            genes.add(r[start])
            genes.add(r[end])
        self.node_id_to_gene = {}
        node_id = 0
        for gen in genes:
            self.node_id_to_gene[node_id] = gen
            node_id += 1
        self.N = len(self.node_id_to_gene)
        self.W = np.zeros((self.N, self.N))
        # Reverse mapping
        self.gene_to_node_id = {gene: idx for idx, gene in self.node_id_to_gene.items()}
        # Fill weight matrix
        for _, r in self.df.iterrows():
            st = self.gene_to_node_id[r[start]]
            fi = self.gene_to_node_id[r[end]]
            wt_val = 1 if wt == 'None' else r[wt]
            if nt == 'Directed':
                self.W[st, fi] = wt_val
            elif nt == 'Undirected':
                self.W[st, fi] = wt_val
                self.W[fi, st] = wt_val
            else:  # Mixed
                if r.get(direct, 1) == 1:
                    self.W[st, fi] = wt_val
                else:
                    self.W[st, fi] = wt_val
                    self.W[fi, st] = wt_val
        if nt == "Undirected":
            self.df.iloc[0]  # Lấy hàng thứ 4 (chỉ số bắt đầu từ 0)
            self.W[self.gene_to_node_id[r[start]], self.gene_to_node_id[r[end]]] -= 1e-5
        # Update mappings attributes
        # node_id_to_gene and gene_to_node_id already set

        self.btn_run.setVisible(True)

    def run_analysis(self):
        self.delta_x.clear(); self.delta_y.clear(); self.delta_curve.setData(self.delta_x, self.delta_y)
        model = inside_competitive_dynamics_model(
            self.N, self.W, W_att = self.b_param['W_att'], num_agent = self.b_param['num_agent'],
            alpha = self.b_param['alpha'],
            gamma = self.b_param['gamma'], 
            lamda = self.b_param['lamda'], 
            decay = 0,
            id_node_to_gene = self.node_id_to_gene, gene_to_node_id = self.gene_to_node_id,
        )
        self.thread = TrainingThread(model)
        self.thread.delta_signal.connect(self.update_delta)
        self.thread.finished_signal.connect(self.show_result)
        self.thread.start()

    def update_delta(self, delta):
        it = len(self.delta_x)
        self.delta_x.append(it); self.delta_y.append(delta)
        self.delta_curve.setData(self.delta_x, self.delta_y)
        # Scientific notation
        txt = f"{delta:.2e}"
        self.delta_label.setText(txt)
        self.delta_label.setPos(it, delta)

    def stop_training(self):
        if self.thread and self.thread.isRunning():
            self.thread.requestInterruption()
            self.statusBar().showMessage("Training stopped.", 2000)

    def show_result(self, df):
        self.result_df = df
        self.result_table.setModel(PandasModel(df))
        self.btn_save.setVisible(True)

    def save_result(self):
        if self.result_df is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Result", "", "Excel Files (*.xlsx)")
        if path:
            self.result_df.to_excel(path, index=False)
            QMessageBox.information(self, "Saved", "Results saved.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
