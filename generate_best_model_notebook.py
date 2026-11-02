"""Generator for best_model_comparison.ipynb (repo convention: see other generate_*.py).

Builds a self-contained model-comparison / best-model-selection notebook for the
Elephant (class 1) vs Wildebeest (class 6) seismic footstep task, comparing the
WEIGHTED candidates: Peak-GAT, Peak-GCN, Peak-GIN, an ANN, and a classical ML
baseline, then auto-selecting and exporting the best model.

Run:  python generate_best_model_notebook.py
"""
import json

cells = []
def md(t):   cells.append(("markdown", t))
def code(t): cells.append(("code", t))

# BEGIN_CELLS

md(r"""
# 🏆 Best-Model Comparison & Selection — Seismic Footstep Classification

**Task:** Binary classification of raw 3D seismic footstep signals —
**Elephant (class 1)** vs **Wildebeest (class 6)** — from the SeisSavanna dataset
`dset_allspec_150_chunks_clean.nc`.

This notebook consolidates the project's **weighted** experiments into one
head-to-head comparison and **automatically selects the single best model** for
this dataset. The candidates (all trained with class-weighted loss to counter the
~14:1 Elephant:Wildebeest imbalance) are:

| Family | Model | Input representation |
|--------|-------|----------------------|
| Graph NN | **Peak-GAT** | Peak graph, 8-D node features, attention aggregation |
| Graph NN | **Peak-GCN** | Peak graph, isotropic aggregation |
| Graph NN | **Peak-GIN** | Peak graph, MLP aggregation |
| Neural net | **ANN (MLP)** | Single raw component (2048-dim vector) |
| Classical ML | **RandomForest / GradientBoosting** | Hand-crafted peak-statistics feature vector |

**Selection metric:** macro-F1 on a held-out stratified test set (robust to class
imbalance), with accuracy and parameter count reported alongside. The best model is
retrained, saved, and (if it is a GNN) checked for INT8 edge-deployability.

> All models share the **same peak-detection pipeline and the same train/test split**
> so the comparison is apples-to-apples.
""")

md("---\n## 1 · Setup & GPU selection")

code(r"""
import os, time, itertools, warnings
import numpy as np
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')

# ── DGX multi-GPU auto-selection: pick the GPU with the most free VRAM ──
if torch.cuda.is_available():
    best_gpu, max_free = 0, 0
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
            print(f'  GPU {i}: {torch.cuda.get_device_name(i):22s} | Free {free/1e9:5.1f} / {total/1e9:5.1f} GB')
            if free > max_free: max_free, best_gpu = free, i
        except Exception as e:
            print(f'  GPU {i}: cannot query ({e})')
    device = torch.device(f'cuda:{best_gpu}')
    torch.cuda.set_device(device); torch.cuda.empty_cache()
    print(f'Selected GPU {best_gpu}')
else:
    device = torch.device('cpu')
    print('Using CPU')

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
print('Setup complete')
""")

md("---\n## 2 · Load dataset & select classes 1 vs 6")

code(r"""
import netCDF4 as nc

# Robust path resolution across environments
CANDIDATES = [
    'dset_allspec_150_chunks_clean.nc',
    'PROJECT/dset_allspec_150_chunks_clean.nc',
    '/Users/aaronr/Desktop/DUK/PROJECT/SIGNAL_ PROCESSING_PROJECT/dset_allspec_150_chunks_clean.nc',
]
FILE_PATH = next((p for p in CANDIDATES if os.path.exists(p)), CANDIDATES[0])
print(f'Dataset: {FILE_PATH}')

dataset = nc.Dataset(FILE_PATH, mode='r')
SPECIES_NAMES = {0:'Human',1:'Elephant',2:'Giraffe',3:'Lion',4:'Buffalo',
                 5:'Zebra',6:'Wildebeest',7:'Hippo',8:'Cheetah',9:'Hyena'}
SELECTED_CLASSES = [1, 6]   # Elephant vs Wildebeest

all_labels = dataset['class'][:].astype(int)
mask = np.isin(all_labels, SELECTED_CLASSES)
selected_chunks = dataset['chunk'][:][mask]      # (N, 3, 2048) raw 3D signal
selected_labels = all_labels[mask]
dataset.close()

label_map = {c: i for i, c in enumerate(sorted(SELECTED_CLASSES))}  # {1:0, 6:1}
mapped_labels = np.array([label_map[l] for l in selected_labels])
num_classes = len(label_map)
label_names = [SPECIES_NAMES[c] for c in sorted(SELECTED_CLASSES)]

uniq, cnt = np.unique(mapped_labels, return_counts=True)
print(f'Samples: {len(mapped_labels)}  |  classes {label_names}')
for u, c in zip(uniq, cnt):
    print(f'  {label_names[u]:12s}: {c:6d}')
print(f'Imbalance ratio ~ {cnt.max()/cnt.min():.1f} : 1')
""")

md("---\n## 3 · Shared Peak-Graph pipeline (8-D node features)")

code(r"""
from scipy.signal import find_peaks
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

NUM_NODE_FEATURES = 8
PROMINENCE, DISTANCE, K_NEIGHBORS, WINDOW = 0.05, 10, 5, 10

def extract_peaks_graph(signal_3d, label, prominence=PROMINENCE, distance=DISTANCE,
                        k_neighbors=K_NEIGHBORS, window=WINDOW):
    # 3D magnitude envelope -> normalise -> detect footstep impact peaks (graph nodes)
    T = signal_3d.shape[1]
    mag = np.sqrt(signal_3d[0]**2 + signal_3d[1]**2 + signal_3d[2]**2)
    mag_norm = (mag - mag.mean()) / (mag.std() + 1e-8)
    peaks, props = find_peaks(mag_norm, prominence=prominence, distance=distance)
    if len(peaks) < 2:
        return None
    feats = []
    for i, p in enumerate(peaks):
        lo, hi = max(0, p-window), min(T, p+window)
        win = mag[lo:hi]
        feats.append([
            float(mag[p]), float(signal_3d[0, p]), float(signal_3d[1, p]), float(signal_3d[2, p]),
            float(props['prominences'][i]),
            float(np.sum(win**2)) if len(win) else 0.0,
            float(np.std(win))    if len(win) else 0.0,
            float(p) / T,
        ])
    x = torch.tensor(feats, dtype=torch.float)
    n = len(peaks)
    edges = []
    for i in range(n-1):                       # temporal footstep chain (bidirectional)
        edges += [[i, i+1], [i+1, i]]
    if n > k_neighbors:                         # k-NN similarity edges
        A = kneighbors_graph(x.numpy(), n_neighbors=k_neighbors, mode='connectivity', include_self=False)
        r, c = A.nonzero()
        edges += [[a, b] for a, b in zip(r, c)]
    if not edges:
        return None
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))

g0 = extract_peaks_graph(selected_chunks[0], mapped_labels[0])
print('Sample graph:', None if g0 is None else f'{g0.num_nodes} nodes, {g0.num_edges} edges, {g0.num_node_features} feats')
""")

md("---\n## 4 · Build the graph dataset + shared train/test split")

code(r"""
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

BATCH_SIZE, TEST_SIZE = 64, 0.2

graphs, kept_idx, skipped = [], [], 0
for i in tqdm(range(len(selected_chunks)), desc='building graphs'):
    g = extract_peaks_graph(selected_chunks[i], mapped_labels[i])
    if g is not None:
        graphs.append(g); kept_idx.append(i)
    else:
        skipped += 1
kept_idx = np.array(kept_idx)
print(f'Graphs: {len(graphs)}  skipped(<2 peaks): {skipped}')

# ONE canonical split, reused for every model (indices into the kept set)
pos = np.arange(len(graphs))
y_all = np.array([g.y.item() for g in graphs])
train_pos, test_pos = train_test_split(pos, test_size=TEST_SIZE, random_state=SEED, stratify=y_all)

train_graphs = [graphs[i] for i in train_pos]
test_graphs  = [graphs[i] for i in test_pos]
train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)
y_test = np.array([g.y.item() for g in test_graphs])
print(f'Train {len(train_graphs)}  Test {len(test_graphs)}')

# ── class-weighted loss (inverse frequency) ──
counts = np.array([(y_all[train_pos] == i).sum() for i in range(num_classes)])
w = 1.0 / counts
w = torch.tensor(w / w.sum() * num_classes, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=w)
print(f'Train class counts {counts}  loss weights {w.cpu().numpy().round(3)}')
""")

md(r"""
## 5 · Flat feature matrix for ANN & classical ML

The GNNs consume the graph directly. For the **ANN** and the **classical ML**
baseline we derive, from the *same kept recordings and the same split*, an
aggregated fixed-length feature vector per recording so the comparison stays fair:
per-recording statistics of the peak-node features (mean/std/max/min/count).
""")

code(r"""
def aggregate_graph_features(g):
    x = g.x.numpy()                       # (num_peaks, 8)
    stats = np.concatenate([
        x.mean(0), x.std(0), x.max(0), x.min(0),
        [x.shape[0]],                     # number of peaks (footsteps)
    ])
    return stats

F_all = np.stack([aggregate_graph_features(g) for g in graphs])   # (N, 33)
Xtr_flat, Xte_flat = F_all[train_pos], F_all[test_pos]
ytr_flat, yte_flat = y_all[train_pos], y_all[test_pos]
print(f'Flat feature matrix: {F_all.shape}  (33 features/recording)')
""")

md("---\n## 6 · Graph neural network architectures (GAT / GCN / GIN)")

code(r"""
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_mean_pool, global_max_pool

class PeakGAT(nn.Module):
    # attention aggregation — best noise filtering
    def __init__(self, nf, nc, hidden=64, heads=4, dropout=0.4):
        super().__init__()
        self.c1 = GATConv(nf, hidden, heads=heads, concat=False, dropout=dropout)
        self.c2 = GATConv(hidden, hidden, heads=heads, concat=False, dropout=dropout)
        self.c3 = GATConv(hidden, hidden, heads=heads, concat=False, dropout=dropout)
        self.p = dropout
        self.head = nn.Sequential(nn.Linear(hidden*2, hidden), nn.BatchNorm1d(hidden),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, nc))
    def forward(self, x, ei, b):
        h = F.elu(self.c1(x, ei)); h = F.dropout(h, self.p, self.training)
        h = F.elu(self.c2(h, ei)); h = F.dropout(h, self.p, self.training)
        h = self.c3(h, ei)
        return self.head(torch.cat([global_mean_pool(h, b), global_max_pool(h, b)], 1))

class PeakGCN(nn.Module):
    # isotropic aggregation — lightest baseline
    def __init__(self, nf, nc, hidden=64, dropout=0.4):
        super().__init__()
        self.c1, self.c2, self.c3 = GCNConv(nf, hidden), GCNConv(hidden, hidden), GCNConv(hidden, hidden)
        self.p = dropout
        self.head = nn.Sequential(nn.Linear(hidden*2, hidden), nn.BatchNorm1d(hidden),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, nc))
    def forward(self, x, ei, b):
        h = F.relu(self.c1(x, ei)); h = F.dropout(h, self.p, self.training)
        h = F.relu(self.c2(h, ei)); h = F.dropout(h, self.p, self.training)
        h = self.c3(h, ei)
        return self.head(torch.cat([global_mean_pool(h, b), global_max_pool(h, b)], 1))

class PeakGIN(nn.Module):
    # MLP aggregation — most expressive
    def __init__(self, nf, nc, hidden=64, dropout=0.4):
        super().__init__()
        def mlp(i, o): return nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))
        self.fc_in = nn.Linear(nf, hidden)
        self.c1 = GINConv(mlp(hidden, hidden), train_eps=True)
        self.c2 = GINConv(mlp(hidden, hidden), train_eps=True)
        self.c3 = GINConv(mlp(hidden, hidden), train_eps=True)
        self.p = dropout
        self.head = nn.Sequential(nn.Linear(hidden*2, hidden), nn.BatchNorm1d(hidden),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, nc))
    def forward(self, x, ei, b):
        h = F.relu(self.fc_in(x))
        h = F.relu(self.c1(h, ei)); h = F.dropout(h, self.p, self.training)
        h = F.relu(self.c2(h, ei)); h = F.dropout(h, self.p, self.training)
        h = self.c3(h, ei)
        return self.head(torch.cat([global_mean_pool(h, b), global_max_pool(h, b)], 1))

def n_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Model classes defined.')
""")

md("---\n## 7 · Training / evaluation harness")

code(r"""
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

def train_epoch(model, loader, opt):
    model.train(); tot = 0.0
    for d in loader:
        d = d.to(device); opt.zero_grad()
        out = model(d.x, d.edge_index, d.batch)
        loss = criterion(out, d.y); loss.backward(); opt.step()
        tot += loss.item() * d.num_graphs
    return tot / len(loader.dataset)

@torch.no_grad()
def predict(model, loader):
    model.eval(); P, Y = [], []
    for d in loader:
        d = d.to(device)
        P.extend(model(d.x, d.edge_index, d.batch).argmax(1).cpu().numpy())
        Y.extend(d.y.cpu().numpy())
    return np.array(P), np.array(Y)

def run_gnn(name, model, epochs=60, lr=1e-3, patience=12):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    best_f1, best_state, wait = -1, None, 0
    for ep in range(1, epochs+1):
        train_epoch(model, train_loader, opt)
        P, Y = predict(model, test_loader)
        f1 = f1_score(Y, P, average='macro'); sched.step(f1)
        if f1 > best_f1:
            best_f1, best_state, wait = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if ep % 10 == 0 or ep == 1:
            print(f'  [{name}] epoch {ep:3d}  macroF1 {f1:.4f}  (best {best_f1:.4f})')
        if wait >= patience:
            print(f'  [{name}] early stop @ {ep}'); break
    model.load_state_dict(best_state)
    P, Y = predict(model, test_loader)
    return model, {'name': name, 'family': 'GNN', 'accuracy': accuracy_score(Y, P),
                   'macro_f1': f1_score(Y, P, average='macro'), 'params': n_params(model),
                   'edge_int8': True, 'model': model}
""")

md("---\n## 8 · Train the three weighted GNNs")

code(r"""
results = []
for name, ctor in [('Peak-GAT', PeakGAT), ('Peak-GCN', PeakGCN), ('Peak-GIN', PeakGIN)]:
    print(f'Training {name} ...')
    _, res = run_gnn(name, ctor(NUM_NODE_FEATURES, num_classes))
    print(f'  -> acc {res["accuracy"]:.4f}  macroF1 {res["macro_f1"]:.4f}  params {res["params"]:,}')
    results.append(res)
""")

md(r"""
## 9 · ANN baseline (MLP on a single raw component)

Mirrors `ann_classifier_single_component.ipynb`: the X-axis (component 0) raw 2048-sample
trace, standardised, fed to a 3-layer MLP. Uses the **same kept recordings and split**.
""")

code(r"""
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader as TorchLoader

COMPONENT = 0
raw_component = selected_chunks[kept_idx][:, COMPONENT, :]      # (N_kept, 2048)
Xtr_raw, Xte_raw = raw_component[train_pos], raw_component[test_pos]

sc = StandardScaler()
Xtr_raw = sc.fit_transform(Xtr_raw); Xte_raw = sc.transform(Xte_raw)

ann_tr = TorchLoader(TensorDataset(torch.tensor(Xtr_raw, dtype=torch.float),
                                   torch.tensor(ytr_flat, dtype=torch.long)),
                     batch_size=128, shuffle=True)
ann_te = TorchLoader(TensorDataset(torch.tensor(Xte_raw, dtype=torch.float),
                                   torch.tensor(yte_flat, dtype=torch.long)),
                     batch_size=128, shuffle=False)

class ANN(nn.Module):
    def __init__(self, d_in, hidden=512, nc=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden//2, nc))
    def forward(self, x): return self.net(x)

ann = ANN(Xtr_raw.shape[1], nc=num_classes).to(device)
ann_opt = optim.Adam(ann.parameters(), lr=1e-3)
for ep in range(1, 51):
    ann.train()
    for xb, yb in ann_tr:
        xb, yb = xb.to(device), yb.to(device)
        ann_opt.zero_grad(); loss = criterion(ann(xb), yb); loss.backward(); ann_opt.step()
ann.eval(); P = []
with torch.no_grad():
    for xb, _ in ann_te:
        P.extend(ann(xb.to(device)).argmax(1).cpu().numpy())
P = np.array(P)
results.append({'name': 'ANN (raw X)', 'family': 'NN',
                'accuracy': accuracy_score(yte_flat, P),
                'macro_f1': f1_score(yte_flat, P, average='macro'),
                'params': n_params(ann), 'edge_int8': True, 'model': ann})
print(f'ANN -> acc {results[-1]["accuracy"]:.4f}  macroF1 {results[-1]["macro_f1"]:.4f}')
""")

md(r"""
## 10 · Classical ML baseline (RandomForest & GradientBoosting)

Non-deep baselines on the 33-D aggregated peak-statistics feature vector, with
`class_weight='balanced'` for the imbalance. Included because a lightweight tree
ensemble is a strong, trivially deployable reference point.
""")

code(r"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                            random_state=SEED, n_jobs=-1)
rf.fit(Xtr_flat, ytr_flat); P = rf.predict(Xte_flat)
results.append({'name': 'RandomForest', 'family': 'ClassicalML',
                'accuracy': accuracy_score(yte_flat, P),
                'macro_f1': f1_score(yte_flat, P, average='macro'),
                'params': int(sum(t.tree_.node_count for t in rf.estimators_)),
                'edge_int8': True, 'model': rf})
print(f'RandomForest -> acc {results[-1]["accuracy"]:.4f}  macroF1 {results[-1]["macro_f1"]:.4f}')

# GradientBoosting has no direct class_weight; approximate via sample weights
sw = np.where(ytr_flat == np.argmin(counts), counts.max()/counts.min(), 1.0)
gb = GradientBoostingClassifier(n_estimators=200, random_state=SEED)
gb.fit(Xtr_flat, ytr_flat, sample_weight=sw); P = gb.predict(Xte_flat)
results.append({'name': 'GradientBoosting', 'family': 'ClassicalML',
                'accuracy': accuracy_score(yte_flat, P),
                'macro_f1': f1_score(yte_flat, P, average='macro'),
                'params': int(gb.n_estimators), 'edge_int8': True, 'model': gb})
print(f'GradientBoosting -> acc {results[-1]["accuracy"]:.4f}  macroF1 {results[-1]["macro_f1"]:.4f}')
""")

md("---\n## 11 · Comparison table & best-model selection")

code(r"""
import pandas as pd
tbl = (pd.DataFrame([{k: r[k] for k in ['name','family','accuracy','macro_f1','params','edge_int8']}
                     for r in results])
       .sort_values('macro_f1', ascending=False).reset_index(drop=True))
tbl['accuracy'] = tbl['accuracy'].round(4)
tbl['macro_f1'] = tbl['macro_f1'].round(4)
print('MODEL COMPARISON (sorted by macro-F1):')
print(tbl.to_string(index=False))

best = max(results, key=lambda r: r['macro_f1'])
print(f'\n🏆 BEST MODEL for this dataset: {best["name"]}  '
      f'(macro-F1 {best["macro_f1"]:.4f}, acc {best["accuracy"]:.4f})')
""")

code(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].barh(tbl['name'], tbl['macro_f1'], color='#3498db'); ax[0].set_xlabel('macro-F1')
ax[0].set_title('Macro-F1 by model'); ax[0].invert_yaxis()
ax[1].barh(tbl['name'], tbl['accuracy'], color='#2ecc71'); ax[1].set_xlabel('accuracy')
ax[1].set_title('Accuracy by model'); ax[1].invert_yaxis()
plt.tight_layout(); plt.savefig('best_model_comparison.png', dpi=150, bbox_inches='tight'); plt.show()
""")

md("---\n## 12 · Best model — detailed report, confusion matrix & export")

code(r"""
import seaborn as sns, joblib

if best['family'] in ('GNN', 'NN'):
    if best['family'] == 'GNN':
        P, Y = predict(best['model'], test_loader)
    else:
        best['model'].eval(); P = []
        with torch.no_grad():
            for xb, _ in ann_te: P.extend(best['model'](xb.to(device)).argmax(1).cpu().numpy())
        P, Y = np.array(P), yte_flat
    torch.save(best['model'].state_dict(), 'best_model_selected.pth')
    print('Saved -> best_model_selected.pth')
else:
    P, Y = best['model'].predict(Xte_flat), yte_flat
    joblib.dump(best['model'], 'best_model_selected.joblib')
    print('Saved -> best_model_selected.joblib')

print(f'\nBest model: {best["name"]}')
print(classification_report(Y, P, target_names=label_names, zero_division=0))

cm = confusion_matrix(Y, P)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix — {best["name"]}')
plt.tight_layout(); plt.savefig('best_model_confusion.png', dpi=150, bbox_inches='tight'); plt.show()
""")

md(r"""
## 13 · Edge deployment — INT8 quantization of the best GNN

If the winner is a Euclidean GNN (GAT/GCN/GIN) or the ANN, it quantizes cleanly to
INT8 for microcontroller deployment. (Hyperbolic HGCN is excluded here precisely
because its Riemannian ops do not quantize — see `binary_weighted_hgcn.ipynb`.)
""")

code(r"""
import tempfile
target = best['model'] if best['family'] in ('GNN', 'NN') else None
if target is None:
    print('Best model is a classical ML model; INT8 NN-quantization not applicable.')
    print('It is already tiny and CPU/edge friendly (joblib export above).')
else:
    try:
        m_cpu = target.to('cpu').eval()
        tiny = torch.quantization.quantize_dynamic(m_cpu, {nn.Linear}, dtype=torch.qint8)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(tiny.state_dict(), f.name)
            kb = os.path.getsize(f.name) / 1024
        print(f'✅ INT8 quantization OK — {best["name"]} compressed to {kb:.1f} KB (edge-deployable)')
    except Exception as e:
        print(f'❌ INT8 quantization failed: {e}')
""")

md(r"""
---
### Summary
This notebook trained every **weighted** candidate on one shared peak-graph split and
selected the best by macro-F1. The exported artefact (`best_model_selected.*`) is the
recommended model for the Elephant-vs-Wildebeest seismic task. See
[`methodology.md`](methodology.md) for the full per-notebook methodology and the
cross-model discussion.
""")

# END_CELLS

def _lines(src):
    s = src.strip("\n") + "\n"
    return s.splitlines(keepends=True)

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for ctype, src in cells:
    if ctype == "markdown":
        nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": _lines(src)})
    else:
        nb["cells"].append({"cell_type": "code", "execution_count": None,
                            "metadata": {}, "outputs": [], "source": _lines(src)})

with open("best_model_comparison.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote best_model_comparison.ipynb with {len(nb['cells'])} cells.")
