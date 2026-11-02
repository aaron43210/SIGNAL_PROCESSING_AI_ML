#!/usr/bin/env python3
"""Rigorous, leakage-aware benchmark of every model family on SeisSavanna
(Elephantidae vs Giraffidae), parallelised over all free GPUs.

Protocol (identical for every model)
  * one common cohort: binary windows with at least two detected peaks;
  * 5-fold cross-validation grouped by recording station (whole stations held out);
  * inside each outer training set, a station-grouped validation split used for every
    hyperparameter choice and for early stopping; each test fold is used once;
  * seeds 0, 1, 2; out-of-fold predictions saved for paired McNemar tests.

Parallelism
  The benchmark is hundreds of small, independent training runs, so instead of DDP (which
  splits one run across GPUs and synchronises every batch) each free GPU runs one or more
  worker processes that pull independent jobs from a file-based queue. Results are written
  per job, so the run can be stopped and restarted at any point.

Commands
  python rigorous_benchmark.py all        prepare data, run every phase, aggregate
  python rigorous_benchmark.py prepare    cache the binary windows, features and splits
  python rigorous_benchmark.py run        run the training phases on the free GPUs
  python rigorous_benchmark.py aggregate  tables, McNemar tests, edge profile, zip
  python rigorous_benchmark.py status     progress of every phase

  Useful options: --gpus 0,3,5  --max-gpus 4  --workers-per-gpu 2  --smoke  --data PATH
"""

import argparse
import copy
import hashlib
import io
import itertools
import json
import math
import os
import pickle
import platform
import random
import subprocess
import sys
import time
import traceback
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings('ignore')
for _m in ('torch_scatter', 'torch_sparse'):        # PyG runs without them; a mismatched
    sys.modules.setdefault(_m, None)                # binary can crash on import

# XGBoost before PyTorch: they bundle different OpenMP runtimes, and loading PyTorch's first
# can crash the process at the first XGBoost fit.
from xgboost import XGBClassifier                   # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- configuration
DEFAULTS = dict(
    classes=[1, 6],                 # 1 = Elephantidae, 6 = Giraffidae (positive class)
    seeds=[0, 1, 2],
    n_outer=5, n_inner=4,
    detector=dict(prominence=0.05, distance=10, window=10, k=5),
    grids=dict(
        gat=dict(hidden=[32, 64], heads=[2, 4], lr=[1e-3, 5e-4], scale=['raw', 'std']),
        gcn=dict(encoding=['raw', 'stats'], lr=[1e-3, 5e-4]),
        kan=dict(width=[[32, 16], [64, 32]], lr=[1e-3, 5e-4]),
        ann=dict(component=['x', 'xyz'], hidden=[256, 512], lr=[1e-3, 1e-4]),
        mlp=dict(hidden=[64, 128], lr=[1e-3, 5e-4]),
        cnn=dict(channels=[16, 32], lr=[1e-3, 5e-4]),
    ),
    budget=dict(gat_search_epochs=20, gat_search_patience=5, gat_epochs=100, gat_patience=15,
                gat_batch=64, nn_epochs=60, nn_patience=10, nn_batch=128, weight_decay=1e-4,
                gat_dropout=0.4, ablation_seeds=[0]),
    ml_grids={
        'Logistic reg.': [dict(C=c) for c in (0.1, 1.0, 10.0)],
        'SVM (RBF)': [dict(C=c) for c in (1.0, 10.0)],
        'k-NN': [dict(n_neighbors=k, weights=w) for k in (7, 15) for w in ('uniform', 'distance')],
        'Random forest': [dict(max_depth=d) for d in (None, 20)],
        'Grad. boosting': [dict(learning_rate=0.1)],
        'XGBoost': [dict(max_depth=d, learning_rate=l) for d in (3, 5, 7) for l in (0.05, 0.1)],
    },
    detector_sweep=dict(prominence=[0.05, 0.1, 0.25, 0.5], distance=[10, 40]),
    attention=dict(max_graphs=2000, fractions=[0.1, 0.25, 0.5]),
    edge=dict(n_windows=200),
    run=dict(ml=True, ann=True, mlp=True, cnn=True, kan=True, gcn=True, gat=True, gat_ablation=True,
             full_cohort=True, detector_sweep=True, attention=True),
    smoke=False, smoke_n=3000,
)

# label -> (kind, cohort) for the neural models trained on arrays
NN_MODELS = [('ANN', 'ann', 'peak'), ('MLP (28 features)', 'mlp', 'peak'), ('1D-CNN', 'cnn', 'peak'),
             ('KAN', 'kan', 'peak'), ('GCN', 'gcn', 'peak'), ('ANN [all windows]', 'ann', 'all')]
SCALED = {'Logistic reg.', 'SVM (RBF)', 'k-NN'}
DETERMINISTIC = {'Logistic reg.', 'SVM (RBF)', 'k-NN'}


def smoke_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.update(smoke=True, seeds=[0], n_outer=3)
    cfg['grids'] = dict(gat=dict(hidden=[32], heads=[2], lr=[1e-3], scale=['raw', 'std']),
                        gcn=dict(encoding=['raw', 'stats'], lr=[1e-3]), kan=dict(width=[[32, 16]], lr=[1e-3]),
                        ann=dict(component=['x'], hidden=[256], lr=[1e-3]), mlp=dict(hidden=[64], lr=[1e-3]),
                        cnn=dict(channels=[16], lr=[1e-3]))
    cfg['budget'].update(gat_search_epochs=2, gat_search_patience=1, gat_epochs=2, gat_patience=1,
                         nn_epochs=2, nn_patience=1)
    cfg['ml_grids'] = {k: v[:1] for k, v in cfg['ml_grids'].items()}
    cfg['detector_sweep'] = dict(prominence=[0.05, 0.25], distance=[10])
    cfg['attention']['max_graphs'] = 60
    cfg['edge']['n_windows'] = 10
    return cfg


CFG = None          # loaded configuration
OUT = None          # output directory (Path)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def slug(name):
    return ''.join(c if c.isalnum() else '_' for c in name).strip('_')


def cfg_id(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def grid_configs(kind):
    g = CFG['grids'][kind]
    return [dict(zip(g, v)) for v in itertools.product(*g.values())]


def load_config(out):
    global CFG, OUT
    OUT = Path(out)
    CFG = json.load(open(OUT / 'config.json'))


# ----------------------------------------------------------------------------- data state
Y = STATION = CHUNKS = FEATS = ML28 = POOLED = N_PEAKS = None
COHORT = ALLWIN = None
SPLITS = {}         # 'peak' / 'all' -> list of folds (dicts of index arrays)


def load_state(need_chunks=True):
    global Y, STATION, CHUNKS, FEATS, ML28, POOLED, N_PEAKS, COHORT, ALLWIN, SPLITS
    meta = np.load(OUT / 'cache' / 'meta.npz')
    Y, STATION = meta['Y'].astype(np.int64), meta['STATION']
    if need_chunks:
        CHUNKS = np.load(OUT / 'cache' / 'chunks.npy', mmap_mode='r')     # shared page cache
    FEATS = pickle.load(open(OUT / 'cache' / 'features_main.pkl', 'rb'))
    N_PEAKS = np.array([r['n'] for r in FEATS])
    ML28 = np.stack([r['ml28'] for r in FEATS])
    POOLED = np.full((len(Y), 24), np.nan, dtype=np.float32)
    for i, r in enumerate(FEATS):
        if r['n'] >= 2:
            POOLED[i] = r['pooled']
    COHORT = np.flatnonzero(N_PEAKS >= 2)
    ALLWIN = np.arange(len(Y))
    sp = json.load(open(OUT / 'splits.json'))
    SPLITS = {k: [{kk: (np.asarray(vv) if isinstance(vv, list) else vv) for kk, vv in d.items()} for d in v]
              for k, v in sp.items()}


# ----------------------------------------------------------------------------- features
from scipy.signal import find_peaks                                     # noqa: E402
from sklearn.neighbors import kneighbors_graph                          # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold                # noqa: E402


def ml28_features(sig):
    """28 engineered features, as in ml_classifier.ipynb."""
    feats = []
    for ax in range(3):
        s = sig[ax]
        zcr = np.mean(np.abs(np.diff(np.sign(s)))) / 2.0
        feats += [s.mean(), s.std(), s.max(), np.abs(s).max(), float(np.sum(s ** 2)), zcr]
    mag = np.sqrt(sig[0] ** 2 + sig[1] ** 2 + sig[2] ** 2)
    feats += [mag.mean(), mag.std(), mag.max(), float(np.sum(mag ** 2))]
    mag_norm = (mag - mag.mean()) / (mag.std() + 1e-8)
    peaks, props = find_peaks(mag_norm, prominence=0.05, distance=10)
    if len(peaks) > 0:
        heights, proms = mag[peaks], props['prominences']
        gaps = np.diff(peaks) if len(peaks) > 1 else np.array([0.0])
        feats += [len(peaks), heights.mean(), heights.std(), proms.mean(), gaps.mean(), heights.max()]
    else:
        feats += [0, 0, 0, 0, 0, 0]
    return np.nan_to_num(np.array(feats, dtype=np.float32))


def peak_nodes(sig, prominence, distance, window):
    """8 node features per detected peak, as in class1_and_class6_weighted.ipynb."""
    T = sig.shape[1]
    mag = np.sqrt(sig[0] ** 2 + sig[1] ** 2 + sig[2] ** 2)
    mag_norm = (mag - mag.mean()) / (mag.std() + 1e-8)
    peaks, props = find_peaks(mag_norm, prominence=prominence, distance=distance)
    rows = []
    for i, p in enumerate(peaks):
        lo, hi = max(0, p - window), min(T, p + window)
        win = mag[lo:hi]
        rows.append([float(mag[p]), float(sig[0, p]), float(sig[1, p]), float(sig[2, p]),
                     float(props['prominences'][i]),
                     float(np.sum(win ** 2)) if len(win) else 0.0,
                     float(np.std(win)) if len(win) else 0.0, float(p) / T])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 8)


def knn_edges_sklearn(x, k):
    """Directed k-NN edges exactly as in the original graph builder (row -> neighbour)."""
    if len(x) <= k:
        return np.zeros((2, 0), dtype=np.int64)
    r, c = kneighbors_graph(x, n_neighbors=k, mode='connectivity', include_self=False).nonzero()
    return np.vstack([r, c]).astype(np.int64)


def process_batch(sigs, det, with_ml, with_graph):
    out = []
    for sig in sigs:
        x = peak_nodes(np.asarray(sig), det['prominence'], det['distance'], det['window'])
        rec = dict(n=len(x))
        if len(x) >= 2:
            rec['pooled'] = np.concatenate([x.mean(0), x.max(0), x.std(0)]).astype(np.float32)
            if with_graph:
                rec['x'] = x
                rec['knn'] = knn_edges_sklearn(x, det['k'])
        if with_ml:
            rec['ml28'] = ml28_features(np.asarray(sig))
        out.append(rec)
    return out


def extract_features(chunks, det, with_ml, with_graph, n_jobs):
    from joblib import Parallel, delayed
    bs = 500
    parts = Parallel(n_jobs=n_jobs)(delayed(process_batch)(np.asarray(chunks[a:a + bs]), det, with_ml, with_graph)
                                    for a in range(0, len(chunks), bs))
    return [r for p in parts for r in p]


def make_random_splits(idx, y, n_outer, n_inner, seed=42):
    """Control only: window-level stratified splits that ignore stations, as in the original
    notebooks. Used to show how much of a score comes from the split itself."""
    from sklearn.model_selection import StratifiedKFold
    folds = []
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    for f, (tr_pos, te_pos) in enumerate(outer.split(idx, y[idx])):
        tr, te = idx[tr_pos], idx[te_pos]
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 100 + f)
        itr, iva = next(inner.split(tr, y[tr]))
        folds.append(dict(fold=f, train=np.sort(tr[itr]), val=np.sort(tr[iva]), test=np.sort(te)))
    return folds


def make_splits(idx, y, station, n_outer, n_inner, seed=42):
    """Outer folds hold out whole stations; inside each outer training set one of n_inner
    station groups becomes the validation split."""
    folds = []
    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    for f, (tr_pos, te_pos) in enumerate(outer.split(idx, y[idx], station[idx])):
        tr, te = idx[tr_pos], idx[te_pos]
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + 100 + f)
        itr, iva = next(inner.split(tr, y[tr], station[tr]))
        d = dict(fold=f, train=np.sort(tr[itr]), val=np.sort(tr[iva]), test=np.sort(te))
        for part in ('train', 'val', 'test'):
            assert len(np.unique(y[d[part]])) == 2, f'fold {f}: {part} lacks a class'
        assert not set(station[d['test']].tolist()) & set(station[tr].tolist()), 'station leak'
        folds.append(d)
    return folds


def find_dataset(name='dset_allspec_150_chunks_clean.nc'):
    """Look next to the script, in the usual project folders, then up to 4 levels below home."""
    home, user = Path.home(), os.environ.get('USER', '')
    roots = [Path.cwd(), HERE, HERE.parent, HERE.parent / 'PROJECT', home, home / 'PROJECT',
             Path('/jupyter') / user, Path('/jupyter') / user / 'PROJECT']
    for r in roots:
        if (r / name).exists():
            return str(r / name)
    for top in (home, Path('/jupyter') / user):
        if not top.exists():
            continue
        base = len(top.parts)
        for dirpath, dirnames, filenames in os.walk(top):
            if name in filenames:
                return os.path.join(dirpath, name)
            if len(Path(dirpath).parts) - base >= 4:
                dirnames[:] = []                    # do not descend further
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in ('site-packages', 'node_modules')]
    return None


def prepare(args):
    import netCDF4 as nc
    (OUT / 'cache').mkdir(parents=True, exist_ok=True)
    meta_p, chunks_p = OUT / 'cache' / 'meta.npz', OUT / 'cache' / 'chunks.npy'
    if not (meta_p.exists() and chunks_p.exists()):
        path = args.data or find_dataset()
        assert path and os.path.exists(path), 'dataset not found; pass --data /path/to/dset_allspec_150_chunks_clean.nc'
        log('reading', path)
        ds = nc.Dataset(path)
        ds.set_auto_mask(False)
        labels = np.asarray(ds['class'][:]).astype(int)
        station = np.asarray(ds['seis'][:]).astype(int)
        distance = np.asarray(ds['distance'][:]).astype(np.float32)
        sel = np.flatnonzero(np.isin(labels, CFG['classes']))
        if CFG['smoke']:
            sel = np.sort(np.random.default_rng(0).choice(sel, CFG['smoke_n'], replace=False))
        allc = ds['chunk'][:]
        np.save(chunks_p, np.ascontiguousarray(allc[sel], dtype=np.float32))
        del allc
        ds.close()
        np.savez(meta_p, Y=(labels[sel] == CFG['classes'][1]).astype(np.int64), STATION=station[sel],
                 DISTANCE=distance[sel], SEL=sel)
        log(f'cached {len(sel):,} windows')
    meta = np.load(meta_p)
    y, station = meta['Y'], meta['STATION']
    chunks = np.load(chunks_p, mmap_mode='r')
    fp = OUT / 'cache' / 'features_main.pkl'
    if not fp.exists():
        t0 = time.time()
        feats = extract_features(chunks, CFG['detector'], True, True, args.cpu_jobs)
        pickle.dump(feats, open(fp, 'wb'), protocol=4)
        log(f'features for {len(feats):,} windows in {time.time() - t0:.0f}s')
    feats = pickle.load(open(fp, 'rb'))
    n = np.array([r['n'] for r in feats])
    cohort = np.flatnonzero(n >= 2)
    sp = dict(peak=make_splits(cohort, y, station, CFG['n_outer'], CFG['n_inner']),
              all=make_splits(np.arange(len(y)), y, station, CFG['n_outer'], CFG['n_inner']),
              random_peak=make_random_splits(cohort, y, CFG['n_outer'], CFG['n_inner']),
              random_all=make_random_splits(np.arange(len(y)), y, CFG['n_outer'], CFG['n_inner']))
    json.dump({k: [{kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in d.items()} for d in v]
               for k, v in sp.items()}, open(OUT / 'splits.json', 'w'))
    rows = []
    for name, folds in sp.items():
        for d in folds:
            for part in ('train', 'val', 'test'):
                ii = d[part]
                rows.append(dict(cohort=name, fold=d['fold'], part=part, windows=len(ii), giraffe=int(y[ii].sum()),
                                 stations=' '.join(map(str, sorted(set(station[ii].tolist()))))))
    (OUT / 'tables').mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'tables' / 'splits.csv', index=False)
    log(f'cohort with >=2 peaks: {len(cohort):,} of {len(y):,} windows; '
        f'kept Elephantidae {np.mean(n[y == 0] >= 2):.3f}, Giraffidae {np.mean(n[y == 1] >= 2):.3f}')
    log('splits written; prepare done')


# ----------------------------------------------------------------------------- metrics and saving
import torch                                                            # noqa: E402
import torch.nn as nn                                                   # noqa: E402
import torch.nn.functional as F                                         # noqa: E402
from torch.utils.data import DataLoader as TorchLoader, TensorDataset   # noqa: E402
from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef, f1_score, accuracy_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score, average_precision_score,
                             confusion_matrix)

DEVICE = torch.device('cpu')


def metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(n=int(len(y)), accuracy=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred),
               macro_f1=f1_score(y, pred, average='macro', zero_division=0), mcc=matthews_corrcoef(y, pred),
               prec_g=precision_score(y, pred, zero_division=0), rec_g=recall_score(y, pred, zero_division=0),
               f1_g=f1_score(y, pred, zero_division=0), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    try:
        out['auroc'], out['auprc'] = roc_auc_score(y, prob), average_precision_score(y, prob)
    except ValueError:
        out['auroc'] = out['auprc'] = float('nan')
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def run_path(model, fold, seed):
    return OUT / 'runs' / slug(model) / f'f{fold}_s{seed}.json'


def is_done(model, fold, seed):
    return run_path(model, fold, seed).exists()


def save_run(model, fold, seed, idx, prob, info):
    p = run_path(model, fold, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    (OUT / 'preds').mkdir(exist_ok=True)
    m = metrics(Y[idx], prob)
    np.savez_compressed(OUT / 'preds' / f'{slug(model)}_f{fold}_s{seed}.npz', idx=idx, y=Y[idx],
                        prob=prob.astype(np.float32))
    tmp = p.with_suffix('.tmp')
    json.dump(dict(model=model, fold=fold, seed=seed, test=m, **info), open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, p)                              # written last and atomically: marks the run done
    log(f'{model} fold {fold} seed {seed}: BA {m["bal_acc"]:.3f} MCC {m["mcc"]:.3f} Giraffe R {m["rec_g"]:.3f}')
    return m


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def class_weights(ytr):
    c = np.bincount(ytr, minlength=2).astype(float)
    return torch.tensor(len(ytr) / (2 * c), dtype=torch.float32)


def weighted_nll(y, prob, w):
    p = np.clip(np.where(y == 1, prob, 1 - prob), 1e-7, 1)
    return float(np.sum(-w[y] * np.log(p)) / np.sum(w[y]))


# ----------------------------------------------------------------------------- models
class ANN(nn.Module):               # architecture of ann_classifier_weighted.ipynb (also the feature MLP)
    def __init__(self, d_in, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(hidden // 2, 2))

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """Compact 1D-CNN on the raw (3, 2048) window; about 24k (c=16) to 94k (c=32) parameters."""
    def __init__(self, c=16, n_cls=2):
        super().__init__()

        def block(ci, co, stride, pool):
            layers = [nn.Conv1d(ci, co, 9, stride=stride, padding=4), nn.BatchNorm1d(co), nn.ReLU()]
            return layers + ([nn.MaxPool1d(pool)] if pool else [])

        self.features = nn.Sequential(*block(3, c, 2, 4), *block(c, 2 * c, 1, 4), *block(2 * c, 4 * c, 1, None))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(8 * c, n_cls))

    def forward(self, x):
        h = self.features(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


class KANLinear(nn.Module):         # verbatim from kan_weighted.ipynb
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1,
                 scale_base=1.0, scale_spline=1.0, base_activation=nn.SiLU, grid_range=(-1, 1)):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
        self.register_buffer('grid', grid.expand(in_features, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.base_activation = base_activation()
        self.scale_noise, self.scale_base, self.scale_spline = scale_noise, scale_base, scale_spline
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                     * self.scale_noise / self.grid_size)
            x_fit = self.grid.T[self.spline_order:-self.spline_order]
            self.spline_weight.data.copy_(self.scale_spline * self._curve2coeff(x_fit, noise))

    def _b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)])
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def _curve2coeff(self, x, y):
        A = self._b_splines(x).transpose(0, 1)
        return torch.linalg.lstsq(A, y.transpose(0, 1)).solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x):
        base = F.linear(self.base_activation(x), self.base_weight)
        spline = F.linear(self._b_splines(x).reshape(x.size(0), -1),
                          self.scaled_spline_weight.reshape(self.out_features, -1))
        return base + spline


class KAN(nn.Module):
    def __init__(self, sizes, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([KANLinear(a, b) for a, b in zip(sizes[:-1], sizes[1:])])
        self.dropout = dropout

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class DenseGCN(nn.Module):
    """Two-layer GCN on the fixed 64-node segment chain (+ self-loops) of gcn(final).ipynb:
    H' = A_hat H W + b with a dense normalised adjacency."""
    def __init__(self, f_in, h1=64, h2=32, n_cls=2, n_nodes=64, bn_dropout=None):
        super().__init__()
        A = torch.eye(n_nodes)
        i = torch.arange(n_nodes - 1)
        A[i, i + 1] = 1
        A[i + 1, i] = 1
        dinv = A.sum(1).rsqrt()
        self.register_buffer('A', dinv[:, None] * A * dinv[None, :])
        self.w1, self.b1 = nn.Linear(f_in, h1, bias=False), nn.Parameter(torch.zeros(h1))
        self.w2, self.b2 = nn.Linear(h1, h2, bias=False), nn.Parameter(torch.zeros(h2))
        self.out = nn.Linear(h2, n_cls)
        self.bn = None
        if bn_dropout is not None:
            self.bn = nn.ModuleList([nn.BatchNorm1d(h1), nn.BatchNorm1d(h2)])
            self.p = bn_dropout

    def _layer(self, x, w, b, k):
        h = torch.einsum('ij,bjf->bif', self.A, w(x)) + b
        if self.bn is not None:
            h = self.bn[k](h.transpose(1, 2)).transpose(1, 2)
        h = F.relu(h)
        if self.bn is not None:
            h = F.dropout(h, self.p, self.training)
        return h

    def forward(self, x):
        return self.out(self._layer(self._layer(x, self.w1, self.b1, 0), self.w2, self.b2, 1).mean(1))


from torch_geometric.data import Data                                   # noqa: E402
from torch_geometric.loader import DataLoader as GraphLoader            # noqa: E402
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool  # noqa: E402
from torch_geometric.utils import subgraph                              # noqa: E402


class PeakGAT(nn.Module):           # architecture of class1_and_class6_weighted.ipynb
    def __init__(self, f_in=8, hidden=64, heads=4, dropout=0.4, n_cls=2):
        super().__init__()
        self.gat1 = GATConv(f_in, hidden, heads=heads, concat=False)
        self.gat2 = GATConv(hidden, hidden, heads=heads, concat=False)
        self.gat3 = GATConv(hidden, hidden, heads=heads, concat=False)
        self.dropout = dropout
        self.classifier = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, n_cls))

    def forward(self, x, edge_index, batch, return_att=False):
        atts = []

        def conv(layer, h):
            if return_att:
                h, (ei, a) = layer(h, edge_index, return_attention_weights=True)
                atts.append((ei, a))
                return h
            return layer(h, edge_index)

        h = F.dropout(F.elu(conv(self.gat1, x)), self.dropout, self.training)
        h = F.dropout(F.elu(conv(self.gat2, h)), self.dropout, self.training)
        h = conv(self.gat3, h)
        logits = self.classifier(torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], 1))
        return (logits, atts) if return_att else logits


# ----------------------------------------------------------------------------- array-model inputs and training
THREADS = 1


def chunk_rows(bi):
    return torch.from_numpy(np.ascontiguousarray(CHUNKS[np.asarray(bi)], dtype=np.float32))


def batched_stats(fn, idx, bs=2048):
    """Per-feature mean and std of fn(windows) over idx, accumulated in float64 batches."""
    s = ss = None
    n = 0
    for a in range(0, len(idx), bs):
        x = fn(torch.as_tensor(idx[a:a + bs])).double()
        x = x.reshape(-1, x.shape[-1])
        s = x.sum(0) if s is None else s + x.sum(0)
        ss = (x ** 2).sum(0) if ss is None else ss + (x ** 2).sum(0)
        n += x.shape[0]
    mu = s / n
    return mu.float(), torch.sqrt(torch.clamp(ss / n - mu ** 2, min=1e-12)).float()


def segments(sig_std):              # (B, 3, 2048) -> (B, 64 nodes, 3 components, 32 samples)
    return sig_std.reshape(sig_std.shape[0], 3, 64, 32).permute(0, 2, 1, 3)


def segment_stats(seg):             # (B, 64, 3, 32) -> (B, 64, 21)
    rms = torch.sqrt((seg ** 2).mean(-1))
    st = torch.stack([seg.mean(-1), seg.std(-1), seg.amin(-1), seg.amax(-1), rms, (seg ** 2).sum(-1),
                      seg.amax(-1) - seg.amin(-1)], dim=-1)
    return st.reshape(seg.shape[0], 64, 21)


def nn_inputs(kind, cfg, tr):
    """(get_x, d_in) for an array model; all standardisation uses training windows only."""
    dev = DEVICE
    if kind == 'ann':
        raw = (lambda bi: chunk_rows(bi)[:, 0, :]) if cfg['component'] == 'x' else \
              (lambda bi: chunk_rows(bi).reshape(len(bi), -1))
        mu, sd = batched_stats(raw, tr)
        mu, sd = mu.to(dev), sd.to(dev)
        return (lambda bi: (raw(bi).to(dev) - mu) / sd), int(mu.numel())
    if kind == 'mlp':
        M = torch.from_numpy(ML28)
        mu, sd = M[tr].mean(0).to(dev), M[tr].std(0).clamp_min(1e-6).to(dev)
        return (lambda bi: (M[bi].to(dev) - mu) / sd), 28
    if kind == 'kan':
        P = torch.from_numpy(POOLED)
        mu, sd = P[tr].mean(0).to(dev), P[tr].std(0).clamp_min(1e-6).to(dev)
        return (lambda bi: (P[bi].to(dev) - mu) / sd), 24
    if kind in ('cnn', 'gcn'):
        cmu, csd = batched_stats(lambda bi: chunk_rows(bi).permute(0, 2, 1), tr, bs=512)
        cmu, csd = cmu.to(dev), csd.to(dev)
        std_sig = lambda bi: (chunk_rows(bi).to(dev) - cmu[None, :, None]) / (csd[None, :, None] + 1e-12)
        if kind == 'cnn':
            return std_sig, 3
        if cfg['encoding'] == 'raw':
            return (lambda bi: segments(std_sig(bi)).reshape(len(bi), 64, 96)), 96
        stat = lambda bi: segment_stats(segments(std_sig(bi)))
        mu, sd = batched_stats(lambda bi: stat(bi).cpu(), tr, bs=512)
        mu, sd = mu.to(dev), sd.to(dev)
        return (lambda bi: (stat(bi) - mu) / sd), 21
    raise ValueError(kind)


def build_nn(kind, cfg, d_in):
    if kind in ('ann', 'mlp'):
        return lambda: ANN(d_in, cfg['hidden'])
    if kind == 'cnn':
        return lambda: CNN1D(cfg['channels'])
    if kind == 'kan':
        return lambda: KAN([d_in, *cfg['width'], 2])
    if kind == 'gcn':
        return lambda: DenseGCN(d_in, bn_dropout=0.3 if cfg['encoding'] == 'stats' else None)
    raise ValueError(kind)


@torch.no_grad()
def predict_nn(model, get_x, idx, batch=1024):
    model.eval()
    out = []
    for a in range(0, len(idx), batch):
        out.append(F.softmax(model(get_x(torch.as_tensor(idx[a:a + batch]))), 1)[:, 1].float().cpu())
    return torch.cat(out).numpy()


def fit_nn(build, get_x, tr, va, lr, epochs, patience, batch, seed, grad_clip=None):
    """Weighted cross-entropy, Adam, early stopping on validation balanced accuracy."""
    set_seed(seed)
    model = build().to(DEVICE)
    w = class_weights(Y[tr])
    crit = nn.CrossEntropyLoss(weight=w.to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=CFG['budget']['weight_decay'])
    loader = TorchLoader(TensorDataset(torch.as_tensor(tr)), batch_size=batch, shuffle=True,
                         generator=torch.Generator().manual_seed(seed), drop_last=len(tr) > batch)
    y_t = torch.as_tensor(Y)
    best_ba, best_state, best_ep, bad, hist = -1.0, None, 0, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for (bi,) in loader:
            loss = crit(model(get_x(bi)), y_t[bi].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += loss.item() * len(bi)
            n += len(bi)
        pv = predict_nn(model, get_x, va)
        ba = balanced_accuracy_score(Y[va], pv >= 0.5)
        hist.append(dict(epoch=ep, train_loss=tot / max(n, 1), val_loss=weighted_nll(Y[va], pv, w.numpy()), val_ba=ba))
        if ba > best_ba + 1e-6:
            best_ba, best_state, best_ep, bad = ba, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_ba, best_ep, hist


# ----------------------------------------------------------------------------- peak graphs
def chain_edges(n):
    i = np.arange(n - 1)
    return np.vstack([np.concatenate([i, i + 1]), np.concatenate([i + 1, i])]).astype(np.int64)


def knn_edges_np(x, k):
    n = len(x)
    if n <= k:
        return np.zeros((2, 0), dtype=np.int64)
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nb = np.argpartition(d, k, axis=1)[:, :k]
    return np.vstack([np.repeat(np.arange(n), k), nb.reshape(-1)]).astype(np.int64)


def node_stats(tr):
    X = np.concatenate([FEATS[i]['x'] for i in tr])
    return X.mean(0), X.std(0) + 1e-8


def make_graph(i, variant='full', stats=None, seed=0):
    x = FEATS[i]['x']
    n = len(x)
    if stats is not None:
        x = (x - stats[0]) / stats[1]
    k = CFG['detector']['k']
    knn = FEATS[i]['knn'] if stats is None else knn_edges_np(x, k)
    if variant == 'full':
        ei = np.hstack([chain_edges(n), knn])
    elif variant == 'temporal':
        ei = chain_edges(n)
    elif variant == 'knn':
        ei = knn
    elif variant == 'random':                       # same edge count as the full graph
        m = 2 * (n - 1) + knn.shape[1]
        rng = np.random.default_rng(seed * 1_000_003 + int(i))
        src = rng.integers(0, n, m)
        ei = np.vstack([src, (src + rng.integers(1, n, m)) % n])
    elif variant == 'none':
        ei = np.zeros((2, 0), dtype=np.int64)
    else:
        raise ValueError(variant)
    return Data(x=torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                edge_index=torch.from_numpy(np.ascontiguousarray(ei, dtype=np.int64)),
                y=torch.tensor([int(Y[i])]), idx=torch.tensor([int(i)]))


def fold_graphs(d, variant='full', scale='raw', seed=0):
    stats = node_stats(d['train']) if scale == 'std' else None
    return {p: [make_graph(i, variant, stats, seed) for i in d[p]] for p in ('train', 'val', 'test')}


@torch.no_grad()
def predict_gat(model, graphs, batch=512):
    model.eval()
    out = []
    for b in GraphLoader(graphs, batch_size=batch, shuffle=False):
        b = b.to(DEVICE)
        out.append(F.softmax(model(b.x, b.edge_index, b.batch), 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def fit_gat(cfg, g_tr, g_va, y_tr, y_va, epochs, patience, seed):
    set_seed(seed)
    b = CFG['budget']
    model = PeakGAT(hidden=cfg['hidden'], heads=cfg['heads'], dropout=b['gat_dropout']).to(DEVICE)
    w = class_weights(y_tr)
    crit = nn.CrossEntropyLoss(weight=w.to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=b['weight_decay'])
    loader = GraphLoader(g_tr, batch_size=b['gat_batch'], shuffle=True,
                         generator=torch.Generator().manual_seed(seed), drop_last=len(g_tr) > b['gat_batch'])
    best_ba, best_state, best_ep, bad, hist = -1.0, None, 0, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for bt in loader:
            bt = bt.to(DEVICE)
            loss = crit(model(bt.x, bt.edge_index, bt.batch), bt.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * bt.num_graphs
            n += bt.num_graphs
        pv = predict_gat(model, g_va)
        ba = balanced_accuracy_score(y_va, pv >= 0.5)
        hist.append(dict(epoch=ep, train_loss=tot / max(n, 1), val_loss=weighted_nll(y_va, pv, w.numpy()), val_ba=ba))
        if ba > best_ba + 1e-6:
            best_ba, best_state, best_ep, bad = ba, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_ba, best_ep, hist


# ----------------------------------------------------------------------------- classical ML
def make_estimator(name, params, seed, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    if name == 'Logistic reg.':
        return LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed, **params)
    if name == 'SVM (RBF)':
        return SVC(kernel='rbf', class_weight='balanced', gamma='scale', random_state=seed, **params)
    if name == 'k-NN':
        return KNeighborsClassifier(n_jobs=THREADS, **params)
    if name == 'Random forest':
        return RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=seed,
                                      n_jobs=THREADS, **params)
    if name == 'Grad. boosting':
        return GradientBoostingClassifier(n_estimators=200, random_state=seed, **params)
    if name == 'XGBoost':
        spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
        return XGBClassifier(n_estimators=300, scale_pos_weight=spw, eval_metric='logloss', tree_method='hist',
                             random_state=seed, n_jobs=THREADS, **params)
    raise ValueError(name)


def fit_predict(est_name, params, seed, Xtr, ytr, Xte):
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight
    est = make_estimator(est_name, params, seed, ytr)
    if est_name in SCALED:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if est_name == 'Grad. boosting':
        est.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
    else:
        est.fit(Xtr, ytr)
    if est_name == 'SVM (RBF)':                      # logistic of the margin: 0.5 <=> decision 0
        prob = 1.0 / (1.0 + np.exp(-est.decision_function(Xte)))
    else:
        prob = est.predict_proba(Xte)[:, 1]
    return est, prob


# ----------------------------------------------------------------------------- jobs
def grid_dir(label, fold):
    p = OUT / 'grid' / slug(label) / f'f{fold}'
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_grid(label, fold, cfg, row):
    p = grid_dir(label, fold) / f'{cfg_id(cfg)}.json'
    json.dump(dict(config=cfg, **row), open(p.with_suffix('.tmp'), 'w'), default=str)
    os.replace(p.with_suffix('.tmp'), p)


def best_config(label, fold, kind):
    rows = [json.load(open(p)) for p in grid_dir(label, fold).glob('*.json')]
    expected = len(grid_configs(kind))
    if len(rows) < expected:
        raise RuntimeError(f'{label} fold {fold}: {len(rows)}/{expected} grid configurations finished')
    order = {cfg_id(c): k for k, c in enumerate(grid_configs(kind))}
    rows.sort(key=lambda r: (-r['val_ba'], order.get(cfg_id(r['config']), 0)))
    return rows[0]['config'], rows[0]['val_ba']


def job_ml(j):
    label, est, fold = j['label'], j['est'], j['fold']
    X = ML28 if j['features'] == 'ml28' else POOLED
    d = SPLITS[j['cohort']][fold]
    tr, va, te = d['train'], d['val'], d['test']
    seeds = [0] if est in DETERMINISTIC else CFG['seeds']
    if all(is_done(label, fold, s) for s in seeds):
        return
    rows = []
    for params in CFG['ml_grids'][est]:
        t0 = time.time()
        _, pv = fit_predict(est, params, 0, X[tr], Y[tr], X[va])
        rows.append(dict(params=json.dumps(params), val_ba=balanced_accuracy_score(Y[va], pv >= 0.5),
                         val_mcc=matthews_corrcoef(Y[va], pv >= 0.5), seconds=time.time() - t0))
    gd = grid_dir(label, fold)
    pd.DataFrame(rows).to_csv(gd / 'grid.csv', index=False)
    best = max(rows, key=lambda r: r['val_ba'])
    params = json.loads(best['params'])
    trva = np.concatenate([tr, va])                 # refit the chosen setting on train + validation
    for s in seeds:
        if is_done(label, fold, s):
            continue
        t0 = time.time()
        model, prob = fit_predict(est, params, s, X[trva], Y[trva], X[te])
        if fold == 0 and s == 0:
            (OUT / 'models').mkdir(exist_ok=True)
            pickle.dump(model, open(OUT / 'models' / f'{slug(label)}_f0_s0.pkl', 'wb'))
        save_run(label, fold, s, te, prob, dict(params=params, val_ba=best['val_ba'], family='ML',
                                                features=j['features'], fit_seconds=time.time() - t0))


def job_nn_search(j):
    d = SPLITS[j['cohort']][j['fold']]
    cfg, kind = j['config'], j['kind']
    get_x, d_in = nn_inputs(kind, cfg, d['train'])
    b = CFG['budget']
    t0 = time.time()
    _, vba, ep, _ = fit_nn(build_nn(kind, cfg, d_in), get_x, d['train'], d['val'], cfg['lr'], b['nn_epochs'],
                           b['nn_patience'], b['nn_batch'], 0, 1.0 if kind == 'kan' else None)
    write_grid(j['label'], j['fold'], cfg, dict(val_ba=vba, best_epoch=ep, seconds=time.time() - t0))


def job_nn_final(j):
    label, kind, fold, s = j['label'], j['kind'], j['fold'], j['seed']
    if is_done(label, fold, s):
        return
    cfg, vba = best_config(label, fold, kind)
    d = SPLITS[j['cohort']][fold]
    get_x, d_in = nn_inputs(kind, cfg, d['train'])
    b = CFG['budget']
    t0 = time.time()
    model, val_ba, ep, hist = fit_nn(build_nn(kind, cfg, d_in), get_x, d['train'], d['val'], cfg['lr'],
                                     b['nn_epochs'], b['nn_patience'], b['nn_batch'], s,
                                     1.0 if kind == 'kan' else None)
    prob = predict_nn(model, get_x, d['test'])
    (OUT / 'models').mkdir(exist_ok=True)
    torch.save(model.state_dict(), OUT / 'models' / f'{slug(label)}_f{fold}_s{s}.pt')
    json.dump(dict(config=cfg, d_in=d_in, kind=kind), open(OUT / 'models' / f'{slug(label)}_f{fold}_config.json', 'w'))
    save_run(label, fold, s, d['test'], prob, dict(config=cfg, val_ba=val_ba, grid_val_ba=vba, best_epoch=ep,
                                                   history=hist, family=kind.upper(), fit_seconds=time.time() - t0,
                                                   params=sum(p.numel() for p in model.parameters())))


def job_gat_search(j):
    d = SPLITS['peak'][j['fold']]
    cfg, b = j['config'], CFG['budget']
    g = fold_graphs(d, 'full', cfg['scale'])
    t0 = time.time()
    _, vba, ep, _ = fit_gat(cfg, g['train'], g['val'], Y[d['train']], Y[d['val']],
                            b['gat_search_epochs'], b['gat_search_patience'], 0)
    write_grid('GAT', j['fold'], cfg, dict(val_ba=vba, best_epoch=ep, seconds=time.time() - t0))


def job_gat_final(j):
    label, fold, s = j['label'], j['fold'], j['seed']
    if is_done(label, fold, s):
        return
    if 'fixed_config' in j:                         # control: the original notebook's configuration
        cfg, gvba = j['fixed_config'], None
    else:
        cfg, gvba = best_config('GAT', fold, 'gat')
    variant = j.get('variant', 'full')
    if j.get('scale_flip'):
        cfg = dict(cfg, scale='std' if cfg['scale'] == 'raw' else 'raw')
    d, b = SPLITS[j.get('cohort', 'peak')][fold], CFG['budget']
    g = fold_graphs(d, variant, cfg['scale'], seed=s)
    t0 = time.time()
    model, vba, ep, hist = fit_gat(cfg, g['train'], g['val'], Y[d['train']], Y[d['val']],
                                   b['gat_epochs'], b['gat_patience'], s)
    prob = predict_gat(model, g['test'])
    if label == 'GAT':
        (OUT / 'models').mkdir(exist_ok=True)
        torch.save(model.state_dict(), OUT / 'models' / f'GAT_f{fold}_s{s}.pt')
        json.dump(dict(config=cfg), open(OUT / 'models' / f'GAT_f{fold}_config.json', 'w'))
    save_run(label, fold, s, d['test'], prob, dict(config=cfg, variant=variant, val_ba=vba, grid_val_ba=gvba,
                                                   best_epoch=ep, history=hist, family='GAT',
                                                   fit_seconds=time.time() - t0,
                                                   params=sum(p.numel() for p in model.parameters())))


@torch.no_grad()
def node_importance(model, g):
    """Attention each node sends to other nodes, averaged over heads, summed over layers."""
    model.eval()
    b = next(iter(GraphLoader([g], batch_size=1))).to(DEVICE)
    _, atts = model(b.x, b.edge_index, b.batch, return_att=True)
    imp = torch.zeros(g.num_nodes, device=DEVICE)
    for ei, a in atts:
        keep = ei[0] != ei[1]
        imp.index_add_(0, ei[0][keep], a[keep].mean(1))
    return imp.cpu().numpy()


def delete_nodes(g, drop):
    keep = torch.ones(g.num_nodes, dtype=torch.bool)
    keep[torch.as_tensor(drop, dtype=torch.long)] = False
    ei, _ = subgraph(keep, g.edge_index, relabel_nodes=True, num_nodes=g.num_nodes)
    return Data(x=g.x[keep], edge_index=ei, y=g.y, idx=g.idx)


def job_attention(j):
    fold = j['fold']
    cfg = json.load(open(OUT / 'models' / f'GAT_f{fold}_config.json'))['config']
    model = PeakGAT(hidden=cfg['hidden'], heads=cfg['heads'], dropout=CFG['budget']['gat_dropout']).to(DEVICE)
    model.load_state_dict(torch.load(OUT / 'models' / f'GAT_f{fold}_s0.pt', map_location=DEVICE))
    d = SPLITS['peak'][fold]
    rng = np.random.default_rng(fold)
    te = np.sort(rng.choice(d['test'], min(CFG['attention']['max_graphs'], len(d['test'])), replace=False))
    stats = node_stats(d['train']) if cfg['scale'] == 'std' else None
    graphs = [make_graph(i, 'full', stats) for i in te]
    y = Y[te]
    p0 = predict_gat(model, graphs)
    pt0 = np.where(y == 1, p0, 1 - p0)
    imps = [node_importance(model, g) for g in graphs]
    rows = []
    for frac in CFG['attention']['fractions']:
        top, rnd = [], []
        for g, imp in zip(graphs, imps):
            n = g.num_nodes
            k = min(max(1, int(round(frac * n))), n - 1)
            top.append(delete_nodes(g, np.argsort(-imp)[:k]))
            rnd.append(delete_nodes(g, rng.choice(n, k, replace=False)))
        for method, gs in (('top attention', top), ('random', rnd)):
            p = predict_gat(model, gs)
            pt = np.where(y == 1, p, 1 - p)
            rows.append(dict(fold=fold, fraction=frac, method=method, graphs=len(gs),
                             mean_drop_true_prob=float(np.mean(pt0 - pt)),
                             flip_rate=float(np.mean((p >= 0.5) != (p0 >= 0.5))),
                             bal_acc=balanced_accuracy_score(y, p >= 0.5),
                             bal_acc_intact=balanced_accuracy_score(y, p0 >= 0.5)))
    (OUT / 'tables' / 'attention').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'tables' / 'attention' / f'f{fold}.csv', index=False)


def job_detector(j):
    det = dict(CFG['detector'], prominence=j['prominence'], distance=j['distance'])
    feats = extract_features(CHUNKS, det, False, False, THREADS)
    n = np.array([r['n'] for r in feats])
    keep = n >= 2
    row = dict(prominence=det['prominence'], distance_samples=det['distance'], distance_ms=det['distance'] * 5,
               kept_elephant=float(keep[Y == 0].mean()), kept_giraffe=float(keep[Y == 1].mean()),
               nodes_mean=float(n[keep].mean()) if keep.any() else 0.0,
               nodes_median=float(np.median(n[keep])) if keep.any() else 0.0,
               peaks_per_second=float(n[keep].mean() / 10.24) if keep.any() else 0.0)
    try:
        coh = np.flatnonzero(keep)
        X = np.full((len(Y), 24), np.nan, dtype=np.float32)
        for i in coh:
            X[i] = feats[i]['pooled']
        probs, idxs = [], []
        for d in make_splits(coh, Y, STATION, CFG['n_outer'], CFG['n_inner']):
            trva = np.concatenate([d['train'], d['val']])
            _, p = fit_predict('XGBoost', dict(max_depth=5, learning_rate=0.1), 0, X[trva], Y[trva], X[d['test']])
            probs.append(p)
            idxs.append(d['test'])
        m = metrics(Y[np.concatenate(idxs)], np.concatenate(probs))
        row.update(cohort=int(len(coh)), xgb_pooled_bal_acc=m['bal_acc'], xgb_pooled_mcc=m['mcc'])
    except AssertionError as e:
        row.update(note=str(e))
    (OUT / 'tables' / 'detector').mkdir(parents=True, exist_ok=True)
    json.dump(row, open(OUT / 'tables' / 'detector' / f"p{det['prominence']}_d{det['distance']}.json", 'w'))


JOB_FUNCS = dict(ml=job_ml, nn_search=job_nn_search, nn_final=job_nn_final, gat_search=job_gat_search,
                 gat_final=job_gat_final, attention=job_attention, detector=job_detector)


def dry_run_job(j):
    """Scheduler test only: no training; writes the minimum outputs later phases depend on."""
    time.sleep(0.05)
    if j['type'] in ('nn_search', 'gat_search'):
        write_grid(j['label'], j['fold'], j['config'], dict(val_ba=random.random(), best_epoch=1, seconds=0.0))
    if j['type'] == 'gat_final' and j['label'] == 'GAT':
        (OUT / 'models').mkdir(exist_ok=True)
        (OUT / 'models' / f"GAT_f{j['fold']}_s{j['seed']}.pt").touch()


# ----------------------------------------------------------------------------- planning
PHASES = ['1_search', '2_final', '3_extras', '4_control']


def plan(phase):
    r, folds, jobs = CFG['run'], range(CFG['n_outer']), []
    nn_on = [(lab, k, c) for lab, k, c in NN_MODELS if r.get(k) and (c == 'peak' or r['full_cohort'])]
    if phase == '1_search':
        if r['gat']:
            jobs += [dict(type='gat_search', label='GAT', fold=f, config=c, prio=0)
                     for f in folds for c in grid_configs('gat')]
        for lab, k, c in nn_on:
            jobs += [dict(type='nn_search', label=lab, kind=k, cohort=c, fold=f, config=cf,
                          prio=1 if k in ('cnn', 'ann') else 3) for f in folds for cf in grid_configs(k)]
        if r['ml']:
            for est in CFG['ml_grids']:
                jobs += [dict(type='ml', label=est, est=est, features='ml28', cohort='peak', fold=f,
                              prio=1 if est == 'SVM (RBF)' else 2) for f in folds]
            jobs += [dict(type='ml', label='XGBoost (pooled peaks)', est='XGBoost', features='pooled',
                          cohort='peak', fold=f, prio=2) for f in folds]
            if r['full_cohort']:
                jobs += [dict(type='ml', label='XGBoost [all windows]', est='XGBoost', features='ml28',
                              cohort='all', fold=f, prio=2) for f in folds]
    elif phase == '2_final':
        if r['gat']:
            jobs += [dict(type='gat_final', label='GAT', fold=f, seed=s, prio=0) for f in folds for s in CFG['seeds']]
            if r['gat_ablation']:
                for f in folds:
                    for s in CFG['budget']['ablation_seeds']:
                        jobs += [dict(type='gat_final', label=f'GAT edges={v}', variant=v, fold=f, seed=s, prio=1)
                                 for v in ('temporal', 'knn', 'random', 'none')]
                        jobs.append(dict(type='gat_final', label='GAT nodes=other scaling', scale_flip=True,
                                         fold=f, seed=s, prio=1))
        for lab, k, c in nn_on:
            jobs += [dict(type='nn_final', label=lab, kind=k, cohort=c, fold=f, seed=s, prio=2)
                     for f in folds for s in CFG['seeds']]
    elif phase == '3_extras':
        if r['attention'] and r['gat']:
            jobs += [dict(type='attention', label='GAT attention', fold=f, prio=0) for f in folds
                     if (OUT / 'models' / f'GAT_f{f}_s0.pt').exists()]
        if r['detector_sweep']:
            jobs += [dict(type='detector', label='detector', prominence=p, distance=d, prio=1)
                     for p in CFG['detector_sweep']['prominence'] for d in CFG['detector_sweep']['distance']]
    elif phase == '4_control' and r.get('control', True):
        # Same code, but window-level random splits as in the original notebooks: isolates how much
        # of the original scores came from the split (leakage between near-duplicate windows).
        if r['gat']:
            orig = dict(hidden=64, heads=4, lr=1e-3, scale='raw')
            jobs += [dict(type='gat_final', label='GAT [random split]', cohort='random_peak', fixed_config=orig,
                          fold=f, seed=s, prio=0) for f in folds for s in CFG['seeds']]
        if r['ml']:
            for est in ('XGBoost', 'Logistic reg.'):
                jobs += [dict(type='ml', label=f'{est} [random split]', est=est, features='ml28',
                              cohort='random_all', fold=f, prio=1) for f in folds]
    return jobs


def job_id(j):
    key = f"{j['prio']}_{j['type']}__{slug(j['label'])}"
    if 'fold' in j:
        key += f"__f{j['fold']}"
    if 'config' in j:
        key += '__' + cfg_id(j['config'])
    if 'seed' in j:
        key += f"__s{j['seed']}"
    if j['type'] == 'detector':
        key += f"__p{j['prominence']}_d{j['distance']}"
    return key


# ----------------------------------------------------------------------------- queue and workers
def qdir(phase, sub):
    p = OUT / 'queue' / phase / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def enqueue(phase, jobs):
    pend, claimed, done, failed = (qdir(phase, s) for s in ('pending', 'claimed', 'done', 'failed'))
    for c in claimed.glob('*.json'):                # claims left by an interrupted run
        name = c.name.split('__', 1)[1]
        if (done / (name[:-5] + '.ok')).exists():
            c.unlink()
        else:
            os.replace(c, pend / name)
    for f in failed.glob('*.txt'):                  # failed jobs are retried on every run
        f.unlink()
    for j in jobs:
        jid = job_id(j)
        if not (done / f'{jid}.ok').exists() and not (pend / f'{jid}.json').exists():
            json.dump(j, open(pend / f'{jid}.json', 'w'))


def worker(args):
    global DEVICE, THREADS
    load_config(args.out)
    THREADS = args.threads
    torch.set_num_threads(THREADS)
    DEVICE = torch.device('cuda:0') if args.gpu != 'cpu' and torch.cuda.is_available() else torch.device('cpu')
    log(f'worker {args.wid} on {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"}), '
        f'{THREADS} CPU threads')
    load_state(need_chunks=True)
    pend, claimed_dir, done, failed = (qdir(args.phase, s) for s in ('pending', 'claimed', 'done', 'failed'))
    while True:
        claimed = None
        for p in sorted(pend.glob('*.json')):
            target = claimed_dir / f'{args.wid}__{p.name}'
            try:
                os.rename(p, target)                # atomic: exactly one worker wins
                claimed = target
                break
            except OSError:
                continue
        if claimed is None:
            break
        j = json.load(open(claimed))
        jid = claimed.name.split('__', 1)[1][:-5]
        t0 = time.time()
        log('start', jid)
        try:
            (dry_run_job if args.dry_run else JOB_FUNCS[j['type']])(j)
            (done / f'{jid}.ok').write_text(f'{time.time() - t0:.1f}')
            log('done', jid, f'{time.time() - t0:.0f}s')
        except Exception:
            (failed / f'{jid}.txt').write_text(traceback.format_exc())
            log('FAILED', jid)
            traceback.print_exc()
        finally:
            claimed.unlink(missing_ok=True)
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
    log(f'worker {args.wid}: no jobs left')


def detect_gpus(args):
    if args.gpus:
        return [int(x) for x in args.gpus.split(',') if x.strip()]
    gpus = []
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
        for line in out.strip().splitlines():
            idx, used, util = (int(float(v)) for v in line.split(','))
            free = used <= args.max_used_mb and util <= args.max_util
            log(f'GPU {idx}: {used} MiB used, {util}% utilisation -> {"free" if free else "busy, skipped"}')
            if free:
                gpus.append(idx)
    except Exception as e:
        log('nvidia-smi unavailable:', e)
    return gpus[:args.max_gpus] if args.max_gpus else gpus


def run_phase(phase, args, slots, threads):
    jobs = plan(phase)
    enqueue(phase, jobs)
    pend, done, failed = qdir(phase, 'pending'), qdir(phase, 'done'), qdir(phase, 'failed')
    ids = [job_id(j) for j in jobs]
    n_done0 = sum((done / f'{i}.ok').exists() for i in ids)
    if not any(pend.glob('*.json')):
        log(f'phase {phase}: all {len(ids)} jobs already done')
        return
    log(f'phase {phase}: {len(ids) - n_done0} of {len(ids)} jobs to run on {len(slots)} workers')
    (OUT / 'logs').mkdir(exist_ok=True)
    procs = []
    for g, k in slots:
        wid = f'g{g}w{k}'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if g == 'cpu' else str(g), OMP_NUM_THREADS=str(threads),
                   MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--out', str(OUT), '--phase', phase,
               '--wid', wid, '--gpu', str(g), '--threads', str(threads)] + (['--dry-run'] if args.dry_run else [])
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(OUT / 'logs' / f'{phase}_{wid}.log', 'a'),
                                      stderr=subprocess.STDOUT))
        time.sleep(1 if args.dry_run else 5)        # stagger start-up (each worker loads the cache)
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(args.poll)
        n_done = sum((done / f'{i}.ok').exists() for i in ids)
        n_fail = len(list(failed.glob('*.txt')))
        rate = (n_done - n_done0) / max(time.time() - t0, 1)
        left = len(ids) - n_done - n_fail
        eta = f'{left / rate / 3600:.1f} h' if rate > 0 else '?'
        log(f'phase {phase}: {n_done}/{len(ids)} done, {n_fail} failed, ETA {eta}')
    n_done = sum((done / f'{i}.ok').exists() for i in ids)
    fails = sorted(failed.glob('*.txt'))
    log(f'phase {phase} finished: {n_done}/{len(ids)} done, {len(fails)} failed '
        f'({(time.time() - t0) / 3600:.2f} h)')
    for f in fails:
        log('  failed:', f.stem, '->', f.read_text().strip().splitlines()[-1])


def cmd_run(args):
    gpus = detect_gpus(args)
    if gpus:
        slots = [(g, k) for g in gpus for k in range(args.workers_per_gpu)]
        log(f'using GPUs {gpus} with {args.workers_per_gpu} worker(s) each')
    else:
        slots = [('cpu', k) for k in range(args.cpu_workers)]
        log(f'no free GPU: running {args.cpu_workers} CPU workers')
    threads = max(1, (os.cpu_count() or 2) // len(slots))
    for phase in PHASES:
        run_phase(phase, args, slots, threads)


def cmd_status(args):
    for phase in PHASES:
        c = {s: len(list(qdir(phase, s).glob('*'))) for s in ('pending', 'claimed', 'done', 'failed')}
        print(f'{phase:10s} pending {c["pending"]:5d}  running {c["claimed"]:4d}  done {c["done"]:5d}  failed {c["failed"]:4d}')
        for f in sorted(qdir(phase, 'failed').glob('*.txt')):
            print('   failed:', f.stem, '->', f.read_text().strip().splitlines()[-1])


# ----------------------------------------------------------------------------- aggregation
def load_preds(model, seed):
    parts = sorted((OUT / 'preds').glob(f'{slug(model)}_f*_s{seed}.npz'))
    if not parts:
        return None
    z = [np.load(p) for p in parts]
    return (np.concatenate([a['idx'] for a in z]), np.concatenate([a['y'] for a in z]),
            np.concatenate([a['prob'] for a in z]), len(parts))


def nbytes_state(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def int8_bytes(model):
    try:
        q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).cpu(), {nn.Linear}, dtype=torch.qint8)
        return nbytes_state(q)
    except Exception:
        return None


def median_ms(fn, items, reps=3):
    ts = []
    for it in items:
        best = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(it)
            best.append(time.perf_counter() - t0)
        ts.append(np.median(best))
    return 1000 * float(np.median(ts)), 1000 * float(np.percentile(ts, 95))


def edge_profile():
    """Exact parameter counts and sizes; single-thread host-CPU latency as a relative proxy."""
    global DEVICE
    DEVICE = torch.device('cpu')
    torch.set_num_threads(1)
    d = SPLITS['peak'][0]
    win = np.random.default_rng(0).choice(d['test'], min(CFG['edge']['n_windows'], len(d['test'])), replace=False)
    det, rows = CFG['detector'], []
    mp = OUT / 'models' / 'GAT_f0_s0.pt'
    if mp.exists():
        cfg = json.load(open(OUT / 'models' / 'GAT_f0_config.json'))['config']
        m = PeakGAT(hidden=cfg['hidden'], heads=cfg['heads'], dropout=CFG['budget']['gat_dropout'])
        m.load_state_dict(torch.load(mp, map_location='cpu'))
        m.eval()
        stats = node_stats(d['train']) if cfg['scale'] == 'std' else None
        pre = median_ms(lambda i: knn_edges_sklearn(peak_nodes(np.asarray(CHUNKS[i]), det['prominence'],
                                                               det['distance'], det['window']), det['k']), win)
        gs = [make_graph(i, 'full', stats) for i in win]
        with torch.no_grad():
            inf = median_ms(lambda g: m(g.x, g.edge_index, torch.zeros(g.num_nodes, dtype=torch.long)), gs)
        rows.append(dict(model='GAT', params=sum(p.numel() for p in m.parameters()), fp32_bytes=nbytes_state(m),
                         int8_bytes=int8_bytes(m), pre_ms=pre[0], pre_ms_p95=pre[1], inf_ms=inf[0], inf_ms_p95=inf[1]))
    for label, kind, cohort in NN_MODELS:
        mp = OUT / 'models' / f'{slug(label)}_f0_s0.pt'
        if cohort != 'peak' or not mp.exists():
            continue
        meta = json.load(open(OUT / 'models' / f'{slug(label)}_f0_config.json'))
        m = build_nn(kind, meta['config'], meta['d_in'])()
        m.load_state_dict(torch.load(mp, map_location='cpu'))
        m.eval()
        get_x, _ = nn_inputs(kind, meta['config'], d['train'])
        if kind == 'mlp':
            pre = median_ms(lambda i: ml28_features(np.asarray(CHUNKS[i])), win)
        elif kind == 'kan':
            pre = median_ms(lambda i: peak_nodes(np.asarray(CHUNKS[i]), det['prominence'], det['distance'],
                                                 det['window']), win)
        else:
            pre = median_ms(lambda i: get_x(torch.as_tensor([i])), win)
        with torch.no_grad():
            inf = median_ms(lambda i: m(get_x(torch.as_tensor([i]))), win)
        rows.append(dict(model=label, params=sum(p.numel() for p in m.parameters()), fp32_bytes=nbytes_state(m),
                         int8_bytes=int8_bytes(m), pre_ms=pre[0], pre_ms_p95=pre[1], inf_ms=inf[0], inf_ms_p95=inf[1]))
    for label in CFG['ml_grids']:
        mp = OUT / 'models' / f'{slug(label)}_f0_s0.pkl'
        if not mp.exists():
            continue
        est = pickle.load(open(mp, 'rb'))
        row = dict(model=label, pickle_bytes=mp.stat().st_size)
        if label == 'XGBoost':
            df = est.get_booster().trees_to_dataframe()
            splits, leaves = int((df.Feature != 'Leaf').sum()), int((df.Feature == 'Leaf').sum())
            row.update(xgb_raw_bytes=len(est.get_booster().save_raw('ubj')), xgb_splits=splits, xgb_leaves=leaves,
                       xgb_c_array_bytes=splits * 9 + leaves * 4)
        pre = median_ms(lambda i: ml28_features(np.asarray(CHUNKS[i])), win)
        row.update(pre_ms=pre[0], pre_ms_p95=pre[1])
        if label not in SCALED:
            inf = median_ms(lambda i: est.predict(ML28[i:i + 1]), win)
            row.update(inf_ms=inf[0], inf_ms_p95=inf[1])
        rows.append(row)
    return pd.DataFrame(rows)


def cmd_aggregate(args):
    from scipy.stats import binomtest
    load_state(need_chunks=True)
    T = OUT / 'tables'
    T.mkdir(exist_ok=True)
    recs = []
    for p in (OUT / 'runs').glob('*/*.json'):
        r = json.load(open(p))
        recs.append(dict(model=r['model'], fold=r['fold'], seed=r['seed'], **r['test']))
    runs = pd.DataFrame(recs)
    if runs.empty:
        log('no finished runs yet')
        return
    summary = []
    for model in sorted(runs.model.unique()):
        per_seed = []
        for s in sorted(runs[runs.model == model].seed.unique()):
            lp = load_preds(model, s)
            if lp is not None:
                per_seed.append(dict(metrics(lp[1], lp[2]), n_folds=lp[3]))
        ps = pd.DataFrame(per_seed)
        fold_ba = runs[runs.model == model].groupby('fold').bal_acc.mean()
        row = dict(model=model, seeds=len(ps), folds=int(ps.n_folds.min()), windows=int(ps.n.iloc[0]))
        for k in ('bal_acc', 'mcc', 'macro_f1', 'accuracy', 'prec_g', 'rec_g', 'f1_g', 'auroc', 'auprc'):
            row[k] = ps[k].mean()
            row[k + '_sd_seeds'] = ps[k].std(ddof=1) if len(ps) > 1 else 0.0
        row['bal_acc_fold_mean'] = fold_ba.mean()
        row['bal_acc_fold_sd'] = fold_ba.std(ddof=1) if len(fold_ba) > 1 else 0.0
        summary.append(row)
    summary = pd.DataFrame(summary).sort_values('bal_acc', ascending=False)
    summary.to_csv(T / 'summary_by_model.csv', index=False)
    print(summary[['model', 'folds', 'seeds', 'windows', 'bal_acc', 'bal_acc_sd_seeds', 'bal_acc_fold_sd',
                   'mcc', 'macro_f1', 'rec_g', 'auroc']].round(3).to_string(index=False))

    primary = ['GAT', 'GCN', 'KAN', 'ANN', 'MLP (28 features)', '1D-CNN', 'Logistic reg.', 'SVM (RBF)', 'k-NN',
               'Random forest', 'Grad. boosting', 'XGBoost', 'XGBoost (pooled peaks)']
    preds0 = {}
    for m in primary:
        lp = load_preds(m, 0)
        if lp is not None:
            o = np.argsort(lp[0])
            preds0[m] = (lp[0][o], lp[1][o], lp[2][o] >= 0.5)
    tests = []
    for a, b in itertools.combinations(preds0, 2):
        (ia, ya, pa), (ib, yb, pb) = preds0[a], preds0[b]
        if not np.array_equal(ia, ib):
            continue
        ca, cb = pa == ya, pb == yb
        n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
        tests.append(dict(model_a=a, model_b=b, a_right_b_wrong=n01, a_wrong_b_right=n10,
                          p_value=binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0,
                          bal_acc_a=balanced_accuracy_score(ya, pa), bal_acc_b=balanced_accuracy_score(yb, pb)))
    mc = pd.DataFrame(tests)
    if len(mc):
        order, adj, running = np.argsort(mc.p_value.values), np.empty(len(mc)), 0.0
        for rank, i in enumerate(order):            # Holm step-down
            running = max(running, min(1.0, (len(mc) - rank) * mc.p_value.values[i]))
            adj[i] = running
        mc['p_holm'] = adj
        mc.to_csv(T / 'mcnemar_pairwise.csv', index=False)

    grids = []
    for p in (OUT / 'grid').glob('*/f*/*.json'):
        r = json.load(open(p))
        grids.append(dict(model=p.parent.parent.name, fold=int(p.parent.name[1:]), config=json.dumps(r['config']),
                          scale=r['config'].get('scale'), val_ba=r['val_ba'], best_epoch=r.get('best_epoch'),
                          seconds=r.get('seconds')))
    grid_df = pd.DataFrame(grids)
    if len(grid_df):
        grid_df.to_csv(T / 'grid_search_all.csv', index=False)
    att = [pd.read_csv(p) for p in (T / 'attention').glob('f*.csv')] if (T / 'attention').exists() else []
    if att:
        pd.concat(att).to_csv(T / 'attention_faithfulness.csv', index=False)
    det = [json.load(open(p)) for p in (T / 'detector').glob('*.json')] if (T / 'detector').exists() else []
    if det:
        pd.DataFrame(det).to_csv(T / 'detector_sweep.csv', index=False)
    try:
        edge = edge_profile()
        edge.to_csv(T / 'edge_profile.csv', index=False)
        print(edge.round(3).to_string(index=False))
    except Exception:
        log('edge profile failed:\n' + traceback.format_exc())

    def table(name):
        p = T / name
        return pd.read_csv(p).to_dict(orient='records') if p.exists() else None

    env = dict(python=platform.python_version(), torch=torch.__version__, host=platform.node(),
               cpu_count=os.cpu_count(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    try:
        import sklearn, scipy, torch_geometric, xgboost
        env.update(sklearn=sklearn.__version__, scipy=scipy.__version__, torch_geometric=torch_geometric.__version__,
                   xgboost=xgboost.__version__, numpy=np.__version__, pandas=pd.__version__)
    except Exception:
        pass
    paper = dict(environment=env, config=CFG,
                 cohort=dict(binary_windows=int(len(Y)), peak_cohort=int(len(COHORT)),
                             kept_elephant=float((N_PEAKS[Y == 0] >= 2).mean()),
                             kept_giraffe=float((N_PEAKS[Y == 1] >= 2).mean())),
                 summary=summary.to_dict(orient='records'), mcnemar=mc.to_dict(orient='records'),
                 attention=table('attention_faithfulness.csv'), detector_sweep=table('detector_sweep.csv'),
                 edge=table('edge_profile.csv'), splits=table('splits.csv'))
    json.dump(paper, open(OUT / 'paper_results.json', 'w'), indent=1, default=str)
    zp = OUT.parent / f'{OUT.name}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob('*'):
            if p.is_file() and not ({'cache', 'queue'} & set(p.parts)) and p.suffix not in ('.pt', '.pkl', '.tmp'):
                zf.write(p, p.relative_to(OUT.parent))
    log(f'send this file back: {zp} ({zp.stat().st_size / 1e6:.1f} MB)')


# ----------------------------------------------------------------------------- command line
def init(args):
    global OUT, CFG
    OUT = Path(args.out or (HERE / ('results_rigorous_smoke' if args.smoke else 'results_rigorous'))).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'config.json'
    if p.exists():
        CFG = json.load(open(p))
    else:
        CFG = json.loads(json.dumps(smoke_config(DEFAULTS) if args.smoke else DEFAULTS))
        json.dump(CFG, open(p, 'w'), indent=1)
    for sub in ('logs', 'tables'):
        (OUT / sub).mkdir(exist_ok=True)
    log(f'output: {OUT} | smoke test: {CFG["smoke"]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['all', 'prepare', 'run', 'aggregate', 'status', 'worker'])
    ap.add_argument('--out', help='output folder (default: results_rigorous next to this script)')
    ap.add_argument('--data', help='path to dset_allspec_150_chunks_clean.nc')
    ap.add_argument('--smoke', action='store_true', help='short end-to-end test on 3,000 windows')
    ap.add_argument('--gpus', help='comma-separated GPU ids to use (default: every free GPU)')
    ap.add_argument('--max-gpus', type=int, default=0, help='use at most this many free GPUs (0 = no limit)')
    ap.add_argument('--workers-per-gpu', type=int, default=2)
    ap.add_argument('--max-used-mb', type=int, default=4000, help='a GPU counts as free below this memory use')
    ap.add_argument('--max-util', type=int, default=30, help='a GPU counts as free below this utilisation (%%)')
    ap.add_argument('--cpu-workers', type=int, default=2, help='workers when no GPU is free')
    ap.add_argument('--cpu-jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help='CPU processes for feature extraction')
    ap.add_argument('--poll', type=int, default=60, help='progress report interval (s)')
    ap.add_argument('--dry-run', action='store_true', help='test the scheduler without training')
    # internal (worker processes)
    ap.add_argument('--phase')
    ap.add_argument('--wid')
    ap.add_argument('--gpu', default='cpu')
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()
    if args.command == 'worker':
        return worker(args)
    init(args)
    if args.command in ('all', 'prepare'):
        prepare(args)
    if args.command in ('all', 'run'):
        cmd_run(args)
    if args.command in ('all', 'aggregate') and not args.dry_run:
        cmd_aggregate(args)
    if args.command == 'status':
        cmd_status(args)


if __name__ == '__main__':
    main()
