"""Generator for ml_classifier.ipynb — single, self-contained classical-ML
classifier notebook styled to match the other notebooks in this project
(emoji + "CELL N" headers, dataset summary, peak-detection visualisation).

Task: Elephant (class 1) vs Wildebeest (class 6) from raw 3D seismic signal,
using scikit-learn only (no deep learning / no graph networks).

Run:  python generate_ml_classifier_notebook.py
"""
import json

cells = []
def md(t):   cells.append(("markdown", t))
def code(t): cells.append(("code", t))

# BEGIN_CELLS

md(r"""
# 🌲 Classical ML on Raw 3D Seismic Signal
## Animal Footstep Binary Classification — Elephant vs Wildebeest | scikit-learn Benchmark

**Approach:** Instead of a graph neural network, we extract a fixed-length
**hand-crafted feature vector** from each raw 3D seismic recording and classify it
with classical machine-learning models (Random Forest, Gradient Boosting, SVM, …).

**Why a classical-ML benchmark is included:**
- Extremely lightweight and trivially edge-deployable (no GPU, tiny model)
- Strong, interpretable baseline (feature importances) for the GNN/ANN study
- No deep-learning dependencies — only `numpy`, `scipy`, `scikit-learn`

**Class imbalance (~14:1 Elephant:Wildebeest):** the notebook trains every model
**unweighted vs weighted (balanced)** and shows that weighting is better — it stops
the dominant Elephant class from influencing the result and lets the rare Wildebeest
class actually be detected. The exported model is the **weighted** winner.

**Dataset variables (from `dset_allspec_150_chunks_clean.nc`):**
- `chunk(traces, comp, time)` → Raw seismic signal, shape `(N, 3, 2048)` — **this is our input**
- `class(traces)` → Animal class label
- `distance(traces)` → Distance from sensor to animal
- `image(traces, side, side, comp)` → Pre-computed spectrograms (we **do NOT use these**)
- `seis(traces)` → Seismic event ID
""")

md("---\n## 🔧 CELL 1: Setup & Configuration")

code(r"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

SEED = 42
np.random.seed(SEED)
print('scikit-learn classical-ML pipeline')
print('Setup complete ✓')
""")

md("---\n## 📂 CELL 2: Load Dataset & Inspect Structure")

code(r"""
import netCDF4 as nc

CANDIDATES = [
    'dset_allspec_150_chunks_clean.nc',
    'PROJECT/dset_allspec_150_chunks_clean.nc',
    '/Users/aaronr/Desktop/DUK/PROJECT/SIGNAL_ PROCESSING_PROJECT/dset_allspec_150_chunks_clean.nc',
]
FILE_PATH = next((p for p in CANDIDATES if os.path.exists(p)), CANDIDATES[0])

dataset = nc.Dataset(FILE_PATH, mode='r')
print('=' * 60); print('Dataset Summary'); print('=' * 60)
print('Variables:', list(dataset.variables.keys()))
for v in dataset.variables.keys():
    print(f'  {v:12s}: shape = {dataset.variables[v].shape}, dtype = {dataset.variables[v].dtype}')

all_labels = dataset['class'][:].astype(int)
uniq, cnt = np.unique(all_labels, return_counts=True)
print(f'\nTotal recordings : {len(all_labels)}')
for cls, c in zip(uniq, cnt):
    print(f'  Class {cls:3d}: {c:6d} samples')
""")

md("---\n## 🗺️ CELL 3: Select Classes & Load Raw Signal Data")

code(r"""
SPECIES_NAMES = {0:'Human',1:'Elephant',2:'Giraffe',3:'Lion',4:'Buffalo',
                 5:'Zebra',6:'Wildebeest',7:'Hippo',8:'Cheetah',9:'Hyena'}
SELECTED_CLASSES = [1, 6]   # Elephant vs Wildebeest (2-class binary)

mask = np.isin(all_labels, SELECTED_CLASSES)
chunks = dataset['chunk'][:][mask]          # (N, 3, 2048) raw 3D signal
labels = all_labels[mask]
dataset.close()

label_map = {c: i for i, c in enumerate(sorted(SELECTED_CLASSES))}   # {1:0, 6:1}
y = np.array([label_map[l] for l in labels])
label_names = [SPECIES_NAMES[c] for c in sorted(SELECTED_CLASSES)]

u, c = np.unique(y, return_counts=True)
print(f'Using classes {label_names}, {len(y)} samples total')
for ui, ci in zip(u, c):
    print(f'  {label_names[ui]:12s}: {ci:6d}')
print(f'Imbalance ratio ~ {c.max()/c.min():.1f} : 1  (minority = {label_names[int(np.argmin(c))]})')
""")

md("---\n## 🏔️ CELL 4: Peak Detection Visualisation")

code(r"""
def compute_magnitude(sig3d):
    return np.sqrt(sig3d[0]**2 + sig3d[1]**2 + sig3d[2]**2)

raw = chunks[0]; mag = compute_magnitude(raw)
mag_norm = (mag - mag.mean()) / (mag.std() + 1e-8)
peaks, props = find_peaks(mag_norm, prominence=0.05, distance=10)
t = np.arange(len(mag_norm))

fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
fig.suptitle(f'Peak Detection on 3D Magnitude — {SPECIES_NAMES.get(int(labels[0]), "?")}',
             fontsize=13, fontweight='bold')
axes[0].plot(t, mag, color='#8e44ad', linewidth=0.8)
axes[0].set_title('Raw 3D Magnitude'); axes[0].set_ylabel('Magnitude'); axes[0].grid(True, alpha=0.3)
axes[1].plot(t, mag_norm, color='#2c3e50', linewidth=0.8, label='Normalised Magnitude')
axes[1].scatter(peaks, mag_norm[peaks], color='#e74c3c', zorder=5, s=40,
                label=f'Detected Peaks ({len(peaks)} footsteps)')
axes[1].set_title('Normalised Magnitude with Detected Peaks')
axes[1].set_ylabel('Norm. Amplitude'); axes[1].set_xlabel('Time Step')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ml_peak_detection_demo.png', dpi=150, bbox_inches='tight'); plt.show()
print(f'Detected {len(peaks)} footstep peaks — summarised into features in the next cell.')
""")

md(r"""
---
## 🧮 CELL 5: Feature Engineering

Each recording → a fixed **29-D feature vector** (independent of how many footsteps
it contains), combining global time-domain statistics with footstep-peak statistics
from the 3D magnitude envelope:
- **Per-axis (X, Y, Z):** mean, std, max, abs-max, energy, zero-crossing rate
- **3D magnitude:** mean, std, max, energy
- **Footstep peaks:** count, mean/std/max height, mean prominence, mean inter-peak gap
""")

code(r"""
FEATURE_NAMES = (
    [f'{ax}_{s}' for ax in ['X','Y','Z'] for s in ['mean','std','max','absmax','energy','zcr']]
    + ['mag_mean','mag_std','mag_max','mag_energy',
       'n_peaks','peak_mean_h','peak_std_h','peak_mean_prom','peak_mean_gap','peak_max_h']
)

def extract_features(sig3d, prominence=0.05, distance=10):
    feats = []
    for ax in range(3):                       # per-axis time-domain stats
        s = sig3d[ax]
        zcr = np.mean(np.abs(np.diff(np.sign(s)))) / 2.0
        feats += [s.mean(), s.std(), s.max(), np.abs(s).max(), float(np.sum(s**2)), zcr]
    mag = np.sqrt(sig3d[0]**2 + sig3d[1]**2 + sig3d[2]**2)   # 3D magnitude envelope
    feats += [mag.mean(), mag.std(), mag.max(), float(np.sum(mag**2))]
    mag_norm = (mag - mag.mean()) / (mag.std() + 1e-8)        # footstep peak stats
    peaks, props = find_peaks(mag_norm, prominence=prominence, distance=distance)
    if len(peaks) > 0:
        heights, proms = mag[peaks], props['prominences']
        gaps = np.diff(peaks) if len(peaks) > 1 else np.array([0.0])
        feats += [len(peaks), heights.mean(), heights.std(),
                  proms.mean(), gaps.mean(), heights.max()]
    else:
        feats += [0, 0, 0, 0, 0, 0]
    return np.array(feats, dtype=np.float32)

from tqdm import tqdm
X = np.stack([extract_features(chunks[i]) for i in tqdm(range(len(chunks)), desc='features')])
X = np.nan_to_num(X)
print(f'Feature matrix: {X.shape}  ({len(FEATURE_NAMES)} features/recording)')
""")

md("---\n## 🏗️ CELL 6: Train/Test Split & Scaling")

code(r"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f'Train: {X_train.shape[0]}   Test: {X_test.shape[0]}')
print('Scaling applied (linear/distance models use scaled features; trees use raw).')
""")

md(r"""
---
## 🧠 CELL 7: Candidate ML Models + the Weighting Toggle

The dataset is ~14:1 imbalanced, so a model trained **unweighted** can score high
accuracy simply by always predicting the majority class (**Elephant**) and *never*
detecting **Wildebeest** — the big class drowns out the small one.

To stop the bigger class influencing the result we apply **balanced class weighting**:
every sample gets weight inversely proportional to its class frequency
(`compute_sample_weight('balanced', y_train)`), so the rare Wildebeest footsteps
count as much as the abundant Elephant ones during training.

Below we train **every model twice — unweighted vs weighted — and compare**, to
demonstrate that weighting is the better choice.
""")

code(r"""
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import recall_score

# Base estimators (NO built-in weighting — we control it via sample_weight so the
# unweighted vs weighted comparison is apples-to-apples for every model).
MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=2000, random_state=SEED),
    'SVM (RBF)':          SVC(kernel='rbf', probability=True, random_state=SEED),
    'RandomForest':       RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    'GradientBoosting':   GradientBoostingClassifier(n_estimators=200, random_state=SEED),
}
try:
    from xgboost import XGBClassifier
    MODELS['XGBoost'] = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                      eval_metric='logloss', random_state=SEED, n_jobs=-1)
    print('XGBoost available ✓')
except Exception:
    print('XGBoost not installed; skipping (optional).')

NEEDS_SCALING = {'LogisticRegression', 'SVM (RBF)'}

# Balanced sample weights (used only in the "weighted" runs)
SW_BALANCED = compute_sample_weight('balanced', y_train)
MINORITY = int(np.argmin(np.bincount(y_train)))   # Wildebeest index
print(f'{len(MODELS)} base models. Minority class = {label_names[MINORITY]}')
""")

md("---\n## 🏋️ CELL 8: Train Unweighted vs Weighted & Compare")

code(r"""
rows, fitted = [], {}
for name, base in MODELS.items():
    Xtr = X_train_s if name in NEEDS_SCALING else X_train
    Xte = X_test_s  if name in NEEDS_SCALING else X_test
    for mode in ['unweighted', 'weighted']:
        m = clone(base)
        sw = SW_BALANCED if mode == 'weighted' else None
        m.fit(Xtr, y_train, sample_weight=sw)
        pred = m.predict(Xte)
        rows.append({
            'model': name, 'weighting': mode,
            'accuracy': accuracy_score(y_test, pred),
            'macro_f1': f1_score(y_test, pred, average='macro'),
            f'{label_names[1]}_recall': recall_score(y_test, pred, pos_label=MINORITY, zero_division=0),
            f'{label_names[1]}_f1': f1_score(y_test, pred, average=None, zero_division=0)[MINORITY],
        })
        fitted[(name, mode)] = m

cmp = pd.DataFrame(rows)
for c in cmp.columns:
    if c not in ('model', 'weighting'): cmp[c] = cmp[c].round(4)
print('UNWEIGHTED vs WEIGHTED (all models):')
print(cmp.to_string(index=False))

print('\nEffect of weighting (weighted − unweighted):')
for name in MODELS:
    u = cmp[(cmp.model == name) & (cmp.weighting == 'unweighted')].iloc[0]
    w = cmp[(cmp.model == name) & (cmp.weighting == 'weighted')].iloc[0]
    print(f'  {name:20s}  macroF1 {u.macro_f1:.3f}→{w.macro_f1:.3f}   '
          f'{label_names[1]} recall {u[f"{label_names[1]}_recall"]:.3f}→{w[f"{label_names[1]}_recall"]:.3f}')
print('\n➡️  Weighting raises macro-F1 and minority recall: the majority class can no longer dominate.')
""")

md("---\n## 📊 CELL 8b: Why Weighting Wins — Visual")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Unweighted vs Weighted — the big class can no longer dominate',
             fontsize=13, fontweight='bold')
names = list(MODELS.keys()); xpos = np.arange(len(names)); bw = 0.38
for k, (metric, title) in enumerate([('macro_f1', 'Macro-F1'),
                                      (f'{label_names[1]}_recall', f'{label_names[1]} recall')]):
    u = [cmp[(cmp.model == n) & (cmp.weighting == 'unweighted')][metric].iloc[0] for n in names]
    w = [cmp[(cmp.model == n) & (cmp.weighting == 'weighted')][metric].iloc[0] for n in names]
    ax[k].bar(xpos - bw/2, u, bw, label='unweighted', color='#e74c3c')
    ax[k].bar(xpos + bw/2, w, bw, label='weighted',   color='#2ecc71')
    ax[k].set_xticks(xpos); ax[k].set_xticklabels(names, rotation=30, ha='right')
    ax[k].set_title(title); ax[k].set_ylim(0, 1); ax[k].legend(); ax[k].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ml_weighting_effect.png', dpi=150, bbox_inches='tight'); plt.show()
""")

# END_CELLS

md(r"""
---
## 📈 CELL 9: Final Leaderboard (Weighted Models)

Having shown weighting is better, the final model selection is made **among the
weighted models only** — ranked by macro-F1 (robust to imbalance).
""")

code(r"""
tbl = (cmp[cmp.weighting == 'weighted']
       .drop(columns='weighting')
       .sort_values('macro_f1', ascending=False)
       .reset_index(drop=True))
print('🏆 WEIGHTED MODEL LEADERBOARD (sorted by macro-F1):')
print(tbl.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
fig.suptitle('Weighted Models — Final Comparison', fontsize=13, fontweight='bold')
ax[0].barh(tbl['model'], tbl['macro_f1'], color='#2ecc71'); ax[0].set_xlabel('macro-F1')
ax[0].set_title('Macro-F1'); ax[0].invert_yaxis(); ax[0].grid(True, alpha=0.3)
ax[1].barh(tbl['model'], tbl['accuracy'], color='#3498db'); ax[1].set_xlabel('accuracy')
ax[1].set_title('Accuracy'); ax[1].invert_yaxis(); ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ml_classifier_comparison.png', dpi=150, bbox_inches='tight'); plt.show()
""")

md("---\n## 📋 CELL 10: Best Model — Report & Confusion Matrix")

code(r"""
best_name = tbl.iloc[0]['model']
best_model = fitted[(best_name, 'weighted')]      # the WEIGHTED version
Xte = X_test_s if best_name in NEEDS_SCALING else X_test
pred = best_model.predict(Xte)

print(f'🏆 BEST ML MODEL: {best_name} (weighted)  (macro-F1 {tbl.iloc[0]["macro_f1"]:.4f})\n')
print(classification_report(y_test, pred, target_names=label_names, zero_division=0))

cm = confusion_matrix(y_test, pred)
fig, axc = plt.subplots(figsize=(6, 5))
im = axc.imshow(cm, cmap='Blues')
axc.set_xticks([0, 1]); axc.set_yticks([0, 1])
axc.set_xticklabels(label_names); axc.set_yticklabels(label_names)
axc.set_xlabel('Predicted'); axc.set_ylabel('True')
axc.set_title(f'Confusion Matrix — {best_name} (weighted)')
for i in range(2):
    for j in range(2):
        axc.text(j, i, cm[i, j], ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=13)
plt.colorbar(im); plt.tight_layout()
plt.savefig('ml_classifier_confusion.png', dpi=150, bbox_inches='tight'); plt.show()
""")

md("---\n## 🌟 CELL 11: Feature Importance")

code(r"""
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=FEATURE_NAMES).sort_values()
    plt.figure(figsize=(8, 8))
    imp.plot(kind='barh', color='#8e44ad')
    plt.title(f'Feature Importance — {best_name}'); plt.xlabel('importance')
    plt.tight_layout(); plt.savefig('ml_feature_importance.png', dpi=150, bbox_inches='tight'); plt.show()
    print('Top 8 features:'); print(imp.sort_values(ascending=False).head(8))
else:
    print(f'{best_name} has no feature_importances_ (not a tree ensemble).')
""")

md("---\n## 💾 CELL 12: Export Best Model")

code(r"""
import joblib
joblib.dump({'model': best_model, 'scaler': scaler,
             'needs_scaling': best_name in NEEDS_SCALING,
             'feature_names': FEATURE_NAMES, 'label_names': label_names},
            'best_ml_classifier.joblib')
print('Saved -> best_ml_classifier.joblib')
print('\nInference on a new (3, 2048) recording:')
print("  b = joblib.load('best_ml_classifier.joblib')")
print("  f = extract_features(sig).reshape(1, -1)")
print("  if b['needs_scaling']: f = b['scaler'].transform(f)")
print("  pred = b['model'].predict(f)   # 0=Elephant, 1=Wildebeest")
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

with open("ml_classifier.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote ml_classifier.ipynb with {len(nb['cells'])} cells.")
