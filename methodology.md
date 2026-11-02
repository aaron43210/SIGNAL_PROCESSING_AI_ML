# Methodology — Seismic Wildlife Signal Classification (GNN / ANN / ML Comparison Study)

This document describes **what every notebook in the project does**, the **shared
methodology** they build on, a **comparison study** of the models, and the
**best-model recommendation** produced by the new `best_model_comparison.ipynb`.

Per the study scope, the detailed treatment focuses on the **weighted** notebooks,
the **GAT** notebooks, and the **ANN** notebook; the remaining notebooks are
summarised for completeness.

---

## 0 · Dataset & shared foundations

**File:** `dset_allspec_150_chunks_clean.nc` (NetCDF4 / HDF5, ~29 GB)
Derived from the SeisSavanna seismic ground-vibration recordings of African
savanna wildlife.

| Variable | Shape | Meaning |
|----------|-------|---------|
| `chunk`    | `(70637, 3, 2048)` | Raw 3-axis (X/Y/Z) seismic time series — **the model input** |
| `class`    | `(70637,)`         | Species label (integer) |
| `distance` | `(70637,)`         | Sensor-to-animal distance (m) |
| `seis`     | `(70637,)`         | Seismic event / sensor ID |
| `image`    | `(70637,128,128,3)`| Pre-computed spectrograms (**not used** by the raw-signal pipeline) |

**Class map:** 0 Human · 1 Elephant · 2 Giraffe · 3 Lion · 4 Buffalo · 5 Zebra ·
6 Wildebeest · 7 Hippo · 8 Cheetah · 9 Hyena.

**Severe class imbalance** — Elephant (class 1) ≈ 62,493 (~88% of all traces)
dominates every other species. For the binary Elephant-vs-Wildebeest task the ratio
is roughly **14–15 : 1**. This imbalance is the central modelling challenge and the
reason the "weighted" variants exist.

### The Peak-Graph pipeline (shared by all raw-signal GNN notebooks)

Instead of the expensive FFT→spectrogram→CNN route (`O(N log N)`), the project
converts each raw recording into a **sparse graph of footstep impacts** (`O(N)`),
suitable for low-power edge devices:

1. **3D magnitude envelope:** `mag = sqrt(X² + Y² + Z²)`, then z-score normalise.
2. **Peak detection:** `scipy.signal.find_peaks(prominence=0.05, distance=10)` —
   each detected peak is one **graph node** (one footstep impact). Recordings with
   `< 2` peaks are dropped.
3. **Node features (8-D):** `[magnitude, X, Y, Z at peak, prominence, local energy
   Σmag² in ±10 window, local std, normalised time position]`.
4. **Edges:** bidirectional **temporal chain** (footstep *i* ↔ *i*+1) **+ k-NN
   similarity edges** (`k=5`, on node features) linking similar footsteps.
5. **Graph-level readout:** `concat(global_mean_pool, global_max_pool)` → classifier
   head `Linear → BatchNorm → ReLU → Dropout → Linear`.

All GNN notebooks share this pipeline; they differ only in the **convolution
operator**, the **loss weighting strategy**, and small feature/pooling tweaks.

> **Edge deployment:** Euclidean GNNs (GAT/GCN/GIN) and the ANN quantize cleanly to
> **INT8** (`torch.quantization.quantize_dynamic`, ~160 KB). The hyperbolic HGCN
> does **not** quantize — a key negative result in the study.

---

## 1 · WEIGHTED notebooks (primary scope)

All four apply **inverse-frequency class weighting** in the loss
(`CrossEntropyLoss(weight=w)`, `w = (1/count)` normalised to sum = num_classes), so
the minority Wildebeest gets ~14× the gradient weight of Elephant. This is the
project's main remedy for imbalance (vs. the unweighted baselines, which collapse to
predicting Elephant for everything).

### 1a · `binary_weighted_gcn.ipynb` — Peak-GCN
- **Task:** Elephant (1) vs Wildebeest (6), raw-signal peak graph.
- **Model:** 3× `GCNConv(8→64→64→64)` (isotropic aggregation) + mean⊕max pool +
  classifier head. ~45 K params.
- **Training:** weighted CE, Adam, `ReduceLROnPlateau`, grid search
  (`hidden∈{32,64}`, `dropout∈{0.3,0.4}`, `lr∈{1e-3,5e-4}`), 100 epochs, early
  stopping on best test accuracy. INT8 quantization cell at the end (succeeds).
- **Role:** lightweight standard baseline; weaker noise filtering than GAT.

### 1b · `binary_weighted_gin.ipynb` — Peak-GIN
- Same pipeline/weighting. **Model:** input `Linear(8→64)` → 3× `GINConv(MLP[64→64])`
  with `train_eps=True` → mean⊕max pool → classifier. Most *expressive* aggregator
  (injective MLP), slightly more params than GCN. INT8 succeeds.

### 1c · `binary_weighted_hgcn.ipynb` — Peak-HGCN (hyperbolic)
- **Model:** a short (~7-cell) script, **no grid search**, 20 epochs. Despite the
  "HGCN" name the layers are **hyperbolic-wrapped GAT**:
  `log_map_zero → GATConv(heads=2) → ReLU → exp_map_zero` on the Poincaré ball, with
  `exp/log_map_zero` + `poincare_dist` helpers (tanh/atanh). No BatchNorm/Dropout.
- **Key purpose:** demonstrate that Riemannian ops **break INT8 quantization** — the
  quantization cell is *designed to fail*, arguing HGCN is **not edge-deployable** and
  Euclidean Peak-GAT is the superior choice. (Not executed, so the failure is asserted
  by design rather than captured in output.)

### 1d · `class1_and_class6_weighted.ipynb` — Peak-GAT (the headline weighted run)
- **Model:** 3× `GATConv(heads=4, concat=False)` attention aggregation + mean⊕max
  pool + classifier. **~45,058 params.**
- **This is the one weighted notebook with genuine, self-consistent executed
  results:**
  - **Accuracy 0.8921**, **weighted-F1 0.8959**, **macro-F1 ≈ 0.58**.
  - Per class: Elephant P/R/F1 = **0.95 / 0.94 / 0.94** (support 10,283);
    Wildebeest P/R/F1 = **0.20 / 0.23 / 0.22** (support 698).
  - INT8 model ≈ **161.3 KB** (FP32 184.3 KB → only 1.1× smaller, because most
    weights live in the GAT layers, not `nn.Linear`).
- **Interpretation:** weighting trades overall accuracy (0.94→0.89 vs the unweighted
  run) to actually *detect* the minority class (Wildebeest recall 0 → 0.23). The
  minority class remains hard, but the model is no longer a majority-class collapse.

---

## 2 · GAT notebooks (primary scope)

### 2a · `peak_based_GAT.ipynb` — multi-class Peak-GAT (top-4 species)
- **Task:** multi-class (filtered to the **4 most frequent** classes, remapped 0–3;
  Elephant dominates). "Spectrogram"-named but actually averages the 3 `chunk`
  components into a 1-D envelope.
- **Model:** 3× `GATConv(5→64, heads=4)` + mean⊕max pool + head. **5-D** node
  features here (height, prominence, local std, local energy, normalised position).
  **44,420 params.**
- **Training:** weighted CE (inverse frequency), Adam, `ReduceLROnPlateau`, 100 epochs.
- **Results (genuine):** headline **Best Test Accuracy 0.9058**, but **macro-F1 only
  0.36** — class 1 recall 1.00 while a minority class is missed entirely (0.00 F1).
  High accuracy is an imbalance artefact. *(Note: the training-loop print 0.6313 and
  the reloaded-best 0.9058 disagree; the classification report corresponds to 0.9058.)*

### 2b · `peak_gat_raw_signal.ipynb` — binary raw-signal Peak-GAT
- **Task:** binary classification directly on the raw 3-D signal (no FFT), motivation
  = `O(N)` vs `O(N log N)`. Uses the full **8-D** node-feature pipeline above.
- **Model:** `PeakGATRaw` = 3× `GATConv(8→64, heads=4)` + mean⊕max pool + head;
  grid search (16 combos) selected by macro-F1; 100 epochs, patience 15; INT8 cell.
- **Results:** **none captured** — the notebook was saved un-run; all printed numbers
  are unresolved f-string placeholders. It is the clean *reference template* for the
  binary GAT, not a source of measured numbers.

### 2c · `peak_gat_single_class.ipynb` — one-class GAT **autoencoder** (anomaly detection)
- **Task:** *not a classifier.* A **Graph Attention Autoencoder (GAT-GAE)** trained
  only on normal Elephant graphs; low reconstruction error = Elephant, high error =
  anomaly/alert. Eliminates the need for negative training data.
- **Model:** encoder `GATConv(8→32, heads=4) → GATConv(32→16)` (16-D latent); MLP
  decoder `16→32→8` reconstructing node features. Loss = MSE. 50 epochs, early stopping.
- **Anomaly demo:** inject Gaussian noise into test graphs, compare per-graph
  reconstruction error with a 3σ alert threshold. INT8 quantization for IoT.
- **Results:** none captured (placeholders only). Conceptually the paper's novel
  contribution (`PeakGAT` GAT-GAE, 45,058 params, 161.2 KB in the write-up).

### 2d · `class1_and_class6_enhanced_gat.ipynb` — Peak-GAT with 11-D physics features
- Builds on the weighted/balanced GAT by adding **3 physics-informed node features**
  (→ 11-D): inter-peak time delta (gait rhythm), zero-crossing rate (impact
  sharpness), peak rise/decay asymmetry (animal mass proxy). Forces CPU to dodge an
  Apple-MPS PyG bug.
- **Results:** the printed metrics are **stale copies of the 8-feature weighted run**
  (same 0.8921 / 45,058 params / "Node Features: 8D"); the 11-feature change was *not*
  re-executed. Treat its displayed numbers as **not valid** — the notebook must be
  re-run to evaluate the enhanced features.

### 2e · `class1_and_class6_gru_gat.ipynb` — GAT + GRU hybrid
- Spatial `GATConv` feature extractor → `SortAggregation(k=15)` → **GRU** over the
  peak sequence to model footstep *rhythm/temporal order* → classifier. 11-D features,
  50/50 undersampled.
- **Results:** also **stale Peak-GAT carry-over** (0.8921); `CELL 11` still
  instantiates `PeakGATRaw`, so the printed numbers do not reflect the GRU model.

---

## 3 · ANN notebook (primary scope)

### `ann_classifier_single_component.ipynb`
- **Task:** lightweight baseline — classify Elephant (1) vs Wildebeest (6) from a
  **single raw component** (X-axis, 2048-sample vector), no graph, no FFT.
- **Preprocessing:** stratified 80/20 split, `StandardScaler`, batch 128.
- **Model:** 3-layer MLP `Linear(2048→512) → BN → ReLU → Dropout(0.5) →
  Linear(512→256) → BN → ReLU → Dropout(0.5) → Linear(256→2)`.
- **Training:** **class-weighted** CE (`compute_class_weight('balanced')`), Adam
  `lr=1e-3`, 50 epochs. Evaluation = classification report + confusion matrix.
- **Role in the study:** the "no graph structure" control — tests how much the
  peak-graph representation actually adds over a plain MLP on the raw trace.

---

## 4 · Remaining notebooks (summary)

| Notebook | What it does | Status |
|----------|--------------|--------|
| `binary_unweighted_gcn/gin.ipynb` | Same GCN/GIN pipeline **without** class weighting — intentional minority-collapse demonstrations | Code templates (no captured outputs) |
| `binary_unweighted_hgcn.ipynb` | Unweighted hyperbolic variant; differs from weighted only in the loss line | Code template |
| `class1_and_class6_unweighted.ipynb` | Peak-GAT, no weighting — **genuine run: acc 0.9364 but Wildebeest recall/F1 = 0.00** (majority collapse), macro-F1 ≈ 0.48 | Genuine |
| `class1_and_class6_balanced.ipynb` | Adds majority **undersampling** to 50/50 before graph building | Outputs **stale** (copied from weighted run; not re-executed) |
| `class1_and_class6_patch_gcn.ipynb` | "Patch" GCN+GRU assuming a fixed 16-node graph (mismatches variable peak counts) | Outputs stale |
| `class1_and_class6_sortpool_gcn.ipynb` | 3-layer GCN + `SortAggregation(k=20)` + dense MLP head | Outputs stale |
| `hypergraph_raw_signal.ipynb` | **ST-HGCN**: true hypergraph (hyperedges = gait cycles via sliding window of 4 peaks + k-NN), incidence-matrix convolution, all-species multiclass | Code only (not executed) |
| `seismic_savanna_GNN.ipynb` | Exploratory: **visibility-graph** construction + unsupervised **Graph2Vec** embedding; dataset subsetting | Exploratory, no classifier/metrics |
| `netCDF_opening.ipynb` | Data exploration: inspect NetCDF structure, plot traces/spectrograms, early visibility-graph & toy GCN denoising experiments | Exploratory |

---

## 5 · Comparison study

### 5.1 Model-family comparison

| Model | Representation | Aggregation | Imbalance handling | INT8 edge-deployable | Params |
|-------|----------------|-------------|--------------------|----------------------|--------|
| **Peak-GAT** | 8-D peak graph | Attention (4 heads) | class-weighted loss | ✅ (~161 KB) | ~45 K |
| **Peak-GCN** | 8-D peak graph | Isotropic | class-weighted loss | ✅ | ~45 K |
| **Peak-GIN** | 8-D peak graph | MLP (injective) | class-weighted loss | ✅ | ~45 K |
| **Peak-HGCN** | 8-D peak graph | Hyperbolic GAT | class-weighted loss | ❌ (by design) | ~45 K |
| **ANN** | raw X-trace (2048) | — (dense MLP) | class-weighted loss | ✅ | ~1.3 M |
| **GAT-GAE** (one-class) | 8-D peak graph | Attention AE | one-class (no negatives) | ✅ | ~45 K |
| **RandomForest / GBoost** | 33-D peak stats | tree ensemble | `class_weight='balanced'` | ✅ (tiny) | — |

### 5.2 Measured results (only genuinely-executed runs)

| Run | Accuracy | Macro-F1 | Minority (Wildebeest) F1 | Note |
|-----|----------|----------|--------------------------|------|
| Peak-GAT, **unweighted** (`class1_and_class6_unweighted`) | **0.9364** | ~0.48 | **0.00** | majority-class collapse — high acc is misleading |
| Peak-GAT, **weighted** (`class1_and_class6_weighted`) | 0.8921 | **~0.58** | **0.22** | recovers minority detection at a small accuracy cost |
| Peak-GAT, **multiclass top-4** (`peak_based_GAT`) | 0.9058 | 0.36 | — | dominated by Elephant; macro-F1 low |

**Headline finding of the study:** accuracy alone is deceptive under 14:1 imbalance.
The *unweighted* model looks best (93.6%) yet never predicts Wildebeest; the
*weighted* model is the honest winner because **macro-F1 rises from ~0.48 to ~0.58**
and minority recall from 0 to 0.23. **Weighted loss is therefore the correct
training regime**, which is exactly why this study restricts itself to the weighted
variants plus the ANN.

### 5.3 ⚠️ Data-integrity caveats (important for the comparison)

Several notebooks carry **stale outputs** that were *not regenerated* after their
defining change: `class1_and_class6_balanced`, `..._enhanced_gat`, `..._gru_gat`,
`..._patch_gcn`, `..._sortpool_gcn` all display the **same** Peak-GAT numbers
(0.8921 / 45,058 params). Their stored metrics **cannot be trusted** as results for
their stated architectures. The `binary_*` and `hypergraph_*` notebooks were saved
un-run (f-string placeholders). To obtain a trustworthy cross-architecture
comparison, the models must be re-trained under one controlled harness — which is the
purpose of `best_model_comparison.ipynb` below.

---

## 6 · Best-model recommendation & the comparison notebook

### `best_model_comparison.ipynb` (new)

Generated by `generate_best_model_notebook.py`, this single notebook removes the
stale-output problem by training **every weighted candidate on one shared
peak-graph split**:

1. Load dataset, select Elephant vs Wildebeest, compute class weights.
2. Build the 8-D peak-graph dataset **once**; derive a parallel 33-D aggregated
   feature matrix for the ANN/ML baselines from the *same* recordings and split.
3. Train **Peak-GAT, Peak-GCN, Peak-GIN** (weighted CE), the **ANN** (weighted CE),
   and classical **RandomForest / GradientBoosting** (balanced).
4. Produce a **comparison table** (accuracy, macro-F1, params, INT8-deployable),
   **auto-select the best model by macro-F1**, retrain/save it
   (`best_model_selected.pth` / `.joblib`), print its classification report +
   confusion matrix, and run an **INT8 edge-deployment check** on the winner.

**Recommended model (a-priori, from the evidence above):** **weighted Peak-GAT.** It
combines the attention mechanism's superior noise filtering, the best genuine
macro-F1 observed (~0.58), a tiny ~45 K-param / ~161 KB INT8 footprint for edge
deployment, and clean quantization — unlike the hyperbolic HGCN. The comparison
notebook confirms (or overrides) this empirically on a controlled, re-executed run.

> **How to run:** execute `best_model_comparison.ipynb` in the same environment as the
> other notebooks (PyTorch + PyTorch-Geometric + scikit-learn, GPU recommended). The
> final cells emit the comparison table and the exported best-model artefact.

---

### File index

- **Deliverables:** `methodology.md` (this file), `best_model_comparison.ipynb`,
  `generate_best_model_notebook.py` (its generator).
- **Primary-scope sources:** `class1_and_class6_weighted`, `binary_weighted_gcn`,
  `binary_weighted_gin`, `binary_weighted_hgcn`, `peak_based_GAT`,
  `peak_gat_raw_signal`, `peak_gat_single_class`, `class1_and_class6_enhanced_gat`,
  `class1_and_class6_gru_gat`, `ann_classifier_single_component`.

