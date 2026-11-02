# Comparative Study Plan: ML vs. DL vs. GNN vs. KAN for Seismic Wildlife Classification on ESP32-S3

Status: draft plan (v1) — 2026-09-03
Dataset: `dset_allspec_150_chunks_clean.nc` (29.5 GB, not copied into this folder — read in place)

---

## 1. What this project actually is

The dataset is **seismic footstep classification**, not audio/image classification. Ten
seismometer stations recorded ground-vibration "chunks" (3-axis, 2048 samples each) triggered by
animal/human footsteps in a savanna environment, at distances of 11–150 m from the sensor. The
task: identify *which species* produced the footstep, from a compact embedded sensor node.

This matches the prior work already in `PROJECT/` (ANN, GCN/GAT/GIN, KAN notebooks) — this plan
extends that work into a rigorous 4-way comparison with a real embedded-deployment leg on
ESP32-S3, rather than stopping at desktop accuracy numbers.

### 1.1 Confirmed dataset structure (inspected directly, not assumed)

| Variable | Shape | Meaning |
|---|---|---|
| `chunk` | (70637, 3, 2048) float32 | raw 3-axis (X/Y/Z) vibration waveform per event |
| `image` | (70637, 128, 128, 3) int64 | precomputed spectrogram-style image per event |
| `class` | (70637,) int64 | species label |
| `distance` | (70637,) float32 | sensor-to-source distance, 11.4–149.8 m |
| `seis` | (70637,) int64 | which of 10 seismometer stations recorded it |

Species mapping used in prior notebooks (`SPECIES_NAMES`):
`0 Human, 1 Elephant, 2 Giraffe, 3 Lion, 4 Buffalo, 5 Zebra, 6 Wildebeest, 7 Hippo, 8 Cheetah, 9 Hyena`
(class `11` is unmapped in prior work — 4 samples; treat as "unknown/other" and exclude, confirm with domain source if available).

### 1.2 Class distribution — the single biggest constraint on this whole study

```
class  0 (Human)      2073   ( 2.9%)
class  1 (Elephant)  62493   (88.5%)   <- dominates the dataset
class  2 (Giraffe)     164   ( 0.2%)
class  3 (Lion)          2   ( 0.0%)   <- unusable for ML
class  4 (Buffalo)     271   ( 0.4%)
class  5 (Zebra)       769   ( 1.1%)
class  6 (Wildebeest) 4321   ( 6.1%)
class  7 (Hippo)        42   ( 0.1%)   <- unusable for ML
class  8 (Cheetah)      10   ( 0.0%)   <- unusable for ML
class  9 (Hyena)       488   ( 0.7%)
class 11 (unknown)       4   ( 0.0%)   <- unusable for ML
```

**Decision:** run the comparative study on two tiers, defined once and reused identically across
all four methods so comparisons stay fair:

- **Core-6 set** (primary comparison): classes `{0, 1, 2, 4, 5, 6, 9}` — 7 classes with ≥150
  samples (Human, Elephant, Giraffe, Buffalo, Zebra, Wildebeest, Hyena). This is where accuracy
  numbers are trustworthy enough to compare architectures.
- **Full-11 set** (secondary/stretch): all classes, reported with heavy caveats — classes with
  <50 samples cannot get a statistically meaningful test set no matter what architecture is used.
  Report macro-F1 here mainly to show how gracefully each method degrades, not as a headline number.
- Elephant (class 1) will still dominate Core-6 (62.5k of ~70.6k) → **use stratified splits +
  macro-F1 / balanced accuracy as the primary metric everywhere**, never plain accuracy alone
  (a model that always predicts "Elephant" scores ~89% accuracy and is worthless).

---

## 2. Target hardware: ESP32-S3 constraints that drive every downstream decision

| Resource | Typical ESP32-S3 (N16R8 module) | Implication |
|---|---|---|
| CPU | Dual-core Xtensa LX7 @ up to 240 MHz, w/ SIMD (vector) instructions for INT8/INT16 | Favors INT8-quantized, matmul-heavy models; no hardware FPU for double, single-precision float ops are slow relative to INT8 |
| SRAM | ~512 KB (parts reserved for stack/OS/WiFi if used) | Working set (activations) must fit in low hundreds of KB |
| PSRAM (optional, octal SPI) | 0, 2, or 8 MB depending on module | If present, usable for larger weight tensors, but slower access than SRAM — budget for it explicitly, don't assume it |
| Flash | 4–16 MB typical dev boards | Model weights (flash-resident, e.g. via `.rodata`) rarely the bottleneck; RAM usually is |
| NN runtime support | ESP-DL (Espressif, INT8-native, optimized conv/dense/pool) and TFLite-Micro (`esp-tflite-micro`) both officially supported | **Only ML (as generated C) and DL (CNN/MLP) have first-class runtime support.** GNN message-passing ops and KAN spline ops do **not** exist in either runtime — see §6/§7 for the workaround this plan commits to. |
| No OS assumptions | Bare-metal / FreeRTOS via ESP-IDF | Inference must be a single-threaded, deterministic C function for clean latency/energy measurement |

**Working RAM/latency/energy budget for this study** (documented once, applied to every method):
- Peak RAM (weights + activations) target: **≤ 200 KB** without PSRAM, **≤ 1.5 MB** if PSRAM variant is used as a stretch comparison.
- Inference latency target: **≤ 50 ms** per event (footstep events are sparse in time — this is not a hard real-time audio-rate constraint, so the budget is generous, but 50 ms is what defines "usable" for a battery node reporting infrequently).
- Model flash footprint target: **≤ 512 KB** per model.
- These are *targets to report against*, not hard cutoffs — part of the deliverable is showing where each of the 4 methods lands relative to them.

---

## 3. Common experimental protocol (defined once, shared by all 4 tracks)

Fairness of the comparison depends entirely on this being identical across methods — this is
worth doing carefully before touching any model code.

1. **Fixed split**: stratified train/val/test = 70/10/20, fixed `random_state`, computed once per
   tier (Core-6, Full-11) and saved to `data/splits/*.npz` as index arrays — every training script
   loads the same indices, never re-splits.
2. **Grouping caveat to test for**: check whether the same footstep event is duplicated across
   sensors (`seis` has 10 stations) — if the same physical event appears at multiple stations,
   a naive random split leaks information across train/test. Do a **group-aware split by event
   (or by station-holdout)** as a secondary robustness check; report both if it changes results.
3. **Preprocessing shared by every method**: per-channel (X/Y/Z) z-score normalization fit on
   train only; no leakage from val/test into normalization statistics.
4. **Metrics logged identically for every model** (desktop stage):
   - Accuracy, macro-F1, balanced accuracy, per-class recall, confusion matrix
   - 5-fold stratified CV on train+val for hyperparameter selection (not just a single split)
5. **Metrics logged identically for every model** (embedded stage, see §8):
   - On-device inference latency (mean/p95 over ≥200 runs), peak RAM, flash size, energy per
     inference, accuracy drop from quantization (desktop fp32 vs. on-device INT8).
6. **Statistical comparison**: paired McNemar's test (or 5×2cv paired t-test) between each pair of
   methods' predictions on the same test set — "Method A beats Method B" needs a p-value, not
   just a higher point estimate, given how small some class counts are.

---

## 4. Track A — Classical Machine Learning

**Goal**: a strong, interpretable, cheap-to-deploy baseline — this is what the other 3 methods
have to justify their extra complexity against.

- **Features** (per event, computed on the raw 3-axis 2048-sample chunk — reuse peak-detection
  logic already built in `gcn_classifier.ipynb`/`kan.ipynb`):
  - Time-domain: mean, std, RMS, min/max, skew, kurtosis, zero-crossing rate — per axis and on
    the 3D magnitude envelope
  - Frequency-domain: FFT-based spectral centroid, bandwidth, dominant frequency, band energies
    (a handful of bands appropriate to seismic footstep frequency range)
  - Footstep-peak statistics (already implemented): peak count, inter-peak interval mean/std,
    peak magnitude mean/max/std, total energy — this is the same pooled-24D feature family used
    for the KAN track, reused here so Track A and Track D can be compared apples-to-apples on
    identical inputs, isolating "does the classifier matter" from "do the features matter"
  - `distance` included as an optional extra feature in one ablation (does knowing sensor-source
    distance help disambiguate species vs. leaking a shortcut?)
- **Models**: Random Forest, Gradient Boosted Trees (XGBoost or LightGBM), SVM-RBF, k-NN,
  Logistic Regression (linear baseline). Select the best via 5-fold CV macro-F1 on Core-6.
- **Class imbalance handling**: `class_weight='balanced'`, and compare against SMOTE/ADASYN
  oversampling on the minority classes as an ablation.
- **Embedded export path**: tree ensembles → C code via `m2cgen` or `emlearn` (pure `if/else` /
  lookup arrays, no runtime needed at all — this is the *easiest* of the 4 methods to get onto
  ESP32-S3, and that ease is itself a comparison finding worth reporting). SVM/LogReg → hand-rolled
  dot-product C function.

---

## 5. Track B — Deep Learning

**Goal**: compare a compact 1D-CNN on the raw waveform against the existing ANN/MLP baseline on
pooled features, then take the better of the two forward for compression + deployment.

- **Architecture 1 (already exists)**: `ANNClassifier` from `ann_classifier_weighted.ipynb` —
  MLP over the 24D pooled peak features. Re-run under the shared protocol (§3) for a fair number.
- **Architecture 2 (new)**: compact 1D-CNN directly on the raw `(3, 2048)` waveform —
  depthwise-separable 1D convolutions (MobileNet-style) to keep parameter/FLOP count low, global
  average pooling, small dense head. This tests whether learned raw-waveform features beat
  handcrafted peak features.
- **Architecture 3 (secondary, desktop-only)**: small 2D-CNN on the existing `image` (128×128×3)
  representation, purely to check whether the precomputed spectrogram image contains more
  discriminative signal than the raw waveform. **Not a deployment candidate** — 128×128×3 input
  is far outside the ESP32-S3 RAM budget without aggressive downsampling (would need to shrink to
  something like 32×32 to fit, which itself is a finding worth reporting: "the shipped image
  representation isn't edge-viable at native resolution").
- **Compression pipeline** (this is the DL track's real contribution vs. ML): 
  - Post-training INT8 quantization (PTQ) as the baseline
  - Quantization-aware training (QAT) to recover accuracy lost to PTQ
  - Structured pruning (magnitude-based, channel-level) + re-training
  - Optional: knowledge distillation from the 2D-CNN (teacher) into the 1D-CNN (student) to see
    if the image model's extra accuracy can be transferred into a deployable model
- **Embedded export path**: TensorFlow Lite → `esp-tflite-micro`, or convert to ESP-DL's native
  format for better INT8 SIMD utilization. Benchmark both runtimes for the same model — which
  runtime the paper/report recommends is itself a result.

---

## 6. Track C — Graph Neural Networks (GCN / GAT / GIN)

**Goal**: evaluate whether modeling a footstep event as a *graph of impact peaks* (rather than a
fixed-size vector or raw waveform) improves classification — and confront directly the fact that
GNNs have **no embedded inference runtime** on ESP32-S3.

- **Graph construction (already implemented, reuse as-is)**: footstep peaks detected via
  `scipy.signal.find_peaks` on the 3D magnitude envelope → each peak is a node with an 8D feature
  vector (magnitude, X/Y/Z amplitude, prominence, local energy, local std, normalized position);
  edges = temporal chain + k-NN similarity (k=5).
- **Desktop comparison**: GCN (already implemented, `best_species_gcn_classifier.pth` exists),
  GAT (already implemented in the class1/class6 notebooks — extend to all Core-6 classes), and
  GIN (referenced by `GIN.png` — implement if not already done) trained under the shared protocol
  of §3. This produces the "which GNN variant wins" sub-result before spending effort deploying one.
- **The embedded deployment problem, and the plan to solve it**: PyTorch Geometric's sparse
  message-passing has no ESP32 runtime equivalent. The path forward:
  1. **Fix the graph size**: cap peaks per event at a constant `N_MAX` (pad with zero-nodes +
     mask, or truncate lowest-prominence peaks beyond `N_MAX`) — turns the variable-size graph
     into a fixed-shape dense tensor problem, which is what makes step 2 possible.
  2. **Densify the message passing**: with fixed `N_MAX`, a GCN layer reduces to
     `H' = σ(D^-1/2 · A · D^-1/2 · H · W)` — a fixed-size adjacency matrix and dense matmuls only.
     This has no PyG-specific ops left; it can be exported as a small sequence of matmuls, same
     representation family as the DL track's dense layers.
  3. **Deploy as hand-written C**: because there's no standard op for even this densified form in
     TFLite-Micro's op set, write a small dedicated C inference function (a handful of matmul +
     normalization + ReLU calls, adjacency matrix baked in per-sample from peak detection done
     on-device) rather than forcing it through a general graph runtime. This is more code than the
     ML/DL tracks but is a fixed, boundable amount — feasible within the study's timeline.
  4. GAT's attention weights and GIN's sum-aggregation both reduce similarly under a fixed `N_MAX`
     — implement whichever of the three wins the desktop comparison, not all three, for the
     embedded leg (time-box this — deploying one well beats deploying three badly).
- **Expected finding to report explicitly**: GNN accuracy vs. the RAM/latency cost of computing
  peak detection + fixed-size adjacency on-device *before* the matmuls even start — this
  preprocessing cost is unique to this track and should be measured separately from "model
  inference time" so the comparison with ML/DL/KAN isn't misleading.

---

## 7. Track D — Kolmogorov-Arnold Networks (KAN)

**Goal**: test the core KAN claim (comparable accuracy with far fewer/more-interpretable
parameters than an MLP) on this dataset, and confront the fact that spline evaluation has no
embedded runtime either.

- **Architecture (already implemented, reuse as-is)**: `KANLinear` from `kan.ipynb` /
  `kan_weighted.ipynb` — B-spline basis (`grid_size=5`, `spline_order=3` → 8 basis functions per
  edge) + base residual path, stacked `24 → 32 → 16 → num_classes` over the same pooled 24D peak
  features used in Tracks A and B. Re-run under the shared protocol of §3.
- **Direct apples-to-apples ablation**: since Track A (ML) and Track B (ANN) can both also
  consume the identical 24D pooled feature vector, this gives a clean 3-way comparison
  (RF/GBT vs. MLP vs. KAN) **on the exact same input representation** — isolating the classifier
  architecture as the only variable. This is the most scientifically clean comparison in the whole
  study and should be highlighted as such in the final report.
- **Parameter-efficiency analysis** (the actual KAN research question): plot accuracy vs.
  parameter count for KAN vs. MLP at matched hidden widths — does KAN reach the same accuracy with
  fewer trainable parameters here, as the original KAN paper claims for other domains?
- **The embedded deployment problem, and the plan to solve it**: no runtime evaluates B-splines
  on ESP32. The path forward:
  1. **Quantize each edge's spline to a lookup table (LUT)**: since `grid_range=(-1, 1)` is fixed
     and inputs are standardized, precompute each edge's spline output at e.g. 32–64 evenly spaced
     input values post-training, quantize to INT8, and store as a static C array. Inference
     becomes: clip input → index into LUT (+ linear interpolation between two nearest bins) →
     accumulate. This is cheap (no runtime spline math at all) and bounded in size
     (`in_features × out_features × LUT_resolution` bytes per layer — compute this exactly for the
     `24→32→16→num_classes` architecture before committing, to confirm it fits the flash budget).
  2. **Validate LUT fidelity**: compare LUT-approximated KAN accuracy against the exact PyTorch
     spline evaluation on the test set before deploying — report the approximation gap explicitly,
     it's a real cost of this method that ML/DL don't pay.
  3. Alternative considered and rejected for this study: distilling the trained KAN into an MLP
     student defeats the point of testing KAN specifically (the report would just be re-measuring
     an MLP) — keep the LUT approach as the faithful representation of "KAN on embedded hardware."

---

## 8. Embedded benchmarking methodology (applies to all 4 deployed models)

- **Firmware**: one ESP-IDF project per method (`firmware/{ml,dl,gnn,kan}_inference/`) sharing a
  common harness (`firmware/esp32s3_common/`) for timing, serial I/O, and memory reporting, so the
  *only* thing that differs between the four firmware builds is the inference function.
- **Test vector delivery**: since there's no live seismometer attached to the ESP32-S3, feed a
  fixed, pre-agreed batch of test-set samples (already extracted to feature vectors / raw windows)
  over UART/serial from a PC into the device, run inference on-device, and read back predictions —
  this keeps embedded evaluation fully reproducible and comparable to the desktop numbers using
  the exact same test-set samples.
- **Latency**: `esp_timer_get_time()` around the inference call only (excludes UART I/O), mean +
  p95 over ≥200 repeated inferences per sample batch.
- **Memory**: flash size from `idf.py size` (per-component breakdown), peak heap/stack via
  `heap_caps_get_info()` / static analysis of `.bss`/`.data` for statically-allocated buffers.
- **Energy**: current draw via an external power profiler (e.g., a INA219-based board or a Nordic
  Power Profiler Kit II) at fixed voltage, integrated over the inference window → mJ/inference.
  If no power-measurement hardware is available, state this explicitly as a limitation and report
  latency × rated active-mode current as an estimated upper bound instead of a measured value —
  do not fabricate a measured number.
- **Accuracy on-device**: re-run the full test set's predictions on-device (batched over serial)
  and recompute macro-F1 from the device's actual INT8/LUT outputs, not just the desktop
  quantization-simulated accuracy — catches any float/fixed-point export bugs.

---

## 9. Final comparative analysis (the deliverable)

- **Master results table**: rows = {ML, DL, GNN, KAN}, columns = {macro-F1 (Core-6), macro-F1
  (Full-11), accuracy, flash KB, peak RAM KB, latency ms (mean/p95), energy mJ/inference,
  parameter count, preprocessing cost if applicable (GNN's peak/adjacency step)}.
- **Pareto-front plot**: accuracy (y) vs. latency (x), point size = memory footprint, one point
  per method — the figure that actually answers "which method should ship on this hardware."
- **Statistical significance table**: pairwise McNemar's test results between methods' on-device
  predictions.
- **Qualitative findings section**: deployability effort (lines of custom C code, hours to port),
  approximation costs paid (INT8 for ML/DL, densified-graph + custom-C for GNN, LUT for KAN),
  and a recommendation of which method to actually ship, conditioned on whether the priority is
  accuracy, latency, memory, or engineering effort.

---

## 10. Suggested timeline (10 weeks, adjust to actual availability)

| Week | Work |
|---|---|
| 1 | Finalize Core-6/Full-11 splits, shared preprocessing, shared metrics code (§3) |
| 2 | Track A: feature extraction + ML model selection (CV) |
| 3 | Track B: re-run ANN, build + train 1D-CNN, (optional) 2D-CNN on image |
| 4 | Track C: re-run GCN, extend GAT/GIN to Core-6, pick best variant |
| 5 | Track D: re-run KAN, run parameter-efficiency ablation vs. MLP |
| 6 | Compression: PTQ/QAT/pruning for ML+DL; fixed-`N_MAX` densification for GNN; LUT export for KAN |
| 7 | ESP-IDF harness (`esp32s3_common`) + Track A & B firmware, on-device benchmarking |
| 8 | Track C firmware (custom densified-GNN C code) + Track D firmware (spline LUT C code), on-device benchmarking |
| 9 | Full on-device accuracy re-validation, energy measurement pass, statistical tests |
| 10 | Comparative report, Pareto plots, recommendation writeup |

---

## 11. Key risks

1. **Rare-class statistical validity** — classes 3/7/8/11 (2–42 samples) cannot support a
   trustworthy test set under any method; the Full-11 numbers will be noisy regardless of
   architecture. Mitigate by reporting Core-6 as the headline result and Full-11 only as a
   qualitative "degradation" comparison, never as the primary claim.
2. **Train/test leakage via multi-station duplication** — must verify whether events repeat across
   the 10 `seis` stations before trusting the random split (§3.2).
3. **GNN/KAN have no native embedded runtime** — this is not a risk to mitigate away, it *is* one
   of the study's findings; budget real engineering time (§6, §7) for the custom-C paths rather
   than assuming an off-the-shelf exporter exists.
4. **No power-measurement hardware on hand** — confirm availability early (Week 6); if unavailable,
   the energy column becomes an estimated bound, stated as such, not a measured result.
5. **PSRAM-dependent boards vs. SRAM-only boards** — pick one specific ESP32-S3 board/module up
   front and state its exact RAM/flash config in the report; "ESP32-S3" alone is not one hardware
   spec.

---

## 12. Folder structure created for this work

```
wildlife_esp32_comparative_study/
├── PLAN.md                      <- this document
├── docs/                        <- hardware notes, dataset notes, literature
├── data/
│   ├── processed/                 <- extracted feature .npz per tier (Core-6 / Full-11)
│   └── splits/                    <- saved train/val/test index arrays (shared across methods)
├── notebooks/                   <- exploratory / prototyping notebooks for this study
├── src/
│   ├── data/                      <- netCDF loading, split generation
│   ├── features/                  <- peak extraction, handcrafted features, spectrogram utils
│   ├── models/{ml,dl,gnn,kan}/    <- model definitions per track
│   ├── training/                  <- one training script per track, shared protocol
│   ├── compression/                <- quantization, pruning, LUT export, graph densification
│   ├── export/                    <- TFLite / ESP-DL / C-code generation per track
│   └── evaluation/                 <- shared metrics, desktop + on-device benchmarking
├── firmware/
│   ├── esp32s3_common/            <- shared ESP-IDF harness (timing, serial I/O, mem reporting)
│   └── {ml,dl,gnn,kan}_inference/ <- one firmware project per method
├── results/{accuracy,latency,memory,energy,comparative_report}/
└── models_saved/{ml,dl,gnn,kan}/  <- trained model artifacts per track
```

Existing prior-work assets to reuse (not duplicate) from `PROJECT/`:
`ann_classifier_weighted.ipynb`, `ann_classifier_unweighted.ipynb`, `gcn_classifier.ipynb`,
`gcn.ipynb`, `gcn_weighted.ipynb`, `kan.ipynb`, `kan_weighted.ipynb`,
`class1_and_class6_{weighted,unweighted,balanced}.ipynb`, `seismic_savanna_GNN.ipynb`,
`best_species_gcn_classifier.pth`, `best_species_kan_classifier.pth`.
