# Wildlife Seismic Classification — ESP32-S3 Comparative Study

4-way comparative study (Classical ML vs. Deep Learning vs. GNN vs. KAN) for classifying animal
species from seismic footstep vibrations, with a real embedded-deployment leg on ESP32-S3.

Start here: **[PLAN.md](PLAN.md)** — full plan (dataset analysis, hardware constraints, per-method
methodology, embedded benchmarking protocol, timeline, risks).

Reference docs:
- [docs/dataset_notes.md](docs/dataset_notes.md) — confirmed structure/stats of the source netCDF
- [docs/esp32s3_hardware_notes.md](docs/esp32s3_hardware_notes.md) — target hardware budget

This folder is a scaffold (directory layout only, per PLAN.md §12) — implementation has not
started yet. Prior related notebooks/models already in `PROJECT/` (ANN, GCN, GAT, KAN) are reused,
not duplicated, per the plan.
