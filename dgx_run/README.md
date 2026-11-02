# Running the rigorous benchmark on the DGX

Everything runs from one script in the background, so you can close your laptop. It uses
every free GPU, resumes after any interruption, and produces one zip file to send back.

## 1. Copy
Copy this whole `dgx_run/` folder to the DGX and put `dset_allspec_150_chunks_clean.nc`
either inside it or in its parent folder (or pass `--data /path/to/file.nc`).

## 2. Test first
```bash
cd dgx_run
bash run_dgx.sh --check      # shows which Python and dataset it found; starts nothing
bash run_dgx.sh --smoke      # roughly 10–20 minutes, runs in the background
```
This runs every model and every phase on 3,000 windows with tiny budgets, writing to
`results_rigorous_smoke/`. When the log ends with `send this file back`, it worked.
If anything fails, send me the log (`run_*.log`) and `results_rigorous_smoke/logs/`.

## 3. Full run
```bash
bash run_dgx.sh
```
Log out whenever you like.

**Python is found automatically.** The launcher prefers an environment named `project_venv`
(a virtualenv in the dgx_run folder, its parent, your home, `~/PROJECT/`, `/jupyter/$USER/`,
anywhere up to four levels below those, or a conda environment of that name), then any
activated environment, `~/PROJECT/venv`, and finally `python3`/`python`. It uses the first one
that has every package; if `project_venv` has PyTorch but is missing lighter packages, it
installs them into `project_venv`. It prints the interpreter it chose. To force one:
`PYTHON=/path/to/python bash run_dgx.sh`.

The dataset is also found automatically (next to the script, in `PROJECT/`, or up to four
levels below your home folder); otherwise pass `--data /path/to/dset_allspec_150_chunks_clean.nc`.

- Follow progress: `tail -f run_*.log` (prints done/total and an ETA every minute)
- Status: the launcher prints the exact command (it uses the chosen Python)
- Stop: `pkill -f rigorous_benchmark.py`; run `bash run_dgx.sh` again to resume where it stopped
- Failed jobs are listed in the log and retried automatically on the next run

## 4. Send back
`results_rigorous.zip` (next to this README). It contains tables, per-window predictions,
training histories, grid-search results and the configuration, but no raw data or model weights.

## How the GPUs are used
The benchmark is hundreds of small, independent training runs (folds x seeds x configurations
x models), so each free GPU runs `--workers-per-gpu` worker processes (default 2) that take
jobs from a shared queue. This scales almost linearly with the number of GPUs. DDP would
instead split a single run across GPUs and synchronise every batch, which gives little or no
speed-up for models this small.

A GPU counts as free if it uses less than 4 GB and 30% utilisation when the run starts, so
GPUs in use by others are left alone. Options:

| Option | Meaning |
|---|---|
| `--gpus 0,1,2,3` | use exactly these GPUs |
| `--max-gpus 4` | use at most 4 free GPUs |
| `--workers-per-gpu 3` | more workers per GPU (the models are small; 2 to 4 usually works) |
| `--max-used-mb 8000 --max-util 50` | looser definition of a free GPU |
| `--data PATH` | location of the NetCDF file |

## If the DGX uses a job scheduler (SLURM)
If long background processes are not allowed on the login node, submit it as a job instead:
```bash
#!/bin/bash
#SBATCH --job-name=seis-benchmark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=run_%j.log
cd "$SLURM_SUBMIT_DIR"
python rigorous_benchmark.py all
```

## What it runs
1. **Search:** GAT 16-configuration grid (including raw versus standardised node features),
   grids for ANN, feature MLP, 1D-CNN, KAN and GCN, and the six classical estimators,
   all selected on validation stations.
2. **Final:** 3 seeds of each selected model; GAT edge ablations (temporal-only, k-NN-only,
   random, none) and the node-scaling ablation.
3. **Extras:** attention-faithfulness test and peak-detector sensitivity sweep.
4. **Aggregate:** summary table, McNemar tests, single-thread host-CPU edge profile,
   `paper_results.json` and the zip.

Protocol: one cohort (windows with at least two peaks), 5-fold cross-validation grouped by
recording station, an inner station-grouped validation split for all tuning and early
stopping, each test fold used once.
