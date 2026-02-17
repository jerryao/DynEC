# DynEC (Reproducible Package)

This folder is prepared for GitHub submission. It contains:

- `Sichuan2024Dataset/`: an anonymized, open-access subset of real smart meter data (City-A/B/C, year 2024).
- `Sichuan2024_Experiments/`: runnable evaluation script and generated result files.

## Quick Start

### 1) Prerequisites

- Python 3.10+ recommended
- Required Python packages: `numpy`, `pandas`

### 2) Run evaluation (monthly)

From this `DynEC/` directory:

```bash
python Sichuan2024_Experiments/run_2024_experiment.py --time-step month --method both --k 8 --max-iter 50 --seeds 1,2,3 --internal-metrics-sample 200 --out-dir Sichuan2024_Experiments/results
```

Outputs:

- `Sichuan2024_Experiments/results/results_*.csv`
- `Sichuan2024_Experiments/results/summary_*.json`

### 3) Run evaluation (daily)

```bash
python Sichuan2024_Experiments/run_2024_experiment.py --time-step day --method both --k 8 --max-iter 50 --seeds 1,2,3 --internal-metrics-sample 200 --out-dir Sichuan2024_Experiments/results
```

## Dataset Notes

See `Sichuan2024Dataset/README.md` for:

- file formats (`profiles_daily.csv`, `users.csv`, `events.csv`, `meta.json`)
- the definition and provenance of `true_cluster` used for ARI reporting

## Reproducibility Notes

- Use `--seeds` to report mean ± std over multiple runs.
- `--internal-metrics-sample` approximates Silhouette/DBI by sampling users for efficiency.
