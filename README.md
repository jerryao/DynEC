# DynEC (Reproducible Package)

This repository accompanies the manuscript:

- **DynEC: Dynamic Evolutionary Clustering for Power Users via Multi-View Graph Neural Networks** (Zhao et al., manuscript under revision)

This folder is prepared for GitHub submission. It contains:

- `Sichuan2024Dataset/`: an anonymized, open-access subset of real smart meter data (City-A/B/C, year 2024).
- `Sichuan2024_Experiments/`: runnable evaluation script and generated result files.

## Sichuan2024Dataset

`Sichuan2024Dataset/` is an anonymized, open-access subset of real-world smart meter data collected from three representative districts (City-A/B/C) in Sichuan Province, China, covering the full year of 2024.

Key properties:

- **Time span:** 2024-01-01 to 2024-12-31 (366 days)
- **Resolution:** daily profiles with 24 hourly readings (`h0`..`h23`)
- **Scale:** City-A (800 users), City-B (500 users), City-C (650 users)
- **Ground truth:** includes `true_cluster` labels for ARI evaluation (see `Sichuan2024Dataset/README.md` for label provenance)
- **Privacy:** user identifiers are anonymized; load values are normalized to preserve shape patterns while masking absolute consumption
- **Policy context:** influenced by Sichuan Time-of-Use (TOU) pricing policy (official reference is recorded in `Sichuan2024Dataset/meta.json`)

## Citation

If you use this repository (code or dataset), please cite the paper above and/or cite this repository URL:

- https://github.com/jerryao/DynEC

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
