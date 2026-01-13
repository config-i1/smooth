# ETS Benchmark Setup Instructions

Benchmark comparison between smooth (Python), statsforecast, and smooth (R).

## Quick Start

```bash
cd python/tests/speed_tests

# Step 1: Extract data from R packages (run once)
Rscript extract_data_from_R.R

# Step 2: Install Python dependencies
pip install statsforecast tqdm

# Step 3: Run benchmark
python benchmark_ets.py --dataset m3_monthly --n_series 10
```

---

## Setup Details

### Step 1: Extract Data from R

Run the R script once to extract all competition datasets:

```bash
Rscript extract_data_from_R.R
```

This creates `benchmark_data/` folder with CSV files for each dataset.

**R packages required:**
```r
install.packages(c("Mcomp", "Tcomp"))
```

**Datasets extracted:**
| Dataset | Series | Horizon | Seasonality |
|---------|--------|---------|-------------|
| m3_monthly | 1,428 | 18 | 12 |
| m3_quarterly | 756 | 8 | 4 |
| m3_yearly | 645 | 6 | 1 |
| m3_other | 174 | 8 | 1 |
| m1_monthly | 617 | 18 | 12 |
| m1_quarterly | 203 | 8 | 4 |
| m1_yearly | 181 | 6 | 1 |
| tourism_monthly | 366 | 24 | 12 |
| tourism_quarterly | 427 | 8 | 4 |
| tourism_yearly | 518 | 4 | 1 |

### Step 2: Python Dependencies

Only two extra packages needed:

```bash
pip install statsforecast tqdm
```

| Package | Purpose |
|---------|---------|
| `statsforecast` | Competitor ETS implementation |
| `tqdm` | Progress bars |

Core packages (pandas, numpy) are already installed with smooth.

---

## Running Benchmarks

### Python Benchmark

```bash
# List available datasets
python benchmark_ets.py --list

# Quick test (10 series)
python benchmark_ets.py --dataset m3_monthly --n_series 10

# Full benchmark on one dataset
python benchmark_ets.py --dataset m3_monthly --output results.csv

# All datasets
python benchmark_ets.py --all --output results_all.csv

# Specific methods only
python benchmark_ets.py --dataset m3_monthly --methods smooth_py_back statsforecast
```

### R Benchmark

```bash
# Quick test
Rscript benchmark_ets_R.R --n_series 10

# Full M3 benchmark
Rscript benchmark_ets_R.R --dataset M3

# Specific subset
Rscript benchmark_ets_R.R --dataset M3 --subset MONTHLY
```

---

## Output

Results are saved as CSV files:

| File | Contents |
|------|----------|
| `results_*.csv` | Per-series detailed results |
| `results_*_summary.csv` | Aggregated metrics by method |

**Example output:**
```
Dataset: m3_monthly
----------------------------------------------------------------------
Method              MAE       RMSE      sMAPE      MASE   Avg Time
----------------------------------------------------------------------
smooth_py_back    1234.50   1567.80    12.34%    1.234    0.0450s
smooth_py_opt     1212.30   1545.60    12.12%    1.212    0.1230s
statsforecast     1256.70   1589.00    12.56%    1.256    0.0120s
----------------------------------------------------------------------
```

---

## File Structure

```
python/tests/speed_tests/
├── benchmark_data/           # Extracted data (created by R script)
│   ├── m3_monthly/
│   │   ├── metadata.csv
│   │   ├── series_0001_train.csv
│   │   ├── series_0001_test.csv
│   │   └── ...
│   ├── m3_quarterly/
│   └── ...
├── extract_data_from_R.R     # Data extraction script
├── utils_benchmark.py        # Data loading and metrics
├── benchmark_ets.py          # Python benchmark script
├── benchmark_ets_R.R         # R benchmark script
└── BENCHMARK_SETUP.md        # This file
```

---

## Methods Compared

| Method | Package | Description |
|--------|---------|-------------|
| `smooth_py_back` | smooth (Python) | ADAM with backcasting init |
| `smooth_py_opt` | smooth (Python) | ADAM with optimal init |
| `statsforecast` | statsforecast | AutoETS |
| `smooth_back` | smooth (R) | adam() with init="back" |
| `smooth_opt` | smooth (R) | adam() with init="opt" |
| `ets` | forecast (R) | ets() baseline |

---

## Troubleshooting

**"No datasets found"**
```bash
# Run the data extraction script first
Rscript extract_data_from_R.R
```

**R package not found**
```r
install.packages("Mcomp")
install.packages("Tcomp")  # Optional, for Tourism data
```

**statsforecast import error**
```bash
pip install --upgrade statsforecast
```
