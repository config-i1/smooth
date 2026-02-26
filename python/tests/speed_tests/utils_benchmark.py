"""
Utility functions for ETS benchmark comparisons.
Loads data from CSV files extracted from R packages (Mcomp, Tcomp).

Run extract_data_from_R.R first to create the data files.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Default data directory (relative to this file)
DEFAULT_DATA_DIR = Path(__file__).parent / "benchmark_data"


# =============================================================================
# Error Metrics
# =============================================================================

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (0-200 scale)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator == 0, 1, denominator)
    return 200 * np.mean(np.abs(y_true - y_pred) / denominator)


def mase(y_true, y_pred, y_train, seasonality=1):
    """Mean Absolute Scaled Error."""
    y_train = np.asarray(y_train)
    if seasonality <= 1:
        naive_errors = np.abs(np.diff(y_train))
    else:
        naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])

    scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1
    if scale == 0:
        scale = 1
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale


def compute_all_metrics(y_true, y_pred, y_train, seasonality=1):
    """Compute all error metrics at once."""
    return {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'mase': mase(y_true, y_pred, y_train, seasonality),
    }


# =============================================================================
# Data Loading from Extracted CSV Files
# =============================================================================

class BenchmarkDataset:
    """Container for a benchmark dataset with train/test splits."""

    def __init__(self, name, data_dir=None):
        """
        Load a benchmark dataset from extracted CSV files.

        Args:
            name: Dataset name (e.g., 'm3_monthly', 'tourism_quarterly')
            data_dir: Directory containing extracted data (default: ./benchmark_data)
        """
        self.name = name
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.dataset_path = self.data_dir / name

        if not self.dataset_path.exists():
            available = list_available_datasets(self.data_dir)
            raise FileNotFoundError(
                f"Dataset '{name}' not found at {self.dataset_path}\n"
                f"Available datasets: {available}\n"
                f"Run 'Rscript extract_data_from_R.R' to create data files."
            )

        # Load metadata
        self.metadata = pd.read_csv(self.dataset_path / "metadata.csv")
        self.n_series = len(self.metadata)
        self.horizon = int(self.metadata['horizon'].iloc[0])
        self.frequency = int(self.metadata['frequency'].iloc[0])

    def __len__(self):
        return self.n_series

    def __iter__(self):
        """Iterate over all series."""
        for _, row in self.metadata.iterrows():
            yield self.get_series(row['series_id'])

    def get_series(self, series_id):
        """
        Load a single series.

        Returns:
            dict with 'train', 'test', 'horizon', 'frequency', 'series_id'
        """
        train_path = self.dataset_path / f"{series_id}_train.csv"
        test_path = self.dataset_path / f"{series_id}_test.csv"

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        meta_row = self.metadata[self.metadata['series_id'] == series_id].iloc[0]

        return {
            'series_id': series_id,
            'train': train_df['y'].values,
            'test': test_df['y'].values,
            'horizon': int(meta_row['horizon']),
            'frequency': int(meta_row['frequency']),
        }

    def get_series_ids(self):
        """Get list of all series IDs."""
        return self.metadata['series_id'].tolist()


def load_dataset(name, data_dir=None):
    """
    Load a benchmark dataset.

    Args:
        name: Dataset name (e.g., 'm3_monthly', 'tourism_quarterly')
        data_dir: Directory containing extracted data

    Returns:
        BenchmarkDataset object
    """
    return BenchmarkDataset(name, data_dir)


def list_available_datasets(data_dir=None):
    """List all available datasets in the data directory."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    if not data_dir.exists():
        return []

    datasets = []
    for path in data_dir.iterdir():
        if path.is_dir() and (path / "metadata.csv").exists():
            datasets.append(path.name)

    return sorted(datasets)


def print_available_datasets(data_dir=None):
    """Print available datasets with their metadata."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    datasets = list_available_datasets(data_dir)

    if not datasets:
        print(f"No datasets found in {data_dir}")
        print("Run 'Rscript extract_data_from_R.R' to extract data from R packages.")
        return

    print("\nAvailable datasets:")
    print("-" * 60)
    print(f"{'Dataset':<20} {'Series':<10} {'Horizon':<10} {'Frequency':<10}")
    print("-" * 60)

    for name in datasets:
        ds = BenchmarkDataset(name, data_dir)
        print(f"{name:<20} {ds.n_series:<10} {ds.horizon:<10} {ds.frequency:<10}")

    print("-" * 60)
    print(f"\nData directory: {data_dir}")


# =============================================================================
# Dataset Info (for reference)
# =============================================================================

EXPECTED_DATASETS = {
    'm3_monthly': {'horizon': 18, 'seasonality': 12, 'series': 1428},
    'm3_quarterly': {'horizon': 8, 'seasonality': 4, 'series': 756},
    'm3_yearly': {'horizon': 6, 'seasonality': 1, 'series': 645},
    'm3_other': {'horizon': 8, 'seasonality': 1, 'series': 174},
    'm1_monthly': {'horizon': 18, 'seasonality': 12, 'series': 617},
    'm1_quarterly': {'horizon': 8, 'seasonality': 4, 'series': 203},
    'm1_yearly': {'horizon': 6, 'seasonality': 1, 'series': 181},
    'tourism_monthly': {'horizon': 24, 'seasonality': 12, 'series': 366},
    'tourism_quarterly': {'horizon': 8, 'seasonality': 4, 'series': 427},
    'tourism_yearly': {'horizon': 4, 'seasonality': 1, 'series': 518},
}


def get_seasonality(dataset_name):
    """Get seasonality for a dataset."""
    if dataset_name in EXPECTED_DATASETS:
        return EXPECTED_DATASETS[dataset_name]['seasonality']
    # Infer from name
    if 'monthly' in dataset_name:
        return 12
    elif 'quarterly' in dataset_name:
        return 4
    else:
        return 1


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == '__main__':
    print_available_datasets()

    datasets = list_available_datasets()
    if datasets:
        print(f"\nTesting load of '{datasets[0]}'...")
        ds = load_dataset(datasets[0])
        print(f"  Loaded {ds.n_series} series")
        print(f"  Horizon: {ds.horizon}, Frequency: {ds.frequency}")

        # Test loading one series
        series = ds.get_series(ds.get_series_ids()[0])
        print(f"  First series: train={len(series['train'])}, test={len(series['test'])}")
