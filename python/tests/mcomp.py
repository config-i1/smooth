"""
M-Competition datasets loader.

Downloads and parses M1 and M3 competition datasets from the Monash Time Series
Forecasting Repository, providing an interface similar to R's Mcomp package.

Usage:
    from tests.mcomp import M1, M3, load_m1, load_m3

    # Access series by index (1-based, like R)
    series = M3[2568]
    print(series['x'])   # Training data
    print(series['xx'])  # Test data
    print(series['h'])   # Forecast horizon

    # Or load fresh from repository
    M3_data = load_m3()
"""
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

# Cache directory for downloaded data
CACHE_DIR = Path(__file__).parent / ".mcomp_cache"

# Zenodo record IDs and download URLs
# M1 datasets
M1_YEARLY_URL = "https://zenodo.org/records/4656193/files/m1_yearly_dataset.zip?download=1"
M1_QUARTERLY_URL = "https://zenodo.org/records/4656154/files/m1_quarterly_dataset.zip?download=1"
M1_MONTHLY_URL = "https://zenodo.org/records/4656159/files/m1_monthly_dataset.zip?download=1"

# M3 datasets
M3_YEARLY_URL = "https://zenodo.org/records/4656222/files/m3_yearly_dataset.zip?download=1"
M3_QUARTERLY_URL = "https://zenodo.org/records/4656262/files/m3_quarterly_dataset.zip?download=1"
M3_MONTHLY_URL = "https://zenodo.org/records/4656298/files/m3_monthly_dataset.zip?download=1"
M3_OTHER_URL = "https://zenodo.org/records/4656335/files/m3_other_dataset.zip?download=1"

# Forecast horizons by frequency (as defined in M-competitions)
M1_HORIZONS = {"yearly": 6, "quarterly": 8, "monthly": 18}
M3_HORIZONS = {"yearly": 6, "quarterly": 8, "monthly": 18, "other": 8}


class MCompSeries:
    """
    A single M-competition time series.

    Attributes
    ----------
    sn : str
        Series name/identifier
    x : np.ndarray
        Training data (in-sample)
    xx : np.ndarray
        Test data (out-of-sample)
    h : int
        Forecast horizon
    period : int
        Seasonal period (1=yearly, 4=quarterly, 12=monthly)
    type : str
        Series type (yearly, quarterly, monthly, other)
    n : int
        Length of training data
    """

    def __init__(
        self,
        sn: str,
        x: np.ndarray,
        xx: np.ndarray,
        h: int,
        period: int,
        series_type: str,
    ):
        self.sn = sn
        self.x = x
        self.xx = xx
        self.h = h
        self.period = period
        self.type = series_type
        self.n = len(x)

    def __repr__(self) -> str:
        return f"MCompSeries(sn='{self.sn}', n={self.n}, h={self.h}, type='{self.type}')"

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access for R-like interface."""
        return getattr(self, key)

    def keys(self) -> List[str]:
        """Return available keys."""
        return ["sn", "x", "xx", "h", "period", "type", "n"]


class MCompDataset:
    """
    M-competition dataset container.

    Provides dictionary-like access to series, supporting both 0-based and
    1-based indexing (1-based by default, like R's Mcomp package).

    Examples
    --------
    >>> M3 = load_m3()
    >>> series = M3[2568]  # 1-based index (R-style)
    >>> print(series['x'])  # Training data
    """

    def __init__(self, series_dict: Dict[int, MCompSeries], name: str = "M"):
        self._series = series_dict
        self._name = name
        self._keys_sorted = sorted(series_dict.keys())

    def __getitem__(self, key: int) -> MCompSeries:
        """
        Get series by 1-based index (R-style).

        Parameters
        ----------
        key : int
            1-based series index

        Returns
        -------
        MCompSeries
            The requested time series
        """
        if key in self._series:
            return self._series[key]
        else:
            raise KeyError(f"Series {key} not found in {self._name} dataset")

    def __len__(self) -> int:
        return len(self._series)

    def __iter__(self):
        for key in self._keys_sorted:
            yield self._series[key]

    def __repr__(self) -> str:
        return f"{self._name} Dataset: {len(self)} series"

    def keys(self) -> List[int]:
        """Return all series indices."""
        return self._keys_sorted

    def items(self):
        """Iterate over (index, series) pairs."""
        for key in self._keys_sorted:
            yield key, self._series[key]

    def subset(self, series_type: str) -> "MCompDataset":
        """
        Get subset of series by type.

        Parameters
        ----------
        series_type : str
            One of 'yearly', 'quarterly', 'monthly', 'other'

        Returns
        -------
        MCompDataset
            Subset containing only series of specified type
        """
        filtered = {k: v for k, v in self._series.items() if v.type == series_type}
        return MCompDataset(filtered, f"{self._name}_{series_type}")


def _download_and_extract(url: str, cache_dir: Path, filename: str) -> Path:
    """
    Download zip file and extract .tsf file.

    Parameters
    ----------
    url : str
        URL to download
    cache_dir : Path
        Directory to cache files
    filename : str
        Base filename (without extension)

    Returns
    -------
    Path
        Path to extracted .tsf file
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"{filename}.zip"
    tsf_path = cache_dir / f"{filename}.tsf"

    # Download if not cached
    if not tsf_path.exists():
        if not zip_path.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, zip_path)

        # Extract
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Find the .tsf file in the archive
            tsf_files = [n for n in zf.namelist() if n.endswith('.tsf')]
            if tsf_files:
                # Extract to cache dir with our naming
                with zf.open(tsf_files[0]) as src:
                    with open(tsf_path, 'wb') as dst:
                        dst.write(src.read())

        # Clean up zip
        zip_path.unlink()

    return tsf_path


def _parse_tsf_file(filepath: Path) -> tuple:
    """
    Parse a .tsf file from Monash repository.

    Parameters
    ----------
    filepath : Path
        Path to .tsf file

    Returns
    -------
    tuple
        (metadata dict, list of series dicts)
    """
    series_list = []
    metadata = {}
    in_data = False
    attribute_names = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse metadata
            if line.startswith("@"):
                lower_line = line.lower()
                if lower_line == "@data":
                    in_data = True
                    continue

                # Parse attribute definitions
                if lower_line.startswith("@attribute"):
                    parts = line.split()
                    if len(parts) >= 2:
                        attr_name = parts[1]
                        attribute_names.append(attr_name)
                    continue

                # Parse other metadata
                parts = line[1:].split(" ", 1)
                if len(parts) == 2:
                    key, value = parts
                    # Clean up value
                    value = value.strip()
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    metadata[key.lower()] = value
                continue

            # Parse data lines
            if in_data:
                # Format is: series_name:timestamp:value1,value2,value3,...
                # or: series_name:value1,value2,value3,...
                # Find the last colon that precedes numeric data
                parts = line.split(":")

                if len(parts) >= 2:
                    series_info = {}
                    series_info["series_name"] = parts[0].strip()

                    # Find which part contains the values (comma-separated numbers)
                    values_str = None
                    for i in range(len(parts) - 1, -1, -1):
                        part = parts[i].strip()
                        if "," in part or (part and part[0].isdigit()):
                            values_str = part
                            break

                    if values_str:
                        values = []
                        for v in values_str.split(","):
                            v = v.strip()
                            if v and v != "?":
                                try:
                                    values.append(float(v))
                                except ValueError:
                                    pass
                        series_info["values"] = np.array(values)
                        series_list.append(series_info)

    return metadata, series_list


def _parse_competition_tsf(
    filepath: Path,
    series_type: str,
    horizon: int,
    start_index: int = 1
) -> Dict[int, MCompSeries]:
    """
    Parse M-competition .tsf file and split into train/test.

    Parameters
    ----------
    filepath : Path
        Path to .tsf file
    series_type : str
        Type of series (yearly, quarterly, monthly, other)
    horizon : int
        Forecast horizon for test split
    start_index : int
        Starting index for series numbering (sequential from this value)

    Returns
    -------
    Dict[int, MCompSeries]
        Dictionary mapping series number to MCompSeries
    """
    period_map = {"yearly": 1, "quarterly": 4, "monthly": 12, "other": 1}
    period = period_map.get(series_type, 1)

    series_dict = {}
    metadata, raw_series = _parse_tsf_file(filepath)

    for idx, s in enumerate(raw_series):
        values = s.get("values", np.array([]))
        name = s.get("series_name", f"N{start_index + idx}")

        # Use sequential numbering from start_index
        sn = start_index + idx

        # Split into train and test
        if len(values) > horizon:
            x = values[:-horizon]
            xx = values[-horizon:]
        else:
            x = values
            xx = np.array([])

        series_dict[sn] = MCompSeries(
            sn=name,
            x=x,
            xx=xx,
            h=horizon,
            period=period,
            series_type=series_type,
        )

    return series_dict


def load_m3(force_download: bool = False) -> MCompDataset:
    """
    Load M3 competition dataset.

    Downloads data from Monash repository if not cached.

    Series are numbered according to M3 competition convention:
    - Yearly (1-645): 645 series
    - Quarterly (646-1401): 756 series
    - Monthly (1402-2829): 1428 series
    - Other (2830-3003): 174 series

    Parameters
    ----------
    force_download : bool, default=False
        If True, re-download even if cached

    Returns
    -------
    MCompDataset
        M3 dataset with all 3003 series

    Examples
    --------
    >>> M3 = load_m3()
    >>> series = M3[2568]
    >>> print(f"Training length: {len(series['x'])}")
    >>> print(f"Test length: {len(series['xx'])}")
    """
    cache_dir = CACHE_DIR / "m3"

    if force_download:
        import shutil
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    all_series = {}

    # M3 series numbering (matching R's Mcomp package):
    # Yearly: 1-645, Quarterly: 646-1401, Monthly: 1402-2829, Other: 2830-3003
    datasets = [
        ("yearly", M3_YEARLY_URL, "m3_yearly_dataset", 1),
        ("quarterly", M3_QUARTERLY_URL, "m3_quarterly_dataset", 646),
        ("monthly", M3_MONTHLY_URL, "m3_monthly_dataset", 1402),
        ("other", M3_OTHER_URL, "m3_other_dataset", 2830),
    ]

    for series_type, url, filename, start_idx in datasets:
        filepath = _download_and_extract(url, cache_dir, filename)
        horizon = M3_HORIZONS[series_type]
        series_dict = _parse_competition_tsf(
            filepath, series_type, horizon, start_index=start_idx
        )
        all_series.update(series_dict)

    print(f"Loaded M3 dataset: {len(all_series)} series")
    return MCompDataset(all_series, "M3")


def load_m1(force_download: bool = False) -> MCompDataset:
    """
    Load M1 competition dataset.

    Downloads data from Monash repository if not cached.

    Series are numbered according to M1 competition convention:
    - Yearly (1-181): 181 series
    - Quarterly (182-384): 203 series
    - Monthly (385-1001): 617 series

    Parameters
    ----------
    force_download : bool, default=False
        If True, re-download even if cached

    Returns
    -------
    MCompDataset
        M1 dataset with all 1001 series

    Examples
    --------
    >>> M1 = load_m1()
    >>> series = M1[1]
    >>> print(f"Training length: {len(series['x'])}")
    """
    cache_dir = CACHE_DIR / "m1"

    if force_download:
        import shutil
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    all_series = {}

    # M1 series numbering (matching R's Mcomp package):
    # Yearly: 1-181, Quarterly: 182-384, Monthly: 385-1001
    datasets = [
        ("yearly", M1_YEARLY_URL, "m1_yearly_dataset", 1),
        ("quarterly", M1_QUARTERLY_URL, "m1_quarterly_dataset", 182),
        ("monthly", M1_MONTHLY_URL, "m1_monthly_dataset", 385),
    ]

    for series_type, url, filename, start_idx in datasets:
        filepath = _download_and_extract(url, cache_dir, filename)
        horizon = M1_HORIZONS[series_type]
        series_dict = _parse_competition_tsf(
            filepath, series_type, horizon, start_index=start_idx
        )
        all_series.update(series_dict)

    print(f"Loaded M1 dataset: {len(all_series)} series")
    return MCompDataset(all_series, "M1")


# Lazy-loaded module-level datasets
_M1: Optional[MCompDataset] = None
_M3: Optional[MCompDataset] = None


class _LazyDataset:
    """Lazy-loading wrapper for M-competition datasets."""

    def __init__(self, loader, name: str):
        self._loader = loader
        self._data = None
        self._name = name

    def _ensure_loaded(self):
        if self._data is None:
            self._data = self._loader()

    def __getitem__(self, key):
        self._ensure_loaded()
        return self._data[key]

    def __len__(self):
        self._ensure_loaded()
        return len(self._data)

    def __iter__(self):
        self._ensure_loaded()
        return iter(self._data)

    def __repr__(self):
        if self._data is None:
            return f"{self._name} Dataset (not loaded yet - access any series to load)"
        return repr(self._data)

    def keys(self):
        self._ensure_loaded()
        return self._data.keys()

    def items(self):
        self._ensure_loaded()
        return self._data.items()

    def subset(self, series_type: str):
        self._ensure_loaded()
        return self._data.subset(series_type)


# Module-level lazy datasets for convenient access
M1 = _LazyDataset(load_m1, "M1")
M3 = _LazyDataset(load_m3, "M3")


if __name__ == "__main__":
    # Test the loader
    print("=" * 60)
    print("Loading M3 dataset...")
    print("=" * 60)
    m3 = load_m3()
    print(f"\nTotal series: {len(m3)}")

    # Show counts by type
    print("\nSeries by type:")
    for t in ["yearly", "quarterly", "monthly", "other"]:
        subset = m3.subset(t)
        print(f"  {t}: {len(subset)} series")

    # Show some examples
    print("\nFirst 5 series:")
    for i, (idx, series) in enumerate(m3.items()):
        if i >= 5:
            break
        print(f"  [{idx}] {series}")

    # Access specific series
    print("\n" + "=" * 60)
    print("Accessing specific series (M3 first yearly series):")
    print("=" * 60)
    first_key = m3.keys()[0]
    s = m3[first_key]
    print(f"  Name: {s['sn']}")
    print(f"  Training data (first 5): {s['x'][:5]}")
    print(f"  Training data length: {len(s['x'])}")
    print(f"  Test data: {s['xx']}")
    print(f"  Test data length: {len(s['xx'])}")
    print(f"  Horizon: {s['h']}")
    print(f"  Period: {s['period']}")
    print(f"  Type: {s['type']}")

    # Test M1
    print("\n" + "=" * 60)
    print("Loading M1 dataset...")
    print("=" * 60)
    m1 = load_m1()
    print(f"\nTotal series: {len(m1)}")

    print("\nSeries by type:")
    for t in ["yearly", "quarterly", "monthly"]:
        subset = m1.subset(t)
        print(f"  {t}: {len(subset)} series")
