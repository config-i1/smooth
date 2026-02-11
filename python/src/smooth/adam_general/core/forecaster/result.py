"""ForecastResult: structured container for ADAM forecasts."""

import numpy as np
import pandas as pd


class ForecastResult:
    """Structured forecast result matching R's ``forecast.adam()`` output.

    Attributes
    ----------
    mean : pd.Series
        Point forecasts indexed by forecast period.
    lower : pd.DataFrame or None
        Lower prediction bounds. Columns are quantile values (e.g. 0.025).
        None when ``interval="none"`` or ``side="upper"``.
    upper : pd.DataFrame or None
        Upper prediction bounds. Columns are quantile values (e.g. 0.975).
        None when ``interval="none"`` or ``side="lower"``.
    level : float or list of float
        Original confidence level(s) requested.
    side : str
        ``"both"``, ``"upper"``, or ``"lower"``.
    interval : str
        Resolved interval type (``"none"``, ``"approximate"``, ``"simulated"``).

    Notes
    -----
    Also supports DataFrame-style access (``result["mean"]``,
    ``result.columns``, ``result.shape``) for backward compatibility.
    """

    __slots__ = ("mean", "lower", "upper", "level", "side", "interval")

    def __init__(self, mean, lower, upper, level, side, interval):
        self.mean = mean
        self.lower = lower
        self.upper = upper
        self.level = level
        self.side = side
        self.interval = interval

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self):
        return repr(self.to_dataframe())

    def __len__(self):
        return len(self.mean)

    # ------------------------------------------------------------------
    # Backward-compatible DataFrame-style access
    # ------------------------------------------------------------------

    @property
    def columns(self):
        """Column names matching the flat DataFrame layout."""
        return self.to_dataframe().columns

    @property
    def shape(self):
        """(n_rows, n_columns) matching the flat DataFrame layout."""
        n_cols = 1  # mean
        if self.lower is not None:
            n_cols += self.lower.shape[1]
        if self.upper is not None:
            n_cols += self.upper.shape[1]
        return (len(self.mean), n_cols)

    @property
    def index(self):
        """Forecast period index."""
        return self.mean.index

    @property
    def values(self):
        """Numpy array matching the flat DataFrame layout."""
        return self.to_dataframe().values

    def __getitem__(self, key):
        """Access columns by name (e.g. ``result["mean"]``)."""
        return self.to_dataframe()[key]

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """Convert to a flat DataFrame with prefixed column names."""
        parts = {"mean": self.mean.values}

        if self.lower is not None:
            for col in self.lower.columns:
                parts[f"lower_{col}"] = self.lower[col].values
        if self.upper is not None:
            for col in self.upper.columns:
                parts[f"upper_{col}"] = self.upper[col].values

        return pd.DataFrame(parts, index=self.mean.index)
