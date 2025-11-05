from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import math
import warnings

import numpy as np

try:
    # statsmodels provides a LOWESS implementation similar in spirit to R's
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess  # type: ignore
    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_STATSMODELS = False

try:
    # Optional: Friedman’s supersmoother (third-party). If unavailable, we fallback.
    from supersmoother import SuperSmoother  # type: ignore
    _HAS_SUPERSMOOTHER = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SUPERSMOOTHER = False


ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass
class MsDecomposeResult:
    y: np.ndarray
    states: np.ndarray
    initial: np.ndarray
    seasonal: Optional[List[np.ndarray]]
    fitted: np.ndarray
    gta: np.ndarray  # Global Trend, Additive (intercept, slope) on original/additive scale
    gtm: np.ndarray  # Global Trend, Multiplicative (exp of additive fit)
    loss: str
    lags: List[int]
    type: str  # "additive" or "multiplicative"
    y_name: str
    smoother: str


def _moving_average(y: np.ndarray, order: int) -> np.ndarray:
    if order <= 1:
        return y.copy()
    # Centered moving average with half-weights for ends when even window
    if (np.sum(~np.isnan(y)) == order) or (order % 2 == 1):
        weights = np.full(order, 1.0 / order)
    else:
        weights = np.concatenate(([0.5], np.ones(order - 1), [0.5])) / order
    valid = ~np.isnan(y)
    filtered = np.full_like(y, np.nan, dtype=float)
    yy = y[valid]
    # Convolve only valid part, then place back – keep it simple and centered
    conv = np.convolve(yy, weights, mode="same")
    filtered[valid] = conv
    return filtered


def _lowess(y: np.ndarray, frac: float) -> np.ndarray:
    if not _HAS_STATSMODELS:
        raise RuntimeError(
            "LOWESS smoother requires statsmodels. Install statsmodels or choose smoother='ma'."
        )
    x = np.arange(1, y.size + 1, dtype=float)
    valid = ~np.isnan(y)
    smoothed = np.full_like(y, np.nan, dtype=float)
    if valid.any():
        # statsmodels.lowess returns Nx2 array [x, y_smooth]
        out = sm_lowess(y[valid], x[valid], frac=frac, return_sorted=False)
        smoothed[valid] = out
    return smoothed


def _supsmu(y: np.ndarray, span: Union[str, float]) -> np.ndarray:
    if not _HAS_SUPERSMOOTHER:
        raise RuntimeError(
            "supsmu smoother requires the 'supersmoother' package. Install it or choose another smoother."
        )
    x = np.arange(1, y.size + 1, dtype=float)
    valid = ~np.isnan(y)
    smoothed = np.full_like(y, np.nan, dtype=float)
    if valid.any():
        model = SuperSmoother()
        # supersmoother auto-tunes internally; 'span' is not directly exposed. Keep API parity minimal.
        model.fit(x[valid], y[valid])
        smoothed[valid] = model.predict(x[valid])
    return smoothed


def msdecompose(
    y: ArrayLike,
    lags: Sequence[int] = (12,),
    type: str = "multiplicative",
    smoother: str = "ma",
    *,
    y_name: Optional[str] = None,
    lowess_order: Optional[float] = None,
) -> MsDecomposeResult:
    """Multiple seasonal classical decomposition (Python version).

    Parameters
    - y: 1D array-like time series
    - lags: Iterable of seasonal periods (e.g., [12] or [48, 336])
    - type: 'additive' or 'multiplicative'
    - smoother: 'ma' (moving average), 'lowess', or 'supsmu'
    - y_name: optional descriptive name of the series
    - lowess_order: optional order parameter mapped to LOWESS fraction as 1/order

    Returns MsDecomposeResult with fields modeled after the R implementation.
    """
    if type not in ("additive", "multiplicative"):
        raise ValueError("type must be 'additive' or 'multiplicative'")
    if smoother not in ("ma", "lowess", "supsmu"):
        raise ValueError("smoother must be 'ma', 'lowess', or 'supsmu'")

    y_arr = np.asarray(y, dtype=float).reshape(-1)
    n = y_arr.size
    if n == 0:
        raise ValueError("y must be non-empty")

    lags_sorted = sorted(set(int(l) for l in lags))
    seasonal_lags = any(l > 1 for l in lags_sorted)

    # Warn and switch to lowess if MA not applicable per R logic
    if smoother == "ma" and n <= min(lags_sorted or [1]):
        warnings.warn(
            "The minimum lag is larger than the sample size. Moving average does not work. Switching to LOWESS.",
            RuntimeWarning,
        )
        smoother = "lowess"

    # Track missing and non-positive handling for multiplicative
    y_na = np.isnan(y_arr)
    shifted_data = False

    if type == "multiplicative":
        if np.any(y_arr[~y_na] <= 0):
            y_na = np.logical_or(y_na, y_arr <= 0)
        with np.errstate(divide="ignore"):
            y_insample = np.log(y_arr)
    else:
        y_insample = y_arr.copy()

    # Impute missing via regression on polynomial + seasonal harmonics (simple analogue)
    if np.any(y_na):
        t = np.arange(1, n + 1, dtype=float)
        max_lag = max(lags_sorted) if lags_sorted else 1
        # degree up to min(max(trunc(n/10),1),5) similar to R
        degree = int(min(max(int(n / 10), 1), 5))
        poly_terms = np.vstack([t ** d for d in range(0, degree + 1)]).T
        # Seasonal harmonics using max lag
        harmonics = [np.sin(math.pi * t * k / max_lag) for k in range(1, max_lag + 1)]
        X = np.column_stack([poly_terms] + harmonics)
        valid = ~y_na
        # Least squares fit
        coef, *_ = np.linalg.lstsq(X[valid], y_insample[valid], rcond=None)
        y_insample[y_na] = (X @ coef)[y_na]

    # Helper: smoothing function
    def smoothing_function(series: np.ndarray, order: Optional[int]) -> np.ndarray:
        if smoother == "ma":
            ord_val = order if order and order > 0 else 1
            return _moving_average(series, int(ord_val))
        elif smoother == "lowess":
            # Map order to fraction ~ 1/order (as in the R code path)
            if lowess_order is not None:
                ord_val = lowess_order
            else:
                # Defaults: when order is None or equals certain values in R, they set to 1.5
                if (order is None) or (order == 1):
                    ord_val = 1.5
                else:
                    ord_val = float(order)
            frac = max(1.0 / max(ord_val, 1e-6), 1.0 / n)
            frac = min(max(frac, 1.0 / n), 1.0)  # clamp
            return _lowess(series, frac)
        else:  # supsmu
            # In R they choose span="cv" when order matches certain values; here we auto-tune
            span = "cv" if (order is None or order == 1) else max(1.0 / float(order), 1e-3)
            return _supsmu(series, span)

    # Build smoothed lists similar to R: ySmooth[[1]] holds actuals, subsequent are smoothed
    y_smooth: List[np.ndarray] = [y_insample.copy()]
    for lag in lags_sorted:
        y_smooth.append(smoothing_function(y_insample, order=lag))
    trend = y_smooth[len(lags_sorted)]

    # Produce cleared series per seasonal lag
    y_clear: List[np.ndarray] = []
    if seasonal_lags:
        for i in range(len(lags_sorted)):
            y_clear.append(y_smooth[i] - y_smooth[i + 1])

    # Seasonal patterns per lag
    patterns: Optional[List[np.ndarray]]
    if seasonal_lags:
        patterns_list: List[np.ndarray] = []
        for i, lag in enumerate(lags_sorted):
            pattern = np.full(n, np.nan, dtype=float)
            for j in range(lag):
                idx = np.arange(j, n, lag)
                y_seasonal = y_clear[i][idx]
                if smoother == "ma":
                    # average of available values for this seasonal position
                    val = np.nanmean(y_seasonal)
                    pattern[idx] = val
                else:
                    sm = smoothing_function(y_seasonal, order=None)
                    # align smoothed values back into series
                    pattern[idx[: sm[~np.isnan(sm)].size]] = sm[~np.isnan(sm)]
            # trim to length and de-mean
            pattern = pattern[:n]
            pattern = pattern - np.nanmean(pattern)
            patterns_list.append(pattern)
        patterns = patterns_list
    else:
        patterns = None

    # Deterministic trend (additive) fit on trend component for ADAM backcasting analogue
    t = np.arange(1, n + 1, dtype=float)
    valid_trend = ~np.isnan(trend)
    X = np.column_stack([np.ones(np.sum(valid_trend)), t[valid_trend]])
    coef_add, *_ = np.linalg.lstsq(X, trend[valid_trend], rcond=None)
    trend_determ_add = coef_add.copy()

    # Initial level and slope based on the last smoothed pre-trend component
    last_smooth = y_smooth[len(lags_sorted) - 1] if len(lags_sorted) > 0 else y_smooth[0]
    diffs = np.diff(last_smooth)
    initial_level = last_smooth[~np.isnan(last_smooth)][0]
    initial_trend = np.nanmean(diffs)
    if smoother == "ma":
        initial_level = initial_level - initial_trend * math.floor(max(lags_sorted) if lags_sorted else 1 / 2)
    initial = np.array([initial_level, initial_trend], dtype=float)

    # Move trend back to start it off-sample (align with ADAM behavior)
    if lags_sorted:
        trend_determ_add[0] = trend_determ_add[0] - trend_determ_add[1] * max(lags_sorted)

    # Return to original scale for multiplicative branch
    if type == "multiplicative":
        initial = np.exp(initial)
        trend_on_original = np.exp(trend)
        # compute multiplicative deterministic trend (exp of additive fit)
        gtm = np.exp(trend_determ_add)
        # recompute additive trend coefficients on exp(trend)
        coef_add2, *_ = np.linalg.lstsq(X, trend_on_original[valid_trend], rcond=None)
        trend_determ_add = coef_add2
        if lags_sorted:
            trend_determ_add[0] = trend_determ_add[0] - trend_determ_add[1] * max(lags_sorted)
        if patterns is not None:
            patterns = [np.exp(p) for p in patterns]
        gta = trend_determ_add
    else:
        # Handle non-positive trend by shifting for multiplicative fit
        non_positive = np.any(trend[valid_trend] <= 0)
        trend_for_mult = trend.copy()
        trend_min = 0.0
        if non_positive:
            trend_min = np.nanmin(trend_for_mult)
            trend_for_mult = trend_for_mult - trend_min + 1.0
        coef_mult, *_ = np.linalg.lstsq(X, np.log(trend_for_mult[valid_trend]), rcond=None)
        gtm = np.exp(coef_mult)
        if non_positive:
            gtm[0] = gtm[0] - trend_min - 1.0
        gta = trend_determ_add

    # Build states matrix and fitted values
    fitted = trend.copy()
    if patterns is not None:
        # states: [level, trend, seasonal_1, seasonal_2, ...]
        seasonal_mat = np.column_stack(patterns)
        states = np.column_stack([
            trend,
            np.concatenate([[np.nan], np.diff(trend)]),
            seasonal_mat,
        ])
        if type == "additive":
            for i, lag in enumerate(lags_sorted):
                fitted = fitted + np.resize(patterns[i], n)
        else:
            for i, lag in enumerate(lags_sorted):
                fitted = fitted * np.resize(patterns[i], n)
    else:
        states = np.column_stack([trend, np.concatenate([[np.nan], np.diff(trend)])])

    return MsDecomposeResult(
        y=y_arr,
        states=states,
        initial=initial,
        seasonal=patterns,
        fitted=fitted,
        gta=gta,
        gtm=gtm,
        loss="MSE",
        lags=list(lags_sorted),
        type=type,
        y_name=y_name or "y",
        smoother=smoother,
    )


