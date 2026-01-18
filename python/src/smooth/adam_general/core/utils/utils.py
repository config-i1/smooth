import numpy as np
from scipy import stats
from scipy.linalg import eigvals
from scipy.special import gamma, digamma, beta
import pandas as pd

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

def msdecompose(y, lags=[12], type="additive", smoother="lowess"):
    """
    Multiple seasonal decomposition of time series with multiple frequencies.

    This function performs **classical seasonal decomposition** for time series with multiple
    seasonal patterns (e.g., hourly data with daily and weekly seasonality, or daily data
    with weekly and yearly patterns). It extends the standard STL decomposition to handle
    multiple seasonal periods simultaneously.

    The decomposition separates the time series into:

    - **Trend**: Long-term movement (captured via smoothing)
    - **Seasonal components**: One for each seasonal period in `lags`
    - **Remainder** (not explicitly returned but implied): y - trend - seasonals

    **Decomposition Method**:

    For **additive** decomposition:

    .. math::

        y_t = \\text{Trend}_t + \\sum_i \\text{Seasonal}_i(t) + \\epsilon_t

    For **multiplicative** decomposition:

    .. math::

        y_t = \\text{Trend}_t \\times \\prod_i \\text{Seasonal}_i(t) \\times \\epsilon_t

    **Algorithm Steps**:

    1. **Log Transform** (if multiplicative): Apply log to convert to additive form
    2. **Missing Value Imputation**: Fill NaN values using polynomial + Fourier regression
    3. **Iterative Smoothing**: For each lag period (sorted ascending):

       - Apply smoother with window = lag period
       - Extract seasonal pattern as residual from next smoother level
       - Remove seasonal mean to center patterns

    4. **Trend Extraction**: Final smoothed series is the trend
    5. **Initial States**: Compute level and slope from trend for model initialization

    **Smoother Types**:

    - **"ma"**: Moving average with window = lag period. Fast but less flexible.
      Automatically switches to LOWESS if sample size < minimum lag.

    - **"lowess"** (default): Locally weighted scatterplot smoothing. Robust to outliers,
      adapts to local patterns. Equivalent to R's `lowess()`.

    - **"supsmu"**: Friedman's super smoother (uses LOWESS implementation in Python).
      Adaptive bandwidth selection.

    - **"global"**: Global linear regression with intercept and deterministic trend.
      Fits a straight line to the data.

    Parameters
    ----------
    y : array-like
        Time series data to decompose. Can contain NaN values (will be imputed).
        Shape: (T,) where T is the number of observations.

    lags : list or array, default=[12]
        Seasonal periods to extract. Examples:

        - [12]: Monthly data with yearly seasonality
        - [24]: Hourly data with daily seasonality
        - [7, 365.25]: Daily data with weekly and yearly seasonality
        - [24, 168]: Hourly data with daily (24h) and weekly (7×24=168h) patterns

        Must contain positive integers. Lags are sorted automatically.

    type : str, default="additive"
        Decomposition type:

        - **"additive"**: Components are summed (for stable seasonality)
        - **"multiplicative"**: Components are multiplied (for proportional seasonality,
          requires y > 0)

    smoother : str, default="lowess"
        Smoothing method for trend and seasonal extraction:

        - **"lowess"**: LOWESS with adaptive span (recommended, **default**)
        - **"supsmu"**: Super smoother (uses LOWESS in Python)
        - **"ma"**: Simple moving average (faster but less robust)
        - **"global"**: Global linear regression (straight line fit)

    Returns
    -------
    dict
        Dictionary containing decomposition results:

        - **'states'** (numpy.ndarray): Matrix of extracted states, shape (T, n_states).
          Columns are [Level, Trend, Seasonal_1, Seasonal_2, ..., Seasonal_n].
          These states can be used as initial values for ADAM model estimation.

        - **'initial'** (dict): Dictionary with initial values, containing:
          - **'nonseasonal'** (dict): Dictionary with 'level' and 'trend' keys.
            Level is the initial value at t=0, trend is the slope per period.
            Computed from the first non-NaN trend values and adjusted back by lags_max.
          - **'seasonal'** (list of numpy.ndarray): List of seasonal initial values.
            Each seasonal[i] contains the first lags[i] values from pattern i.

        - **'trend'** (numpy.ndarray): Extracted trend component, shape (T,).
          Long-term movement after removing seasonal patterns.

        - **'seasonal'** (list of numpy.ndarray): Seasonal patterns, one array per lag.
          Each seasonal[i] has shape (T,) and is centered (mean = 0).

        - **'component'** (list): Component type descriptions (for compatibility)

        - **'lags'** (numpy.ndarray): Sorted unique lag periods used

        - **'type'** (str): Decomposition type ('additive' or 'multiplicative')

    Raises
    ------
    ValueError
        If type not in ['additive', 'multiplicative']
        If smoother not in ['ma', 'lowess', 'supsmu']
    ImportError
        If smoother='lowess' or 'supsmu' but statsmodels is not installed

    Notes
    -----
    **Missing Values**:

    NaN values are automatically imputed using a regression model:

    .. math::

        \\hat{y}_t = \\sum_{k=0}^d \\beta_k t^k + \\sum_{j=1}^m \\alpha_j \\sin(\\pi t j / m)

    where d is polynomial degree (up to 5) and m is the maximum lag.
    This preserves trend and seasonal structure during imputation.

    **Multiplicative Decomposition**:

    Requires strictly positive data. If y ≤ 0, those values are treated as missing.
    Internally works on log(y), then exponentiates results.

    **Smoother Span Selection**:

    For LOWESS, span (bandwidth) is automatically selected based on lag period:

    - For lag = 1: span = 2/3 (R's default)
    - For lag = T: span = 2/3
    - Otherwise: span = 1 / lag
    - Minimum span: 3 / T (ensures smoothness)

    **Seasonal Centering**:

    Each seasonal pattern is centered to have mean zero. This ensures identifiability:
    trend captures the level, seasonals capture deviations.

    **Performance**:

    - Moving average: Very fast (~1ms for T=1000)
    - LOWESS: Moderate (~10-50ms depending on T)
    - Multiple lags: Time scales linearly with number of lags

    **Use in ADAM**:

    The decomposition is used for initial state estimation when initial="backcasting"
    or when the model includes seasonal components. The extracted states provide
    reasonable starting values for the level, trend, and seasonal components.

    **Comparison to STL**:

    Unlike STL (Seasonal-Trend decomposition using Loess), which handles only one
    seasonal period, msdecompose handles **multiple** seasonal periods by iteratively
    removing each seasonal component.

    See Also
    --------
    creator : Uses msdecompose results for initial state estimation
    initialiser : May use decomposition results for parameter initialization

    Examples
    --------
    Decompose monthly data with yearly seasonality::

        >>> y = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        ...               115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140])
        >>> result = msdecompose(y, lags=[12], type='additive', smoother='lowess')
        >>> print(result['trend'])  # Trend component
        >>> print(result['seasonal'][0])  # Yearly seasonal pattern
        >>> print(result['initial']['nonseasonal']['level'])  # Initial level
        >>> print(result['initial']['nonseasonal']['trend'])  # Initial trend
        >>> print(result['initial']['seasonal'][0])  # First 12 seasonal values

    Decompose hourly data with daily and weekly seasonality::

        >>> hourly_data = np.random.randn(24 * 7 * 4)  # 4 weeks of hourly data
        >>> result = msdecompose(hourly_data, lags=[24, 168],  # 24h and 7*24h
        ...                      type='additive', smoother='lowess')
        >>> daily_pattern = result['seasonal'][0]  # 24-hour pattern
        >>> weekly_pattern = result['seasonal'][1]  # Weekly pattern

    Multiplicative decomposition for positive data::

        >>> sales = np.array([100, 120, 150, 140, 130, 160, 200, 210, 180, 140, 110, 130])
        >>> result = msdecompose(sales, lags=[12], type='multiplicative')
        >>> # Seasonality proportional to level

    Use decomposition for ADAM initialization::

        >>> result = msdecompose(y, lags=[12], type='additive')
        >>> initial_level = result['initial']['nonseasonal']['level']
        >>> initial_trend = result['initial']['nonseasonal']['trend']
        >>> initial_seasonal = result['initial']['seasonal'][0]  # First 12 values
        >>> # Pass to ADAM's initials parameter
    """
    # Argument validation
    if type not in ["additive", "multiplicative"]:
        raise ValueError("type must be 'additive' or 'multiplicative'")
    if smoother not in ["ma", "lowess", "supsmu", "global"]:
        raise ValueError("smoother must be 'ma', 'lowess', 'supsmu', or 'global'")

    # Check statsmodels availability for lowess
    if smoother in ["lowess", "supsmu"] and not HAS_STATSMODELS:
        raise ImportError("statsmodels is required for lowess/supsmu smoother. "
                         "Install it with: pip install statsmodels")

    # Variable name handling
    y_name = "y"

    # Data preparation
    y = np.asarray(y)
    obs_in_sample = len(y)

    # Handle empty lags case - treat as lags=[1] to match R behavior
    # In R, msdecompose is never called with empty lags, but the Python code
    # filters lags to remove lag=1, which can result in empty lags.
    # We treat empty lags as lags=[1] to ensure consistent smoothing behavior.
    if len(lags) == 0:
        lags = [1]

    seasonal_lags = any(lag > 1 for lag in lags)

    # Smoothing function definition
    def smoothing_function_ma(y, order):
        """Moving average smoother"""
        # Convert y to float to avoid integer overflow
        y = y.astype(float)
        if order == np.sum(~np.isnan(y)) or order % 2 != 0:
            # Odd order or order equals non-NA count: simple moving average
            k = order
            weights = np.ones(k) / order
        else:
            # Even order: use filter of length order + 1
            k = order + 1
            weights = np.array([0.5] + [1] * (order - 1) + [0.5]) / order
        half_k = (k - 1) // 2  # e.g., for k=13, half_k=6
        trend = np.full_like(y, np.nan)
        for i in range(half_k, len(y) - half_k):
            trend[i] = np.sum(y[i - half_k : i + half_k + 1] * weights)
        return trend

    def smoothing_function_lowess(y, order):
        """LOWESS smoother (equivalent to R's lowess/supsmu)"""
        y = y.astype(float)
        n = len(y)
        x = np.arange(1, n + 1)

        # Calculate span similar to R's supsmu
        # R uses span = max(3/n, 1/order) for supsmu
        if order == 1:
            # R's lowess uses f = 2/3 for default
            span = 2/3
        elif order == lags[len(lags)-1] or order == obs_in_sample:
             span = 2/3 # 1/(1.5)
        elif order > n:
            span = 3 / n
        else:
            span = 1 / order

        # Ensure span is reasonable (between 0 and 1)
        span = max(min(span, 1.0), 3 / n)

        # Handle missing values
        valid_mask = ~np.isnan(y)
        if not np.any(valid_mask):
            return np.full_like(y, np.nan)

        # Apply LOWESS
        # statsmodels lowess returns array of (x, y) pairs
        # R's lowess uses iter=3 by default
        smoothed = sm_lowess(y[valid_mask], x[valid_mask], frac=span,
                            return_sorted=True, it=3)

        # Map back to original indices
        result = np.full_like(y, np.nan)
        result[valid_mask] = smoothed[:, 1]

        return result

    def smoothing_function_global(y, order=None):
        """Global linear regression smoother"""
        y = y.astype(float)
        n = len(y)
        X = np.column_stack([np.ones(n), np.arange(1, n + 1)])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return y - (y - X @ coef)  # Returns fitted values: X @ coef

    # Initial data processing
    # obs_in_sample is already defined above

    # Select smoothing function based on smoother type
    if smoother == "ma":
        smoothing_function = smoothing_function_ma
    elif smoother == "global":
        smoothing_function = smoothing_function_global
    else:  # lowess or supsmu
        smoothing_function = smoothing_function_lowess

    # DEBUG: Print selected smoother
    import os
    DEBUG_DECOMP = os.environ.get("DEBUG_CREATOR") == "True"
    if DEBUG_DECOMP:
        print(f"[MSDECOMPOSE DEBUG] Smoother selected: {smoother}")

    # Check if MA smoother works with the given sample size
    if smoother == "ma" and obs_in_sample <= min(lags):
        import warnings
        warnings.warn(
            "The minimum lag is larger than the sample size. "
            "Moving average does not work in this case. "
            "Switching smoother to LOWESS.",
            stacklevel=2
        )
        smoother = "lowess"
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels is required for lowess smoother. "
                             "Install it with: pip install statsmodels")
        smoothing_function = smoothing_function_lowess

    y_na_values = np.isnan(y)
    if type == "multiplicative":
        shifted_data = False
        if np.any(y[~y_na_values] <= 0):
            y_na_values = y_na_values | (y <= 0)
        y_insample = np.log(y)
    else:
        y_insample = y.copy()

    # Missing value imputation
    if np.any(y_na_values):
        degree = min(max(int(np.floor(obs_in_sample / 10)), 1), 5)
        t = np.arange(1, obs_in_sample + 1)
        X_poly = np.vander(t, degree + 1, increasing=True)
        max_lag = np.max(lags)
        X_sin = np.column_stack([np.sin(np.pi * t * k / max_lag) for k in range(1, max_lag + 1)])
        X = np.column_stack((X_poly, X_sin))
        coef = np.linalg.lstsq(X[~y_na_values], y_insample[~y_na_values], rcond=None)[0]
        y_insample[y_na_values] = X[y_na_values] @ coef

    # Smoothing and trend extraction
    lags = np.sort(np.unique(lags))
    
    lags_length = len(lags)
    y_smooth = [None] * (lags_length + 1)
    y_smooth[0] = y_insample
    for i in range(lags_length):
        y_smooth[i + 1] = smoothing_function(y_insample, order=lags[i])
    trend = y_smooth[lags_length]

    # Cleared series
    if seasonal_lags:
        y_clear = [None] * lags_length
        for i in range(lags_length):
            y_clear[i] = y_smooth[i] - y_smooth[i + 1]

    # Seasonal patterns
    if seasonal_lags:
        patterns = []
        for i in range(lags_length):
            pattern_i = np.full(obs_in_sample, np.nan)
            for j in range(lags[i]):
                indices = np.arange(j, obs_in_sample, lags[i])
                y_seasonal = y_clear[i][indices]
                y_seasonal_non_na = y_seasonal[~np.isnan(y_seasonal)]
                
                if len(y_seasonal_non_na) > 0:
                    if smoother == "ma":
                        y_seasonal_smooth = np.mean(y_seasonal_non_na)
                        pattern_i[indices[~np.isnan(y_seasonal)]] = y_seasonal_smooth
                    else:
                        y_seasonal_smooth = smoothing_function(y_seasonal_non_na, order=obs_in_sample)
                        pattern_i[indices[~np.isnan(y_seasonal)]] = y_seasonal_smooth
            
            if np.any(~np.isnan(pattern_i)):
                pattern_i -= np.nanmean(pattern_i)
            patterns.append(pattern_i)
    else:
        patterns = None

    # Initial level and trend
    # Create initial as a dict with nonseasonal and seasonal components
    initial = {"nonseasonal": {}, "seasonal": []}

    # Calculate nonseasonal initial values (level and trend)
    data_for_initial = y_smooth[lags_length]  # Matches R's ySmooth[[ySmoothLength]]
    valid_data_for_initial = data_for_initial[~np.isnan(data_for_initial)]
    if len(valid_data_for_initial) == 0:
        init_level = 0.0
        init_trend = 0.0
    else:
        init_level = valid_data_for_initial[0]
        diffs = np.diff(valid_data_for_initial)
        init_trend = np.nanmean(diffs) if len(diffs) > 0 else 0.0

    lags_max = max(lags)

    # Fix the initial for MA smoother (lines 200-202 in R)
    if smoother == "ma":
        init_level -= init_trend * np.floor(lags_max / 2)

    # Lag things back to get values useful for ADAM (lines 204-206 in R)
    init_level -= init_trend * lags_max

    # Store in nonseasonal dict
    initial["nonseasonal"] = {"level": init_level, "trend": init_trend}

    # Deterministic trend fit to the smooth series
    # This is required by adam() with backcasting
    valid_trend = ~np.isnan(trend)
    X_determ = np.column_stack([np.ones(obs_in_sample), np.arange(1, obs_in_sample + 1)])

    # DEBUG: Print trend values before regression
    if DEBUG_DECOMP:
        print(f"[MSDECOMPOSE DEBUG] Trend before regression:")
        print(f"  trend length: {len(trend)}")
        print(f"  trend valid count: {np.sum(valid_trend)}")
        print(f"  trend[0:5] (first 5): {trend[0:5]}")
        print(f"  trend[-5:] (last 5): {trend[-5:]}")
        print(f"  trend mean: {np.mean(trend[valid_trend])}")

    gta = np.linalg.lstsq(X_determ[valid_trend], trend[valid_trend], rcond=None)[0]

    # DEBUG: Print gta before adjustment
    if DEBUG_DECOMP:
        print(f"[MSDECOMPOSE DEBUG] gta before adjustment: {gta}")
        print(f"[MSDECOMPOSE DEBUG] max(lags): {max(lags)}")
        print(f"[MSDECOMPOSE DEBUG] gta adjustment: -{gta[1]} * {max(lags)} = {-gta[1] * max(lags)}")

    # Move the trend back to start it off-sample in case of ADAM
    gta[0] = gta[0] - gta[1] * max(lags)

    # Return to the original scale
    if type == "multiplicative":
        # Transform nonseasonal initial values back to exponential scale
        initial["nonseasonal"]["level"] = np.exp(initial["nonseasonal"]["level"])
        initial["nonseasonal"]["trend"] = np.exp(initial["nonseasonal"]["trend"])
        trend_exp = np.exp(trend)

        # Sort out additive/multiplicative trend for ADAM
        gtm = np.exp(gta.copy())

        # Recalculate gta on the exponential scale
        gta = np.linalg.lstsq(X_determ[valid_trend], trend_exp[valid_trend], rcond=None)[0]
        gta[0] = gta[0] - gta[1] * max(lags)

        trend = trend_exp
        if seasonal_lags:
            patterns = [np.exp(pattern) for pattern in patterns]
    else:
        # Get the deterministic multiplicative trend for additive decomposition
        # Shift the trend if it contains negative values
        non_positive_values = False
        trend_for_gtm = trend.copy()
        if np.any(trend[valid_trend] <= 0):
            non_positive_values = True
            trend_min = np.nanmin(trend)
            trend_for_gtm = trend - trend_min + 1

        gtm = np.linalg.lstsq(X_determ[valid_trend], np.log(trend_for_gtm[valid_trend]), rcond=None)[0]
        gtm = np.exp(gtm)

        # Correct the initial level
        if non_positive_values:
            gtm[0] = gtm[0] - trend_min - 1

    # Extract seasonal initial values (first lags[i] values from each pattern)
    # Lines 256-258 in R
    if seasonal_lags:
        for i in range(lags_length):
            initial["seasonal"].append(patterns[i][:lags[i]])

    # Fitted values and states
    y_fitted = trend.copy()
    if seasonal_lags:
        states = np.column_stack((trend, np.concatenate(([np.nan], np.diff(trend))), np.column_stack(patterns)))
        if type == "additive":
            for i in range(lags_length):
                pattern_rep = np.tile(patterns[i], int(np.ceil(obs_in_sample / lags[i])))[:obs_in_sample]
                y_fitted += pattern_rep
        else:
            for i in range(lags_length):
                pattern_rep = np.tile(patterns[i], int(np.ceil(obs_in_sample / lags[i])))[:obs_in_sample]
                y_fitted *= pattern_rep
    else:
        states = np.column_stack((trend, np.concatenate(([np.nan], np.diff(trend)))))

    # Fix for the "NA" in trend in case of global trend (lines 266-268 in R)
    if smoother == "global":
        states[:, 1] = np.nanmean(states[:, 1])

    # Return structure
    result = {
        "y": y,
        "states": states,
        "initial": initial,
        "seasonal": patterns,
        "fitted": y_fitted,
        # gta is the Global Trend, Additive. gtm is the Global Trend, Multiplicative
        "gta": gta,
        "gtm": gtm,
        "loss": "MSE",
        "lags": lags,
        "type": type,
        "yName": y_name,
        "smoother": smoother
    }
    return result


def calculate_acf(data, nlags=40):
    """
    Calculate Autocorrelation Function for numpy array or pandas Series.
    
    Parameters:
    data (np.array or pd.Series): Input time series data
    nlags (int): Number of lags to calculate ACF for
    
    Returns:
    np.array: ACF values
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    return acf(data, nlags=nlags, fft=False)


def calculate_pacf(data, nlags=40):
    """
    Calculate Partial Autocorrelation Function for numpy array or pandas Series.
    
    Parameters:
    data (np.array or pd.Series): Input time series data
    nlags (int): Number of lags to calculate PACF for
    
    Returns:
    np.array: PACF values
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    return pacf(data, nlags=nlags, method='ols')


def calculate_likelihood(distribution, Etype, y, y_fitted, scale, other):
    
    # Fixes the output dimension
    y = y.reshape(-1,1) 
    
    if distribution == "dnorm":
        if Etype == "A":
            return stats.norm.logpdf(y, loc=y_fitted, scale=scale)
        else:  # "M"
            return stats.norm.logpdf(y, loc=y_fitted, scale=scale*y_fitted)
    elif distribution == "dlaplace":
        if Etype == "A":
            return stats.laplace.logpdf(y, loc=y_fitted, scale=scale)
        else:  # "M"
            return stats.laplace.logpdf(y, loc=y_fitted, scale=scale*y_fitted)
    elif distribution == "ds":
        if Etype == "A":
            return stats.t.logpdf(y, df=2, loc=y_fitted, scale=scale)
        else:  # "M"
            return stats.t.logpdf(y, df=2, loc=y_fitted, scale=scale*np.sqrt(y_fitted))
    elif distribution == "dgnorm":
        # Implement generalized normal distribution
        pass
    elif distribution == "dalaplace":
        # Implement asymmetric Laplace distribution
        pass
    elif distribution == "dlnorm":
        return stats.lognorm.logpdf(y, s=scale, scale=np.exp(np.log(y_fitted) - scale**2/2))
    elif distribution == "dllaplace":
        return stats.laplace.logpdf(np.log(y), loc=np.log(y_fitted), scale=scale) - np.log(y)
    elif distribution == "dls":
        return stats.t.logpdf(np.log(y), df=2, loc=np.log(y_fitted), scale=scale) - np.log(y)
    elif distribution == "dlgnorm":
        # Implement log-generalized normal distribution
        pass
    elif distribution == "dinvgauss":
        return stats.invgauss.logpdf(y, mu=np.abs(y_fitted), scale=np.abs(scale/y_fitted))
    elif distribution == "dgamma":
        return stats.gamma.logpdf(y, a=1/scale, scale=scale*np.abs(y_fitted))

def calculate_entropy(distribution, scale, other, obsZero, y_fitted):
    if distribution == "dnorm":
        return obsZero * (np.log(np.sqrt(2 * np.pi) * scale) + 0.5)
    elif distribution == "dlnorm":
        return obsZero * (np.log(np.sqrt(2 * np.pi) * scale) + 0.5) - scale**2 / 2
    elif distribution == "dlogis":
        return obsZero * 2
    elif distribution in ["dlaplace", "dllaplace", "dalaplace"]:
        return obsZero * (1 + np.log(2 * scale))
    elif distribution in ["ds", "dls"]:
        return obsZero * (2 + 2 * np.log(2 * scale))
    elif distribution in ["dgnorm", "dlgnorm"]:
        return obsZero * (1 / other - np.log(other / (2 * scale * gamma(1 / other))))
    elif distribution == "dt":
        return obsZero * ((scale + 1) / 2 * (digamma((scale + 1) / 2) - digamma(scale / 2)) +
                          np.log(np.sqrt(scale) * beta(scale / 2, 0.5)))
    elif distribution == "dinvgauss":
        return 0.5 * (obsZero * (np.log(np.pi / 2) + 1 + np.log(scale)) - np.sum(np.log(y_fitted)))
    elif distribution == "dgamma":
        return obsZero * (1 / scale + np.log(gamma(1 / scale)) + (1 - 1 / scale) * digamma(1 / scale)) + \
               np.sum(np.log(scale * y_fitted))

def calculate_multistep_loss(loss, adamErrors, obsInSample, horizon):
    if loss == "MSEh":
        return np.sum(adamErrors[:, horizon-1]**2) / (obsInSample - horizon)
    elif loss == "TMSE":
        return np.sum(np.sum(adamErrors**2, axis=0) / (obsInSample - horizon))
    elif loss == "GTMSE":
        return np.sum(np.log(np.sum(adamErrors**2, axis=0) / (obsInSample - horizon)))
    elif loss == "MSCE":
        return np.sum(np.sum(adamErrors, axis=1)**2) / (obsInSample - horizon)
    elif loss == "MAEh":
        return np.sum(np.abs(adamErrors[:, horizon-1])) / (obsInSample - horizon)
    elif loss == "TMAE":
        return np.sum(np.sum(np.abs(adamErrors), axis=0) / (obsInSample - horizon))
    elif loss == "GTMAE":
        return np.sum(np.log(np.sum(np.abs(adamErrors), axis=0) / (obsInSample - horizon)))
    elif loss == "MACE":
        return np.sum(np.abs(np.sum(adamErrors, axis=1))) / (obsInSample - horizon)
    elif loss == "HAMh":
        return np.sum(np.sqrt(np.abs(adamErrors[:, horizon-1]))) / (obsInSample - horizon)
    elif loss == "THAM":
        return np.sum(np.sum(np.sqrt(np.abs(adamErrors)), axis=0) / (obsInSample - horizon))
    elif loss == "GTHAM":
        return np.sum(np.log(np.sum(np.sqrt(np.abs(adamErrors)), axis=0) / (obsInSample - horizon)))
    elif loss == "CHAM":
        return np.sum(np.sqrt(np.abs(np.sum(adamErrors, axis=1)))) / (obsInSample - horizon)
    elif loss == "GPL":
        return np.log(np.linalg.det(adamErrors.T @ adamErrors / (obsInSample - horizon)))
    else:
        return 0
    

def scaler(distribution, Etype, errors, y_fitted, obs_in_sample, other):
    """
    Calculate scale parameter for the provided parameters.

    Parameters:
    - distribution (str): The distribution type
    - Etype (str): Error type ('A' for additive, 'M' for multiplicative)
    - errors (np.array): Array of errors
    - y_fitted (np.array): Array of fitted values
    - obs_in_sample (int): Number of observations in sample
    - other (float): Additional parameter for some distributions

    Returns:
    float: The calculated scale parameter
    """
    
    # Helper function to safely compute complex logarithm
    def safe_log(x):
        return np.log(np.abs(x) + 1j * (x.imag - x.real))
    
    if distribution == "dnorm":
        return np.sqrt(np.sum(errors**2) / obs_in_sample)
    
    elif distribution == "dlaplace":
        return np.sum(np.abs(errors)) / obs_in_sample
    
    elif distribution == "ds":
        return np.sum(np.sqrt(np.abs(errors))) / (obs_in_sample * 2)
    
    elif distribution == "dgnorm":
        return (other * np.sum(np.abs(errors)**other) / obs_in_sample)**(1 / other)
    
    elif distribution == "dalaplace":
        return np.sum(errors * (other - (errors <= 0) * 1)) / obs_in_sample
    
    elif distribution == "dlnorm":
        if Etype == "A":
            temp = 1 - np.sqrt(np.abs(1 - np.sum(np.log(np.abs(1 + errors / y_fitted))**2) / obs_in_sample))
        else:  # "M"
            temp = 1 - np.sqrt(np.abs(1 - np.sum(np.log(1 + errors)**2) / obs_in_sample))
        return np.sqrt(2 * np.abs(temp))
    
    elif distribution == "dllaplace":
        if Etype == "A":
            return np.real(np.sum(np.abs(safe_log(1 + errors / y_fitted))) / obs_in_sample)
        else:  # "M"
            return np.sum(np.abs(np.log(1 + errors))) / obs_in_sample
    
    elif distribution == "dls":
        if Etype == "A":
            return np.real(np.sum(np.sqrt(np.abs(safe_log(1 + errors / y_fitted)))) / obs_in_sample)
        else:  # "M"
            return np.sum(np.sqrt(np.abs(np.log(1 + errors)))) / obs_in_sample
    
    elif distribution == "dlgnorm":
        if Etype == "A":
            return np.real((other * np.sum(np.abs(safe_log(1 + errors / y_fitted))**other) / obs_in_sample)**(1 / other))
        else:  # "M"
            return (other * np.sum(np.abs(safe_log(1 + errors))**other) / obs_in_sample)**(1 / other)
    
    elif distribution == "dinvgauss":
        if Etype == "A":
            return np.sum((errors / y_fitted)**2 / (1 + errors / y_fitted)) / obs_in_sample
        else:  # "M"
            return np.sum(errors**2 / (1 + errors)) / obs_in_sample
    
    elif distribution == "dgamma":
        if Etype == "A":
            return np.sum((errors / y_fitted)**2) / obs_in_sample
        else:  # "M"
            return np.sum(errors**2) / obs_in_sample
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

def measurement_inverter(measurement):
    """
    Invert the measurement matrix, setting infinite values to zero.
    This is needed for the stability check for xreg models with regressors="adapt".

    Parameters:
    - measurement (np.array): The measurement matrix to invert

    Returns:
    np.array: The inverted measurement matrix
    """
    # Create a copy to avoid modifying the original array
    inverted = np.array(measurement, copy=True)

    # Invert all elements
    np.divide(1, inverted, out=inverted, where=inverted!=0)

    # Set infinite values to zero
    inverted[np.isinf(inverted)] = 0

    return inverted

def smooth_eigens(persistence, transition, measurement,
                  lags_model_all, xreg_model, obs_in_sample,
                  has_delta_persistence=False):
    lags_unique = np.unique(lags_model_all)
    lags_unique_length = len(lags_unique)
    eigen_values = np.zeros(len(lags_model_all), dtype=complex)

    # Eigen values checks do not work for xreg. So, check the average condition
    if xreg_model and has_delta_persistence:
        # We check the condition on average
        return np.linalg.eigvals(
            transition -
            np.diag(persistence.flatten()) @
            measurement_inverter(measurement[:obs_in_sample, :]).T @
            measurement[:obs_in_sample, :] / obs_in_sample
        )
    else:
        for i in range(lags_unique_length):
            mask = lags_model_all == lags_unique[i]
            eigen_values[mask] = np.linalg.eigvals(
                transition[np.ix_(mask, mask)] -
                persistence[mask].reshape(-1, 1) @
                measurement[obs_in_sample - 1, mask].reshape(1, -1)
            )
        return eigen_values
