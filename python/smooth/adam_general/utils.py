import numpy as np



def ma(y, order):
    if order % 2 == 0:
        weights = np.concatenate([[0.5], np.ones(order - 1), [0.5]]) / order
    else:
        weights = np.ones(order) / order
    
    result = np.convolve(y, weights, mode='valid')
    
    # Pad the result with NaNs at the edges to match the original length
    pad_width = (len(y) - len(result)) // 2
    return np.pad(result, (pad_width, pad_width), mode='constant', constant_values=np.nan)


def msdecompose(y, lags=[12], type="additive"):
    """
    Multiple Seasonal Classical Decomposition

    This function decomposes multiple seasonal time series into components using
    the principles of classical decomposition.

    The function applies centered moving averages to smooth the original series
    and obtain level, trend, and seasonal components of the series. It supports
    both additive and multiplicative decomposition methods.

    Parameters
    ----------
    y : array-like
        Vector or array containing the time series data to be decomposed.
    lags : list of int, optional (default=[12])
        List of lags corresponding to the frequencies in the data.
        For example, [7, 365] for weekly and yearly seasonality in daily data.
    type : {'additive', 'multiplicative'}, optional (default='additive')
        The type of decomposition. If 'multiplicative' is selected,
        then the logarithm of data is taken prior to the decomposition.

    Returns
    -------
    dict
        A dictionary containing the following components:
        - y : array-like
            The original time series.
        - initial : array-like
            The estimates of the initial level and trend.
        - trend : array-like
            The long-term trend in the data.
        - seasonal : list of array-like
            List of seasonal patterns for each specified lag.
        - loss : str
            The loss function used (always 'MSE' for this implementation).
        - lags : list of int
            The provided lags used for decomposition.
        - type : str
            The selected type of decomposition ('additive' or 'multiplicative').
        - yName : str
            The name of the provided data (always 'y' in this implementation).

    Notes
    -----
    - The function handles missing values by imputing them using a polynomial
      and trigonometric regression.
    - For multiplicative decomposition, non-positive values are treated as missing.
    - The seasonal components are centered around zero for additive decomposition
      and around one for multiplicative decomposition.

    Examples
    --------
    >>> import numpy as np
    >>> from msedecompose import msedecompose
    >>> 
    >>> # Generate sample data with multiple seasonalities
    >>> t = np.arange(1000)
    >>> y = 10 + 0.01*t + 5*np.sin(2*np.pi*t/7) + 3*np.sin(2*np.pi*t/365) + np.random.normal(0, 1, 1000)
    >>> 
    >>> # Perform decomposition
    >>> result = msedecompose(y, lags=[7, 365], type="additive")
    >>> 
    >>> # Access components
    >>> trend = result['trend']
    >>> weekly_seasonal = result['seasonal'][0]
    >>> yearly_seasonal = result['seasonal'][1]

 
    """
    # ensure type is valid and lags a list
    if type not in ["additive", "multiplicative"]:
        raise ValueError("type must be 'additive' or 'multiplicative'")
    
    # ensure lags is a list
    if not isinstance(lags, list):
        raise ValueError("lags must be a list")
    
    y = np.asarray(y)
    obs_in_sample = len(y)
    y_na_values = np.isnan(y)

    # transform the data if needed and split the sample
    if type == "multiplicative":
        shifted_data = False
        if any(y[~y_na_values] <= 0):
            y_na_values[:] = y_na_values | (y <= 0)
            
        y_insample = np.log(y)
    
    else:
        y_insample = y

    # treat the missing values
    if any(y_na_values):
        # create the design matrix
        X = np.c_[np.ones(obs_in_sample), np.poly(np.arange(1, obs_in_sample + 1), degree=min(max(int(obs_in_sample/10),1),5)).T, np.sin(np.pi * np.outer(np.arange(1, obs_in_sample + 1), np.arange(1, max(lags) + 1)) / max(lags))]
        # We use the least squares method to fit the model
        lm_fit = np.linalg.lstsq(X[~y_na_values], y_insample[~y_na_values], rcond=None)
        # replace the missing values with the fitted values
        y_insample[y_na_values] = np.dot(X, lm_fit[0])[y_na_values]
        del X
    
    obs = len(y)
    lags = sorted(set(lags))
    lags_length = len(lags)
    
    # List of smoothed values
    y_smooth = [None] * (lags_length + 1)
    y_smooth[0] = y_insample  # Put actuals in the first element of the list
    
    # List of cleared values
    y_clear = [None] * lags_length
    
    # Smooth time series with different lags
    for i in range(lags_length):
        y_smooth[i + 1] = ma(y_insample, lags[i])
    
    trend = y_smooth[lags_length]

    # Produce the cleared series
    for i in range(lags_length):
        y_clear[i] = y_smooth[i] - y_smooth[i + 1]

    # The seasonal patterns
    patterns = [None] * lags_length
    for i in range(lags_length):
        patterns[i] = np.array([np.nanmean(y_clear[i][j::lags[i]]) for j in range(lags[i])])
        patterns[i] -= np.nanmean(patterns[i])

    # Initial level and trend
    valid_trend = trend[~np.isnan(trend)]
    initial = np.array([
        valid_trend[0],
        np.nanmean(np.diff(valid_trend))
    ])
    
    # Fix the initial, to get to the beginning of the sample
    initial[0] -= initial[1] * np.floor(max(lags) / 2)

    # Return to the original scale
    if type == "multiplicative":
        initial = np.exp(initial)
        trend = np.exp(trend)
        patterns = [np.exp(pattern) for pattern in patterns]
        if shifted_data:
            initial[0] -= 1
            trend -= 1

    # Prepare the return structure
    result = {
        "y": y,
        "initial": initial,
        "trend": trend,
        "seasonal": patterns,
        "loss": "MSE",
        "lags": lags,
        "type": type,
        "yName": "y"  # You might want to pass the actual name as an argument
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
