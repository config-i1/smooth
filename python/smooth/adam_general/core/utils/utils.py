import numpy as np
from scipy import stats
from scipy.linalg import eigvals
from scipy.special import gamma, digamma, beta


def msdecompose(y, lags=[12], type="additive", smoother="ma"):
    """
    Decomposes a time series assuming multiple frequencies provided in lags.
    Uses only 'ma' smoother and avoids statsmodels package.
    
    Parameters:
    - y: numpy array, the time series data
    - lags: list or array, seasonal periods
    - type: str, 'additive' or 'multiplicative'
    - smoother: str, set to 'ma' only in this implementation
    
    Returns:
    - dict: decomposition results including states, fitted values, etc.
    """
    # Argument validation
    if type not in ["additive", "multiplicative"]:
        raise ValueError("type must be 'additive' or 'multiplicative'")
    if smoother != "ma":
        raise ValueError("Only 'ma' smoother is supported in this implementation")

    # Variable name handling
    y_name = "y"

    # Data preparation
    y = np.asarray(y)
    seasonal_lags = any(lag > 1 for lag in lags)

    # Smoothing function definition
    def smoothing_function(y, order):
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

    # Initial data processing
    obs_in_sample = len(y)
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
                    y_seasonal_smooth = np.mean(y_seasonal_non_na)
                    pattern_i[indices] = y_seasonal_smooth
            if np.any(~np.isnan(pattern_i)):
                pattern_i -= np.nanmean(pattern_i)
            patterns.append(pattern_i)
    else:
        patterns = None

    # Initial level and trend
    data_for_initial = y_smooth[lags_length - 1]  # Matches R's ySmooth[[lagsLength]]
    valid_data_for_initial = data_for_initial[~np.isnan(data_for_initial)]
    if len(valid_data_for_initial) == 0:
        init_level = 0.0
        init_trend = 0.0
    else:
        init_level = valid_data_for_initial[0]
        diffs = np.diff(valid_data_for_initial)
        init_trend = np.nanmean(diffs) if len(diffs) > 0 else 0.0
    initial = np.array([init_level, init_trend], dtype=float)
    #print(lags)
    initial[0] -= initial[1] * np.floor(max(lags) / 2)
    # Multiplicative adjustment
    if type == "multiplicative":
        initial = np.exp(initial)
        trend = np.exp(trend)
        if seasonal_lags:
            patterns = [np.exp(pattern) for pattern in patterns]

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

    # Return structure
    result = {
        "y": y,
        "states": states,
        "initial": initial,
        "seasonal": patterns,
        "fitted": y_fitted,
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
