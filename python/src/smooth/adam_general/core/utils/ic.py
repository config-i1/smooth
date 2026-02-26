import numpy as np


def AIC(loglik, nobs=None, df=None):  # noqa: N802
    """
    Calculate Akaike Information Criterion

    Parameters
    ----------
    loglik : float or object with loglik attribute
        Log-likelihood value
    nobs : int, optional
        Number of observations
    df : int, optional
        Degrees of freedom (number of parameters)

    Returns
    -------
    float
        AIC value
    """
    # Extract loglik value if object is passed
    if hasattr(loglik, "loglik"):
        loglik = loglik.loglik
    if hasattr(loglik, "df"):
        df = loglik.df

    return -2 * loglik + 2 * df


def AICc(loglik, nobs=None, df=None):  # noqa: N802
    """
    Calculate corrected Akaike Information Criterion

    Parameters
    ----------
    loglik : float or object with loglik attribute
        Log-likelihood value
    nobs : int, optional
        Number of observations
    df : int, optional
        Degrees of freedom (number of parameters)

    Returns
    -------
    float
        AICc value
    """
    # Extract loglik value if object is passed
    if hasattr(loglik, "loglik"):
        loglik = loglik.loglik
    if hasattr(loglik, "nobs"):
        nobs = loglik.nobs
    if hasattr(loglik, "df"):
        df = loglik.df

    aic = AIC(loglik, nobs, df)
    denominator = nobs - df - 1

    if denominator == 0:
        return float("inf")
    else:
        return aic + (2 * df * (df + 1)) / denominator


def BIC(loglik, nobs=None, df=None):  # noqa: N802
    """
    Calculate Bayesian Information Criterion

    Parameters
    ----------
    loglik : float or object with loglik attribute
        Log-likelihood value
    nobs : int, optional
        Number of observations
    df : int, optional
        Degrees of freedom (number of parameters)

    Returns
    -------
    float
        BIC value
    """
    # Extract loglik value if object is passed
    if hasattr(loglik, "loglik"):
        loglik = loglik.loglik
    if hasattr(loglik, "nobs"):
        nobs = loglik.nobs
    if hasattr(loglik, "df"):
        df = loglik.df

    return -2 * loglik + np.log(nobs) * df


def BICc(loglik, nobs=None, df=None):  # noqa: N802
    """
    Calculate corrected Bayesian Information Criterion

    Parameters
    ----------
    loglik : float or object with loglik attribute
        Log-likelihood value
    nobs : int, optional
        Number of observations
    df : int, optional
        Degrees of freedom (number of parameters)

    Returns
    -------
    float
        BICc value
    """
    # Extract loglik value if object is passed
    if hasattr(loglik, "loglik"):
        loglik = loglik.loglik
    if hasattr(loglik, "nobs"):
        nobs = loglik.nobs
    if hasattr(loglik, "df"):
        df = loglik.df

    bic = BIC(loglik, nobs, df)
    denominator = nobs - df - 1

    if denominator == 0:
        return float("inf")
    else:
        return bic + (np.log(nobs) * df * (df + 1)) / denominator


def ic_function(ic_name, loglik):
    """
    Select information criterion function based on name

    Parameters
    ----------
    ic_name : str
        Name of information criterion ('AIC', 'AICc', 'BIC', or 'BICc')

    Returns
    -------
    function
        Selected information criterion function
    """
    value = loglik["value"]
    nobs = loglik["nobs"]
    df = loglik["df"]
    ic_functions = {
        "AIC": AIC(value, nobs, df),
        "AICc": AICc(value, nobs, df),
        "BIC": BIC(value, nobs, df),
        "BICc": BICc(value, nobs, df),
    }

    if ic_name not in ic_functions:
        valid_names = list(ic_functions.keys())
        raise ValueError(
            f"Invalid information criterion: {ic_name}. Must be one of {valid_names}"
        )

    return ic_functions[ic_name]


def calculate_ic_weights(ic_values, threshold=1e-5):
    """
    Calculate Akaike weights from information criterion values.

    Akaike weights represent the relative likelihood of each model being the best
    model given the data. They are derived from information criterion values using
    the formula: w_i = exp(-0.5 * delta_i) / sum(exp(-0.5 * delta_j))
    where delta_i = IC_i - IC_min.

    Parameters
    ----------
    ic_values : dict
        Dictionary mapping model names to their IC values.
    threshold : float, default=1e-5
        Weights below this threshold are zeroed out and remaining weights
        are renormalized to sum to 1.0.

    Returns
    -------
    dict
        Dictionary mapping model names to normalized weights (summing to 1.0).

    Examples
    --------
    >>> ic_values = {"ANN": 100.5, "AAN": 98.2, "AAA": 99.1}
    >>> weights = calculate_ic_weights(ic_values)
    >>> sum(weights.values())  # Should be approximately 1.0
    1.0
    """
    if not ic_values:
        return {}

    model_names = list(ic_values.keys())
    ic_array = np.array(list(ic_values.values()))

    # Calculate delta IC (difference from minimum)
    ic_best = np.min(ic_array)
    delta_ic = ic_array - ic_best

    # Calculate raw weights
    weights = np.exp(-0.5 * delta_ic)

    # Normalize
    weights = weights / np.sum(weights)

    # Zero out tiny weights and renormalize
    weights[weights < threshold] = 0
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)

    return dict(zip(model_names, weights))
