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
        raise ValueError(
            f"Invalid information criterion: {ic_name}. Must be one of {list(ic_functions.keys())}"
        )

    return ic_functions[ic_name]
