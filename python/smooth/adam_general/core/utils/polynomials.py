"""
ARIMA polynomial utilities.

This module provides functions for creating ARIMA polynomials from parameters.
"""

import numpy as np

try:
    from smooth import _adamCore
except ImportError:
    _adamCore = None
    import warnings
    warnings.warn(
        "_adamCore C++ module not found. Please rebuild: cd python && pip install -e .",
        ImportWarning
    )


def adam_polynomialiser(
    parameters,
    ar_orders,
    i_orders,
    ma_orders,
    ar_estimate,
    ma_estimate,
    arma_parameters,
    lags
):
    """
    Create ARIMA polynomials from parameters.

    This function wraps the C++ adamCore.polynomialise() method to generate
    AR, I, MA, and ARI polynomials for ARIMA models.

    Parameters
    ----------
    parameters : float or np.ndarray
        Parameter vector B containing AR and MA coefficients to estimate.
        If a scalar (0 or similar), it's converted to an empty vector.
    ar_orders : np.ndarray
        AR orders for each lag (e.g., [1, 1] for ARIMA(1,0,0)(1,0,0)_12)
    i_orders : np.ndarray
        Integration orders for each lag
    ma_orders : np.ndarray
        MA orders for each lag
    ar_estimate : bool
        Whether AR parameters should be estimated (True) or are provided (False)
    ma_estimate : bool
        Whether MA parameters should be estimated (True) or are provided (False)
    arma_parameters : np.ndarray or None
        Fixed ARMA parameters (if ar_estimate=False or ma_estimate=False)
    lags : np.ndarray
        Lags for ARIMA components (e.g., [1, 12] for non-seasonal and seasonal)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'ar_polynomial': AR polynomial coefficients
        - 'i_polynomial': I polynomial coefficients
        - 'ari_polynomial': ARI (combined AR and I) polynomial coefficients
        - 'ma_polynomial': MA polynomial coefficients

    Notes
    -----
    The function requires instantiating an adamCore object, which needs
    basic model structure information. For polynomial generation, we use
    minimal settings since the polynomialise method only depends on
    ARIMA parameters.

    Examples
    --------
    >>> ar_orders = np.array([1, 0], dtype=np.uint32)
    >>> i_orders = np.array([1, 0], dtype=np.uint32)
    >>> ma_orders = np.array([1, 0], dtype=np.uint32)
    >>> lags = np.array([1, 12], dtype=np.uint32)
    >>> params = np.array([0.5, -0.3])  # AR and MA coefficients
    >>> result = adam_polynomialiser(params, ar_orders, i_orders, ma_orders,
    ...                              True, True, None, lags)
    >>> result['ari_polynomial']  # Combined AR and I polynomial
    """
    if _adamCore is None:
        raise ImportError("_adamCore module not compiled. Run: cd python && pip install -e .")

    # Convert inputs to proper types
    if np.isscalar(parameters):
        parameters = np.array([], dtype=np.float64)
    else:
        parameters = np.asarray(parameters, dtype=np.float64).ravel()

    # Use uint64 for arma::uvec consistency
    ar_orders = np.asarray(ar_orders, dtype=np.uint64).ravel()
    i_orders = np.asarray(i_orders, dtype=np.uint64).ravel()
    ma_orders = np.asarray(ma_orders, dtype=np.uint64).ravel()
    lags_arima = np.asarray(lags, dtype=np.uint64).ravel()

    if arma_parameters is None:
        arma_parameters = np.array([], dtype=np.float64)
    else:
        arma_parameters = np.asarray(arma_parameters, dtype=np.float64).ravel()

    # Create minimal adamCore instance
    # For polynomialiser, we need valid constructor params but they don't affect the polynomial
    # Use minimal valid settings
    n_arima = np.sum(ar_orders + ma_orders)

    # Create a minimal lags vector for constructor (just use the ARIMA lags)
    minimal_lags = lags_arima if len(lags_arima) > 0 else np.array([1], dtype=np.uint64)

    adam_core = _adamCore.adamCore(
        lags=minimal_lags,
        E='A',  # Additive error (doesn't matter for polynomials)
        T='N',  # No trend
        S='N',  # No seasonality
        nNonSeasonal=0,
        nSeasonal=0,
        nETS=0,
        nArima=int(n_arima),
        nXreg=0,
        constant=False,
        adamETS=False
    )

    # Call polynomialise method
    # C++ signature: polynomialise(B, arOrders, iOrders, maOrders, arEstimate,
    #                              maEstimate, armaParameters, lagsARIMA)
    result = adam_core.polynomialise(
        parameters,
        ar_orders,
        i_orders,
        ma_orders,
        bool(ar_estimate),
        bool(ma_estimate),
        arma_parameters,
        lags_arima
    )

    # Convert PolyResult to dict for backward compatibility
    return {
        'ar_polynomial': np.array(result.arPolynomial),
        'i_polynomial': np.array(result.iPolynomial),
        'ari_polynomial': np.array(result.ariPolynomial),
        'ma_polynomial': np.array(result.maPolynomial)
    }


__all__ = ['adam_polynomialiser']
