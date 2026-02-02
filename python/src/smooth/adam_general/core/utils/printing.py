"""
Printing utilities for ADAM models.

This module provides functions to generate formatted summaries of fitted ADAM models,
similar to R's print.adam() method.
"""

import time
from typing import Any, Dict, Optional

import numpy as np

from smooth.adam_general.core.utils.ic import AIC, BIC, AICc, BICc


def _format_distribution(distribution: str, other: Optional[Dict] = None) -> str:
    """
    Format distribution name for display.

    Parameters
    ----------
    distribution : str
        Distribution code (e.g., 'dnorm', 'dgamma')
    other : Optional[Dict]
        Dictionary with additional parameters (shape, nu, etc.)

    Returns
    -------
    str
        Human-readable distribution name
    """
    other = other or {}

    dist_names = {
        "dnorm": "Normal",
        "dlaplace": "Laplace",
        "ds": "S",
        "dlogis": "Logistic",
        "dlnorm": "Log-Normal",
        "dllaplace": "Log-Laplace",
        "dls": "Log-S",
        "dinvgauss": "Inverse Gaussian",
        "dgamma": "Gamma",
    }

    if distribution == "dgnorm":
        shape = other.get("shape", "?")
        return (
            f"Generalised Normal with shape={shape:.4f}"
            if isinstance(shape, float)
            else f"Generalised Normal with shape={shape}"
        )
    elif distribution == "dt":
        nu = other.get("nu", "?")
        return (
            f"Student t with df={nu:.4f}"
            if isinstance(nu, float)
            else f"Student t with df={nu}"
        )
    elif distribution == "dalaplace":
        alpha = other.get("alpha", "?")
        return (
            f"Asymmetric Laplace with alpha={alpha:.4f}"
            if isinstance(alpha, float)
            else f"Asymmetric Laplace with alpha={alpha}"
        )
    elif distribution == "dlgnorm":
        shape = other.get("shape", "?")
        return (
            f"Log-Generalised Normal with shape={shape:.4f}"
            if isinstance(shape, float)
            else f"Log-Generalised Normal with shape={shape}"
        )

    return dist_names.get(distribution, distribution)


def _get_persistence_from_model(model: Any) -> Dict[str, Any]:
    """
    Extract persistence parameters from a fitted ADAM model.

    This function extracts persistence values from the filled matrices
    or calculates them from the B parameter vector if needed.

    Parameters
    ----------
    model : ADAM
        Fitted ADAM model

    Returns
    -------
    Dict[str, Any]
        Dictionary with persistence values: alpha, beta, gamma (list), xreg
    """
    result = {"alpha": None, "beta": None, "gamma": [], "xreg": None}

    # Try to get from prepared_model first (if predict was called)
    if hasattr(model, "prepared_model") and model.prepared_model:
        vec_g = model.prepared_model.get("persistence")
        if vec_g is not None and len(vec_g) > 0:
            return _extract_persistence_from_vec_g(model, vec_g)

    # Otherwise, fill matrices to get vec_g
    if (
        hasattr(model, "adam_estimated")
        and model.adam_estimated
        and hasattr(model, "adam_created")
        and model.adam_created
    ):
        B = model.adam_estimated.get("B")
        if B is not None and len(B) > 0:
            # Import filler to fill matrices with estimated parameters
            try:
                from smooth.adam_general.core.creator import filler

                # Make a deep copy of matrices to avoid modifying originals
                matrices_copy = {
                    "mat_vt": model.adam_created["mat_vt"].copy(),
                    "mat_wt": model.adam_created["mat_wt"].copy(),
                    "mat_f": model.adam_created["mat_f"].copy(),
                    "vec_g": model.adam_created["vec_g"].copy(),
                    "arima_polynomials": model.adam_created.get("arima_polynomials"),
                }

                filled = filler(
                    B=B,
                    model_type_dict=model.model_type_dict,
                    components_dict=model.components_dict,
                    lags_dict=model.lags_dict,
                    matrices_dict=matrices_copy,
                    persistence_checked=model.persistence_results,
                    initials_checked=model.initials_results,
                    arima_checked=model.arima_results,
                    explanatory_checked=model.explanatory_dict,
                    phi_dict=model.phi_dict,
                    constants_checked=model.constant_dict,
                )

                vec_g = filled.get("vec_g")
                if vec_g is not None:
                    return _extract_persistence_from_vec_g(model, vec_g)
            except Exception:
                pass

    return result


def _extract_persistence_from_vec_g(model: Any, vec_g: np.ndarray) -> Dict[str, Any]:
    """
    Extract persistence values from vec_g array.

    Parameters
    ----------
    model : ADAM
        Fitted ADAM model
    vec_g : np.ndarray
        Persistence vector

    Returns
    -------
    Dict[str, Any]
        Dictionary with persistence values
    """
    result = {"alpha": None, "beta": None, "gamma": [], "xreg": None}

    if vec_g is None or len(vec_g) == 0:
        return result

    vec_g = np.array(vec_g).flatten()
    idx = 0

    # Check model type
    model_type = model.model_type_dict if hasattr(model, "model_type_dict") else {}
    components = model.components_dict if hasattr(model, "components_dict") else {}

    n_ets = components.get("components_number_ets", 0)
    n_ets_seasonal = components.get("components_number_ets_seasonal", 0)

    # Alpha (level)
    if n_ets > 0 and idx < len(vec_g):
        result["alpha"] = vec_g[idx]
        idx += 1

    # Beta (trend)
    if model_type.get("model_is_trendy", False) and idx < len(vec_g):
        result["beta"] = vec_g[idx]
        idx += 1

    # Gamma (seasonal) - can be multiple
    if model_type.get("model_is_seasonal", False) and n_ets_seasonal > 0:
        for _ in range(n_ets_seasonal):
            if idx < len(vec_g):
                result["gamma"].append(vec_g[idx])
                idx += 1

    return result


def _format_persistence_vector(model: Any, digits: int = 4) -> str:
    """
    Format persistence vector for display.

    Parameters
    ----------
    model : ADAM
        Fitted ADAM model
    digits : int
        Number of decimal places

    Returns
    -------
    str
        Formatted persistence vector string
    """
    persistence = _get_persistence_from_model(model)

    names = []
    values = []

    # Level (alpha)
    if persistence["alpha"] is not None:
        names.append("alpha")
        values.append(persistence["alpha"])

    # Trend (beta)
    if persistence["beta"] is not None:
        names.append("beta")
        values.append(persistence["beta"])

    # Seasonal (gamma)
    if persistence["gamma"]:
        for i, val in enumerate(persistence["gamma"]):
            if len(persistence["gamma"]) > 1:
                names.append(f"gamma{i + 1}")
            else:
                names.append("gamma")
            values.append(val)

    if not names:
        return ""

    # Format header and values
    name_strs = [f"{n:>{max(len(n), digits + 2)}}" for n in names]
    value_strs = [
        f"{v:{max(len(names[i]), digits + 2)}.{digits}f}" for i, v in enumerate(values)
    ]

    header = " ".join(name_strs)
    vals = " ".join(value_strs)

    return f"{header}\n{vals}"


def _format_arma_parameters(model: Any, digits: int = 4) -> str:
    """
    Format ARMA parameters for display.

    Parameters
    ----------
    model : ADAM
        Fitted ADAM model
    digits : int
        Number of decimal places

    Returns
    -------
    str
        Formatted ARMA parameters string
    """
    if not hasattr(model, "arima_results") or not model.arima_results:
        return ""

    arima = model.arima_results
    if not arima.get("arima_model", False):
        return ""

    lines = []

    # AR parameters
    ar_params = arima.get("ar_parameters", [])
    if ar_params and len(ar_params) > 0:
        ar_orders = arima.get("ar_orders", [])
        lags = model.lags_dict.get("lags", [1])

        lines.append("AR parameters:")
        param_idx = 0
        for i, order in enumerate(ar_orders):
            if order > 0:
                lag = lags[i] if i < len(lags) else 1
                for j in range(order):
                    if param_idx < len(ar_params):
                        lines.append(
                            f"  AR({j + 1}) Lag {lag}: "
                            f"{ar_params[param_idx]:.{digits}f}"
                        )
                        param_idx += 1

    # MA parameters
    ma_params = arima.get("ma_parameters", [])
    if ma_params and len(ma_params) > 0:
        ma_orders = arima.get("ma_orders", [])
        lags = model.lags_dict.get("lags", [1])

        lines.append("MA parameters:")
        param_idx = 0
        for i, order in enumerate(ma_orders):
            if order > 0:
                lag = lags[i] if i < len(lags) else 1
                for j in range(order):
                    if param_idx < len(ma_params):
                        lines.append(
                            f"  MA({j + 1}) Lag {lag}: "
                            f"{ma_params[param_idx]:.{digits}f}"
                        )
                        param_idx += 1

    return "\n".join(lines)


def _format_information_criteria(
    log_lik_value: float, nobs: int, n_params: int, digits: int = 4
) -> str:
    """
    Format information criteria for display.

    Parameters
    ----------
    log_lik_value : float
        Log-likelihood value
    nobs : int
        Number of observations
    n_params : int
        Number of estimated parameters
    digits : int
        Number of decimal places

    Returns
    -------
    str
        Formatted information criteria string
    """
    aic = AIC(log_lik_value, nobs, n_params)
    aicc = AICc(log_lik_value, nobs, n_params)
    bic = BIC(log_lik_value, nobs, n_params)
    bicc = BICc(log_lik_value, nobs, n_params)

    # Format with alignment
    width = max(digits + 5, 8)
    header = f"{'AIC':>{width}} {'AICc':>{width}} {'BIC':>{width}} {'BICc':>{width}}"
    values = (
        f"{aic:{width}.{digits}f} "
        f"{aicc:{width}.{digits}f} "
        f"{bic:{width}.{digits}f} "
        f"{bicc:{width}.{digits}f}"
    )

    return f"{header}\n{values}"


def _compute_forecast_errors(
    y_holdout: np.ndarray,
    y_fitted_holdout: np.ndarray,
    y_in_sample: np.ndarray,
    period: int = 1,
) -> Dict[str, float]:
    """
    Compute forecast error metrics.

    Parameters
    ----------
    y_holdout : np.ndarray
        Actual holdout values
    y_fitted_holdout : np.ndarray
        Forecasted values for holdout period
    y_in_sample : np.ndarray
        In-sample actual values (for scaling)
    period : int
        Seasonal period for MASE calculation

    Returns
    -------
    Dict[str, float]
        Dictionary of error metrics
    """
    errors = y_holdout - y_fitted_holdout

    # Basic errors
    me = np.mean(errors)
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)

    # Scaled errors
    y_mean = np.mean(y_in_sample)
    sce = np.sum(errors) / np.sum(y_in_sample) if np.sum(y_in_sample) != 0 else np.nan
    smae = mae / y_mean if y_mean != 0 else np.nan
    smse = mse / (y_mean**2) if y_mean != 0 else np.nan

    # Asymmetry
    pos_errors = np.sum(errors[errors > 0])
    neg_errors = np.abs(np.sum(errors[errors < 0]))
    asymmetry = (
        (pos_errors - neg_errors) / (pos_errors + neg_errors)
        if (pos_errors + neg_errors) != 0
        else 0
    )

    # MASE and RMSSE (using naive seasonal forecast as benchmark)
    if len(y_in_sample) > period:
        naive_errors = y_in_sample[period:] - y_in_sample[:-period]
        scale = np.mean(np.abs(naive_errors))
        mase = mae / scale if scale != 0 else np.nan
        rmsse = (
            rmse / np.sqrt(np.mean(naive_errors**2))
            if np.mean(naive_errors**2) != 0
            else np.nan
        )
    else:
        mase = np.nan
        rmsse = np.nan

    # Relative errors (vs naive)
    naive_forecast = y_in_sample[-1] if len(y_in_sample) > 0 else 0
    naive_mae = np.mean(np.abs(y_holdout - naive_forecast))
    naive_rmse = np.sqrt(np.mean((y_holdout - naive_forecast) ** 2))
    rmae = mae / naive_mae if naive_mae != 0 else np.nan
    rrmse = rmse / naive_rmse if naive_rmse != 0 else np.nan

    return {
        "ME": me,
        "MAE": mae,
        "RMSE": rmse,
        "sCE": sce,
        "asymmetry": asymmetry,
        "sMAE": smae,
        "sMSE": smse,
        "MASE": mase,
        "RMSSE": rmsse,
        "rMAE": rmae,
        "rRMSE": rrmse,
    }


def _format_forecast_errors(errors: Dict[str, float], digits: int = 3) -> str:
    """
    Format forecast errors for display.

    Parameters
    ----------
    errors : Dict[str, float]
        Dictionary of error metrics
    digits : int
        Number of decimal places

    Returns
    -------
    str
        Formatted error metrics string
    """
    lines = []

    # Line 1: ME, MAE, RMSE
    line1 = "; ".join(
        [
            f"ME: {errors['ME']:.{digits}f}",
            f"MAE: {errors['MAE']:.{digits}f}",
            f"RMSE: {errors['RMSE']:.{digits}f}",
        ]
    )
    lines.append(line1)

    # Line 2: sCE, Asymmetry, sMAE, sMSE (as percentages)
    line2 = "; ".join(
        [
            f"sCE: {errors['sCE'] * 100:.{digits}f}%",
            f"Asymmetry: {errors['asymmetry'] * 100:.1f}%",
            f"sMAE: {errors['sMAE'] * 100:.{digits}f}%",
            f"sMSE: {errors['sMSE'] * 100:.{digits}f}%",
        ]
    )
    lines.append(line2)

    # Line 3: MASE, RMSSE, rMAE, rRMSE
    line3 = "; ".join(
        [
            f"MASE: {errors['MASE']:.{digits}f}",
            f"RMSSE: {errors['RMSSE']:.{digits}f}",
            f"rMAE: {errors['rMAE']:.{digits}f}",
            f"rRMSE: {errors['rRMSE']:.{digits}f}",
        ]
    )
    lines.append(line3)

    return "\n".join(lines)


def model_summary(model: Any, digits: int = 4) -> str:
    """
    Generate a formatted summary of a fitted ADAM model.

    This function produces output similar to R's print.adam() method.

    Parameters
    ----------
    model : ADAM
        Fitted ADAM model instance
    digits : int, default=4
        Number of decimal places for numeric output

    Returns
    -------
    str
        Formatted model summary string

    Examples
    --------
    >>> from smooth import ADAM
    >>> model = ADAM(model="AAN")
    >>> model.fit(y)
    >>> print(model_summary(model))
    """
    lines = []

    # Time elapsed
    if hasattr(model, "start_time"):
        elapsed = time.time() - model.start_time
        lines.append(f"Time elapsed: {elapsed:.2f} seconds")

    # Model type
    model_name = _get_model_name(model)
    function_name = _get_function_name(model)
    lines.append(f"Model estimated using {function_name}() function: {model_name}")

    # Initialization type
    init_type = _get_initialization_type(model)
    lines.append(f"With {init_type} initialisation")

    # Distribution
    distribution = _get_distribution(model)
    dist_str = _format_distribution(distribution, getattr(model, "other", None))
    lines.append(f"Distribution assumed in the model: {dist_str}")

    # Loss function
    loss_str = _format_loss(model, digits)
    lines.append(loss_str)

    # Constant/Drift if present
    constant_str = _format_constant(model, digits)
    if constant_str:
        lines.append(constant_str)

    # Persistence vector
    if _is_ets_model(model):
        persistence_str = _format_persistence_vector(model, digits)
        if persistence_str:
            lines.append("Persistence vector g:")
            lines.append(persistence_str)

        # Damping parameter
        phi_str = _format_phi(model, digits)
        if phi_str:
            lines.append(phi_str)

    # ARMA parameters
    arma_str = _format_arma_parameters(model, digits)
    if arma_str:
        lines.append("ARMA parameters of the model:")
        lines.append(arma_str)

    # Sample size and parameters
    nobs = _get_nobs(model)
    n_params = _get_n_params(model)
    lines.append(f"Sample size: {nobs}")
    lines.append(f"Number of estimated parameters: {n_params}")
    lines.append(f"Number of degrees of freedom: {nobs - n_params}")

    # Information criteria
    if _can_compute_ic(model):
        log_lik = _get_log_likelihood(model)
        if log_lik is not None:
            lines.append("Information criteria:")
            lines.append(_format_information_criteria(log_lik, nobs, n_params, digits))
    else:
        lines.append(
            "Information criteria are unavailable for the chosen loss & distribution."
        )

    # Forecast errors (if holdout)
    errors_str = _format_holdout_errors(model, digits)
    if errors_str:
        lines.append("Forecast errors:")
        lines.append(errors_str)

    return "\n".join(lines)


def _get_model_name(model: Any) -> str:
    """Get the model name string (e.g., 'ETS(AAN)')."""
    # Use the model attribute directly - it's updated in _set_fitted_attributes()
    if hasattr(model, "model") and model.model:
        return model.model
    return "Unknown"


def _get_function_name(model: Any) -> str:
    """Get the function name used to estimate the model."""
    # Check if it's an ES model (subclass)
    if model.__class__.__name__ == "ES":
        return "ES"
    return "ADAM"


def _get_initialization_type(model: Any) -> str:
    """Get the initialization type string."""
    if hasattr(model, "initials_results") and model.initials_results:
        return model.initials_results.get("initial_type", "unknown")
    if hasattr(model, "initial"):
        if isinstance(model.initial, str):
            return model.initial
        return "provided"
    return "unknown"


def _get_distribution(model: Any) -> str:
    """Get the distribution code."""
    if hasattr(model, "general") and model.general:
        return model.general.get(
            "distribution_new", model.general.get("distribution", "dnorm")
        )
    if hasattr(model, "distribution") and model.distribution:
        return model.distribution
    return "dnorm"


def _format_loss(model: Any, digits: int) -> str:
    """Format loss function information."""
    loss = "likelihood"
    loss_value = None
    lambda_val = None

    if hasattr(model, "general") and model.general:
        loss = model.general.get("loss", "likelihood")
        # Get lambda for LASSO/RIDGE
        if loss in ["LASSO", "RIDGE"]:
            lambda_val = model.general.get("lambda")

    if hasattr(model, "adam_estimated") and model.adam_estimated:
        if "CF_value" in model.adam_estimated:
            loss_value = model.adam_estimated["CF_value"]

    result = f"Loss function type: {loss}"
    if loss_value is not None:
        result += f"; Loss function value: {loss_value:.{digits}f}"
    if lambda_val is not None:
        result += f"; lambda= {lambda_val}"

    return result


def _format_constant(model: Any, digits: int) -> str:
    """Format constant/drift value if present."""
    if hasattr(model, "constant_dict") and model.constant_dict:
        if model.constant_dict.get("constant_estimate", False):
            constant_val = model.constant_dict.get("constant_value")
            if constant_val is not None:
                return f"Intercept/Drift value: {constant_val:.{digits}f}"
    return ""


def _format_phi(model: Any, digits: int) -> str:
    """Format damping parameter if present."""
    # First check if model has damped trend - don't print if not damped
    if not hasattr(model, "model_type_dict") or not model.model_type_dict:
        return ""

    if not model.model_type_dict.get("damped", False):
        return ""

    # Model is damped, now get the phi value
    phi_val = None

    # Try to get from phi_dict first
    if hasattr(model, "phi_dict") and model.phi_dict:
        phi_val = model.phi_dict.get("phi")

    # Fallback to phi_ attribute
    if phi_val is None and hasattr(model, "phi_") and model.phi_ is not None:
        phi_val = model.phi_

    # Try prepared_model
    if phi_val is None and hasattr(model, "prepared_model") and model.prepared_model:
        phi_val = model.prepared_model.get("phi")

    if phi_val is not None:
        return f"Damping parameter: {phi_val:.{digits}f}"

    return ""


def _is_ets_model(model: Any) -> bool:
    """Check if model has ETS components."""
    if hasattr(model, "model_type_dict") and model.model_type_dict:
        return model.model_type_dict.get("ets_model", False)
    return False


def _get_nobs(model: Any) -> int:
    """Get number of observations."""
    if hasattr(model, "observations_dict") and model.observations_dict:
        return model.observations_dict.get("obs_in_sample", 0)
    return 0


def _get_n_params(model: Any) -> int:
    """Get number of estimated parameters (for degrees of freedom calculation)."""
    # Use n_param table if available (preferred)
    if hasattr(model, "n_param") and model.n_param:
        return model.n_param.n_param_estimated

    # Fallback to n_param_estimated attribute
    if hasattr(model, "n_param_estimated"):
        return model.n_param_estimated

    # Fallback to general dict n_param
    if hasattr(model, "general") and model.general:
        n_param = model.general.get("n_param")
        if n_param:
            return n_param.n_param_estimated

        # Legacy format fallback
        params_number = model.general.get("parameters_number", [[0]])
        if params_number and len(params_number) > 0:
            return (
                params_number[0][0]
                if isinstance(params_number[0], list)
                else params_number[0]
            )

    return 0


def _format_n_param_table(model: Any) -> str:
    """Format the n_param table for display."""
    n_param = None

    # Try to get n_param from model
    if hasattr(model, "n_param") and model.n_param:
        n_param = model.n_param
    elif hasattr(model, "general") and model.general:
        n_param = model.general.get("n_param")

    if n_param is None:
        return ""

    return str(n_param)


def _can_compute_ic(model: Any) -> bool:
    """Check if information criteria can be computed."""
    loss = "likelihood"
    distribution = "dnorm"

    if hasattr(model, "general") and model.general:
        loss = model.general.get("loss", "likelihood")
        distribution = model.general.get(
            "distribution_new", model.general.get("distribution", "dnorm")
        )

    # IC available for likelihood or matching loss/distribution combos
    if loss == "likelihood":
        return True
    if loss in ["MSE", "MSEh", "MSCE", "GPL"] and distribution == "dnorm":
        return True
    if loss in ["MAE", "MAEh", "MACE"] and distribution == "dlaplace":
        return True
    if loss in ["HAM", "HAMh", "CHAM"] and distribution == "ds":
        return True

    return False


def _get_log_likelihood(model: Any) -> Optional[float]:
    """Get log-likelihood value."""
    if hasattr(model, "adam_estimated") and model.adam_estimated:
        log_lik_dict = model.adam_estimated.get("log_lik_adam_value", {})
        if isinstance(log_lik_dict, dict):
            return log_lik_dict.get("value")
        return log_lik_dict
    return None


def _format_holdout_errors(model: Any, digits: int) -> str:
    """Format holdout forecast errors if available."""
    if not hasattr(model, "general") or not model.general:
        return ""

    if not model.general.get("holdout", False):
        return ""

    # Get holdout data and forecasts
    if not hasattr(model, "observations_dict") or not model.observations_dict:
        return ""

    y_holdout = model.observations_dict.get("y_holdout")
    y_in_sample = model.observations_dict.get("y_in_sample")

    if y_holdout is None or len(y_holdout) == 0:
        return ""

    # Get forecasts for holdout period
    if hasattr(model, "forecast_results") and model.forecast_results:
        y_forecast = model.forecast_results.get("forecast")
        if y_forecast is not None and len(y_forecast) >= len(y_holdout):
            y_forecast = y_forecast[: len(y_holdout)]
        else:
            return ""
    else:
        return ""

    # Get period for MASE calculation
    period = 1
    if hasattr(model, "lags_dict") and model.lags_dict:
        lags = model.lags_dict.get("lags", [1])
        period = max(lags) if lags else 1

    # Compute errors
    errors = _compute_forecast_errors(y_holdout, y_forecast, y_in_sample, period)
    return _format_forecast_errors(errors, digits)
