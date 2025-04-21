"""
ADAM Forecasting Package - Parameter Checker Module

This module handles the validation and processing of input parameters for ADAM models.
It provides functions to check and transform user inputs into the standardized format
required by the model estimation and forecasting functions.
"""

import numpy as np


def _warn(msg, silent=False):
    """
    Helper to show warnings in a style closer to R.

    Parameters
    ----------
    msg : str
        Warning message
    silent : bool, optional
        Whether to suppress warnings
    """
    if not silent:
        print(f"Warning: {msg}")


def _check_occurrence(
    data, occurrence, frequency=None, silent=False, holdout=False, h=0
):
    """
    Check and handle 'occurrence' parameter for intermittent demand data.

    Parameters
    ----------
    data : array-like
        Input time series data
    occurrence : str
        Occurrence type ('none', 'auto', 'fixed', etc.)
    frequency : str, optional
        Time series frequency
    silent : bool, optional
        Whether to suppress warnings
    holdout : bool, optional
        Whether to use holdout
    h : int, optional
        Forecast horizon

    Returns
    -------
    dict
        Dictionary with occurrence details and nonzero counts
    """
    data_list = list(data) if not isinstance(data, list) else data
    obs_in_sample = len(data_list)
    obs_all = obs_in_sample + (1 - holdout) * h

    # Identify non-zero observations
    nonzero_indices = [
        i for i, val in enumerate(data_list) if val is not None and val != 0
    ]
    obs_nonzero = len(nonzero_indices)

    # If all zeroes, fallback
    if all(val == 0 for val in data_list):
        _warn("You have a sample with zeroes only. Your forecast will be zero.", silent)
        return {
            "occurrence": "none",
            "occurrence_model": False,
            "obs_in_sample": obs_in_sample,
            "obs_nonzero": 0,
            "obs_all": obs_all,
        }

    # Validate the occurrence choice
    valid_occ = [
        "none",
        "auto",
        "fixed",
        "general",
        "odds-ratio",
        "inverse-odds-ratio",
        "direct",
        "provided",
    ]
    if occurrence not in valid_occ:
        _warn(f"Invalid occurrence: {occurrence}. Switching to 'none'.", silent)
        occurrence = "none"

    occurrence_model = occurrence not in ["none", "provided"]
    return {
        "occurrence": occurrence,
        "occurrence_model": occurrence_model,
        "obs_in_sample": obs_in_sample,
        "obs_nonzero": obs_nonzero,
        "obs_all": obs_all,
    }


def _check_lags(lags, obs_in_sample, silent=False):
    """
    Validate or adjust the set of lags.

    Parameters
    ----------
    lags : list
        List of lag values
    obs_in_sample : int
        Number of in-sample observations
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with lags information including seasonal lags
    """
    # Remove any zero-lags
    lags = [lg for lg in lags if lg != 0]

    # Force 1 in lags (for level)
    if 1 not in lags:
        lags.insert(0, 1)

    # Must be positive
    if any(lg <= 0 for lg in lags):
        raise ValueError(
            "Right! Why don't you try complex lags then, mister smart guy? (Lag <= 0 given)"
        )

    # Create lagsModel (matrix in R, list here)
    lags_model = sorted(set(lags))

    # Get seasonal lags (all lags > 1)
    lags_model_seasonal = [lag for lag in lags_model if lag > 1]
    max_lag = max(lags) if lags else 1

    if max_lag >= obs_in_sample:
        msg = (
            f"The maximum lags value is {max_lag}, while sample size is {obs_in_sample}. "
            f"I cannot guarantee that I'll be able to fit the model."
        )
        _warn(msg, silent)

    return {
        "lags": sorted(set(lags)),
        "lags_model": lags_model,
        "lags_model_seasonal": lags_model_seasonal,
        "lags_length": len(lags_model),
        "max_lag": max_lag,
    }


def _expand_component_code(comp_char, allow_multiplicative=True):
    """
    Expand a single component character into a list of valid possibilities.

    Parameters
    ----------
    comp_char : str
        Component character (E, T, S)
    allow_multiplicative : bool, optional
        Whether multiplicative models are allowed

    Returns
    -------
    list
        List of possible component values
    """
    possible = set()

    # Handle each special case:
    if comp_char == "Z":
        # Expand to A,N + M if allowed
        possible.update(["A", "N"])
        if allow_multiplicative:
            possible.add("M")
    elif comp_char == "C":
        # "C" is effectively "combine all" in R
        possible.update(["A", "N"])
        if allow_multiplicative:
            possible.add("M")
    elif comp_char == "F":
        # "full" => A, N, plus M if multiplicative
        possible.update(["A", "N"])
        if allow_multiplicative:
            possible.add("M")
    elif comp_char == "P":
        # "pure" => A plus M if multiplicative
        possible.update(["A"])
        if allow_multiplicative:
            possible.add("M")
    elif comp_char == "X":
        # R logic converts X->A
        possible.update(["A"])
    elif comp_char == "Y":
        # R logic converts Y->M if allowed, else A
        if allow_multiplicative:
            possible.update(["M"])
        else:
            possible.update(["A"])
    else:
        # If it's one of 'A','M','N' or an unknown letter, just return that
        possible.add(comp_char)

    return list(possible)


def _build_models_pool_from_components(
    error_type, trend_type, season_type, damped, allow_multiplicative
):
    """
    Build a models pool by fully enumerating expansions for E, T, S.

    Parameters
    ----------
    error_type : str
        Error component type
    trend_type : str
        Trend component type
    season_type : str
        Seasonal component type
    damped : bool
        Whether trend is damped
    allow_multiplicative : bool
        Whether multiplicative models are allowed

    Returns
    -------
    tuple
        (candidate_models, combined_mode)
    """
    err_options = _expand_component_code(error_type, allow_multiplicative)
    trend_options = _expand_component_code(trend_type, allow_multiplicative)
    seas_options = _expand_component_code(season_type, allow_multiplicative)

    # Check if 'C' is in any expansions => combined_mode
    combined_mode = "C" in error_type or "C" in trend_type or "C" in season_type

    # Build candidate models
    candidate_models = []
    for e in err_options:
        for t in trend_options:
            for s in seas_options:
                # Add 'd' if damped
                if damped and t not in ["N"]:
                    candidate_models.append(f"{e}{t}d{s}")
                else:
                    candidate_models.append(f"{e}{t}{s}")

    candidate_models = list(set(candidate_models))  # unique
    candidate_models.sort()

    return candidate_models, combined_mode


def _check_model_composition(model_str, allow_multiplicative=True, silent=False):
    """
    Parse and validate model composition string.

    Parameters
    ----------
    model_str : str
        String like "ANN", "ZZZ", etc. representing model components
    allow_multiplicative : bool, optional
        Whether multiplicative models are allowed
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with model components and configuration
    """
    # Initialize defaults
    error_type = trend_type = season_type = "N"
    damped = False
    model_do = "estimate"
    models_pool = None

    # Validate model string
    if not isinstance(model_str, str):
        if not silent:
            _warn(
                f"Invalid model type: {model_str}. Should be a string. Switching to 'ZZZ'."
            )
        model_str = "ZZZ"

    # Handle 4-character models (with damping)
    if len(model_str) == 4:
        error_type = model_str[0]
        trend_type = model_str[1]
        season_type = model_str[3]
        if model_str[2] != "d":
            if not silent:
                _warn(f"Invalid damped trend specification in {model_str}. Using 'd'.")
        damped = True

    # Handle 3-character models
    elif len(model_str) == 3:
        error_type = model_str[0]
        trend_type = model_str[1]
        season_type = model_str[2]
        damped = trend_type in ["Z", "X", "Y"]

    else:
        if not silent:
            _warn(f"Invalid model string length: {model_str}. Switching to 'ZZZ'.")
        model_str = "ZZZ"
        error_type = trend_type = season_type = "Z"
        damped = True

    # Validate components
    valid_error = ["Z", "X", "Y", "A", "M", "C", "N", "F", "P"]
    valid_trend = ["Z", "X", "Y", "N", "A", "M", "C", "F", "P"]
    valid_season = ["Z", "X", "Y", "N", "A", "M", "C", "F", "P"]

    if error_type not in valid_error:
        if not silent:
            _warn(f"Invalid error type: {error_type}. Switching to 'Z'.")
        error_type = "Z"
        model_do = "select"

    if trend_type not in valid_trend:
        if not silent:
            _warn(f"Invalid trend type: {trend_type}. Switching to 'Z'.")
        trend_type = "Z"
        model_do = "select"

    if season_type not in valid_season:
        if not silent:
            _warn(f"Invalid seasonal type: {season_type}. Switching to 'Z'.")
        season_type = "Z"
        model_do = "select"

    # Handle model selection/combination mode
    if "C" in [error_type, trend_type, season_type]:
        model_do = "combine"
        # Replace C with Z for actual fitting
        if error_type == "C":
            error_type = "Z"
        if trend_type == "C":
            trend_type = "Z"
        if season_type == "C":
            season_type = "Z"
    elif any(
        c in ["Z", "X", "Y", "F", "P"] for c in [error_type, trend_type, season_type]
    ):
        model_do = "select"

    # Handle multiplicative restrictions
    if not allow_multiplicative:
        if error_type == "M":
            error_type = "A"
        if trend_type == "M":
            trend_type = "A"
        if season_type == "M":
            season_type = "A"
        if error_type == "Y":
            error_type = "X"
        if trend_type == "Y":
            trend_type = "X"
        if season_type == "Y":
            season_type = "X"

    # Generate models pool if needed
    if model_do in ["select", "combine"]:
        models_pool, _ = _build_models_pool_from_components(
            error_type, trend_type, season_type, damped, allow_multiplicative
        )

    # Return model components and info
    return {
        "model": model_str,
        "error_type": error_type,
        "trend_type": trend_type,
        "season_type": season_type,
        "damped": damped,
        "model_do": model_do,
        "models_pool": models_pool,
        "allow_multiplicative": allow_multiplicative,
    }


def _generate_models_pool(
    error_type, trend_type, season_type, allow_multiplicative, silent=False
):
    """
    Generate a pool of models based on component specifications.

    Parameters
    ----------
    error_type : str
        Error component type
    trend_type : str
        Trend component type
    season_type : str
        Seasonal component type
    allow_multiplicative : bool
        Whether multiplicative models are allowed
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    tuple
        (pool_small, pool_errors, pool_trends, pool_seasonals, check_trend, check_seasonal)
    """
    # Print status if not silent
    if not silent:
        print("Forming the pool of models based on... ", end="")

    # Define the whole pool of errors, trends, and seasonals
    if not allow_multiplicative:
        pool_errors = ["A"]
        pool_trends = ["N", "A", "Ad"]
        pool_seasonals = ["N", "A"]
    else:
        pool_errors = ["A", "M"]
        pool_trends = ["N", "A", "Ad", "M", "Md"]
        pool_seasonals = ["N", "A", "M"]

    # Prepare error type
    if error_type != "Z":
        pool_errors = [error_type]
        pool_errors_small = [error_type]
    else:
        pool_errors_small = ["A"]

    # Prepare trend type
    if trend_type != "Z":
        if trend_type == "X":
            pool_trends_small = ["N", "A"]
            pool_trends = ["N", "A", "Ad"]
            check_trend = True
        elif trend_type == "Y":
            pool_trends_small = ["N", "M"]
            pool_trends = ["N", "M", "Md"]
            check_trend = True
        else:
            damped = "d" in trend_type
            if damped:
                pool_trends = pool_trends_small = [trend_type]
            else:
                pool_trends = pool_trends_small = [trend_type]
            check_trend = False
    else:
        pool_trends_small = ["N", "A"]
        check_trend = True

    # Prepare seasonal type
    if season_type != "Z":
        if season_type == "X":
            pool_seasonals = pool_seasonals_small = ["N", "A"]
            check_seasonal = True
        elif season_type == "Y":
            pool_seasonals_small = ["N", "M"]
            pool_seasonals = ["N", "M"]
            check_seasonal = True
        else:
            pool_seasonals_small = [season_type]
            pool_seasonals = [season_type]
            check_seasonal = False
    else:
        pool_seasonals_small = ["N", "A", "M"]
        check_seasonal = True

    # Create the small pool
    pool_small = []
    for error in pool_errors_small:
        for trend in pool_trends_small:
            for seasonal in pool_seasonals_small:
                pool_small.append(error + trend + seasonal)

    # Align error and seasonality, if the error was not forced to be additive
    if any(model[2] == "M" for model in pool_small) and error_type not in ["A", "X"]:
        for i, model in enumerate(pool_small):
            if model[2] == "M":
                pool_small[i] = "M" + model[1:]

    return (
        pool_small,
        pool_errors,
        pool_trends,
        pool_seasonals,
        check_trend,
        check_seasonal,
    )


def _check_ets_model(model, distribution, data, silent=False):
    """
    Check ETS model and validate compatibility with data.

    Parameters
    ----------
    model : str
        Model string specification
    distribution : str
        Probability distribution to use
    data : array-like
        Time series data
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with ETS model information
    """
    # If ARIMA, return default ETS off settings
    if isinstance(model, (list, tuple)) or model in ["auto.arima", "ARIMA"]:
        return {
            "ets_model": False,
            "model": "ANN",
            "error_type": "A",
            "trend_type": "N",
            "season_type": "N",
            "damped": False,
            "allow_multiplicative": True,
        }

    # Determine if multiplicative models are allowed
    data_arr = np.array(data)
    allow_multiplicative = not np.any(np.asarray(data_arr) <= 0)

    # Check if pure ARIMA is requested (not hybrid)
    if isinstance(model, str) and model.upper() in ["ARIMA", "AUTO.ARIMA"]:
        return {
            "ets_model": False,
            "model": "ANN",
            "error_type": "A",
            "trend_type": "N",
            "season_type": "N",
            "damped": False,
            "allow_multiplicative": allow_multiplicative,
        }

    # Check if this is an ETS model
    if isinstance(model, str) and len(model) in [3, 4]:
        # Parse model configuration
        model_info = _check_model_composition(model, allow_multiplicative, silent)

        # Check for multiplicative compatibility with data
        if not allow_multiplicative:
            if model_info["error_type"] == "M":
                _warn(
                    "Switching to additive error because data has non-positive values.",
                    silent,
                )
                model_info["error_type"] = "A"
            if model_info["trend_type"] == "M":
                _warn(
                    "Switching to additive trend because data has non-positive values.",
                    silent,
                )
                model_info["trend_type"] = "A"
            if model_info["season_type"] == "M":
                _warn(
                    "Switching to additive seasonal because data has non-positive values.",
                    silent,
                )
                model_info["season_type"] = "A"

            # Update model string based on changes
            model_str = f"{model_info['error_type']}{model_info['trend_type']}"
            if model_info["damped"]:
                model_str += "d"
            model_str += model_info["season_type"]
            model_info["model"] = model_str

        # Indicate this is an ETS model
        model_info["ets_model"] = True
        model_info["allow_multiplicative"] = allow_multiplicative

        return model_info

    # Default to ANN for unrecognized models
    _warn(f"Unknown model type: {model}. Switching to 'ANN'.", silent)
    return {
        "ets_model": True,
        "model": "ANN",
        "error_type": "A",
        "trend_type": "N",
        "season_type": "N",
        "damped": False,
        "allow_multiplicative": allow_multiplicative,
    }


def _expand_orders(orders):
    """
    Expand ARIMA orders into AR, I, MA components.

    Parameters
    ----------
    orders : list, tuple, int, or None
        ARIMA order specification

    Returns
    -------
    tuple
        (ar_orders, i_orders, ma_orders)
    """
    # Default values
    ar_orders = i_orders = ma_orders = [0]

    if orders is None:
        return ar_orders, i_orders, ma_orders

    # Handle different input types
    if isinstance(orders, (list, tuple)):
        if len(orders) >= 3:
            ar_orders = (
                [orders[0]] if isinstance(orders[0], (int, float)) else orders[0]
            )
            i_orders = [orders[1]] if isinstance(orders[1], (int, float)) else orders[1]
            ma_orders = (
                [orders[2]] if isinstance(orders[2], (int, float)) else orders[2]
            )
    elif isinstance(orders, (int, float)):
        # Single value -> assume AR order
        ar_orders = [orders]

    return ar_orders, i_orders, ma_orders


def _check_arima(orders, validated_lags, silent=False):
    """
    Check and validate ARIMA model specification.

    Parameters
    ----------
    orders : list, tuple, int, or None
        ARIMA order specification
    validated_lags : list
        List of validated lags
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with ARIMA model information
    """
    # Initialize with default values
    arima_result = {
        "arima_model": False,
        "ar_orders": [0],
        "i_orders": [0],
        "ma_orders": [0],
        "ar_required": False,
        "i_required": False,
        "ma_required": False,
        "ar_estimate": False,
        "ma_estimate": False,
        "arma_parameters": None,
        "lags_model_arima": [],
        "non_zero_ari": [],
        "non_zero_ma": [],
        "select": False,
    }

    # If no orders specified, return default values
    if orders is None:
        return arima_result

    # Parse orders into components
    ar_orders, i_orders, ma_orders = _expand_orders(orders)

    # Check for valid ARIMA component
    if (sum(ar_orders) + sum(i_orders) + sum(ma_orders)) > 0:
        arima_result["arima_model"] = True
    else:
        return arima_result

    # Update ARIMA information
    arima_result["ar_orders"] = ar_orders
    arima_result["i_orders"] = i_orders
    arima_result["ma_orders"] = ma_orders

    # Determine required components
    arima_result["ar_required"] = sum(ar_orders) > 0
    arima_result["i_required"] = sum(i_orders) > 0
    arima_result["ma_required"] = sum(ma_orders) > 0

    # Calculate non-zero indices
    arima_result["non_zero_ari"] = [
        i for i, val in enumerate(ar_orders + i_orders) if val > 0
    ]
    arima_result["non_zero_ma"] = [i for i, val in enumerate(ma_orders) if val > 0]

    # Set estimation flags - always estimate parameters if they are non-zero
    arima_result["ar_estimate"] = arima_result["ar_required"]
    arima_result["ma_estimate"] = arima_result["ma_required"]

    # Set up lags for ARIMA
    arima_lags = []
    if arima_result["ar_required"]:
        for ar_order in ar_orders:
            arima_lags.extend(range(1, ar_order + 1))

    arima_result["lags_model_arima"] = sorted(set(arima_lags))

    # Initialize ARMA parameters (will be filled during estimation)
    arima_parameters = {}

    if arima_result["ar_required"]:
        for i in range(sum(ar_orders)):
            arima_parameters[f"phi{i+1}"] = 0.0

    if arima_result["ma_required"]:
        for i in range(sum(ma_orders)):
            arima_parameters[f"theta{i+1}"] = 0.0

    arima_result["arma_parameters"] = arima_parameters if arima_parameters else None

    return arima_result


def _check_distribution_loss(distribution, loss, silent=False):
    """
    Check distribution and loss function compatibility.

    Parameters
    ----------
    distribution : str
        Probability distribution
    loss : str
        Loss function name
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with validated distribution and loss
    """
    # Valid distribution types
    valid_distributions = [
        "default",
        "dnorm",
        "dlaplace",
        "ds",
        "dlogitnorm",
        "dlogis",
        "dt",
        "dgnorm",
        "dgamma",
        "dpois",
    ]

    # Valid loss functions
    valid_losses = [
        "likelihood",
        "MSE",
        "MAE",
        "HAM",
        "MSEh",
        "MAEh",
        "HAMh",
        "MSCE",
        "MACE",
        "CHAM",
        "TMSE",
        "GTMSE",
        "LASSO",
        "RIDGE",
    ]

    # Check distribution
    if distribution not in valid_distributions:
        _warn(f"Unknown distribution: {distribution}. Switching to 'default'.", silent)
        distribution = "default"

    # Check loss function
    if loss not in valid_losses:
        _warn(f"Unknown loss function: {loss}. Switching to 'likelihood'.", silent)
        loss = "likelihood"

    return {"distribution": distribution, "loss": loss}


def _check_outliers(outliers_mode, silent=False):
    """
    Check outliers handling mode.

    Parameters
    ----------
    outliers_mode : str
        Outliers handling mode
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    str
        Validated outliers mode
    """
    valid_modes = ["ignore", "replace"]

    if outliers_mode not in valid_modes:
        _warn(f"Unknown outliers mode: {outliers_mode}. Switching to 'ignore'.", silent)
        return "ignore"

    return outliers_mode


def _check_phi(phi, damped, silent=False):
    """
    Check damping parameter phi.

    Parameters
    ----------
    phi : float or None
        Damping parameter value
    damped : bool
        Whether trend is damped
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with phi information
    """
    # If damped, but phi not provided, use default
    if damped and phi is None:
        phi = 0.95
        phi_estimate = True
    # If damped and phi provided as numeric
    elif damped and isinstance(phi, (int, float)):
        if phi >= 1 or phi <= 0:
            _warn(
                f"Invalid phi value: {phi}. Must be in (0, 1). Setting to 0.95.", silent
            )
            phi = 0.95
        phi_estimate = False
    # If damped and phi provided as boolean/string "estimate"
    elif damped and (
        phi is True or (isinstance(phi, str) and phi.lower() == "estimate")
    ):
        phi = 0.95
        phi_estimate = True
    # If not damped, phi is fixed at 1
    elif not damped:
        phi = 1.0
        phi_estimate = False
    # Default fallback
    else:
        _warn(f"Invalid phi specification: {phi}. Using default value.", silent)
        phi = 0.95 if damped else 1.0
        phi_estimate = damped

    return {"phi": phi, "phi_estimate": phi_estimate}


def _check_persistence(
    persistence,
    ets_model,
    trend_type,
    season_type,
    lags_model_seasonal,
    xreg_model=False,
    silent=False,
):
    """
    Check persistence parameters.

    Parameters
    ----------
    persistence : float, list, dict, or None
        Persistence parameter specification
    ets_model : bool
        Whether ETS model is used
    trend_type : str
        Trend component type
    season_type : str
        Seasonal component type
    lags_model_seasonal : list
        List of seasonal lags
    xreg_model : bool, optional
        Whether external regressors are used
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with persistence parameters
    """
    # Initialize defaults
    result = {
        "persistence": None,
        "persistence_estimate": True,
        "persistence_level": None,
        "persistence_level_estimate": True,
        "persistence_trend": None,
        "persistence_trend_estimate": True,
        "persistence_seasonal": None,
        "persistence_seasonal_estimate": True,
        "persistence_xreg": None,
        "persistence_xreg_estimate": True,
        "persistence_xreg_provided": False,
    }

    # Handle None case
    if persistence is None:
        return result

    # Handle single numeric value
    if isinstance(persistence, (int, float)):
        # Apply the same value to all components
        result["persistence"] = persistence
        result["persistence_level"] = persistence
        result["persistence_trend"] = persistence
        result["persistence_seasonal"] = persistence

        # Mark all as not estimated
        result["persistence_estimate"] = False
        result["persistence_level_estimate"] = False
        result["persistence_trend_estimate"] = False
        result["persistence_seasonal_estimate"] = False

        return result

    # Handle list/tuple of values
    if isinstance(persistence, (list, tuple)):
        # Check if the length is appropriate for the model
        expected_length = (
            1 + (trend_type != "N") + (len(lags_model_seasonal) > 0) + xreg_model
        )

        if len(persistence) > expected_length:
            _warn(
                f"Too many persistence values provided ({len(persistence)}). Expected at most {expected_length}.",
                silent,
            )

        # Assign values based on position
        pos = 0
        if ets_model:
            if pos < len(persistence):
                result["persistence_level"] = persistence[pos]
                result["persistence_level_estimate"] = False
                pos += 1

            if trend_type != "N" and pos < len(persistence):
                result["persistence_trend"] = persistence[pos]
                result["persistence_trend_estimate"] = False
                pos += 1

            if len(lags_model_seasonal) > 0 and pos < len(persistence):
                result["persistence_seasonal"] = persistence[pos]
                result["persistence_seasonal_estimate"] = False
                pos += 1

        if xreg_model and pos < len(persistence):
            result["persistence_xreg"] = persistence[pos]
            result["persistence_xreg_estimate"] = False
            result["persistence_xreg_provided"] = True

        # Mark overall persistence as not estimated
        result["persistence_estimate"] = False

        return result

    # Handle dictionary of named parameters
    if isinstance(persistence, dict):
        # Process level persistence
        if "level" in persistence:
            result["persistence_level"] = persistence["level"]
            result["persistence_level_estimate"] = False

        # Process trend persistence
        if "trend" in persistence and trend_type != "N":
            result["persistence_trend"] = persistence["trend"]
            result["persistence_trend_estimate"] = False

        # Process seasonal persistence
        if "seasonal" in persistence and len(lags_model_seasonal) > 0:
            result["persistence_seasonal"] = persistence["seasonal"]
            result["persistence_seasonal_estimate"] = False

        # Process xreg persistence
        if "xreg" in persistence and xreg_model:
            result["persistence_xreg"] = persistence["xreg"]
            result["persistence_xreg_estimate"] = False
            result["persistence_xreg_provided"] = True

        # Mark overall persistence as not estimated
        result["persistence_estimate"] = False

        return result

    # Invalid persistence type
    _warn(
        f"Invalid persistence type: {type(persistence)}. Using default values.", silent
    )
    return result


def _check_initial(
    initial,
    ets_model,
    trend_type,
    season_type,
    arima_model=False,
    xreg_model=False,
    silent=False,
):
    """
    Check initial values for state components.

    Parameters
    ----------
    initial : float, list, dict, or None
        Initial values specification
    ets_model : bool
        Whether ETS model is used
    trend_type : str
        Trend component type
    season_type : str
        Seasonal component type
    arima_model : bool, optional
        Whether ARIMA model is used
    xreg_model : bool, optional
        Whether external regressors are used
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with initial values
    """
    # Initialize defaults
    result = {
        "initial": None,
        "initial_type": "optimal",
        "initial_estimate": True,
        "initial_level": None,
        "initial_level_estimate": True,
        "initial_trend": None,
        "initial_trend_estimate": True,
        "initial_seasonal": None,
        "initial_seasonal_estimate": True,
        "initial_arima": None,
        "initial_arima_estimate": True,
        "initial_arima_number": 0,
        "initial_xreg_estimate": True,
        "initial_xreg_provided": False,
    }

    # Handle None case
    if initial is None:
        return result

    # Handle "optimal" or "backcasting" strings
    if isinstance(initial, str):
        if initial.lower() in ["optimal", "backcasting"]:
            result["initial_type"] = initial.lower()
            return result
        else:
            _warn(f"Unknown initial value method: {initial}. Using 'optimal'.", silent)
            return result

    # Handle numeric values
    if isinstance(initial, (int, float)):
        # Set all initial values to this value
        result["initial"] = initial
        result["initial_level"] = initial
        result["initial_trend"] = initial if trend_type != "N" else None
        result["initial_seasonal"] = initial if season_type != "N" else None
        result["initial_arima"] = initial if arima_model else None

        # Mark all as not estimated
        result["initial_estimate"] = False
        result["initial_level_estimate"] = False
        result["initial_trend_estimate"] = False
        result["initial_seasonal_estimate"] = False
        result["initial_arima_estimate"] = False

        # Set type to backcasting
        result["initial_type"] = "provided"

        return result

    # Handle list/tuple of values
    if isinstance(initial, (list, tuple)):
        result["initial_type"] = "provided"

        # Figure out the expected number of values
        expected_components = (
            1 + (trend_type != "N") + (season_type != "N") + arima_model
        )

        if len(initial) > expected_components:
            _warn(
                f"Too many initial values provided ({len(initial)}). Expected at most {expected_components}.",
                silent,
            )

        # Assign values based on position
        pos = 0

        # Level
        if ets_model and pos < len(initial):
            result["initial_level"] = initial[pos]
            result["initial_level_estimate"] = False
            pos += 1

        # Trend
        if ets_model and trend_type != "N" and pos < len(initial):
            result["initial_trend"] = initial[pos]
            result["initial_trend_estimate"] = False
            pos += 1

        # Seasonal
        if ets_model and season_type != "N" and pos < len(initial):
            result["initial_seasonal"] = initial[pos]
            result["initial_seasonal_estimate"] = False
            pos += 1

        # ARIMA
        if arima_model and pos < len(initial):
            result["initial_arima"] = initial[pos]
            result["initial_arima_estimate"] = False

            # Set ARIMA initial number
            if isinstance(initial[pos], (list, np.ndarray)):
                result["initial_arima_number"] = len(initial[pos])
            else:
                result["initial_arima_number"] = 1

            pos += 1

        # Mark overall initial as not estimated
        result["initial_estimate"] = False

        return result

    # Handle dictionary of named parameters
    if isinstance(initial, dict):
        result["initial_type"] = "provided"

        # Process level initial
        if "level" in initial:
            result["initial_level"] = initial["level"]
            result["initial_level_estimate"] = False

        # Process trend initial
        if "trend" in initial and trend_type != "N":
            result["initial_trend"] = initial["trend"]
            result["initial_trend_estimate"] = False

        # Process seasonal initial
        if "seasonal" in initial and season_type != "N":
            result["initial_seasonal"] = initial["seasonal"]
            result["initial_seasonal_estimate"] = False

        # Process ARIMA initial
        if "arima" in initial and arima_model:
            result["initial_arima"] = initial["arima"]
            result["initial_arima_estimate"] = False

            # Set ARIMA initial number
            if isinstance(initial["arima"], (list, np.ndarray)):
                result["initial_arima_number"] = len(initial["arima"])
            else:
                result["initial_arima_number"] = 1

        # Mark overall initial as not estimated
        result["initial_estimate"] = False

        return result

    # Invalid initial type
    _warn(f"Invalid initial type: {type(initial)}. Using default values.", silent)
    return result


def _check_constant(constant, silent=False):
    """
    Check constant term parameter.

    Parameters
    ----------
    constant : bool, float, or None
        Constant term specification
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with constant term information
    """
    # Initialize with defaults
    result = {
        "constant_required": False,
        "constant_estimate": False,
        "constant_value": 0.0,
        "constant_name": "constant",
    }

    # Handle None case
    if constant is None:
        return result

    # Handle boolean case
    if isinstance(constant, bool):
        result["constant_required"] = constant
        result["constant_estimate"] = constant
        return result

    # Handle numeric case
    if isinstance(constant, (int, float)):
        result["constant_required"] = True
        result["constant_estimate"] = False
        result["constant_value"] = float(constant)
        return result

    # Invalid type
    _warn(
        f"Invalid constant type: {type(constant)}. Using default value (False).", silent
    )
    return result


def _initialize_estimation_params(
    loss, lambda_param, ets_info, arima_info, silent=False
):
    """
    Initialize parameters for estimation.

    Parameters
    ----------
    loss : str
        Loss function name
    lambda_param : float
        Lambda parameter for LASSO/RIDGE
    ets_info : dict
        ETS model information
    arima_info : dict
        ARIMA model information
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with estimation parameters
    """
    # Initialize result
    result = {"lambda": lambda_param}

    # Handle specialized loss functions
    if loss in ["LASSO", "RIDGE"]:
        # Adjust lambda if needed
        if lambda_param <= 0:
            _warn(f"Lambda must be positive for {loss}. Setting it to 1.", silent)
            result["lambda"] = 1.0

        # Set lambda directly in the result
        result["lambda_"] = result["lambda"]

    # Add ARMA parameters if needed
    if arima_info["arima_model"]:
        result["arma_params"] = arima_info["arma_parameters"]

    return result


def _organize_model_type_info(ets_info, arima_info, xreg_model=False):
    """
    Organize model type information into a consolidated dictionary.

    Parameters
    ----------
    ets_info : dict
        ETS model information
    arima_info : dict
        ARIMA model information
    xreg_model : bool, optional
        Whether external regressors are used

    Returns
    -------
    dict
        Consolidated model type information
    """
    # Create standard model type dictionary
    model_type_dict = {
        "ets_model": ets_info["ets_model"],
        "arima_model": arima_info["arima_model"],
        "xreg_model": xreg_model,
        "model": ets_info["model"],
        "error_type": ets_info["error_type"],
        "trend_type": ets_info["trend_type"],
        "season_type": ets_info["season_type"],
        "damped": ets_info["damped"],
        "allow_multiplicative": ets_info["allow_multiplicative"],
        "model_do": "estimate",  # Default, will be updated if needed
        "models_pool": None,  # Will be populated for model selection
        "model_is_trendy": ets_info["trend_type"] != "N",
        "model_is_seasonal": ets_info["season_type"] != "N",
    }

    return model_type_dict


def _organize_components_info(ets_info, arima_info, lags_model_seasonal):
    """
    Organize model components information.

    Parameters
    ----------
    ets_info : dict
        ETS model information
    arima_info : dict
        ARIMA model information
    lags_model_seasonal : list
        List of seasonal lags

    Returns
    -------
    dict
        Dictionary with components information
    """
    # Calculate number of ETS components
    if ets_info["ets_model"]:
        components_number_ets = 1 + (ets_info["trend_type"] != "N")
        components_number_ets_seasonal = len(lags_model_seasonal)
    else:
        components_number_ets = 0
        components_number_ets_seasonal = 0

    # Calculate number of ARIMA components
    components_number_arima = (
        (
            sum(arima_info["ar_orders"])
            + sum(arima_info["ma_orders"])
            + sum(arima_info["i_orders"])
        )
        if arima_info["arima_model"]
        else 0
    )

    # Create components dictionary
    components_dict = {
        "components_number_ets": components_number_ets,
        "components_number_ets_seasonal": components_number_ets_seasonal,
        "components_number_arima": components_number_arima,
    }

    return components_dict


def _organize_lags_info(
    validated_lags, lags_model, lags_model_seasonal, lags_model_arima, xreg_model=False
):
    """
    Organize lags information.

    Parameters
    ----------
    validated_lags : list
        List of validated lags
    lags_model : list
        List of model lags
    lags_model_seasonal : list
        List of seasonal lags
    lags_model_arima : list
        List of ARIMA lags
    xreg_model : bool, optional
        Whether external regressors are used

    Returns
    -------
    dict
        Dictionary with lags information
    """
    # Calculate the maximum lag
    lags_model_max = max(validated_lags) if validated_lags else 1

    # Create lags dictionary
    lags_dict = {
        "lags": validated_lags,
        "lags_model": lags_model,
        "lags_model_seasonal": lags_model_seasonal,
        "lags_model_arima": lags_model_arima,
        "lags_length": len(lags_model),
        "lags_model_max": lags_model_max,
        "lags_model_all": sorted(set(lags_model + lags_model_arima)),
    }

    return lags_dict


def _organize_occurrence_info(occurrence, occurrence_model, obs_in_sample, h=0):
    """
    Organize occurrence information.

    Parameters
    ----------
    occurrence : str
        Occurrence type
    occurrence_model : bool
        Whether occurrence model is used
    obs_in_sample : int
        Number of in-sample observations
    h : int, optional
        Forecast horizon

    Returns
    -------
    dict
        Dictionary with occurrence information
    """
    # Create occurrence dictionary
    occurrence_dict = {
        "occurrence": occurrence,
        "occurrence_model": occurrence_model,
        "oes_model": "none",  # Default OES model type
        "probability": None,  # Will be filled during estimation
        "occurrence_probability": None,  # Will be filled during estimation
        "occurrence_parameters": None,  # Will be filled during estimation
        "occurrence_y": None,  # Will be filled during estimation
    }

    return occurrence_dict


def _organize_phi_info(phi_val, phi_estimate):
    """
    Organize phi (damping) parameter information.

    Parameters
    ----------
    phi_val : float
        Phi value
    phi_estimate : bool
        Whether phi should be estimated

    Returns
    -------
    dict
        Dictionary with phi information
    """
    return {"phi": phi_val, "phi_estimate": phi_estimate}


def _calculate_ot_logical(
    data,
    occurrence,
    occurrence_model,
    obs_in_sample,
    frequency=None,
    h=0,
    holdout=False,
):
    """
    Calculate logical observation vector and observation time indices.

    Parameters
    ----------
    data : array-like
        Input time series data
    occurrence : str
        Occurrence type
    occurrence_model : bool
        Whether occurrence model is used
    obs_in_sample : int
        Number of in-sample observations
    frequency : str, optional
        Time series frequency
    h : int, optional
        Forecast horizon
    holdout : bool, optional
        Whether to use holdout data

    Returns
    -------
    dict
        Dictionary with observation information
    """
    # Convert data to numpy array if needed
    if hasattr(data, "values"):
        y_in_sample = (
            data.values.flatten() if hasattr(data.values, "flatten") else data.values
        )
    else:
        y_in_sample = np.asarray(data).flatten()

    # Handle holdout if requested and possible
    y_holdout = None
    if holdout and h > 0 and len(y_in_sample) > h:
        # Split the data
        y_holdout = y_in_sample[-h:]
        y_in_sample = y_in_sample[:-h]

    # Initial calculation - data != 0
    ot_logical = y_in_sample != 0

    # If occurrence is "none" and all values are non-zero, set all to True
    if occurrence == "none" and all(ot_logical):
        ot_logical = np.ones_like(ot_logical, dtype=bool)

    # If occurrence model is not used and occurrence is not "provided"
    if not occurrence_model and occurrence != "provided":
        ot_logical = np.ones_like(ot_logical, dtype=bool)

    # Determine frequency
    freq = "1"  # Default
    if (
        hasattr(data, "index")
        and hasattr(data.index, "freq")
        and data.index.freq is not None
    ):
        freq = data.index.freq

    # Get start time if available
    y_start = 0  # Default
    if hasattr(data, "index") and len(data.index) > 0:
        y_start = data.index[0]

    # Handle forecast start time
    if hasattr(data, "index") and len(data.index) > 0:
        if holdout and h > 0:
            y_forecast_start = data.index[-h]
        else:
            # Last data point + 1 period
            try:
                # For DatetimeIndex
                from pandas import DatetimeIndex

                if isinstance(data.index, DatetimeIndex):
                    # Get the last index and add one frequency unit
                    from pandas import Timedelta

                    last_idx = data.index[-1]
                    freq_delta = Timedelta(freq)
                    y_forecast_start = last_idx + freq_delta
                else:
                    # For numeric index
                    y_forecast_start = data.index[-1] + 1
            except (ImportError, AttributeError, ValueError):
                # Fallback: use the last index + 1
                y_forecast_start = data.index[-1] + 1
    else:
        # For non-indexed data, just use the total length
        y_forecast_start = len(y_in_sample)

    # Create basic result
    result = {
        "ot_logical": ot_logical,
        "ot": np.where(ot_logical, 1, 0),
        "y_in_sample": y_in_sample,
        "y_holdout": y_holdout,
        "frequency": freq,
        "y_start": y_start,
        "y_forecast_start": y_forecast_start,
    }

    # Add index information if available
    if hasattr(data, "index"):
        if holdout and h > 0:
            result["y_in_sample_index"] = data.index[:-h]
            result["y_forecast_index"] = data.index[-h:]
        else:
            result["y_in_sample_index"] = data.index
            # Create forecast index
            try:
                import pandas as pd

                result["y_forecast_index"] = pd.date_range(
                    start=y_forecast_start, periods=h, freq=freq
                )
            except (ImportError, ValueError, TypeError):
                # Fallback for non-time data
                result["y_forecast_index"] = None

    return result


def _calculate_parameters_number(
    ets_info, arima_info, xreg_info=None, constant_required=False
):
    """
    Calculate number of parameters in the model.

    Parameters
    ----------
    ets_info : dict
        ETS model information
    arima_info : dict
        ARIMA model information
    xreg_info : dict, optional
        External regressors information
    constant_required : bool, optional
        Whether constant is required

    Returns
    -------
    numpy.ndarray
        Array with parameter counts
    """
    # ETS parameters
    ets_param_count = 0
    if ets_info["ets_model"]:
        # Level
        ets_param_count += 1

        # Trend
        if ets_info["trend_type"] != "N":
            ets_param_count += 1

            # Damping
            if ets_info["damped"]:
                ets_param_count += 1

        # Seasonal
        if ets_info["season_type"] != "N":
            # One seasonal parameter per seasonal lag
            ets_param_count += len(ets_info.get("seasonal_lags", []))

    # ARIMA parameters
    arima_param_count = 0
    if arima_info["arima_model"]:
        # AR parameters
        if arima_info["ar_required"] and arima_info["ar_estimate"]:
            arima_param_count += sum(arima_info["ar_orders"])

        # MA parameters
        if arima_info["ma_required"] and arima_info["ma_estimate"]:
            arima_param_count += sum(arima_info["ma_orders"])

    # Exogenous variables parameters
    xreg_param_count = 0
    if xreg_info is not None and xreg_info.get("xreg_model", False):
        xreg_param_count = xreg_info.get("xreg_number", 0)

    # Constant parameter
    constant_param_count = 1 if constant_required else 0

    # Total parameters
    total_params = (
        ets_param_count + arima_param_count + xreg_param_count + constant_param_count
    )

    # Return as a 2D array with shape (1, 3)
    return np.array([[ets_param_count, arima_param_count, total_params]])


def _adjust_model_for_sample_size(
    model_info,
    obs_nonzero,
    lags_model_max,
    allow_multiplicative=True,
    xreg_number=0,
    silent=False,
):
    """
    Adjust model complexity based on sample size.

    Parameters
    ----------
    model_info : dict
        Model type information
    obs_nonzero : int
        Number of non-zero observations
    lags_model_max : int
        Maximum lag value
    allow_multiplicative : bool, optional
        Whether multiplicative models are allowed
    xreg_number : int, optional
        Number of external regressors
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Updated model information
    """
    result = model_info.copy()

    # Effective sample size after accounting for lags
    effective_sample = obs_nonzero - lags_model_max

    # Calculate number of parameters needed
    n_params = 0

    # Level parameter
    if model_info["ets_model"]:
        n_params += 1

    # Trend parameter
    if model_info["ets_model"] and model_info["trend_type"] != "N":
        n_params += 1

        # Damping parameter
        if model_info["damped"]:
            n_params += 1

    # Seasonal parameters
    if (
        model_info["ets_model"]
        and model_info["season_type"] != "N"
        and model_info.get("model_is_seasonal", False)
    ):
        # One parameter per seasonal lag
        n_params += model_info.get("components_number_ets_seasonal", 0)

    # External regressors
    n_params += xreg_number

    # Check if we have enough observations
    if effective_sample <= n_params:
        # Not enough observations, simplify the model
        _warn(
            f"Not enough non-zero observations ({obs_nonzero}) for the model with {n_params} parameters. "
            f"Switching to a simpler model.",
            silent,
        )

        # Start by reducing seasonality
        if result["season_type"] != "N":
            result["season_type"] = "N"
            result["model_is_seasonal"] = False
            n_params -= model_info.get("components_number_ets_seasonal", 0)

        # Then remove damping
        if result["damped"] and result["trend_type"] != "N":
            result["damped"] = False
            n_params -= 1

        # Finally remove trend
        if effective_sample <= n_params and result["trend_type"] != "N":
            result["trend_type"] = "N"
            result["model_is_trendy"] = False
            n_params -= 1

        # Update model string
        if result["damped"] and result["trend_type"] != "N":
            result["model"] = (
                f"{result['error_type']}{result['trend_type']}d{result['season_type']}"
            )
        else:
            result["model"] = (
                f"{result['error_type']}{result['trend_type']}{result['season_type']}"
            )

    return result


def parameters_checker(
    data,
    model,
    lags,
    orders=None,
    constant=False,
    outliers="ignore",
    level=0.99,
    persistence=None,
    phi=None,
    initial=None,
    distribution="default",
    loss="likelihood",
    h=0,
    holdout=False,
    occurrence="none",
    ic="AICc",
    bounds="usual",
    silent=False,
    model_do="estimate",
    fast=False,
    models_pool=None,
    lambda_param=None,
    frequency=None,
    interval="parametric",
    interval_level=[0.95],
    side="both",
    cumulative=False,
    nsim=1000,
    scenarios=100,
    ellipsis=None,
):
    """
    Check and process model parameters.

    This function validates all input parameters for an ADAM model and
    converts them into the standardized format required by model estimation.

    Parameters
    ----------
    data : array-like
        Input time series data
    model : str
        Model type specification (e.g., "ANN", "ZZZ")
    lags : list
        List of lag values
    orders : list, tuple, or None, optional
        ARIMA orders
    constant : bool or float, optional
        Whether to include a constant term
    outliers : str, optional
        Outliers handling mode
    level : float, optional
        Confidence level
    persistence : float, list, dict, or None, optional
        Persistence parameter specification
    phi : float or None, optional
        Damping parameter
    initial : float, list, dict, or None, optional
        Initial values specification
    distribution : str, optional
        Probability distribution
    loss : str, optional
        Loss function name
    h : int, optional
        Forecast horizon
    holdout : bool, optional
        Whether to use holdout data
    occurrence : str, optional
        Occurrence type for intermittent demand
    ic : str, optional
        Information criterion
    bounds : str, optional
        Parameter bounds type
    silent : bool, optional
        Whether to suppress warnings
    model_do : str, optional
        Model action ("estimate", "select", "combine")
    fast : bool, optional
        Whether to use fast estimation
    models_pool : list or None, optional
        Pool of models for selection
    lambda_param : float or None, optional
        Lambda parameter for LASSO/RIDGE
    frequency : str or None, optional
        Time series frequency
    interval : str, optional
        Prediction interval type
    interval_level : list, optional
        Prediction interval levels
    side : str, optional
        Prediction interval side
    cumulative : bool, optional
        Whether to use cumulative forecasts
    nsim : int, optional
        Number of simulations
    scenarios : int, optional
        Number of scenarios
    ellipsis : dict or None, optional
        Additional parameters

    Returns
    -------
    tuple
        Dictionaries with validated parameters
    """
    #####################
    # 1) Check Occurrence
    #####################
    occ_info = _check_occurrence(data, occurrence, frequency, silent, holdout, h)
    obs_in_sample = occ_info["obs_in_sample"]
    obs_nonzero = occ_info["obs_nonzero"]
    occurrence_model = occ_info["occurrence_model"]

    #####################
    # 2) Check Lags
    #####################
    lags_info = _check_lags(lags, obs_in_sample, silent)
    validated_lags = lags_info["lags"]
    lags_model = lags_info["lags_model"]
    lags_model_seasonal = lags_info["lags_model_seasonal"]
    max_lag = lags_info["max_lag"]

    #####################
    # 3) Check ETS Model
    #####################
    ets_info = _check_ets_model(model, distribution, data, silent)
    ets_model = ets_info["ets_model"]

    #####################
    # 4) Check ARIMA
    #####################
    arima_info = _check_arima(orders, validated_lags, silent)
    arima_model = arima_info["arima_model"]
    ar_orders = arima_info["ar_orders"]
    i_orders = arima_info["i_orders"]
    ma_orders = arima_info["ma_orders"]
    lags_model_arima = arima_info["lags_model_arima"]

    #####################
    # 5) Check Distribution & Loss
    #####################
    dist_info = _check_distribution_loss(distribution, loss, silent)
    distribution = dist_info["distribution"]
    loss = dist_info["loss"]

    #####################
    # 6) Check Outliers
    #####################
    outliers_mode = _check_outliers(outliers, silent)

    #####################
    # 7) Check Phi
    #####################
    phi_info = _check_phi(phi, ets_info["damped"], silent)
    phi_val = phi_info["phi"]
    phi_estimate = phi_info["phi_estimate"]

    #####################
    # 8) Check Persistence
    #####################
    persist_info = _check_persistence(
        persistence=persistence,
        ets_model=ets_model,
        trend_type=ets_info["trend_type"],
        season_type=ets_info["season_type"],
        lags_model_seasonal=lags_model_seasonal,
        xreg_model=False,  # Will be updated when xreg is implemented
        silent=silent,
    )

    #####################
    # 9) Check Initial Values
    #####################
    init_info = _check_initial(
        initial=initial,
        ets_model=ets_model,
        trend_type=ets_info["trend_type"],
        season_type=ets_info["season_type"],
        arima_model=arima_model,
        xreg_model=False,  # Will be updated when xreg is implemented
        silent=silent,
    )

    #####################
    # 10) Check Constant
    #####################
    constant_dict = _check_constant(constant, silent)

    #####################
    # 11) Check Bounds
    #####################
    if bounds not in ["usual", "admissible", "none"]:
        _warn(f"Unknown bounds='{bounds}'. Switching to 'usual'.", silent)
        bounds = "usual"

    #####################
    # 12) Check Holdout
    #####################
    if holdout and h <= 0:
        _warn(
            "holdout=TRUE but horizon 'h' is not positive. No real holdout can be made.",
            silent,
        )

    #####################
    # 13) Check Model Pool
    #####################
    # Check if multiplicative models are allowed
    if hasattr(data, "values"):
        actual_values = (
            data.values.flatten() if hasattr(data.values, "flatten") else data.values
        )
    else:
        actual_values = np.asarray(data).flatten()

    allow_multiplicative = not (
        (any(y <= 0 for y in actual_values if not np.isnan(y)) and not occurrence_model)
        or (occurrence_model and any(y < 0 for y in actual_values if not np.isnan(y)))
    )

    # Setup model type dictionary
    model_type_dict = _organize_model_type_info(ets_info, arima_info, xreg_model=False)

    # Check if model is valid for the sample size
    model_type_dict = _adjust_model_for_sample_size(
        model_info=model_type_dict,
        obs_nonzero=obs_nonzero,
        lags_model_max=max_lag,
        allow_multiplicative=allow_multiplicative,
        xreg_number=0,  # Will be updated when xreg is implemented
        silent=silent,
    )

    # Replace the model pools if provided
    if models_pool is not None:
        model_type_dict["models_pool"] = models_pool

    # Components info
    components_dict = _organize_components_info(
        ets_info, arima_info, lags_model_seasonal
    )

    # Create lags dictionary
    lags_dict = _organize_lags_info(
        validated_lags=validated_lags,
        lags_model=lags_model,
        lags_model_seasonal=lags_model_seasonal,
        lags_model_arima=lags_model_arima,
        xreg_model=False,  # Will be updated when xreg is implemented
    )

    # Create occurrence dictionary
    occurrence_dict = _organize_occurrence_info(
        occurrence=occ_info["occurrence"],
        occurrence_model=occurrence_model,
        obs_in_sample=obs_in_sample,
        h=h,
    )

    # Create phi dictionary
    phi_dict = _organize_phi_info(phi_val=phi_val, phi_estimate=phi_estimate)

    # Calculate observation logical vector
    ot_info = _calculate_ot_logical(
        data=data,
        occurrence=occurrence_dict["occurrence"],
        occurrence_model=occurrence_dict["occurrence_model"],
        obs_in_sample=obs_in_sample,
        frequency=frequency,
        h=h,
        holdout=holdout,
    )

    # Create observations dictionary
    observations_dict = {
        "obs_in_sample": obs_in_sample,
        "obs_nonzero": obs_nonzero,
        "obs_all": occ_info["obs_all"],
        "ot_logical": ot_info["ot_logical"],
        "ot": ot_info["ot"],
        "y_in_sample": ot_info.get("y_in_sample", data),
        "y_holdout": ot_info.get("y_holdout", None),
        "frequency": ot_info["frequency"],
        "y_start": ot_info["y_start"],
        "y_in_sample_index": ot_info.get("y_in_sample_index", None),
        "y_forecast_start": ot_info["y_forecast_start"],
        "y_forecast_index": ot_info.get("y_forecast_index", None),
    }

    # Create general dictionary with remaining parameters
    general_dict = {
        "distribution": distribution,
        "loss": loss,
        "outliers": outliers_mode,
        "h": h,
        "holdout": holdout,
        "ic": ic,
        "bounds": bounds,
        "model_do": model_do,
        "fast": fast,
        "models_pool": models_pool,
        "interval": interval,
        "interval_level": interval_level,
        "side": side,
        "cumulative": cumulative,
        "nsim": nsim,
        "scenarios": scenarios,
        "ellipsis": ellipsis,
    }

    # Initialize estimation parameters if needed
    if model_do == "estimate":
        est_params = _initialize_estimation_params(
            loss=loss,
            lambda_param=lambda_param or 1,  # Default to 1 if not provided
            ets_info=ets_info,
            arima_info=arima_info,
            silent=silent,
        )
        # Update general dict with estimation parameters
        general_dict.update(
            {
                "lambda": est_params.get("lambda", 1),
                "arma_params": est_params.get("arma_params", None),
            }
        )

    # Create persistence dictionary
    persistence_dict = {
        "persistence": persist_info["persistence"],
        "persistence_estimate": persist_info["persistence_estimate"],
        "persistence_level": persist_info["persistence_level"],
        "persistence_level_estimate": persist_info["persistence_level_estimate"],
        "persistence_trend": persist_info["persistence_trend"],
        "persistence_trend_estimate": persist_info["persistence_trend_estimate"],
        "persistence_seasonal": persist_info["persistence_seasonal"],
        "persistence_seasonal_estimate": persist_info["persistence_seasonal_estimate"],
        "persistence_xreg": persist_info["persistence_xreg"],
        "persistence_xreg_estimate": persist_info["persistence_xreg_estimate"],
        "persistence_xreg_provided": persist_info["persistence_xreg_provided"],
    }

    # Create initials dictionary
    initials_dict = {
        "initial": init_info["initial"],
        "initial_type": init_info["initial_type"],
        "initial_estimate": init_info["initial_estimate"],
        "initial_level": init_info["initial_level"],
        "initial_level_estimate": init_info["initial_level_estimate"],
        "initial_trend": init_info["initial_trend"],
        "initial_trend_estimate": init_info["initial_trend_estimate"],
        "initial_seasonal": init_info["initial_seasonal"],
        "initial_seasonal_estimate": init_info["initial_seasonal_estimate"],
        "initial_arima": init_info["initial_arima"],
        "initial_arima_estimate": init_info["initial_arima_estimate"],
        "initial_arima_number": init_info["initial_arima_number"],
        "initial_xreg_estimate": init_info["initial_xreg_estimate"],
        "initial_xreg_provided": init_info["initial_xreg_provided"],
    }

    # Create ARIMA dictionary
    arima_dict = {
        "arima_model": arima_model,
        "ar_orders": ar_orders,
        "i_orders": i_orders,
        "ma_orders": ma_orders,
        "ar_required": arima_info.get("ar_required", False),
        "i_required": arima_info.get("i_required", False),
        "ma_required": arima_info.get("ma_required", False),
        "ar_estimate": arima_info.get("ar_estimate", False),
        "ma_estimate": arima_info.get("ma_estimate", False),
        "arma_parameters": arima_info.get("arma_parameters", None),
        "non_zero_ari": arima_info.get("non_zero_ari", []),
        "non_zero_ma": arima_info.get("non_zero_ma", []),
        "select": arima_info.get("select", False),
    }

    # Initialize explanatory variables dictionary
    xreg_dict = {
        "xreg_model": False,
        "regressors": None,
        "xreg_model_initials": None,
        "xreg_data": None,
        "xreg_number": 0,
        "xreg_names": None,
        "response_name": None,
        "formula": None,
        "xreg_parameters_missing": None,
        "xreg_parameters_included": None,
        "xreg_parameters_estimated": None,
        "xreg_parameters_persistence": None,
    }

    # Calculate number of parameters
    params_info = _calculate_parameters_number(
        ets_info=ets_info,
        arima_info=arima_info,
        xreg_info=None,  # Will be updated when xreg is implemented
        constant_required=constant_dict["constant_required"],
    )

    # Set parameters number in general dict
    general_dict["parameters_number"] = params_info

    # Return all dictionaries
    return (
        general_dict,
        observations_dict,
        persistence_dict,
        initials_dict,
        arima_dict,
        constant_dict,
        model_type_dict,
        components_dict,
        lags_dict,
        occurrence_dict,
        phi_dict,
        xreg_dict,
        params_info,
    )
