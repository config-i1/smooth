import numpy as np
import pandas as pd


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
    obs_all = obs_in_sample + (h if holdout else 0)
    # Identify non-zero observations
    nonzero_indices = [i for i, val in enumerate(data_list) if val is not None and val != 0]
    obs_nonzero = len(nonzero_indices)

    # If all zeroes, fallback
    if all(val == 0 for val in data):
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
    error_type_char, trend_type_char, season_type_char, damped_original_model, allow_multiplicative
):
    """
    Build a models pool by fully enumerating expansions for E, T, S.
    This version aims to replicate the pool generation of the older _generate_models_pool.

    Parameters
    ----------
    error_type_char : str
        Error component character (e.g., 'Z', 'A', 'F', 'P')
        These are characters *after* initial processing in _check_model_composition
        (e.g., C->Z, M->A if not allow_multiplicative for specific components).
    trend_type_char : str
        Trend component character, similarly processed.
    season_type_char : str
        Seasonal component character, similarly processed.
    damped_original_model : bool
        Damping flag from the original model string (largely ignored here for pool generation,
        as damping is part of trend strings like "Ad").
    allow_multiplicative : bool
        Whether multiplicative models are generally allowed for pool expansion (e.g. for Z).

    Returns
    -------
    tuple
        (candidate_models, combined_mode_placeholder)
    """
    candidate_models = []

    # Handle full pool case ("F")
    if "F" in [error_type_char, trend_type_char, season_type_char]:
        candidate_models = ["ANN", "AAN", "AAdN", "AMN", "AMdN",
                            "ANA", "AAA", "AAdA", "AMA", "AMdA",
                            "ANM", "AAM", "AAdM", "AMM", "AMdM"]
        if allow_multiplicative: # This global allow_multiplicative governs the F/P pool extension
            candidate_models.extend([
                "MNN", "MAN", "MAdN", "MMN", "MMdN",
                "MNA", "MAA", "MAdA", "MMA", "MMdA",
                "MNM", "MAM", "MAdM", "MMM", "MMdM"
            ])
    # Handle pure models case ("P")
    elif "P" in [error_type_char, trend_type_char, season_type_char]:
        candidate_models = ["ANN", "AAN", "AAdN", "ANA", "AAA", "AAdA"]
        if allow_multiplicative: # This global allow_multiplicative governs the F/P pool extension
            candidate_models.extend(["MNN", "MMN", "MMdN", "MNM", "MMM", "MMdM"])
    # Handle standard selection case
    else:
        # Check for multiplicative requests on non-positive data
        if not allow_multiplicative:
            if error_type_char in ['Y', 'Z']:
                import warnings
                warnings.warn(
                    "Multiplicative error models cannot be used on non-positive data. "
                    "Switching to additive error (A).",
                    UserWarning
                )
                error_type_char = 'A'
            if trend_type_char in ['Y', 'Z']:
                import warnings
                warnings.warn(
                    "Multiplicative trend models cannot be used on non-positive data. "
                    "Switching to additive trend selection (X).",
                    UserWarning
                )
                trend_type_char = 'X'
            if season_type_char in ['Y', 'Z']:
                import warnings
                warnings.warn(
                    "Multiplicative seasonal models cannot be used on non-positive data. "
                    "Switching to additive seasonal selection (X).",
                    UserWarning
                )
                season_type_char = 'X'

        # Determine Error Options
        if error_type_char in ['A', 'M']:
            actual_error_options = [error_type_char]
        else: # For 'Z', 'N', 'X', 'Y' etc.
            actual_error_options = ["A", "M"] if allow_multiplicative else ["A"]

        # Determine Trend Options
        # Note: trend_type_char would already be 'A' or 'Ad' if originally 'M' or 'Md'
        # and allow_multiplicative was false, due to pre-processing in _check_model_composition.
        if trend_type_char in ['N', 'A', 'M', 'Ad', 'Md']:
            actual_trend_options_with_damping = [trend_type_char]
        elif trend_type_char == 'Z':
            actual_trend_options_with_damping = ["N", "A", "M", "Ad", "Md"]
        elif trend_type_char == 'Y':
            actual_trend_options_with_damping = ["N", "M", "Md"]
        else:  # 'X' or any other
            actual_trend_options_with_damping = ["N", "A", "Ad"]

        # Determine Season Options
        # Note: season_type_char would already be 'A' if originally 'M'
        # and allow_multiplicative was false.
        if season_type_char in ['N', 'A', 'M']:
            actual_season_options = [season_type_char]
        elif season_type_char == 'Z':
            actual_season_options = ["N", "A", "M"]
        elif season_type_char == 'Y':
            actual_season_options = ["N", "M"]
        else:  # 'X' or any other
            actual_season_options = ["N", "A"]
        
        for e_opt in actual_error_options:
            for t_opt in actual_trend_options_with_damping:
                # t_opt already includes 'd' if it's a damped trend (e.g., "Ad")
                for s_opt in actual_season_options:
                    candidate_models.append(f"{e_opt}{t_opt}{s_opt}")

    candidate_models = sorted(list(set(candidate_models)))
    # combined_mode is determined upstream in _check_model_composition and used there.
    # This function just needs to return the pool.
    return candidate_models, False


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
        # print(models_pool) # Removed print statement

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
    orders : list, tuple, int, dict, or None
        ARIMA order specification. Can be:
        - dict with 'ar', 'i', 'ma' keys
        - list/tuple of [ar, i, ma] values
        - single int (interpreted as AR order)

    Returns
    -------
    tuple
        (ar_orders, i_orders, ma_orders)
    """
    # Default values
    ar_orders = i_orders = ma_orders = [0]

    if orders is None:
        return ar_orders, i_orders, ma_orders

    # Handle dict input (from ADAM class)
    if isinstance(orders, dict):
        ar = orders.get("ar", 0)
        i = orders.get("i", 0)
        ma = orders.get("ma", 0)
        ar_orders = [ar] if isinstance(ar, (int, float)) else list(ar) if ar else [0]
        i_orders = [i] if isinstance(i, (int, float)) else list(i) if i else [0]
        ma_orders = [ma] if isinstance(ma, (int, float)) else list(ma) if ma else [0]
    # Handle list/tuple input
    elif isinstance(orders, (list, tuple)):
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

    This function mirrors R's parametersChecker ARIMA handling in adamGeneral.R lines 519-666.

    Parameters
    ----------
    orders : list, tuple, int, or None
        ARIMA order specification
    validated_lags : list
        List of validated lags (must include 1 as first element for non-seasonal)
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with ARIMA model information including:
        - non_zero_ari: Nx2 matrix [polynomial_index, state_index] (0-indexed for Python)
        - non_zero_ma: Nx2 matrix [polynomial_index, state_index] (0-indexed for Python)
        - lags_model_arima: list of lag values for ARIMA states
        - components_number_arima: number of ARIMA state components
    """
    import numpy as np

    # Initialize with default values - R lines 652-666
    arima_result = {
        "arima_model": False,
        "ar_orders": None,
        "i_orders": None,
        "ma_orders": None,
        "ar_required": False,
        "i_required": False,
        "ma_required": False,
        "ar_estimate": False,
        "ma_estimate": False,
        "arma_parameters": None,
        "lags_model_arima": [],
        "non_zero_ari": np.zeros((0, 2), dtype=int),
        "non_zero_ma": np.zeros((0, 2), dtype=int),
        "components_number_arima": 0,
        "components_names_arima": [],
        "initial_arima_number": 0,
        "select": False,
    }

    # If no orders specified, return default values
    if orders is None:
        return arima_result

    # Parse orders into components - R lines 521-538
    ar_orders, i_orders, ma_orders = _expand_orders(orders)

    # Check for valid ARIMA component - R lines 541-542
    if (sum(ar_orders) + sum(i_orders) + sum(ma_orders)) == 0:
        return arima_result

    arima_result["arima_model"] = True

    # See if AR/I/MA is needed - R lines 544-560
    ar_required = sum(ar_orders) > 0
    i_required = sum(i_orders) > 0
    ma_required = sum(ma_orders) > 0

    arima_result["ar_required"] = ar_required
    arima_result["i_required"] = i_required
    arima_result["ma_required"] = ma_required

    # Ensure lags start with 1
    lags = list(validated_lags) if validated_lags else [1]
    if lags[0] != 1:
        lags = [1] + lags

    # Define maxOrder and align orders with lags - R lines 563-578
    max_order = max(len(ar_orders), len(i_orders), len(ma_orders), len(lags))

    # Pad orders with zeros to match max_order
    ar_orders = list(ar_orders) + [0] * (max_order - len(ar_orders))
    i_orders = list(i_orders) + [0] * (max_order - len(i_orders))
    ma_orders = list(ma_orders) + [0] * (max_order - len(ma_orders))

    # If lags shorter than max_order, filter orders by non-zero lags - R lines 573-578
    if len(lags) < max_order:
        lags_new = list(lags) + [0] * (max_order - len(lags))
        # Filter to keep only orders where lags are non-zero
        ar_orders = [ar_orders[i] for i in range(len(lags_new)) if lags_new[i] != 0]
        i_orders = [i_orders[i] for i in range(len(lags_new)) if lags_new[i] != 0]
        ma_orders = [ma_orders[i] for i in range(len(lags_new)) if lags_new[i] != 0]
    else:
        # Make sure lags matches the length
        lags = list(lags[:max_order])

    # If after filtering all orders are zero, return no ARIMA - R lines 632-646
    if all(o == 0 for o in ar_orders + i_orders + ma_orders):
        return arima_result

    arima_result["ar_orders"] = ar_orders
    arima_result["i_orders"] = i_orders
    arima_result["ma_orders"] = ma_orders

    # Define the non-zero values via polynomial computation - R lines 580-616
    # This computes all possible lag positions in the ARI and MA polynomials
    ari_values = []
    ma_values = []

    for i in range(len(lags)):
        # ARI values for this lag - R lines 583-588
        ari_for_lag = [0]
        if ar_orders[i] > 0:
            ari_for_lag.extend(range(1, ar_orders[i] + 1))
        if i_orders[i] > 0:
            ari_for_lag.extend(range(ar_orders[i] + 1, ar_orders[i] + i_orders[i] + 1))
        # Multiply by lag and take unique - R line 588
        ari_for_lag = list(set([v * lags[i] for v in ari_for_lag]))
        ari_values.append(ari_for_lag)

        # MA values for this lag - R line 589
        ma_for_lag = [0]
        if ma_orders[i] > 0:
            ma_for_lag.extend(range(1, ma_orders[i] + 1))
        ma_for_lag = list(set([v * lags[i] for v in ma_for_lag]))
        ma_values.append(ma_for_lag)

    # Produce ARI polynomial lag positions - R lines 592-603
    # This creates all combinations of lag positions across seasonal factors
    def expand_polynomial(values_list):
        """Expand polynomial by multiplying across seasonal factors."""
        if len(values_list) == 0:
            return [0]
        result = values_list[0]
        for i in range(1, len(values_list)):
            new_result = []
            for r in result:
                for v in values_list[i]:
                    new_result.append(r + v)
            result = new_result
        return result

    ari_polynomial = expand_polynomial(ari_values)
    ma_polynomial = expand_polynomial(ma_values)

    # What are the non-zero ARI and MA polynomials? - R lines 618-625
    # Remove the first element (which corresponds to L^0 = 1 coefficient) and get unique
    non_zero_ari_lags = sorted(set([x for x in ari_polynomial if x > 0]))
    non_zero_ma_lags = sorted(set([x for x in ma_polynomial if x > 0]))

    # Lags for the ARIMA components - R line 623
    lags_model_arima = sorted(set(non_zero_ari_lags + non_zero_ma_lags))

    if len(lags_model_arima) == 0:
        # No ARIMA states needed
        arima_result["arima_model"] = False
        return arima_result

    # Create nonZeroARI matrix - R line 624
    # Column 0: polynomial index (position in ariPolynomial, 0-indexed for Python)
    # Column 1: state index (position in lagsModelARIMA, 0-indexed for Python)
    non_zero_ari = []
    for lag in non_zero_ari_lags:
        poly_idx = lag  # The lag value itself serves as index into polynomial
        state_idx = lags_model_arima.index(lag)
        non_zero_ari.append([poly_idx, state_idx])

    non_zero_ma = []
    for lag in non_zero_ma_lags:
        poly_idx = lag
        state_idx = lags_model_arima.index(lag)
        non_zero_ma.append([poly_idx, state_idx])

    # Convert to numpy arrays
    non_zero_ari = np.array(non_zero_ari, dtype=int) if non_zero_ari else np.zeros((0, 2), dtype=int)
    non_zero_ma = np.array(non_zero_ma, dtype=int) if non_zero_ma else np.zeros((0, 2), dtype=int)

    # Number of components - R line 628
    components_number_arima = len(lags_model_arima)

    # Component names - R lines 629-630
    if components_number_arima > 1:
        components_names_arima = [f"ARIMAState{i+1}" for i in range(components_number_arima)]
    else:
        components_names_arima = ["ARIMAState1"] if components_number_arima > 0 else []

    # Number of initials needed - R line 649
    initial_arima_number = max(lags_model_arima) if lags_model_arima else 0

    # Update result
    arima_result["non_zero_ari"] = non_zero_ari
    arima_result["non_zero_ma"] = non_zero_ma
    arima_result["lags_model_arima"] = lags_model_arima
    arima_result["components_number_arima"] = components_number_arima
    arima_result["components_names_arima"] = components_names_arima
    arima_result["initial_arima_number"] = initial_arima_number

    # Set estimation flags - always estimate parameters if they are required
    arima_result["ar_estimate"] = ar_required
    arima_result["ma_estimate"] = ma_required

    # Initialize ARMA parameters (will be filled during estimation)
    arima_parameters = []
    if ar_required:
        for i in range(len(ar_orders)):
            for j in range(ar_orders[i]):
                arima_parameters.append(0.0)
    if ma_required:
        for i in range(len(ma_orders)):
            for j in range(ma_orders[i]):
                arima_parameters.append(0.0)

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
    n_seasonal = len(lags_model_seasonal) if lags_model_seasonal else 0
    result = {
        "persistence": None,
        "persistence_estimate": True,
        "persistence_level": None,
        "persistence_level_estimate": True,
        "persistence_trend": None,
        "persistence_trend_estimate": True,
        "persistence_seasonal": [None] * n_seasonal,
        "persistence_seasonal_estimate": [True] * n_seasonal,
        "persistence_xreg": None,
        "persistence_xreg_estimate": True,
        "persistence_xreg_provided": False
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
        if initial.lower() in ["optimal", "backcasting", "complete", "two-stage"]:
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
        "constant_value": None,
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
        "model_do": ets_info.get("model_do", "estimate"),  # Use model_do from ets_info
        "models_pool": ets_info.get("models_pool", None),  # Use models_pool from ets_info
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
        components_number_ets = 1 + (ets_info["trend_type"] != "N") + (ets_info["season_type"] != "N")
        components_number_ets_seasonal = len(lags_model_seasonal) if ets_info["season_type"] != "N" else 0
        components_number_ets_non_seasonal = components_number_ets - components_number_ets_seasonal
    else:
        components_number_ets = 0
        components_number_ets_seasonal = 0
        components_number_ets_non_seasonal = 0

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
        "components_number_ets_non_seasonal": components_number_ets_non_seasonal,
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
    if frequency is not None:
        freq = frequency
    else:
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
                    if hasattr(data.index, 'freq') and data.index.freq is not None:
                        y_forecast_start = data.index[-1] + data.index.freq
                    else:
                        # Fallback for numeric index without freq - use integer
                        y_forecast_start = int(data.index[-1]) + 1
            except (ImportError, AttributeError, ValueError):
                # Fallback: use the last index + freq
                if hasattr(data.index, 'freq') and data.index.freq is not None:
                    y_forecast_start = data.index[-1] + data.index.freq
                else:
                    # Ultimate fallback for numeric index - use integer
                    y_forecast_start = int(data.index[-1]) + 1
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


def _calculate_parameters_number(ets_info, arima_info, xreg_info=None, constant_required=False):
    """Calculate number of parameters for different model components.
    
    Returns a 2x1 array-like structure similar to R's parametersNumber matrix:
    - Row 1: Number of states/components
    - Row 2: Number of parameters to estimate
    """
    # Initialize parameters number matrix (2x1)
    parameters_number = [[0], [0]]  # Mimics R's matrix(0,2,1)
    
    # Count states (first row)
    if ets_info["ets_model"]:
        # Add level component
        parameters_number[0][0] += 1
        # Add trend if present
        if ets_info["trend_type"] != "N":
            parameters_number[0][0] += 1
        # Add seasonal if present
        if ets_info["season_type"] != "N":
            parameters_number[0][0] += 1
    
    # Count parameters to estimate (second row)
    if ets_info["ets_model"]:
        # Level persistence
        parameters_number[1][0] += 1
        # Trend persistence if present
        if ets_info["trend_type"] != "N":
            parameters_number[1][0] += 1
            # Additional parameter for damped trend
            if ets_info["damped"]:
                parameters_number[1][0] += 1
        # Seasonal persistence if present
        if ets_info["season_type"] != "N":
            parameters_number[1][0] += 1
    
    # Add ARIMA parameters if present
    if arima_info["arima_model"]:
        # Add number of ARMA parameters
        parameters_number[1][0] += len(arima_info.get("arma_parameters", []))
    
    # Add constant if required
    if constant_required:
        parameters_number[1][0] += 1
    
    # Handle pure constant model case (no ETS, no ARIMA, no xreg)
    if not ets_info["ets_model"] and not arima_info["arima_model"] and not xreg_info:
        parameters_number[0][0] = 0
        parameters_number[1][0] = 2  # Matches R code line 3047
    
    return parameters_number
    #return {
    #    "parameters_number": parameters_number,
    #    "n_states": parameters_number[0][0],
    #    "n_params": parameters_number[1][0]
    #}


def _calculate_n_param_max(
    ets_model,
    persistence_level_estimate,
    model_is_trendy,
    persistence_trend_estimate,
    model_is_seasonal,
    persistence_seasonal_estimate,
    phi_estimate,
    initial_type,
    initial_level_estimate,
    initial_trend_estimate,
    initial_seasonal_estimate,
    lags_model_seasonal,
    arima_model=False,
    initial_arima_number=0,
    ar_required=False,
    ar_estimate=False,
    ar_orders=None,
    ma_required=False,
    ma_estimate=False,
    ma_orders=None,
    xreg_model=False,
    xreg_number=0,
    initial_xreg_estimate=False,
    persistence_xreg_estimate=False,
):
    """
    Calculate maximum number of parameters for the model.

    Follows R's adamGeneral.R lines 2641-2651.

    Parameters
    ----------
    ets_model : bool
        Whether ETS model is used
    persistence_level_estimate : bool
        Whether level persistence is estimated
    model_is_trendy : bool
        Whether model has trend component
    persistence_trend_estimate : bool
        Whether trend persistence is estimated
    model_is_seasonal : bool
        Whether model has seasonal component
    persistence_seasonal_estimate : list or bool
        Whether seasonal persistence(s) are estimated
    phi_estimate : bool
        Whether damping parameter is estimated
    initial_type : str
        Initial type ('optimal', 'two-stage', 'backcasting', 'provided')
    initial_level_estimate : bool
        Whether initial level is estimated
    initial_trend_estimate : bool
        Whether initial trend is estimated
    initial_seasonal_estimate : list or bool
        Whether initial seasonal(s) are estimated
    lags_model_seasonal : list
        List of seasonal lags
    arima_model : bool
        Whether ARIMA model is used
    initial_arima_number : int
        Number of ARIMA initial states
    ar_required : bool
        Whether AR is required
    ar_estimate : bool
        Whether AR is estimated
    ar_orders : list
        List of AR orders
    ma_required : bool
        Whether MA is required
    ma_estimate : bool
        Whether MA is estimated
    ma_orders : list
        List of MA orders
    xreg_model : bool
        Whether xreg model is used
    xreg_number : int
        Number of external regressors
    initial_xreg_estimate : bool
        Whether xreg initials are estimated
    persistence_xreg_estimate : bool
        Whether xreg persistence is estimated

    Returns
    -------
    int
        Maximum number of parameters
    """
    # Handle list/bool for persistence_seasonal_estimate
    if isinstance(persistence_seasonal_estimate, (list, np.ndarray)):
        sum_persistence_seasonal = sum(persistence_seasonal_estimate)
    else:
        sum_persistence_seasonal = int(persistence_seasonal_estimate) if persistence_seasonal_estimate else 0

    # Handle list/bool for initial_seasonal_estimate - multiply by lags as in R
    # R: sum(initialSeasonalEstimate*lagsModelSeasonal)
    if lags_model_seasonal and initial_seasonal_estimate:
        if isinstance(initial_seasonal_estimate, (list, np.ndarray)):
            # Element-wise multiplication of estimates and lags, then sum
            init_seasonal_arr = np.array(initial_seasonal_estimate, dtype=int)
            lags_arr = np.array(lags_model_seasonal)
            # Broadcast if lengths differ (repeat estimates if shorter)
            if len(init_seasonal_arr) < len(lags_arr):
                init_seasonal_arr = np.tile(init_seasonal_arr, len(lags_arr))[:len(lags_arr)]
            elif len(init_seasonal_arr) > len(lags_arr):
                init_seasonal_arr = init_seasonal_arr[:len(lags_arr)]
            sum_initial_seasonal_lags = int(np.sum(init_seasonal_arr * lags_arr))
        else:
            # Single boolean - multiply by sum of lags
            sum_initial_seasonal_lags = int(initial_seasonal_estimate) * sum(lags_model_seasonal)
    else:
        sum_initial_seasonal_lags = 0

    # Check if initial type requires estimation
    initial_needs_estimation = initial_type in ["optimal", "two-stage"]
    initial_needs_xreg = initial_type in ["backcasting", "optimal", "two-stage"]

    # ETS component
    ets_params = 0
    if ets_model:
        ets_params = (
            int(persistence_level_estimate) +
            int(model_is_trendy) * int(persistence_trend_estimate) +
            int(model_is_seasonal) * sum_persistence_seasonal +
            int(phi_estimate) +
            int(initial_needs_estimation) * (
                int(initial_level_estimate) +
                int(initial_trend_estimate) +
                sum_initial_seasonal_lags
            )
        )

    # ARIMA component
    arima_params = 0
    if arima_model:
        ar_orders_sum = sum(ar_orders) if ar_orders else 0
        ma_orders_sum = sum(ma_orders) if ma_orders else 0
        arima_params = (
            int(initial_needs_estimation) * initial_arima_number +
            int(ar_required) * int(ar_estimate) * ar_orders_sum +
            int(ma_required) * int(ma_estimate) * ma_orders_sum
        )

    # Xreg component
    xreg_params = 0
    if xreg_model:
        xreg_params = xreg_number * (
            int(initial_needs_xreg) * int(initial_xreg_estimate) +
            int(persistence_xreg_estimate)
        )

    # Total: 1 (scale) + ETS + ARIMA + xreg
    return 1 + ets_params + arima_params + xreg_params


def _restrict_models_pool_for_sample_size(
    obs_nonzero,
    lags_model_max,
    model_do,
    error_type,
    trend_type,
    season_type,
    models_pool,
    allow_multiplicative=True,
    xreg_number=0,
    silent=False,
    n_param_max=None,
    damped=False,
):
    """
    Restrict the models pool based on sample size.

    Follows the logic from R's adamGeneral.R lines 2641-2944.
    Only applies restrictions when obs_nonzero <= n_param_max.

    Parameters
    ----------
    obs_nonzero : int
        Number of non-zero observations
    lags_model_max : int
        Maximum lag value
    model_do : str
        Model action: 'estimate', 'select', 'combine', or 'use'
    error_type : str
        Error type character
    trend_type : str
        Trend type character
    season_type : str
        Season type character
    models_pool : list or None
        Current models pool
    allow_multiplicative : bool
        Whether multiplicative models are allowed
    xreg_number : int
        Number of external regressors
    silent : bool
        Whether to suppress warnings
    n_param_max : int or None
        Maximum number of parameters. If None, restrictions always apply.
        If provided, restrictions only apply when obs_nonzero <= n_param_max.
    damped : bool
        Whether the model has damped trend

    Returns
    -------
    dict
        Updated model configuration with restricted pool
    """
    n_param_exo = xreg_number * 2  # xreg coefficients + persistence

    result = {
        "model_do": model_do,
        "error_type": error_type,
        "trend_type": trend_type,
        "season_type": season_type,
        "models_pool": models_pool,
        "persistence_level": None,
        "persistence_estimate": True,
        "initial_type": None,
        "initial_estimate": True,
        "phi_estimate": True,
        "damped": damped,
    }

    # Only apply restrictions if n_param_max is None or obs_nonzero <= n_param_max
    # Following R line 2655: if(obsNonzero <= nParamMax)
    if n_param_max is not None and obs_nonzero > n_param_max:
        return result

    # Print message when restrictions are being applied (R lines 2657-2661)
    if n_param_max is not None and not silent:
        print(f"Number of non-zero observations is {obs_nonzero}, "
              f"while the maximum number of parameters to estimate is {n_param_max}.\n"
              "Updating pool of models.")

    # If pool not specified and select/combine mode, build restricted pool
    if obs_nonzero > (3 + n_param_exo) and models_pool is None and model_do in ["select", "combine"]:
        new_pool = ["ANN"]
        if allow_multiplicative:
            new_pool.append("MNN")

        # Enough for trend model
        if obs_nonzero > (5 + n_param_exo):
            if trend_type in ["Z", "X", "A"]:
                new_pool.append("AAN")
            if allow_multiplicative and trend_type in ["Z", "Y", "M"]:
                new_pool.extend(["AMN", "MAN", "MMN"])

        # Enough for damped trend model
        if obs_nonzero > (6 + n_param_exo):
            if trend_type in ["Z", "X", "A"]:
                new_pool.append("AAdN")
            if allow_multiplicative and trend_type in ["Z", "Y", "M"]:
                new_pool.extend(["AMdN", "MAdN", "MMdN"])

        # Enough for seasonal model
        if obs_nonzero > lags_model_max and lags_model_max != 1:
            if season_type in ["Z", "X", "A"]:
                new_pool.append("ANA")
            if allow_multiplicative and season_type in ["Z", "Y", "M"]:
                new_pool.extend(["ANM", "MNA", "MNM"])

        # Enough for seasonal model with trend
        if (obs_nonzero > (6 + lags_model_max + n_param_exo) and
            obs_nonzero > 2 * lags_model_max and lags_model_max != 1):
            if trend_type in ["Z", "X", "A"] and season_type in ["Z", "X", "A"]:
                new_pool.append("AAA")
            if allow_multiplicative:
                if trend_type in ["Z", "X", "A"] and season_type in ["Z", "Y", "A"]:
                    new_pool.append("MAA")
                if trend_type in ["Z", "X", "A"] and season_type in ["Z", "Y", "M"]:
                    new_pool.extend(["AAM", "MAM"])
                if trend_type in ["Z", "Y", "M"] and season_type in ["Z", "X", "A"]:
                    new_pool.extend(["AMA", "MMA"])
                if trend_type in ["Z", "Y", "M"] and season_type in ["Z", "Y", "M"]:
                    new_pool.extend(["AMM", "MMM"])

        result["models_pool"] = new_pool
        if model_do == "combine":
            result["model_do"] = "combine"
        else:
            result["model_do"] = "select"

        if not silent:
            _warn(f"Not enough observations for full model pool. Fitting restricted pool: {new_pool}")

    # If pool is provided, filter it based on available observations
    elif obs_nonzero > (3 + n_param_exo) and models_pool is not None:
        filtered_pool = list(models_pool)

        # Remove damped seasonal models if not enough obs
        if obs_nonzero <= (6 + lags_model_max + 1 + n_param_exo):
            filtered_pool = [m for m in filtered_pool if not (len(m) == 4 and m[-1] in ["A", "M"])]

        # Remove seasonal + trend models if not enough obs
        if obs_nonzero <= (5 + lags_model_max + 1 + n_param_exo):
            filtered_pool = [m for m in filtered_pool if not (m[1] != "N" and m[-1] != "N")]

        # Remove seasonal models if not enough obs
        if obs_nonzero <= lags_model_max:
            filtered_pool = [m for m in filtered_pool if m[-1] == "N"]

        # Remove damped trend if not enough obs
        if obs_nonzero <= (6 + n_param_exo):
            filtered_pool = [m for m in filtered_pool if len(m) != 4]

        # Remove any trend if not enough obs
        if obs_nonzero <= (5 + n_param_exo):
            filtered_pool = [m for m in filtered_pool if m[1] == "N"]

        result["models_pool"] = list(set(filtered_pool))
        if not silent and len(filtered_pool) < len(models_pool):
            _warn(f"Pool restricted due to sample size: {filtered_pool}")

    # Handle estimate/use mode (R lines 2778-2832)
    elif obs_nonzero > (3 + n_param_exo) and model_do in ["estimate", "use"]:
        # Build model string: E + T + (d if damped) + S
        model = error_type + trend_type
        if damped:
            model += "d"
        model += season_type
        original_model = model

        # 1. Remove damped from seasonal models if not enough obs (R lines 2780-2790)
        if obs_nonzero <= (6 + lags_model_max + 1 + n_param_exo):
            if len(model) == 4:  # Damped model with seasonal
                if not silent:
                    _warn(f"Not enough non-zero observations for ETS({model})! Fitting what I can...")
                model = model[:2] + model[3]  # Remove 'd': AAdA -> AAA

        # 2. Remove trend from seasonal models if not enough obs (R lines 2791-2798)
        if obs_nonzero <= (5 + lags_model_max + 1 + n_param_exo):
            if model[1] != "N":  # Has trend
                if not silent:
                    _warn(f"Not enough non-zero observations for ETS({model})! Fitting what I can...")
                model = model[0] + "N" + model[2]  # Remove trend: AAA -> ANA

        # 3. Remove seasonal if not enough obs (R lines 2799-2805)
        if obs_nonzero <= lags_model_max:
            if model[-1] != "N":  # Has seasonal
                if not silent:
                    _warn(f"Not enough non-zero observations for ETS({model})! Fitting what I can...")
                model = model[:2] + "N"  # Remove seasonal: ANA -> ANN

        # 4. Remove damped from non-seasonal models if not enough obs (R lines 2806-2814)
        if obs_nonzero <= (6 + n_param_exo):
            if len(model) == 4:  # Damped model (non-seasonal at this point)
                if not silent:
                    _warn(f"Not enough non-zero observations for ETS({model})! Fitting what I can...")
                model = model[:2] + model[3]  # Remove 'd': AAdN -> AAN

        # 5. Remove any trend if not enough obs (R lines 2815-2821)
        if obs_nonzero <= (5 + n_param_exo):
            if model[1] != "N":  # Has trend
                if not silent:
                    _warn(f"Not enough non-zero observations for ETS({model})! Fitting what I can...")
                model = model[0] + "N" + model[2]  # Remove trend: AAN -> ANN

        # Update result based on simplified model (R lines 2822-2832)
        result["error_type"] = model[0]
        result["trend_type"] = model[1]
        result["season_type"] = model[-1]  # Last character is always season

        # Update damped and phi_estimate based on final model
        model_is_trendy = model[1] != "N"
        result["damped"] = model_is_trendy and len(model) == 4
        result["phi_estimate"] = result["damped"]

    # Extreme cases
    if obs_nonzero == 4:
        if error_type in ["A", "M"]:
            result["model_do"] = "estimate"
            result["trend_type"] = "N"
            result["season_type"] = "N"
        else:
            result["models_pool"] = ["ANN", "MNN"] if allow_multiplicative else ["ANN"]
            result["model_do"] = "select"
            result["error_type"] = "Z"
            result["trend_type"] = "N"
            result["season_type"] = "N"
            if not silent:
                _warn("Very small sample. Only level models available.")
        result["phi_estimate"] = False
        result["damped"] = False

    elif obs_nonzero == 3:
        if error_type in ["A", "M"]:
            result["model_do"] = "estimate"
            result["trend_type"] = "N"
            result["season_type"] = "N"
        else:
            result["models_pool"] = ["ANN", "MNN"] if allow_multiplicative else ["ANN"]
            result["model_do"] = "select"
            result["error_type"] = "Z"
            result["trend_type"] = "N"
            result["season_type"] = "N"
        result["persistence_level"] = 0
        result["persistence_estimate"] = False
        result["phi_estimate"] = False
        result["damped"] = False
        if not silent:
            _warn("Very small sample. Persistence set to zero.")

    elif obs_nonzero == 2:
        result["models_pool"] = None
        result["persistence_level"] = 0
        result["persistence_estimate"] = False
        result["initial_type"] = "provided"
        result["initial_estimate"] = False
        result["model_do"] = "use"
        result["error_type"] = "A"
        result["trend_type"] = "N"
        result["season_type"] = "N"
        result["phi_estimate"] = False
        result["damped"] = False
        if not silent:
            _warn("Sample too small. Using fixed ANN model.")

    elif obs_nonzero == 1:
        result["models_pool"] = None
        result["persistence_level"] = 0
        result["persistence_estimate"] = False
        result["initial_type"] = "provided"
        result["initial_estimate"] = False
        result["model_do"] = "use"
        result["error_type"] = "A"
        result["trend_type"] = "N"
        result["season_type"] = "N"
        result["phi_estimate"] = False
        result["damped"] = False
        if not silent:
            _warn("Only one observation. Using Naive forecast.")

    elif obs_nonzero == 0:
        result["models_pool"] = None
        result["persistence_level"] = 0
        result["persistence_estimate"] = False
        result["initial_type"] = "provided"
        result["initial_estimate"] = False
        result["model_do"] = "use"
        result["error_type"] = "A"
        result["trend_type"] = "N"
        result["season_type"] = "N"
        result["phi_estimate"] = False
        result["damped"] = False
        if not silent:
            _warn("No non-zero observations. Forecast will be zero.")

    elif obs_nonzero <= 3 + n_param_exo:
        raise ValueError(f"Not enough observations ({obs_nonzero}) for model estimation.")

    return result


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
    n_iterations=None,
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
    Validate and process all ADAM model parameters before estimation.

    This is the central parameter validation function that checks all user inputs for
    consistency, converts them to standardized internal formats, and sets up the complete
    model specification. It acts as a gatekeeper before model estimation, ensuring that:

    - Model specifications are valid (ETS components, ARIMA orders)
    - Data properties are appropriate (sufficient observations, valid lags)
    - Parameter specifications are consistent (bounds, distributions, loss functions)
    - Initial values and persistence parameters are properly formatted
    - Information criteria and occurrence models are correctly configured

    The function performs comprehensive validation similar to R's adam() parameter checking,
    transforming user-friendly inputs into the detailed dictionaries required by the
    estimation engine.

    **Validation Process**:

    1. **Occurrence Checking**: Validate intermittent demand settings
    2. **Lags Validation**: Ensure lags are compatible with data length
    3. **ETS Model Parsing**: Decode model string (e.g., "AAA", "ZXZ") into components
    4. **ARIMA Validation**: Check orders and stationarity requirements
    5. **Distribution & Loss**: Verify compatibility (e.g., multiplicative error requires positive data)
    6. **Outliers**: Configure outlier detection if requested
    7. **Damping (φ)**: Validate damping parameter for damped trend models
    8. **Persistence**: Process smoothing parameters (α, β, γ) - fixed or to be estimated
    9. **Initial States**: Configure initialization method (optimal, backcasting, provided)
    10. **Constants**: Set up intercept term if required
    11. **Model Pool**: Generate model pool for automatic selection ("ZZZ", "XXX", etc.)
    12. **Profiles**: Initialize time-varying parameter structures
    13. **Observations**: Format data and compute necessary statistics
    14. **Assembly**: Package all validated parameters into organized dictionaries

    Parameters
    ----------
    data : array-like, pandas.Series, or pandas.DataFrame
        Time series data for model estimation. Must be numeric and one-dimensional.
        Can handle intermittent demand (data with zeros).

        - **Length requirement**: Must have sufficient observations for the model
          (at least max(lags) + max(orders) observations)
        - **Missing values**: Handled by converting to numeric with ``pd.to_numeric``

    model : str or list of str
        ETS model specification or list of models for selection.

        Model string format: ``E + T + S`` where:

        - **E** (Error): "A" (Additive) or "M" (Multiplicative)
        - **T** (Trend): "N" (None), "A" (Additive), "Ad" (Additive damped),
          "M" (Multiplicative), "Md" (Multiplicative damped)
        - **S** (Seasonality): "N" (None), "A" (Additive), "M" (Multiplicative)

        Special codes for automatic selection:

        - **"Z"**: Select from all options (Branch & Bound algorithm)
        - **"X"**: Select only additive components
        - **"Y"**: Select only multiplicative components
        - **"C"**: Combine forecasts using IC weights
        - **"P"**: Select between pure additive and pure multiplicative
        - **"F"**: Full search across all 30 possible models
        - **"S"**: Sensible 19-model pool with finite variance

        Examples: "ANN" (Simple exponential smoothing), "AAA" (Holt-Winters additive),
        "MAM" (Multiplicative error with additive trend and multiplicative seasonality),
        "ZXZ" (Auto-select error and seasonality, only additive trend)

    lags : numpy.ndarray or list
        Seasonal lags vector. For multiple seasonality, provide multiple lags.

        - Non-seasonal model: ``lags=[1]``
        - Monthly with annual seasonality: ``lags=[1, 12]``
        - Hourly with daily and weekly seasonality: ``lags=[1, 24, 168]``

        **Important**: First lag is typically 1 for level/trend. Subsequent lags define
        seasonal patterns. Length must not exceed number of observations.

    orders : dict, list, tuple, or None, default=None
        ARIMA component specification. If None, pure ETS model is estimated.

        **Format options**:

        1. **Dict format** (recommended for clarity)::

            orders = {
                'ar': [p, P],    # AR orders: non-seasonal p, seasonal P
                'i': [d, D],     # Integration: non-seasonal d, seasonal D
                'ma': [q, Q],    # MA orders: non-seasonal q, seasonal Q
                'select': False  # Whether to auto-select orders
            }

        2. **List/tuple format**: ``[p, d, q]`` for non-seasonal ARIMA(p,d,q)

        If ``'select': True``, automatic order selection is performed (similar to auto.arima).

        Examples:

        - ``orders={'ar': [1, 0], 'i': [1, 0], 'ma': [1, 0]}``: ARIMA(1,1,1)
        - ``orders={'ar': [0, 1], 'i': [0, 1], 'ma': [0, 1]}``: Seasonal ARIMA(0,0,0)(1,1,1)
        - ``orders=[1, 1, 1]``: Non-seasonal ARIMA(1,1,1)

    constant : bool or float, default=False
        Whether to include a constant (intercept) term in the model.

        - ``False``: No constant
        - ``True``: Estimate constant
        - ``float``: Fixed constant value (not estimated)

        The constant is particularly useful for:

        - Models without trend when data has non-zero mean
        - ARIMA models with drift

    outliers : str, default="ignore"
        Outlier detection and handling method.

        - **"ignore"**: No outlier handling (default)
        - **"detect"**: Detect outliers using tsoutliers package
        - **"use"**: Use provided outlier indicators

        *Note: Outlier handling is not fully implemented in Python version yet.*

    level : float, default=0.99
        Confidence level for outlier detection (if outliers != "ignore").
        Typical values: 0.95 (5% significance), 0.99 (1% significance).

    persistence : dict, list, float, or None, default=None
        Smoothing parameters specification (α, β, γ).

        **Format options**:

        1. **None** (default): All smoothing parameters are estimated
        2. **Dict format** for granular control::

            persistence = {
                'alpha': 0.3,     # Level smoothing (or None to estimate)
                'beta': 0.1,      # Trend smoothing (or None to estimate)
                'gamma': 0.05     # Seasonal smoothing (or None to estimate)
            }

        3. **List format**: ``[α, β, γ]`` with None for parameters to estimate
        4. **Float**: Single value used for all estimated smoothing parameters (starting value)

        **Constraints**: During estimation, smoothing parameters are constrained to [0,1]
        with additional restrictions: β ≤ α, γ ≤ 1-α (usual bounds).

    phi : float or None, default=None
        Damping parameter for damped trend models (Ad or Md).

        - **None**: Estimate φ (if model has damped trend)
        - **Float in (0,1]**: Fixed damping value
        - **1.0**: No damping (equivalent to non-damped trend)

        Lower values (e.g., 0.8-0.95) produce more conservative long-term forecasts
        by damping the trend contribution over time.

    initial : str, dict, list, or None, default=None
        Initial state values specification.

        **Initialization methods**:

        - **"optimal"**: Optimize initial states along with other parameters (default)
        - **"backcasting"**: Use backcasting with 2 iterations and head refinement
        - **"complete"**: Full backcasting without subsequent optimization
        - **"two-stage"**: First backcast, then optimize using backcasted values as starting point

        **Fixed initial values**::

            initial = {
                'level': 100,                    # Initial level
                'trend': 5,                      # Initial trend (if trendy)
                'seasonal': [0.9, 1.0, 1.1, ...] # Initial seasonal indices (if seasonal)
            }

        **Hybrid approach**: Dict with some values specified and others set to None for estimation.

    n_iterations : int or None, default=None
        Number of backcasting iterations when initial="backcasting" or "complete".

        - **None**: Use default (2 for backcasting)
        - **int**: Custom iteration count (typically 2-5)

        More iterations improve initial state estimates but increase computation time.

    distribution : str, default="default"
        Error term probability distribution.

        Supported distributions:

        - **"default"**: Automatic selection based on error type and loss

          * Additive error → Normal (dnorm)
          * Multiplicative error → Gamma (dgamma)

        - **"dnorm"**: Normal distribution (Gaussian)
        - **"dlaplace"**: Laplace distribution (for MAE loss)
        - **"ds"**: S distribution (for HAM loss)
        - **"dgnorm"**: Generalized Normal distribution
        - **"dlnorm"**: Log-Normal distribution
        - **"dgamma"**: Gamma distribution (for multiplicative errors)
        - **"dinvgauss"**: Inverse Gaussian distribution

        The distribution affects likelihood calculation and prediction intervals.

    loss : str, default="likelihood"
        Loss function for parameter optimization.

        **One-step losses**:

        - **"likelihood"**: Maximum likelihood estimation (default)
        - **"MSE"**: Mean Squared Error
        - **"MAE"**: Mean Absolute Error
        - **"HAM"**: Half Absolute Moment (geometric mean of absolute errors)
        - **"LASSO"**: L1-regularized loss (for variable selection)
        - **"RIDGE"**: L2-regularized loss (for shrinkage)

        **Multi-step losses** (h-step ahead):

        - **"MSEh"**: h-step ahead MSE
        - **"MAEh"**: h-step ahead MAE
        - **"HAMh"**: h-step ahead HAM

        For LASSO/RIDGE, set ``lambda_param`` to control regularization strength.

    h : int, default=0
        Forecast horizon (number of steps ahead to forecast).

        - Used for holdout validation if ``holdout=True``
        - Required for multi-step losses (MSEh, MAEh, HAMh)
        - Sets prediction interval horizon

    holdout : bool, default=False
        Whether to split data into training and holdout samples.

        - ``False``: Use all data for estimation
        - ``True``: Last ``h`` observations become holdout sample for validation

        Useful for out-of-sample accuracy assessment.

    occurrence : str, default="none"
        Occurrence model for intermittent demand (data with zeros).

        - **"none"**: No occurrence model (continuous demand)
        - **"auto"**: Automatically select occurrence model
        - **"fixed"**: Fixed probability
        - **"general"**: General occurrence model
        - **"odds-ratio"**: Odds-ratio based model
        - **"inverse-odds-ratio"**: Inverse odds-ratio model
        - **"direct"**: Direct probability model
        - **"provided"**: User-provided occurrence indicators

        Occurrence models are essential for intermittent demand forecasting (e.g., spare parts).

    ic : str, default="AICc"
        Information criterion for model selection.

        - **"AIC"**: Akaike Information Criterion
        - **"AICc"**: Corrected AIC (recommended for small samples)
        - **"BIC"**: Bayesian Information Criterion (more parsimonious)
        - **"BICc"**: Corrected BIC

        Lower IC values indicate better models. AICc is default as it performs well
        across sample sizes.

    bounds : str, default="usual"
        Parameter constraint type during optimization.

        - **"usual"**: Classical restrictions (α,β,γ ∈ [0,1], β ≤ α, γ ≤ 1-α, φ ∈ [0,1])
        - **"admissible"**: Stability constraints based on eigenvalues of transition matrix
        - **"none"**: No constraints (not recommended)

        "usual" bounds are recommended for most applications. "admissible" allows more
        flexibility but may produce unstable forecasts.

    silent : bool, default=False
        Whether to suppress warning messages.

        - ``False``: Display warnings about model specification issues
        - ``True``: Silent mode (no warnings)

    model_do : str, default="estimate"
        Action to perform with the model.

        - **"estimate"**: Estimate specified model
        - **"select"**: Automatic model selection from pool
        - **"combine"**: Combine forecasts from multiple models (*not implemented yet*)

    fast : bool, default=False
        Whether to use faster (but possibly less accurate) estimation.

        - ``False``: Standard estimation
        - ``True``: Reduced accuracy for speed (fewer iterations, looser tolerances)

    models_pool : list of str or None, default=None
        Custom pool of models for selection (when model_do="select").

        Example: ``models_pool=["ANN", "AAN", "AAdN", "AAA"]``

        If None, pool is generated automatically based on model specification
        (e.g., "ZXZ" generates appropriate pool).

    lambda_param : float or None, default=None
        Regularization parameter for LASSO/RIDGE losses.

        - **0**: No regularization (pure MSE)
        - **1**: Full regularization (parameters shrunk to zero/heavily penalized)
        - **(0,1)**: Partial regularization

        Typical values: 0.01-0.1 for moderate regularization.

    frequency : str or None, default=None
        Time series frequency for date/time indexing.

        Pandas frequency strings: "D" (daily), "W" (weekly), "M" (monthly),
        "Q" (quarterly), "Y" (yearly), "H" (hourly), etc.

        If None, inferred from data if it has DatetimeIndex.

    interval : str, default="parametric"
        Prediction interval calculation method.

        - **"parametric"**: Analytical intervals based on assumed distribution
        - **"simulation"**: Simulation-based intervals
        - **"bootstrap"**: Bootstrap intervals

    interval_level : list of float, default=[0.95]
        Confidence level(s) for prediction intervals.

        Examples: ``[0.80, 0.95]`` for 80% and 95% intervals.

    side : str, default="both"
        Which prediction interval bounds to compute.

        - **"both"**: Lower and upper bounds
        - **"lower"**: Lower bound only
        - **"upper"**: Upper bound only

    cumulative : bool, default=False
        Whether to compute cumulative forecasts (sum over horizon).
        Useful for total demand forecasting.

    nsim : int, default=1000
        Number of simulations for simulation-based prediction intervals.

    scenarios : int, default=100
        Number of scenarios for scenario-based forecasting.

    ellipsis : dict or None, default=None
        Additional parameters passed through (for extensibility).

    Returns
    -------
    tuple of 13 dict
        Tuple containing validated and organized parameters:

        1. **general_dict** : General configuration (loss, distribution, bounds, ic, h, holdout)
        2. **observations_dict** : Data and observation-related information
        3. **persistence_results** : Validated persistence parameters
        4. **initials_results** : Validated initial state specifications
        5. **arima_results** : ARIMA component specifications
        6. **constant_dict** : Constant term configuration
        7. **model_type_dict** : Model type information (ETS, ARIMA, components)
        8. **components_dict** : Component counts and structure
        9. **lags_dict** : Lag structure and related information
        10. **occurrence_dict** : Occurrence model configuration
        11. **phi_dict** : Damping parameter specification
        12. **explanatory_dict** : External regressors configuration (not fully implemented)
        13. **params_info** : Parameter count information

    Raises
    ------
    ValueError
        If parameters are invalid or inconsistent:

        - Data is non-numeric or empty
        - Lags exceed data length
        - Model string is malformed
        - ARIMA orders are negative
        - Incompatible distribution/loss combination
        - Insufficient data for model complexity

    Notes
    -----
    **Parameter Checking Philosophy**:

    This function aims to fail early with clear error messages rather than allowing
    invalid configurations to proceed to estimation. It provides helpful warnings when
    suboptimal choices are detected (e.g., multiplicative seasonality with negative data).

    **Relationship to R Implementation**:

    This function consolidates checks that are distributed across multiple functions in
    the R package (adam, adamSelection, etc.). The Python version performs equivalent
    validation but returns more structured outputs (dictionaries) rather than R's
    list objects.

    **Performance**:

    Parameter checking is fast (< 1ms typically). The main computational cost is in
    model estimation, not validation.

    See Also
    --------
    estimator : Main estimation function that uses validated parameters
    selector : Model selection function
    ADAM : User-facing class that wraps parameter_checker and estimator

    Examples
    --------
    Basic validation for simple exponential smoothing::

        >>> data = np.array([10, 12, 15, 13, 16, 18, 20, 19, 22, 25])
        >>> results = parameters_checker(
        ...     data=data,
        ...     model="ANN",
        ...     lags=[1],
        ...     silent=True
        ... )
        >>> general, obs, persist, initials, arima, const, model_type, *rest = results
        >>> print(model_type['model'])
        'ANN'

    Validation with automatic model selection::

        >>> results = parameters_checker(
        ...     data=data,
        ...     model="ZXZ",  # Auto-select error and seasonality, only additive trend
        ...     lags=[1, 12],
        ...     model_do="select",
        ...     ic="AICc",
        ...     silent=True
        ... )

    ARIMA component with fixed smoothing::

        >>> results = parameters_checker(
        ...     data=data,
        ...     model="AAN",
        ...     lags=[1],
        ...     orders={'ar': [1, 0], 'i': [0, 0], 'ma': [0, 0]},
        ...     persistence={'alpha': 0.3, 'beta': 0.1},
        ...     silent=True
        ... )
    """
    #####################
    # 1) Check Occurrence
    #####################
     # Extract values if DataFrame/Series and ensure numeric
    if hasattr(data, 'values'):
        data_values = data.values
        if isinstance(data_values, np.ndarray):
            data_values = data_values.flatten()
        # Convert to numeric if needed
        data_values = pd.to_numeric(data_values, errors='coerce')
    else:
        # Convert to numeric if needed
        try:
            data_values = pd.to_numeric(data, errors='coerce')
        except:
            raise ValueError("Data must be numeric or convertible to numeric values")


    occ_info = _check_occurrence(data_values, occurrence, frequency, silent, holdout, h)
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

    # Process n_iterations parameter (for backcasting)
    # Default behavior matches R: 2 iterations for backcasting/complete, 1 otherwise
    if n_iterations is None:
        if init_info["initial_type"] in ["backcasting", "complete"]:
            n_iterations = 2
        else:
            n_iterations = 1
    else:
        # Validate user-provided n_iterations
        if not isinstance(n_iterations, int) or n_iterations < 1:
            _warn(f"n_iterations must be a positive integer. Using default value.", silent)
            if init_info["initial_type"] in ["backcasting", "complete"]:
                n_iterations = 2
            else:
                n_iterations = 1

    # Add to init_info
    init_info["n_iterations"] = n_iterations

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

    # Calculate n_param_max to determine if pool restriction is needed (R lines 2641-2651)
    model_is_trendy = ets_info["trend_type"] not in ["N", None]
    model_is_seasonal = ets_info["season_type"] not in ["N", None] and len(lags_model_seasonal) > 0

    n_param_max = _calculate_n_param_max(
        ets_model=ets_model,
        persistence_level_estimate=persist_info.get("persistence_level_estimate", True),
        model_is_trendy=model_is_trendy,
        persistence_trend_estimate=persist_info.get("persistence_trend_estimate", True),
        model_is_seasonal=model_is_seasonal,
        persistence_seasonal_estimate=persist_info.get("persistence_seasonal_estimate", [True] * len(lags_model_seasonal)),
        phi_estimate=phi_estimate,
        initial_type=init_info.get("initial_type", "optimal"),
        initial_level_estimate=init_info.get("initial_level_estimate", True),
        initial_trend_estimate=init_info.get("initial_trend_estimate", True),
        initial_seasonal_estimate=init_info.get("initial_seasonal_estimate", [True] * len(lags_model_seasonal)),
        lags_model_seasonal=lags_model_seasonal,
        arima_model=arima_model,
        initial_arima_number=arima_info.get("initial_arima_number", 0),
        ar_required=any(ar_orders) if ar_orders else False,
        ar_estimate=arima_info.get("ar_estimate", True),
        ar_orders=ar_orders,
        ma_required=any(ma_orders) if ma_orders else False,
        ma_estimate=arima_info.get("ma_estimate", True),
        ma_orders=ma_orders,
        xreg_model=False,  # Will be updated when xreg is implemented
        xreg_number=0,
        initial_xreg_estimate=False,
        persistence_xreg_estimate=False,
    )

    # Restrict models pool based on sample size (R lines 2641-2944)
    # Only apply if obs_nonzero <= n_param_max (R line 2655)
    pool_restriction = _restrict_models_pool_for_sample_size(
        obs_nonzero=obs_nonzero,
        lags_model_max=max_lag,
        model_do=model_do,
        error_type=ets_info["error_type"],
        trend_type=ets_info["trend_type"],
        season_type=ets_info["season_type"],
        models_pool=models_pool if models_pool is not None else ets_info.get("models_pool"),
        allow_multiplicative=allow_multiplicative,
        xreg_number=0,  # Will be updated when xreg is implemented
        silent=silent,
        n_param_max=n_param_max,
        damped=ets_info.get("damped", False),
    )

    # Update ets_info with restricted values
    model_changed = False
    if pool_restriction["error_type"] != ets_info["error_type"]:
        ets_info["error_type"] = pool_restriction["error_type"]
        model_changed = True
    if pool_restriction["trend_type"] != ets_info["trend_type"]:
        ets_info["trend_type"] = pool_restriction["trend_type"]
        model_changed = True
    if pool_restriction["season_type"] != ets_info["season_type"]:
        ets_info["season_type"] = pool_restriction["season_type"]
        model_changed = True
    if pool_restriction["models_pool"] is not None:
        ets_info["models_pool"] = pool_restriction["models_pool"]
    if pool_restriction["model_do"] != model_do:
        model_do = pool_restriction["model_do"]
        ets_info["model_do"] = model_do
    # Update damped flag if it was changed
    if pool_restriction["damped"] != ets_info.get("damped", False):
        ets_info["damped"] = pool_restriction["damped"]
        model_changed = True

    # Rebuild model string if any component changed
    if model_changed:
        new_model = ets_info["error_type"] + ets_info["trend_type"]
        if ets_info["damped"] and ets_info["trend_type"] != "N":
            new_model += "d"
        new_model += ets_info["season_type"]
        ets_info["model"] = new_model

    # Update persistence if restricted
    if pool_restriction["persistence_level"] is not None:
        persist_info["persistence"] = pool_restriction["persistence_level"]
    if not pool_restriction["persistence_estimate"]:
        persist_info["persistence_estimate"] = False
        persist_info["persistence_level_estimate"] = False

    # Update phi_estimate if restricted
    if not pool_restriction["phi_estimate"]:
        phi_estimate = False

    # Update initial if restricted
    if pool_restriction["initial_type"] is not None:
        initial = pool_restriction["initial_type"]
    if not pool_restriction["initial_estimate"]:
        init_info["initial_estimate"] = False
        init_info["initial_level_estimate"] = False

    # Setup model type dictionary
    model_type_dict = _organize_model_type_info(ets_info, arima_info, xreg_model=False)

    # Apply additional sample size adjustments
    model_type_dict = _adjust_model_for_sample_size(
        model_info=model_type_dict,
        obs_nonzero=obs_nonzero,
        lags_model_max=max_lag,
        allow_multiplicative=allow_multiplicative,
        xreg_number=0,
        silent=silent,
    )

    # Update models_pool from restriction
    if pool_restriction["models_pool"] is not None:
        model_type_dict["models_pool"] = pool_restriction["models_pool"]
    elif models_pool is not None:
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
    # Use actual y_in_sample length to account for holdout split
    actual_y_in_sample = ot_info.get("y_in_sample", data)
    actual_obs_in_sample = len(actual_y_in_sample) if hasattr(actual_y_in_sample, '__len__') else obs_in_sample
    observations_dict = {
        "obs_in_sample": actual_obs_in_sample,
        "obs_nonzero": obs_nonzero,
        "obs_all": occ_info["obs_all"],
        #"obs_states": obs_states,
        "ot_logical": ot_info["ot_logical"],
        "ot": ot_info["ot"],
        "y_in_sample": ot_info.get("y_in_sample", data),
        "y_holdout": ot_info.get("y_holdout", None),
        "frequency": ot_info["frequency"],
        "y_start": ot_info["y_start"],
        "y_in_sample_index": ot_info.get("y_in_sample_index", None),
        "y_forecast_start": ot_info["y_forecast_start"],
        "y_forecast_index": ot_info.get("y_forecast_index", None)
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
        "n_iterations": init_info["n_iterations"],
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

    # Calculate number of parameters using the new n_param table structure
    from smooth.adam_general.core.utils.n_param import build_n_param_table

    n_param = build_n_param_table(
        model_type_dict=model_type_dict,
        persistence_checked=persistence_dict,
        initials_checked=initials_dict,
        arima_checked=arima_dict,
        phi_dict=phi_dict,
        constants_checked=constant_dict,
        explanatory_checked=xreg_dict,
        general=general_dict,
    )

    # Also keep legacy format for backward compatibility
    params_info = _calculate_parameters_number(
        ets_info=ets_info,
        arima_info=arima_info,
        xreg_info=None,
        constant_required=constant_dict["constant_required"],
    )

    # Set parameters number in general dict
    general_dict["parameters_number"] = params_info
    general_dict["n_param"] = n_param

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
