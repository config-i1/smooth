import numpy as np
import pandas as pd

def _warn(msg, silent=False):
    """Helper to show warnings in a style closer to R."""
    if not silent:
        print(f"Warning: {msg}")

def _check_occurrence(data, occurrence, frequency = None, silent=False, holdout=False, h=0):
    """
    Check / handle 'occurrence' parameter similarly to R code.
    Return a dict with occurrence details and nonzero counts.
    """
    data_list = list(data) if not isinstance(data, list) else data
    obs_in_sample = len(data_list)
    obs_all = obs_in_sample + (1 - holdout) * h
    # Identify non-zero observations
    nonzero_indices = [i for i, val in enumerate(data_list) if val is not None and val != 0]
    obs_nonzero = len(nonzero_indices)
    # If all zeroes, fallback
    if all(val == 0 for val in data_list):
        _warn("You have a sample with zeroes only. Your forecast will be zero.", silent)
        return {
            "occurrence": "none",
            "occurrence_model": False,
            "obs_in_sample": obs_in_sample,
            "obs_nonzero": 0,
            "obs_all": obs_all
        }

    # Validate the occurrence choice
    valid_occ = ["none","auto","fixed","general","odds-ratio",
                 "inverse-odds-ratio","direct","provided"]
    if occurrence not in valid_occ:
        _warn(f"Invalid occurrence: {occurrence}. Switching to 'none'.", silent)
        occurrence = "none"

    occurrence_model = (occurrence not in ["none","provided"])
    return {
        "occurrence": occurrence,
        "occurrence_model": occurrence_model,
        "obs_in_sample": obs_in_sample,
        "obs_nonzero": obs_nonzero,
        "obs_all": obs_all
    }

def _check_lags(lags, obs_in_sample, silent=False):
    """
    Validate or tweak the set of lags. Force at least lag=1, remove zeros if any.
    Warn if largest lag >= obs_in_sample.
    Return dictionary with lags info including seasonal lags.
    """
    # Remove any zero-lags
    lags = [lg for lg in lags if lg != 0]
    # Force 1 in lags (for level)
    if 1 not in lags:
        lags.insert(0, 1)
    # Must be positive
    if any(lg <= 0 for lg in lags):
        raise ValueError("Right! Why don't you try complex lags then, mister smart guy? (Lag <= 0 given)")

    # Create lagsModel (matrix in R, list here)
    lags_model = sorted(set(lags))
    
    # Get seasonal lags (all lags > 1)
    lags_model_seasonal = [lag for lag in lags_model if lag > 1]
    max_lag = max(lags) if lags else 1
    if max_lag >= obs_in_sample:
        _warn(f"The maximum lags value is {max_lag}, while sample size is {obs_in_sample}. I cannot guarantee that I'll be able to fit the model.", silent)

    return {
        "lags": sorted(set(lags)),
        "lags_model": lags_model,
        "lags_model_seasonal": lags_model_seasonal,
        "lags_length": len(lags_model),
        "max_lag": max_lag
    }

def _expand_component_code(comp_char, allow_multiplicative=True):
    """
    Expand a single component character into a list of valid possibilities,
    following the R approach more fully:
      - 'Z' => ['A','M','N'] (if multiplicative allowed) or ['A','N'] otherwise
      - 'F' => "full" => ['A','M','N'] if multiplicative allowed
      - 'P' => "pure" => ['A','M'] if multiplicative allowed, else ['A']
      - 'C' => "combine all" => we treat like 'Z' expansions but mark combine
      - 'X','Y' => specialized expansions to A or M
      - 'N','A','M' => remain as-is
    """
    possible = set()
    # Handle each special case:
    if comp_char == 'Z':
        # Expand to A,N + M if allowed
        possible.update(['A','N'])
        if allow_multiplicative:
            possible.add('M')
    elif comp_char == 'C':
        # "C" is effectively "combine all" in R. We'll expand the same way as 'Z'
        # but in _build_models_pool_from_components we will detect that we've used 'C'
        # and set combined_mode = True.
        possible.update(['A','N'])
        if allow_multiplicative:
            possible.add('M')
    elif comp_char == 'F':
        # "full" => A, N, plus M if multiplicative
        possible.update(['A','N'])
        if allow_multiplicative:
            possible.add('M')
    elif comp_char == 'P':
        # "pure" => A plus M if multiplicative
        possible.update(['A'])
        if allow_multiplicative:
            possible.add('M')
    elif comp_char == 'X':
        # R logic converts X->A
        possible.update(['A'])
    elif comp_char == 'Y':
        # R logic converts Y->M if allowed, else A
        if allow_multiplicative:
            possible.update(['M'])
        else:
            possible.update(['A'])
    else:
        # If it's one of 'A','M','N' or an unknown letter, just return that
        possible.add(comp_char)
    return list(possible)

def _build_models_pool_from_components(error_type, trend_type, season_type, damped, allow_multiplicative):
    """
    Build a models pool by fully enumerating expansions for E,T,S if any of them
    are Z, F, P, or C. If 'C' appears, we also set combined_mode = True.
    This more closely replicates the R approach of enumerating candidate models.
    """
    err_options   = _expand_component_code(error_type, allow_multiplicative)
    trend_options = _expand_component_code(trend_type, allow_multiplicative)
    seas_options  = _expand_component_code(season_type, allow_multiplicative)

    # Check if 'C' is in any expansions => combined_mode
    combined_mode = ('C' in error_type or 'C' in trend_type or 'C' in season_type)

    # Build candidate models
    candidate_models = []
    for e in err_options:
        for t in trend_options:
            for s in seas_options:
                # Add 'd' if damped
                if damped and t not in ['N']:
                    candidate_models.append(f"{e}{t}d{s}")
                else:
                    candidate_models.append(f"{e}{t}{s}")

    candidate_models = list(set(candidate_models))  # unique
    candidate_models.sort()
    return candidate_models, combined_mode

def _check_model_composition(model_str, allow_multiplicative=True, silent=False):
    """Parse and validate model composition string.
    
    Args:
        model_str: String like "ANN", "ZZZ", etc. representing model components
        allow_multiplicative: Whether multiplicative models are allowed
        silent: Whether to suppress warnings
    
    Returns:
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
            _warn(f"Invalid model type: {model_str}. Should be a string. Switching to 'ZZZ'.")
        model_str = "ZZZ"
        
    # Handle 4-character models (with damping)
    if len(model_str) == 4:
        error_type = model_str[0]
        trend_type = model_str[1]
        season_type = model_str[3]
        if model_str[2] != 'd':
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
    valid_error = ["Z", "X", "Y", "A", "M", "C", "N"]
    valid_trend = ["Z", "X", "Y", "N", "A", "M", "C"] 
    valid_season = ["Z", "X", "Y", "N", "A", "M", "C"]

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
    elif any(c in ["Z", "X", "Y", "F", "P"] for c in [error_type, trend_type, season_type]):
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

    # Generate models pool for selection/combination if needed
    if model_do in ["select", "combine"]:
        models_pool = _generate_models_pool(error_type, trend_type, season_type, 
                                         allow_multiplicative, silent)

    return {
        "error_type": error_type,
        "trend_type": trend_type,
        "season_type": season_type,
        "damped": damped,
        "model_do": model_do,
        "models_pool": models_pool
    }

def _generate_models_pool(error_type, trend_type, season_type, allow_multiplicative, silent=False):
    """Generate pool of candidate models based on components."""
    pool = []
    
    # Handle full pool case ("FFF")
    if "F" in [error_type, trend_type, season_type]:
        pool = ["ANN", "AAN", "AAdN", "AMN", "AMdN",
                "ANA", "AAA", "AAdA", "AMA", "AMdA",
                "ANM", "AAM", "AAdM", "AMM", "AMdM"]
        if allow_multiplicative:
            pool.extend([
                "MNN", "MAN", "MAdN", "MMN", "MMdN",
                "MNA", "MAA", "MAdA", "MMA", "MMdA",
                "MNM", "MAM", "MAdM", "MMM", "MMdM"
            ])
            
    # Handle pure models case ("PPP")
    elif "P" in [error_type, trend_type, season_type]:
        pool = ["ANN", "AAN", "AAdN", "ANA", "AAA", "AAdA"]
        if allow_multiplicative:
            pool.extend(["MNN", "MMN", "MMdN", "MNM", "MMM", "MMdM"])
            
    # Handle standard selection case
    else:
        # Generate based on provided components
        error_options = ["A", "M"] if allow_multiplicative else ["A"]
        trend_options = ["N", "A", "Ad"] if trend_type != "N" else ["N"]
        season_options = ["N", "A", "M"] if season_type != "N" and allow_multiplicative else ["N", "A"] if season_type != "N" else ["N"]
        
        if error_type in ["A", "M"]:
            error_options = [error_type]
        if trend_type in ["A", "M", "N"]:
            trend_options = [trend_type]
        if season_type in ["A", "M", "N"]:
            season_options = [season_type]
            
        for e in error_options:
            for t in trend_options:
                for s in season_options:
                    if "d" in t:
                        pool.append(f"{e}{t[0]}d{s}")
                    else:
                        pool.append(f"{e}{t}{s}")
                        
    return sorted(list(set(pool)))

def _check_ets_model(model, distribution, data, silent=False):
    """
    Full check for ETS logic. Return dictionary with error_type, trend_type, season_type, damped, ets_model...
    If error_type='N', skip ETS model.
    """

 # Convert DataFrame/Series to numeric values if needed
    if hasattr(data, 'values'):
        data_values = data.values.flatten() if hasattr(data, 'flatten') else data.values
    else:
        data_values = data


    # Check multiplicative components first
    data_min = min(data_values) if hasattr(data, 'values') else min(data)
    allow_multiplicative = True
    if data_min <= 0 and distribution in ["dlnorm", "dgamma", "dinvgauss", "dllaplace", "dlgnorm", "dls"]:
        allow_multiplicative = False

    # Now call _check_model_composition with all required arguments
    model_info = _check_model_composition(model, allow_multiplicative, silent)
    # Extract components
    error_type = model_info["error_type"]
    trend_type = model_info["trend_type"]
    season_type = model_info["season_type"]
    damped = model_info["damped"]

    # If error_type='N', that means no ETS
    if error_type == "N":
        trend_type = "N"
        season_type = "N"
        damped = False
        return {
            "ets_model": False,
            "error_type": "N",
            "trend_type": "N",
            "season_type": "N",
            "damped": False,
            "allow_multiplicative": False,
            "model": "NNN"
        }

    # Warn about multiplicative components if needed
    if not allow_multiplicative and not silent and any(x == "M" for x in (error_type, trend_type, season_type)):
        _warn("Your data contains non-positive values, so the ETS might break for multiplicative components.")

    # Construct final model string
    if len(model) == 4:
        model = f"{error_type}{trend_type}d{season_type}"
    else:
        model = f"{error_type}{trend_type}{season_type}"

    return {
        "ets_model": True,
        "model": model,
        "error_type": error_type,
        "trend_type": trend_type,
        "season_type": season_type,
        "damped": damped,
        "allow_multiplicative": allow_multiplicative
    }

def _expand_orders(orders):
    """Tiny helper to convert orders to lists: (ar, i, ma)."""
    if isinstance(orders, dict):
        ar_orders = orders.get("ar", [])
        i_orders = orders.get("i", [])
        ma_orders = orders.get("ma", [])
    elif isinstance(orders, (list, tuple)) and len(orders) == 3:
        ar_orders, i_orders, ma_orders = orders
    else:
        ar_orders, i_orders, ma_orders = 0, 0, 0
    if not isinstance(ar_orders, (list, tuple)):
        ar_orders = [ar_orders]
    if not isinstance(i_orders, (list, tuple)):
        i_orders = [i_orders]
    if not isinstance(ma_orders, (list, tuple)):
        ma_orders = [ma_orders]
    return ar_orders, i_orders, ma_orders

def _check_arima(orders, validated_lags, silent=False):
    """
    Validate ARIMA portion and return complete ARIMA information.
    
    Args:
        orders: ARIMA orders (can be dict, list/tuple, or single values)
        validated_lags: List of validated lags
        silent: Whether to suppress warnings
    
    Returns:
        dict: Complete ARIMA information including all components
    """
    ar_orders, i_orders, ma_orders = _expand_orders(orders)
    
    # Sum orders to determine if ARIMA is needed
    s_ar = sum(ar_orders)
    s_i = sum(i_orders)
    s_ma = sum(ma_orders)
    arima_model = (s_ar + s_i + s_ma) > 0

    # Initialize result dictionary
    result = {
        "arima_model": arima_model,
        "ar_orders": ar_orders,
        "i_orders": i_orders,
        "ma_orders": ma_orders,
        "ar_required": False,
        "i_required": False,
        "ma_required": False,
        "ar_estimate": False,
        "ma_estimate": False,
        "arma_parameters": None,
        "non_zero_ari": [],
        "non_zero_ma": [],
        "lags_model_arima": [],
        "select": False
    }

    if not arima_model:
        return result

    # Set required flags based on orders
    result["ar_required"] = (s_ar > 0)
    result["i_required"] = (s_i > 0)
    result["ma_required"] = (s_ma > 0)
    
    # Set estimation flags - if component is required, it needs to be estimated
    result["ar_estimate"] = result["ar_required"]
    result["ma_estimate"] = result["ma_required"]

    # Create ariValues and maValues lists
    ari_values = []
    ma_values = []
    for i, lag in enumerate(validated_lags):
        # AR and I orders combined
        ari = [0]  # Start with 0
        if ar_orders[i] > 0:
            ari.extend(range(min(1, ar_orders[i]), ar_orders[i] + 1))
        if i_orders[i] > 0:
            ari.extend(range(ar_orders[i] + 1, ar_orders[i] + i_orders[i] + 1))
        ari_values.append([x * lag for x in set(ari)])
        
        # MA orders
        ma = [0]  # Start with 0
        if ma_orders[i] > 0:
            ma.extend(range(min(1, ma_orders[i]), ma_orders[i] + 1))
        ma_values.append([x * lag for x in set(ma)])

    # Get non-zero values
    result["non_zero_ari"] = sorted(set([x for sublist in ari_values for x in sublist if x != 0]))
    result["non_zero_ma"] = sorted(set([x for sublist in ma_values for x in sublist if x != 0]))
    
    # Combine and sort unique lags for ARIMA components
    result["lags_model_arima"] = sorted(set(result["non_zero_ari"] + result["non_zero_ma"]))

    # Initialize ARMA parameters if needed
    if result["ar_required"] or result["ma_required"]:
        result["arma_parameters"] = []
        # Add AR parameters
        if result["ar_required"]:
            for lag in validated_lags:
                for i in range(max(ar_orders)):
                    result["arma_parameters"].append({
                        "name": f"phi{i+1}[{lag}]",
                        "value": None
                    })
        # Add MA parameters
        if result["ma_required"]:
            for lag in validated_lags:
                for i in range(max(ma_orders)):
                    result["arma_parameters"].append({
                        "name": f"theta{i+1}[{lag}]",
                        "value": None
                    })

    # Warning for high frequency data
    if max(validated_lags) >= 24 and not silent:
        _warn("The estimation of ARIMA model with large lags might take time. Consider initial='backcasting' or simpler orders.", silent)

    return result

def _check_distribution_loss(distribution, loss, silent=False):
    """Check distribution and loss from a known set, falling back to defaults if invalid."""
    valid_distributions = {
        "default","dnorm","dlaplace","dalaplace","ds","dgnorm",
        "dlnorm","dinvgauss","dgamma"
    }
    if distribution not in valid_distributions:
        _warn(f"distribution '{distribution}' not recognized. Using 'default'.", silent)
        distribution = "default"

    valid_losses = {
        "likelihood","MSE","MAE","HAM","LASSO","RIDGE",
        "MSEh","TMSE","GTMSE","MSCE","MAEh","TMAE","GTMAE","MACE",
        "HAMh","THAM","GTHAM","CHAM","GPL",
        "aMSEh","aTMSE","aGTMSE","aMSCE","aGPL","custom"
    }
    if loss not in valid_losses:
        _warn(f"loss '{loss}' is not recognized. Using 'likelihood'.", silent)
        loss = "likelihood"

    return {"distribution": distribution, "loss": loss}

def _check_outliers(outliers_mode, silent=False):
    """Ensure outliers mode is one of 'ignore','use','select'."""
    choices = ["ignore","use","select"]
    if outliers_mode not in choices:
        _warn(f"outliers='{outliers_mode}' not recognized. Switching to 'ignore'.", silent)
        outliers_mode = "ignore"
    return outliers_mode

def _check_phi(phi, damped, silent=False):
    """
    If damped is True, we want 0<phi<2.
    If user supplies invalid phi => fallback to 0.95.
    If user doesn't supply => default to 0.95 if damped, else 1.
    """
    if damped:
        if phi is None:
            phi_val = 0.95
            phi_est = True
        else:
            # numeric?
            try:
                p = float(phi)
                if p <= 0 or p >= 2:
                    _warn(f"Damping parameter should lie in (0,2). Changing to estimate with initial=0.95", silent)
                    phi_val = 0.95
                    phi_est = True
                else:
                    phi_val = p
                    phi_est = False
            except:
                _warn(f"Provided value of phi is invalid. Using 0.95 as a guess, will estimate it.", silent)
                phi_val = 0.95
                phi_est = True
    else:
        # not damped => phi=1
        phi_val = 1.0
        phi_est = False

    return {"phi": phi_val, "phi_estimate": phi_est}

def _check_persistence(persistence, ets_model, trend_type, season_type, lags_model_seasonal, xreg_model=False, silent=False):
    """
    Check persistence parameters and return dictionary with all persistence-related variables.
    Mirrors R code's persistence handling.
    
    Args:
        persistence: The persistence parameter (can be None, numeric, list, or dict)
        ets_model: Boolean indicating if ETS model is used
        trend_type: The trend type ('N', 'A', 'M' etc)
        season_type: The seasonal type ('N', 'A', 'M' etc)
        lags_model_seasonal: List of seasonal lags
        xreg_model: Boolean indicating if explanatory variables are used
        silent: Whether to suppress warnings
    
    Returns:
        dict: Dictionary containing all persistence-related variables
    """
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
    
    # If no persistence provided, return defaults
    if persistence is None:
        return result

    # Handle different persistence input types
    if isinstance(persistence, dict):
        # Named dictionary case (similar to R's named list)
        if "level" in persistence or "alpha" in persistence:
            result["persistence_level"] = persistence.get("level", persistence.get("alpha"))
            result["persistence_level_estimate"] = False
            
        if "trend" in persistence or "beta" in persistence:
            result["persistence_trend"] = persistence.get("trend", persistence.get("beta"))
            result["persistence_trend_estimate"] = False
            
        if "seasonal" in persistence or "gamma" in persistence:
            seasonal_value = persistence.get("seasonal", persistence.get("gamma"))
            if isinstance(seasonal_value, (list, tuple)):
                # Set ALL components based on length match
                all_estimate = len(seasonal_value) != len(lags_model_seasonal)
                result["persistence_seasonal_estimate"] = [all_estimate] * n_seasonal
                if not all_estimate:
                    result["persistence_seasonal"] = list(seasonal_value)
            else:
                # Non-list value provided - set ALL to False
                result["persistence_seasonal_estimate"] = [False] * n_seasonal
                result["persistence_seasonal"][0] = seasonal_value

        if "xreg" in persistence or "delta" in persistence:
            result["persistence_xreg"] = persistence.get("xreg", persistence.get("delta"))
            result["persistence_xreg_estimate"] = False
            result["persistence_xreg_provided"] = True

    elif isinstance(persistence, (list, tuple)):
        # List/tuple case
        if len(persistence) > 0 and persistence[0] is not None:
            result["persistence_level"] = persistence[0]
            result["persistence_level_estimate"] = False
            
        if len(persistence) > 1 and persistence[1] is not None:
            result["persistence_trend"] = persistence[1]
            result["persistence_trend_estimate"] = False
            
        if len(persistence) > 2 and persistence[2] is not None:
            result["persistence_seasonal"] = persistence[2]
            result["persistence_seasonal_estimate"] = False
            
        if len(persistence) > 3 and persistence[3] is not None:
            result["persistence_xreg"] = persistence[3]
            result["persistence_xreg_estimate"] = False
            result["persistence_xreg_provided"] = True

    elif isinstance(persistence, (int, float)):
        # Single numeric value case
        result["persistence_level"] = float(persistence)
        result["persistence_level_estimate"] = False
        
    else:
        _warn("Persistence is not numeric/list/dict. We'll estimate it instead.", silent)
        return result

    # Update main persistence estimate flag based on component estimates
    result["persistence_estimate"] = any([
        result["persistence_level_estimate"] and ets_model,
        result["persistence_trend_estimate"] and trend_type != "N",
        result["persistence_seasonal_estimate"] and season_type != "N",
        result["persistence_xreg_estimate"] and xreg_model
    ])

    # Make sure only relevant components are estimated
    if not ets_model:
        result["persistence_level_estimate"] = False
        result["persistence_level"] = None
        
    if trend_type == "N":
        result["persistence_trend_estimate"] = False
        result["persistence_trend"] = None
        
    if season_type == "N":
        result["persistence_seasonal_estimate"] = False
        result["persistence_seasonal"] = None
        
    if not xreg_model:
        result["persistence_xreg_estimate"] = False
        result["persistence_xreg"] = None
        result["persistence_xreg_provided"] = False

    return result

def _check_initial(initial, ets_model, trend_type, season_type, arima_model=False, xreg_model=False, silent=False):
    """
    Check initial parameters and return dictionary with all initial-related variables.
    Mirrors R code's initial handling.
    
    Args:
        initial: The initial parameter (can be None, numeric, list/tuple, or dict)
        ets_model: Boolean indicating if ETS model is used
        trend_type: The trend type ('N', 'A', 'M' etc)
        season_type: The seasonal type ('N', 'A', 'M' etc)
        arima_model: Boolean indicating if ARIMA model is used
        xreg_model: Boolean indicating if explanatory variables are used
        silent: Whether to suppress warnings
    
    Returns:
        dict: Dictionary containing all initial-related variables
    """
    result = {
        "initial": initial,  # Store original value
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
        "initial_arima_number": 0,  # Will be set properly if ARIMA model is used
        "initial_xreg_estimate": True,
        "initial_xreg_provided": False
    }
    
    # If no initial provided, return defaults with optimal type
    if initial is None:
        #if not silent:
            #print("Initial value is not selected. Switching to optimal.")
        return result

    # Handle string types
    if isinstance(initial, str):
        valid_types = ["optimal", "backcasting", "complete", "provided"]
        if initial not in valid_types:
            _warn(f"Initial '{initial}' not recognized. Using 'optimal'.", silent)
            return result
        
        result["initial_type"] = initial
        # Set estimate flags based on type
        is_estimate = (initial != "provided" and initial != "complete")
        result["initial_estimate"] = is_estimate
        result["initial_level_estimate"] = is_estimate
        result["initial_trend_estimate"] = is_estimate
        result["initial_seasonal_estimate"] = is_estimate
        result["initial_arima_estimate"] = is_estimate
        result["initial_xreg_estimate"] = is_estimate
        return result

    # Handle dictionary case (similar to R's named list)
    if isinstance(initial, dict):
        result["initial_type"] = "provided"
        result["initial_estimate"] = False
        
        if "level" in initial:
            result["initial_level"] = initial["level"]
            result["initial_level_estimate"] = False
            
        if "trend" in initial:
            result["initial_trend"] = initial["trend"]
            result["initial_trend_estimate"] = False
            
        if "seasonal" in initial:
            result["initial_seasonal"] = initial["seasonal"]
            result["initial_seasonal_estimate"] = False
            
        if "arima" in initial:
            result["initial_arima"] = initial["arima"]
            result["initial_arima_estimate"] = False
            
        if "xreg" in initial:
            result["initial_xreg_provided"] = True
            result["initial_xreg_estimate"] = False

    # Handle numeric or list/tuple case
    elif isinstance(initial, (int, float, list, tuple)):
        result["initial_type"] = "provided"
        result["initial_estimate"] = False
        
        # Convert to list for consistent handling
        init_values = [initial] if isinstance(initial, (int, float)) else list(initial)
        
        # Assign values based on position
        if len(init_values) > 0:
            result["initial_level"] = init_values[0]
            result["initial_level_estimate"] = False
            
        if len(init_values) > 1 and trend_type != "N":
            result["initial_trend"] = init_values[1]
            result["initial_trend_estimate"] = False
            
        if len(init_values) > 2 and season_type != "N":
            result["initial_seasonal"] = init_values[2]
            result["initial_seasonal_estimate"] = False
            
        if len(init_values) > 3 and arima_model:
            result["initial_arima"] = init_values[3]
            result["initial_arima_estimate"] = False
    
    else:
        _warn("Initial vector is not numeric! Using optimal initialization.", silent)
        return result

    # Make sure only relevant components are estimated
    if not ets_model:
        result["initial_level_estimate"] = False
        result["initial_level"] = None
        
    if trend_type == "N":
        result["initial_trend_estimate"] = False
        result["initial_trend"] = None
        
    if season_type == "N":
        result["initial_seasonal_estimate"] = False
        result["initial_seasonal"] = None
        
    if not arima_model:
        result["initial_arima_estimate"] = False
        result["initial_arima"] = None
        result["initial_arima_number"] = 0
        
    if not xreg_model:
        result["initial_xreg_estimate"] = False
        result["initial_xreg_provided"] = False

    return result

def _check_constant(constant, silent=False):
    """
    R code: numeric => use it as drift/constant, logical => estimate or not.
    """
    if isinstance(constant, bool):
        return {
            "constant_required": constant,
            "constant_estimate": constant,
            "constant_value": None
        }
    if isinstance(constant, (float,int)):
        return {
            "constant_required": True,
            "constant_estimate": False,
            "constant_value": float(constant)
        }
    _warn(f"The parameter 'constant' can only be bool or numeric. Found '{constant}'. Switching to FALSE", silent)
    return {
        "constant_required": False,
        "constant_estimate": False,
        "constant_value": None
    }

def _initialize_estimation_params(
    loss,
    lambda_param,
    ets_info,
    arima_info,
    silent=False
):
    """
    Initialize estimation parameters, particularly for LASSO/RIDGE cases.
    This mirrors the R code's initialization logic in the model_do="estimate" section.
    
    Args:
        loss (str): Loss function type
        lambda_param (float): Lambda parameter for LASSO/RIDGE
        ets_info (dict): ETS model information
        arima_info (dict): ARIMA model information
        silent (bool): Whether to suppress warnings
    
    Returns:
        dict: Dictionary containing initialized parameters and lambda value
    """
    # Only proceed with special initialization if LASSO/RIDGE with lambda=1
    if loss not in ["LASSO", "RIDGE"] or lambda_param != 1:
        return {
            "lambda": lambda_param,
            "persistence_params": None,
            "arma_params": None
        }

    result = {
        "lambda": 0  # Set lambda to 0 for initial estimation
    }

    # Initialize persistence parameters if ETS model
    if ets_info["ets_model"]:
        persistence_params = {
            "estimate": False,
            "level_estimate": False,
            "trend_estimate": False,
            "seasonal_estimate": False,
            "level": 0,
            "trend": 0,
            "seasonal": 0,
            "phi": 1,
            "phi_estimate": False
        }
        result["persistence_params"] = persistence_params

    # Initialize ARMA parameters if ARIMA model
    if arima_info["arima_model"]:
        ar_orders = arima_info["ar_orders"]
        ma_orders = arima_info["ma_orders"]
        lags = sorted(set([1] + (arima_info.get("lags", []) or [])))  # Ensure we have lags

        # Initialize ARMA parameters
        arma_params = {
            "ar_estimate": False,
            "ma_estimate": False,
            "parameters": []
        }

        # Build ARMA parameters list with proper naming
        for lag in lags:
            # Add AR parameters (all set to 1)
            ar_count = ar_orders[0] if ar_orders else 0  # Simplified - might need adjustment
            for i in range(ar_count):
                arma_params["parameters"].append({
                    "name": f"phi{i+1}[{lag}]",
                    "value": 1
                })
            
            # Add MA parameters (all set to 0)
            ma_count = ma_orders[0] if ma_orders else 0  # Simplified - might need adjustment
            for i in range(ma_count):
                arma_params["parameters"].append({
                    "name": f"theta{i+1}[{lag}]",
                    "value": 0
                })

        result["arma_params"] = arma_params

    return result

def _organize_model_type_info(ets_info, arima_info, xreg_model=False):
    """
    Organize basic model type information into a structured dictionary.
    
    Args:
        ets_info (dict): ETS model information
        arima_info (dict): ARIMA model information
        xreg_model (bool): Whether explanatory variables are used
    
    Returns:
        dict: Model type information
    """
    return {
        "ets_model": ets_info["ets_model"],
        "arima_model": arima_info["arima_model"],
        "xreg_model": xreg_model,
        "model": ets_info["model"],
        "error_type": ets_info["error_type"],
        "trend_type": ets_info["trend_type"],
        "season_type": ets_info["season_type"],
        "damped": ets_info["damped"],
        "allow_multiplicative": ets_info["allow_multiplicative"],
        "model_is_trendy": ets_info["trend_type"] != "N",
        "model_is_seasonal": ets_info["season_type"] != "N",
        "model_do": "estimate",  # default, can be overridden
        "models_pool": ets_info.get("models_pool", None)
    }

def _organize_components_info(ets_info, arima_info, lags_model_seasonal):
    """
    Organize components information into a structured dictionary.
    
    Args:
        ets_info (dict): ETS model information
        arima_info (dict): ARIMA model information
        lags_model_seasonal (list): List of seasonal lags
    
    Returns:
        dict: Components information
    """
    # Calculate ETS components
    components_number_ets = sum([
        ets_info["ets_model"],  # level component if ETS
        ets_info["trend_type"] != "N",  # trend component if present
        ets_info["season_type"] != "N"  # seasonal component if present
    ])
    
    components_names_ets = [
        name for include, name in [
            (ets_info["ets_model"], "level"),
            (ets_info["trend_type"] != "N", "trend"),
            (ets_info["season_type"] != "N", "seasonal")
        ] if include
    ]
    
    # Calculate seasonal components
    components_number_ets_seasonal = int(ets_info["season_type"] != "N")
    components_number_ets_non_seasonal = components_number_ets - components_number_ets_seasonal
    
    return {
        # ETS components
        "components_number_ets": components_number_ets,
        "components_names_ets": components_names_ets,
        "components_number_ets_seasonal": components_number_ets_seasonal,
        "components_number_ets_non_seasonal": components_number_ets_non_seasonal,
        
        # ARIMA components
        "components_number_arima": len(arima_info.get("lags_model_arima", [])),
        "components_names_arima": [f"ARIMAState{i+1}" for i in range(len(arima_info.get("lags_model_arima", [])))],
        
        # Seasonal info
        "lags_model_seasonal": lags_model_seasonal,
        
        # Total components
        "components_number_total": (components_number_ets + 
                                  len(arima_info.get("lags_model_arima", [])))
    }

def _organize_lags_info(validated_lags, lags_model, lags_model_seasonal, lags_model_arima, xreg_model=False):
    """Organize lags information into a dictionary"""
    # Calculate all model lags (ETS + ARIMA + X)
    if xreg_model:
        # If xreg exists, add ones for each xreg variable
        lags_model_all = sorted(set(lags_model + lags_model_arima + [1]))
    else:
        lags_model_all = sorted(set(lags_model + (lags_model_arima if lags_model_arima else [])))

    # flatten lags_model_all
    return {
        "lags": validated_lags,
        "lags_model": lags_model,
        "lags_model_seasonal": lags_model_seasonal,
        "lags_model_arima": lags_model_arima,
        "lags_model_all": lags_model_all,
        "max_lag": max(lags_model_all) if lags_model_all else 1,
        "lags_model_min": min(lags_model_all) if lags_model_all else 1,
        "lags_length": len(validated_lags)
    }

def _organize_occurrence_info(occurrence, occurrence_model, obs_in_sample, h=0):
    """
    Organize occurrence information into a structured dictionary.
    
    Args:
        occurrence (str): Occurrence type
        occurrence_model (bool): Whether occurrence model is used
        obs_in_sample (int): Number of observations in sample
        h (int): Forecast horizon
    
    Returns:
        dict: Occurrence information
    """
    # Initialize with default values matching R code
    p_fitted = np.ones((obs_in_sample, 1))  # matrix(1, obsInSample, 1) in R
    p_forecast = np.array([np.nan] * h)  # rep(NA,h) in R
    oes_model = None
    occurrence_model_provided = False

    # Handle occurrence model object case (is.occurrence in R)
    if hasattr(occurrence, 'occurrence'):  # equivalent to is.occurrence(occurrence)
        oes_model = occurrence
        occurrence = oes_model.occurrence
        if occurrence == "provided":
            occurrence_model_provided = False
        else:
            occurrence_model_provided = True
        p_fitted = np.matrix(oes_model.fitted).reshape(obs_in_sample, 1)

    # Handle numeric/logical occurrence
    if isinstance(occurrence, (bool, np.bool_)):
        occurrence = int(occurrence)
    
    if isinstance(occurrence, (int, float, np.number)):
        if all(occurrence == 1):
            occurrence = "none"
        else:
            # Check bounds
            if any(o < 0 or o > 1 for o in np.atleast_1d(occurrence)):
                _warn("Parameter 'occurrence' should contain values between zero and one.\nConverting to appropriate vector.")
                occurrence = (occurrence != 0).astype(int)
            
            # Set pFitted from occurrence values
            p_fitted[:] = occurrence[:obs_in_sample]
            
            # Handle forecast values
            if h > 0:
                if len(occurrence) > obs_in_sample:
                    p_forecast = occurrence[obs_in_sample:]
                else:
                    p_forecast = np.repeat(occurrence[-1], h)
                
                # Adjust forecast length
                if len(p_forecast) > h:
                    p_forecast = p_forecast[:h]
                elif len(p_forecast) < h:
                    p_forecast = np.append(p_forecast, np.repeat(p_forecast[-1], h - len(p_forecast)))
            else:
                p_forecast = np.array([np.nan])
            
            occurrence = "provided"
            oes_model = {"fitted": p_fitted, "forecast": p_forecast, "occurrence": "provided"}

    return {
        "occurrence": occurrence,
        "occurrence_model": occurrence_model,
        "occurrence_model_provided": occurrence_model_provided,
        "p_fitted": p_fitted,
        "p_forecast": p_forecast,
        "oes_model": oes_model
    }

def _organize_phi_info(phi_val, phi_estimate):
    """
    Organize phi information into a structured dictionary.
    
    Args:
        phi_val (float): Phi value
        phi_estimate (bool): Whether phi should be estimated
    
    Returns:
        dict: Phi information
    """
    return {
        "phi": phi_val,
        "phi_estimate": phi_estimate
    }

###################
# MAIN ENTRYPOINT #
###################

def parameters_checker(
    data,
    model,
    lags,
    orders = None,
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
    ellipsis=None
):
    """
    Extended parameters_checker that includes initialization logic for estimation.
    """
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

    #####################
    # 1) Occurrence
    #####################
    occ_info = _check_occurrence(data_values, occurrence, silent, holdout, h)
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
    lags_length = lags_info["lags_length"]
    max_lag = lags_info["max_lag"]

    #####################
    # 3) Check ETS Model
    #####################
    ets_info = _check_ets_model(model, distribution, data, silent)
    ets_model = ets_info["ets_model"]   # boolean
    model_str = ets_info["model"]
    error_type = ets_info["error_type"]
    trend_type = ets_info["trend_type"]
    season_type = ets_info["season_type"]
    damped = ets_info["damped"]
    allow_mult = ets_info["allow_multiplicative"]

    #####################
    # 4) ARIMA checks
    #####################
    arima_info = _check_arima(orders, validated_lags, silent)
    arima_model = arima_info["arima_model"]
    ar_orders = arima_info["ar_orders"]
    i_orders = arima_info["i_orders"]
    ma_orders = arima_info["ma_orders"]
    lags_model_arima = arima_info["lags_model_arima"]
    non_zero_ari = arima_info["non_zero_ari"]
    non_zero_ma = arima_info["non_zero_ma"]

    #####################
    # 5) Dist & Loss
    #####################
    dist_info = _check_distribution_loss(distribution, loss, silent)
    distribution = dist_info["distribution"]
    loss = dist_info["loss"]

    #####################
    # 6) Outliers
    #####################
    outliers_mode = _check_outliers(outliers, silent)

    #####################
    # 7) Persistence
    #####################
    persist_info = _check_persistence(
        persistence=persistence,
        ets_model=ets_model,
        trend_type=trend_type,
        season_type=season_type,
        lags_model_seasonal=lags_model_seasonal,
        xreg_model=False,  # You'll need to add xreg handling logic
        silent=silent
    )
    
    #####################
    # 8) Initial
    #####################
    init_info = _check_initial(
        initial=initial,
        ets_model=ets_model,
        trend_type=trend_type,
        season_type=season_type,
        arima_model=arima_model,
        xreg_model=False,  # You'll need to add xreg handling logic
        silent=silent
    )

    # Create initials dictionary
    initials_results = {
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
        "initial_xreg_provided": init_info["initial_xreg_provided"]
    }

    #####################
    # 9) Constant
    #####################
    constant_dict = _check_constant(constant, silent)

    #####################
    # 9.1) Check phi
    #####################
    phi_info = _check_phi(phi, damped, silent)
    phi_val = phi_info["phi"]
    phi_estimate = phi_info["phi_estimate"]

    #####################
    # 10) Validate Bounds
    #####################
    if bounds not in ["usual","admissible","none"]:
        _warn(f"Unknown bounds='{bounds}'. Switching to 'usual'.", silent)
        bounds = "usual"

    #####################
    # 11) holdout logic
    #####################
    if holdout and h <= 0:
        _warn("holdout=TRUE but horizon 'h' is not positive. No real holdout can be made.", silent)

    #####################
    # 12) Model selection fallback
    #####################
    # The R code tries to reduce the model complexity if obs_nonzero < #params, etc.
    # We'll do a simplified fallback if sample is too small. This is a partial approach.
    # (In R code, there's extensive logic around "if(etsModel && obsNonzero <= nParamMax) {...}")
    # We'll do a simpler approach:
    if ets_model and (obs_nonzero < 3):
        # Switch to ANN or do-nothing approach
        _warn("Not enough of non-zero observations for a complicated ETS model. Switching to 'ANN'.", silent)
        error_type, trend_type, season_type = "A", "N", "N"
        damped = False
        phi_val = 1.0
        model_str = "ANN"
    # We might do more checks, but keep it short here.

    # Check if multiplicative models are allowed (using data_values instead of data)
    allow_multiplicative = not ((any(y <= 0 for y in data_values if not pd.isna(y)) and not occurrence_model) or 
                              (occurrence_model and any(y < 0 for y in data_values if not pd.isna(y))))
    
    # Check model composition - reuse components from ets_info if possible
    if ets_model:
        # Reuse the model_info from ets_info to avoid inconsistency
        error_type = ets_info["error_type"]
        trend_type = ets_info["trend_type"]
        season_type = ets_info["season_type"]
        damped = ets_info["damped"]
        model_str = ets_info["model"]
        
        # Check model composition with current allow_multiplicative value
        model_info = _check_model_composition(model_str, allow_multiplicative, silent)
    else:
        # For non-ETS models, check composition directly
        model_info = _check_model_composition(model, allow_multiplicative, silent)
    
    final_model_do = model_info["model_do"]
    candidate_pool = model_info["models_pool"]

    if final_model_do in ["select", "combine"]:
        # This replicates R's auto selection or combination
        # (in R, it enumerates, fits each, then picks or combines)
        fitted_results = []
        for candidate in candidate_pool:
            # parse submodel
            sub_error_type = candidate[0]
            sub_trend_type = candidate[1]
            sub_season_type = candidate[-1]
            sub_damped = ('d' in candidate[2:-1]) if len(candidate) == 4 else False

            # Fit the submodel here (omitted for brevity)...
            # e.g. sub_fit = _fit_submodel(data, sub_error_type, sub_trend_type, sub_season_type, sub_damped, ...)
            # Then store results
            fitted_results.append((candidate, None))  # placeholder

        if final_model_do == "select":
            # In R, you'd pick best by IC or something; let's pick first for the example
            best_model = candidate_pool[0]
        else:
            # "combine": in R, you'd average forecasts from all or use weights
            best_model = candidate_pool[0]  # placeholder

        # Overwrite final model with the chosen "best_model" or a combined approach
        error_type = best_model[0]
        trend_type = best_model[1]
        season_type = best_model[-1]
        damped = ('d' in best_model[2:-1]) if len(best_model) == 4 else False
        if damped and trend_type != 'N':
            final_model_str = f"{error_type}{trend_type}d{season_type}"
        else:
            final_model_str = f"{error_type}{trend_type}{season_type}"
    else:
        # Normal single model
        error_type = model_info["error_type"]
        trend_type = model_info["trend_type"]
        season_type = model_info["season_type"]
        damped      = model_info["damped"]
        if damped and trend_type != 'N':
            final_model_str = f"{error_type}{trend_type}d{season_type}"
        else:
            final_model_str = f"{error_type}{trend_type}{season_type}"

    # ... continue with ARIMA checks, etc. Pass final_model_str onward ...
    # finalize the return or proceed
    # ... rest of the existing function ...

    # Create lags dictionary
    lags_dict = _organize_lags_info(
        validated_lags=validated_lags,
        lags_model=lags_model,
        lags_model_seasonal=lags_model_seasonal,
        lags_model_arima=lags_model_arima,
        xreg_model=False  # Update this when xreg is implemented
    )

    # Create occurrence dictionary
    occurrence_dict = _organize_occurrence_info(
        occurrence=occ_info["occurrence"],
        occurrence_model=occurrence_model,
        obs_in_sample=obs_in_sample,
        h=h
    )

    # Create phi dictionary
    phi_dict = _organize_phi_info(
        phi_val=phi_val,
        phi_estimate=phi_estimate
    )

    # Main results dictionary - remove occurrence and phi info
    # Create observations dictionary
    ot_info = _calculate_ot_logical(data, occurrence_dict["occurrence"], 
                                  occurrence_dict["occurrence_model"], 
                                  obs_in_sample,
                                  frequency, h, holdout)

    observations_dict = {
        "obs_in_sample": obs_in_sample,
        "obs_nonzero": obs_nonzero,
        "obs_all": occ_info["obs_all"],
        "ot_logical": ot_info["ot_logical"],
        "ot": ot_info["ot"],
        "y_in_sample": ot_info.get("y_in_sample", data),  # Use split data if available
        "y_holdout": ot_info.get("y_holdout", None),  # Add holdout data
        "frequency": ot_info["frequency"],
        "y_start": ot_info["y_start"],
        "y_in_sample_index": ot_info.get("y_in_sample_index", None),  # Add the index to observations_dict
        "y_forecast_start": ot_info["y_forecast_start"]  # Make sure this is here too
    }

    # Create general dictionary with remaining parameters
    general = {
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
        "ellipsis": ellipsis
    }

    #####################
    # Initialize Estimation
    #####################
    if model_do == "estimate":
        init_params = _initialize_estimation_params(
            loss=loss,
            lambda_param=lambda_param or 1,  # Default to 1 if not provided
            ets_info=ets_info,
            arima_info=arima_info,
            silent=silent
        )
        # Update results with initialization parameters
        general.update({
            "lambda": init_params["lambda"],
            "persistence_params": init_params.get("persistence_params"),
            "arma_params": init_params.get("arma_params")
        })
    
    # Persistence-specific dictionary
    persistence_results = {
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
        "persistence_xreg_provided": persist_info["persistence_xreg_provided"]
    }

    # ARIMA-specific dictionary
    arima_results = {
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
        "non_zero_ari": non_zero_ari,
        "non_zero_ma": non_zero_ma,
        "select": arima_info.get("select", False)
    }

    # Create model type dictionary
    model_type_dict = _organize_model_type_info(ets_info, arima_info, xreg_model=False)
    
    # Update model_do based on final_model_do instead of input parameter
    model_type_dict["model_do"] = final_model_do
    
    # Add candidate_pool to model_type_dict if model selection is needed
    if final_model_do in ["select", "combine"]:
        model_type_dict["models_pool"] = candidate_pool
    
    # Organize components info
    components_dict = _organize_components_info(ets_info, arima_info, lags_model_seasonal)
    # Initiliaze the explonatory dict -> will not be used for now
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
        "xreg_parameters_persistence": None
    }

    # Calculate parameters number
    # Calculate parameters number
    params_info = _calculate_parameters_number(
        ets_info=ets_info,
        arima_info=arima_info,
        xreg_info=None,  # Add xreg handling if needed
        constant_required=constant_dict["constant_required"]
    )
    
    # Return all dictionaries including new lags_dict
    return (general, 
            observations_dict,
            persistence_results, 
            initials_results, 
            arima_results, 
            constant_dict, 
            model_type_dict, 
            components_dict, 
            lags_dict,
            occurrence_dict,
            phi_dict,
            xreg_dict,
            params_info)

# Calculate otLogical based on the R code logic
def _calculate_ot_logical(data, occurrence, occurrence_model, obs_in_sample, frequency=None, h=0, holdout=False):
    """Calculate otLogical vector and ot based on occurrence type and data
    
    Args:
        data: Input time series data
        occurrence (str): Occurrence type
        occurrence_model (bool): Whether occurrence model is used
        obs_in_sample (int): Number of in-sample observations
        frequency (int, optional): Time series frequency. If None, will be inferred.
        h (int): Forecast horizon
        holdout (bool): Whether to create holdout set
    
    Returns:
        dict: Dictionary containing ot_logical, ot, frequency, y_start and y_holdout
    """
    # Convert data to numpy array if needed
    if hasattr(data, 'values'):
        y_in_sample = data.values.flatten() if hasattr(data.values, 'flatten') else data.values
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

    # Use provided frequency if available, otherwise infer it
    if frequency is not None:
        freq = str(frequency)
    else:
        # Default frequency
        freq = "1"  # Default string frequency
        
        # Try to infer frequency from data if it's a pandas Series/DataFrame
        if hasattr(data, 'index'):
            # Try to get frequency from index
            if hasattr(data.index, 'freq') and data.index.freq is not None:
                # Get the actual frequency string
                freq = data.index.freq
                if not freq:  # If empty string, fallback to default
                    freq = "1"

    # Get start time from index if available
    y_start = 0  # default
    if hasattr(data, 'index') and len(data.index) > 0:
        y_start = data.index[0]

    # Create ot based on otLogical
    # Check if data is a time series (has frequency attribute)
    if hasattr(data, 'freq') or hasattr(data, 'index'):
        # For time series data, create time series ot
        freq_pd = getattr(data, 'freq', None) or getattr(data.index, 'freq', None)
        if hasattr(data.index[0], 'to_timestamp'):
            start = data.index[0].to_timestamp()
        else:
            start = data.index[0]
        
        # Ensure ot_logical is 1-dimensional before creating Series
        ot = pd.Series(ot_logical.ravel().astype(int), 
                      index=pd.date_range(start=start, periods=len(ot_logical), freq=freq_pd))
    else:
        # For non-time series data, create simple array
        ot = ot_logical.ravel().astype(int)

    # Get the index if available
    y_in_sample_index = None
    if hasattr(data, 'index'):
        y_in_sample_index = data.index[:-h] if holdout else data.index

    # Get forecast start based on index
    if holdout:
        y_forecast_start = data.index[obs_in_sample] if hasattr(data, 'index') else obs_in_sample + 1
    else:
        if hasattr(data, 'index'):
            # Get the last observation and add one time unit
            y_forecast_start = data.index[obs_in_sample-1] + (data.index[obs_in_sample-1] - data.index[obs_in_sample-2])
        else:
            y_forecast_start = obs_in_sample

    return {
        "ot_logical": ot_logical,
        "ot": ot,
        "frequency": freq,
        "y_start": y_start,
        "y_holdout": y_holdout,
        "y_in_sample_index": y_in_sample_index,
        "y_forecast_start": y_forecast_start
    }

def _adjust_model_for_sample_size(
    model_info,
    obs_nonzero,
    lags_model_max,
    allow_multiplicative=True,
    xreg_number=0,
    silent=False
):
    """
    Adjust model selection based on sample size, matching R's logic.
    """
    error_type = model_info["error_type"]
    trend_type = model_info["trend_type"]
    season_type = model_info["season_type"]
    model_do = model_info["model_do"]
    models_pool = model_info["models_pool"]
    
    n_param_exo = xreg_number * 2  # For both initial and persistence
    
    # If sample is too small for current model
    if obs_nonzero <= 3 + n_param_exo:
        if obs_nonzero == 3:
            if error_type in ["A", "M"]:
                model_do = "estimate"
                trend_type = season_type = "N"
            else:
                models_pool = ["ANN"]
                if allow_multiplicative:
                    models_pool.append("MNN")
                model_do = "select"
                error_type = trend_type = season_type = "N"
            
            return {
                "error_type": error_type,
                "trend_type": trend_type,
                "season_type": season_type,
                "model_do": model_do,
                "models_pool": models_pool,
                "persistence_estimate": False,
                "persistence_level": 0
            }
        
        elif obs_nonzero == 2:
            return {
                "error_type": "A",
                "trend_type": "N",
                "season_type": "N",
                "model_do": "use",
                "models_pool": None,
                "persistence_estimate": False,
                "persistence_level": 0,
                "initial_estimate": False
            }
        
        elif obs_nonzero == 1:
            return {
                "error_type": "A",
                "trend_type": "N",
                "season_type": "N",
                "model_do": "use",
                "models_pool": None,
                "persistence_estimate": False,
                "persistence_level": 0,
                "initial_estimate": False
            }
    
    # Handle larger but still limited samples
    if obs_nonzero <= 5 + n_param_exo:
        trend_type = "N"
        if len(model_info.get("model", "")) == 4:
            model = f"{error_type}N{season_type}"
        
    if obs_nonzero <= 2 * lags_model_max:
        season_type = "N"
        if models_pool:
            models_pool = [m for m in models_pool if m[-1] == "N"]
    
    return {
        "error_type": error_type,
        "trend_type": trend_type,
        "season_type": season_type,
        "model_do": model_do,
        "models_pool": models_pool
    }

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
    
    return {
        "parameters_number": parameters_number,
        "n_states": parameters_number[0][0],
        "n_params": parameters_number[1][0]
    }
