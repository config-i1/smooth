import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.special import gamma

# Note: adam_cpp instance is passed to functions that need C++ integration
# The adamCore object is created in architector() and passed through the pipeline
from smooth.adam_general.core.creator import adam_profile_creator, filler
from smooth.adam_general.core.utils.utils import scaler
from smooth.adam_general.core.utils.var_covar import sigma, covar_anal, var_anal, matrix_power_wrap
from smooth.adam_general.core.utils.distributions import generate_errors, normalize_errors

def _prepare_forecast_index(observations_dict, general_dict):
    """
    Prepare the index for the forecast Series/DataFrame.

    Parameters
    ----------
    observations_dict : dict
        Dictionary with observation data and related information
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    pandas.Index
        Index for the forecast
    """
    y_forecast_start = observations_dict["y_forecast_start"]
    h = general_dict["h"]
    freq = observations_dict["frequency"]

    # Check if y_forecast_start is a valid timestamp
    try:
        # Try to create a date_range with the start
        if isinstance(y_forecast_start, (pd.Timestamp, np.datetime64)):
            observations_dict["y_forecast_index"] = pd.date_range(
                start=y_forecast_start,
                periods=h,
                freq=freq
            )
        elif isinstance(y_forecast_start, (int, np.integer)):
            # Numeric index - use RangeIndex
            observations_dict["y_forecast_index"] = pd.RangeIndex(
                start=y_forecast_start,
                stop=y_forecast_start + h
            )
        else:
            # Try date_range anyway, may work for some types
            observations_dict["y_forecast_index"] = pd.date_range(
                start=y_forecast_start,
                periods=h,
                freq=freq
            )
    except (TypeError, ValueError):
        # Fallback to RangeIndex
        n_obs = len(observations_dict.get("y_in_sample", []))
        observations_dict["y_forecast_index"] = pd.RangeIndex(
            start=n_obs,
            stop=n_obs + h
        )

    return observations_dict["y_forecast_index"]


def _check_fitted_values(model_prepared, occurrence_dict):
    """
    Check fitted values for NaNs and adjust for occurrence if needed.

    Parameters
    ----------
    model_prepared : dict
        Dictionary with the prepared model including fitted values
    occurrence_dict : dict
        Dictionary with occurrence model parameters

    Returns
    -------
    dict
        Updated model_prepared dictionary
    """
    # Check for NaNs in fitted values
    if np.any(np.isnan(model_prepared["y_fitted"])) or np.any(
        pd.isna(model_prepared["y_fitted"])
    ):
        warnings.warn(
            "Something went wrong in the estimation of the model and NaNs were produced. "
            "If this is a mixed model, consider using the pure ones instead."
        )

    # Apply occurrence model to fitted values if present
    if occurrence_dict["occurrence_model"]:
        model_prepared["y_fitted"][:] = (
            model_prepared["y_fitted"] * occurrence_dict["p_fitted"]
        )

    # Fix cases when we have zeroes in the provided occurrence
    if occurrence_dict["occurrence"] == "provided":
        model_prepared["y_fitted"][~occurrence_dict["ot_logical"]] = (
            model_prepared["y_fitted"][~occurrence_dict["ot_logical"]]
            * occurrence_dict["p_fitted"][~occurrence_dict["ot_logical"]]
        )

    return model_prepared


def _initialize_forecast_series(observations_dict, general_dict):
    """
    Initialize a pandas Series for forecasts with appropriate index.

    Parameters
    ----------
    observations_dict : dict
        Dictionary with observation data and related information
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    pandas.Series
        Initialized forecast series with NaN values
    """
    if general_dict["h"] <= 0:
        return None

    if not isinstance(observations_dict.get("y_in_sample"), pd.Series):
        forecast_series = pd.Series(
            np.full(general_dict["h"], np.nan),
            index=pd.date_range(
                start=observations_dict["y_forecast_start"],
                periods=general_dict["h"],
                freq=observations_dict["frequency"],
            ),
        )
    else:
        forecast_series = pd.Series(
            np.full(general_dict["h"], np.nan),
            index=observations_dict["y_forecast_index"],
        )

    return forecast_series


def _prepare_lookup_table(lags_dict, observations_dict, general_dict):
    """
    Prepare lookup table for forecasting.

    Parameters
    ----------
    lags_dict : dict
        Dictionary with lag-related information
    observations_dict : dict
        Dictionary with observation data and related information
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    numpy.ndarray
        Lookup table for forecasting
    """
    lookup_result = adam_profile_creator(
        lags_model_all=lags_dict["lags_model_all"],
        lags_model_max=lags_dict["lags_model_max"],
        obs_all=observations_dict["obs_in_sample"] + general_dict["h"],
        lags=lags_dict["lags"],
    )
    lookup = lookup_result["index_lookup_table"][
        :, (observations_dict["obs_in_sample"] + lags_dict["lags_model_max"]) :
    ]
    return lookup


def _prepare_matrices_for_forecast(
    model_prepared, observations_dict, lags_dict, general_dict
):
    """
    Prepare matrices required for forecasting.

    Parameters
    ----------
    model_prepared : dict
        Dictionary with the prepared model including matrices
    observations_dict : dict
        Dictionary with observation data and related information
    lags_dict : dict
        Dictionary with lag-related information
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    tuple
        Tuple containing (mat_vt, mat_wt, vec_g, mat_f)
    """
    # Get state matrix
    mat_vt = model_prepared["states"][
        :,
        observations_dict["obs_states"]
        - lags_dict["lags_model_max"] : observations_dict["obs_states"]
        + 1,
    ]

    # Get measurement matrix
    if model_prepared["measurement"].shape[0] < general_dict["h"]:
        mat_wt = np.tile(model_prepared["measurement"][-1], (general_dict["h"], 1))
    else:
        mat_wt = model_prepared["measurement"][-general_dict["h"] :]

    # Get other matrices
    vec_g = model_prepared["persistence"]
    mat_f = model_prepared["transition"]

    return mat_vt, mat_wt, vec_g, mat_f


def _determine_forecast_interval(
    general_dict, model_type_dict, explanatory_checked, lags_dict
):
    """
    Determine the appropriate interval type for forecasting.

    Parameters
    ----------
    general_dict : dict
        Dictionary with general model parameters
    model_type_dict : dict
        Dictionary with model type information
    explanatory_checked : dict
        Dictionary with external regressors information
    lags_dict : dict
        Dictionary with lag-related information

    Returns
    -------
    dict
        Updated general_dict with interval type
    """
    # If this is "prediction", do simulations for multiplicative components
    if general_dict["interval"] == "prediction":
        # Simulate stuff for the ETS only
        if (
            model_type_dict["ets_model"] or explanatory_checked["xreg_number"] > 0
        ) and (
            model_type_dict["trend_type"] == "M"
            or (
                model_type_dict["season_type"] == "M"
                and general_dict["h"] > lags_dict["lags_model_min"]
            )
        ):
            general_dict["interval"] = "simulated"
        else:
            general_dict["interval"] = "approximate"

    return general_dict


def _generate_point_forecasts(
    observations_dict,
    lags_dict,
    model_prepared,
    lookup,
    model_type_dict,
    components_dict,
    explanatory_checked,
    constants_checked,
    general_dict,
    adam_cpp,
):
    """
    Generate point forecasts using adam_cpp.forecast().

    Parameters
    ----------
    observations_dict : dict
        Dictionary with observation data and related information
    lags_dict : dict
        Dictionary with lag-related information
    model_prepared : dict
        Dictionary with the prepared model
    lookup : numpy.ndarray
        Lookup table for forecasting
    model_type_dict : dict
        Dictionary with model type information
    components_dict : dict
        Dictionary with components information
    explanatory_checked : dict
        Dictionary with external regressors information
    constants_checked : dict
        Dictionary with information about constants
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    numpy.ndarray
        Array of point forecasts
    """
    # Get all the necessary matrices for forecasting
    mat_vt, mat_wt, vec_g, mat_f = _prepare_matrices_for_forecast(
        model_prepared, observations_dict, lags_dict, general_dict
    )

    # Prepare data for adam_forecaster
    lags_model_all = np.asfortranarray(
        lags_dict["lags_model_all"], dtype=np.uint64
    ).reshape(-1, 1)
    profiles_recent_table = np.asfortranarray(
        model_prepared["profiles_recent_table"], dtype=np.float64
    )
    index_lookup_table = np.asfortranarray(lookup, dtype=np.uint64)


    # Fix a bug I cant trace
    components_dict['components_number_ets_non_seasonal'] = components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal']

    # Call adam_cpp.forecast() with the prepared inputs
    # Note: E, T, S, nNonSeasonal, nSeasonal, nArima, nXreg, constant are set during adamCore construction
    forecast_result = adam_cpp.forecast(
        matrixWt=np.asfortranarray(mat_wt, dtype=np.float64),
        matrixF=np.asfortranarray(mat_f, dtype=np.float64),
        indexLookupTable=index_lookup_table,
        profilesRecent=profiles_recent_table,
        horizon=general_dict["h"],
    )
    y_forecast = forecast_result.forecast.flatten()

    return y_forecast


def _handle_forecast_safety_checks(
    y_forecast, model_type_dict, model_prepared, general_dict
):
    """
    Perform safety checks on forecasts and issue warnings if needed.

    Parameters
    ----------
    y_forecast : numpy.ndarray
        Array of point forecasts
    model_type_dict : dict
        Dictionary with model type information
    model_prepared : dict
        Dictionary with the prepared model
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    numpy.ndarray
        Corrected forecast array
    """
    # Replace NaN values with zeros
    if np.any(np.isnan(y_forecast)):
        y_forecast[np.isnan(y_forecast)] = 0

    # Issue warning about potentially explosive multiplicative trend
    # Make safety checks
    # If there are NaN values
    if np.any(np.isnan(y_forecast)):
        y_forecast[np.isnan(y_forecast)] = 0

    # Make a warning about the potential explosive trend
    if (model_type_dict["trend_type"] == "M" and not model_type_dict["damped"] and 
        model_prepared["profiles_recent_table"][1,0] > 1 and general_dict["h"] > 10):
        warnings.warn("Your model has a potentially explosive multiplicative trend. "
                    "I cannot do anything about it, so please just be careful.")


    return y_forecast


def _process_occurrence_forecast(occurrence_dict, general_dict):
    """
    Process occurrence forecasts for forecast horizon.

    Parameters
    ----------
    occurrence_dict : dict
        Dictionary with occurrence model parameters
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    numpy.ndarray
        Array of occurrence probabilities for forecast horizon
    """
    # Initialize occurrence model flag
    occurrence_model = False
    # If the occurrence values are provided for the holdout
    if occurrence_dict.get("occurrence") is not None and isinstance(occurrence_dict["occurrence"], bool):
        p_forecast = occurrence_dict["occurrence"] * 1
    elif occurrence_dict.get("occurrence") is not None and isinstance(occurrence_dict["occurrence"], (int, float)):
        p_forecast = occurrence_dict["occurrence"]
    else:
        # If this is a mixture model, produce forecasts for the occurrence
        if occurrence_dict.get("occurrence_model"):
            occurrence_model = True
            if occurrence_dict["occurrence"] == "provided":
                p_forecast = np.ones(general_dict["h"])
            else:
                # TODO: Implement forecast for occurrence model
                pass
        else:
            occurrence_model = False
            # If this was provided occurrence, then use provided values
            if (occurrence_dict.get("occurrence") is not None and 
                occurrence_dict.get("occurrence") == "provided" and
                occurrence_dict.get("p_forecast") is not None):
                p_forecast = occurrence_dict["p_forecast"]
            else:
                p_forecast = np.ones(general_dict["h"])


    # Make sure that the values are of the correct length
    if general_dict["h"] < len(p_forecast):
        p_forecast = p_forecast[:general_dict["h"]]
    elif general_dict["h"] > len(p_forecast):
        p_forecast = np.concatenate([p_forecast, np.repeat(p_forecast[-1], general_dict["h"] - len(p_forecast))])


    return p_forecast, occurrence_model


def _prepare_forecast_intervals(general_dict):
    """
    Prepare parameters for forecast intervals.

    Parameters
    ----------
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    tuple
        Tuple containing level information and side-specific interval levels
    """
    # Fix just in case user used 95 instead of 0.95
    level = general_dict["interval_level"]
    level = [
        level_value / 100 if level_value > 1 else level_value for level_value in level
    ]

    # Handle different interval sides
    if general_dict.get("side") == "both":
        level_low = round((1 - level[0]) / 2, 3)
        level_up = round((1 + level[0]) / 2, 3)
    elif general_dict.get("side") == "upper":
        level_low = None  # Not used for upper-side intervals
        level_up = level
    else:
        level_low = 1 - level
        level_up = None  # Not used for lower-side intervals

    return level, level_low, level_up


def _format_forecast_output(
    y_forecast, observations_dict, level_low, level_up, h_final=None
):
    """
    Format the forecast data into a pandas DataFrame with intervals.

    Parameters
    ----------
    y_forecast : numpy.ndarray
        Array of point forecasts
    observations_dict : dict
        Dictionary with observation data and related information
    level_low : float
        Lower level for prediction interval
    level_up : float
        Upper level for prediction interval
    h_final : int, optional
        Final forecast horizon (may differ from original if cumulative)

    Returns
    -------
    pandas.DataFrame
        DataFrame with forecasts and intervals
    """
    # Use the original horizon if h_final not provided
    if h_final is None:
        h_final = len(y_forecast) if hasattr(y_forecast, "__len__") else 1

    # Convert to dataframe with level_low and level_up as column names
    y_forecast_out = pd.DataFrame(
        {
            "mean": y_forecast,
            # Return NaN for intervals (would be calculated in a more complete implementation)
            f"lower_{level_low}": np.nan,
            f"upper_{level_up}": np.nan,
        },
        index=observations_dict["y_forecast_index"][:h_final],
    )

    return y_forecast_out


def forecaster(
    model_prepared,
    observations_dict,
    general_dict,
    occurrence_dict,
    lags_dict,
    model_type_dict,
    explanatory_checked,
    components_dict,
    constants_checked,
    params_info,
    adam_cpp,
    calculate_intervals,
    interval_method,
    level,
    side,
):
    """
    Generate point forecasts and prediction intervals from an estimated ADAM model.

    This function takes a prepared (fitted) ADAM model and produces forecasts for a
    specified horizon h. It can generate point forecasts, prediction intervals, or both.
    The forecasting process uses the **recursive multi-step ahead** approach where each
    forecast step updates the state vector for the next step.

    **Forecasting Process**:

    1. **Preparation**: Set up forecast index, check fitted values, validate horizon
    2. **Lookup Table**: Create index mapping for accessing lagged states
    3. **Point Forecasts**: Call C++ forecaster to generate h-step ahead predictions
       using the state-space recursion:

       .. math::

           \\hat{y}_{T+h} = w_{T+h}' v_{T+h|T}

           v_{T+h|T} = F v_{T+h-1|T}

    4. **Safety Checks**: Ensure forecasts are valid (no NaN, appropriate bounds)
    5. **Occurrence Adjustment**: Apply occurrence probabilities for intermittent data
    6. **Cumulative Forecasts**: Sum forecasts if cumulative=True (for demand totals)
    7. **Prediction Intervals**: Generate confidence bounds using parametric, simulation,
       or bootstrap methods

    **Prediction Interval Methods**:

    - **Parametric** (default): Analytical intervals based on assumed error distribution
      and state-space variance formulas. Fast and accurate for well-specified models.

    - **Simulation**: Monte Carlo simulation of future paths using estimated model and
      error distribution. More flexible but slower. Recommended for:

      * Multiplicative error models
      * Intermittent demand
      * Non-normal distributions

    - **Bootstrap**: Resampling residuals to generate intervals (not fully implemented yet)

    Parameters
    ----------
    model_prepared : dict
        Prepared model from ``preparator()`` containing:

        - **'states'**: State vector matrix (n_components × T), last columns are
          used as starting point for forecasting
        - **'y_fitted'**: Fitted values for in-sample period
        - **'residuals'**: Model residuals
        - **'mat_wt'**: Measurement matrix
        - **'mat_f'**: Transition matrix
        - **'vec_g'**: Persistence vector
        - **'persistence_level'**: Estimated α (level smoothing)
        - **'persistence_trend'**: Estimated β (if trendy)
        - **'persistence_seasonal'**: Estimated γ (if seasonal)
        - **'phi'**: Damping parameter (if damped trend)
        - **'scale'**: Estimated error scale parameter
        - **'initial_level'**, **'initial_trend'**, **'initial_seasonal'**: Initial states
        - **'arima_polynomials'**: AR/MA polynomials (if ARIMA)

    observations_dict : dict
        Observation information containing:

        - 'y_in_sample': In-sample observed values
        - 'obs_in_sample': Number of in-sample observations
        - 'y_forecast_start': Starting time for forecasts
        - 'y_forecast_index': Pandas index for forecast period
        - 'frequency': Time series frequency (for date indexing)

    general_dict : dict
        General configuration containing:

        - **'h'**: Forecast horizon (number of steps ahead)
        - 'distribution': Error distribution ('dnorm', 'dgamma', etc.)
        - 'cumulative': Whether to produce cumulative forecasts
        - 'nsim': Number of simulations (for simulation method, default 10000)
        - 'interval': Interval type ('parametric', 'simulation', 'bootstrap')

    occurrence_dict : dict
        Intermittent demand configuration containing:

        - 'occurrence_model': Whether occurrence model is active
        - 'p_fitted': Fitted occurrence probabilities (if occurrence model)
        - 'ot_logical': Boolean mask for non-zero observations

    lags_dict : dict
        Lag structure containing:

        - 'lags': Primary lag vector
        - 'lags_model': Lags for each state component
        - 'lags_model_all': Complete lag specification
        - 'lags_model_max': Maximum lag (lookback period)

    model_type_dict : dict
        Model type specification containing:

        - 'error_type': 'A' (additive) or 'M' (multiplicative)
        - 'trend_type': 'N', 'A', 'Ad', 'M', 'Md'
        - 'season_type': 'N', 'A', 'M'
        - 'ets_model': Whether ETS components exist
        - 'arima_model': Whether ARIMA components exist
        - 'model_is_trendy': Trend presence flag
        - 'model_is_seasonal': Seasonality presence flag

    explanatory_checked : dict
        External regressors specification containing:

        - 'xreg_model': Whether regressors are present
        - 'xreg_number': Number of regressors
        - 'xreg_data': Regressor values for forecast period (if available)

    components_dict : dict
        Component counts containing:

        - 'components_number_all': Total state dimension
        - 'components_number_ets': ETS component count
        - 'components_number_arima': ARIMA component count

    constants_checked : dict
        Constant term specification containing:

        - 'constant_required': Whether constant is included

    params_info : list or array
        Parameter count information:

        - params_info[0][0]: Number of states
        - params_info[1][0]: Number of parameters estimated

    calculate_intervals : bool
        Whether to calculate prediction intervals. If False, only point forecasts
        are returned (faster).

    interval_method : str
        Prediction interval method:

        - **"parametric"**: Analytical intervals (fast, assumes correct distribution)
        - **"simulation"**: Monte Carlo intervals (flexible, slower)
        - **"bootstrap"**: Bootstrap intervals (not fully implemented yet)

    level : float or list of float
        Confidence level(s) for prediction intervals.

        - Single value: e.g., 0.95 for 95% intervals
        - List: e.g., [0.80, 0.95] for 80% and 95% intervals (currently only first used)

        Standard values: 0.80 (80%), 0.90 (90%), 0.95 (95%), 0.99 (99%)

    side : str
        Which prediction interval bounds to compute:

        - **"both"**: Lower and upper bounds (default)
        - **"lower"**: Lower bound only
        - **"upper"**: Upper bound only

    Returns
    -------
    pandas.DataFrame
        DataFrame containing forecast results with columns:

        - **'mean'**: Point forecasts (always included)
        - **'lower_{level}'**: Lower prediction interval (if calculate_intervals=True)
        - **'upper_{level}'**: Upper prediction interval (if calculate_intervals=True)

        Index is the forecast period (y_forecast_index from observations_dict).

        Shape: (h, 1) or (h, 3) depending on calculate_intervals.

        If cumulative=True, returns shape (1, ...) with sum of forecasts.

    Notes
    -----
    **Recursive Forecasting**:

    The forecaster uses a recursive approach where:

    1. At h=1: Use final state vector from fitting
    2. At h=2: Update states using h=1 forecast
    3. Continue recursively through horizon h

    This naturally propagates uncertainty as h increases.

    **State Vector Evolution**:

    For each forecast step, the state vector evolves as:

    .. math::

        v_{T+h|T} = F v_{T+h-1|T} + g \\cdot 0

    The error term is zero for point forecasts (expected value).

    **Interval Width Growth**:

    Prediction intervals widen with forecast horizon due to:

    - Accumulation of forecast errors (one-step → multi-step)
    - Uncertainty in parameter estimates
    - Uncertainty in state values

    For multiplicative error models, intervals grow faster than additive models.

    **Occurrence Probabilities**:

    For intermittent demand, forecasts are adjusted by occurrence probability:

    .. math::

        E[y_{T+h}] = E[y_{T+h} | \\text{demand}] \\times P(\\text{demand})

    Intervals account for both demand size and demand probability uncertainties.

    **Cumulative Forecasts**:

    When cumulative=True, useful for inventory management:

    - Point forecast: Sum of individual forecasts
    - Intervals: Account for correlation between forecast errors
    - Occurrence: Uses simulation method automatically (complex distribution)

    **Performance**:

    - Point forecasts only: Very fast (~1ms for h=100)
    - Parametric intervals: Fast (~5-10ms)
    - Simulation intervals: Slower (100-500ms depending on nsim)

    **Zero and Negative Forecasts**:

    - Multiplicative models: Forecasts bounded below by small positive value
    - Intermittent data: Zero forecasts occur when occurrence probability < threshold
    - Negative forecasts: Possible with additive error if data trends strongly down

    See Also
    --------
    preparator : Prepares model for forecasting (called before forecaster)
    generate_prediction_interval : Parametric interval calculation
    generate_simulation_interval : Simulation-based interval calculation
    adam_forecaster : C++ backend for fast recursive forecasting

    Examples
    --------
    Generate point forecasts only::

        >>> forecast_df = forecaster(
        ...     model_prepared=prepared_model,
        ...     observations_dict={'y_in_sample': data, 'obs_in_sample': 100, ...},
        ...     general_dict={'h': 12, 'distribution': 'dnorm', ...},
        ...     calculate_intervals=False,
        ...     ...
        ... )
        >>> print(forecast_df['mean'])  # 12 point forecasts

    Generate 95% prediction intervals with parametric method::

        >>> forecast_df = forecaster(
        ...     model_prepared=prepared_model,
        ...     general_dict={'h': 12, 'interval': 'parametric', ...},
        ...     calculate_intervals=True,
        ...     interval_method='parametric',
        ...     level=0.95,
        ...     side='both',
        ...     ...
        ... )
        >>> print(forecast_df.columns)  # ['mean', 'lower_0.025', 'upper_0.975']

    Generate intervals using simulation for complex model::

        >>> forecast_df = forecaster(
        ...     model_prepared=prepared_model,
        ...     general_dict={'h': 24, 'interval': 'simulation', 'nsim': 10000, ...},
        ...     calculate_intervals=True,
        ...     interval_method='simulation',
        ...     level=0.90,
        ...     ...
        ... )
        >>> # More accurate for multiplicative/intermittent models

    Cumulative forecast for inventory planning::

        >>> forecast_df = forecaster(
        ...     general_dict={'h': 12, 'cumulative': True, ...},
        ...     calculate_intervals=True,
        ...     level=0.95,
        ...     ...
        ... )
        >>> total_demand = forecast_df.loc[0, 'mean']  # Sum of 12 periods
        >>> upper_bound = forecast_df.loc[0, 'upper_0.975']  # For safety stock
    """
    # 1. Prepare forecast index
    _prepare_forecast_index(observations_dict, general_dict)
    # 2. Check fitted values for issues and adjust for occurrence
    model_prepared = _check_fitted_values(model_prepared, occurrence_dict)

    # 3. Return empty result if horizon is zero
    if general_dict["h"] <= 0:
        return None

    # 4. Initialize forecast series structure (but we'll use numpy arrays for calculations)
    # We don't need to store this value since we're only using numpy arrays internally
    _initialize_forecast_series(observations_dict, general_dict)

    # 5. Prepare lookup table for forecasting
    lookup = _prepare_lookup_table(lags_dict, observations_dict, general_dict)

    # 6. Determine the appropriate interval type
    general_dict = _determine_forecast_interval(
        general_dict, model_type_dict, explanatory_checked, lags_dict
    )

    # 7. Generate point forecasts using adam_cpp.forecast()
    y_forecast_values = _generate_point_forecasts(
        observations_dict,
        lags_dict,
        model_prepared,
        lookup,
        model_type_dict,
        components_dict,
        explanatory_checked,
        constants_checked,
        general_dict,
        adam_cpp,
    )

    # 8. Perform safety checks on forecasts
    y_forecast_values = _handle_forecast_safety_checks(
        y_forecast_values, model_type_dict, model_prepared, general_dict
    )

    # 9. Process occurrence forecasts
    p_forecast, occurrence_model = _process_occurrence_forecast(
        occurrence_dict, general_dict
    )

    # 10. Apply occurrence probabilities to forecasts
    y_forecast_values = y_forecast_values * p_forecast
    # 11. Handle cumulative forecasts if specified
    h_final = general_dict["h"]
    if general_dict.get("cumulative"):
        y_forecast_values = np.sum(y_forecast_values)
        h_final = 1
        # In case of occurrence model use simulations - the cumulative probability is complex
        if occurrence_model:
            general_dict["interval"] = "simulated"

    # 12. Prepare interval parameters
    if calculate_intervals:
        # assert method is parametric, bootstrap or simulation
        assert interval_method in ["parametric", "simulation", "bootstrap"], "Interval method must be either parametric, simulation, or bootstrap"

        if level is None:
            warnings.warn("No confidence level specified. Using default level of 0.95")
            level = 0.95

        # Ensure level is a scalar for now
        if isinstance(level, list):
            level = level[0]

        level_low, level_up = ensure_level_format(level, side)

        # Route to appropriate interval method
        if interval_method == "simulation":
            nsim = general_dict.get("nsim", 10000)
            y_lower, y_upper = generate_simulation_interval(
                y_forecast_values,
                model_prepared,
                general_dict,
                observations_dict,
                model_type_dict,
                lags_dict,
                components_dict,
                explanatory_checked,
                constants_checked,
                params_info,
                adam_cpp,
                level,
                nsim=nsim
            )
        elif interval_method == "bootstrap":
            warnings.warn("Bootstrap intervals not yet supported. Using parametric.")
            y_lower, y_upper = generate_prediction_interval(
                y_forecast_values, model_prepared, general_dict,
                observations_dict, model_type_dict, lags_dict, params_info, level
            )
        else:  # parametric (default)
            y_lower, y_upper = generate_prediction_interval(
                y_forecast_values, model_prepared, general_dict,
                observations_dict, model_type_dict, lags_dict, params_info, level
            )
    
        y_forecast_out = pd.DataFrame({
            'mean': y_forecast_values,
            f'lower_{level_low}': y_lower,  # Return 0 regardless of calculations
            f'upper_{level_up}': y_upper    # Return 0 regardless of calculations
        }, index=observations_dict["y_forecast_index"])
    else:
        y_forecast_out = pd.DataFrame({
            'mean': y_forecast_values,
        }, index=observations_dict["y_forecast_index"])
        

    return y_forecast_out


def _fill_matrices_if_needed(
    general_dict,
    matrices_dict,
    adam_estimated,
    model_type_dict,
    components_dict,
    lags_dict,
    persistence_checked,
    initials_checked,
    arima_checked,
    explanatory_checked,
    phi_dict,
    constants_checked,
    adam_cpp=None,
):
    """
    Fill matrices with estimated parameters if needed.

    Parameters
    ----------
    general_dict : dict
        Dictionary with general model parameters
    matrices_dict : dict
        Dictionary with model matrices
    adam_estimated : dict
        Dictionary with estimated parameters
    model_type_dict : dict
        Dictionary with model type information
    components_dict : dict
        Dictionary with model components information
    lags_dict : dict
        Dictionary with lag-related information
    persistence_checked : dict
        Dictionary with persistence parameters
    initials_checked : dict
        Dictionary with initial values
    arima_checked : dict
        Dictionary with ARIMA model parameters
    explanatory_checked : dict
        Dictionary with external regressors information
    phi_dict : dict
        Dictionary with damping parameter information
    constants_checked : dict
        Dictionary with information about constants in the model

    Returns
    -------
    dict
        Updated matrices dictionary
    """
    if general_dict.get("model_do") != "use":
        matrices_dict = filler(
            adam_estimated["B"],
            model_type_dict=model_type_dict,
            components_dict=components_dict,
            lags_dict=lags_dict,
            matrices_dict=matrices_dict,
            persistence_checked=persistence_checked,
            initials_checked=initials_checked,
            arima_checked=arima_checked,
            explanatory_checked=explanatory_checked,
            phi_dict=phi_dict,
            constants_checked=constants_checked,
        )

    return matrices_dict


def _prepare_profiles_recent_table(matrices_dict, lags_dict):
    """
    Prepare the recent profiles table for forecasting.

    Parameters
    ----------
    matrices_dict : dict
        Dictionary with model matrices
    lags_dict : dict
        Dictionary with lag-related information

    Returns
    -------
    numpy.ndarray
        Recent profiles table
    numpy.ndarray
        Initial recent profiles table
    """
    profiles_recent_table = matrices_dict["mat_vt"][:, : lags_dict["lags_model_max"]]
    profiles_recent_initial = matrices_dict["mat_vt"][
        :, : lags_dict["lags_model_max"]
    ].copy()

    return profiles_recent_table, profiles_recent_initial


def _prepare_fitter_inputs(
    observations_dict, matrices_dict, lags_dict, profiles_dict, profiles_recent_table
):
    """
    Prepare inputs for the adam_fitter function.

    Parameters
    ----------
    observations_dict : dict
        Dictionary with observation data and related information
    matrices_dict : dict
        Dictionary with model matrices
    lags_dict : dict
        Dictionary with lag-related information
    profiles_dict : dict
        Dictionary with profile information
    profiles_recent_table : numpy.ndarray
        Recent profiles table

    Returns
    -------
    tuple
        Tuple containing arrays prepared for adam_fitter
    """
    # Convert pandas Series/DataFrames to numpy arrays
    y_in_sample = np.asarray(
        observations_dict["y_in_sample"].flatten(), dtype=np.float64
    )
    ot = np.asarray(observations_dict["ot"].flatten(), dtype=np.float64)
    mat_vt = np.asfortranarray(matrices_dict["mat_vt"], dtype=np.float64)
    mat_wt = np.asfortranarray(matrices_dict["mat_wt"], dtype=np.float64)
    mat_f = np.asfortranarray(matrices_dict["mat_f"], dtype=np.float64)
    vec_g = np.asfortranarray(matrices_dict["vec_g"], dtype=np.float64)
    lags_model_all = np.asfortranarray(
        lags_dict["lags_model_all"], dtype=np.uint64
    ).reshape(-1, 1)
    index_lookup_table = np.asfortranarray(
        profiles_dict["index_lookup_table"], dtype=np.uint64
    )
    profiles_recent_table = np.asfortranarray(profiles_recent_table, dtype=np.float64)

    return (
        y_in_sample,
        ot,
        mat_vt,
        mat_wt,
        mat_f,
        vec_g,
        lags_model_all,
        index_lookup_table,
        profiles_recent_table,
    )


def _correct_multiplicative_components(
    matrices_dict, profiles_dict, model_type_dict, components_dict
):
    """
    Correct negative or NaN values in multiplicative components.

    Parameters
    ----------
    matrices_dict : dict
        Dictionary with model matrices
    profiles_dict : dict
        Dictionary with profile information
    model_type_dict : dict
        Dictionary with model type information
    components_dict : dict
        Dictionary with model components information

    Returns
    -------
    dict
        Updated matrices dictionary
    dict
        Updated profiles dictionary
    """

    # Kind of complex here. Sorry people.
    # Thanks for the nice heuristics Ivan!
    
    if model_type_dict["trend_type"] == "M" and (np.any(np.isnan(matrices_dict['mat_vt'][1,:])) or np.any(matrices_dict['mat_vt'][1,:] <= 0)):
            i = np.where(matrices_dict['mat_vt'][1,:] <= 0)[0]
            matrices_dict['mat_vt'][1,i] = 1e-6
            profiles_dict["profiles_recent_table"][1,i] = 1e-6 
    if model_type_dict["season_type"] == "M" and np.all(~np.isnan(matrices_dict['mat_vt'][components_dict["components_number_ets_non_seasonal"]:components_dict["components_number_ets_non_seasonal"]+components_dict["components_number_ets_seasonal"],:])) and \
            np.any(matrices_dict['mat_vt'][components_dict["components_number_ets_non_seasonal"]:components_dict["components_number_ets_non_seasonal"]+components_dict["components_number_ets_seasonal"],:] <= 0):
            i = np.where(matrices_dict['mat_vt'][components_dict["components_number_ets_non_seasonal"]:components_dict["components_number_ets_non_seasonal"]+components_dict["components_number_ets_seasonal"],:] <= 0)[0]
            matrices_dict['mat_vt'][components_dict["components_number_ets_non_seasonal"]:components_dict["components_number_ets_non_seasonal"]+components_dict["components_number_ets_seasonal"],i] = 1e-6
            i = np.where(profiles_dict["profiles_recent_table"][components_dict["components_number_ets_non_seasonal"]:components_dict["components_number_ets_non_seasonal"]+components_dict["components_number_ets_seasonal"],:] <= 0)[0]
            profiles_dict["profiles_recent_table"][components_dict["components_number_ets_non_seasonal"]:components_dict["components_number_ets_non_seasonal"]+components_dict["components_number_ets_seasonal"],i] = 1e-6


    return matrices_dict, profiles_dict


def _initialize_fitted_series(observations_dict):
    """
    Initialize pandas Series for fitted values and errors.

    Parameters
    ----------
    observations_dict : dict
        Dictionary with observation data and related information

    Returns
    -------
    tuple
        Tuple of (y_fitted, errors) pandas Series
    """
    if not isinstance(observations_dict["y_in_sample"], pd.Series):
            y_fitted = pd.Series(np.full(observations_dict["obs_in_sample"], np.nan), 
                            index=pd.date_range(start=observations_dict["y_start"], 
                                            periods=observations_dict["obs_in_sample"], 
                                            freq=observations_dict["frequency"]))
            errors = pd.Series(np.full(observations_dict["obs_in_sample"], np.nan), 
                            index=pd.date_range(start=observations_dict["y_start"], 
                                            periods=observations_dict["obs_in_sample"], 
                                            freq=observations_dict["frequency"]))
    else:
            y_fitted = pd.Series(np.full(observations_dict["obs_in_sample"], np.nan), index=observations_dict["y_in_sample_index"])
            errors = pd.Series(np.full(observations_dict["obs_in_sample"], np.nan), index=observations_dict["y_in_sample_index"])

    return y_fitted, errors


def _update_distribution(general_dict, model_type_dict):
    """
    Update distribution based on error type and loss function.

    Parameters
    ----------
    general_dict : dict
        Dictionary with general model parameters
    model_type_dict : dict
        Dictionary with model type information

    Returns
    -------
    dict
        Updated general dictionary
    """
    if general_dict["distribution"] == "default":
        if general_dict["loss"] == "likelihood":
            if model_type_dict["error_type"] == "A":
                general_dict["distribution"] = "dnorm"
            elif model_type_dict["error_type"] == "M":
                general_dict["distribution"] = "dgamma"
        elif general_dict["loss"] in ["MAEh", "MACE", "MAE"]:
            general_dict["distribution"] = "dlaplace"
        elif general_dict["loss"] in ["HAMh", "CHAM", "HAM"]:
            general_dict["distribution"] = "ds"
        elif general_dict["loss"] in ["MSEh", "MSCE", "MSE", "GPL"]:
            general_dict["distribution"] = "dnorm"
        else:
            general_dict["distribution"] = "dnorm"

    return general_dict


def _process_initial_values(
    model_type_dict,
    lags_dict,
    matrices_dict,
    components_dict,
    arima_checked,
    explanatory_checked,
    initials_checked,
):
    """
    Process and organize initial values.

    Parameters
    ----------
    model_type_dict : dict
        Dictionary with model type information
    lags_dict : dict
        Dictionary with lag-related information
    matrices_dict : dict
        Dictionary with model matrices
    components_dict : dict
        Dictionary with model components information
    arima_checked : dict
        Dictionary with ARIMA model parameters
    explanatory_checked : dict
        Dictionary with external regressors information
    initials_checked : dict
        Dictionary with initial values

    Returns
    -------
    tuple
        Tuple of (initial_value, initial_value_names, initial_estimated)
    """
    # Initial values to return
    initial_value = [None] * (model_type_dict["ets_model"] * (1 + model_type_dict["model_is_trendy"] + model_type_dict["model_is_seasonal"]) + 
                             arima_checked["arima_model"] + explanatory_checked["xreg_model"])
    initial_value_ets = [None] * (model_type_dict["ets_model"] * len(lags_dict["lags_model"]))
    initial_value_names = [""] * (model_type_dict["ets_model"] * (1 + model_type_dict["model_is_trendy"] + model_type_dict["model_is_seasonal"]) + 
                                 arima_checked["arima_model"] + explanatory_checked["xreg_model"])
    
    # The vector that defines what was estimated in the model
    initial_estimated = [False] * (model_type_dict["ets_model"] * (1 + model_type_dict["model_is_trendy"] + model_type_dict["model_is_seasonal"] * components_dict["components_number_ets_seasonal"]) + 
                                 arima_checked["arima_model"] + explanatory_checked["xreg_model"])
   
    # Write down the initials of ETS
    j = 0
    if model_type_dict["ets_model"]:
        # Write down level, trend and seasonal
        for i in range(len(lags_dict["lags_model"])):
            # In case of level / trend, we want to get the very first value
            if lags_dict["lags_model"][i] == 1:
                initial_value_ets[i] = matrices_dict['mat_vt'][i, :lags_dict["lags_model_max"]][0]
            # In cases of seasonal components, they should be at the end of the pre-heat period
            else:
                #print(lags_dict["lags_model"][i][0]) # here we might have an issue for taking the first element of the list
                start_idx = lags_dict["lags_model_max"] - lags_dict["lags_model"][i]
                initial_value_ets[i] = matrices_dict['mat_vt'][i, start_idx:lags_dict["lags_model_max"]]
        
        j = 0
        # Write down level in the final list
        initial_estimated[j] = initials_checked["initial_level_estimate"]
        initial_value[j] = initial_value_ets[j]
        initial_value_names[j] = "level"
        
        if model_type_dict["model_is_trendy"]:
            j = 1
            initial_estimated[j] = initials_checked["initial_trend_estimate"]
            # Write down trend in the final list
            initial_value[j] = initial_value_ets[j]
            # Remove the trend from ETS list
            initial_value_ets[j] = None
            initial_value_names[j] = "trend"

        # Write down the initial seasonals
        if model_type_dict["model_is_seasonal"]:
            # Convert initial_seasonal_estimate to list if it's a boolean (for single seasonality)
            if isinstance(initials_checked['initial_seasonal_estimate'], bool):
                seasonal_estimate_list = [initials_checked['initial_seasonal_estimate']] * components_dict['components_number_ets_seasonal']
            else:
                seasonal_estimate_list = initials_checked['initial_seasonal_estimate']

            initial_estimated[j + 1:j + 1 + components_dict["components_number_ets_seasonal"]] = seasonal_estimate_list
            # Remove the level from ETS list
            initial_value_ets[0] = None
            j += 1
            if len(seasonal_estimate_list) > 1:
                initial_value[j] = [x for x in initial_value_ets if x is not None]
                initial_value_names[j] = "seasonal"
                for k in range(components_dict["components_number_ets_seasonal"]):
                    initial_estimated[j + k] = f"seasonal{k+1}"
            else:
                initial_value[j] = next(x for x in initial_value_ets if x is not None)
                initial_value_names[j] = "seasonal"
                initial_estimated[j] = "seasonal"

    # Write down the ARIMA initials
    if arima_checked["arima_model"]:
        j += 1
        initial_estimated[j] = initials_checked["initial_arima_estimate"]
        if initials_checked["initial_arima_estimate"]:
            initial_value[j] = matrices_dict['mat_vt'][components_dict["components_number_ets"] + components_dict.get("components_number_arima", 0) - 1, :initials_checked["initial_arima_number"]]
        else:
            initial_value[j] = initials_checked["initial_arima"]
        initial_value_names[j] = "arima"
        initial_estimated[j] = "arima"

    # Set names for initial values
    initial_value = {name: value for name, value in zip(initial_value_names, initial_value)}


    return initial_value, initial_value_names, initial_estimated


def _process_arma_parameters(arima_checked, adam_estimated):
    """
    Process ARMA parameters from estimates.

    Parameters
    ----------
    arima_checked : dict
        Dictionary with ARIMA model parameters
    adam_estimated : dict
        Dictionary with estimated parameters

    Returns
    -------
    dict or None
        Dictionary of AR and MA parameters or None if no ARIMA model
    """
    if arima_checked["arima_model"]:
            arma_parameters_list = {}
            j = 0
            if arima_checked["ar_required"] and arima_checked["ar_estimate"]:
                # Avoid damping parameter phi by checking name length > 3
                arma_parameters_list["ar"] = [b for name, b in B.items() if len(name) > 3 and name.startswith("phi")]
                j += 1
            elif arima_checked["ar_required"] and not arima_checked["ar_estimate"]:
                # Avoid damping parameter phi
                arma_parameters_list["ar"] = [p for name, p in arima_checked["arma_parameters"].items() if name.startswith("phi")]
                j += 1
            
            if arima_checked["ma_required"] and arima_checked["ma_estimate"]:
                arma_parameters_list["ma"] = [b for name, b in B.items() if name.startswith("theta")]
            elif arima_checked["ma_required"] and not arima_checked["ma_estimate"]:
                arma_parameters_list["ma"] = [p for name, p in arima_checked["arma_parameters"].items() if name.startswith("theta")]
    else:
        arma_parameters_list = None

    return arma_parameters_list


def _calculate_scale_parameter(
    general_dict, model_type_dict, errors, y_fitted, observations_dict, other
):
    """
    Calculate scale parameter using scaler function.

    Parameters
    ----------
    general_dict : dict
        Dictionary with general model parameters
    model_type_dict : dict
        Dictionary with model type information
    errors : pandas.Series
        Series with model errors
    y_fitted : pandas.Series
        Series with fitted values
    observations_dict : dict
        Dictionary with observation data and related information
    other : dict
        Additional parameters

    Returns
    -------
    float
        Calculated scale parameter
    """
    scale = scaler(
        general_dict["distribution_new"],
        model_type_dict["error_type"],
        errors[observations_dict["ot_logical"]],
        y_fitted[observations_dict["ot_logical"]],
        observations_dict["obs_in_sample"],
        other,
    )

    return scale


def _process_other_parameters(
    constants_checked, adam_estimated, general_dict, arima_checked, lags_dict=None
):
    """
    Process additional parameters like constants and ARIMA polynomials.

    Parameters
    ----------
    constants_checked : dict
        Dictionary with information about constants
    adam_estimated : dict
        Dictionary with estimated parameters
    general_dict : dict
        Dictionary with general model parameters
    arima_checked : dict
        Dictionary with ARIMA model parameters
    lags_dict : dict, optional
        Dictionary with lag-related information

    Returns
    -------
    tuple
        Tuple of (constant_value, other_returned)
    """
    # Get constant value
    # If constant is being estimated, get it from B (last element when estimated)
    # If not estimated but required, get the fixed value from constants_checked
    if constants_checked["constant_estimate"]:
        if len(adam_estimated['B']) > 0:
            constant_value = adam_estimated['B'][-1]
        else:
            constant_value = 0  # Default when no parameters estimated
    elif constants_checked["constant_required"]:
        constant_value = constants_checked.get("constant_value", 0)
    else:
        constant_value = 0

    # Initialize other parameters dictionary
    other_returned = {}

    # Add LASSO/RIDGE lambda if applicable
    if general_dict["loss"] in ["LASSO", "RIDGE"]:
        other_returned["lambda"] = general_dict["lambda_"]

    # Add ARIMA polynomials if applicable
    if arima_checked["arima_model"]:
        other_returned["polynomial"] = adam_estimated["arima_polynomials"]
        other_returned["ARIMA_indices"] = {
            "nonZeroARI": arima_checked["non_zero_ari"],
            "nonZeroMA": arima_checked["non_zero_ma"],
        }

        # Create AR polynomial matrix
        if lags_dict is not None:
            ar_matrix_size = sum(arima_checked["ar_orders"]) * lags_dict["lags"]
            other_returned["ar_polynomial_matrix"] = np.zeros(
                (ar_matrix_size, ar_matrix_size)
            )

            if other_returned["ar_polynomial_matrix"].shape[0] > 1:
                # Set diagonal elements to 1 except first row/col
                other_returned["ar_polynomial_matrix"][1:-1, 2:] = np.eye(
                    other_returned["ar_polynomial_matrix"].shape[0] - 2
                )

                if arima_checked["ar_required"]:
                    other_returned["ar_polynomial_matrix"][:, 0] = -adam_estimated[
                        "arima_polynomials"
                    ]["ar_polynomial"][1:]

        other_returned["arma_parameters"] = arima_checked["arma_parameters"]

    return constant_value, other_returned


def preparator(
    # Model type info
    model_type_dict,
    # Components info
    components_dict,
    # Lags info
    lags_dict,
    # Matrices from creator
    matrices_dict,
    # Parameter dictionaries
    persistence_checked,
    initials_checked,
    arima_checked,
    explanatory_checked,
    phi_dict,
    constants_checked,
    # Other parameters
    observations_dict,
    occurrence_dict,
    general_dict,
    profiles_dict,
    # The parameter vector
    adam_estimated,
    # adamCore C++ object
    adam_cpp,
    # Optional parameters
    bounds="usual",
    other=None,
):
    """
    Prepare estimated ADAM model for forecasting by computing in-sample fit and states.

    This is the **bridge function** between model estimation and forecasting. After
    parameters are optimized by ``estimator()``, ``preparator()`` fills the state-space
    matrices with the estimated parameters, runs the model forward through the in-sample
    period to generate fitted values and final states, and packages everything needed
    for ``forecaster()`` to produce out-of-sample predictions.

    **Preparation Process**:

    1. **Matrix Filling**: If parameters were estimated (not fixed), call ``filler()`` to
       populate mat_vt, mat_wt, mat_f, vec_g with values from optimized parameter vector B

    2. **Profile Setup**: Prepare profile matrices for time-varying parameters (advanced
       feature, typically zeros for standard models)

    3. **Array Preparation**: Convert all inputs to proper numpy arrays with correct shapes
       and data types (Fortran-order for C++ compatibility)

    4. **Model Fitting**: Call C++ ``adam_fitter()`` to run the model forward through
       in-sample data, updating states and computing fitted values:

       .. math::

           y_t^{\\text{fitted}} = w_t' v_{t-l}

           v_t = F v_{t-l} + g \\epsilon_t

       where :math:`\\epsilon_t = (y_t - y_t^{\\text{fitted}}) / r_t`

    5. **Results Packaging**: Extract and organize:

       - Final state vector (for starting forecasts)
       - Fitted values
       - Residuals
       - Scale parameter
       - All estimated parameters (α, β, γ, φ, initials, AR/MA, etc.)

    **Outputs Used by Forecaster**:

    The prepared model dict contains everything ``forecaster()`` needs:

    - **States**: Final values to initialize forecasting
    - **Matrices**: mat_wt, mat_f, vec_g for state-space recursion
    - **Parameters**: For interval calculation (scale, smoothing params)
    - **Fitted values**: For diagnostics and residual analysis

    Parameters
    ----------
    model_type_dict : dict
        Model type specification containing:

        - 'ets_model': Whether ETS components exist
        - 'arima_model': Whether ARIMA components exist
        - 'error_type': 'A' or 'M'
        - 'trend_type': 'N', 'A', 'Ad', 'M', 'Md'
        - 'season_type': 'N', 'A', 'M'
        - 'model_is_trendy': Trend presence flag
        - 'model_is_seasonal': Seasonality presence flag

    components_dict : dict
        Component counts containing:

        - 'components_number_all': Total state dimension
        - 'components_number_ets': ETS component count
        - 'components_number_arima': ARIMA component count

    lags_dict : dict
        Lag structure containing:

        - 'lags': Primary lag vector
        - 'lags_model': Lags for each state component
        - 'lags_model_all': Complete lag specification
        - 'lags_model_max': Maximum lag (lookback period)

    matrices_dict : dict
        State-space matrices from ``creator()`` containing:

        - 'mat_vt': State vector (may have initial parameters or backcasted values)
        - 'mat_wt': Measurement matrix (may have damping placeholders)
        - 'mat_f': Transition matrix
        - 'vec_g': Persistence vector (may have smoothing parameter placeholders)

        These matrices are updated in-place if parameters were estimated.

    persistence_checked : dict
        Persistence specification containing:

        - 'persistence_estimate': Whether smoothing parameters were estimated
        - 'persistence_level_estimate': Whether α was estimated
        - 'persistence_trend_estimate': Whether β was estimated
        - 'persistence_seasonal_estimate': List of flags for γ estimation
        - Fixed values for non-estimated parameters

    initials_checked : dict
        Initial states specification containing:

        - 'initial_type': Initialization method used
        - 'initial_level_estimate': Whether l₀ was estimated
        - 'initial_trend_estimate': Whether b₀ was estimated
        - 'initial_seasonal_estimate': List of flags for s₀ estimation

    arima_checked : dict
        ARIMA specification containing:

        - 'arima_model': Whether ARIMA components exist
        - 'ar_estimate': Whether AR coefficients were estimated
        - 'ma_estimate': Whether MA coefficients were estimated
        - 'ar_orders': AR orders
        - 'ma_orders': MA orders

    explanatory_checked : dict
        External regressors specification containing:

        - 'xreg_model': Whether regressors exist
        - 'xreg_number': Number of regressors
        - 'mat_xt': Regressor data matrix

    phi_dict : dict
        Damping specification containing:

        - 'phi_estimate': Whether φ was estimated
        - 'phi': Damping value (estimated or fixed)

    constants_checked : dict
        Constant term specification containing:

        - 'constant_required': Whether constant is included
        - 'constant_estimate': Whether constant was estimated

    observations_dict : dict
        Observation information containing:

        - 'y_in_sample': Time series data
        - 'obs_in_sample': Number of observations
        - 'ot': Occurrence vector (1 for non-zero observations)

    occurrence_dict : dict
        Intermittent demand specification containing:

        - 'occurrence_model': Whether occurrence model is active
        - 'p_fitted': Fitted occurrence probabilities

    general_dict : dict
        General configuration containing:

        - 'distribution': Error distribution
        - 'loss': Loss function used in estimation

    profiles_dict : dict
        Profile matrices for time-varying parameters containing:

        - 'profiles_recent_table': Recent profile values
        - 'index_lookup_table': Index mapping for profile access

    adam_estimated : dict
        Estimation results from ``estimator()`` containing:

        - **'B'**: Optimized parameter vector
        - 'CF_value': Final cost function value
        - 'n_param_estimated': Number of estimated parameters
        - 'log_lik_adam_value': Log-likelihood information
        - 'arima_polynomials': AR/MA polynomials (if ARIMA)

    bounds : str, default="usual"
        Bound type used in estimation ('usual', 'admissible', 'none').
        Currently unused in preparator but kept for compatibility.

    other : float or None, default=None
        Additional distribution parameter (for certain distributions).
        Currently unused in preparator.

    Returns
    -------
    dict
        Prepared model dictionary containing:

        - **'states'** (numpy.ndarray): State vector matrix, shape (n_components, T+max_lag).
          Final columns (at T) are starting point for forecasting.

        - **'y_fitted'** (numpy.ndarray): In-sample fitted values, shape (T,)

        - **'residuals'** (numpy.ndarray): In-sample residuals, shape (T,)

        - **'mat_wt'** (numpy.ndarray): Measurement matrix (for forecasting)

        - **'mat_f'** (numpy.ndarray): Transition matrix (for forecasting)

        - **'vec_g'** (numpy.ndarray): Persistence vector (for forecasting)

        - **'scale'** (float): Error scale parameter (standard deviation for additive,
          scale for multiplicative)

        - **'persistence_level'** (float): Estimated α (if applicable)

        - **'persistence_trend'** (float): Estimated β (if trendy)

        - **'persistence_seasonal'** (list): Estimated γ values (if seasonal)

        - **'phi'** (float): Damping parameter (if damped trend)

        - **'initial_level'** (float): Level initial state

        - **'initial_trend'** (float): Trend initial state (if trendy)

        - **'initial_seasonal'** (list): Seasonal initial states (if seasonal)

        - **'ar_parameters'** (list): AR coefficients (if ARIMA)

        - **'ma_parameters'** (list): MA coefficients (if ARIMA)

        - **'xreg_parameters'** (list): Regression coefficients (if regressors)

        - **'constant'** (float): Constant term (if included)

        - **'arima_polynomials'** (dict): AR/MA polynomial matrices (if ARIMA)

        - **'loglik'** (float): Log-likelihood value

        - **'n_param'** (int): Number of estimated parameters

    Notes
    -----
    **Matrix Ordering**:

    All matrices use **Fortran order** (column-major) for C++ compatibility. Do not
    change to C-order as it will cause incorrect results in adam_fitter.

    **Fitted Values vs Residuals**:

    - **Fitted values**: One-step-ahead predictions using actual past observations
    - **Residuals**: y_t - y_fitted_t (not scaled)
    - **Scaled residuals**: ε_t = residuals_t / scale

    For multiplicative models, residuals are relative errors.

    **Initial States in Output**:

    The initial states returned (initial_level, initial_trend, etc.) are:

    - Values at time t=0 (before first observation)
    - Either estimated, backcasted, or user-provided depending on initial_type
    - Extracted from first max_lag columns of mat_vt

    **When is Filler Called?**:

    ``filler()`` is called only if parameters were actually estimated. If all parameters
    were fixed (e.g., using a previously estimated model), matrices already contain
    correct values and filler is skipped.

    **ARIMA Polynomials**:

    For ARIMA models, the arima_polynomials dict contains companion matrix representations
    of AR and MA polynomials, used for state-space forecasting.

    **Performance**:

    The C++ adam_fitter is very fast (~1-5ms for T=1000 observations). The preparator
    overhead is minimal.

    See Also
    --------
    estimator : Calls preparator after optimization to get final fitted model
    forecaster : Uses prepared model to generate forecasts
    filler : Fills matrices with parameter values (called by preparator if needed)
    adam_fitter : C++ backend for computing fitted values and states

    Examples
    --------
    Prepare model after estimation::

        >>> prepared = preparator(
        ...     model_type_dict={'ets_model': True, 'arima_model': False, ...},
        ...     components_dict={'components_number_all': 13, ...},
        ...     lags_dict={'lags': np.array([1, 12]), ...},
        ...     matrices_dict={'mat_vt': mat_vt, 'mat_wt': mat_wt, ...},
        ...     persistence_checked={'persistence_estimate': True, ...},
        ...     initials_checked={'initial_type': 'optimal', ...},
        ...     observations_dict={'y_in_sample': data, 'obs_in_sample': 100, ...},
        ...     adam_estimated={'B': optimized_params, 'log_lik_adam_value': {...}, ...},
        ...     ...
        ... )
        >>> print(prepared['y_fitted'])  # In-sample fitted values
        >>> print(prepared['states'][:, -1])  # Final state vector for forecasting
        >>> print(prepared['scale'])  # Error scale for prediction intervals

    Extract estimated parameters::

        >>> alpha = prepared['persistence_level']
        >>> beta = prepared['persistence_trend']
        >>> l0 = prepared['initial_level']
        >>> print(f"Smoothing: α={alpha:.3f}, β={beta:.3f}, Initial level: {l0:.2f}")

    Use prepared model for forecasting::

        >>> forecasts = forecaster(
        ...     model_prepared=prepared,  # Pass prepared model
        ...     observations_dict=obs_dict,
        ...     general_dict={'h': 12, ...},
        ...     ...
        ... )
    """
    # 1. Fill matrices with estimated parameters if needed
    matrices_dict = _fill_matrices_if_needed(
        general_dict,
        matrices_dict,
        adam_estimated,
        model_type_dict,
        components_dict,
        lags_dict,
        persistence_checked,
        initials_checked,
        arima_checked,
        explanatory_checked,
        phi_dict,
        constants_checked,
    )

    # 2. Prepare profiles recent table
    profiles_recent_table, profiles_recent_initial = _prepare_profiles_recent_table(
        matrices_dict, lags_dict
    )

    # 3. Prepare inputs for adam_fitter
    (
        y_in_sample,
        ot,
        mat_vt,
        mat_wt,
        mat_f,
        vec_g,
        lags_model_all,
        index_lookup_table,
        profiles_recent_table_fortran,
    ) = _prepare_fitter_inputs(
        observations_dict,
        matrices_dict,
        lags_dict,
        profiles_dict,
        profiles_recent_table,
    )

    # 4. Run adam_fitter to get fitted values and states
    # refineHead should always be True (fixed backcasting issue)
    refine_head = True
    # Use conventional ETS for now (adamETS=False)
    adam_ets = False

    # Check if initial_type is a list or string and compute backcast correctly
    if isinstance(initials_checked['initial_type'], list):
        backcast_value_prep = any([t == "complete" or t == "backcasting" for t in initials_checked['initial_type']])
    else:
        backcast_value_prep = initials_checked['initial_type'] in ["complete", "backcasting"]

    # Call adam_cpp.fit() with the prepared inputs
    # Note: E, T, S, nNonSeasonal, nSeasonal, nArima, nXreg, constant are set during adamCore construction
    adam_fitted = adam_cpp.fit(
        matrixVt=mat_vt,
        matrixWt=mat_wt,
        matrixF=mat_f,
        vectorG=vec_g,
        indexLookupTable=index_lookup_table,
        profilesRecent=profiles_recent_table_fortran,
        vectorYt=y_in_sample,
        vectorOt=ot,
        backcast=backcast_value_prep,
        nIterations=initials_checked["n_iterations"],
        refineHead=refine_head,
    )
    # 5. Correct negative or NaN values in multiplicative components
    matrices_dict, profiles_dict = _correct_multiplicative_components(
        matrices_dict, profiles_dict, model_type_dict, components_dict
    )
    # 6. Initialize fitted values and errors series
    y_fitted, errors = _initialize_fitted_series(observations_dict)
    # 7. Fill in fitted values and errors from adam_cpp.fit() results
    errors[:] = adam_fitted.errors.flatten()
    y_fitted[:] = adam_fitted.fitted.flatten()
    # 8. Update distribution based on error type and loss function
    general_dict = _update_distribution(general_dict, model_type_dict)

    # 9. Process initial values for all components
    initial_value, initial_value_names, initial_estimated = _process_initial_values(
        model_type_dict,
        lags_dict,
        matrices_dict,
        components_dict,
        arima_checked,
        explanatory_checked,
        initials_checked,
    )

    # 10. Handle external regressors
    if explanatory_checked["xreg_model"] and explanatory_checked.get("regressors") != "adapt":
        explanatory_checked["regressors"] = "use"
    elif not explanatory_checked["xreg_model"]:
        explanatory_checked["regressors"] = None 

    # 11. Process ARMA parameters
    arma_parameters_list = _process_arma_parameters(arima_checked, adam_estimated)

    # 12. Calculate scale parameter
    scale = _calculate_scale_parameter(
        general_dict, model_type_dict, errors, y_fitted, observations_dict, other
    )

    # 13. Process constant and other parameters
    constant_value, other_returned = _process_other_parameters(
        constants_checked, adam_estimated, general_dict, arima_checked, lags_dict
    )

    # 14. Update parameters number
    general_dict["parameters_number"][0][2] = np.sum(
        general_dict["parameters_number"][0][:2]
    )

    # 15. Return the prepared model
    return {
        "model": model_type_dict["model"],
        "time_elapsed": None,  # Time calculation could be added if needed
        "holdout": general_dict["holdout"],
        "y_fitted": y_fitted,
        "residuals": errors,
        "states": adam_fitted.states,
        "profiles_recent_table": adam_fitted.profile,
        "persistence": matrices_dict["vec_g"],
        "transition": matrices_dict["mat_f"],
        "measurement": matrices_dict["mat_wt"],
        "phi": phi_dict["phi"],
        "initial": initial_value,
        "initial_type": initials_checked["initial_type"],
        "initial_estimated": initial_estimated,
        "orders": general_dict.get("orders"),
        "arma": arma_parameters_list,
        "constant": constant_value,
        "n_param": general_dict["parameters_number"],
        "occurrence": occurrence_dict["oes_model"],
        "formula": explanatory_checked.get("formula"),
        "regressors": explanatory_checked.get("regressors"),
        "loss": general_dict["loss"],
        "loss_value": adam_estimated["CF_value"],
        "log_lik": adam_estimated["log_lik_adam_value"],
        "distribution": general_dict["distribution"],
        "scale": scale,
        "other": other_returned,
        "B": adam_estimated["B"],
        "lags": lags_dict["lags"],
        "lags_all": lags_dict["lags_model_all"],
        "FI": general_dict.get("fi"),
    }



def ensure_level_format(level, side):
    
    # Fix just in case user used 95 etc instead of 0.95 
    level = level/100 if level > 1 else level
    
    # Handle different interval sides
    if side == "both":
        level_low = round((1 - level) / 2, 3)
        level_up = round((1 + level) / 2, 3)
        
    elif side == "upper":
        #level_low = np.zeros_like(level) 
        level_up =  level
    else:
        level_low = 1 - level

    return level_low, level_up


def generate_prediction_interval(predictions, 
                                 prepared_model,
                                general, 
                                 observations_dict,
                                 model_type_dict,
                                 lags_dict,
                        
                                params_info, level):
    

    mat_vt, mat_wt, vec_g, mat_f = _prepare_matrices_for_forecast(
        prepared_model, observations_dict, lags_dict, general
    )


    # stimate sigma
    s2 = sigma(observations_dict, params_info, general, prepared_model)**2

    # lines 8015 to 8022
    # line 8404 -> I dont get the (is.scale(object$scale))
    # Skipping for now.
    # Will ask Ivan what this is 

    # Check if model is ETS and has certain distributions with multiplicative errors
    if (model_type_dict['ets_model'] and 
        general['distribution'] in ['dinvgauss', 'dgamma', 'dlnorm', 'dllaplace', 'dls', 'dlgnorm'] and 
        model_type_dict['error_type'] == 'M'):

        # again scale object
        # lines 8425 8428

        v_voc_multi = var_anal(lags_dict['lags_model_all'], general['h'], mat_wt[0], mat_f, vec_g, s2)

        # Lines 8429-8433 in R/adam.R
        # If distribution is one of the log-based ones, transform the variance
        if general['distribution'] in ['dlnorm', 'dls', 'dllaplace', 'dlgnorm']:
            v_voc_multi = np.log(1 + v_voc_multi)
        
        # Lines 8435-8437 in R/adam.R
        # We don't do correct cumulatives in this case...
        if general.get('cumulative', False):
            v_voc_multi = np.sum(v_voc_multi)
    else:
        # Lines 8439-8441 in R/adam.R
        v_voc_multi = covar_anal(lags_dict['lags_model_all'], general['h'], mat_wt, mat_f, vec_g, s2)
        
        # Skipping the is.scale check (lines 8442-8445)
        
        # Lines 8447-8453 in R/adam.R
        # Do either the variance of sum, or a diagonal
        if general.get('cumulative', False):
            v_voc_multi = np.sum(v_voc_multi)
        else:
            v_voc_multi = np.diag(v_voc_multi)

    # Extract extra values which we will include in the function call
    # Now implement prediction intervals based on distribution
    # Translating from R/adam.R lines 8515-8640
    y_forecast = predictions
    y_lower = np.zeros_like(y_forecast)
    y_upper = np.zeros_like(y_forecast)

    level_low = (1 - level) / 2
    level_up = 1 - level_low
    e_type = model_type_dict['error_type']  # "A" or "M"


    distribution = general['distribution']
    other_params = general.get('other', {}) # Handle cases where 'other' might be missing

    if distribution == "dnorm":
        scale = np.sqrt(v_voc_multi)
        loc = 1 if e_type == "M" else 0
        y_lower[:] = stats.norm.ppf(level_low, loc=loc, scale=scale)
        y_upper[:] = stats.norm.ppf(level_up, loc=loc, scale=scale)

    elif distribution == "dlaplace":
        scale = np.sqrt(v_voc_multi / 2)
        loc = 1 if e_type == "M" else 0
        y_lower[:] = stats.laplace.ppf(level_low, loc=loc, scale=scale)
        y_upper[:] = stats.laplace.ppf(level_up, loc=loc, scale=scale)

    elif distribution == "ds":
        # Assuming stats.s_dist exists and follows R's qs(p, location, scale) convention
        # scale = (variance / 120)**0.25
        scale = (v_voc_multi / 120)**0.25
        loc = 1 if e_type == "M" else 0
        try:
            # Check if stats.s_dist exists before calling
            if hasattr(stats, 's_dist') and hasattr(stats.s_dist, 'ppf'):
                y_lower[:] = stats.s_dist.ppf(level_low, loc=loc, scale=scale)
                y_upper[:] = stats.s_dist.ppf(level_up, loc=loc, scale=scale)
            else:
                print("Warning: stats.s_dist not found. Cannot calculate intervals for 'ds'.")
                y_lower[:], y_upper[:] = np.nan, np.nan
        except Exception as e:
            print(f"Error calculating 'ds' interval: {e}")
            y_lower[:], y_upper[:] = np.nan, np.nan


    elif distribution == "dgnorm":
        # stats.gennorm.ppf(q, beta, loc=0, scale=1)
        shape_beta = other_params.get('shape')
        if shape_beta is not None:
            # Handle potential division by zero or issues with gamma function if shape is invalid
            try:
                scale = np.sqrt(v_voc_multi * (gamma(1 / shape_beta) / gamma(3 / shape_beta)))
                loc = 1 if e_type == "M" else 0
                y_lower[:] = stats.gennorm.ppf(level_low, beta=shape_beta, loc=loc, scale=scale)
                y_upper[:] = stats.gennorm.ppf(level_up, beta=shape_beta, loc=loc, scale=scale)
            except (ValueError, ZeroDivisionError) as e:
                print(f"Warning: Could not calculate scale for dgnorm (shape={shape_beta}). Error: {e}")
                y_lower[:], y_upper[:] = np.nan, np.nan
        else:
            print("Warning: Shape parameter 'beta' not found for dgnorm.")
            y_lower[:], y_upper[:] = np.nan, np.nan


    elif distribution == "dlogis":
        # Variance = (scale*pi)^2 / 3 => scale = sqrt(Variance*3) / pi
        scale = np.sqrt(v_voc_multi * 3) / np.pi
        loc = 1 if e_type == "M" else 0
        y_lower[:] = stats.logistic.ppf(level_low, loc=loc, scale=scale)
        y_upper[:] = stats.logistic.ppf(level_up, loc=loc, scale=scale)

    elif distribution == "dt":
        # stats.t.ppf(q, df, loc=0, scale=1)
        df = observations_dict['obs_in_sample'] - params_info['n_param']
        if df <= 0:
            print(f"Warning: Degrees of freedom ({df}) non-positive for dt distribution. Setting intervals to NaN.")
            y_lower[:], y_upper[:] = np.nan, np.nan
        else:
            scale = np.sqrt(v_voc_multi)
            if e_type == "A":
                y_lower[:] = scale * stats.t.ppf(level_low, df)
                y_upper[:] = scale * stats.t.ppf(level_up, df)
            else: # Etype == "M"
                y_lower[:] = (1 + scale * stats.t.ppf(level_low, df))
                y_upper[:] = (1 + scale * stats.t.ppf(level_up, df))

    elif distribution == "dalaplace":
        # Assuming stats.alaplace exists: ppf(q, loc, scale, alpha or kappa)
        alpha = other_params.get('alpha')
        if alpha is not None and 0 < alpha < 1:
            try:
                # Scale parameter from R code
                scale = np.sqrt(v_voc_multi * alpha**2 * (1 - alpha)**2 / (alpha**2 + (1 - alpha)**2))
                loc = 1 if e_type == "M" else 0
                # Assuming the third parameter is alpha/kappa
                # Check if stats.alaplace exists before calling
                if hasattr(stats, 'alaplace') and hasattr(stats.alaplace, 'ppf'):
                    # SciPy <= 1.8 used 'kappa', >= 1.9 uses 'alpha'
                    try:
                        y_lower[:] = stats.alaplace.ppf(level_low, loc=loc, scale=scale, alpha=alpha)
                        y_upper[:] = stats.alaplace.ppf(level_up, loc=loc, scale=scale, alpha=alpha)
                    except TypeError: # Try kappa for older SciPy versions
                        y_lower[:] = stats.alaplace.ppf(level_low, loc=loc, scale=scale, kappa=alpha)
                        y_upper[:] = stats.alaplace.ppf(level_up, loc=loc, scale=scale, kappa=alpha)
                else:
                    print("Warning: stats.alaplace not found. Cannot calculate intervals for 'dalaplace'.")
                    y_lower[:], y_upper[:] = np.nan, np.nan
            except (ValueError, ZeroDivisionError) as e:
                print(f"Warning: Could not calculate scale for dalaplace (alpha={alpha}). Error: {e}")
                y_lower[:], y_upper[:] = np.nan, np.nan
        else:
            print(f"Warning: Alpha parameter ({alpha}) invalid or not found for dalaplace.")
            y_lower[:], y_upper[:] = np.nan, np.nan


    # Log-Distributions (handling depends on whether v_voc_multi is variance of log)
    # Assuming v_voc_multi IS the variance of the log error based on R lines 8429-8433 if Etype=='M'
    # For Etype=='A', R calculates these as if M and then adjusts. Python code does this too.

    elif distribution == "dlnorm":
        # stats.lognorm.ppf(q, s, loc=0, scale=1). s=sdlog, scale=exp(meanlog)
        # Assuming E[1+err]=1 => meanlog = -sdlog^2/2 = -vcovMulti/2
        sdlog = np.sqrt(v_voc_multi)
        meanlog = -v_voc_multi / 2
        scipy_scale = np.exp(meanlog)
        # Calculate quantiles of (1+error) multiplier
        y_lower_mult = stats.lognorm.ppf(level_low, s=sdlog, loc=0, scale=scipy_scale)
        y_upper_mult = stats.lognorm.ppf(level_up, s=sdlog, loc=0, scale=scipy_scale)
        # Final adjustment depends on Etype (handled AFTER this block in R/Python)


    elif distribution == "dllaplace":
        # Corresponds to exp(Laplace(0, b)) where b = sqrt(var_log/2)
        scale_log = np.sqrt(v_voc_multi / 2)
        # Calculate quantiles of (1+error) multiplier
        y_lower_mult = np.exp(stats.laplace.ppf(level_low, loc=0, scale=scale_log))
        y_upper_mult = np.exp(stats.laplace.ppf(level_up, loc=0, scale=scale_log))
        # Final adjustment depends on Etype


    elif distribution == "dls":
        # Corresponds to exp(S(0, b)) where b = (var_log/120)**0.25
        scale_log = (v_voc_multi / 120)**0.25
        # Calculate quantiles of (1+error) multiplier
        try:
            # Check if stats.s_dist exists before calling
            if hasattr(stats, 's_dist') and hasattr(stats.s_dist, 'ppf'):
                y_lower_mult = np.exp(stats.s_dist.ppf(level_low, loc=0, scale=scale_log))
                y_upper_mult = np.exp(stats.s_dist.ppf(level_up, loc=0, scale=scale_log))
            else:
                print("Warning: stats.s_dist not found. Cannot calculate intervals for 'dls'.")
                y_lower_mult, y_upper_mult = np.nan, np.nan
        except Exception as e:
            print(f"Error calculating 'dls' interval: {e}")
            y_lower_mult, y_upper_mult = np.nan, np.nan
        # Final adjustment depends on Etype


    elif distribution == "dlgnorm":
        # Corresponds to exp(GenNorm(0, scale_log, beta))
        shape_beta = other_params.get('shape')
        if shape_beta is not None:
            try:
                scale_log = np.sqrt(v_voc_multi * (gamma(1 / shape_beta) / gamma(3 / shape_beta)))
                # Calculate quantiles of (1+error) multiplier
                y_lower_mult = np.exp(stats.gennorm.ppf(level_low, beta=shape_beta, loc=0, scale=scale_log))
                y_upper_mult = np.exp(stats.gennorm.ppf(level_up, beta=shape_beta, loc=0, scale=scale_log))
            except (ValueError, ZeroDivisionError) as e:
                print(f"Warning: Could not calculate scale for dlgnorm (shape={shape_beta}). Error: {e}")
                y_lower_mult, y_upper_mult = np.nan, np.nan
        else:
            print("Warning: Shape parameter 'beta' not found for dlgnorm.")
            y_lower_mult, y_upper_mult = np.nan, np.nan
        # Final adjustment depends on Etype

    # Distributions naturally multiplicative (or treated as such for intervals)
    elif distribution == "dinvgauss":
        # stats.invgauss.ppf(q, mu, loc=0, scale=1). mu is shape parameter.
        # R: qinvgauss(p, mean=1, dispersion=vcovMulti) -> implies lambda = 1/vcovMulti
        # Map (mean=1, lambda=1/vcovMulti) to scipy's mu. Tentative: mu = 1/vcovMulti?
        # Variance = mean^3 / lambda. If mean=1, Var = 1/lambda. If vcovMulti=Var -> lambda=1/vcovMulti
        # Let's try mu = 1 / vcovMulti as the shape parameter `mu` for scipy
        if np.any(v_voc_multi <= 0):
            print("Warning: Non-positive variance for dinvgauss. Setting intervals to NaN.")
            y_lower[:], y_upper[:] = np.nan, np.nan
        else:
            mu_shape = 1.0 / v_voc_multi # Tentative mapping
            # Calculate quantiles of (1+error) multiplier (mean should be 1)
            y_lower_mult = stats.invgauss.ppf(level_low, mu=mu_shape, loc=0, scale=1) # loc=0, scale=1 for standard form around mu
            y_upper_mult = stats.invgauss.ppf(level_up, mu=mu_shape, loc=0, scale=1)
            # Need to rescale ppf output? Let's assume R's mean=1 implies the output is already centered around 1. Needs verification.


    elif distribution == "dgamma":
        # stats.gamma.ppf(q, a, loc=0, scale=1). a=shape.
        # R: qgamma(p, shape=1/vcovMulti, scale=vcovMulti) -> Mean = shape*scale = 1. Variance = shape*scale^2 = vcovMulti.
        if np.any(v_voc_multi <= 0):
            print("Warning: Non-positive variance for dgamma. Setting intervals to NaN.")
            y_lower[:], y_upper[:] = np.nan, np.nan
        else:
            shape_a = 1.0 / v_voc_multi
            scale_param = v_voc_multi
            # Calculate quantiles of (1+error) multiplier (mean is 1)
            y_lower_mult = stats.gamma.ppf(level_low, a=shape_a, loc=0, scale=scale_param)
            y_upper_mult = stats.gamma.ppf(level_up, a=shape_a, loc=0, scale=scale_param)

    else:
        print(f"Warning: Distribution '{distribution}' not recognized for interval calculation.")
        y_lower[:], y_upper[:] = np.nan, np.nan


    # Final adjustments based on Etype (as done in R lines 8632-8640)
    # This part should come *after* the above block in your main script
    needs_etype_A_adjustment = distribution in ["dlnorm", "dllaplace", "dls", "dlgnorm", "dinvgauss", "dgamma"]

    if needs_etype_A_adjustment and e_type == "A":
        # Calculated _mult quantiles assuming multiplicative form, adjust for additive
        y_lower[:] = (y_lower_mult - 1) * y_forecast
        y_upper[:] = (y_upper_mult - 1) * y_forecast
    elif needs_etype_A_adjustment and e_type == "M":
        # Assign the calculated multiplicative quantiles directly
        y_lower[:] = y_lower_mult
        y_upper[:] = y_upper_mult


    # Create copies to store the final interval bounds
    y_lower_final = y_lower.copy()
    y_upper_final = y_upper.copy()

    # 1. Make sensible values out of extreme quantiles (handle Inf/-Inf)
    if not general["cumulative"]:
        # Check level_low for 0% quantile
        zero_lower_mask = (level_low == 0)
        if np.any(zero_lower_mask):
            if e_type == "A":
                y_lower_final[zero_lower_mask] = -np.inf
            else: # e_type == "M"
                y_lower_final[zero_lower_mask] = 0.0

        # Check level_up for 100% quantile
        one_upper_mask = (level_up == 1)
        if np.any(one_upper_mask):
            y_upper_final[one_upper_mask] = np.inf
    else: # cumulative = True (Dealing with a single value)
        if e_type == "A" and np.any(level_low == 0):
            y_lower_final[:] = -np.inf
        elif e_type == "M" and np.any(level_low == 0):
            y_lower_final[:] = 0.0

        if np.any(level_up == 1):
            y_upper_final[:] = np.inf

    # 2. Substitute NaNs
    nan_lower_mask = np.isnan(y_lower_final)
    if np.any(nan_lower_mask):
        replace_val = 0.0 if e_type == "A" else 1.0
        y_lower_final[nan_lower_mask] = replace_val

    nan_upper_mask = np.isnan(y_upper_final)
    if np.any(nan_upper_mask):
        replace_val = 0.0 if e_type == "A" else 1.0
        y_upper_final[nan_upper_mask] = replace_val

    # 3. Combine intervals with forecasts
    if e_type == "A":
        # y_lower/upper_final currently hold offsets, add forecast
        y_lower_final = y_forecast + y_lower_final
        y_upper_final = y_forecast + y_upper_final
    else: # e_type == "M"
        # y_lower/upper_final currently hold multipliers, multiply forecast
        y_lower_final = y_forecast * y_lower_final
        y_upper_final = y_forecast * y_upper_final

    return y_lower_final, y_upper_final


def generate_simulation_interval(
    predictions,
    prepared_model,
    general_dict,
    observations_dict,
    model_type_dict,
    lags_dict,
    components_dict,
    explanatory_checked,
    constants_checked,
    params_info,
    adam_cpp,
    level,
    nsim=10000,
    external_errors=None
):
    """
    Generate prediction intervals using simulation.

    This implements the simulation-based intervals from R's forecast.adam()
    (lines 8317-8412 in R/adam.R).

    Parameters
    ----------
    predictions : np.ndarray
        Point forecasts.
    prepared_model : dict
        Dictionary with the prepared model.
    general_dict : dict
        Dictionary with general model parameters.
    observations_dict : dict
        Dictionary with observation data.
    model_type_dict : dict
        Dictionary with model type information.
    lags_dict : dict
        Dictionary with lag-related information.
    components_dict : dict
        Dictionary with model components information.
    explanatory_checked : dict
        Dictionary with external regressors information.
    constants_checked : dict
        Dictionary with information about constants.
    params_info : dict
        Dictionary with parameter information.
    level : float
        Confidence level for prediction intervals.
    nsim : int
        Number of simulations to run.
    external_errors : np.ndarray, optional
        Pre-generated error matrix of shape (h, nsim) for deterministic testing.
        If provided, these errors are used instead of generating new ones.
        This allows 100% reproducibility between R and Python by using
        the same random errors.

    Returns
    -------
    tuple
        (y_lower, y_upper) arrays of prediction interval bounds.
    """
    h = general_dict["h"]
    lags_model_max = lags_dict["lags_model_max"]
    lags_model_all = lags_dict["lags_model_all"]

    # Get number of components
    n_components = (components_dict["components_number_ets"] +
                   components_dict.get("components_number_arima", 0) +
                   explanatory_checked["xreg_number"] +
                   int(constants_checked["constant_required"]))

    # 1. Create 3D state array: [components, h+lags_max, nsim]
    arr_vt = np.zeros((n_components, h + lags_model_max, nsim), order='F')

    # Initialize with current states (replicated across nsim)
    mat_vt = prepared_model["states"][:, observations_dict["obs_states"] - lags_model_max:observations_dict["obs_states"] + 1]
    for i in range(nsim):
        arr_vt[:, :lags_model_max, i] = mat_vt[:, :lags_model_max]

    # 2. Calculate degrees of freedom for de-biasing
    # params_info is a list of lists, params_info[0][-1] is the number of parameters
    n_param = params_info[0][-1] if params_info and params_info[0] else 0
    df = observations_dict["obs_in_sample"] - n_param
    if df <= 0:
        df = observations_dict["obs_in_sample"]

    # 3. Get and de-bias scale
    scale_value = prepared_model["scale"] * observations_dict["obs_in_sample"] / df

    # 4. Generate random errors or use external errors
    if external_errors is not None:
        # Use externally provided errors for deterministic testing
        mat_errors = external_errors
        if mat_errors.shape != (h, nsim):
            raise ValueError(f"external_errors shape {mat_errors.shape} does not match (h={h}, nsim={nsim})")
    else:
        distribution = general_dict["distribution"]
        other_params = general_dict.get("other", {})

        # Generate h*nsim errors and reshape to (h, nsim)
        errors_flat = generate_errors(
            distribution=distribution,
            n=h * nsim,
            scale=scale_value,
            obs_in_sample=observations_dict["obs_in_sample"],
            n_param=n_param,
            shape=other_params.get("shape"),
            alpha=other_params.get("alpha")
        )
        mat_errors = errors_flat.reshape((h, nsim), order='F')

    # 5. Normalize errors if nsim <= 500
    e_type = model_type_dict["error_type"]
    if nsim <= 500:
        mat_errors = normalize_errors(mat_errors, e_type)

    # 6. Determine modified error type for additive models with log-distributions
    e_type_modified = e_type
    if e_type == "A" and distribution in ["dlnorm", "dinvgauss", "dgamma", "dls", "dllaplace", "dlgnorm"]:
        e_type_modified = "M"

    # 7. Prepare matrices for simulator
    mat_vt_prep, mat_wt, vec_g, mat_f = _prepare_matrices_for_forecast(
        prepared_model, observations_dict, lags_dict, general_dict
    )

    # Prepare lookup table
    lookup = _prepare_lookup_table(lags_dict, observations_dict, general_dict)

    # Create 3D arrays for F and G (replicated for each simulation)
    arr_f = np.zeros((mat_f.shape[0], mat_f.shape[1], nsim), order='F')
    for i in range(nsim):
        arr_f[:, :, i] = mat_f

    # G matrix: [n_components, nsim]
    mat_g = np.zeros((n_components, nsim), order='F')
    for i in range(nsim):
        mat_g[:, i] = vec_g.flatten()

    # Occurrence matrix (all ones for now - no occurrence model)
    mat_ot = np.ones((h, nsim), order='F')

    # Profiles recent table
    profiles_recent = np.asfortranarray(prepared_model["profiles_recent_table"], dtype=np.float64)

    # Prepare inputs for C++ simulator
    arr_vt_f = np.asfortranarray(arr_vt, dtype=np.float64)
    mat_errors_f = np.asfortranarray(mat_errors, dtype=np.float64)
    mat_ot_f = np.asfortranarray(mat_ot, dtype=np.float64)
    arr_f_f = np.asfortranarray(arr_f, dtype=np.float64)
    mat_wt_f = np.asfortranarray(mat_wt, dtype=np.float64)
    mat_g_f = np.asfortranarray(mat_g, dtype=np.float64)
    lags_f = np.asfortranarray(lags_model_all, dtype=np.uint64).reshape(-1, 1)
    lookup_f = np.asfortranarray(lookup, dtype=np.uint64)

    # Determine adamETS setting (False for conventional ETS)
    adam_ets = False

    # 8. Call adam_cpp.simulate() with the prepared inputs
    # Note: E, T, S, nNonSeasonal, nSeasonal, nArima, nXreg, constant are set during adamCore construction
    sim_result = adam_cpp.simulate(
        matrixErrors=mat_errors_f,
        matrixOt=mat_ot_f,
        arrayVt=arr_vt_f,
        matrixWt=mat_wt_f,
        arrayF=arr_f_f,
        matrixG=mat_g_f,
        indexLookupTable=lookup_f,
        profilesRecent=profiles_recent,
        E=e_type_modified,
    )

    y_simulated = sim_result.data  # Shape: (h, nsim)

    # 9. Handle cumulative forecasts
    if general_dict.get("cumulative", False):
        y_forecast_sim = np.mean(np.sum(y_simulated, axis=0))
        level_low = (1 - level) / 2
        level_up = 1 - level_low
        y_lower = np.array([np.quantile(np.sum(y_simulated, axis=0), level_low)])
        y_upper = np.array([np.quantile(np.sum(y_simulated, axis=0), level_up)])
    else:
        # 10. Calculate quantiles for each horizon
        level_low = (1 - level) / 2
        level_up = 1 - level_low

        y_lower = np.zeros(h)
        y_upper = np.zeros(h)
        y_forecast_sim = np.zeros(h)

        for i in range(h):
            # For multiplicative trend/seasonal, use trimmed mean
            if model_type_dict["trend_type"] == "M" or (
                model_type_dict["season_type"] == "M" and h > lags_dict.get("lags_model_min", 1)
            ):
                # Trim 1% on each side
                y_forecast_sim[i] = stats.trim_mean(y_simulated[i, :], 0.01)
            else:
                y_forecast_sim[i] = np.mean(y_simulated[i, :])

            # Use R's type=7 quantile (linear interpolation)
            y_lower[i] = np.quantile(y_simulated[i, :], level_low, interpolation='linear')
            y_upper[i] = np.quantile(y_simulated[i, :], level_up, interpolation='linear')

    # 11. Convert to relative form (like parametric intervals)
    # R uses the same yForecast for both conversion and final combination:
    # - For additive models: yForecast = point forecast (adamForecast)
    # - For multiplicative: yForecast = simulation mean (overwritten in loop)
    if e_type == "A":
        # For additive: use point forecast for both operations (like R)
        y_lower = y_lower - predictions
        y_upper = y_upper - predictions
    else:
        # For multiplicative: use simulation mean for both operations (like R)
        # Avoid division by zero
        y_lower = np.where(y_forecast_sim != 0, y_lower / y_forecast_sim, 0)
        y_upper = np.where(y_forecast_sim != 0, y_upper / y_forecast_sim, 0)

    # 12. Final combination with forecasts (same as parametric)
    y_lower_final = y_lower.copy()
    y_upper_final = y_upper.copy()

    # Handle NaNs
    nan_lower_mask = np.isnan(y_lower_final)
    if np.any(nan_lower_mask):
        replace_val = 0.0 if e_type == "A" else 1.0
        y_lower_final[nan_lower_mask] = replace_val

    nan_upper_mask = np.isnan(y_upper_final)
    if np.any(nan_upper_mask):
        replace_val = 0.0 if e_type == "A" else 1.0
        y_upper_final[nan_upper_mask] = replace_val

    # Combine with forecasts using the SAME value as used for relative form
    if e_type == "A":
        # For additive: use point forecast (same as subtraction above)
        y_lower_final = predictions + y_lower_final
        y_upper_final = predictions + y_upper_final
    else:
        # For multiplicative: use simulation mean (same as division above)
        y_lower_final = y_forecast_sim * y_lower_final
        y_upper_final = y_forecast_sim * y_upper_final

    return y_lower_final, y_upper_final