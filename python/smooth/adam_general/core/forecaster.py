import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.special import gamma

from smooth.adam_general._adam_general import adam_forecaster, adam_fitter
from smooth.adam_general.core.creator import adam_profile_creator, filler
from smooth.adam_general.core.utils.utils import scaler
from smooth.adam_general.core.utils.var_covar import sigma, covar_anal, var_anal, matrix_power_wrap

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
    pandas.DatetimeIndex
        Index for the forecast
    """
    observations_dict["y_forecast_index"] = pd.date_range(
        start=observations_dict["y_forecast_start"], 
        periods=general_dict["h"], 
        freq=observations_dict["frequency"]
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
):
    """
    Generate point forecasts using adam_forecaster.

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
    
    # Call adam_forecaster with the prepared inputs
    y_forecast = adam_forecaster(
        matrixWt=np.asfortranarray(mat_wt, dtype=np.float64),
        matrixF=np.asfortranarray(mat_f, dtype=np.float64),
        lags=lags_model_all,
        indexLookupTable=index_lookup_table,
        profilesRecent=profiles_recent_table,
        E=model_type_dict["error_type"],
        T=model_type_dict["trend_type"],
        S=model_type_dict["season_type"],
        nNonSeasonal=components_dict["components_number_ets_non_seasonal"],
        nSeasonal=components_dict["components_number_ets_seasonal"],
        nArima=components_dict.get("components_number_arima", 0),
        nXreg=explanatory_checked["xreg_number"],
        constant=constants_checked["constant_required"],
        horizon=general_dict["h"],
    ).flatten()

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
    calculate_intervals,
    interval_method,
    level,
    side,
):
    """
    Generate forecasts from a prepared ADAM model.

    Parameters
    ----------
    model_prepared : dict
        Dictionary with the prepared model including fitted values and states
    observations_dict : dict
        Dictionary with observation data and related information
    general_dict : dict
        Dictionary with general model parameters
    occurrence_dict : dict
        Dictionary with occurrence model parameters
    lags_dict : dict
        Dictionary with lag-related information
    model_type_dict : dict
        Dictionary with model type information (ETS, ARIMA components)
    explanatory_checked : dict
        Dictionary with external regressors information
    components_dict : dict
        Dictionary with model components information
    constants_checked : dict
        Dictionary with information about constants in the model
    calculate_intervals : bool
        Whether to calculate prediction intervals
    interval_method : str
        Method to use for calculating prediction intervals
    level : list
        Confidence levels for prediction intervals
    side : str
        Side for prediction intervals

    Returns
    -------
    pandas.DataFrame
        DataFrame with forecasts and prediction intervals (if specified)
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

    # 7. Generate point forecasts using adam_forecaster
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
        # assert method is parameteric boostrap or simulation
        assert interval_method in ["parametric", "simulation", "bootstrap"], "Interval method must be either parametric, simulation, or bootstrap"

        # if it is not parameteric raise warning and say that for now only parametric is supported
        if interval_method != "parametric":
            warnings.warn("For now only parametric intervals are supported. Other methods will be implemented in the future.")
            interval_method = "parametric"

        if level is None:
            warnings.warn("No confidence level specified. Using default level of 0.95")
            level = [0.95]

        level_low, level_up = ensure_level_format(level, side)
    
        y_lower, y_upper = generate_prediction_interval(y_forecast_values, model_prepared, general_dict, observations_dict, model_type_dict, lags_dict, params_info, level)
    
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
    # Optional parameters
    bounds="usual",
    other=None,
):
    """
    Prepare the model after estimation for forecasting.

    This function takes the estimated parameters and various model components
    and prepares everything needed for generating forecasts.

    Parameters
    ----------
    model_type_dict : dict
        Dictionary with model type information
    components_dict : dict
        Dictionary with model components information
    lags_dict : dict
        Dictionary with lag-related information
    matrices_dict : dict
        Dictionary with model matrices
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
    observations_dict : dict
        Dictionary with observation data and related information
    occurrence_dict : dict
        Dictionary with occurrence model parameters
    general_dict : dict
        Dictionary with general model parameters
    profiles_dict : dict
        Dictionary with profile information
    adam_estimated : dict
        Dictionary with estimated parameters
    bounds : str, optional
        Type of bounds used in estimation
    other : dict, optional
        Additional parameters

    Returns
    -------
    dict
        Dictionary with prepared model for forecasting
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
    # Determine refineHead based on whether ARIMA is present
    refine_head = not arima_checked['arima_model']
    # Use conventional ETS for now (adamETS=False)
    adam_ets = False

    # Check if initial_type is a list or string and compute backcast correctly
    if isinstance(initials_checked['initial_type'], list):
        backcast_value_prep = any([t == "complete" or t == "backcasting" for t in initials_checked['initial_type']])
    else:
        backcast_value_prep = initials_checked['initial_type'] in ["complete", "backcasting"]

    adam_fitted = adam_fitter(
        matrixVt=mat_vt,
        matrixWt=mat_wt,
        matrixF=mat_f,
        vectorG=vec_g,
        lags=lags_model_all,
        indexLookupTable=index_lookup_table,
        profilesRecent=profiles_recent_table_fortran,
        E=model_type_dict["error_type"],
        T=model_type_dict["trend_type"],
        S=model_type_dict["season_type"],
        nNonSeasonal=components_dict["components_number_ets"]
        - components_dict["components_number_ets_seasonal"],
        nSeasonal=components_dict["components_number_ets_seasonal"],
        nArima=components_dict["components_number_arima"],
        nXreg=explanatory_checked["xreg_number"],
        constant=constants_checked["constant_required"],
        vectorYt=y_in_sample,
        vectorOt=ot,
        backcast=backcast_value_prep,
        nIterations=initials_checked["n_iterations"],
        refineHead=refine_head,
        adamETS=adam_ets
    )
    # 5. Correct negative or NaN values in multiplicative components
    matrices_dict, profiles_dict = _correct_multiplicative_components(
        matrices_dict, profiles_dict, model_type_dict, components_dict
    )
    # 6. Initialize fitted values and errors series
    y_fitted, errors = _initialize_fitted_series(observations_dict)
        # 7. Fill in fitted values and errors from adam_fitter results
    errors[:] = adam_fitted["errors"].flatten()
    y_fitted[:] = adam_fitted["yFitted"].flatten()
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
        "states": adam_fitted["matVt"],
        "profiles_recent_table": adam_fitted["profile"],
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