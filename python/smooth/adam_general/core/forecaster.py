import numpy as np
import pandas as pd
import warnings

from smooth.adam_general._adam_general import adam_forecaster, adam_fitter
from smooth.adam_general.core.creator import adam_profile_creator, filler
from smooth.adam_general.core.utils.utils import scaler


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
        freq=observations_dict["frequency"],
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

    lookup = lookup_result["lookup"][
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
    if (
        model_type_dict["trend_type"] == "M"
        and not model_type_dict["damped"]
        and model_prepared["profiles_recent_table"]["profiles_recent_table"][1, 0] > 1
        and general_dict["h"] > 10
    ):
        warnings.warn(
            "Your model has a potentially explosive multiplicative trend. "
            "I cannot do anything about it, so please just be careful."
        )

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

    # Process occurrence based on type
    if occurrence_dict.get("occurrence") is not None and isinstance(
        occurrence_dict["occurrence"], bool
    ):
        # Boolean occurrence
        p_forecast = occurrence_dict["occurrence"] * 1
    elif occurrence_dict.get("occurrence") is not None and isinstance(
        occurrence_dict["occurrence"], (int, float)
    ):
        # Numeric occurrence
        p_forecast = occurrence_dict["occurrence"]
    else:
        # If this is a mixture model, produce forecasts for the occurrence
        if occurrence_dict.get("occurrence_model"):
            occurrence_model = True
            if occurrence_dict["occurrence"] == "provided":
                p_forecast = np.ones(general_dict["h"])
            else:
                # This is where the occurrence model forecast would be implemented
                # For now, default to ones as in the original code
                p_forecast = np.ones(general_dict["h"])
        else:
            occurrence_model = False
            # If this was provided occurrence, then use provided values
            if (
                occurrence_dict.get("occurrence") is not None
                and occurrence_dict.get("occurrence") == "provided"
                and occurrence_dict.get("p_forecast") is not None
            ):
                p_forecast = occurrence_dict["p_forecast"]
            else:
                p_forecast = np.ones(general_dict["h"])

    # Make sure the values are of the correct length
    if general_dict["h"] < len(p_forecast):
        p_forecast = p_forecast[: general_dict["h"]]
    elif general_dict["h"] > len(p_forecast):
        p_forecast = np.concatenate(
            [p_forecast, np.repeat(p_forecast[-1], general_dict["h"] - len(p_forecast))]
        )

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
    level, level_low, level_up = _prepare_forecast_intervals(general_dict)

    # 13. Format and return the final forecast output
    y_forecast_out = _format_forecast_output(
        y_forecast_values, observations_dict, level_low, level_up, h_final
    )

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
        observations_dict["y_in_sample"].values.flatten(), dtype=np.float64
    )
    ot = np.asarray(observations_dict["ot"].values.flatten(), dtype=np.float64)
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
    # Correct for multiplicative trend
    if model_type_dict["trend_type"] == "M" and (
        np.any(np.isnan(matrices_dict["mat_vt"][1, :]))
        or np.any(matrices_dict["mat_vt"][1, :] <= 0)
    ):
        i = np.where(matrices_dict["mat_vt"][1, :] <= 0)[0]
        matrices_dict["mat_vt"][1, i] = 1e-6
        profiles_dict["profiles_recent_table"][1, i] = 1e-6

    # Correct for multiplicative seasonality
    if (
        model_type_dict["season_type"] == "M"
        and np.all(
            ~np.isnan(
                matrices_dict["mat_vt"][
                    components_dict[
                        "components_number_ets_non_seasonal"
                    ] : components_dict["components_number_ets_non_seasonal"]
                    + components_dict["components_number_ets_seasonal"],
                    :,
                ]
            )
        )
        and np.any(
            matrices_dict["mat_vt"][
                components_dict["components_number_ets_non_seasonal"] : components_dict[
                    "components_number_ets_non_seasonal"
                ]
                + components_dict["components_number_ets_seasonal"],
                :,
            ]
            <= 0
        )
    ):

        i = np.where(
            matrices_dict["mat_vt"][
                components_dict["components_number_ets_non_seasonal"] : components_dict[
                    "components_number_ets_non_seasonal"
                ]
                + components_dict["components_number_ets_seasonal"],
                :,
            ]
            <= 0
        )[0]
        matrices_dict["mat_vt"][
            components_dict["components_number_ets_non_seasonal"] : components_dict[
                "components_number_ets_non_seasonal"
            ]
            + components_dict["components_number_ets_seasonal"],
            i,
        ] = 1e-6

        i = np.where(
            profiles_dict["profiles_recent_table"][
                components_dict["components_number_ets_non_seasonal"] : components_dict[
                    "components_number_ets_non_seasonal"
                ]
                + components_dict["components_number_ets_seasonal"],
                :,
            ]
            <= 0
        )[0]
        profiles_dict["profiles_recent_table"][
            components_dict["components_number_ets_non_seasonal"] : components_dict[
                "components_number_ets_non_seasonal"
            ]
            + components_dict["components_number_ets_seasonal"],
            i,
        ] = 1e-6

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
        y_fitted = pd.Series(
            np.full(observations_dict["obs_in_sample"], np.nan),
            index=pd.date_range(
                start=observations_dict["y_start"],
                periods=observations_dict["obs_in_sample"],
                freq=observations_dict["frequency"],
            ),
        )
        errors = pd.Series(
            np.full(observations_dict["obs_in_sample"], np.nan),
            index=pd.date_range(
                start=observations_dict["y_start"],
                periods=observations_dict["obs_in_sample"],
                freq=observations_dict["frequency"],
            ),
        )
    else:
        y_fitted = pd.Series(
            np.full(observations_dict["obs_in_sample"], np.nan),
            index=observations_dict["y_in_sample_index"],
        )
        errors = pd.Series(
            np.full(observations_dict["obs_in_sample"], np.nan),
            index=observations_dict["y_in_sample_index"],
        )

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
    # Initialize arrays for initial values
    initial_values_count = (
        model_type_dict["ets_model"]
        * (
            1
            + model_type_dict["model_is_trendy"]
            + model_type_dict["model_is_seasonal"]
        )
        + arima_checked["arima_model"]
        + explanatory_checked["xreg_model"]
    )

    initial_value = [None] * initial_values_count
    initial_value_ets = [None] * (
        model_type_dict["ets_model"] * len(lags_dict["lags_model"])
    )
    initial_value_names = [""] * initial_values_count

    # Vector that defines what was estimated in the model
    estimated_count = (
        model_type_dict["ets_model"]
        * (
            1
            + model_type_dict["model_is_trendy"]
            + model_type_dict["model_is_seasonal"]
            * components_dict["components_number_ets_seasonal"]
        )
        + arima_checked["arima_model"]
        + explanatory_checked["xreg_model"]
    )

    initial_estimated = [False] * estimated_count

    # Process ETS components if present
    j = 0
    if model_type_dict["ets_model"]:
        # Extract initial values for all components
        for i in range(len(lags_dict["lags_model"])):
            # For level/trend, get the first value
            if lags_dict["lags_model"][i] == 1:
                initial_value_ets[i] = matrices_dict["mat_vt"][
                    i, : lags_dict["lags_model_max"]
                ][0]
            # For seasonal components, get values from end of pre-heat period
            else:
                start_idx = lags_dict["lags_model_max"] - lags_dict["lags_model"][i]
                initial_value_ets[i] = matrices_dict["mat_vt"][
                    i, start_idx : lags_dict["lags_model_max"]
                ]

        # Process level
        j = 0
        initial_estimated[j] = initials_checked["initial_level_estimate"]
        initial_value[j] = initial_value_ets[j]
        initial_value_names[j] = "level"

        # Process trend if present
        if model_type_dict["model_is_trendy"]:
            j = 1
            initial_estimated[j] = initials_checked["initial_trend_estimate"]
            initial_value[j] = initial_value_ets[j]
            initial_value_ets[j] = None  # Remove from ETS list
            initial_value_names[j] = "trend"

        # Process seasonal components if present
        if model_type_dict["model_is_seasonal"]:
            initial_estimated[
                j + 1 : j + 1 + components_dict["components_number_ets_seasonal"]
            ] = initials_checked["initial_seasonal_estimate"]
            initial_value_ets[0] = None  # Remove level from ETS list
            j += 1

            if len(initials_checked["initial_seasonal_estimate"]) > 1:
                initial_value[j] = [x for x in initial_value_ets if x is not None]
                initial_value_names[j] = "seasonal"
                for k in range(components_dict["components_number_ets_seasonal"]):
                    initial_estimated[j + k] = f"seasonal{k+1}"
            else:
                initial_value[j] = next(x for x in initial_value_ets if x is not None)
                initial_value_names[j] = "seasonal"
                initial_estimated[j] = "seasonal"

    # Process ARIMA components if present
    if arima_checked["arima_model"]:
        j += 1
        initial_estimated[j] = initials_checked["initial_arima_estimate"]
        if initials_checked["initial_arima_estimate"]:
            initial_value[j] = matrices_dict["mat_vt"][
                components_dict["components_number_ets"]
                + components_dict.get("components_number_arima", 0)
                - 1,
                : initials_checked["initial_arima_number"],
            ]
        else:
            initial_value[j] = initials_checked["initial_arima"]
        initial_value_names[j] = "arima"
        initial_estimated[j] = "arima"

    # Convert to dictionary with names as keys
    initial_value = {
        name: value for name, value in zip(initial_value_names, initial_value)
    }

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
    # Return None if no ARIMA model
    if not arima_checked["arima_model"]:
        return None

    arma_parameters_list = {}

    # Process AR parameters if present
    j = 0
    if arima_checked["ar_required"] and arima_checked["ar_estimate"]:
        # Avoid damping parameter phi by checking name length > 3
        arma_parameters_list["ar"] = [
            adam_estimated["B"][name]
            for name in adam_estimated["B"]
            if len(name) > 3 and name.startswith("phi")
        ]
        j += 1
    elif arima_checked["ar_required"] and not arima_checked["ar_estimate"]:
        # Use provided parameters
        arma_parameters_list["ar"] = [
            p
            for name, p in arima_checked["arma_parameters"].items()
            if name.startswith("phi")
        ]
        j += 1

    # Process MA parameters if present
    if arima_checked["ma_required"] and arima_checked["ma_estimate"]:
        arma_parameters_list["ma"] = [
            adam_estimated["B"][name]
            for name in adam_estimated["B"]
            if name.startswith("theta")
        ]
    elif arima_checked["ma_required"] and not arima_checked["ma_estimate"]:
        arma_parameters_list["ma"] = [
            p
            for name, p in arima_checked["arma_parameters"].items()
            if name.startswith("theta")
        ]

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
    if constants_checked["constant_estimate"]:
        constant_value = adam_estimated["B"][constants_checked["constant_name"]]
    else:
        constant_value = constants_checked["constant"]

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
        backcast=any(
            [
                t == "complete" or t == "backcasting"
                for t in initials_checked["initial_type"]
            ]
        ),
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
    if (
        explanatory_checked["xreg_model"]
        and explanatory_checked.get("regressors") != "adapt"
    ):
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
