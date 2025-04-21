import numpy as np
from typing import List, Dict, Union, Any
from scipy.optimize import minimize

from smooth.adam_general.core.utils.utils import msdecompose
from smooth.adam_general.core.utils.polynomials import adam_polynomialiser

"""
Model creation module for ADAM (Augmented Dynamic Adaptive Model) forecasting.

This module contains functions for creating model matrices, initializing parameters,
and handling the model architecture for various time series forecasting models 
including ETS, ARIMA, and their combinations.

The main functions are:
- creator: Creates model matrices for ADAM
- initialiser: Initializes parameters for the model
- filler: Fills in model matrices with optimized parameters
- architector: Sets up the model architecture
- adam_profile_creator: Creates profile matrices for ADAM
"""


def creator(
    # Model type info
    model_type_dict,
    # Lags info
    lags_dict,
    # Profiles
    profiles_dict,
    # Observation info
    observations_dict,
    # Parameter dictionaries
    persistence_checked,
    initials_checked,
    arima_checked,
    constants_checked,
    phi_dict,
    # Components info
    components_dict,
    explanatory_checked=None,
):
    """
    Creates the model matrices for ADAM.

    Args:
        model_type_dict: Dictionary containing model type information
        lags_dict: Dictionary containing lags information
        profiles_dict: Dictionary containing profiles information
        observations_dict: Dictionary containing observation information
        persistence_checked: Dictionary of persistence parameters
        initials_checked: Dictionary of initial values
        arima_checked: Dictionary of ARIMA parameters
        constants_checked: Dictionary of constant parameters
        explanatory_checked: Dictionary of explanatory variables parameters
        phi_dict: Dictionary containing phi parameters
        components_dict: Dictionary containing component information

    Returns:
        Dict: Dictionary containing the created model matrices
    """
    # Extract data and parameters
    model_params = _extract_model_parameters(
        model_type_dict,
        components_dict,
        lags_dict,
        observations_dict,
        phi_dict,
        profiles_dict,
    )

    # Setup matrices
    matrices = _setup_matrices(
        model_params,
        observations_dict,
        components_dict,
        explanatory_checked,
        constants_checked,
    )

    # Setup persistence vector and transition matrix
    matrices = _setup_persistence_vector(
        matrices,
        model_params,
        persistence_checked,
        arima_checked,
        constants_checked,
        explanatory_checked,
    )

    # Setup measurement vector and handle damping
    matrices = _setup_measurement_vector(matrices, model_params, explanatory_checked)

    # Handle ARIMA polynomials if needed
    matrices = _handle_polynomial_setup(matrices, model_params, arima_checked)

    # Initialize states
    matrices = _initialize_states(
        matrices,
        model_params,
        profiles_dict,
        observations_dict,
        initials_checked,
        arima_checked,
        explanatory_checked,
        constants_checked,
    )

    # Return created matrices
    return {
        "mat_vt": matrices["mat_vt"],
        "mat_wt": matrices["mat_wt"],
        "mat_f": matrices["mat_f"],
        "vec_g": matrices["vec_g"],
        "arima_polynomials": matrices.get("arima_polynomials", None),
    }


def _extract_model_parameters(
    model_type_dict,
    components_dict,
    lags_dict,
    observations_dict,
    phi_dict,
    profiles_dict,
):
    """
    Extract and organize model parameters from various dictionaries.

    Args:
        model_type_dict: Dictionary containing model type information
        components_dict: Dictionary containing component information
        lags_dict: Dictionary containing lags information
        observations_dict: Dictionary containing observation information
        phi_dict: Dictionary containing phi parameters
        profiles_dict: Dictionary containing profiles information

    Returns:
        Dict: Dictionary containing organized model parameters
    """
    # Extract observation values
    obs_states = observations_dict["obs_states"]
    obs_in_sample = observations_dict["obs_in_sample"]
    obs_all = observations_dict["obs_all"]
    ot_logical = observations_dict["ot_logical"]
    y_in_sample = observations_dict["y_in_sample"]
    obs_nonzero = observations_dict["obs_nonzero"]

    # Extract values from dictionaries
    ets_model = model_type_dict["ets_model"]
    e_type = model_type_dict["error_type"]
    t_type = model_type_dict["trend_type"]
    s_type = model_type_dict["season_type"]
    model_is_trendy = model_type_dict["model_is_trendy"]
    model_is_seasonal = model_type_dict["model_is_seasonal"]

    # Extract component numbers
    components_number_ets = components_dict["components_number_ets"]
    components_number_ets_seasonal = components_dict["components_number_ets_seasonal"]
    components_number_arima = components_dict.get("components_number_arima", 0)

    # Extract phi parameter
    phi = phi_dict["phi"]

    # Extract lags info
    lags = lags_dict["lags"]
    lags_model = lags_dict["lags_model"]
    lags_model_arima = lags_dict["lags_model_arima"]
    lags_model_all = lags_dict["lags_model_all"]
    lags_model_max = lags_dict["lags_model_max"]

    # Extract profiles info
    profiles_recent_table = profiles_dict["profiles_recent_table"]
    profiles_recent_provided = profiles_dict["profiles_recent_provided"]

    return {
        "obs_states": obs_states,
        "obs_in_sample": obs_in_sample,
        "obs_all": obs_all,
        "ot_logical": ot_logical,
        "y_in_sample": y_in_sample,
        "obs_nonzero": obs_nonzero,
        "ets_model": ets_model,
        "e_type": e_type,
        "t_type": t_type,
        "s_type": s_type,
        "model_is_trendy": model_is_trendy,
        "model_is_seasonal": model_is_seasonal,
        "components_number_ets": components_number_ets,
        "components_number_ets_seasonal": components_number_ets_seasonal,
        "components_number_arima": components_number_arima,
        "phi": phi,
        "lags": lags,
        "lags_model": lags_model,
        "lags_model_arima": lags_model_arima,
        "lags_model_all": lags_model_all,
        "lags_model_max": lags_model_max,
        "profiles_recent_table": profiles_recent_table,
        "profiles_recent_provided": profiles_recent_provided,
    }


def _setup_matrices(
    model_params,
    observations_dict,
    components_dict,
    explanatory_checked,
    constants_checked,
):
    """
    Setup the state, measurement, and transition matrices for the model.

    Args:
        model_params: Dictionary containing model parameters
        observations_dict: Dictionary containing observation information
        components_dict: Dictionary containing component information
        explanatory_checked: Dictionary of explanatory variables parameters
        constants_checked: Dictionary containing constant parameters

    Returns:
        Dict: Dictionary containing initialized matrices
    """
    # Get parameters
    obs_states = model_params["obs_states"]
    obs_all = model_params["obs_all"]
    components_number_ets = model_params["components_number_ets"]
    components_number_arima = model_params["components_number_arima"]

    # Matrix of states. Time in columns, components in rows
    mat_vt = np.full(
        (
            components_number_ets
            + components_number_arima
            + explanatory_checked["xreg_number"]
            + constants_checked["constant_required"],
            obs_states,
        ),
        np.nan,
    )

    # Measurement rowvector
    mat_wt = np.ones(
        (
            obs_all,
            components_number_ets
            + components_number_arima
            + explanatory_checked["xreg_number"]
            + constants_checked["constant_required"],
        )
    )

    # Transition matrix
    mat_f = np.eye(
        components_number_ets
        + components_number_arima
        + explanatory_checked["xreg_number"]
        + constants_checked["constant_required"]
    )

    # Persistence vector
    vec_g = np.zeros(
        (
            components_number_ets
            + components_number_arima
            + explanatory_checked["xreg_number"]
            + constants_checked["constant_required"],
            1,
        )
    )

    return {"mat_vt": mat_vt, "mat_wt": mat_wt, "mat_f": mat_f, "vec_g": vec_g}


def _setup_measurement_vector(matrices, model_params, explanatory_checked):
    """
    Setup the measurement vector and handle damping parameters.

    Args:
        matrices: Dictionary containing model matrices
        model_params: Dictionary containing model parameters
        explanatory_checked: Dictionary of explanatory variables parameters

    Returns:
        Dict: Updated matrices dictionary
    """
    # Get matrices
    mat_wt = matrices["mat_wt"]
    mat_f = matrices["mat_f"]

    # Get parameters
    ets_model = model_params["ets_model"]
    model_is_trendy = model_params["model_is_trendy"]
    phi = model_params["phi"]
    components_number_ets = model_params["components_number_ets"]
    components_number_arima = model_params["components_number_arima"]

    # If xreg are provided, then fill in the respective values in Wt vector
    if explanatory_checked["xreg_model"]:
        mat_wt[
            :,
            components_number_ets
            + components_number_arima : components_number_ets
            + components_number_arima
            + explanatory_checked["xreg_number"],
        ] = explanatory_checked["xreg_data"]

    # Damping parameter value
    if ets_model and model_is_trendy:
        mat_f[0, 1] = phi
        mat_f[1, 1] = phi
        mat_wt[:, 1] = phi

    matrices["mat_wt"] = mat_wt
    matrices["mat_f"] = mat_f

    return matrices


def _setup_persistence_vector(
    matrices,
    model_params,
    persistence_checked,
    arima_checked,
    constants_checked,
    explanatory_checked,
):
    """
    Setup the persistence vector for the model.

    Args:
        matrices: Dictionary containing model matrices
        model_params: Dictionary containing model parameters
        persistence_checked: Dictionary of persistence parameters
        arima_checked: Dictionary of ARIMA parameters
        constants_checked: Dictionary containing constant parameters
        explanatory_checked: Dictionary of explanatory variables parameters

    Returns:
        Dict: Updated matrices dictionary
    """
    # Get matrices
    mat_f = matrices["mat_f"]
    vec_g = matrices["vec_g"]

    # Get parameters
    ets_model = model_params["ets_model"]
    model_is_trendy = model_params["model_is_trendy"]
    model_is_seasonal = model_params["model_is_seasonal"]
    components_number_ets = model_params["components_number_ets"]
    components_number_arima = model_params["components_number_arima"]

    j = 0
    # ETS model, persistence
    if ets_model:
        j += 1
        if not persistence_checked["persistence_level_estimate"]:
            vec_g[j - 1, 0] = persistence_checked["persistence_level"]

        if model_is_trendy:
            j += 1
            if not persistence_checked["persistence_trend_estimate"]:
                vec_g[j - 1, 0] = persistence_checked["persistence_trend"]

        if model_is_seasonal:
            if not all(persistence_checked["persistence_seasonal_estimate"]):
                vec_g[
                    j
                    + np.where(
                        np.logical_not(
                            persistence_checked["persistence_seasonal_estimate"]
                        )
                    )[0],
                    0,
                ] = persistence_checked["persistence_seasonal"]

    # ARIMA model, persistence
    if arima_checked["arima_model"]:
        # Remove diagonal from the ARIMA part of the matrix
        mat_f[j : j + components_number_arima, j : j + components_number_arima] = 0
        j += components_number_arima

    # Modify transition to handle drift
    if not arima_checked["arima_model"] and constants_checked["constant_required"]:
        mat_f[0, -1] = 1

    # Regression, persistence
    if explanatory_checked["xreg_model"]:
        if (
            persistence_checked["persistence_xreg_provided"]
            and not persistence_checked["persistence_xreg_estimate"]
        ):
            vec_g[j : j + explanatory_checked["xreg_number"], 0] = persistence_checked[
                "persistence_xreg"
            ]

    matrices["mat_f"] = mat_f
    matrices["vec_g"] = vec_g

    return matrices


def _handle_polynomial_setup(matrices, model_params, arima_checked):
    """
    Handle ARIMA polynomial setup if needed.

    Args:
        matrices: Dictionary containing model matrices
        model_params: Dictionary containing model parameters
        arima_checked: Dictionary of ARIMA parameters

    Returns:
        Dict: Updated matrices dictionary
    """
    # Get matrices
    mat_f = matrices["mat_f"]
    vec_g = matrices["vec_g"]

    # Get parameters
    lags = model_params["lags"]
    components_number_ets = model_params["components_number_ets"]

    # If the arma parameters were provided, fill in the persistence
    arima_polynomials = None
    if arima_checked["arima_model"] and (
        not arima_checked["ar_estimate"] and not arima_checked["ma_estimate"]
    ):
        # Call polynomial
        arima_polynomials = {
            key: np.array(value)
            for key, value in adam_polynomialiser(
                0,
                arima_checked["ar_orders"],
                arima_checked["i_orders"],
                arima_checked["ma_orders"],
                arima_checked["ar_estimate"],
                arima_checked["ma_estimate"],
                arima_checked["arma_parameters"],
                lags,
            ).items()
        }

        # Fill in the transition matrix
        if len(arima_checked["non_zero_ari"]) > 0:
            non_zero_ari = np.array(arima_checked["non_zero_ari"])
            mat_f[
                components_number_ets + non_zero_ari[:, 1],
                components_number_ets + non_zero_ari[:, 1],
            ] = -arima_polynomials["ari_polynomial"][non_zero_ari[:, 0]]

        # Fill in the persistence vector
        if len(arima_checked["non_zero_ari"]) > 0:
            non_zero_ari = np.array(arima_checked["non_zero_ari"])
            vec_g[components_number_ets + non_zero_ari[:, 1], 0] = -arima_polynomials[
                "ari_polynomial"
            ][non_zero_ari[:, 0]]

        if len(arima_checked["non_zero_ma"]) > 0:
            non_zero_ma = np.array(arima_checked["non_zero_ma"])
            vec_g[components_number_ets + non_zero_ma[:, 1], 0] += arima_polynomials[
                "ma_polynomial"
            ][non_zero_ma[:, 0]]

    matrices["mat_f"] = mat_f
    matrices["vec_g"] = vec_g
    matrices["arima_polynomials"] = arima_polynomials

    return matrices


def _initialize_states(
    matrices,
    model_params,
    profiles_dict,
    observations_dict,
    initials_checked,
    arima_checked,
    explanatory_checked,
    constants_checked,
):
    """
    Initialize the state matrix with proper values.

    Args:
        matrices: Dictionary containing model matrices
        model_params: Dictionary containing model parameters
        profiles_dict: Dictionary containing profiles information
        observations_dict: Dictionary containing observation information
        initials_checked: Dictionary of initial values
        arima_checked: Dictionary of ARIMA parameters
        explanatory_checked: Dictionary of explanatory variables parameters
        constants_checked: Dictionary containing constant parameters

    Returns:
        Dict: Updated matrices dictionary
    """
    # Get matrices
    mat_vt = matrices["mat_vt"]

    # Get parameters
    profiles_recent_provided = model_params["profiles_recent_provided"]
    ets_model = model_params["ets_model"]
    model_is_seasonal = model_params["model_is_seasonal"]
    e_type = model_params["e_type"]
    lags_model_max = model_params["lags_model_max"]
    profiles_recent_table = model_params["profiles_recent_table"]

    # If recent profiles are not provided, initialize states
    if not profiles_recent_provided:
        # ETS model initialization
        if ets_model:
            mat_vt = _initialize_ets_states(
                mat_vt,
                model_params,
                initials_checked,
                explanatory_checked,
                observations_dict,
            )

        # ARIMA model initialization
        if arima_checked["arima_model"]:
            mat_vt = _initialize_arima_states(
                mat_vt, model_params, initials_checked, arima_checked, observations_dict
            )

        # Initialize explanatory variables
        if explanatory_checked["xreg_model"]:
            mat_vt = _initialize_xreg_states(
                mat_vt,
                model_params,
                initials_checked,
                explanatory_checked,
                arima_checked,
            )

        # Initialize constant if needed
        if constants_checked["constant_required"]:
            mat_vt = _initialize_constant(
                mat_vt,
                model_params,
                constants_checked,
                initials_checked,
                observations_dict,
                arima_checked,
                explanatory_checked,
                ets_model,
            )
    else:
        # If profiles are provided, use them directly
        mat_vt[:, 0:lags_model_max] = profiles_recent_table

    matrices["mat_vt"] = mat_vt
    return matrices


def _initialize_ets_states(
    mat_vt, model_params, initials_checked, explanatory_checked, observations_dict
):
    """
    Initialize ETS model states.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        initials_checked: Dictionary of initial values
        explanatory_checked: Dictionary of explanatory variables parameters
        observations_dict: Dictionary containing observation information

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    ets_model = model_params["ets_model"]
    model_is_seasonal = model_params["model_is_seasonal"]
    model_is_trendy = model_params["model_is_trendy"]
    e_type = model_params["e_type"]
    t_type = model_params["t_type"]
    s_type = model_params["s_type"]
    lags = model_params["lags"]
    lags_model = model_params["lags_model"]
    lags_model_max = model_params["lags_model_max"]
    components_number_ets_seasonal = model_params["components_number_ets_seasonal"]
    y_in_sample = model_params["y_in_sample"]
    ot_logical = model_params["ot_logical"]
    obs_nonzero = model_params["obs_nonzero"]

    # If initials need to be estimated
    if initials_checked["initial_estimate"]:
        # For seasonal models
        if model_is_seasonal:
            # Handle seasonal model initialization
            if obs_nonzero >= lags_model_max * 2:
                mat_vt = _initialize_ets_seasonal_states_with_decomp(
                    mat_vt, model_params, initials_checked, explanatory_checked
                )
            else:
                mat_vt = _initialize_ets_seasonal_states_small_sample(
                    mat_vt, model_params, initials_checked, explanatory_checked
                )
        else:
            # Handle non-seasonal model initialization
            mat_vt = _initialize_ets_nonseasonal_states(
                mat_vt, model_params, initials_checked
            )

        # Handle special case for multiplicative models
        if (
            initials_checked["initial_level_estimate"]
            and e_type == "M"
            and mat_vt[0, lags_model_max - 1] == 0
        ):
            mat_vt[0, 0:lags_model_max] = np.mean(y_in_sample)

    # If initials are provided, use them directly
    elif (
        not initials_checked["initial_estimate"]
        and initials_checked["initial_type"] == "provided"
    ):
        j = 0
        mat_vt[j, 0:lags_model_max] = initials_checked["initial_level"]
        if model_is_trendy:
            j += 1
            mat_vt[j, 0:lags_model_max] = initials_checked["initial_trend"]
        if model_is_seasonal:
            for i in range(components_number_ets_seasonal):
                # This is misaligned, but that's okay, because this goes directly to profile_recent
                mat_vt[j + i, 0 : lags_model[j + i]] = initials_checked[
                    "initial_seasonal"
                ][i]

    return mat_vt


def _initialize_ets_seasonal_states_with_decomp(
    mat_vt, model_params, initials_checked, explanatory_checked
):
    """
    Initialize ETS seasonal model states using decomposition when sufficient data is available.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        initials_checked: Dictionary of initial values
        explanatory_checked: Dictionary of explanatory variables parameters

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    e_type = model_params["e_type"]
    t_type = model_params["t_type"]
    s_type = model_params["s_type"]
    lags = model_params["lags"]
    lags_model = model_params["lags_model"]
    lags_model_max = model_params["lags_model_max"]
    model_is_trendy = model_params["model_is_trendy"]
    components_number_ets_seasonal = model_params["components_number_ets_seasonal"]
    y_in_sample = model_params["y_in_sample"]
    ot_logical = model_params["ot_logical"]

    # If either e_type or s_type are multiplicative, do multiplicative decomposition
    decomposition_type = (
        "multiplicative" if any(x == "M" for x in [e_type, s_type]) else "additive"
    )
    y_decomposition = msdecompose(
        y_in_sample.values.ravel(),
        [lag for lag in lags if lag != 1],
        type=decomposition_type,
    )
    j = 0

    # Initialize level
    if initials_checked["initial_level_estimate"]:
        mat_vt[j, 0:lags_model_max] = y_decomposition["initial"][0]
        if explanatory_checked["xreg_model"]:
            if e_type == "A":
                mat_vt[j, 0:lags_model_max] -= np.dot(
                    explanatory_checked["xreg_model_initials"][0]["initial_xreg"],
                    explanatory_checked["xreg_data"][0],
                )
            else:
                mat_vt[j, 0:lags_model_max] /= np.exp(
                    np.dot(
                        explanatory_checked["xreg_model_initials"][1]["initial_xreg"],
                        explanatory_checked["xreg_data"][0],
                    )
                )
    else:
        mat_vt[j, 0:lags_model_max] = initials_checked["initial_level"]
    j += 1

    # Initialize trend if needed
    if model_is_trendy:
        if initials_checked["initial_trend_estimate"]:
            # Handle different trend types
            if t_type == "A" and s_type == "M":
                mat_vt[j, 0:lags_model_max] = (
                    np.prod(y_decomposition["initial"]) - y_decomposition["initial"][0]
                )

                # If the initial trend is higher than the lowest value, initialise with zero.
                # This is a failsafe mechanism for the mixed models
                if mat_vt[j, 0] < 0 and abs(mat_vt[j, 0]) > min(
                    abs(y_in_sample[ot_logical])
                ):
                    mat_vt[j, 0:lags_model_max] = 0
            elif t_type == "M" and s_type == "A":
                mat_vt[j, 0:lags_model_max] = sum(
                    abs(y_decomposition["initial"])
                ) / abs(y_decomposition["initial"][0])
            elif t_type == "M":
                # trend is too dangerous, make it start from 1.
                mat_vt[j, 0:lags_model_max] = 1
            else:
                # trend
                mat_vt[j, 0:lags_model_max] = y_decomposition["initial"][1]

            # Safety checks for multiplicative trend models
            if t_type == "M" and np.any(mat_vt[j, 0:lags_model_max] > 1.1):
                mat_vt[j, 0:lags_model_max] = 1
            if t_type == "M" and np.any(mat_vt[0, 0:lags_model_max] < 0):
                mat_vt[0, 0:lags_model_max] = y_in_sample[ot_logical][0]
        else:
            mat_vt[j, 0:lags_model_max] = initials_checked["initial_trend"]
        j += 1

    # Initialize seasonal components
    # For pure models use stuff as is
    if (
        all(x == "A" for x in [e_type, s_type])
        or all(x == "M" for x in [e_type, s_type])
        or (e_type == "A" and s_type == "M")
    ):
        for i in range(components_number_ets_seasonal):
            if initials_checked["initial_seasonal_estimate"]:
                # NOTE: CAREFULL THAT 0 INDEX ON LAGS MODEL FOR MULTIPLE SEASONALITIES
                mat_vt[i + j, 0 : lags_model[i + j]] = y_decomposition["seasonal"][i][
                    0 : lags_model[i + j]
                ]
                # Renormalise the initial seasons
                if s_type == "A":
                    mat_vt[i + j, 0 : lags_model[i + j]] -= np.mean(
                        mat_vt[i + j, 0 : lags_model[i + j]]
                    )
                else:
                    mat_vt[i + j, 0 : lags_model[i + j]] /= np.exp(
                        np.mean(np.log(mat_vt[i + j, 0 : lags_model[i + j]]))
                    )
            else:
                mat_vt[i + j, 0 : lags_model[i + j]] = initials_checked[
                    "initial_seasonal"
                ][i]
    # For mixed models use a different set of initials
    elif e_type == "M" and s_type == "A":
        for i in range(components_number_ets_seasonal):
            if initials_checked["initial_seasonal_estimate"]:
                mat_vt[i + j, 0 : lags_model[i + j]] = np.log(
                    y_decomposition["seasonal"][i][0 : lags_model[i + j]]
                ) * np.min(y_in_sample[ot_logical])
                # Renormalise the initial seasons
                if s_type == "A":
                    mat_vt[i + j, 0 : lags_model[i + j]] -= np.mean(
                        mat_vt[i + j, 0 : lags_model[i + j]]
                    )
                else:
                    mat_vt[i + j, 0 : lags_model[i + j]] /= np.exp(
                        np.mean(np.log(mat_vt[i + j, 0 : lags_model[i + j]]))
                    )
            else:
                mat_vt[i + j, 0 : lags_model[i + j]] = initials_checked[
                    "initial_seasonal"
                ][i]

    return mat_vt


def _initialize_ets_seasonal_states_small_sample(
    mat_vt, model_params, initials_checked, explanatory_checked
):
    """
    Initialize ETS seasonal model states when limited data is available.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        initials_checked: Dictionary of initial values
        explanatory_checked: Dictionary of explanatory variables parameters

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    e_type = model_params["e_type"]
    t_type = model_params["t_type"]
    s_type = model_params["s_type"]
    lags = model_params["lags"]
    lags_model = model_params["lags_model"]
    lags_model_max = model_params["lags_model_max"]
    model_is_trendy = model_params["model_is_trendy"]
    components_number_ets_seasonal = model_params["components_number_ets_seasonal"]
    y_in_sample = model_params["y_in_sample"]
    ot_logical = model_params["ot_logical"]

    # If either e_type or s_type are multiplicative, do multiplicative decomposition
    j = 0
    # level
    if initials_checked["initial_level_estimate"]:
        mat_vt[j, 0:lags_model_max] = np.mean(y_in_sample[0:lags_model_max])
        if explanatory_checked["xreg_model"]:
            if e_type == "A":
                mat_vt[j, 0:lags_model_max] -= np.dot(
                    explanatory_checked["xreg_model_initials"][0]["initial_xreg"],
                    explanatory_checked["xreg_data"][0],
                )
            else:
                mat_vt[j, 0:lags_model_max] /= np.exp(
                    np.dot(
                        explanatory_checked["xreg_model_initials"][1]["initial_xreg"],
                        explanatory_checked["xreg_data"][0],
                    )
                )
    else:
        mat_vt[j, 0:lags_model_max] = initials_checked["initial_level"]
    j += 1

    # Initialize trend if needed
    if model_is_trendy:
        if initials_checked["initial_trend_estimate"]:
            if t_type == "A":
                # trend
                mat_vt[j, 0:lags_model_max] = y_in_sample[1] - y_in_sample[0]
            elif t_type == "M":
                if initials_checked["initial_level_estimate"]:
                    # level fix
                    mat_vt[j - 1, 0:lags_model_max] = np.exp(
                        np.mean(np.log(y_in_sample[ot_logical][0:lags_model_max]))
                    )
                # trend
                mat_vt[j, 0:lags_model_max] = y_in_sample[1] / y_in_sample[0]

        # Safety check for multiplicative trend
        if t_type == "M" and np.any(mat_vt[j, 0:lags_model_max] > 1.1):
            mat_vt[j, 0:lags_model_max] = 1
        else:
            mat_vt[j, 0:lags_model_max] = initials_checked["initial_trend"]
        j += 1

    # Initialize seasonal components
    if s_type == "A":
        for i in range(components_number_ets_seasonal):
            if initials_checked["initial_seasonal_estimate"]:
                mat_vt[i + j - 1, 0 : lags_model[i + j - 1]] = (
                    y_in_sample[0 : lags_model[i + j - 1]] - mat_vt[0, 0]
                )
                # Renormalise the initial seasons
                mat_vt[i + j - 1, 0 : lags_model[i + j - 1]] -= np.mean(
                    mat_vt[i + j - 1, 0 : lags_model[i + j - 1]]
                )
            else:
                mat_vt[i + j - 1, 0 : lags_model[i + j - 1]] = initials_checked[
                    "initial_seasonal"
                ][i]
    # For mixed models use a different set of initials
    else:
        for i in range(components_number_ets_seasonal):
            if initials_checked["initial_seasonal_estimate"]:
                # abs() is needed for mixed ETS+ARIMA
                mat_vt[i + j - 1, 0 : lags_model[i + j - 1]] = y_in_sample[
                    0 : lags_model[i + j - 1]
                ] / abs(mat_vt[0, 0])
                # Renormalise the initial seasons
                mat_vt[i + j - 1, 0 : lags_model[i + j - 1]] /= np.exp(
                    np.mean(np.log(mat_vt[i + j - 1, 0 : lags_model[i + j - 1]]))
                )
            else:
                mat_vt[i + j - 1, 0 : lags_model[i + j - 1]] = initials_checked[
                    "initial_seasonal"
                ][i]

    return mat_vt


def _initialize_ets_nonseasonal_states(mat_vt, model_params, initials_checked):
    """
    Initialize ETS non-seasonal model states.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        initials_checked: Dictionary of initial values

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    lags_model_max = model_params["lags_model_max"]
    model_is_trendy = model_params["model_is_trendy"]
    t_type = model_params["t_type"]
    y_in_sample = model_params["y_in_sample"]
    ot_logical = model_params["ot_logical"]
    obs_in_sample = model_params["obs_in_sample"]

    # level
    if initials_checked["initial_level_estimate"]:
        mat_vt[0, :lags_model_max] = np.mean(
            y_in_sample[: max(lags_model_max, int(obs_in_sample * 0.2))]
        )
    else:
        mat_vt[0, :lags_model_max] = initials_checked["initial_level"]

    # trend
    if model_is_trendy:
        if initials_checked["initial_trend_estimate"]:
            if t_type == "A":
                mat_vt[1, 0:lags_model_max] = np.nanmean(
                    np.diff(
                        y_in_sample[
                            0 : max(lags_model_max + 1, int(obs_in_sample * 0.2))
                        ],
                        axis=0,
                    )
                )
            else:  # t_type == "M"
                mat_vt[1, 0:lags_model_max] = np.exp(
                    np.mean(np.diff(np.log(y_in_sample[ot_logical])))
                )
        else:
            mat_vt[1, 0:lags_model_max] = initials_checked["initial_trend"]

    return mat_vt


def _initialize_arima_states(
    mat_vt, model_params, initials_checked, arima_checked, observations_dict
):
    """
    Initialize ARIMA model states.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        initials_checked: Dictionary of initial values
        arima_checked: Dictionary of ARIMA parameters
        observations_dict: Dictionary containing observation information

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    components_number_ets = model_params["components_number_ets"]
    components_number_arima = model_params["components_number_arima"]
    e_type = model_params["e_type"]
    lags = model_params["lags"]
    y_in_sample = model_params["y_in_sample"]
    ot_logical = model_params["ot_logical"]

    if initials_checked["initial_arima_estimate"]:
        mat_vt[
            components_number_ets : components_number_ets + components_number_arima,
            0 : initials_checked["initial_arima_number"],
        ] = (
            0 if e_type == "A" else 1
        )

        if any(lag > 1 for lag in lags):
            y_decomposition = msdecompose(
                y_in_sample,
                [lag for lag in lags if lag != 1],
                type="additive" if e_type == "A" else "multiplicative",
            )["seasonal"][-1][0]
        else:
            y_decomposition = (
                np.mean(np.diff(y_in_sample[ot_logical]))
                if e_type == "A"
                else np.exp(np.mean(np.diff(np.log(y_in_sample[ot_logical]))))
            )

        mat_vt[
            components_number_ets + components_number_arima - 1,
            0 : initials_checked["initial_arima_number"],
        ] = np.tile(
            y_decomposition,
            int(np.ceil(initials_checked["initial_arima_number"] / max(lags))),
        )[
            : initials_checked["initial_arima_number"]
        ]
    else:
        mat_vt[
            components_number_ets : components_number_ets + components_number_arima,
            0 : initials_checked["initial_arima_number"],
        ] = (
            0 if e_type == "A" else 1
        )
        mat_vt[
            components_number_ets + components_number_arima - 1,
            0 : initials_checked["initial_arima_number"],
        ] = initials_checked["initial_arima"][
            : initials_checked["initial_arima_number"]
        ]

    return mat_vt


def _initialize_xreg_states(
    mat_vt, model_params, initials_checked, explanatory_checked, arima_checked
):
    """
    Initialize explanatory variables states.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        initials_checked: Dictionary of initial values
        explanatory_checked: Dictionary of explanatory variables parameters
        arima_checked: Dictionary of ARIMA parameters

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    components_number_ets = model_params["components_number_ets"]
    components_number_arima = model_params["components_number_arima"]
    e_type = model_params["e_type"]
    lags_model_max = model_params["lags_model_max"]

    if (
        e_type == "A"
        or initials_checked["initial_xreg_provided"]
        or explanatory_checked["xreg_model_initials"][1] is None
    ):
        mat_vt[
            components_number_ets
            + components_number_arima : components_number_ets
            + components_number_arima
            + explanatory_checked["xreg_number"],
            0:lags_model_max,
        ] = explanatory_checked["xreg_model_initials"][0]["initial_xreg"]
    else:
        mat_vt[
            components_number_ets
            + components_number_arima : components_number_ets
            + components_number_arima
            + explanatory_checked["xreg_number"],
            0:lags_model_max,
        ] = explanatory_checked["xreg_model_initials"][1]["initial_xreg"]

    return mat_vt


def _initialize_constant(
    mat_vt,
    model_params,
    constants_checked,
    initials_checked,
    observations_dict,
    arima_checked,
    explanatory_checked,
    ets_model,
):
    """
    Initialize constant term if required.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        constants_checked: Dictionary containing constant parameters
        initials_checked: Dictionary of initial values
        observations_dict: Dictionary containing observation information
        arima_checked: Dictionary of ARIMA parameters
        explanatory_checked: Dictionary of explanatory variables parameters
        ets_model: Boolean indicating if ETS model is used

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    components_number_ets = model_params["components_number_ets"]
    components_number_arima = model_params["components_number_arima"]
    e_type = model_params["e_type"]
    lags_model_max = model_params["lags_model_max"]
    y_in_sample = model_params["y_in_sample"]
    ot_logical = model_params["ot_logical"]

    if constants_checked["constant_estimate"]:
        # Add the mean of data
        if sum(arima_checked["i_orders"]) == 0 and not ets_model:
            mat_vt[
                components_number_ets
                + components_number_arima
                + explanatory_checked["xreg_number"],
                :,
            ] = np.mean(y_in_sample[ot_logical])
        else:
            if e_type == "A":
                mat_vt[
                    components_number_ets
                    + components_number_arima
                    + explanatory_checked["xreg_number"],
                    :,
                ] = np.mean(np.diff(y_in_sample[ot_logical]))
            else:
                mat_vt[
                    components_number_ets
                    + components_number_arima
                    + explanatory_checked["xreg_number"],
                    :,
                ] = np.exp(np.mean(np.diff(np.log(y_in_sample[ot_logical]))))
    else:
        mat_vt[
            components_number_ets
            + components_number_arima
            + explanatory_checked["xreg_number"],
            :,
        ] = constants_checked["constant_value"]

    # If ETS model is used, change the initial level
    if ets_model and initials_checked["initial_level_estimate"]:
        if e_type == "A":
            mat_vt[0, 0:lags_model_max] -= mat_vt[
                components_number_ets
                + components_number_arima
                + explanatory_checked["xreg_number"],
                0,
            ]
        else:
            mat_vt[0, 0:lags_model_max] /= mat_vt[
                components_number_ets
                + components_number_arima
                + explanatory_checked["xreg_number"],
                0,
            ]

    # If ARIMA is done, debias states
    if arima_checked["arima_model"] and initials_checked["initial_arima_estimate"]:
        if e_type == "A":
            mat_vt[
                components_number_ets : components_number_ets + components_number_arima,
                0 : initials_checked["initial_arima_number"],
            ] -= mat_vt[
                components_number_ets
                + components_number_arima
                + explanatory_checked["xreg_number"],
                0,
            ]
        else:
            mat_vt[
                components_number_ets : components_number_ets + components_number_arima,
                0 : initials_checked["initial_arima_number"],
            ] /= mat_vt[
                components_number_ets
                + components_number_arima
                + explanatory_checked["xreg_number"],
                0,
            ]

    return mat_vt


def initialiser(
    # Model type info
    model_type_dict,
    # Components info
    components_dict,
    # Lags info
    lags_dict,
    # Matrices from creator
    adam_created,
    # Parameter dictionaries
    persistence_checked,
    initials_checked,
    arima_checked,
    constants_checked,
    explanatory_checked,
    phi_dict,
    # Other parameters
    observations_dict,
    bounds="usual",
    other=None,
):
    """
    Initialize parameters for the ADAM model.

    Args:
        model_type_dict: Dictionary containing model type information
        components_dict: Dictionary containing component information
        lags_dict: Dictionary containing lags information
        adam_created: Dictionary containing created model matrices
        persistence_checked: Dictionary of persistence parameters
        initials_checked: Dictionary of initial values
        arima_checked: Dictionary of ARIMA parameters
        constants_checked: Dictionary of constant parameters
        explanatory_checked: Dictionary of explanatory variables parameters
        phi_dict: Dictionary containing phi parameters
        observations_dict: Dictionary containing observation information
        bounds: Bounds specification for optimization
        other: Other parameters

    Returns:
        Dict: Dictionary containing initialized parameters
    """
    # Extract model parameters
    model_params = _extract_initialiser_params(
        model_type_dict, components_dict, lags_dict, adam_created, observations_dict
    )

    # Prepare inputs for optimization
    optimization_inputs = _prepare_optimization_inputs(
        model_params,
        persistence_checked,
        initials_checked,
        arima_checked,
        constants_checked,
        explanatory_checked,
        phi_dict,
        bounds,
    )

    # Perform optimization
    optimization_result = _optimize_parameters(optimization_inputs, model_params)

    # Extract and format results
    result = _extract_optimization_results(
        optimization_result, optimization_inputs, model_params
    )

    return result


def _extract_initialiser_params(
    model_type_dict, components_dict, lags_dict, adam_created, observations_dict
):
    """
    Extract parameters needed for initialization from input dictionaries.

    Args:
        model_type_dict: Dictionary containing model type information
        components_dict: Dictionary containing component information
        lags_dict: Dictionary containing lags information
        adam_created: Dictionary containing created model matrices
        observations_dict: Dictionary containing observation information

    Returns:
        Dict: Dictionary containing extracted parameters
    """
    # Extract observation values
    obs_states = observations_dict["obs_states"]
    obs_in_sample = observations_dict["obs_in_sample"]
    obs_all = observations_dict["obs_all"]
    ot_logical = observations_dict["ot_logical"]
    y_in_sample = observations_dict["y_in_sample"]

    # Extract model type values
    error_type = model_type_dict["error_type"]
    trend_type = model_type_dict["trend_type"]
    season_type = model_type_dict["season_type"]
    ets_model = model_type_dict["ets_model"]
    model_is_trendy = model_type_dict["model_is_trendy"]
    model_is_seasonal = model_type_dict["model_is_seasonal"]

    # Extract matrices
    mat_vt = adam_created["mat_vt"]
    mat_wt = adam_created["mat_wt"]
    mat_f = adam_created["mat_f"]
    vec_g = adam_created["vec_g"]

    # Extract component numbers
    components_number_ets = components_dict["components_number_ets"]
    components_number_ets_seasonal = components_dict["components_number_ets_seasonal"]
    components_number_arima = components_dict.get("components_number_arima", 0)

    # Extract lags info
    lags = lags_dict["lags"]
    lags_model = lags_dict["lags_model"]
    lags_model_max = lags_dict["lags_model_max"]

    return {
        "obs_states": obs_states,
        "obs_in_sample": obs_in_sample,
        "obs_all": obs_all,
        "ot_logical": ot_logical,
        "y_in_sample": y_in_sample,
        "error_type": error_type,
        "trend_type": trend_type,
        "season_type": season_type,
        "ets_model": ets_model,
        "model_is_trendy": model_is_trendy,
        "model_is_seasonal": model_is_seasonal,
        "mat_vt": mat_vt,
        "mat_wt": mat_wt,
        "mat_f": mat_f,
        "vec_g": vec_g,
        "components_number_ets": components_number_ets,
        "components_number_ets_seasonal": components_number_ets_seasonal,
        "components_number_arima": components_number_arima,
        "lags": lags,
        "lags_model": lags_model,
        "lags_model_max": lags_model_max,
    }


def _prepare_optimization_inputs(
    model_params,
    persistence_checked,
    initials_checked,
    arima_checked,
    constants_checked,
    explanatory_checked,
    phi_dict,
    bounds,
):
    """
    Prepare inputs for parameter optimization.

    Args:
        model_params: Dictionary containing model parameters
        persistence_checked: Dictionary of persistence parameters
        initials_checked: Dictionary of initial values
        arima_checked: Dictionary of ARIMA parameters
        constants_checked: Dictionary of constant parameters
        explanatory_checked: Dictionary of explanatory variables parameters
        phi_dict: Dictionary containing phi parameters
        bounds: Bounds specification for optimization

    Returns:
        Dict: Dictionary containing prepared optimization inputs
    """
    # Extract relevant parameters
    ets_model = model_params["ets_model"]
    model_is_trendy = model_params["model_is_trendy"]
    model_is_seasonal = model_params["model_is_seasonal"]
    components_number_ets = model_params["components_number_ets"]
    components_number_ets_seasonal = model_params["components_number_ets_seasonal"]
    error_type = model_params["error_type"]

    # Determine bounds
    bound_min = []
    bound_max = []

    # ETS model
    if ets_model:
        # Smoothing parameter for level
        if persistence_checked["persistence_level_estimate"]:
            if bounds == "usual":
                bound_min.append(0)
                bound_max.append(1)
            elif bounds == "admissible":
                bound_min.append(0)
                bound_max.append(2)
            elif bounds == "none":
                bound_min.append(-np.inf)
                bound_max.append(np.inf)

        # Smoothing parameter for trend
        if model_is_trendy and persistence_checked["persistence_trend_estimate"]:
            if bounds == "usual":
                bound_min.append(0)
                bound_max.append(1)
            elif bounds == "admissible":
                bound_min.append(0)
                bound_max.append(2)
            elif bounds == "none":
                bound_min.append(-np.inf)
                bound_max.append(np.inf)

        # Smoothing parameters for seasonality
        if model_is_seasonal:
            for i in range(components_number_ets_seasonal):
                if persistence_checked["persistence_seasonal_estimate"][i]:
                    if bounds == "usual":
                        bound_min.append(0)
                        bound_max.append(1)
                    elif bounds == "admissible":
                        bound_min.append(0)
                        bound_max.append(2)
                    elif bounds == "none":
                        bound_min.append(-np.inf)
                        bound_max.append(np.inf)

    # ARIMA parameters
    if arima_checked["arima_model"]:
        # AR parameters
        if arima_checked["ar_estimate"]:
            for i in range(sum(arima_checked["ar_orders"])):
                if bounds == "usual":
                    bound_min.append(-0.99)
                    bound_max.append(0.99)
                elif bounds == "admissible":
                    bound_min.append(-0.999)
                    bound_max.append(0.999)
                elif bounds == "none":
                    bound_min.append(-np.inf)
                    bound_max.append(np.inf)

        # MA parameters
        if arima_checked["ma_estimate"]:
            for i in range(sum(arima_checked["ma_orders"])):
                if bounds == "usual":
                    bound_min.append(-0.99)
                    bound_max.append(0.99)
                elif bounds == "admissible":
                    bound_min.append(-0.999)
                    bound_max.append(0.999)
                elif bounds == "none":
                    bound_min.append(-np.inf)
                    bound_max.append(np.inf)

    # Regression parameters
    if explanatory_checked["xreg_model"] and explanatory_checked["xreg_estimate"]:
        for i in range(explanatory_checked["xreg_number"]):
            if bounds == "none":
                bound_min.append(-np.inf)
                bound_max.append(np.inf)
            else:
                bound_min.append(-1000)
                bound_max.append(1000)

    # Damping parameter
    if ets_model and model_is_trendy and phi_dict["phi_estimate"]:
        if bounds == "usual":
            bound_min.append(0.8)
            bound_max.append(0.98)
        elif bounds == "admissible":
            bound_min.append(0.5)
            bound_max.append(0.999)
        elif bounds == "none":
            bound_min.append(-np.inf)
            bound_max.append(np.inf)

    # Initial values for ETS components
    if ets_model:
        # Initial level
        if initials_checked["initial_level_estimate"]:
            if error_type == "A" or bounds == "none":
                bound_min.append(-np.inf)
                bound_max.append(np.inf)
            else:
                bound_min.append(1e-10)
                bound_max.append(np.inf)

        # Initial trend
        if model_is_trendy and initials_checked["initial_trend_estimate"]:
            if error_type == "A" or bounds == "none":
                bound_min.append(-np.inf)
                bound_max.append(np.inf)
            else:
                if bounds == "usual":
                    bound_min.append(0.9)
                    bound_max.append(1.1)
                else:
                    bound_min.append(0.5)
                    bound_max.append(2)
        else:
            # This else matches with "if model_is_trendy and initials_checked["initial_trend_estimate"]:"
            if bounds == "usual":
                bound_min.append(0.9)
                bound_max.append(1.1)
            else:
                bound_min.append(0.5)
                bound_max.append(2)

        # Initial seasonal
        if model_is_seasonal and initials_checked["initial_seasonal_estimate"]:
            for i in range(components_number_ets_seasonal):
                for j in range(
                    model_params["lags_model"][i + 1 + int(model_is_trendy)]
                ):
                    if error_type == "A" or bounds == "none":
                        bound_min.append(-np.inf)
                        bound_max.append(np.inf)
                    else:
                        if bounds == "usual":
                            bound_min.append(0.9)
                            bound_max.append(1.1)
                        else:
                            bound_min.append(0.5)
                            bound_max.append(2)

    # ARIMA initial values
    if arima_checked["arima_model"] and initials_checked["initial_arima_estimate"]:
        for i in range(initials_checked["initial_arima_number"]):
            if error_type == "A" or bounds == "none":
                bound_min.append(-np.inf)
                bound_max.append(np.inf)
            else:
                if bounds == "usual":
                    bound_min.append(0.9)
                    bound_max.append(1.1)
                else:
                    bound_min.append(0.5)
                    bound_max.append(2)

    # Constant
    if (
        constants_checked["constant_required"]
        and constants_checked["constant_estimate"]
    ):
        if error_type == "A" or bounds == "none":
            bound_min.append(-np.inf)
            bound_max.append(np.inf)
        else:
            if bounds == "usual":
                bound_min.append(0.9)
                bound_max.append(1.1)
            else:
                bound_min.append(0.5)
                bound_max.append(2)

    return {
        "bound_min": bound_min,
        "bound_max": bound_max,
        "persistence_checked": persistence_checked,
        "initials_checked": initials_checked,
        "arima_checked": arima_checked,
        "constants_checked": constants_checked,
        "explanatory_checked": explanatory_checked,
        "phi_dict": phi_dict,
    }


def _optimize_parameters(optimization_inputs, model_params):
    """
    Optimize model parameters using minimization.

    Args:
        optimization_inputs: Dictionary containing optimization inputs
        model_params: Dictionary containing model parameters

    Returns:
        Dict: Optimization result
    """
    bound_min = optimization_inputs["bound_min"]
    bound_max = optimization_inputs["bound_max"]

    # If no parameters to optimize, return empty result
    if len(bound_min) == 0:
        return {"x": [], "success": True}

    # Form initial vector
    B0 = np.zeros(len(bound_min))

    # Create optimization bounds
    bounds = [(bound_min[i], bound_max[i]) for i in range(len(bound_min))]

    # Define objective function
    def objective(b):
        # Create filled matrices
        filled_matrices = filler(
            b,
            model_type_dict={
                "ets_model": model_params["ets_model"],
                "error_type": model_params["error_type"],
                "trend_type": model_params["trend_type"],
                "season_type": model_params["season_type"],
                "model_is_trendy": model_params["model_is_trendy"],
                "model_is_seasonal": model_params["model_is_seasonal"],
            },
            components_dict={
                "components_number_ets": model_params["components_number_ets"],
                "components_number_ets_seasonal": model_params[
                    "components_number_ets_seasonal"
                ],
                "components_number_arima": model_params["components_number_arima"],
            },
            lags_dict={
                "lags": model_params["lags"],
                "lags_model": model_params["lags_model"],
                "lags_model_max": model_params["lags_model_max"],
            },
            matrices_dict={
                "mat_vt": model_params["mat_vt"].copy(),
                "mat_wt": model_params["mat_wt"].copy(),
                "mat_f": model_params["mat_f"].copy(),
                "vec_g": model_params["vec_g"].copy(),
            },
            persistence_checked=optimization_inputs["persistence_checked"],
            initials_checked=optimization_inputs["initials_checked"],
            arima_checked=optimization_inputs["arima_checked"],
            explanatory_checked=optimization_inputs["explanatory_checked"],
            phi_dict=optimization_inputs["phi_dict"],
            constants_checked=optimization_inputs["constants_checked"],
        )

        # Calculate error using filled matrices
        error = np.zeros(model_params["obs_in_sample"])
        xt = filled_matrices["mat_vt"][:, 0:1]

        for t in range(model_params["obs_in_sample"]):
            # Measurement
            if model_params["error_type"] == "A":
                error[t] = (
                    model_params["y_in_sample"][t]
                    - np.dot(filled_matrices["mat_wt"][t, :], xt).ravel()[0]
                )
            else:
                error[t] = (
                    model_params["y_in_sample"][t]
                    / np.dot(filled_matrices["mat_wt"][t, :], xt).ravel()[0]
                    - 1
                )

            # Transition
            if t < model_params["obs_in_sample"] - 1:
                xt = (
                    np.dot(filled_matrices["mat_f"], xt)
                    + filled_matrices["vec_g"] * error[t]
                )

        # Calculate cost
        return np.mean(error**2)

    # Run optimization
    result = minimize(objective, B0, method="L-BFGS-B", bounds=bounds)

    return result


def _extract_optimization_results(
    optimization_result, optimization_inputs, model_params
):
    """
    Extract and format optimization results.

    Args:
        optimization_result: Result from optimization
        optimization_inputs: Dictionary containing optimization inputs
        model_params: Dictionary containing model parameters

    Returns:
        Dict: Formatted optimization results
    """
    # Fill matrices with optimized parameters
    filled_matrices = filler(
        optimization_result["x"] if len(optimization_result["x"]) > 0 else [],
        model_type_dict={
            "ets_model": model_params["ets_model"],
            "error_type": model_params["error_type"],
            "trend_type": model_params["trend_type"],
            "season_type": model_params["season_type"],
            "model_is_trendy": model_params["model_is_trendy"],
            "model_is_seasonal": model_params["model_is_seasonal"],
        },
        components_dict={
            "components_number_ets": model_params["components_number_ets"],
            "components_number_ets_seasonal": model_params[
                "components_number_ets_seasonal"
            ],
            "components_number_arima": model_params["components_number_arima"],
        },
        lags_dict={
            "lags": model_params["lags"],
            "lags_model": model_params["lags_model"],
            "lags_model_max": model_params["lags_model_max"],
        },
        matrices_dict={
            "mat_vt": model_params["mat_vt"].copy(),
            "mat_wt": model_params["mat_wt"].copy(),
            "mat_f": model_params["mat_f"].copy(),
            "vec_g": model_params["vec_g"].copy(),
        },
        persistence_checked=optimization_inputs["persistence_checked"],
        initials_checked=optimization_inputs["initials_checked"],
        arima_checked=optimization_inputs["arima_checked"],
        explanatory_checked=optimization_inputs["explanatory_checked"],
        phi_dict=optimization_inputs["phi_dict"],
        constants_checked=optimization_inputs["constants_checked"],
    )

    # Return results
    return {
        "mat_vt": filled_matrices["mat_vt"],
        "mat_wt": filled_matrices["mat_wt"],
        "mat_f": filled_matrices["mat_f"],
        "vec_g": filled_matrices["vec_g"],
        "optimization_result": optimization_result,
        "parameters": (
            optimization_result["x"] if len(optimization_result["x"]) > 0 else []
        ),
    }


def filler(
    B,
    model_type_dict,
    components_dict,
    lags_dict,
    matrices_dict,
    persistence_checked,
    initials_checked,
    arima_checked,
    explanatory_checked,
    phi_dict,
    constants_checked,
):
    """
    Fill model matrices with parameter values.

    Args:
        B: Vector of parameters to fill
        model_type_dict: Dictionary containing model type information
        components_dict: Dictionary containing component information
        lags_dict: Dictionary containing lags information
        matrices_dict: Dictionary containing model matrices
        persistence_checked: Dictionary of persistence parameters
        initials_checked: Dictionary of initial values
        arima_checked: Dictionary of ARIMA parameters
        explanatory_checked: Dictionary of explanatory variables parameters
        phi_dict: Dictionary containing phi parameters
        constants_checked: Dictionary containing constant parameters

    Returns:
        Dict: Dictionary containing filled matrices
    """
    # Extract matrices
    mat_vt = matrices_dict["mat_vt"].copy()
    mat_wt = matrices_dict["mat_wt"].copy()
    mat_f = matrices_dict["mat_f"].copy()
    vec_g = matrices_dict["vec_g"].copy()

    # Copy model parameters to avoid modifying original
    persistence = persistence_checked.copy()
    initials = initials_checked.copy()
    arima = arima_checked.copy()
    explanatory = explanatory_checked.copy()
    phi = phi_dict.copy()
    constants = constants_checked.copy()

    # Get model parameters
    ets_model = model_type_dict["ets_model"]
    model_is_trendy = model_type_dict["model_is_trendy"]
    model_is_seasonal = model_type_dict["model_is_seasonal"]
    components_number_ets = components_dict["components_number_ets"]
    components_number_ets_seasonal = components_dict["components_number_ets_seasonal"]
    components_number_arima = components_dict.get("components_number_arima", 0)

    # If no parameters to fill, return original matrices
    if len(B) == 0:
        return matrices_dict

    # Keep track of parameter index
    param_index = 0

    # Fill parameters in sequence
    param_index = _fill_persistence_parameters(
        B, param_index, persistence, ets_model, model_is_trendy, model_is_seasonal
    )

    param_index = _fill_arima_parameters(B, param_index, arima)

    param_index = _fill_explanatory_parameters(B, param_index, explanatory)

    param_index = _fill_phi_parameter(B, param_index, phi, ets_model, model_is_trendy)

    param_index = _fill_initial_states(
        B,
        param_index,
        initials,
        ets_model,
        model_is_trendy,
        model_is_seasonal,
        components_number_ets_seasonal,
        lags_dict,
    )

    param_index = _fill_arima_initial_states(B, param_index, initials, arima)

    param_index = _fill_constant_parameter(B, param_index, constants)

    # Update matrices with filled parameters
    matrices = _update_matrices_with_parameters(
        mat_vt,
        mat_wt,
        mat_f,
        vec_g,
        model_type_dict,
        components_dict,
        persistence,
        initials,
        arima,
        explanatory,
        phi,
        constants,
        lags_dict,
    )

    return matrices


def _fill_persistence_parameters(
    B, param_index, persistence, ets_model, model_is_trendy, model_is_seasonal
):
    """
    Fill persistence parameters from parameter vector.

    Args:
        B: Parameter vector
        param_index: Current parameter index
        persistence: Persistence parameters dictionary
        ets_model: Boolean indicating if ETS model is used
        model_is_trendy: Boolean indicating if model has trend
        model_is_seasonal: Boolean indicating if model has seasonality

    Returns:
        int: Updated parameter index
    """
    if ets_model:
        # Persistence for level
        if persistence["persistence_level_estimate"]:
            persistence["persistence_level"] = B[param_index]
            param_index += 1

        # Persistence for trend
        if model_is_trendy and persistence["persistence_trend_estimate"]:
            persistence["persistence_trend"] = B[param_index]
            param_index += 1

        # Persistence for seasonality
        if model_is_seasonal:
            for i in range(len(persistence["persistence_seasonal"])):
                if persistence["persistence_seasonal_estimate"][i]:
                    persistence["persistence_seasonal"][i] = B[param_index]
                    param_index += 1

    return param_index


def _fill_arima_parameters(B, param_index, arima):
    """
    Fill ARIMA parameters from parameter vector.

    Args:
        B: Parameter vector
        param_index: Current parameter index
        arima: ARIMA parameters dictionary

    Returns:
        int: Updated parameter index
    """
    if arima["arima_model"]:
        # AR parameters
        if arima["ar_estimate"]:
            # Check if we already have parameters
            if "arma_parameters" not in arima:
                arima["arma_parameters"] = []

            # Get total AR parameters
            ar_params_count = sum(arima["ar_orders"])

            # Fill AR parameters
            for i in range(ar_params_count):
                if param_index < len(B):
                    arima["arma_parameters"].append(B[param_index])
                    param_index += 1

        # MA parameters
        if arima["ma_estimate"]:
            # If no AR parameters were estimated, initialize the array
            if "arma_parameters" not in arima:
                arima["arma_parameters"] = []

            # Get total MA parameters
            ma_params_count = sum(arima["ma_orders"])

            # Fill MA parameters
            for i in range(ma_params_count):
                if param_index < len(B):
                    arima["arma_parameters"].append(B[param_index])
                    param_index += 1

    return param_index


def _fill_explanatory_parameters(B, param_index, explanatory):
    """
    Fill explanatory variable parameters from parameter vector.

    Args:
        B: Parameter vector
        param_index: Current parameter index
        explanatory: Explanatory variables parameters dictionary

    Returns:
        int: Updated parameter index
    """
    if explanatory["xreg_model"] and explanatory["xreg_estimate"]:
        if "xreg_parameters" not in explanatory:
            explanatory["xreg_parameters"] = []

        for i in range(explanatory["xreg_number"]):
            if param_index < len(B):
                explanatory["xreg_parameters"].append(B[param_index])
                param_index += 1

    return param_index


def _fill_phi_parameter(B, param_index, phi, ets_model, model_is_trendy):
    """
    Fill phi (damping) parameter from parameter vector.

    Args:
        B: Parameter vector
        param_index: Current parameter index
        phi: Phi parameters dictionary
        ets_model: Boolean indicating if ETS model is used
        model_is_trendy: Boolean indicating if model has trend

    Returns:
        int: Updated parameter index
    """
    if ets_model and model_is_trendy and phi["phi_estimate"]:
        if param_index < len(B):
            phi["phi"] = B[param_index]
            param_index += 1

    return param_index


def _fill_initial_states(
    B,
    param_index,
    initials,
    ets_model,
    model_is_trendy,
    model_is_seasonal,
    components_number_ets_seasonal,
    lags_dict,
):
    """
    Fill initial states from parameter vector.

    Args:
        B: Parameter vector
        param_index: Current parameter index
        initials: Initial values dictionary
        ets_model: Boolean indicating if ETS model is used
        model_is_trendy: Boolean indicating if model has trend
        model_is_seasonal: Boolean indicating if model has seasonality
        components_number_ets_seasonal: Number of seasonal components in ETS model
        lags_dict: Dictionary containing lags information

    Returns:
        int: Updated parameter index
    """
    if ets_model:
        # Initial level
        if initials["initial_level_estimate"]:
            if param_index < len(B):
                initials["initial_level"] = B[param_index]
                param_index += 1

        # Initial trend
        if model_is_trendy and initials["initial_trend_estimate"]:
            if param_index < len(B):
                initials["initial_trend"] = B[param_index]
                param_index += 1

        # Initial seasonal
        if model_is_seasonal and initials["initial_seasonal_estimate"]:
            if "initial_seasonal" not in initials:
                initials["initial_seasonal"] = [None] * components_number_ets_seasonal

            for i in range(components_number_ets_seasonal):
                seasonal_i = []
                for j in range(lags_dict["lags_model"][i + 1 + int(model_is_trendy)]):
                    if param_index < len(B):
                        seasonal_i.append(B[param_index])
                        param_index += 1

                if len(seasonal_i) > 0:
                    initials["initial_seasonal"][i] = seasonal_i

    return param_index


def _fill_arima_initial_states(B, param_index, initials, arima):
    """
    Fill ARIMA initial states from parameter vector.

    Args:
        B: Parameter vector
        param_index: Current parameter index
        initials: Initial values dictionary
        arima: ARIMA parameters dictionary

    Returns:
        int: Updated parameter index
    """
    if arima["arima_model"] and initials["initial_arima_estimate"]:
        initial_arima_values = []

        for i in range(initials["initial_arima_number"]):
            if param_index < len(B):
                initial_arima_values.append(B[param_index])
                param_index += 1

        if len(initial_arima_values) > 0:
            initials["initial_arima"] = initial_arima_values

    return param_index


def _fill_constant_parameter(B, param_index, constants):
    """
    Fill constant parameter from parameter vector.

    Args:
        B: Parameter vector
        param_index: Current parameter index
        constants: Constants parameters dictionary

    Returns:
        int: Updated parameter index
    """
    if constants["constant_required"] and constants["constant_estimate"]:
        if param_index < len(B):
            constants["constant_value"] = B[param_index]
            param_index += 1

    return param_index


def _update_matrices_with_parameters(
    mat_vt,
    mat_wt,
    mat_f,
    vec_g,
    model_type_dict,
    components_dict,
    persistence,
    initials,
    arima,
    explanatory,
    phi,
    constants,
    lags_dict,
):
    """
    Update matrices with filled parameters.

    Args:
        mat_vt: State matrix
        mat_wt: Measurement matrix
        mat_f: Transition matrix
        vec_g: Persistence vector
        model_type_dict: Dictionary containing model type information
        components_dict: Dictionary containing component information
        persistence: Filled persistence parameters dictionary
        initials: Filled initial values dictionary
        arima: Filled ARIMA parameters dictionary
        explanatory: Filled explanatory variables parameters dictionary
        phi: Filled phi parameters dictionary
        constants: Filled constants parameters dictionary
        lags_dict: Dictionary containing lags information

    Returns:
        Dict: Dictionary containing updated matrices
    """
    # Get model parameters
    ets_model = model_type_dict["ets_model"]
    model_is_trendy = model_type_dict["model_is_trendy"]
    error_type = model_type_dict["error_type"]
    components_number_ets = components_dict["components_number_ets"]
    components_number_arima = components_dict.get("components_number_arima", 0)
    lags = lags_dict["lags"]
    lags_model_max = lags_dict["lags_model_max"]

    # Update with persistence parameters
    if ets_model:
        j = 0
        # Level persistence
        vec_g[j, 0] = persistence["persistence_level"]
        j += 1

        # Trend persistence
        if model_is_trendy:
            vec_g[j, 0] = persistence["persistence_trend"]
            j += 1

        # Seasonal persistence
        for gamma in persistence["persistence_seasonal"]:
            vec_g[j, 0] = gamma
            j += 1

        # Damping parameter
        if model_is_trendy:
            mat_f[0, 1] = phi["phi"]
            mat_f[1, 1] = phi["phi"]
            mat_wt[:, 1] = phi["phi"]

        # Initial states
        j = 0
        mat_vt[j, 0:lags_model_max] = initials["initial_level"]
        j += 1

        if model_is_trendy:
            mat_vt[j, 0:lags_model_max] = initials["initial_trend"]
            j += 1

        if model_type_dict["model_is_seasonal"]:
            for i, seasonal in enumerate(initials["initial_seasonal"]):
                if seasonal is not None:
                    mat_vt[j + i, 0 : len(seasonal)] = seasonal

    # Update with ARIMA parameters
    if arima["arima_model"]:
        # Handle ARIMA parameters
        if "arma_parameters" in arima and len(arima["arma_parameters"]) > 0:
            # Generate polynomials from parameters
            arima_polynomials = {
                key: np.array(value)
                for key, value in adam_polynomialiser(
                    0,
                    arima["ar_orders"],
                    arima["i_orders"],
                    arima["ma_orders"],
                    arima["ar_estimate"],
                    arima["ma_estimate"],
                    arima["arma_parameters"],
                    lags,
                ).items()
            }

            # Fill in the transition matrix
            if len(arima["non_zero_ari"]) > 0:
                non_zero_ari = np.array(arima["non_zero_ari"])
                mat_f[
                    components_number_ets + non_zero_ari[:, 1],
                    components_number_ets + non_zero_ari[:, 1],
                ] = -arima_polynomials["ari_polynomial"][non_zero_ari[:, 0]]

            # Fill in the persistence vector
            if len(arima["non_zero_ari"]) > 0:
                non_zero_ari = np.array(arima["non_zero_ari"])
                vec_g[components_number_ets + non_zero_ari[:, 1], 0] = (
                    -arima_polynomials["ari_polynomial"][non_zero_ari[:, 0]]
                )

            if len(arima["non_zero_ma"]) > 0:
                non_zero_ma = np.array(arima["non_zero_ma"])
                vec_g[
                    components_number_ets + non_zero_ma[:, 1], 0
                ] += arima_polynomials["ma_polynomial"][non_zero_ma[:, 0]]

        # Initial ARIMA states
        if "initial_arima" in initials and initials["initial_arima"] is not None:
            mat_vt[
                components_number_ets : components_number_ets + components_number_arima,
                0 : min(
                    len(initials["initial_arima"]), initials["initial_arima_number"]
                ),
            ] = np.array(
                initials["initial_arima"][: initials["initial_arima_number"]]
            ).reshape(
                1, -1
            )

    # Update with explanatory variables parameters
    if (
        explanatory["xreg_model"]
        and "xreg_parameters" in explanatory
        and len(explanatory["xreg_parameters"]) > 0
    ):
        # Fill in the persistence vector
        for i, beta in enumerate(explanatory["xreg_parameters"]):
            if i < explanatory["xreg_number"]:
                vec_g[components_number_ets + components_number_arima + i, 0] = beta

    # Update with constant parameter
    if constants["constant_required"]:
        # Fill in constant value
        if "constant_value" in constants:
            mat_vt[
                components_number_ets
                + components_number_arima
                + explanatory["xreg_number"],
                :,
            ] = constants["constant_value"]

        # Modify transition for drift
        if not arima["arima_model"]:
            mat_f[0, -1] = 1

    return {"mat_vt": mat_vt, "mat_wt": mat_wt, "mat_f": mat_f, "vec_g": vec_g}


def architector(
    # Model type info
    model_type_dict: Dict[str, Any],
    # Lags info
    lags_dict: Dict[str, Any],
    # Observation info
    observations_dict: Dict[str, Any],
    # Optional model components
    arima_checked: Dict[str, Any] = None,
    explanatory_checked: Dict[str, Any] = None,
    constants_checked: Dict[str, Any] = None,
    # Profiles
    profiles_recent_table: Union[np.ndarray, None] = None,
    profiles_recent_provided: bool = False,
) -> Dict[str, Any]:
    """
    Set up the model architecture for ADAM.

    Args:
        model_type_dict: Dictionary containing model type information
        lags_dict: Dictionary containing lags information
        observations_dict: Dictionary containing observation information
        arima_checked: Dictionary of ARIMA parameters
        explanatory_checked: Dictionary of explanatory variables parameters
        constants_checked: Dictionary of constant parameters
        profiles_recent_table: Table of recent profiles
        profiles_recent_provided: Whether recent profiles are provided

    Returns:
        Tuple: Updated model dictionaries
    """
    # Set up components for the model
    components_dict = _setup_components(model_type_dict, arima_checked)

    # Set up lags
    lags_dict = _setup_lags(lags_dict, model_type_dict, components_dict)

    # Set up profiles
    profiles_dict = _create_profiles(
        profiles_recent_provided, profiles_recent_table, lags_dict, observations_dict
    )

    return model_type_dict, components_dict, lags_dict, observations_dict, profiles_dict


def _setup_components(model_type_dict, arima_checked):
    """
    Set up components for the model architecture.

    Args:
        model_type_dict: Dictionary containing model type information
        arima_checked: Dictionary of ARIMA parameters

    Returns:
        Dict: Dictionary containing component information
    """
    # Initialize components dict
    components_dict = {}

    # Determine ETS components
    if model_type_dict["ets_model"]:
        # Basic number of components: level is always present
        components_number_ets = 1

        # Add trend component if needed
        if model_type_dict["model_is_trendy"]:
            components_number_ets += 1

        # Add seasonal components if needed
        components_number_ets_seasonal = 0
        if model_type_dict["model_is_seasonal"]:
            # Count number of seasonal components
            components_number_ets_seasonal = len(
                [lag for lag in model_type_dict.get("seasonal_periods", []) if lag > 1]
            )
            components_number_ets += components_number_ets_seasonal

        # Store in dictionary
        components_dict["components_number_ets"] = components_number_ets
        components_dict["components_number_ets_seasonal"] = (
            components_number_ets_seasonal
        )
    else:
        # No ETS components
        components_dict["components_number_ets"] = 0
        components_dict["components_number_ets_seasonal"] = 0

    # Determine ARIMA components
    if arima_checked and arima_checked["arima_model"]:
        # Total number of ARIMA components
        components_number_arima = sum(arima_checked["ar_orders"]) + sum(
            arima_checked["ma_orders"]
        )
        components_dict["components_number_arima"] = components_number_arima
    else:
        components_dict["components_number_arima"] = 0

    return components_dict


def _setup_lags(lags_dict, model_type_dict, components_dict):
    """
    Set up lags for the model architecture.

    Args:
        lags_dict: Dictionary containing lags information
        model_type_dict: Dictionary containing model type information
        components_dict: Dictionary containing component information

    Returns:
        Dict: Updated lags dictionary
    """
    # Extract parameters
    lags = lags_dict["lags"]

    # Calculate model lags for each component
    lags_model = []

    # ETS components
    if model_type_dict["ets_model"]:
        # Level always has lag 1
        lags_model.append(1)

        # Trend component has lag 1
        if model_type_dict["model_is_trendy"]:
            lags_model.append(1)

        # Seasonal components have lags corresponding to seasonal periods
        if model_type_dict["model_is_seasonal"]:
            for lag in lags:
                if lag > 1:
                    lags_model.append(lag)

    # ARIMA components
    lags_model_arima = []
    if (
        "components_number_arima" in components_dict
        and components_dict["components_number_arima"] > 0
    ):
        max_lag = max(lags)
        lags_model_arima = [max_lag] * components_dict["components_number_arima"]

    # Combine all lags
    lags_model_all = lags_model + lags_model_arima

    # Find maximum lag
    lags_model_max = max(lags_model_all) if lags_model_all else 1

    # Update lags dictionary
    lags_dict_updated = lags_dict.copy()
    lags_dict_updated["lags_model"] = lags_model
    lags_dict_updated["lags_model_arima"] = lags_model_arima
    lags_dict_updated["lags_model_all"] = lags_model_all
    lags_dict_updated["lags_model_max"] = lags_model_max

    return lags_dict_updated


def _create_profiles(
    profiles_recent_provided, profiles_recent_table, lags_dict, observations_dict
):
    """
    Create profiles for the model architecture.

    Args:
        profiles_recent_provided: Whether recent profiles are provided
        profiles_recent_table: Table of recent profiles
        lags_dict: Dictionary containing lags information
        observations_dict: Dictionary containing observation information

    Returns:
        Dict: Dictionary containing profile information
    """
    # Initialize profiles dictionary
    profiles_dict = {
        "profiles_recent_provided": profiles_recent_provided,
        "profiles_recent_table": profiles_recent_table,
    }

    # If profiles are not provided, create them
    if not profiles_recent_provided:
        # Create profile matrices
        profiles = adam_profile_creator(
            lags_model_all=lags_dict["lags_model_all"],
            lags_model_max=lags_dict["lags_model_max"],
            obs_all=observations_dict["obs_all"],
            lags=lags_dict["lags"],
            y_index=observations_dict.get("y_index", None),
            y_classes=observations_dict.get("y_classes", None),
        )

        # Store profiles in dictionary
        profiles_dict["profiles_recent_table"] = profiles["profiles_recent_table"]

    return profiles_dict


def adam_profile_creator(
    lags_model_all: List[List[int]],
    lags_model_max: int,
    obs_all: int,
    lags: Union[List[int], None] = None,
    y_index: Union[List, None] = None,
    y_classes: Union[List, None] = None,
) -> Dict[str, np.ndarray]:
    """
    Create profile matrices for ADAM.

    Args:
        lags_model_all: List of all model lags
        lags_model_max: Maximum lag in the model
        obs_all: Total number of observations
        lags: List of seasonal periods
        y_index: Index for time series
        y_classes: Classes for time series

    Returns:
        Dict: Dictionary containing profile matrices
    """
    # Determine profile dimensions
    profile_dims = _determine_profile_dimensions(lags_model_all, obs_all)

    # Initialize profile matrices
    profiles = _initialize_profile_matrices(profile_dims)

    # Fill profile matrices
    profiles = _fill_profile_matrices(
        profiles, lags_model_all, lags_model_max, obs_all, lags, y_index, y_classes
    )

    return profiles


def _determine_profile_dimensions(lags_model_all, obs_all):
    """
    Determine dimensions for profile matrices.

    Args:
        lags_model_all: List of all model lags
        obs_all: Total number of observations

    Returns:
        Dict: Dictionary containing profile dimensions
    """
    # Get number of components from the lags
    n_components = len(lags_model_all)

    # Calculate total number of lagged components
    n_lagged_components = sum(lags_model_all)

    # Calculate total observation space
    n_obs_all = obs_all

    return {
        "n_components": n_components,
        "n_lagged_components": n_lagged_components,
        "n_obs_all": n_obs_all,
    }


def _initialize_profile_matrices(profile_dims):
    """
    Initialize profile matrices.

    Args:
        profile_dims: Dictionary containing profile dimensions

    Returns:
        Dict: Dictionary containing initialized profile matrices
    """
    n_components = profile_dims["n_components"]
    n_lagged_components = profile_dims["n_lagged_components"]
    n_obs_all = profile_dims["n_obs_all"]

    # Initialize recent profiles table
    # Using dimensions consistent with original implementation (n_components x lags_model_max)
    profiles_recent_table = np.zeros((n_components, n_lagged_components))

    return {"profiles_recent_table": profiles_recent_table}


def _fill_profile_matrices(
    profiles, lags_model_all, lags_model_max, obs_all, lags, y_index, y_classes
):
    """
    Fill profile matrices with proper values.

    Args:
        profiles: Dictionary containing profile matrices
        lags_model_all: List of all model lags
        lags_model_max: Maximum lag in the model
        obs_all: Total number of observations
        lags: List of seasonal periods
        y_index: Index for time series
        y_classes: Classes for time series

    Returns:
        Dict: Dictionary containing filled profile matrices
    """
    # Initialize column counter
    col = 0

    # Loop through each component
    for i, component_lags in enumerate(lags_model_all):
        # Loop through lags for this component
        for j in range(component_lags):
            # Generate index values for this lag
            idx_values = _generate_lagged_indexes(
                i, j, lags_model_all, lags_model_max, obs_all, lags, y_index, y_classes
            )

            # Fill the profile table with these values
            profiles["profiles_recent_table"][:, col] = idx_values

            # Update column counter
            col += 1

    return profiles


def _generate_lagged_indexes(
    i, j, lags_model_all, lags_model_max, obs_all, lags, y_index, y_classes
):
    """
    Generate lagged indexes for a specific component and lag.

    Args:
        i: Component index
        j: Lag index
        lags_model_all: List of all model lags
        lags_model_max: Maximum lag in the model
        obs_all: Total number of observations
        lags: List of seasonal periods
        y_index: Index for time series
        y_classes: Classes for time series

    Returns:
        np.ndarray: Array of lagged indexes
    """
    # Initialize index values
    idx_values = np.zeros(obs_all)

    # Calculate lag
    if i < len(lags_model_all) and lags_model_all[i] > 0:
        lag = j + 1
    else:
        lag = 1

    # Fill index values
    for t in range(obs_all):
        if t >= lags_model_max - lag:
            t_lag = t - lag
            if t_lag >= 0:
                idx_values[t] = t_lag + 1
            else:
                idx_values[t] = 0
        else:
            idx_values[t] = 0

    return idx_values
