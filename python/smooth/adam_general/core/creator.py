import numpy as np
from typing import List, Dict, Union, Any
from scipy.optimize import minimize

from smooth.adam_general.core.utils.utils import msdecompose, calculate_acf, calculate_pacf
from smooth.adam_general.core.utils.polynomials import adam_polynomialiser

import warnings
# Suppress divide by zero warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')


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
    obs_nonzero = observations_dict['obs_nonzero']

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
        y_in_sample.ravel(),
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
        mat_vt[0, :lags_model_max] = np.mean(y_in_sample[:max(lags_model_max, int(np.ceil(obs_in_sample * 0.2)))])
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
    Initialize parameters for the ADAM model. Determines initial parameter
    values (B) and their bounds (Bl, Bu) for optimization.

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
        other: Other parameters (currently unused)

    Returns:
        Dict: Dictionary containing initialized parameters B, Bl, Bu and names.
              Example: {'B': B, 'Bl': Bl, 'Bu': Bu, 'names': names}
    """
    persistence_estimate_vector = [
        persistence_checked['persistence_level_estimate'],
        model_type_dict["model_is_trendy"] and persistence_checked['persistence_trend_estimate'],
        model_type_dict["model_is_seasonal"] and any(persistence_checked['persistence_seasonal_estimate'])
    ]
    total_params = (
        model_type_dict["ets_model"] * (sum(persistence_estimate_vector) + phi_dict['phi_estimate']) +
        explanatory_checked['xreg_model'] * persistence_checked['persistence_xreg_estimate'] * max(explanatory_checked['xreg_parameters_persistence'] or [0]) +
        arima_checked['arima_model'] * (arima_checked['ar_estimate'] * sum(arima_checked['ar_orders'] or []) + arima_checked['ma_estimate'] * sum(arima_checked['ma_orders'] or [])) +
        model_type_dict["ets_model"] * (initials_checked['initial_type'] not in ["complete", "backcasting"]) * (
            initials_checked['initial_level_estimate'] +
            (model_type_dict["model_is_trendy"] * initials_checked['initial_trend_estimate']) +
            (model_type_dict["model_is_seasonal"] * sum(initials_checked['initial_seasonal_estimate'] * (np.array(lags_dict["lags_model_seasonal"] or []) - 1)))
        ) +
        (initials_checked['initial_type'] not in ["complete", "backcasting"]) * arima_checked['arima_model'] * (initials_checked['initial_arima_number'] or 0) * initials_checked['initial_arima_estimate'] +
        (initials_checked['initial_type'] != "complete") * explanatory_checked['xreg_model'] * initials_checked['initial_xreg_estimate'] * sum(explanatory_checked['xreg_parameters_estimated'] or []) +
        constants_checked['constant_estimate']
    )

    B = np.zeros(total_params)
    Bl = np.zeros(total_params)
    Bu = np.zeros(total_params)
    names = []

    j = 0

    if model_type_dict["ets_model"]:
        if persistence_checked['persistence_estimate'] and any(persistence_estimate_vector):
            if any(ptype == "M" for ptype in [model_type_dict["error_type"], model_type_dict["trend_type"], model_type_dict["season_type"]]):
                if ((model_type_dict["error_type"] == "A" and model_type_dict["trend_type"] == "A" and model_type_dict["season_type"] == "M") or
                    (model_type_dict["error_type"] == "A" and model_type_dict["trend_type"] == "M" and model_type_dict["season_type"] == "A") or
                    (initials_checked['initial_type'] in ["complete", "backcasting"] and
                     ((model_type_dict["error_type"] == "M" and model_type_dict["trend_type"] == "A" and model_type_dict["season_type"] == "A") or
                      (model_type_dict["error_type"] == "M" and model_type_dict["trend_type"] == "A" and model_type_dict["season_type"] == "M")))):
                    B[j:j+sum(persistence_estimate_vector)] = [0.01, 0] + [0] * components_dict["components_number_ets_seasonal"]
                elif model_type_dict["error_type"] == "M" and model_type_dict["trend_type"] == "M" and model_type_dict["season_type"] == "A":
                    B[j:j+sum(persistence_estimate_vector)] = [0, 0] + [0] * components_dict["components_number_ets_seasonal"]
                elif model_type_dict["error_type"] == "M" and model_type_dict["trend_type"] == "A":
                    if initials_checked['initial_type'] in ["complete", "backcasting"]:
                        B[j:j+sum(persistence_estimate_vector)] = [0.1, 0] + [0.3] * components_dict["components_number_ets_seasonal"]
                    else:
                        B[j:j+sum(persistence_estimate_vector)] = [0.2, 0.01] + [0.3] * components_dict["components_number_ets_seasonal"]
                elif model_type_dict["error_type"] == "M" and model_type_dict["trend_type"] == "M":
                    B[j:j+sum(persistence_estimate_vector)] = [0.1, 0.05] + [0.3] * components_dict["components_number_ets_seasonal"]
                else:
                    initial_values = [0.1]
                    if model_type_dict["model_is_trendy"]:
                        initial_values.append(0.05)
                    if model_type_dict["model_is_seasonal"]:
                        initial_values.extend([0.3] * components_dict["components_number_ets_seasonal"])
                    
                    B[j:j+sum(persistence_estimate_vector)] = [val for val, estimate in zip(initial_values, persistence_estimate_vector) if estimate]
            
            else:
                
                initial_values = [0.1]
                if model_type_dict["model_is_trendy"]:
                    initial_values.append(0.05)
                if model_type_dict["model_is_seasonal"]:
                    initial_values.extend([0.3] * components_dict["components_number_ets_seasonal"])
                
                B[j:j+sum(persistence_estimate_vector)] = [val for val, estimate in zip(initial_values, persistence_estimate_vector) if estimate]

            if bounds == "usual":
                Bl[j:j+sum(persistence_estimate_vector)] = 0
                Bu[j:j+sum(persistence_estimate_vector)] = 1
            else:
                Bl[j:j+sum(persistence_estimate_vector)] = -5
                Bu[j:j+sum(persistence_estimate_vector)] = 5

            # Names for B
            if persistence_checked['persistence_level_estimate']:
                names.append("alpha")
                j += 1
            if model_type_dict["model_is_trendy"] and persistence_checked['persistence_trend_estimate']:
                names.append("beta")
                j += 1
            if model_type_dict["model_is_seasonal"] and any(persistence_checked['persistence_seasonal_estimate']):
                if components_dict["components_number_ets_seasonal"] > 1:
                    names.extend([f"gamma{i}" for i in range(1, components_dict["components_number_ets_seasonal"]+1)])
                else:
                    names.append("gamma")
                j += sum(persistence_checked['persistence_seasonal_estimate'])

    if explanatory_checked['xreg_model'] and persistence_checked['persistence_xreg_estimate']:
        xreg_persistence_number = max(explanatory_checked['xreg_parameters_persistence'])
        B[j:j+xreg_persistence_number] = 0.01 if model_type_dict["error_type"] == "A" else 0
        Bl[j:j+xreg_persistence_number] = -5
        Bu[j:j+xreg_persistence_number] = 5
        names.extend([f"delta{i+1}" for i in range(xreg_persistence_number)])
        j += xreg_persistence_number

    if model_type_dict["ets_model"] and phi_dict['phi_estimate']:
        B[j] = 0.95
        names.append("phi")
        Bl[j] = 0
        Bu[j] = 1
        j += 1

    if arima_checked['arima_model']:
        if any([arima_checked['ar_estimate'], arima_checked['ma_estimate']]):
            acf_values = [-0.1] * sum(arima_checked['ma_orders'] * lags_dict["lags"])
            pacf_values = [0.1] * sum(arima_checked['ar_orders'] * lags_dict["lags"])
            
            if not (model_type_dict["ets_model"] or all(arima_checked['i_orders'] == 0)):
                y_differenced = observations_dict['y_in_sample'].copy()
                # Implement differencing if needed
                if any(arima_checked['i_orders'] > 0):
                    for i, order in enumerate(arima_checked['i_orders']):
                        if order > 0:
                            y_differenced = np.diff(y_differenced, n=order, axis=0)
                
                # ACF/PACF calculation for non-seasonal models
                if all(np.array(lags_dict["lags"]) <= 1):
                    if arima_checked['ma_required'] and arima_checked['ma_estimate']:
                        acf_values[:min(sum(arima_checked['ma_orders'] * lags_dict["lags"]), len(y_differenced) - 1)] = calculate_acf(y_differenced, nlags=max(1, sum(arima_checked['ma_orders'] * lags_dict["lags"])))[1:]
                    if arima_checked['ar_required'] and arima_checked['ar_estimate']:
                        pacf_values[:min(sum(arima_checked['ar_orders'] * lags_dict["lags"]), len(y_differenced) - 1)] = calculate_pacf(y_differenced, nlags=max(1, sum(arima_checked['ar_orders'] * lags_dict["lags"])))
            
            for i, lag in enumerate(lags_dict["lags"]):
                if arima_checked['ar_required'] and arima_checked['ar_estimate'] and arima_checked['ar_orders'][i] > 0:
                    B[j:j+arima_checked['ar_orders'][i]] = pacf_values[i*lag:(i+1)*lag][:arima_checked['ar_orders'][i]]
                    if sum(B[j:j+arima_checked['ar_orders'][i]]) > 1:
                        B[j:j+arima_checked['ar_orders'][i]] = B[j:j+arima_checked['ar_orders'][i]] / sum(B[j:j+arima_checked['ar_orders'][i]]) - 0.01
                    Bl[j:j+arima_checked['ar_orders'][i]] = -5
                    Bu[j:j+arima_checked['ar_orders'][i]] = 5
                    names.extend([f"phi{k+1}[{lag}]" for k in range(arima_checked['ar_orders'][i])])
                    j += arima_checked['ar_orders'][i]
                
                if arima_checked['ma_required'] and arima_checked['ma_estimate'] and arima_checked['ma_orders'][i] > 0:
                    B[j:j+arima_checked['ma_orders'][i]] = acf_values[i*lag:(i+1)*lag][:arima_checked['ma_orders'][i]]
                    if sum(B[j:j+arima_checked['ma_orders'][i]]) > 1:
                        B[j:j+arima_checked['ma_orders'][i]] = B[j:j+arima_checked['ma_orders'][i]] / sum(B[j:j+arima_checked['ma_orders'][i]]) - 0.01
                    Bl[j:j+arima_checked['ma_orders'][i]] = -5
                    Bu[j:j+arima_checked['ma_orders'][i]] = 5
                    names.extend([f"theta{k+1}[{lag}]" for k in range(arima_checked['ma_orders'][i])])
                    j += arima_checked['ma_orders'][i]

    if model_type_dict["ets_model"] and initials_checked['initial_type'] not in ["complete", "backcasting"] and initials_checked['initial_estimate']:
        if initials_checked['initial_level_estimate']:
            B[j] = adam_created['mat_vt'][0, 0]
            Bl[j] = -np.inf if model_type_dict["error_type"] == "A" else 0
            Bu[j] = np.inf
            names.append("level")
            j += 1
        if model_type_dict["model_is_trendy"] and initials_checked['initial_trend_estimate']:
            B[j] = adam_created['mat_vt'][1, 0]
            Bl[j] = -np.inf if model_type_dict["trend_type"] == "A" else 0
            Bu[j] = np.inf
            names.append("trend")
            j += 1

        
        if model_type_dict["model_is_seasonal"] and (isinstance(initials_checked['initial_seasonal_estimate'], bool) and initials_checked['initial_seasonal_estimate'] or isinstance(initials_checked['initial_seasonal_estimate'], list) and any(initials_checked['initial_seasonal_estimate'])):
            if components_dict['components_number_ets_seasonal'] > 1:
                for k in range(components_dict['components_number_ets_seasonal']):
                    if initials_checked['initial_seasonal_estimate'][k]:
                        # Get the correct seasonal component index and lag
                        seasonal_index = components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'] + k
                        lag = lags_dict['lags'][seasonal_index]

                        # Get the values from mat_vt (make sure dimensions match)
                        seasonal_values = adam_created['mat_vt'][seasonal_index, :lag-1]

                        # Assign to B with matching dimensions
                        B[j:j+lag-1] = seasonal_values
                        
                        if model_type_dict['season_type'] == "A":
                            Bl[j:j+lag-1] = -np.inf
                            Bu[j:j+lag-1] = np.inf
                        else:
                            Bl[j:j+lag-1] = 0
                            Bu[j:j+lag-1] = np.inf
                        names.extend([f"seasonal_{m}" for m in range(2, lag)])
                        j += lag - 1
            else:
                
                # Get the correct seasonal component index and lag
                seasonal_index = components_dict["components_number_ets"] - 1
                temp_lag = lags_dict["lags_model"][seasonal_index]
                seasonal_values = adam_created['mat_vt'][seasonal_index, :temp_lag-1]
                # Assign to B with matching dimensions
                B[j:j+temp_lag-1] = seasonal_values
                if model_type_dict['season_type'] == "A":
                    Bl[j:j+temp_lag-1] = -np.inf
                    Bu[j:j+temp_lag-1] = np.inf
                else:
                    Bl[j:j+temp_lag-1] = 0
                    Bu[j:j+temp_lag-1] = np.inf
                #names.extend([f"seasonal_{m}" for m in range(2, temp_lag)])
                j += temp_lag - 1
    if initials_checked['initial_type'] not in ["complete", "backcasting"] and arima_checked['arima_model'] and initials_checked['initial_arima_estimate']:
        B[j:j+initials_checked['initial_arima_number']] = adam_created['mat_vt'][components_dict["components_number_ets"] + components_dict["components_number_arima"], :initials_checked['initial_arima_number']]
        names.extend([f"ARIMAState{n}" for n in range(1, initials_checked['initial_arima_number']+1)])
        if model_type_dict["error_type"] == "A":
            Bl[j:j+initials_checked['initial_arima_number']] = -np.inf
            Bu[j:j+initials_checked['initial_arima_number']] = np.inf
        else:
            B[j:j+initials_checked['initial_arima_number']] = np.abs(B[j:j+initials_checked['initial_arima_number']])
            Bl[j:j+initials_checked['initial_arima_number']] = 0
            Bu[j:j+initials_checked['initial_arima_number']] = np.inf
        j += initials_checked['initial_arima_number']

    if initials_checked['initial_type'] != "complete" and initials_checked['initial_xreg_estimate'] and explanatory_checked['xreg_model']:
        xreg_number_to_estimate = sum(explanatory_checked['xreg_parameters_estimated'])
        if xreg_number_to_estimate > 0:
            B[j:j+xreg_number_to_estimate] = adam_created['mat_vt'][components_dict["components_number_ets"] + components_dict["components_number_arima"], 0]
            names.extend([f"xreg{idx+1}" for idx in range(xreg_number_to_estimate)])
            Bl[j:j+xreg_number_to_estimate] = -np.inf
            Bu[j:j+xreg_number_to_estimate] = np.inf
            j += xreg_number_to_estimate

    if constants_checked['constant_estimate']:
        j += 1
        if adam_created['mat_vt'].shape[0] > components_dict["components_number_ets"] + components_dict["components_number_arima"] + explanatory_checked['xreg_number']:
            B[j-1] = adam_created['mat_vt'][components_dict["components_number_ets"] + components_dict["components_number_arima"] + explanatory_checked['xreg_number'], 0]
        else:
            B[j-1] = 0  # or some other default value
        names.append(constants_checked['constant_name'] or "constant")
        if model_type_dict["ets_model"] or (arima_checked['i_orders'] is not None and sum(arima_checked['i_orders']) != 0):
            if model_type_dict["error_type"] == "A":
                Bu[j-1] = np.quantile(np.diff(observations_dict['y_in_sample'][observations_dict['ot_logical']], axis=0), 0.6)
                Bl[j-1] = -Bu[j-1]
            else:
                Bu[j-1] = np.exp(np.quantile(np.diff(np.log(observations_dict['y_in_sample'][observations_dict['ot_logical']]), axis=0), 0.6))
                Bl[j-1] = np.exp(np.quantile(np.diff(np.log(observations_dict['y_in_sample'][observations_dict['ot_logical']]), axis=0), 0.4))
            
            if Bu[j-1] <= Bl[j-1]:
                Bu[j-1] = np.inf
                Bl[j-1] = -np.inf if model_type_dict["error_type"] == "A" else 0
            
            if B[j-1] <= Bl[j-1]:
                Bl[j-1] = -np.inf if model_type_dict["error_type"] == "A" else 0
            if B[j-1] >= Bu[j-1]:
                Bu[j-1] = np.inf
        else:
            Bu[j-1] = max(abs(observations_dict['y_in_sample'][observations_dict['ot_logical']]), abs(B[j-1]) * 1.01)
            Bl[j-1] = -Bu[j-1]

    # assuming no other parameters for now
    #if initials_checked['other_parameter_estimate']:
    #    j += 1
    #    B[j-1] = other
    #    names.append("other")
    #    Bl[j-1] = 1e-10
    #    Bu[j-1] = np.inf
    return {
        "B": B,
        "Bl": Bl,
        "Bu": Bu,
        "names": names
    }


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


def _calculate_initial_parameters_and_bounds(
    model_params,
    persistence_checked,
    initials_checked,
    arima_checked,
    constants_checked,
    explanatory_checked,
    phi_dict,
    bounds,
    adam_created, # Added
    observations_dict # Added
):
    """
    Calculate initial parameter vector B, bounds Bl and Bu, and names.
    Combines logic from old initialiser and _prepare_optimization_inputs.
    """
    # Extract relevant parameters from model_params and dicts
    ets_model = model_params['ets_model']
    model_is_trendy = model_params['model_is_trendy']
    model_is_seasonal = model_params['model_is_seasonal']
    error_type = model_params['error_type']
    trend_type = model_params['trend_type']
    season_type = model_params['season_type']
    components_number_ets = model_params['components_number_ets']
    components_number_ets_seasonal = model_params['components_number_ets_seasonal']
    lags_dict = {
        'lags': model_params['lags'],
        'lags_model': model_params['lags_model'],
        'lags_model_max': model_params['lags_model_max'],
        #'lags_model_seasonal': model_params.get('lags_model_seasonal') # Needs to be passed correctly
    }
    mat_vt = adam_created['mat_vt'] # Used for initial state estimates
    y_in_sample = observations_dict['y_in_sample']
    ot_logical = observations_dict['ot_logical']

    # Calculate the total number of parameters to estimate
    # --- Calculate persistence params ---
    est_level = persistence_checked.get('persistence_level_estimate', False)
    est_trend = model_is_trendy and persistence_checked.get('persistence_trend_estimate', False)
    seas_est_val = persistence_checked.get('persistence_seasonal_estimate', False)
    num_seasonal_persistence_params = 0 # Initialize
    if isinstance(seas_est_val, bool):
        est_seasonal = model_is_seasonal and seas_est_val
        # num_seasonal_persistence_params = 1 if est_seasonal else 0 # Old logic was sum of components
        if est_seasonal: # if est_seasonal is True, it means all seasonal components are estimated
            num_seasonal_persistence_params = components_number_ets_seasonal if components_number_ets_seasonal > 0 else 1 if model_is_seasonal else 0
    elif isinstance(seas_est_val, (list, np.ndarray)):
        est_seasonal = model_is_seasonal and any(seas_est_val)
        num_seasonal_persistence_params = sum(p for p in seas_est_val if p)
    else:
        est_seasonal = False
        num_seasonal_persistence_params = 0

    num_ets_persistence_params = sum([est_level, est_trend, num_seasonal_persistence_params if isinstance(seas_est_val, bool) and seas_est_val else num_seasonal_persistence_params])


    # --- Xreg persistence parameters (deltas) ---
    est_xreg_persistence = explanatory_checked.get('xreg_model', False) and persistence_checked.get('persistence_xreg_estimate', False)
    num_xreg_persistence_params = 0
    if est_xreg_persistence:
        # The old code used max(explanatory_checked['xreg_parameters_persistence'] or [0])
        # This implies xreg_parameters_persistence is a list of integers indicating which xreg variables have persistence
        # and the number of delta parameters is the max value in that list.
        # For simplicity, if persistence_xreg_estimate is True, we might assume one delta per xreg, or follow old logic if xreg_parameters_persistence is available.
        # The old code's logic for xreg_parameters_persistence was:
        # max(explanatory_checked['xreg_parameters_persistence'] or [0])
        # This suggests 'xreg_parameters_persistence' is a list like [1, 2] if 2 delta params
        # Let's assume for now if est_xreg_persistence is true, it's for all xreg components.
        # A more robust way would be to use explanatory_checked['xreg_parameters_persistence'] if available.
        # Given the old code: max(explanatory_checked['xreg_parameters_persistence'] or [0])
        # If 'xreg_parameters_persistence' is not in explanatory_checked or is empty, this defaults to 0.
        # This needs careful check against how 'xreg_parameters_persistence' is defined.
        # Let's assume xreg_parameters_persistence is a list like [1,1,0] for 3 xregs, 2 have persistence estimates
        if explanatory_checked.get('xreg_parameters_persistence'):
             num_xreg_persistence_params = sum(explanatory_checked.get('xreg_parameters_persistence', [])) # Number of true flags
        else: # Fallback if not defined, assume one per xreg if main flag is true
             num_xreg_persistence_params = explanatory_checked.get('xreg_number', 0)


    num_persistence_params = num_ets_persistence_params # This will be expanded

    # --- Calculate ARIMA params ---
    est_ar = arima_checked.get('ar_estimate', False)
    est_ma = arima_checked.get('ma_estimate', False)
    num_ar_params = sum(arima_checked.get('ar_orders', [])) if est_ar else 0
    num_ma_params = sum(arima_checked.get('ma_orders', [])) if est_ma else 0
    num_arima_params = num_ar_params + num_ma_params

    # --- Calculate Xreg params ---
    est_xreg = explanatory_checked.get('xreg_model', False) and explanatory_checked.get('xreg_estimate', False)
    num_xreg_params = explanatory_checked.get('xreg_number', 0) if est_xreg else 0

    # --- Calculate Phi param ---
    est_phi = ets_model and model_is_trendy and phi_dict.get('phi_estimate', False)

    # --- Calculate Initial state params ---
    initials_active = initials_checked.get('initial_type') not in ["complete", "backcasting"]
    est_init_level = ets_model and initials_active and initials_checked.get('initial_level_estimate', False)
    est_init_trend = ets_model and model_is_trendy and initials_active and initials_checked.get('initial_trend_estimate', False)

    est_init_seasonal_val = initials_checked.get('initial_seasonal_estimate', False)
    num_initial_seasonal_params = 0
    est_init_seasonal = False
    if model_is_seasonal and initials_active:
        if isinstance(est_init_seasonal_val, bool):
            if est_init_seasonal_val:
                 est_init_seasonal = True
                 # Needs lags_model to be accurate, using approximation
                 num_initial_seasonal_params = sum(lag - 1 for lag in model_params['lags_model'][1+int(model_is_trendy):])
        elif isinstance(est_init_seasonal_val, (list, np.ndarray)):
            if any(est_init_seasonal_val):
                 est_init_seasonal = True
                 lags_seasonal = model_params['lags_model'][1+int(model_is_trendy):]
                 num_initial_seasonal_params = sum(lags_seasonal[i] - 1 for i, est in enumerate(est_init_seasonal_val) if est)

    est_init_arima = initials_active and arima_checked.get('arima_model', False) and initials_checked.get('initial_arima_estimate', False)
    num_init_arima_params = initials_checked.get('initial_arima_number', 0) if est_init_arima else 0

    # --- Calculate Constant param ---
    est_constant = constants_checked.get('constant_estimate', False)

    # --- Summing up ---
    total_params = int(
        num_ets_persistence_params + # Changed from num_persistence_params
        num_xreg_persistence_params + # Added for delta
        num_arima_params +
        num_xreg_params +
        est_phi +
        est_init_level +
        est_init_trend +
        num_initial_seasonal_params +
        num_init_arima_params +
        est_constant
    )

    # Initialize arrays
    B = np.zeros(total_params)
    Bl = np.zeros(total_params)
    Bu = np.zeros(total_params)
    names = []
    param_idx = 0

    # --- Populate B, Bl, Bu, names based on old initialiser logic ---

    # ETS Persistence Parameters
    if ets_model:
        # Level Persistence (alpha)
        if est_level:
            B[param_idx] = 0.1 # Default initial value
            if bounds == "usual": Bl[param_idx], Bu[param_idx] = 0, 1
            else: Bl[param_idx], Bu[param_idx] = -5, 5 # Old code's else
            names.append("alpha")
            param_idx += 1

        # Trend Persistence (beta)
        if est_trend:
            B[param_idx] = 0.05 # Default initial value
            if bounds == "usual": Bl[param_idx], Bu[param_idx] = 0, 1
            else: Bl[param_idx], Bu[param_idx] = -5, 5 # Old code's else
            names.append("beta")
            param_idx += 1

        # Seasonal Persistence (gamma)
        if est_seasonal:
            B[param_idx:param_idx + num_seasonal_persistence_params] = 0.05 # Default initial value (approximation)
            # Old code had more complex B init for seasonal persistence based on model types
            # For Bl, Bu:
            if bounds == "usual":
                Bl[param_idx:param_idx + num_seasonal_persistence_params] = 0
                Bu[param_idx:param_idx + num_seasonal_persistence_params] = 1
            else:
                Bl[param_idx:param_idx + num_seasonal_persistence_params] = -5
                Bu[param_idx:param_idx + num_seasonal_persistence_params] = 5 # Old code's else

            if isinstance(seas_est_val, bool) and seas_est_val: # Single gamma if all estimated together
                 if components_number_ets_seasonal > 1 :
                     names.extend([f"gamma{k+1}" for k in range(num_seasonal_persistence_params)])
                 elif num_seasonal_persistence_params ==1 : # handles single seasonal component
                     names.append("gamma")
            elif isinstance(seas_est_val, (list, np.ndarray)): # individual gammas
                true_indices = [i + 1 for i, est in enumerate(seas_est_val) if est]
                if len(true_indices) == 1 and num_seasonal_persistence_params ==1 :
                     names.append("gamma")
                else:
                     names.extend([f"gamma{k}" for k in true_indices])
            elif num_seasonal_persistence_params == 1: # Catch single seasonal from components_number_ets_seasonal
                 names.append("gamma")


            param_idx += num_seasonal_persistence_params

    # Xreg Persistence Parameters (delta) - ADDED SECTION
    if est_xreg_persistence and num_xreg_persistence_params > 0:
        # Default B values from old code (0.01 for A, 0 for M error type)
        # This depends on how xreg_parameters_persistence is structured.
        # Assuming one delta per estimated xreg persistence.
        B[param_idx:param_idx + num_xreg_persistence_params] = 0.01 if error_type == "A" else 0
        Bl[param_idx:param_idx + num_xreg_persistence_params] = -5
        Bu[param_idx:param_idx + num_xreg_persistence_params] = 5
        # Naming based on the number of such parameters. Old code used "delta1", "delta2"...
        # This relies on num_xreg_persistence_params correctly reflecting count of deltas
        xreg_pers_indices = []
        if explanatory_checked.get('xreg_parameters_persistence'):
            xreg_pers_indices = [i + 1 for i, est_flag in enumerate(explanatory_checked['xreg_parameters_persistence']) if est_flag]
        
        if len(xreg_pers_indices) == num_xreg_persistence_params and num_xreg_persistence_params > 0:
            if len(xreg_pers_indices) == 1:
                 names.append(f"delta{xreg_pers_indices[0]}") # Or just "delta" if only one possible
            else:
                 names.extend([f"delta{k}" for k in xreg_pers_indices])
        else: # Fallback naming if mismatch
            names.extend([f"delta{k+1}" for k in range(num_xreg_persistence_params)])
        param_idx += num_xreg_persistence_params


    # ARIMA Parameters
    if arima_checked.get('arima_model', False):
        if est_ar:
            # Initial AR values using PACF (simplified from old code)
            try:
                # Ensure nlags is at least 1
                nlags_ar = max(1, num_ar_params)
                pacf_values = calculate_pacf(y_in_sample[ot_logical], nlags=nlags_ar)
                # Ensure pacf_values has the correct length
                if len(pacf_values) >= num_ar_params:
                    B[param_idx:param_idx + num_ar_params] = pacf_values[:num_ar_params]
                else:
                    # Handle cases where PACF calculation returns fewer values than expected
                    B[param_idx:param_idx + len(pacf_values)] = pacf_values
                    B[param_idx + len(pacf_values):param_idx + num_ar_params] = 0.1 # Pad with fallback

            except Exception as e:
                 # print(f"PACF calculation failed: {e}") # Optional debug print
                 B[param_idx:param_idx + num_ar_params] = 0.1 # Fallback

            # Old code: Bl=-5, Bu=5
            Bl[param_idx:param_idx + num_ar_params] = -5
            Bu[param_idx:param_idx + num_ar_params] = 5
            # Naming needs refinement based on orders and lags
            names.extend([f"ar_{k+1}" for k in range(num_ar_params)]) # Simplified, old code had per-lag naming
            param_idx += num_ar_params

        if est_ma:
             # Initial MA values using ACF (simplified from old code)
            try:
                # Ensure nlags is at least 1
                nlags_ma = max(1, num_ma_params)
                acf_values = calculate_acf(y_in_sample[ot_logical], nlags=nlags_ma)[1:] # Exclude lag 0
                # Ensure acf_values has the correct length
                if len(acf_values) >= num_ma_params:
                     B[param_idx:param_idx + num_ma_params] = acf_values[:num_ma_params]
                else:
                    # Handle cases where ACF calculation returns fewer values than expected
                    B[param_idx:param_idx + len(acf_values)] = acf_values
                    B[param_idx + len(acf_values):param_idx + num_ma_params] = -0.1 # Pad with fallback
            except Exception as e:
                 # print(f"ACF calculation failed: {e}") # Optional debug print
                 B[param_idx:param_idx + num_ma_params] = -0.1 # Fallback

            # Old code: Bl=-5, Bu=5
            Bl[param_idx:param_idx + num_ma_params] = -5
            Bu[param_idx:param_idx + num_ma_params] = 5
            # Naming needs refinement based on orders and lags
            names.extend([f"ma_{k+1}" for k in range(num_ma_params)]) # Simplified, old code had per-lag naming
            param_idx += num_ma_params

    # Explanatory Variable Parameters
    if est_xreg:
        # Use initial values if available from _initialize_xreg_states, otherwise 0
        try:
            initial_xreg_vals = mat_vt[components_number_ets + model_params['components_number_arima'] : components_number_ets + model_params['components_number_arima'] + num_xreg_params, 0]
            B[param_idx:param_idx + num_xreg_params] = initial_xreg_vals
        except IndexError:
            B[param_idx:param_idx + num_xreg_params] = 0 # Fallback if mat_vt shape is unexpected

        # Old code: Bl=-inf, Bu=inf
        Bl[param_idx:param_idx + num_xreg_params] = -np.inf
        Bu[param_idx:param_idx + num_xreg_params] = np.inf
        names.extend([f"xreg_{k+1}" for k in range(num_xreg_params)])
        param_idx += num_xreg_params

    # Phi (Damping) Parameter
    if est_phi:
        B[param_idx] = 0.95 # Default initial value
        # Old code: Bl=0, Bu=1
        Bl[param_idx], Bu[param_idx] = 0, 1
        names.append("phi")
        param_idx += 1

    # Initial ETS States
    if ets_model and initials_active:
        # Initial Level
        if est_init_level:
            B[param_idx] = mat_vt[0, 0] # Use value from creator initialization
            # Old code: Bl = -np.inf if error_type == "A" else 0
            if error_type == "A":
                Bl[param_idx], Bu[param_idx] = -np.inf, np.inf
            else:
                Bl[param_idx], Bu[param_idx] = 0, np.inf
            names.append("init_level")
            param_idx += 1

        # Initial Trend
        if est_init_trend:
            B[param_idx] = mat_vt[1, 0] # Use value from creator initialization
            # Old code: Bl = -np.inf if trend_type == "A" else 0
            if trend_type == "A":
                Bl[param_idx], Bu[param_idx] = -np.inf, np.inf
            else: # Multiplicative trend
                Bl[param_idx], Bu[param_idx] = 0, np.inf
            names.append("init_trend")
            param_idx += 1

        # Initial Seasonal
        if est_init_seasonal:
             seasonal_comp_start_idx = 1 + int(model_is_trendy)
             param_count_seasonal = 0
             name_idx_seasonal = 1

             current_est_init_seas_val = est_init_seasonal_val
             # Ensure it's a list for iteration if it was a single True boolean
             if isinstance(current_est_init_seas_val, bool):
                 current_est_init_seas_val = [current_est_init_seas_val] * components_number_ets_seasonal

             for i in range(components_number_ets_seasonal):
                 if current_est_init_seas_val[i]:
                     try:
                         current_lag = model_params['lags_model'][seasonal_comp_start_idx + i]
                         num_params_this_comp = current_lag - 1
                         if num_params_this_comp > 0:
                             # Get initial values from mat_vt (calculated in creator)
                             initial_vals_seas = mat_vt[seasonal_comp_start_idx + i, :num_params_this_comp]
                             B[param_idx + param_count_seasonal : param_idx + param_count_seasonal + num_params_this_comp] = initial_vals_seas

                             # Old code: Bl = -np.inf if season_type == "A" else 0
                             if season_type == "A":
                                 Bl[param_idx + param_count_seasonal : param_idx + param_count_seasonal + num_params_this_comp] = -np.inf
                                 Bu[param_idx + param_count_seasonal : param_idx + param_count_seasonal + num_params_this_comp] = np.inf
                             else: # Multiplicative seasonality
                                 Bl[param_idx + param_count_seasonal : param_idx + param_count_seasonal + num_params_this_comp] = 0
                                 Bu[param_idx + param_count_seasonal : param_idx + param_count_seasonal + num_params_this_comp] = np.inf
                             
                             season_suffix = f"_{name_idx_seasonal}" if components_number_ets_seasonal > 1 and isinstance(est_init_seasonal_val, list) and len(est_init_seasonal_val) >1 else "" # check if est_init_seasonal_val implies multiple distinct seasonalities
                             names.extend([f"init_seas{season_suffix}_{k+1}" for k in range(num_params_this_comp)])
                             param_count_seasonal += num_params_this_comp
                     except IndexError:
                         # Handle potential index errors if lags_model is incorrect
                         print(f"Warning: Could not determine initial seasonal parameters for component {i+1}.")
                         pass # Skip this component's parameters

                     name_idx_seasonal += 1
             param_idx += param_count_seasonal

    # Initial ARIMA States
    if est_init_arima:
        if num_init_arima_params > 0:
            arima_state_start_idx = components_number_ets
            # Get initial values from mat_vt
            try:
                # Calculate expected shape and slice
                num_arima_components = model_params['components_number_arima']
                expected_len = num_arima_components * num_init_arima_params
                if mat_vt.shape[1] >= num_init_arima_params:
                    initial_arima_flat = mat_vt[arima_state_start_idx : arima_state_start_idx + num_arima_components, :num_init_arima_params].flatten()
                    # Ensure we don't exceed the length of B array slice
                    slice_len = min(len(initial_arima_flat), len(B[param_idx:param_idx + num_init_arima_params]))
                    B[param_idx:param_idx + slice_len] = initial_arima_flat[:slice_len]
                else:
                     B[param_idx:param_idx + num_init_arima_params] = 0 if error_type == "A" else 1 # Fallback
            except IndexError:
                 B[param_idx:param_idx + num_init_arima_params] = 0 if error_type == "A" else 1 # Fallback

            # Old code: Bl = -np.inf if error_type == "A" else 0
            if error_type == "A":
                Bl[param_idx:param_idx + num_init_arima_params] = -np.inf
                Bu[param_idx:param_idx + num_init_arima_params] = np.inf
            else:
                # Ensure initial values are non-negative for multiplicative errors as per old logic (implicitly by Bl=0)
                B[param_idx:param_idx + num_init_arima_params] = np.maximum(B[param_idx:param_idx + num_init_arima_params], 0) # Old used abs() then Bl=0
                Bl[param_idx:param_idx + num_init_arima_params] = 0
                Bu[param_idx:param_idx + num_init_arima_params] = np.inf
            names.extend([f"init_arima_{k+1}" for k in range(num_init_arima_params)])
            param_idx += num_init_arima_params

    # Constant Parameter
    if est_constant:
        try:
            constant_idx_in_vt = components_number_ets + model_params['components_number_arima'] + explanatory_checked.get('xreg_number', 0)
            B[param_idx] = mat_vt[constant_idx_in_vt, 0] # Use value from creator initialization
        except IndexError:
             B[param_idx] = 0 # Fallback

        # Bounds calculation similar to old code
        if (arima_checked.get('i_orders') and sum(arima_checked.get('i_orders',[])) != 0) or ets_model:
            try:
                valid_ot_logical = ot_logical & np.isfinite(y_in_sample)
                if np.sum(valid_ot_logical) > 1:
                    # Ensure y has positive values for log
                    log_y_valid = y_in_sample[valid_ot_logical]
                    if error_type != "A":
                         log_y_valid = log_y_valid[log_y_valid > 1e-10]
                         if len(log_y_valid) < 2: raise ValueError("Not enough positive values for log diff")
                         diff_log_y = np.diff(np.log(log_y_valid))
                    else:
                         # diff_log_y = np.array([]) # Not needed for Additive
                         pass # Not needed for Additive case for old bounds

                    diff_y = np.diff(y_in_sample[valid_ot_logical])

                    if error_type == "A": # Old code direct logic
                        bound_val = np.quantile(diff_y, 0.6) # Old: not abs()
                        # Ensure bound_val is not NaN if diff_y is empty or all same
                        if not np.isfinite(bound_val) or (len(diff_y) > 0 and np.all(diff_y == diff_y[0])): # if diff_y results in non-finite quantile
                             Bl[param_idx], Bu[param_idx] = -np.inf, np.inf
                        else:
                             Bl[param_idx] = -np.abs(bound_val) # ensure symmetry around 0 if bound_val can be negative
                             Bu[param_idx] = np.abs(bound_val)
                             if Bu[param_idx] <= Bl[param_idx]: Bl[param_idx], Bu[param_idx] = -np.inf, np.inf
                    elif bounds == "none": # New code path, keep for flexibility, though old didn't have it
                        Bl[param_idx], Bu[param_idx] = -np.inf, np.inf
                    else: # Multiplicative or Mixed from old logic
                        Bl[param_idx] = np.exp(np.quantile(diff_log_y, 0.4))
                        Bu[param_idx] = np.exp(np.quantile(diff_log_y, 0.6))
                        if Bu[param_idx] <= Bl[param_idx] or not np.isfinite(Bu[param_idx]): Bl[param_idx], Bu[param_idx] = 0, np.inf # Old: 0, not 1e-10
                else:
                     raise ValueError("Not enough valid observations for diff")
            except Exception as e:
                 # print(f"Constant bounds calculation failed: {e}") # Optional debug
                 Bl[param_idx], Bu[param_idx] = (-np.inf, np.inf) if error_type == "A" else (0, np.inf) # Old: 0, not 1e-10
        else: # Not ETS and no differencing
             y_abs_max = np.abs(y_in_sample[ot_logical]).max() if np.sum(ot_logical)>0 else 1
             current_B_val = B[param_idx] # Value already set or default 0
             Bl[param_idx] = -max(y_abs_max, abs(current_B_val) * 1.01 if np.isfinite(current_B_val) else y_abs_max) # handle non-finite B
             Bu[param_idx] = -Bl[param_idx]
             if not (np.isfinite(Bl[param_idx]) and np.isfinite(Bu[param_idx])): # Fallback if calculation fails
                Bl[param_idx], Bu[param_idx] = -np.inf, np.inf


        # Ensure B is within bounds
        B[param_idx] = np.clip(B[param_idx], Bl[param_idx], Bu[param_idx])

        names.append(constants_checked.get('constant_name', "constant"))
        param_idx += 1

    # Final check for parameter count consistency
    if param_idx != total_params:
        #print(f"Warning: Parameter count mismatch! Expected {total_params}, got {param_idx}. Adjusting arrays.")
        # Attempt to resize arrays - this might indicate logic error elsewhere
        B = B[:param_idx]
        Bl = Bl[:param_idx]
        Bu = Bu[:param_idx]
    # Return the dictionary in the expected format
    return {"B": B, "Bl": Bl, "Bu": Bu, "names": names}


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
    # Ensure model_is_trendy and model_is_seasonal flags are consistently set.
    # A trend_type of "N" means no trend.
    model_type_dict["model_is_trendy"] = model_type_dict.get("trend_type", "N") != "N"
    # A season_type of "N" means no seasonality.
    model_type_dict["model_is_seasonal"] = model_type_dict.get("season_type", "N") != "N"

    # Set up components for the model
    components_dict = _setup_components(model_type_dict, arima_checked, lags_dict)
    # Set up lags
    lags_dict = _setup_lags(lags_dict, model_type_dict, components_dict)

    # Set up profiles
    profiles_dict = _create_profiles(
        profiles_recent_provided, profiles_recent_table, lags_dict, observations_dict
    )

    # Update obs states
    observations_dict["obs_states"] = observations_dict["obs_in_sample"] + lags_dict["lags_model_max"]

    return model_type_dict, components_dict, lags_dict, observations_dict, profiles_dict


def _setup_components(model_type_dict, arima_checked, lags_dict):
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
            # Count number of seasonal components based on the original lags provided.
            # A seasonal lag is any lag > 1 in the original lags list for the model.
            original_lags = lags_dict.get("lags", []) 
            components_number_ets_seasonal = sum(1 for lag_period in original_lags if lag_period > 1)
            components_number_ets += components_number_ets_seasonal
        # Store in dictionary
        components_dict["components_number_ets"] = components_number_ets
        components_dict["components_number_ets_seasonal"] = (
            components_number_ets_seasonal
        )
        components_dict["components_number_ets_non_seasonal"] = (
            components_number_ets - components_number_ets_seasonal
        )
    else:
        # No ETS components
        components_dict["components_number_ets"] = 0
        components_dict["components_number_ets_seasonal"] = 0
        components_dict["components_number_ets_non_seasonal"] = 0

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
            lags_model_seasonal = []
            
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
    lags_dict_updated["lags"] = lags_model
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
        profiles_dict["index_lookup_table"] = profiles["index_lookup_table"]
    return profiles_dict


def adam_profile_creator(
    lags_model_all: List[List[int]],
    lags_model_max: int,
    obs_all: int,
    lags: Union[List[int], None] = None,
    y_index: Union[List, None] = None,
    y_classes: Union[List, None] = None
) -> Dict[str, np.ndarray]:
    """
    Creates recent profile and the lookup table for ADAM.

    Args:
        lags_model_all: All lags used in the model for ETS + ARIMA + xreg.
        lags_model_max: The maximum lag used in the model.
        obs_all: Number of observations to create.
        lags: The original lags provided by user (optional).
        y_index: The indices needed to get the specific dates (optional).
        y_classes: The class used for the actual data (optional).

    Returns:
        A dictionary with 'recent' (profiles_recent_table) and 'lookup'
        (index_lookup_table) as keys.
    """
    # Initialize matrices
    # Flatten lags_model_all to handle it properly
    # This is needed because in R, the lagsModelAll is a flat vector, but in Python it's a list of lists
    
    profiles_recent_table = np.zeros((len(lags_model_all), lags_model_max))
    index_lookup_table = np.ones((len(lags_model_all), obs_all + lags_model_max))
    profile_indices = (
        np.arange(1, lags_model_max * len(lags_model_all) + 1)
        .reshape(-1, len(lags_model_all))
        .T
    )

    # Update matrices based on lagsModelAll
    # Update matrices based on lagsModelAll
    for i, lag in enumerate(lags_model_all):
        # Create the matrix with profiles based on the provided lags.
        # For every row, fill the first 'lag' elements from 1 to lag
        profiles_recent_table[i, : lag] = np.arange(1, lag + 1)

        # For the i-th row in indexLookupTable, fill with a repeated sequence starting
        # from lagsModelMax to the end of the row.
        # The repeated sequence is the i-th row of profileIndices, repeated enough times
        # to cover 'obsAll' observations.
        # '- 1' at the end adjusts these values to Python's zero-based indexing.
        # Fix the array size mismatch - ensure we're using the correct range
        index_lookup_table[i, lags_model_max:(lags_model_max+obs_all)] = (
            np.tile(
                profile_indices[i, :lags_model_all[i]],
                int(np.ceil((obs_all) / lags_model_all[i]))
            )[0:(obs_all)] - 1
        )
        
        # Fix the head of the data, before the sample starts
        # (equivalent to the tail() operation in R code)
        unique_indices = np.unique(index_lookup_table[i, lags_model_max:(lags_model_max+obs_all-1)])
        index_lookup_table[i, :lags_model_max] = np.tile(unique_indices, lags_model_max)[:lags_model_max]
    # Convert to int!
    index_lookup_table = index_lookup_table.astype(int)

    # Note: I skip handling of special cases (e.g., daylight saving time, leap years)
    profiles = {
        "profiles_recent_table": np.array(profiles_recent_table, dtype="float64"),
        "index_lookup_table": np.array(index_lookup_table, dtype="int64"),
    }
    return profiles


def filler(B, 
           model_type_dict,
           components_dict,
           lags_dict,
           matrices_dict,
           persistence_checked,
           initials_checked,
           arima_checked,
           explanatory_checked,
           phi_dict,
           constants_checked):
    """
    Updates model matrices based on parameter values.
    """
    j = 0
    # Fill in persistence
    if persistence_checked['persistence_estimate']:
        # Persistence of ETS
        if model_type_dict['ets_model']:
            i = 0
            # alpha
            if persistence_checked['persistence_level_estimate']:
                j += 1
                matrices_dict['vec_g'][i] = B[j-1]
            # beta
            if model_type_dict['model_is_trendy']:
                i = 1
                if persistence_checked['persistence_trend_estimate']:
                    j += 1
                    matrices_dict['vec_g'][i] = B[j-1]
                    
            
            # gamma1, gamma2, ...
            if model_type_dict['model_is_seasonal']:
                if any(persistence_checked['persistence_seasonal_estimate']):
                    seasonal_indices = i + np.where(persistence_checked['persistence_seasonal_estimate'])[0] + 1
                    matrices_dict['vec_g'][seasonal_indices] = B[j:j+sum(persistence_checked['persistence_seasonal_estimate'])]
                    j += sum(persistence_checked['persistence_seasonal_estimate'])
                i = components_dict['components_number_ets'] - 1
        
        # Persistence of xreg
        if explanatory_checked['xreg_model'] and persistence_checked['persistence_xreg_estimate']:
            xreg_persistence_number = max(explanatory_checked['xreg_parameters_persistence'])
            xreg_indices = slice(j + components_dict['components_number_arima'], j + components_dict['components_number_arima'] + len(explanatory_checked['xreg_parameters_persistence']))
            matrices_dict['vec_g'][xreg_indices] = B[j:j+xreg_persistence_number][np.array(explanatory_checked['xreg_parameters_persistence']) - 1]
            j += xreg_persistence_number
    
    # Damping parameter
    if model_type_dict['ets_model'] and phi_dict['phi_estimate']:
        j += 1
        matrices_dict['mat_wt'][:, 1] = B[j-1]
        matrices_dict['mat_f'][0:2, 1] = B[j-1]
    
    # ARMA parameters
    if arima_checked['arima_model']:
        # Call the function returning ARI and MA polynomials
        try:
            arima_polynomials = adam_polynomialiser(
                B[j:j+sum(np.array(arima_checked['ar_orders'])*arima_checked['ar_estimate'] + 
                         np.array(arima_checked['ma_orders'])*arima_checked['ma_estimate'])],
                arima_checked['ar_orders'],
                arima_checked['ma_orders'],
                arima_checked['i_orders'],
                arima_checked['ar_estimate'],
                arima_checked['ma_estimate'],
                arima_checked['arma_parameters'],
                lags_dict['lags']
            )
            arima_polynomials = {k: np.array(v) for k, v in arima_polynomials.items()}
            
            # Fill in the transition matrix
            if len(arima_checked['non_zero_ari']) > 0:
                matrices_dict['mat_f'][components_dict['components_number_ets'] + arima_checked['non_zero_ari'][:, 1], 
                       components_dict['components_number_ets']:components_dict['components_number_ets'] + components_dict['components_number_arima'] + constants_checked['constant_estimate']] = \
                     -arima_polynomials['ariPolynomial'][arima_checked['non_zero_ari'][:, 0]]
            
            # Fill in the persistence vector
            if len(arima_checked['non_zero_ari']) > 0:
                matrices_dict['vec_g'][components_dict['components_number_ets'] + arima_checked['non_zero_ari'][:, 1]] = -arima_polynomials['ariPolynomial'][arima_checked['non_zero_ari'][:, 0]]
            if len(arima_checked['non_zero_ma']) > 0:
                matrices_dict['vec_g'][components_dict['components_number_ets'] + arima_checked['non_zero_ma'][:, 1]] += arima_polynomials['maPolynomial'][arima_checked['non_zero_ma'][:, 0]]
        except Exception as e:
            print(f"DEBUG - Error in ARIMA processing: {e}")
        
        j += sum(np.array(arima_checked['ar_orders'])*arima_checked['ar_estimate'] + 
                np.array(arima_checked['ma_orders'])*arima_checked['ma_estimate'])
    
    # Initials of ETS
    if model_type_dict['ets_model'] and initials_checked['initial_type'] not in ['complete', 'backcasting'] and initials_checked['initial_estimate']:
        i = 0
        if initials_checked['initial_level_estimate']:
            j += 1
            matrices_dict['mat_vt'][i, :lags_dict['lags_model_max']] = B[j-1]
            
        i += 1
        if model_type_dict['model_is_trendy'] and initials_checked['initial_trend_estimate']:
            j += 1
            matrices_dict['mat_vt'][i, :lags_dict['lags_model_max']] = B[j-1]
            i += 1
        
        if model_type_dict["model_is_seasonal"] and (isinstance(initials_checked['initial_seasonal_estimate'], bool) and initials_checked['initial_seasonal_estimate'] or isinstance(initials_checked['initial_seasonal_estimate'], list) and any(initials_checked['initial_seasonal_estimate'])):
            for k in range(components_dict['components_number_ets_seasonal']):
                # Convert initial_seasonal_estimate to a list if it's not already
                # This is for handling single seasonalities
                if isinstance(initials_checked['initial_seasonal_estimate'], bool):
                    initials_checked['initial_seasonal_estimate'] = [initials_checked['initial_seasonal_estimate']] * components_dict['components_number_ets_seasonal']

                if initials_checked['initial_seasonal_estimate'][k]:
                    # added a -1 because the seasonal index is 0-based in R
                    seasonal_index = components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'] + k
                    lag = lags_dict['lags'][seasonal_index]

                    matrices_dict['mat_vt'][seasonal_index, :lag - 1] = B[j:j+lag]
                    
                    if model_type_dict['season_type'] == "A":
                        matrices_dict['mat_vt'][seasonal_index, lag-1] = -np.sum(B[j:j+lag])
                    else:  # "M"
                        matrices_dict['mat_vt'][seasonal_index, lag-1] = 1 / np.prod(B[j:j+lag])

                    j += lag - 1
    
    # Initials of ARIMA
    if arima_checked['arima_model']:
        if initials_checked['initial_type'] not in ['complete', 'backcasting'] and initials_checked['initial_arima_estimate']:
            #print(f"DEBUG - Processing ARIMA initial values starting at index {j}")
            arima_index = components_dict['components_number_ets'] + components_dict['components_number_arima'] - 1
            

            matrices_dict['mat_vt'][arima_index, :initials_checked['initial_arima_number']] = B[j:j+initials_checked['initial_arima_number']]
            
            if model_type_dict['error_type'] == "A":
                ari_indices = components_dict['components_number_ets'] + arima_checked['non_zero_ari'][:, 1]
                matrices_dict['mat_vt'][ari_indices, :initials_checked['initial_arima_number']] = \
                    np.dot(arima_polynomials['ariPolynomial'][arima_checked['non_zero_ari'][:, 0]], 
                            B[j:j+initials_checked['initial_arima_number']].reshape(1, -1)) / arima_polynomials['ariPolynomial'][-1]
            else:  # "M"
                ari_indices = components_dict['components_number_ets'] + arima_checked['non_zero_ari'][:, 1]
                matrices_dict['mat_vt'][ari_indices, :initials_checked['initial_arima_number']] = \
                    np.exp(np.dot(arima_polynomials['ariPolynomial'][arima_checked['non_zero_ari'][:, 0]], 
                                    np.log(B[j:j+initials_checked['initial_arima_number']]).reshape(1, -1)) / arima_polynomials['ariPolynomial'][-1])
            
        
            j += initials_checked['initial_arima_number']
        elif any([arima_checked['ar_estimate'], arima_checked['ma_estimate']]):
            if model_type_dict['error_type'] == "A":
                matrices_dict['mat_vt'][components_dict['components_number_ets'] + arima_checked['non_zero_ari'][:, 1], 
                                        :initials_checked['initial_arima_number']] = \
                    np.dot(arima_polynomials['ariPolynomial'][arima_checked['non_zero_ari'][:, 0]], 
                            matrices_dict['mat_vt'][components_dict['components_number_ets'] + components_dict['components_number_arima'] - 1, 
                                                    :initials_checked['initial_arima_number']].reshape(1, -1)) / \
                    arima_polynomials['ariPolynomial'][-1]
            else:  # "M"
                matrices_dict['mat_vt'][components_dict['components_number_ets'] + arima_checked['non_zero_ari'][:, 1], 
                                        :initials_checked['initial_arima_number']] = \
                    np.exp(np.dot(arima_polynomials['ariPolynomial'][arima_checked['non_zero_ari'][:, 0]], 
                                    np.log(matrices_dict['mat_vt'][components_dict['components_number_ets'] + components_dict['components_number_arima'] - 1, 
                                                                :initials_checked['initial_arima_number']]).reshape(1, -1)) / \
                            arima_polynomials['ariPolynomial'][-1])
            
    # Xreg initial values
    if explanatory_checked['xreg_model'] and (initials_checked['initial_type'] != "complete") and initials_checked['initial_estimate'] and initials_checked['initial_xreg_estimate']:
        xreg_number_to_estimate = sum(explanatory_checked['xreg_parameters_estimated'])
        xreg_indices = components_dict['components_number_ets'] + components_dict['components_number_arima'] + np.where(explanatory_checked['xreg_parameters_estimated'] == 1)[0]
    
        matrices_dict['mat_vt'][xreg_indices, :lags_dict['lags_model_max']] = B[j:j+xreg_number_to_estimate]

        j += xreg_number_to_estimate
    
    # Constant
    if constants_checked['constant_estimate']:
        constant_index = components_dict['components_number_ets'] + components_dict['components_number_arima'] + explanatory_checked['xreg_number']
        
        matrices_dict['mat_vt'][constant_index, :] = B[j]
    return {
        'mat_vt': matrices_dict['mat_vt'],
        'mat_wt': matrices_dict['mat_wt'],
        'mat_f': matrices_dict['mat_f'],
        'vec_g': matrices_dict['vec_g'],
        'arima_polynomials': matrices_dict['arima_polynomials']
    }