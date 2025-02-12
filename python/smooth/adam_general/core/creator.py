import numpy as np
from typing import List, Optional, Dict, Any, Union
from core.utils.utils import (
    measurement_inverter, scaler, msdecompose, calculate_acf, 
    calculate_pacf, calculate_likelihood, calculate_entropy, 
    calculate_multistep_loss
)
from core.utils.polynomials import adam_polynomialiser
from scipy import stats
from scipy.linalg import eigvals
from scipy.optimize import minimize
import pandas as pd


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
            
            explanatory_checked = None,
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
    """
    
    # Extract observation values
    obs_states = observations_dict["obs_states"]
    obs_in_sample = observations_dict["obs_in_sample"]
    obs_all = observations_dict["obs_all"]
    ot_logical = observations_dict["ot_logical"]
    y_in_sample = observations_dict["y_in_sample"]
    obs_nonzero = observations_dict['obs_nonzero']
    # Extract values from dictionaries
    ets_model = model_type_dict["ets_model"]
    e_type = model_type_dict["error_type"] 
    t_type = model_type_dict["trend_type"]
    s_type = model_type_dict["season_type"]
    model_is_trendy = model_type_dict["model_is_trendy"]
    model_is_seasonal = model_type_dict["model_is_seasonal"]
    
    components_number_ets = components_dict["components_number_ets"]
    components_number_ets_seasonal = components_dict["components_number_ets_seasonal"]
    components_number_arima = components_dict.get("components_number_arima", 0)
    
    phi = phi_dict["phi"]
    
    lags = lags_dict["lags"]
    lags_model = lags_dict["lags_model"]
    lags_model_arima = lags_dict["lags_model_arima"]
    lags_model_all = lags_dict["lags_model_all"]
    lags_model_max = lags_dict["lags_model_max"]

    profiles_recent_table = profiles_dict["profiles_recent_table"]
    profiles_recent_provided = profiles_dict["profiles_recent_provided"]
    
    # Matrix of states. Time in columns, components in rows
    mat_vt = np.full((components_number_ets + components_number_arima + 
                      explanatory_checked['xreg_number'] + constants_checked['constant_required'], 
                      obs_states), np.nan)

    # Measurement rowvector
    mat_wt = np.ones((obs_all, components_number_ets + components_number_arima + 
                      explanatory_checked['xreg_number'] + constants_checked['constant_required']))

    # If xreg are provided, then fill in the respective values in Wt vector
    if explanatory_checked['xreg_model']:
        mat_wt[:, components_number_ets + components_number_arima:
                  components_number_ets + components_number_arima + explanatory_checked['xreg_number']] = \
            explanatory_checked['xreg_data']

    # Transition matrix
    mat_f = np.eye(components_number_ets + components_number_arima + explanatory_checked['xreg_number'] + constants_checked['constant_required'])

    # Persistence vector
    vec_g = np.zeros((components_number_ets + components_number_arima + explanatory_checked['xreg_number'] + constants_checked['constant_required'], 1))
    #vec_g_index = components_names_ets + components_names_arima + xreg_names + constant_name

    

    j = 0
    # ETS model, persistence
    if ets_model:
        j += 1
        if not persistence_checked['persistence_level_estimate']:
            vec_g[j-1, 0] = persistence_checked['persistence_level']
        
        if model_is_trendy:
            j += 1
            if not persistence_checked['persistence_trend_estimate']:
                vec_g[j-1, 0] = persistence_checked['persistence_trend']
        
        if model_is_seasonal:
            if not all(persistence_checked['persistence_seasonal_estimate']):
                vec_g[j + np.where(np.logical_not(persistence_checked['persistence_seasonal_estimate']))[0], 0] = persistence_checked['persistence_seasonal']
            
    # ARIMA model, names for persistence
    if arima_checked['arima_model']:
        # Remove diagonal from the ARIMA part of the matrix
        mat_f[j:j+components_number_arima, j:j+components_number_arima] = 0
        #if components_number_arima > 1:
            #vec_g_index[j:j+components_number_arima] = [f"psi{i}" for i in range(1, components_number_arima+1)]
        #else:
            #vec_g_index[j] = "psi"
        j += components_number_arima

    # Modify transition to do drift
    if not arima_checked['arima_model'] and constants_checked['constant_required']:
        mat_f[0, -1] = 1

    # Regression, persistence
    if explanatory_checked['xreg_model']:
        if persistence_checked['persistence_xreg_provided'] and not persistence_checked['persistence_xreg_estimate']:
            vec_g[j:j+explanatory_checked['xreg_number'], 0] = persistence_checked['persistence_xreg']

    # Damping parameter value
    if ets_model and model_is_trendy:
        mat_f[0, 1] = phi
        mat_f[1, 1] = phi
        mat_wt[:, 1] = phi

    # If the arma parameters were provided, fill in the persistence
    if arima_checked['arima_model'] and (not arima_checked['ar_estimate'] and not arima_checked['ma_estimate']):
        # Call polynomial
        arima_polynomials = {key: np.array(value) for key, value in adam_polynomialiser(
            0, arima_checked['ar_orders'], arima_checked['i_orders'], arima_checked['ma_orders'],
            arima_checked['ar_estimate'], arima_checked['ma_estimate'], arima_checked['arma_parameters'], lags
        ).items()}
        
        # Fill in the transition matrix
        if len(arima_checked['non_zero_ari']) > 0:
            non_zero_ari = np.array(arima_checked['non_zero_ari'])
            mat_f[components_number_ets + non_zero_ari[:, 1], components_number_ets + non_zero_ari[:, 1]] = \
                -arima_polynomials['ari_polynomial'][non_zero_ari[:, 0]]
        
        # Fill in the persistence vector
        if len(arima_checked['non_zero_ari']) > 0:
            non_zero_ari = np.array(arima_checked['non_zero_ari'])
            vec_g[components_number_ets + non_zero_ari[:, 1], 0] = \
                -arima_polynomials['ari_polynomial'][non_zero_ari[:, 0]]
        
        if len(arima_checked['non_zero_ma']) > 0:
            non_zero_ma = np.array(arima_checked['non_zero_ma'])
            vec_g[components_number_ets + non_zero_ma[:, 1], 0] += \
                arima_polynomials['ma_polynomial'][non_zero_ma[:, 0]]
    else:
        arima_polynomials = None

    if not profiles_recent_provided:
        # ETS model, initial state
        # If something needs to be estimated...
        if ets_model:
            if initials_checked['initial_estimate']:
                # For the seasonal models
                if model_is_seasonal:
                    
                    if obs_nonzero >= lags_model_max * 2:
                        # If either e_type or s_type are multiplicative, do multiplicative decomposition
                        decomposition_type = "multiplicative" if any(x == "M" for x in [e_type, s_type]) else "additive"
                        y_decomposition = msdecompose(y_in_sample.values.ravel(), [lag for lag in lags if lag != 1], type=decomposition_type)
                        j = 0
                        # level
                        if initials_checked['initial_level_estimate']:
                            mat_vt[j, 0:lags_model_max] = y_decomposition['initial'][0]
                            if explanatory_checked['xreg_model']:
                                if e_type == "A":
                                    mat_vt[j, 0:lags_model_max] -= np.dot(explanatory_checked['xreg_model_initials'][0]['initial_xreg'], explanatory_checked['xreg_data'][0])
                                else:
                                    mat_vt[j, 0:lags_model_max] /= np.exp(np.dot(explanatory_checked['xreg_model_initials'][1]['initial_xreg'], explanatory_checked['xreg_data'][0]))
                        else:
                            mat_vt[j, 0:lags_model_max] = initials_checked['initial_level']
                        j += 1
                        # If trend is needed
                        if model_is_trendy:
                            if initials_checked['initial_trend_estimate']:
                                if t_type == "A" and s_type == "M":
                                    mat_vt[j, 0:lags_model_max] = np.prod(y_decomposition['initial']) - y_decomposition['initial'][0]
                                    # If the initial trend is higher than the lowest value, initialise with zero.
                                    # This is a failsafe mechanism for the mixed models
                                    if mat_vt[j, 0] < 0 and abs(mat_vt[j, 0]) > min(abs(y_in_sample[ot_logical])):
                                        mat_vt[j, 0:lags_model_max] = 0
                                elif t_type == "M" and s_type == "A":
                                    mat_vt[j, 0:lags_model_max] = sum(abs(y_decomposition['initial'])) / abs(y_decomposition['initial'][0])
                                elif t_type == "M":
                                    # trend is too dangerous, make it start from 1.
                                    mat_vt[j, 0:lags_model_max] = 1
                                else:
                                    # trend
                                    mat_vt[j, 0:lags_model_max] = y_decomposition['initial'][1]
                                
                                # This is a failsafe for multiplicative trend models, so that the thing does not explode
                                if t_type == "M" and np.any(mat_vt[j, 0:lags_model_max] > 1.1):
                                    mat_vt[j, 0:lags_model_max] = 1
                                # This is a failsafe for multiplicative trend models, so that the thing does not explode
                                if t_type == "M" and np.any(mat_vt[0, 0:lags_model_max] < 0):
                                    mat_vt[0, 0:lags_model_max] = y_in_sample[ot_logical][0]
                            else:
                                mat_vt[j, 0:lags_model_max] = initials_checked['initial_trend']
                            j += 1

                        # Seasonal components
                        # For pure models use stuff as is
                        if all(x == "A" for x in [e_type, s_type]) or all(x == "M" for x in [e_type, s_type]) or (e_type == "A" and s_type == "M"):
                            for i in range(components_number_ets_seasonal):
                                if initials_checked['initial_seasonal_estimate']:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = y_decomposition['seasonal'][i]
                                    # Renormalise the initial seasons
                                    if s_type == "A":
                                        mat_vt[i+j-1, 0:lags_model[i+j-1]] -= \
                                            np.mean(mat_vt[i+j-1, 0:lags_model[i+j-1]])
                                    else:
                                        mat_vt[i+j-1, 0:lags_model[i+j-1]] /= \
                                            np.exp(np.mean(np.log(mat_vt[i+j-1, 0:lags_model[i+j-1]])))
                                else:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = initials_checked['initial_seasonal'][i]

                        # For mixed models use a different set of initials
                        elif e_type == "M" and s_type == "A":
                            for i in range(components_number_ets_seasonal):
                                if initials_checked['initial_seasonal_estimate']:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = np.log(y_decomposition['seasonal'][i]) * min(y_in_sample[ot_logical])
                                    # Renormalise the initial seasons
                                    if s_type == "A":
                                        mat_vt[i+j-1, 0:lags_model[i+j-1]] -= np.mean(mat_vt[i+j-1, 0:lags_model[i+j-1]])
                                    else:
                                        mat_vt[i+j-1, 0:lags_model[i+j-1]] /= np.exp(np.mean(np.log(mat_vt[i+j-1, 0:lags_model[i+j-1]])))
                                else:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = initials_checked['initial_seasonal'][i]
                    else:
                        # If either e_type or s_type are multiplicative, do multiplicative decomposition
                        j = 0
                        # level
                        if initials_checked['initial_level_estimate']:
                            mat_vt[j, 0:lags_model_max] = np.mean(y_in_sample[0:lags_model_max])
                            if explanatory_checked['xreg_model']:
                                if e_type == "A":
                                    mat_vt[j, 0:lags_model_max] -= np.dot(explanatory_checked['xreg_model_initials'][0]['initial_xreg'], explanatory_checked['xreg_data'][0])
                                else:
                                    mat_vt[j, 0:lags_model_max] /= np.exp(np.dot(explanatory_checked['xreg_model_initials'][1]['initial_xreg'], explanatory_checked['xreg_data'][0]))
                        else:
                            mat_vt[j, 0:lags_model_max] = initials_checked['initial_level']
                        j += 1
                        if model_is_trendy:
                            if initials_checked['initial_trend_estimate']:
                                if t_type == "A":
                                    # trend
                                    mat_vt[j, 0:lags_model_max] = y_in_sample[1] - y_in_sample[0]
                                elif t_type == "M":
                                    if initials_checked['initial_level_estimate']:
                                        # level fix
                                        mat_vt[j-1, 0:lags_model_max] = np.exp(np.mean(np.log(y_in_sample[ot_logical][0:lags_model_max])))
                                    # trend
                                    mat_vt[j, 0:lags_model_max] = y_in_sample[1] / y_in_sample[0]
                                # This is a failsafe for multiplicative trend models, so that the thing does not explode
                                if t_type == "M" and np.any(mat_vt[j, 0:lags_model_max] > 1.1):
                                    mat_vt[j, 0:lags_model_max] = 1
                            else:
                                mat_vt[j, 0:lags_model_max] = initials_checked['initial_trend']

                            # Do roll back. Especially useful for backcasting and multisteps
                            if t_type == "A":
                                mat_vt[j-1, 0:lags_model_max] = mat_vt[j-1, 0] - mat_vt[j, 0] * lags_model_max
                            elif t_type == "M":
                                mat_vt[j-1, 0:lags_model_max] = mat_vt[j-1, 0] / mat_vt[j, 0]**lags_model_max
                            j += 1

                        # Seasonal components
                        if s_type == "A":
                            for i in range(components_number_ets_seasonal):
                                if initials_checked['initial_seasonal_estimate']:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = y_in_sample[0:lags_model[i+j-1]] - mat_vt[0, 0]
                                    # Renormalise the initial seasons
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] -= np.mean(mat_vt[i+j-1, 0:lags_model[i+j-1]])
                                else:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = initials_checked['initial_seasonal'][i]
                        # For mixed models use a different set of initials
                        else:
                            for i in range(components_number_ets_seasonal):
                                if initials_checked['initial_seasonal_estimate']:
                                    # abs() is needed for mixed ETS+ARIMA
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = y_in_sample[0:lags_model[i+j-1]] / abs(mat_vt[0, 0])
                                    # Renormalise the initial seasons
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] /= np.exp(np.mean(np.log(mat_vt[i+j-1, 0:lags_model[i+j-1]])))
                                else:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = initials_checked['initial_seasonal'][i]
                else:
                    # Non-seasonal models
                    # level
                    if initials_checked['initial_level_estimate']:
                        mat_vt[0, 0:lags_model_max] = np.mean(y_in_sample[:max(lags_model_max, int(np.ceil(obs_in_sample * 0.2)))])
                    else:
                        mat_vt[0, 0:lags_model_max] = initials_checked['initial_level']
                    if model_is_trendy:
                        if initials_checked['initial_trend_estimate']:
                            if t_type == "A":
                                mat_vt[1, 0:lags_model_max] = np.mean(np.diff(y_in_sample[0:max(lags_model_max + 1, int(obs_in_sample * 0.2))]))
                            else:  # t_type == "M"
                                mat_vt[1, 0:lags_model_max] = np.exp(np.mean(np.diff(np.log(y_in_sample[ot_logical]))))
                        else:
                            mat_vt[1, 0:lags_model_max] = initials_checked['initial_trend']

                if initials_checked['initial_level_estimate'] and e_type == "M" and mat_vt[0, lags_model_max-1] == 0:
                    mat_vt[0, 0:lags_model_max] = np.mean(y_in_sample)

            # Else, insert the provided ones... make sure that this is not a backcasting
            elif not initials_checked['initial_estimate'] and initials_checked['initial_type'] == "provided":
                j = 0
                mat_vt[j, 0:lags_model_max] = initials_checked['initial_level']
                if model_is_trendy:
                    j += 1
                    mat_vt[j, 0:lags_model_max] = initials_checked['initial_trend']
                if model_is_seasonal:
                    for i in range(components_number_ets_seasonal):
                        # This is misaligned, but that's okay, because this goes directly to profile_recent
                        mat_vt[j+i, 0:lags_model[j+i]] = initials_checked['initial_seasonal'][i]
                j += components_number_ets_seasonal

        # If ARIMA orders are specified, prepare initials
        if arima_checked['arima_model']:
            if initials_checked['initial_arima_estimate']:
                mat_vt[components_number_ets:components_number_ets+components_number_arima, 0:initials_checked['initial_arima_number']] = 0 if e_type == "A" else 1
                
                if any(lag > 1 for lag in lags):
                    y_decomposition = msdecompose(y_in_sample, [lag for lag in lags if lag != 1], 
                                                type="additive" if e_type == "A" else "multiplicative")['seasonal'][-1][0]
                else:
                    y_decomposition = np.mean(np.diff(y_in_sample[ot_logical])) if e_type == "A" else np.exp(np.mean(np.diff(np.log(y_in_sample[ot_logical]))))
                
                mat_vt[components_number_ets+components_number_arima-1, 0:initials_checked['initial_arima_number']] = \
                    np.tile(y_decomposition, int(np.ceil(initials_checked['initial_arima_number'] / max(lags))))[:initials_checked['initial_arima_number']]
            
            else:
                mat_vt[components_number_ets:components_number_ets+components_number_arima, 0:initials_checked['initial_arima_number']] = 0 if e_type == "A" else 1
                mat_vt[components_number_ets+components_number_arima-1, 0:initials_checked['initial_arima_number']] = initials_checked['initial_arima'][:initials_checked['initial_arima_number']]

        # Fill in the initials for xreg
        if explanatory_checked['xreg_model']:
            if e_type == "A" or initials_checked['initial_xreg_provided'] or explanatory_checked['xreg_model_initials'][1] is None:
                mat_vt[components_number_ets+components_number_arima:components_number_ets+components_number_arima+explanatory_checked['xreg_number'], 0:lags_model_max] = \
                    explanatory_checked['xreg_model_initials'][0]['initial_xreg']
            else:
                mat_vt[components_number_ets+components_number_arima:components_number_ets+components_number_arima+explanatory_checked['xreg_number'], 0:lags_model_max] = \
                    explanatory_checked['xreg_model_initials'][1]['initial_xreg']

        # Add constant if needed
        if constants_checked['constant_required']:
            if constants_checked['constant_estimate']:
                # Add the mean of data
                if sum(arima_checked['i_orders']) == 0 and not ets_model:
                    mat_vt[components_number_ets+components_number_arima+explanatory_checked['xreg_number'], :] = np.mean(y_in_sample[ot_logical])
                # Add first differences
                else:
                    if e_type == "A":
                        mat_vt[components_number_ets+components_number_arima+explanatory_checked['xreg_number'], :] = np.mean(np.diff(y_in_sample[ot_logical]))
                    else:
                        mat_vt[components_number_ets+components_number_arima+explanatory_checked['xreg_number'], :] = np.exp(np.mean(np.diff(np.log(y_in_sample[ot_logical]))))
            else:
                mat_vt[components_number_ets+components_number_arima+explanatory_checked['xreg_number'], :] = constants_checked['constant_value']
            
            # If ETS model is used, change the initial level
            if ets_model and initials_checked['initial_level_estimate']:
                if e_type == "A":
                    mat_vt[0, 0:lags_model_max] -= mat_vt[components_number_ets+components_number_arima+explanatory_checked['xreg_number'], 0]
                else:
                    mat_vt[0, 0:lags_model_max] /= mat_vt[components_number_ets+components_number_arima+explanatory_checked['xreg_number'], 0]
            
            # If ARIMA is done, debias states
            if arima_checked['arima_model'] and initials_checked['initial_arima_estimate']:
                if e_type == "A":
                    mat_vt[components_number_ets+non_zero_ari[:, 1], 0:initials_checked['initial_arima_number']] -= \
                        mat_vt[components_number_ets+components_number_arima+explanatory_checked['xreg_number'], 0]
                else:
                    mat_vt[components_number_ets+non_zero_ari[:, 1], 0:initials_checked['initial_arima_number']] /= \
                        mat_vt[components_number_ets+components_number_arima+explanatory_checked['xreg_number'], 0]
    else:
        mat_vt[:, 0:lags_model_max] = profiles_recent_table

    return {'mat_vt': mat_vt, 
            'mat_wt': mat_wt, 
            'mat_f': mat_f, 
            'vec_g': vec_g, 
            'arima_polynomials': arima_polynomials}


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
    other=None
):
    """
    Initializes parameters for ADAM models.
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
        #+ initials_checked['other_parameter_estimate']
    )

    B = np.full(total_params, np.nan)
    Bl = np.full(total_params, np.nan)
    Bu = np.full(total_params, np.nan)
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
                        B[j:j+sum(persistence_estimate_vector)] = [0.1, 0] + [0.01] * components_dict["components_number_ets_seasonal"]
                    else:
                        B[j:j+sum(persistence_estimate_vector)] = [0.2, 0.01] + [0.01] * components_dict["components_number_ets_seasonal"]
                elif model_type_dict["error_type"] == "M" and model_type_dict["trend_type"] == "M":
                    B[j:j+sum(persistence_estimate_vector)] = [0.1, 0.05] + [0.01] * components_dict["components_number_ets_seasonal"]
                else:
                    initial_values = [0.1]
                    if model_type_dict["model_is_trendy"]:
                        initial_values.append(0.05)
                    if model_type_dict["model_is_seasonal"]:
                        initial_values.extend([0.11] * components_dict["components_number_ets_seasonal"])
                    
                    B[j:j+sum(persistence_estimate_vector)] = [val for val, estimate in zip(initial_values, persistence_estimate_vector) if estimate]
            else:
                initial_values = [0.1]
                if model_type_dict["model_is_trendy"]:
                    initial_values.append(0.05)
                if model_type_dict["model_is_seasonal"]:
                    initial_values.extend([0.11] * components_dict["components_number_ets_seasonal"])
                
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
        if model_type_dict["model_is_seasonal"] and any(initials_checked['initial_seasonal_estimate']):
            for k in range(components_dict["components_number_ets_seasonal"]):
                if initials_checked['initial_seasonal_estimate'][k]:
                    B[j:j+lags_dict["lags_model_seasonal"][k]-1] = adam_created['mat_vt'][components_dict["components_number_ets"] + k, 1:lags_dict["lags_model_seasonal"][k]]
                    if model_type_dict["season_type"] == "A":
                        Bl[j:j+lags_dict["lags_model_seasonal"][k]-1] = -np.inf
                        Bu[j:j+lags_dict["lags_model_seasonal"][k]-1] = np.inf
                    else:
                        Bl[j:j+lags_dict["lags_model_seasonal"][k]-1] = 0
                        Bu[j:j+lags_dict["lags_model_seasonal"][k]-1] = np.inf
                    names.extend([f"seasonal{k+1}_{m}" for m in range(2, lags_dict["lags_model_seasonal"][k])])
                    j += lags_dict["lags_model_seasonal"][k] - 1

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
                Bu[j-1] = np.quantile(np.diff(observations_dict['y_in_sample'][observations_dict['ot_logical']]), 0.6)
                Bl[j-1] = -Bu[j-1]
            else:
                Bu[j-1] = np.exp(np.quantile(np.diff(np.log(observations_dict['y_in_sample'][observations_dict['ot_logical']])), 0.6))
                Bl[j-1] = np.exp(np.quantile(np.diff(np.log(observations_dict['y_in_sample'][observations_dict['ot_logical']])), 0.4))
            
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
        "B": B[:j],
        "Bl": Bl[:j],
        "Bu": Bu[:j],
        "names": names
    }



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
                    matrices_dict['vec_g'][i + np.where(persistence_checked['persistence_seasonal_estimate'])[0]] = B[j:j+sum(persistence_checked['persistence_seasonal_estimate'])]
                    j += sum(persistence_checked['persistence_seasonal_estimate'])
                i = components_dict['components_number_ets'] - 1
        
        # Persistence of xreg
        if explanatory_checked['xreg_model'] and persistence_checked['persistence_xreg_estimate']:
            xreg_persistence_number = max(explanatory_checked['xreg_parameters_persistence'])
            matrices_dict['vec_g'][j + components_dict['components_number_arima']:j + components_dict['components_number_arima'] + len(explanatory_checked['xreg_parameters_persistence'])] = \
                B[j:j+xreg_persistence_number][np.array(explanatory_checked['xreg_parameters_persistence']) - 1]
            j += xreg_persistence_number
    
    # Damping parameter
    if model_type_dict['ets_model'] and phi_dict['phi_estimate']:
        j += 1
        matrices_dict['mat_wt'][:, 1] = B[j-1]
        matrices_dict['mat_f'][0:2, 1] = B[j-1]
    
    # ARMA parameters
    if arima_checked['arima_model']:
        # Call the function returning ARI and MA polynomials
        arima_polynomials = adam_polynomialiser(
            B[j:j+sum(np.array(arima_checked['ar_orders'])*arima_checked['ar_estimate'] + 
                     np.array(arima_checked['ma_orders'])*arima_checked['ma_estimate'])],
            arima_checked['ar_orders'], arima_checked['i_orders'], arima_checked['ma_orders'],
            arima_checked['ar_estimate'], arima_checked['ma_estimate'], 
            arima_checked['arma_parameters'], lags_dict['lags']
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
        if model_type_dict['model_is_seasonal'] and any(initials_checked['initial_seasonal_estimate']):
            for k in range(components_dict['components_number_ets_seasonal']):
                if initials_checked['initial_seasonal_estimate'][k]:
                    matrices_dict['mat_vt'][components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'] + k, 
                           1:lags_dict['lags_model'][components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'] + k]] = \
                        B[j:j+lags_dict['lags_model'][components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'] + k] - 2]
                    if model_type_dict['season_type'] == "A":
                        matrices_dict['mat_vt'][components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'] + k, 
                               lags_dict['lags_model'][components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'] + k] - 1] = \
                            -np.sum(B[j:j+lags_dict['lags_model'][components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'] + k] - 2])
                    else:  # "M"
                        matrices_dict['mat_vt'][components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'] + k, 
                               lags_dict['lags_model'][components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'] + k] - 1] = \
                            1 / np.prod(B[j:j+lags_dict['lags_model'][components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'] + k] - 2])
                    j += lags_dict['lags_model'][components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'] + k] - 1
    
    # Initials of ARIMA
    if arima_checked['arima_model']:
        if initials_checked['initial_type'] not in ['complete', 'backcasting'] and initials_checked['initial_arima_estimate']:
            matrices_dict['mat_vt'][components_dict['components_number_ets'] + components_dict['components_number_arima'] - 1, 
                                  :initials_checked['initial_arima_number']] = B[j:j+initials_checked['initial_arima_number']]
            if model_type_dict['error_type'] == "A":
                matrices_dict['mat_vt'][components_dict['components_number_ets'] + arima_checked['non_zero_ari'][:, 1], 
                                      :initials_checked['initial_arima_number']] = \
                    np.dot(arima_polynomials['ariPolynomial'][arima_checked['non_zero_ari'][:, 0]], 
                           B[j:j+initials_checked['initial_arima_number']].reshape(1, -1)) / arima_polynomials['ariPolynomial'][-1]
            else:  # "M"
                matrices_dict['mat_vt'][components_dict['components_number_ets'] + arima_checked['non_zero_ari'][:, 1], 
                                      :initials_checked['initial_arima_number']] = \
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
    
    # Initials of the xreg
    if explanatory_checked['xreg_model'] and (initials_checked['initial_type'] != "complete") and initials_checked['initial_estimate'] and initials_checked['initial_xreg_estimate']:
        xreg_number_to_estimate = sum(explanatory_checked['xreg_parameters_estimated'])
        matrices_dict['mat_vt'][components_dict['components_number_ets'] + components_dict['components_number_arima'] + 
                               np.where(explanatory_checked['xreg_parameters_estimated'] == 1)[0], 
                               :lags_dict['lags_model_max']] = B[j:j+xreg_number_to_estimate]
        j += xreg_number_to_estimate
        # Normalise initials
        for i in np.where(explanatory_checked['xreg_parameters_missing'] != 0)[0]:
            matrices_dict['mat_vt'][components_dict['components_number_ets'] + components_dict['components_number_arima'] + i, 
                                  :lags_dict['lags_model_max']] = \
                -np.sum(matrices_dict['mat_vt'][components_dict['components_number_ets'] + components_dict['components_number_arima'] + 
                                               np.where(explanatory_checked['xreg_parameters_included'] == 
                                                      explanatory_checked['xreg_parameters_missing'][i])[0], 
                                               :lags_dict['lags_model_max']])
    
    # Constant
    if constants_checked['constant_estimate']:
        matrices_dict['mat_vt'][components_dict['components_number_ets'] + components_dict['components_number_arima'] + 
                               explanatory_checked['xreg_number'], :] = B[j]
    
    return {
        'mat_vt': matrices_dict['mat_vt'],
        'mat_wt': matrices_dict['mat_wt'],
        'mat_f': matrices_dict['mat_f'],
        'vec_g': matrices_dict['vec_g'],
        'arima_polynomials': matrices_dict['arima_polynomials']
    }



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
    profiles_recent_table = np.zeros((len(lags_model_all), lags_model_max))
    index_lookup_table = np.ones((len(lags_model_all), obs_all + lags_model_max))
    profile_indices = (
        np.arange(1, lags_model_max * len(lags_model_all) + 1)
        .reshape(-1, len(lags_model_all))
        .T
    )

    # Update matrices based on lagsModelAll
    for i, lag in enumerate(lags_model_all):
        # Create the matrix with profiles based on the provided lags.
        # For every row, fill the first 'lag' elements from 1 to lag
        profiles_recent_table[i, : lag[0]] = np.arange(1, lag[0] + 1)

        # For the i-th row in indexLookupTable, fill with a repeated sequence starting
        # from lagsModelMax to the end of the row.
        # The repeated sequence is the i-th row of profileIndices, repeated enough times
        # to cover 'obsAll' observations.
        # '- 1' at the end adjusts these values to Python's zero-based indexing.
        index_lookup_table[i, lags_model_max : (lags_model_max + obs_all)] = (
            np.tile(
                profile_indices[i, : lags_model_all[i][0]],
                int(np.ceil(obs_all / lags_model_all[i][0])),
            )[0:obs_all]
            - 1
        )

        # Extract unique values from from lagsModelMax to lagsModelMax + obsAll of
        # indexLookupTable
        unique_values = np.unique(
            index_lookup_table[i, lags_model_max : lags_model_max + obs_all]  # noqa
        )

        # fix the head of teh data before the sample starts
        # Repeat the unique values lagsModelMax times and then trim the sequence to only
        # keep the first lagsModelMax elements
        index_lookup_table[i, :lags_model_max] = np.tile(unique_values, lags_model_max)[
            -lags_model_max:
        ]

    # Convert to int!
    index_lookup_table = index_lookup_table.astype(int)

    # Note: I skip andling of special cases (e.g., daylight saving time, leap years)
    return {
        "recent": np.array(profiles_recent_table, dtype="float64"),
        "lookup": np.array(index_lookup_table, dtype="int64"),
    }


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
    profiles_recent_provided: bool = False
) -> Dict[str, Any]:
    """
    Constructs the architecture for ADAM models.

    Args:
        model_type_dict: Dictionary containing model type information (ets_model, error_type, etc.)
        lags_dict: Dictionary containing lags information
        observations_dict: Dictionary containing observation information
        initial_type: Type of initial values
        arima_checked: Dictionary containing ARIMA model parameters
        explanatory_checked: Dictionary containing explanatory variables info
        constants_checked: Dictionary containing constant term info
        profiles_recent_table: Pre-computed recent profiles table (optional)
        profiles_recent_provided: Whether profiles_recent_table is provided

    Returns:
        Dictionary containing model architecture components
    """
    # Extract values from dictionaries
    ets_model = model_type_dict["ets_model"]
    E_type = model_type_dict["error_type"]
    T_type = model_type_dict["trend_type"]
    S_type = model_type_dict["season_type"]
    
    lags = lags_dict["lags"]
    lags_model_seasonal = lags_dict.get("lags_model_seasonal", [])
    
    # Set defaults for optional parameters
    arima_model = False if arima_checked is None else arima_checked["arima_model"]
    lags_model_ARIMA = [] if arima_checked is None else arima_checked.get("lags_model_arima", [])
    
    xreg_model = False if explanatory_checked is None else explanatory_checked["xreg_model"]
    xreg_number = 0 if explanatory_checked is None else explanatory_checked.get("xreg_number", 0)
    
    constant_required = False if constants_checked is None else constants_checked["constant_required"]

    components = {}

    # If there is ETS
    if ets_model:
        model_is_trendy = T_type != "N"
        if model_is_trendy:
            # Make lags (1, 1)
            lags_model = [[1, 1]]
            components_names_ETS = ["level", "trend"]
        else:
            # Make lags (1, ...)
            lags_model = [[1]]
            components_names_ETS = ["level"]
        
        model_is_seasonal = S_type != "N"
        if model_is_seasonal:
            # If the lags are for the non-seasonal model
            lags_model.extend([[lag] for lag in lags_model_seasonal])
            components_number_ETS_seasonal = len(lags_model_seasonal)
            if components_number_ETS_seasonal > 1:
                components_names_ETS.extend([f"seasonal{i+1}" for i in range(components_number_ETS_seasonal)])
            else:
                components_names_ETS.append("seasonal")
        else:
            components_number_ETS_seasonal = 0
        
        lags_model_all = lags_model
        components_number_ETS = len(lags_model)
    else:
        model_is_trendy = model_is_seasonal = False
        components_number_ETS = components_number_ETS_seasonal = 0
        components_names_ETS = []
        lags_model_all = lags_model = []

    # If there is ARIMA
    components_number_ARIMA = 0
    components_names_ARIMA = []
    if arima_model:
        lags_model_all = lags_model + [[lag] for lag in lags_model_ARIMA]
        components_number_ARIMA = len(lags_model_ARIMA)
        components_names_ARIMA = [f"arima{i+1}" for i in range(components_number_ARIMA)]

    # If constant is needed, add it
    if constant_required:
        lags_model_all.append([1])

    # If there are xreg
    if xreg_model:
        lags_model_all.extend([[1]] * xreg_number)

    lags_model_max = max(max(lag) for lag in lags_model_all) if lags_model_all else 1

    # Define the number of cols that should be in the matvt
    obs_states = observations_dict["obs_in_sample"] + lags_model_max

    # Create ADAM profiles for correct treatment of seasonality
    adam_profiles = adam_profile_creator(lags_model_all, lags_model_max, observations_dict["obs_in_sample"] + lags_model_max,
                                         lags=lags, y_index=None, y_classes=None)
    if profiles_recent_provided:
        profiles_recent_table = profiles_recent_table[:, :lags_model_max]
    else:
        profiles_recent_table = adam_profiles['recent']
    index_lookup_table = adam_profiles['lookup']

    # Update model type info
    model_type_dict.update({
        "model_is_trendy": model_is_trendy,
        "model_is_seasonal": model_is_seasonal
    })

    # Create components dict
    components_dict = {
        "components_number_ets": components_number_ETS,
        "components_number_ets_seasonal": components_number_ETS_seasonal,
        "components_names_ets": components_names_ETS,
        "components_number_arima": components_number_ARIMA,
        "components_names_arima": components_names_ARIMA
    }

    # Update lags dict
    lags_dict.update({
        "lags_model": lags_model,
        "lags_model_all": lags_model_all,
        "lags_model_max": lags_model_max
    })

    # Update observations dict with new info
    observations_dict["obs_states"] = obs_states

    profile_dict = {
        "profiles_recent_table": profiles_recent_table,
        'profiles_recent_provided': profiles_recent_provided,
        "index_lookup_table": index_lookup_table
    }

    # Return all required information
    return (
        model_type_dict,
        components_dict,
        lags_dict,
        observations_dict,
        profile_dict
    )