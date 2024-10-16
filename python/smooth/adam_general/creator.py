import numpy as np
from typing import List, Optional, Dict, Any
from utils import msdecompose, calculate_acf, calculate_pacf

def creator(ets_model, e_type, t_type, s_type, model_is_trendy, model_is_seasonal,
            lags, lags_model, lags_model_arima, lags_model_all, lags_model_max,
            profiles_recent_table, profiles_recent_provided,
            obs_states, obs_in_sample, obs_all, components_number_ets, components_number_ets_seasonal,
            components_names_ets, ot_logical, y_in_sample,
            # Persistence and phi
            persistence=None, persistence_estimate=True,
            persistence_level=None, persistence_level_estimate=True,
            persistence_trend=None, persistence_trend_estimate=True,
            persistence_seasonal=None, persistence_seasonal_estimate=True,
            persistence_xreg=None, persistence_xreg_estimate=True, persistence_xreg_provided=False,
            phi=1,
            # Initials
            initial_type="optimal", initial_estimate=True,
            initial_level=None, initial_level_estimate=True,
            initial_trend=None, initial_trend_estimate=True,
            initial_seasonal=None, initial_seasonal_estimate=True,
            initial_arima=None, initial_arima_estimate=True, initial_arima_number=None,
            initial_xreg_estimate=True, initial_xreg_provided=False,
            # ARIMA elements
            arima_model=False, ar_estimate=True, i_required=False, ma_estimate=True, arma_parameters=None,
            ar_orders=None, i_orders=None, ma_orders=None, ar_required=False, ma_required=False,
            non_zero_ari=None, non_zero_ma=None,
            components_number_arima=0, components_names_arima=None,
            # Explanatory variables
            xreg_model=False, xreg_model_initials=None, xreg_data=None, xreg_number=0, xreg_names=None,
            xreg_parameters_persistence=None,
            # Constant
            constant_required=False, constant_estimate=True, constant_value=None, constant_name=None):
    
    # Matrix of states. Time in columns, components in rows
    mat_vt = np.full((components_number_ets + components_number_arima + xreg_number + constant_required, obs_states), np.nan)

    # Measurement rowvector
    mat_wt = np.ones((obs_all, components_number_ets + components_number_arima + xreg_number + constant_required))

    # If xreg are provided, then fill in the respective values in Wt vector
    if xreg_model:
        mat_wt[:, components_number_ets + components_number_arima:components_number_ets + components_number_arima + xreg_number] = xreg_data

    # Transition matrix
    mat_f = np.eye(components_number_ets + components_number_arima + xreg_number + constant_required)

    # Persistence vector
    vec_g = np.zeros((components_number_ets + components_number_arima + xreg_number + constant_required, 1))
    #vec_g_index = components_names_ets + components_names_arima + xreg_names + constant_name

    obs_nonzero = np.sum(y_in_sample != 0)

    j = 0
    # ETS model, persistence
    if ets_model:
        j += 1
        #vec_g_index[j-1] = "alpha"
        if not persistence_level_estimate:
            vec_g[j-1, 0] = persistence_level
        
        if model_is_trendy:
            j += 1
            #vec_g_index[j-1] = "beta"
            if not persistence_trend_estimate:
                vec_g[j-1, 0] = persistence_trend
        
        if model_is_seasonal:
            if not all(persistence_seasonal_estimate):
                vec_g[j + np.where(np.logical_not(persistence_seasonal_estimate))[0], 0] = persistence_seasonal
            
            #if components_number_ets_seasonal > 1:
                #vec_g_index[j:j+components_number_ets_seasonal] = [f"gamma{i}" for i in range(1, components_number_ets_seasonal+1)]
            #  else:
            #    vec_g_index[j] = "gamma"
            
    # ARIMA model, names for persistence
    if arima_model:
        # Remove diagonal from the ARIMA part of the matrix
        mat_f[j:j+components_number_arima, j:j+components_number_arima] = 0
        #if components_number_arima > 1:
            #vec_g_index[j:j+components_number_arima] = [f"psi{i}" for i in range(1, components_number_arima+1)]
        #else:
            #vec_g_index[j] = "psi"
        j += components_number_arima

    # Modify transition to do drift
    if not arima_model and constant_required:
        mat_f[0, -1] = 1

    # Regression, persistence
    if xreg_model:
        if persistence_xreg_provided and not persistence_xreg_estimate:
            vec_g[j:j+xreg_number, 0] = persistence_xreg
        #vec_g_index[j:j+xreg_number] = [f"delta{i}" for i in xreg_parameters_persistence]

    # Damping parameter value
    if ets_model and model_is_trendy:
        mat_f[0, 1] = phi
        mat_f[1, 1] = phi
        mat_wt[:, 1] = phi

    # If the arma parameters were provided, fill in the persistence
    if arima_model and (not ar_estimate and not ma_estimate):
        # Call polynomial
        arima_polynomials = {key: np.array(value) for key, value in adam_polynomialiser(
            0, ar_orders, i_orders, ma_orders,
            ar_estimate, ma_estimate, arma_parameters, lags
        ).items()}
        
        # Fill in the transition matrix
        if non_zero_ari.shape[0] > 0:
            mat_f[components_number_ets + non_zero_ari[:, 1], components_number_ets + non_zero_ari[:, 1]] = \
                -arima_polynomials['ari_polynomial'][non_zero_ari[:, 0]]
        
        # Fill in the persistence vector
        if non_zero_ari.shape[0] > 0:
            vec_g[components_number_ets + non_zero_ari[:, 1], 0] = \
                -arima_polynomials['ari_polynomial'][non_zero_ari[:, 0]]
        
        if non_zero_ma.shape[0] > 0:
            vec_g[components_number_ets + non_zero_ma[:, 1], 0] += \
                arima_polynomials['ma_polynomial'][non_zero_ma[:, 0]]
    else:
        arima_polynomials = None

    if not profiles_recent_provided:
        # ETS model, initial state
        # If something needs to be estimated...
        if ets_model:
            if initial_estimate:
                # For the seasonal models
                if model_is_seasonal:
                    if obs_nonzero >= lags_model_max * 2:
                        # If either e_type or s_type are multiplicative, do multiplicative decomposition
                        decomposition_type = "multiplicative" if any(x == "M" for x in [e_type, s_type]) else "additive"
                        y_decomposition = msdecompose(y_in_sample, [lag for lag in lags if lag != 1], type=decomposition_type)
                        j = 0
                        # level
                        if initial_level_estimate:
                            mat_vt[j, 0:lags_model_max] = y_decomposition['initial'][0]
                            if xreg_model:
                                if e_type == "A":
                                    mat_vt[j, 0:lags_model_max] -= np.dot(xreg_model_initials[0]['initial_xreg'], xreg_data[0])
                                else:
                                    mat_vt[j, 0:lags_model_max] /= np.exp(np.dot(xreg_model_initials[1]['initial_xreg'], xreg_data[0]))
                        else:
                            mat_vt[j, 0:lags_model_max] = initial_level
                        j += 1
                        # If trend is needed
                        if model_is_trendy:
                            if initial_trend_estimate:
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
                                mat_vt[j, 0:lags_model_max] = initial_trend
                            j += 1

                        # Seasonal components
                        # For pure models use stuff as is
                        if all(x == "A" for x in [e_type, s_type]) or all(x == "M" for x in [e_type, s_type]) or (e_type == "A" and s_type == "M"):
                            for i in range(components_number_ets_seasonal):
                                if initial_seasonal_estimate[i]:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = y_decomposition['seasonal'][i]
                                    # Renormalise the initial seasons
                                    if s_type == "A":
                                        mat_vt[i+j-1, 0:lags_model[i+j-1]] -= \
                                            np.mean(mat_vt[i+j-1, 0:lags_model[i+j-1]])
                                    else:
                                        mat_vt[i+j-1, 0:lags_model[i+j-1]] /= \
                                            np.exp(np.mean(np.log(mat_vt[i+j-1, 0:lags_model[i+j-1]])))
                                else:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = initial_seasonal[i]

                        # For mixed models use a different set of initials
                        elif e_type == "M" and s_type == "A":
                            for i in range(components_number_ets_seasonal):
                                if initial_seasonal_estimate[i]:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = np.log(y_decomposition['seasonal'][i]) * min(y_in_sample[ot_logical])
                                    # Renormalise the initial seasons
                                    if s_type == "A":
                                        mat_vt[i+j-1, 0:lags_model[i+j-1]] -= np.mean(mat_vt[i+j-1, 0:lags_model[i+j-1]])
                                    else:
                                        mat_vt[i+j-1, 0:lags_model[i+j-1]] /= np.exp(np.mean(np.log(mat_vt[i+j-1, 0:lags_model[i+j-1]])))
                                else:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = initial_seasonal[i]
                    else:
                        # If either e_type or s_type are multiplicative, do multiplicative decomposition
                        j = 0
                        # level
                        if initial_level_estimate:
                            mat_vt[j, 0:lags_model_max] = np.mean(y_in_sample[0:lags_model_max])
                            if xreg_model:
                                if e_type == "A":
                                    mat_vt[j, 0:lags_model_max] -= np.dot(xreg_model_initials[0]['initial_xreg'], xreg_data[0])
                                else:
                                    mat_vt[j, 0:lags_model_max] /= np.exp(np.dot(xreg_model_initials[1]['initial_xreg'], xreg_data[0]))
                        else:
                            mat_vt[j, 0:lags_model_max] = initial_level
                        j += 1
                        if model_is_trendy:
                            if initial_trend_estimate:
                                if t_type == "A":
                                    # trend
                                    mat_vt[j, 0:lags_model_max] = y_in_sample[1] - y_in_sample[0]
                                elif t_type == "M":
                                    if initial_level_estimate:
                                        # level fix
                                        mat_vt[j-1, 0:lags_model_max] = np.exp(np.mean(np.log(y_in_sample[ot_logical][0:lags_model_max])))
                                    # trend
                                    mat_vt[j, 0:lags_model_max] = y_in_sample[1] / y_in_sample[0]
                                # This is a failsafe for multiplicative trend models, so that the thing does not explode
                                if t_type == "M" and np.any(mat_vt[j, 0:lags_model_max] > 1.1):
                                    mat_vt[j, 0:lags_model_max] = 1
                            else:
                                mat_vt[j, 0:lags_model_max] = initial_trend

                            # Do roll back. Especially useful for backcasting and multisteps
                            if t_type == "A":
                                mat_vt[j-1, 0:lags_model_max] = mat_vt[j-1, 0] - mat_vt[j, 0] * lags_model_max
                            elif t_type == "M":
                                mat_vt[j-1, 0:lags_model_max] = mat_vt[j-1, 0] / mat_vt[j, 0]**lags_model_max
                            j += 1

                        # Seasonal components
                        if s_type == "A":
                            for i in range(components_number_ets_seasonal):
                                if initial_seasonal_estimate[i]:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = y_in_sample[0:lags_model[i+j-1]] - mat_vt[0, 0]
                                    # Renormalise the initial seasons
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] -= np.mean(mat_vt[i+j-1, 0:lags_model[i+j-1]])
                                else:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = initial_seasonal[i]
                        # For mixed models use a different set of initials
                        else:
                            for i in range(components_number_ets_seasonal):
                                if initial_seasonal_estimate[i]:
                                    # abs() is needed for mixed ETS+ARIMA
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = y_in_sample[0:lags_model[i+j-1]] / abs(mat_vt[0, 0])
                                    # Renormalise the initial seasons
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] /= np.exp(np.mean(np.log(mat_vt[i+j-1, 0:lags_model[i+j-1]])))
                                else:
                                    mat_vt[i+j-1, 0:lags_model[i+j-1]] = initial_seasonal[i]
                else:
                    # Non-seasonal models
                    # level
                    if initial_level_estimate:
                        mat_vt[0, 0:lags_model_max] = np.mean(y_in_sample[0:max(lags_model_max, int(obs_in_sample * 0.2))])
                    else:
                        mat_vt[0, 0:lags_model_max] = initial_level
                    
                    if model_is_trendy:
                        if initial_trend_estimate:
                            if t_type == "A":
                                mat_vt[1, 0:lags_model_max] = np.mean(np.diff(y_in_sample[0:max(lags_model_max + 1, int(obs_in_sample * 0.2))]))
                            else:  # t_type == "M"
                                mat_vt[1, 0:lags_model_max] = np.exp(np.mean(np.diff(np.log(y_in_sample[ot_logical]))))
                        else:
                            mat_vt[1, 0:lags_model_max] = initial_trend

                if initial_level_estimate and e_type == "M" and mat_vt[0, lags_model_max-1] == 0:
                    mat_vt[0, 0:lags_model_max] = np.mean(y_in_sample)

            # Else, insert the provided ones... make sure that this is not a backcasting
            elif not initial_estimate and initial_type == "provided":
                j = 0
                mat_vt[j, 0:lags_model_max] = initial_level
                if model_is_trendy:
                    j += 1
                    mat_vt[j, 0:lags_model_max] = initial_trend
                if model_is_seasonal:
                    for i in range(components_number_ets_seasonal):
                        # This is misaligned, but that's okay, because this goes directly to profile_recent
                        mat_vt[j+i, 0:lags_model[j+i]] = initial_seasonal[i]
                j += components_number_ets_seasonal

        # If ARIMA orders are specified, prepare initials
        if arima_model:
            if initial_arima_estimate:
                mat_vt[components_number_ets:components_number_ets+components_number_arima, 0:initial_arima_number] = 0 if e_type == "A" else 1
                
                if any(lag > 1 for lag in lags):
                    y_decomposition = msdecompose(y_in_sample, [lag for lag in lags if lag != 1], 
                                                type="additive" if e_type == "A" else "multiplicative")['seasonal'][-1][0]
                else:
                    y_decomposition = np.mean(np.diff(y_in_sample[ot_logical])) if e_type == "A" else np.exp(np.mean(np.diff(np.log(y_in_sample[ot_logical]))))
                
                mat_vt[components_number_ets+components_number_arima-1, 0:initial_arima_number] = \
                    np.tile(y_decomposition, int(np.ceil(initial_arima_number / max(lags))))[:initial_arima_number]
            
            else:
                mat_vt[components_number_ets:components_number_ets+components_number_arima, 0:initial_arima_number] = 0 if e_type == "A" else 1
                mat_vt[components_number_ets+components_number_arima-1, 0:initial_arima_number] = initial_arima[:initial_arima_number]

        # Fill in the initials for xreg
        if xreg_model:
            if e_type == "A" or initial_xreg_provided or xreg_model_initials[1] is None:
                mat_vt[components_number_ets+components_number_arima:components_number_ets+components_number_arima+xreg_number, 0:lags_model_max] = \
                    xreg_model_initials[0]['initial_xreg']
            else:
                mat_vt[components_number_ets+components_number_arima:components_number_ets+components_number_arima+xreg_number, 0:lags_model_max] = \
                    xreg_model_initials[1]['initial_xreg']

        # Add constant if needed
        if constant_required:
            if constant_estimate:
                # Add the mean of data
                if sum(i_orders) == 0 and not ets_model:
                    mat_vt[components_number_ets+components_number_arima+xreg_number, :] = np.mean(y_in_sample[ot_logical])
                # Add first differences
                else:
                    if e_type == "A":
                        mat_vt[components_number_ets+components_number_arima+xreg_number, :] = np.mean(np.diff(y_in_sample[ot_logical]))
                    else:
                        mat_vt[components_number_ets+components_number_arima+xreg_number, :] = np.exp(np.mean(np.diff(np.log(y_in_sample[ot_logical]))))
            else:
                mat_vt[components_number_ets+components_number_arima+xreg_number, :] = constant_value
            
            # If ETS model is used, change the initial level
            if ets_model and initial_level_estimate:
                if e_type == "A":
                    mat_vt[0, 0:lags_model_max] -= mat_vt[components_number_ets+components_number_arima+xreg_number, 0]
                else:
                    mat_vt[0, 0:lags_model_max] /= mat_vt[components_number_ets+components_number_arima+xreg_number, 0]
            
            # If ARIMA is done, debias states
            if arima_model and initial_arima_estimate:
                if e_type == "A":
                    mat_vt[components_number_ets+non_zero_ari[:, 1], 0:initial_arima_number] -= \
                        mat_vt[components_number_ets+components_number_arima+xreg_number, 0]
                else:
                    mat_vt[components_number_ets+non_zero_ari[:, 1], 0:initial_arima_number] /= \
                        mat_vt[components_number_ets+components_number_arima+xreg_number, 0]
    else:
        mat_vt[:, 0:lags_model_max] = profiles_recent_table

    return {'mat_vt': mat_vt, 'mat_wt': mat_wt, 'mat_f': mat_f, 'vec_g': vec_g, 'arima_polynomials': arima_polynomials}



import numpy as np

from statsmodels.tsa.stattools import acf, pacf
import numpy as np

def initialiser(
    ets_model, e_type, t_type, s_type, model_is_trendy, model_is_seasonal,
    components_number_ets_non_seasonal, components_number_ets_seasonal, components_number_ets,
    lags, lags_model, lags_model_seasonal, lags_model_arima, lags_model_max,
    mat_vt,
    # Persistence values
    persistence_estimate=True,
    persistence_level_estimate=True, persistence_trend_estimate=True,
    persistence_seasonal_estimate=True, persistence_xreg_estimate=True,
    # Initials
    phi_estimate=True, initial_type="optimal", initial_estimate=True,
    initial_level_estimate=True, initial_trend_estimate=True, initial_seasonal_estimate=True,
    initial_arima_estimate=True, initial_xreg_estimate=True,
    # ARIMA elements
    arima_model=False, ar_required=False, ma_required=False, 
    ar_estimate=True, ma_estimate=True, 
    ar_orders=None, ma_orders=None, i_orders=None,
    components_number_arima=0, components_names_arima=None, initial_arima_number=None,
    # Explanatory variables
    xreg_model=False, xreg_number=0,
    xreg_parameters_estimated=None, xreg_parameters_persistence=None,
    # Constant and other stuff
    constant_estimate=True, constant_name=None, other_parameter_estimate=False,
    # Additional parameters
    y_in_sample=None, ot_logical=None, bounds="usual", other=None
):
    persistence_estimate_vector = [
        persistence_level_estimate,
        model_is_trendy and persistence_trend_estimate,
        model_is_seasonal and any(persistence_seasonal_estimate)
    ]
    total_params = (
        ets_model * (sum(persistence_estimate_vector) + phi_estimate) +
        xreg_model * persistence_xreg_estimate * max(xreg_parameters_persistence or [0]) +
        arima_model * (ar_estimate * sum(ar_orders or []) + ma_estimate * sum(ma_orders or [])) +
        ets_model * (initial_type not in ["complete", "backcasting"]) * (
            initial_level_estimate +
            (model_is_trendy * initial_trend_estimate) +
            (model_is_seasonal * sum(initial_seasonal_estimate * (np.array(lags_model_seasonal or []) - 1)))
        ) +
        (initial_type not in ["complete", "backcasting"]) * arima_model * (initial_arima_number or 0) * initial_arima_estimate +
        (initial_type != "complete") * xreg_model * initial_xreg_estimate * sum(xreg_parameters_estimated or []) +
        constant_estimate + other_parameter_estimate
    )

    B = np.full(total_params, np.nan)
    Bl = np.full(total_params, np.nan)
    Bu = np.full(total_params, np.nan)
    names = []

    j = 0

    if ets_model:
        if persistence_estimate and any(persistence_estimate_vector):
            if any(ptype == "M" for ptype in [e_type, t_type, s_type]):
                if ((e_type == "A" and t_type == "A" and s_type == "M") or
                    (e_type == "A" and t_type == "M" and s_type == "A") or
                    (initial_type in ["complete", "backcasting"] and
                     ((e_type == "M" and t_type == "A" and s_type == "A") or
                      (e_type == "M" and t_type == "A" and s_type == "M")))):
                    B[j:j+sum(persistence_estimate_vector)] = [0.01, 0] + [0] * components_number_ets_seasonal
                elif e_type == "M" and t_type == "M" and s_type == "A":
                    B[j:j+sum(persistence_estimate_vector)] = [0, 0] + [0] * components_number_ets_seasonal
                elif e_type == "M" and t_type == "A":
                    if initial_type in ["complete", "backcasting"]:
                        B[j:j+sum(persistence_estimate_vector)] = [0.1, 0] + [0.01] * components_number_ets_seasonal
                    else:
                        B[j:j+sum(persistence_estimate_vector)] = [0.2, 0.01] + [0.01] * components_number_ets_seasonal
                elif e_type == "M" and t_type == "M":
                    B[j:j+sum(persistence_estimate_vector)] = [0.1, 0.05] + [0.01] * components_number_ets_seasonal
                else:
                    initial_values = [0.1]
                    if model_is_trendy:
                        initial_values.append(0.05)
                    if model_is_seasonal:
                        initial_values.extend([0.11] * components_number_ets_seasonal)
                    
                    B[j:j+sum(persistence_estimate_vector)] = [val for val, estimate in zip(initial_values, persistence_estimate_vector) if estimate]
            else:
                initial_values = [0.1]
                if model_is_trendy:
                    initial_values.append(0.05)
                if model_is_seasonal:
                    initial_values.extend([0.11] * components_number_ets_seasonal)
                
                B[j:j+sum(persistence_estimate_vector)] = [val for val, estimate in zip(initial_values, persistence_estimate_vector) if estimate]

            if bounds == "usual":
                Bl[j:j+sum(persistence_estimate_vector)] = 0
                Bu[j:j+sum(persistence_estimate_vector)] = 1
            else:
                Bl[j:j+sum(persistence_estimate_vector)] = -5
                Bu[j:j+sum(persistence_estimate_vector)] = 5

            # Names for B
            if persistence_level_estimate:
                names.append("alpha")
                j += 1
            if model_is_trendy and persistence_trend_estimate:
                names.append("beta")
                j += 1
            if model_is_seasonal and any(persistence_seasonal_estimate):
                if components_number_ets_seasonal > 1:
                    names.extend([f"gamma{i}" for i in range(1, components_number_ets_seasonal+1)])
                else:
                    names.append("gamma")
                j += sum(persistence_seasonal_estimate)

    if xreg_model and persistence_xreg_estimate:
        xreg_persistence_number = max(xreg_parameters_persistence)
        B[j:j+xreg_persistence_number] = 0.01 if e_type == "A" else 0
        Bl[j:j+xreg_persistence_number] = -5
        Bu[j:j+xreg_persistence_number] = 5
        names.extend([f"delta{i+1}" for i in range(xreg_persistence_number)])
        j += xreg_persistence_number

    if ets_model and phi_estimate:
        B[j] = 0.95
        names.append("phi")
        Bl[j] = 0
        Bu[j] = 1
        j += 1

    if arima_model:
        if any([ar_estimate, ma_estimate]):
            acf_values = [-0.1] * sum(ma_orders * lags)
            pacf_values = [0.1] * sum(ar_orders * lags)
            
            if not (ets_model or all(i_orders == 0)):
                y_differenced = y_in_sample.copy()
                # Implement differencing if needed
                if any(i_orders > 0):
                    for i, order in enumerate(i_orders):
                        if order > 0:
                            y_differenced = np.diff(y_differenced, n=order, axis=0)
                
                # ACF/PACF calculation for non-seasonal models
                if all(np.array(lags) <= 1):
                    if ma_required and ma_estimate:
                        acf_values[:min(sum(ma_orders * lags), len(y_differenced) - 1)] = calculate_acf(y_differenced, nlags=max(1, sum(ma_orders * lags)))[1:]
                    if ar_required and ar_estimate:
                        pacf_values[:min(sum(ar_orders * lags), len(y_differenced) - 1)] = calculate_pacf(y_differenced, nlags=max(1, sum(ar_orders * lags)))
            
            for i, lag in enumerate(lags):
                if ar_required and ar_estimate and ar_orders[i] > 0:
                    B[j:j+ar_orders[i]] = pacf_values[i*lag:(i+1)*lag][:ar_orders[i]]
                    if sum(B[j:j+ar_orders[i]]) > 1:
                        B[j:j+ar_orders[i]] = B[j:j+ar_orders[i]] / sum(B[j:j+ar_orders[i]]) - 0.01
                    Bl[j:j+ar_orders[i]] = -5
                    Bu[j:j+ar_orders[i]] = 5
                    names.extend([f"phi{k+1}[{lag}]" for k in range(ar_orders[i])])
                    j += ar_orders[i]
                
                if ma_required and ma_estimate and ma_orders[i] > 0:
                    B[j:j+ma_orders[i]] = acf_values[i*lag:(i+1)*lag][:ma_orders[i]]
                    if sum(B[j:j+ma_orders[i]]) > 1:
                        B[j:j+ma_orders[i]] = B[j:j+ma_orders[i]] / sum(B[j:j+ma_orders[i]]) - 0.01
                    Bl[j:j+ma_orders[i]] = -5
                    Bu[j:j+ma_orders[i]] = 5
                    names.extend([f"theta{k+1}[{lag}]" for k in range(ma_orders[i])])
                    j += ma_orders[i]

    if ets_model and initial_type not in ["complete", "backcasting"] and initial_estimate:
        if initial_level_estimate:
            B[j] = mat_vt[0, 0]
            Bl[j] = -np.inf if e_type == "A" else 0
            Bu[j] = np.inf
            names.append("level")
            j += 1
        if model_is_trendy and initial_trend_estimate:
            B[j] = mat_vt[1, 0]
            Bl[j] = -np.inf if t_type == "A" else 0
            Bu[j] = np.inf
            names.append("trend")
            j += 1
        if model_is_seasonal and any(initial_seasonal_estimate):
            for k in range(components_number_ets_seasonal):
                if initial_seasonal_estimate[k]:
                    B[j:j+lags_model[components_number_ets_non_seasonal+k]-1] = mat_vt[components_number_ets_non_seasonal+k, 1:lags_model[components_number_ets_non_seasonal+k]]
                    if s_type == "A":
                        Bl[j:j+lags_model[components_number_ets_non_seasonal+k]-1] = -np.inf
                        Bu[j:j+lags_model[components_number_ets_non_seasonal+k]-1] = np.inf
                    else:
                        Bl[j:j+lags_model[components_number_ets_non_seasonal+k]-1] = 0
                        Bu[j:j+lags_model[components_number_ets_non_seasonal+k]-1] = np.inf
                    names.extend([f"seasonal{k+1}_{m}" for m in range(2, lags_model[components_number_ets_non_seasonal+k])])
                    j += lags_model[components_number_ets_non_seasonal+k] - 1

    if initial_type not in ["complete", "backcasting"] and arima_model and initial_arima_estimate:
        B[j:j+initial_arima_number] = mat_vt[components_number_ets+components_number_arima, :initial_arima_number]
        names.extend([f"ARIMAState{n}" for n in range(1, initial_arima_number+1)])
        if e_type == "A":
            Bl[j:j+initial_arima_number] = -np.inf
            Bu[j:j+initial_arima_number] = np.inf
        else:
            B[j:j+initial_arima_number] = np.abs(B[j:j+initial_arima_number])
            Bl[j:j+initial_arima_number] = 0
            Bu[j:j+initial_arima_number] = np.inf
        j += initial_arima_number

    if initial_type != "complete" and initial_xreg_estimate and xreg_model:
        xreg_number_to_estimate = sum(xreg_parameters_estimated or [])
        if xreg_number_to_estimate > 0:
            B[j:j+xreg_number_to_estimate] = mat_vt[components_number_ets+components_number_arima:components_number_ets+components_number_arima+xreg_number, 0]
            names.extend([f"xreg{idx+1}" for idx in range(xreg_number_to_estimate)])
            Bl[j:j+xreg_number_to_estimate] = -np.inf
            Bu[j:j+xreg_number_to_estimate] = np.inf
            j += xreg_number_to_estimate

    if constant_estimate:
        j += 1
        if mat_vt.shape[0] > components_number_ets + components_number_arima + xreg_number:
            B[j-1] = mat_vt[components_number_ets + components_number_arima + xreg_number, 0]
        else:
            B[j-1] = 0  # or some other default value
        names.append(constant_name or "constant")
        if ets_model or (i_orders is not None and sum(i_orders) != 0):
            if e_type == "A":
                Bu[j-1] = np.quantile(np.diff(y_in_sample[ot_logical]), 0.6)
                Bl[j-1] = -Bu[j-1]
            else:
                Bu[j-1] = np.exp(np.quantile(np.diff(np.log(y_in_sample[ot_logical])), 0.6))
                Bl[j-1] = np.exp(np.quantile(np.diff(np.log(y_in_sample[ot_logical])), 0.4))
            
            if Bu[j-1] <= Bl[j-1]:
                Bu[j-1] = np.inf
                Bl[j-1] = -np.inf if e_type == "A" else 0
            
            if B[j-1] <= Bl[j-1]:
                Bl[j-1] = -np.inf if e_type == "A" else 0
            if B[j-1] >= Bu[j-1]:
                Bu[j-1] = np.inf
        else:
            Bu[j-1] = max(abs(y_in_sample[ot_logical]), abs(B[j-1]) * 1.01)
            Bl[j-1] = -Bu[j-1]

    if other_parameter_estimate:
        j += 1
        B[j-1] = other
        names.append("other")
        Bl[j-1] = 1e-10
        Bu[j-1] = np.inf

    return {
        "B": B[:j],
        "Bl": Bl[:j],
        "Bu": Bu[:j],
        "names": names
    }
    