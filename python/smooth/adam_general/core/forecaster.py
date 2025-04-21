import numpy as np
import pandas as pd
import warnings

from core.utils.var_covar import sigma, covar_anal, var_anal
from scipy import stats
from scipy.special import gamma
import numpy as np

from smooth.adam_general._adam_general import adam_forecaster, adam_fitter
from smooth.adam_general.core.creator import adam_profile_creator
from smooth.adam_general.core.creator import filler
from smooth.adam_general.core.utils.utils import scaler

def forecaster(model_prepared,
                  observations_dict,
                  general_dict,
                  occurrence_dict,
                  lags_dict,
                  model_type_dict,
                  explanatory_checked,
                  components_dict,
                  constants_checked,
                  params_info,
                  level
                  ):
    
    observations_dict["y_forecast_index"] = pd.date_range(
        start=observations_dict["y_forecast_start"], 
        periods=general_dict["h"], 
        freq=observations_dict["frequency"]
    )

    # Check what was returned in the end
    if np.any(np.isnan(model_prepared['y_fitted'])) or np.any(pd.isna(model_prepared['y_fitted'])):
        warnings.warn("Something went wrong in the estimation of the model and NaNs were produced. "
                        "If this is a mixed model, consider using the pure ones instead.")

    if occurrence_dict["occurrence_model"]:
        model_prepared['y_fitted'][:] = model_prepared['y_fitted'] * occurrence_dict["p_fitted"]

    # Fix the cases, when we have zeroes in the provided occurrence
    if occurrence_dict["occurrence"] == "provided":
        model_prepared['y_fitted'][~occurrence_dict["ot_logical"]] = model_prepared['y_fitted'][~occurrence_dict["ot_logical"]] * occurrence_dict["p_fitted"][~occurrence_dict["ot_logical"]]

    # Produce forecasts if the horizon is non-zero
    if general_dict["h"] > 0:
        if not isinstance(observations_dict.get("y_in_sample"), pd.Series):
            y_forecast = pd.Series(np.full(general_dict["h"], np.nan), 
                                    index=pd.date_range(start=observations_dict["y_forecast_start"], 
                                                    periods=general_dict["h"], 
                                                    freq=observations_dict["frequency"]))
        else:
            y_forecast = pd.Series(np.full(general_dict["h"], np.nan), 
                                    index=observations_dict["y_forecast_index"])
            


    # Prepare the lookup
    lookup_result = adam_profile_creator(
                lags_model_all=lags_dict["lags_model_all"], 
                lags_model_max=lags_dict["lags_model_max"], 
                obs_all=observations_dict["obs_in_sample"]+general_dict["h"],
                lags=lags_dict["lags"], 
            )
    lookup = lookup_result['lookup'][:, (observations_dict["obs_in_sample"]+lags_dict["lags_model_max"]):]

    
    # get the matrices
    # get the matrcies
    mat_vt = model_prepared['states'][:, observations_dict["obs_states"]-lags_dict["lags_model_max"]:observations_dict["obs_states"]+1]
    if model_prepared['measurement'].shape[0] < general_dict["h"]:
        mat_wt = np.tile(model_prepared['measurement'][-1], (general_dict["h"], 1))
    else:
        mat_wt = model_prepared['measurement'][-general_dict["h"]:]
    #mat_f = adam_fitted['matF']
    vec_g = model_prepared['persistence']
    mat_f = model_prepared['transition']


    # If this is "prediction", do simulations for multiplicative components
    if general_dict["interval"] == "prediction":
        # Simulate stuff for the ETS only
        if ((model_type_dict['ets_model'] or explanatory_checked["xreg_number"] > 0) and 
            (model_type_dict["trend_type"] == "M" or 
            (model_type_dict["season_type"] == "M" and general_dict["h"] > lags_dict["lags_model_min"]))):
            general_dict["interval"] = "simulated"
        else:
            general_dict["interval"] = "approximate"


    # Produce point forecasts for non-multiplicative trend / seasonality
    # Do this for cases when h<=m as well and prediction/confidence/simulated interval
    # skip for now 
    #if ((model_type_dict["trend_type"] != "M" and 
    #    (model_type_dict["season_type"] != "M" or 
    #    (model_type_dict["season_type"] == "M" and general_dict["h"] <= lags_dict["lags_model_min"]))) or
    #    general_dict["interval"] in ["nonparametric", "semiparametric", "empirical", "approximate"]):

    y_in_sample = np.asarray(observations_dict['y_in_sample'].values.flatten(), dtype=np.float64)
    ot = np.asarray(observations_dict['ot'].values.flatten(), dtype=np.float64)
    lags_model_all = np.asfortranarray(lags_dict['lags_model_all'], dtype=np.uint64).reshape(-1,1)  # Make sure it's a 1D array
    profiles_recent_table = np.asfortranarray(model_prepared['profiles_recent_table'], dtype=np.float64)
    index_lookup_table = np.asfortranarray(lookup, dtype=np.uint64)


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
            horizon=general_dict["h"]
        ).flatten() 
    #print(y_forecast)
    #else:
        # If we do simulations, leave it for later
        #if general_dict["interval"] == "simulated":
        #    y_forecast = np.zeros(general_dict["h"])
        # If we don't, do simulations to get mean
    #else:
        # TODO: Implement forecast function with simulationo for interval
        #pass 
    
    # Make safety checks
    # If there are NaN values
    if np.any(np.isnan(y_forecast)):
        y_forecast[np.isnan(y_forecast)] = 0

    # Make a warning about the potential explosive trend
    if (model_type_dict["trend_type"] == "M" and not model_type_dict["damped"] and 
        model_prepared["profiles_recent_table"][1,0] > 1 and general_dict["h"] > 10):
        warnings.warn("Your model has a potentially explosive multiplicative trend. "
                    "I cannot do anything about it, so please just be careful.")

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


    # How many levels did user ask to produce
    n_levels = len(general_dict["interval_level"])

    # Cumulative forecasts have only one observation
    if general_dict.get("cumulative"):
        # h_final is the number of elements we will have in the final forecast
        h_final = 1
        # In case of occurrence model use simulations - the cumulative probability is complex
        if occurrence_model:
            general_dict["interval"] = "simulated"
    else:
        h_final = general_dict["h"]


    # Create necessary arrays for the forecasts
    #if general_dict.get("cumulative"):
     #    y_forecast = np.sum(y_forecast * p_forecast)
    #    y_upper = y_lower = np.zeros((h_final, n_levels))
    #else:
    #    y_forecast = y_forecast * p_forecast
    #    y_upper = y_lower = np.zeros((h_final, n_levels))


    # Handle intervals if specified

    # if general_dict.get("interval") != None:
    # TODO: Implement all intervals. for now I assume its none.
    # next version comment in the following code
    if general_dict.get("interval") != None:


        if level is not None:
            # Fix just in case user used 95 etc instead of 0.95 
            level = level/100 if level > 1 else level
            
            # Handle different interval sides
            if general_dict.get("side") == "both":
                level_low = round((1 - level) / 2, 3)
                level_up = round((1 + level) / 2, 3)
                
            elif general_dict.get("side") == "upper":
                #level_low = np.zeros_like(level) 
                level_up =  level
            else:
                level_low = 1 - level

            y_lower, y_upper = generate_prediction_interval(y_forecast, mat_wt, mat_f, vec_g, model_prepared, general_dict, observations_dict, model_type_dict, lags_dict, params_info, level)
                #level_up = np.ones_like(level)
        else:
            y_lower = np.nan
            y_upper = np.nan
        
        # Convert the dataframe with level_low and level_up as column names
        y_forecast_out = pd.DataFrame({
            'mean': y_forecast,
            f'lower_{level_low}': y_lower,  # Return 0 regardless of calculations
            f'upper_{level_up}': y_upper    # Return 0 regardless of calculations
        }, index=observations_dict["y_forecast_index"])
    else:
        y_forecast_out = pd.DataFrame({
            'mean': y_forecast,
        }, index=observations_dict["y_forecast_index"])
        
    return y_forecast_out

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
    other=None
):
    

    # Fill in the matrices if needed
    if general_dict.get("model_do") != "use":
        matrices_dict = filler(
            adam_estimated['B'],
            model_type_dict = model_type_dict,
            components_dict = components_dict,
            lags_dict = lags_dict,
            matrices_dict = matrices_dict,
            persistence_checked = persistence_checked,
            initials_checked = initials_checked,
            arima_checked = arima_checked,
            explanatory_checked = explanatory_checked,
            phi_dict = phi_dict,
            constants_checked = constants_checked
        )

        # Write down phi

    # Write down the initials in the recent profile
    profiles_recent_table = matrices_dict['mat_vt'][:, :lags_dict["lags_model_max"]]
    profiles_recent_initial = matrices_dict['mat_vt'][:, :lags_dict["lags_model_max"]].copy()



    # Prepare variables for the estimation
    # Convert pandas Series/DataFrames to numpy arrays
    y_in_sample = np.asarray(observations_dict['y_in_sample'].values.flatten(), dtype=np.float64)
    ot = np.asarray(observations_dict['ot'].values.flatten(), dtype=np.float64)
    mat_vt = np.asfortranarray(matrices_dict['mat_vt'], dtype=np.float64)
    mat_wt = np.asfortranarray(matrices_dict['mat_wt'], dtype=np.float64)
    mat_f = np.asfortranarray(matrices_dict['mat_f'], dtype=np.float64)
    vec_g = np.asfortranarray(matrices_dict['vec_g'], dtype=np.float64) # Make sure it's a 1D array
    lags_model_all = np.asfortranarray(lags_dict['lags_model_all'], dtype=np.uint64).reshape(-1,1)  # Make sure it's a 1D array
    index_lookup_table = np.asfortranarray(profiles_dict['index_lookup_table'], dtype=np.uint64)
    profiles_recent_table = np.asfortranarray(profiles_recent_table, dtype=np.float64)


    adam_fitted = adam_fitter(
        matrixVt=mat_vt,
        matrixWt=mat_wt,
        matrixF=mat_f,
        vectorG=vec_g,
        lags=lags_model_all, 
        indexLookupTable=index_lookup_table,
        profilesRecent=profiles_recent_table,
        E=model_type_dict['error_type'],
        T=model_type_dict['trend_type'],
        S=model_type_dict['season_type'],

        # non seasonal makes the calculation here
        nNonSeasonal=components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'],
        nSeasonal=components_dict['components_number_ets_seasonal'],
        nArima=components_dict['components_number_arima'],
        nXreg=explanatory_checked['xreg_number'],
        constant=constants_checked['constant_required'],
        vectorYt=y_in_sample,  # Ensure correct mapping
        vectorOt=ot,  # Ensure correct mapping
        backcast=any([t == "complete" or t == "backcasting" for t in initials_checked['initial_type']])    
        )
    
    # Make sure that there are no negative values in multiplicative components
    # This might appear in case of bounds="a"
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

        # Prepare fitted and error with ts / zoo
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

    errors[:] = adam_fitted["errors"].flatten()
    y_fitted[:] = adam_fitted["yFitted"].flatten() 


    # If the distribution is default, change it according to the error term
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
            initial_estimated[j + 1:j + 1 + components_dict["components_number_ets_seasonal"]] = initials_checked["initial_seasonal_estimate"]
            # Remove the level from ETS list
            initial_value_ets[0] = None
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


    # Remove xreg persistence if needed
    if explanatory_checked["xreg_model"] and explanatory_checked.get("regressors") != "adapt":
        explanatory_checked["regressors"] = "use"
    elif not explanatory_checked["xreg_model"]:
        explanatory_checked["regressors"] = None 

    # Handle ARMA parameters
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

    # Handle distribution parameters
    # for now I am skipping this one
    if general_dict["distribution_new"] in ["dalaplace", "dgnorm", "dlgnorm", "dt"] and initials_checked["other_parameter_estimate"]:
        other = abs(adam_estimated['B'],[-1])

    # Calculate scale parameter using scaler function
    scale = scaler(general_dict["distribution_new"], 
                   model_type_dict["error_type"], 
                   errors[observations_dict["ot_logical"]], 
                   y_fitted[observations_dict["ot_logical"]], 
                   observations_dict["obs_in_sample"], 
                   other)

    # Record constant if estimated
    if constants_checked["constant_estimate"]:
        constant_value = adam_estimated['B'],[constants_checked["constant_name"]]
    else:
        constant_value = adam_estimated['B'][-1]
    # Prepare distribution parameters to return
    other_returned = {}
    
    # Write down parameters for distribution (always positive)
    # we skip the distributional parameters for now
    
    #if initials_checked["other_parameter_estimate"]:
    #    param_value = abs(adam_estimated['B'][-1])
    #else:
    #    param_value = other

    # Set parameter name based on distribution
    #if general_dict["distribution"] == "dalaplace":
    #    other_returned["alpha"] = param_value
    #elif general_dict["distribution"] in ["dgnorm", "dlgnorm"]:
    #    other_returned["shape"] = param_value
    #elif general_dict["distribution"] == "dt":
    #    other_returned["nu"] = param_value

    # Add LASSO/RIDGE lambda if applicable
    if general_dict["loss"] in ["LASSO", "RIDGE"]:
        other_returned["lambda"] = general_dict["lambda_"]

    # Return ARIMA polynomials and indices for persistence and transition
    if arima_checked["arima_model"]:
        other_returned["polynomial"] = adam_estimated['arima_polynomials']
        other_returned["ARIMA_indices"] = {
            "nonZeroARI": arima_checked["non_zero_ari"], 
            "nonZeroMA": arima_checked["non_zero_ma"]
        }
        other_returned["ar_polynomial_matrix"] = np.zeros((sum(arima_checked["ar_orders"]) * lags_dict["lags"], 
                                                         sum(arima_checked["ar_orders"]) * lags_dict["lags"]))
        
        if other_returned["ar_polynomial_matrix"].shape[0] > 1:
            # Set diagonal elements to 1 except first row/col
            other_returned["ar_polynomial_matrix"][1:-1, 2:] = np.eye(other_returned["ar_polynomial_matrix"].shape[0]-2)
            
            if arima_checked["ar_required"]:
                other_returned["ar_polynomial_matrix"][:, 0] = -arima_polynomials["ar_polynomial"][1:]
                
        other_returned["arma_parameters"] = arima_checked["arma_parameters"]

    # Update parameters number
    # There is an issue here that I need to fix with the parameters number
    general_dict["parameters_number"][0][2] = np.sum(general_dict["parameters_number"][0][:2]) 

    return {
            "model": model_type_dict["model"],
            "time_elapsed": None, # here will count the time
            #"data": np.column_stack((None, explanatory_checked["xreg_data"])),
            "holdout": general_dict["holdout"],
            "y_fitted": y_fitted,
            "residuals": errors,
            "states": adam_fitted['matVt'],
            "profiles_recent_table": adam_fitted['profile'],
            "persistence": matrices_dict['vec_g'],
            "transition": matrices_dict['mat_f'],
            "measurement": matrices_dict['mat_wt'],
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
            "B": adam_estimated['B'],
            "lags": lags_dict["lags"],
            "lags_all": lags_dict["lags_model_all"],
            "FI": general_dict.get("fi")
    }
            

def generate_prediction_interval(predictions, 
                                 mat_wt, 
                                 mat_f, 
                                 vec_g, 
                                 prepared_model, 
                                 general, 
                                 observations_dict,
                                 model_type_dict,
                                 lags_dict,
                        
                                params_info, level):
    

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