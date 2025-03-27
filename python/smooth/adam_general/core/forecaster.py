import numpy as np
import pandas as pd
import warnings

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
                  constants_checked
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
    #if general_dict.get("interval") != None:


    # Fix just in case user used 95 etc instead of 0.95 
    level = general_dict["interval_level"]
    level = [l/100 if l > 1 else l for l in level]
    

    # Handle different interval sides
    if general_dict.get("side") == "both":
        level_low = round((1 - level[0]) / 2, 3)
        level_up = round((1 + level[0]) / 2, 3)
        
    elif general_dict.get("side") == "upper":
        #level_low = np.zeros_like(level) 
        level_up =  level
    else:
        level_low = 1 - level
        #level_up = np.ones_like(level)
    # Convert the dataframe with level_low and level_up as column names
    y_forecast_out = pd.DataFrame({
        'mean': y_forecast,
        f'lower_{level_low}': np.nan,  # Return 0 regardless of calculations
        f'upper_{level_up}': np.nan    # Return 0 regardless of calculations
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
            