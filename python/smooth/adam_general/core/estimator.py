import numpy as np
import nlopt
from core.utils.ic import ic_function
import pandas as pd
from core.creator import initialiser, creator, architector
from core.utils.cost_functions import CF, log_Lik_ADAM
from smooth.adam_general._adam_general import adam_fitter, adam_forecaster
from core.creator import creator, initialiser, architector, filler
import warnings
from core.utils.utils import scaler


def estimator(
        general_dict,
        model_type_dict,
        lags_dict,
        observations_dict,
        arima_dict,
        constant_dict,
        explanatory_dict,
        profiles_recent_table,
        profiles_recent_provided,
        persistence_dict,
        initials_dict,
        phi_dict,
        components_dict,
        occurrence_dict,

        multisteps = False,
        lb = None,
        ub = None,
        maxtime = None,
        print_level = 1, # 1 or 0
        maxeval = None,
):

    

    # Create the basic variables
    model_type_dict, components_dict, lags_dict, observations_dict, profile_dict = architector(
    model_type_dict = model_type_dict,
    lags_dict = lags_dict,
    observations_dict = observations_dict,
    arima_checked = arima_dict,
    constants_checked = constant_dict,
    explanatory_checked = explanatory_dict,
    profiles_recent_table = profiles_recent_table,
    profiles_recent_provided = profiles_recent_provided
)

    # Create the matrices for the specific ETS model
    adam_created = creator(
        model_type_dict = model_type_dict,
        lags_dict = lags_dict,
        profiles_dict = profile_dict,
        observations_dict = observations_dict,

        persistence_checked = persistence_dict,
        initials_checked = initials_dict,
        arima_checked = arima_dict,
        constants_checked = constant_dict,
        phi_dict = phi_dict,
        components_dict = components_dict,
        explanatory_checked = explanatory_dict
    )
    
    # Initialize B
    # Initialize B
    b_values = initialiser(
        model_type_dict = model_type_dict,
        components_dict = components_dict,
        lags_dict = lags_dict,
        adam_created = adam_created,
        persistence_checked = persistence_dict,
        initials_checked = initials_dict,
        arima_checked = arima_dict,
        constants_checked = constant_dict,
        explanatory_checked = explanatory_dict,
        observations_dict = observations_dict,
        bounds = general_dict['bounds'],
        phi_dict = phi_dict,
    )

    # The following is a translation from R -> why do we need it?
    #B = b_values['B']
    #if B is not None:
    #    if isinstance(B, dict):
     #       B = {k: v for k, v in B.items() if k in b_values['B']}
     #       b_values['B'].update(B)
        #else:
        #    b_values['B'][:] = B
        #    B = dict(zip(b_values['names'], B))

    # Instead I do this:
    # Create the vector of initials for the optimisation
    #if B is None:
    B = b_values['B']
    #if lb is None:
    lb = b_values['Bl']
    #if ub is None:
    ub = b_values['Bu']
    

    #if(!is.null(B)){
    #    if(!is.null(names(B))){
    #        B <- B[names(B) %in% names(BValues$B)];
    #        BValues$B[] <- B;
    #    }
    #    else{
     #       BValues$B[] <- B;
     #       names(B) <- names(BValues$B);
     #   }
    #}




    # Preheat the initial state of ARIMA. Do this only for optimal initials and if B is not provided
    if model_type_dict['arima_model'] and initials_dict['initial_type'] == "optimal" and initials_dict['initial_arima_estimate'] and B is None:
        ... # will add later!
        

    

    # Companion matrices for the polynomials calculation -> stationarity / stability checks
    if model_type_dict['arima_model']:
        # AR polynomials
        ar_polynomial_matrix = np.zeros((np.sum(arima_dict['ar_orders']) * lags_dict['lags'], np.sum(arima_dict['ar_orders']) * lags_dict['lags']))
        if ar_polynomial_matrix.shape[0] > 1:
            ar_polynomial_matrix[1:, :-1] = np.eye(ar_polynomial_matrix.shape[0] - 1)
        # MA polynomials
        ma_polynomial_matrix = np.zeros((np.sum(arima_dict['ma_orders']) * lags_dict['lags'], np.sum(arima_dict['ma_orders']) * lags_dict['lags']))
        if ma_polynomial_matrix.shape[0] > 1:
            ma_polynomial_matrix[1:, :-1] = np.eye(ma_polynomial_matrix.shape[0] - 1)
    else:
        ma_polynomial_matrix = ar_polynomial_matrix = None

    # If the distribution is default, change it according to the error term
    if general_dict['distribution'] == "default":
        if general_dict['loss'] == "likelihood":
            general_dict['distribution_new'] = "dnorm" if model_type_dict['error_type'] == "A" else "dgamma"
        elif general_dict['loss'] in ["MAEh", "MACE", "MAE"]:
            general_dict['distribution_new'] = "dlaplace"
        elif general_dict['loss'] in ["HAMh", "CHAM", "HAM"]:
            general_dict['distribution_new'] = "ds"
        else:
            general_dict['distribution_new'] = "dnorm"
    else:
        general_dict['distribution_new'] = general_dict['distribution']
    # Print initial parameters if print_level is 41
    print_level_hidden = print_level
    if print_level == 1:
        #print("Initial parameters:", B)
        print_level = 0

    # Set maxeval based on parameters
    maxeval_used = maxeval
    if maxeval is None:
        maxeval_used = len(B) * 200
        
        # If xreg model, do more iterations
        if explanatory_dict['xreg_model']:
            maxeval_used = len(B) * 150
            maxeval_used = max(1500, maxeval_used)

    # Handle LASSO/RIDGE denominator calculation
    if general_dict['loss'] in ["LASSO", "RIDGE"]:
        if explanatory_dict['xreg_number'] > 0:
            # Calculate standard deviation for each column of matWt
            general_dict['denominator'] = np.std(adam_created['mat_wt'], axis=0)
            # Replace infinite values with 1
            general_dict['denominator'][np.isinf(general_dict['denominator'])] = 1
        else:
            general_dict['denominator'] = None
        # Calculate denominator for y values
        general_dict['y_denominator'] = max(np.std(np.diff(observations_dict['y_in_sample'])), 1)
    else:
        general_dict['denominator'] = None
        general_dict['y_denominator'] = None

    general_dict['multisteps'] = multisteps



    # Parameters are chosen to speed up the optimisation process and have decent accuracy
    #opts = {
    #    'algorithm': algorithm,
    #    'xtol_rel': xtol_rel,
    #    'xtol_abs': xtol_abs,
    #    'ftol_rel': ftol_rel,
        #   'ftol_abs': ftol_abs,
    #    'maxeval': maxeval_used,
    #    'maxtime': maxtime,
        #   'print_level': print_level
    #}
    # Create nlopt optimizer object
    opt = nlopt.opt(nlopt.LN_BOBYQA, len(B))  # Use BOBYQA algorithm which is better for this type of problem
    
    # Set bounds
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_xtol_rel(1e-6)  # Match R's tolerance
    opt.set_ftol_rel(1e-8)  # Match R's tolerance
    opt.set_ftol_abs(0)     # Match R's tolerance
    opt.set_xtol_abs(1e-8)  # Match R's tolerance

    # Increase maxeval to match or exceed R's value
    if maxeval is None:
        # Increase the default multiplier to ensure we run at least as many iterations as R
        maxeval_used = len(B) * 200  # Increased from 120 to 200
        
        # If xreg model, do more iterations
        if explanatory_dict['xreg_model']:
            maxeval_used = len(B) * 150  # Increased from 100 to 150
            maxeval_used = max(1500, maxeval_used)  # Increased from 1000 to 1500

    opt.set_maxeval(maxeval_used)

    # Remove the default timeout to allow the optimizer to run until maxeval is reached
    if maxtime is not None:
        opt.set_maxtime(maxtime)
    else:
        # Set a much longer timeout (30 minutes instead of 5)
        opt.set_maxtime(1800)  # 30 minutes default timeout


    iteration_count = [0]  
    # Define objective function wrapper since nlopt expects different signature
    def objective_wrapper(x, grad):
        """
        Wrapper for the objective function.
        """

        iteration_count[0] += 1
        
        # Calculate the cost function
        cf_value = CF(
            B = x,
            model_type_dict = model_type_dict,
            components_dict = components_dict,
            lags_dict = lags_dict,
            matrices_dict = adam_created,
            persistence_checked = persistence_dict,
            initials_checked = initials_dict,
            arima_checked = arima_dict,
            explanatory_checked = explanatory_dict,
            phi_dict = phi_dict,
            constants_checked = constant_dict,
            observations_dict = observations_dict,
            profile_dict = profile_dict,
            general = general_dict,
            bounds = "usual"
        )
        
        # Limit extreme values to prevent numerical instability
        if not np.isfinite(cf_value) or cf_value > 1e10:
            return 1e10
        
        #print(f"Iteration {iteration_count[0]}: Cost = {cf_value}")
       # print(f"Parameters: {x}")
        
        return cf_value
    
    # Set objective function
    opt.set_min_objective(objective_wrapper)
    
    # Print initial values before optimization
    #print(f"DEBUG - Starting optimization with initial parameters: {B}")
    #print(f"DEBUG - Lower bounds: {lb}")
    #print(f"DEBUG - Upper bounds: {ub}")
    
    try:
        # Run optimization
        x = opt.optimize(B)
        res_fun = opt.last_optimum_value()
        res = type('OptimizeResult', (), {
            'x': x,
            'fun': res_fun,
            'success': True
        })
        #print(f"Optimization completed after {iteration_count[0]} iterations")
        #print(f"Final parameters: {x}")
        #print(f"Final CF value: {res_fun}")
    except Exception as e:
        #print(f"Optimization failed after {iteration_count[0]} iterations: {str(e)}")  
        #print(f"Optimization failed: {str(e)}")
        res = type('OptimizeResult', (), {
            'x': B,
            'fun': 1e+300,
            'success': False
        })

    # Check the obtained parameters and the loss value and remove redundant parameters
    # Cases to consider:
    # 1. Some smoothing parameters are zero or one;
    # 2. The cost function value is -Inf (due to no variability in the sample);

    # Prepare the values to return
    B[:] = res.x
    CF_value = res.fun
    # A fix for the special case of LASSO/RIDGE with lambda==1
    if any(general_dict['loss'] == loss_type for loss_type in ["LASSO", "RIDGE"]) and general_dict['lambda_'] == 1:
        CF_value = 0
    n_param_estimated = len(B)


    # Return a proper logLik class equivalent
    log_lik_adam_value = log_Lik_ADAM(
        B,
        model_type_dict,
        components_dict,
        lags_dict,
        adam_created,
        persistence_dict,
        initials_dict,
        arima_dict,
        explanatory_dict,
        phi_dict,
        constant_dict,
        observations_dict,
        occurrence_dict,
        general_dict,
        profile_dict,
        multisteps = False
    )

    # In case of likelihood, we typically have one more parameter to estimate - scale.
    log_lik_adam_value = {
        'value': log_lik_adam_value,
        'nobs': observations_dict['obs_in_sample'],
        'df': n_param_estimated + (1 if general_dict['loss'] == "likelihood" else 0)
    }
    #print(f"DEBUG - Log likelihood value: {log_lik_adam_value}")

    # Here I will add regressors when I have olm
    # line 3032 - 3322

    return {
        'B': B,
        'CF_value': CF_value,
        'n_param_estimated': n_param_estimated,
        'log_lik_adam_value': log_lik_adam_value,
        
        # skiping the regressions for now
        # 'xreg_model': xreg_model,
        # 'xreg_data': xreg_data,
        # 'xreg_number': xreg_number,
        # 'xreg_names': xreg_names,
        # 'xreg_model_initials': xreg_model_initials,
        # 'formula': formula,
        # 'initial_xreg_estimate': initial_xreg_estimate,
        # 'persistence_xreg_estimate': persistence_xreg_estimate,
        # 'xreg_parameters_missing': xreg_parameters_missing,
        # 'xreg_parameters_included': xreg_parameters_included,
        # 'xreg_parameters_estimated': xreg_parameters_estimated,
        # 'xreg_parameters_persistence': xreg_parameters_persistence,
        'arima_polynomials': adam_created['arima_polynomials']
    }
    
import math

def selector(
        model_type_dict,
        phi_dict,
        general_dict, 
        lags_dict, 
        observations_dict, 
        arima_dict,
        constant_dict,
        explanatory_dict,
        occurrence_dict,
        components_dict,
        profiles_recent_table,
        profiles_recent_provided,
        persistence_results,
        initials_results,

        criterion = "AICc",
        silent = False
):
    """Creates a pool of models and selects the best of them"""
    
    

    # Note:
    # If we call the selector we need custom dictionairies to pass each time!
    # I need to find a way to pass it every time

    # Check if the pool was provided. In case of "no", form the big and the small ones
    if model_type_dict['models_pool'] is None:
        # The variable saying that the pool was not provided.
        if not silent:
            print("Forming the pool of models based on... ", end="")

        # Define the whole pool of errors
        if not model_type_dict['allow_multiplicative']:
            pool_errors = ["A"]
            pool_trends = ["N", "A", "Ad"] 
            pool_seasonals = ["N", "A"]
        else:
            pool_errors = ["A", "M"]
            pool_trends = ["N", "A", "Ad", "M", "Md"]
            pool_seasonals = ["N", "A", "M"]

        # Some preparation variables
        # If e_type is not Z, then check on additive errors
        if model_type_dict['error_type'] != "Z":
            pool_errors = pool_errors_small = model_type_dict['error_type']
        else:
            pool_errors_small = "A"

        # If t_type is not Z, then create a pool with specified type
        if model_type_dict['trend_type'] != "Z":
            if model_type_dict['trend_type'] == "X":
                pool_trends_small = ["N", "A"]
                pool_trends = ["N", "A", "Ad"]
                check_trend = True
            elif model_type_dict['trend_type'] == "Y":
                pool_trends_small = ["N", "M"]
                pool_trends = ["N", "M", "Md"]
                check_trend = True
            else:
                if model_type_dict['damped']:
                    pool_trends = pool_trends_small = [model_type_dict['trend_type'] + "d"]
                else:
                    pool_trends = pool_trends_small = [model_type_dict['trend_type']]
                check_trend = False
        else:
            pool_trends_small = ["N", "A"]
            check_trend = True

        # If s_type is not Z, then create specific pools
        if model_type_dict['season_type'] != "Z":
            if model_type_dict['season_type'] == "X":
                pool_seasonals = pool_seasonals_small = ["N", "A"]
                check_seasonal = True
            elif model_type_dict['season_type'] == "Y":
                pool_seasonals_small = ["N", "M"]
                pool_seasonals = ["N", "M"]
                check_seasonal = True
            else:
                pool_seasonals_small = [model_type_dict['season_type']]
                pool_seasonals = [model_type_dict['season_type']]
                check_seasonal = False
        else:
            pool_seasonals_small = ["N", "A", "M"]
            check_seasonal = True

        # If ZZZ, then the vector is: "ANN" "ANA" "ANM" "AAN" "AAA" "AAM"
        # Otherwise depends on the provided restrictions
        pool_small = []
        for error in pool_errors_small:
            for trend in pool_trends_small:
                for seasonal in pool_seasonals_small:
                    pool_small.append(error + trend + seasonal)

        # Align error and seasonality, if the error was not forced to be additive
        # The new pool: "ANN" "ANA" "MNM" "AAN" "AAA" "MAM"
        if any(model[2] == "M" for model in pool_small) and model_type_dict['error_type'] not in ["A", "X"]:
            for i, model in enumerate(pool_small):
                if model[2] == "M":
                    pool_small[i] = "M" + model[1:]

        models_tested = None
        model_current = None

        # Counter + checks for the components
        j = 1
        i = 0
        check = True
        best_i = best_j = 1
        results = [None] * len(pool_small)
        
        # Branch and bound is here
        while check:

            # here just update the values on the dictionaries
            # I think its going to temporary work

            i += 1
            model_current = pool_small[j-1]

            # create a copy of the model_type_dict and the phi_dict
            model_type_dict_temp = model_type_dict.copy()
            model_type_dict_temp['model'] = model_current
            phi_dict_temp = phi_dict.copy()

            # Replace the values on the dictionary
            model_type_dict_temp['error_type'] = model_current[0]
            model_type_dict_temp['trend_type'] = model_current[1]
            
            if len(model_current) == 4:
                phi_dict_temp['phi'] = 0.95
                phi_dict_temp['phi_estimate'] = True
                model_type_dict_temp['season_type'] = model_current[3]
            else:
                phi_dict_temp['phi'] = 1
                phi_dict_temp['phi_estimate'] = False
                model_type_dict_temp['season_type'] = model_current[2]

            #print('estimator', flush=True)
            results[i-1] = {}
            results[i-1]['adam_estimated'] = estimator(
                        general_dict= general_dict,
                        model_type_dict = model_type_dict_temp,
                        lags_dict = lags_dict,
                        observations_dict = observations_dict,
                        arima_dict=arima_dict,
                        constant_dict=constant_dict,
                        explanatory_dict=explanatory_dict,
                        profiles_recent_table= profiles_recent_table,
                        profiles_recent_provided= profiles_recent_provided,
                        persistence_dict=persistence_results,
                        initials_dict=initials_results,
                        occurrence_dict=occurrence_dict,
                        phi_dict=phi_dict,
                        components_dict=components_dict,
                    )

            # this need further itteration on how to return outputs
            results[i-1]["IC"] = ic_function(general_dict['ic'],loglik=results[i-1]['adam_estimated']["log_lik_adam_value"])
            results[i-1]['model_type_dict'] = model_type_dict_temp
            results[i-1]['phi_dict'] = phi_dict_temp
            results[i-1]['model'] = model_current

            if phi_dict_temp['phi_estimate']:
                results[i-1]['phi_dict']["phi"] = results[i-1]["B"].get("phi")
            else:
                results[i-1]['phi_dict']["phi"] = 1
                
            #results[i-1]['model'] = model_current

            if models_tested is None:
                models_tested = [model_current]
            else:
                models_tested.append(model_current)

            if j > 1:
                # If the first is better than the second, then choose first
                if results[best_i-1]["IC"] <= results[i-1]["IC"]:
                    # If Ttype is the same, then we check seasonality
                    if model_current[1] == pool_small[best_j-1][1]:
                        pool_seasonals = results[best_i-1]["model_type_dict"]["season_type"]
                        check_seasonal = False
                        j = [k+1 for k in range(len(pool_small)) 
                                if pool_small[k] != pool_small[best_j-1] and 
                                pool_small[k][-1] == pool_seasonals]
                    # Otherwise we checked trend
                    else:
                        pool_trends = results[best_j-1]["model_type_dict"]["trend_type"]
                        check_trend = False
                else:
                    # If the trend is the same
                    if model_current[1] == pool_small[best_i-1][1]:
                        pool_seasonals = [s for s in pool_seasonals if s != model_type_dict_temp['season_type']]
                        if len(pool_seasonals) > 1:
                            # Select another seasonal model, not from previous iteration and not current
                            best_j = j
                            best_i = i
                            j = 3
                        else:
                            best_j = j
                            best_i = i
                            # Move to checking the trend
                            j = [k+1 for k in range(len(pool_small)) 
                                    if pool_small[k][-1] == pool_seasonals[0] and
                                    pool_small[k][1] != model_current[1]]
                            check_seasonal = False
                    else:
                        pool_trends = [t for t in pool_trends if t != model_type_dict_temp['trend_type']]
                        best_i = i
                        best_j = j
                        check_trend = False

                if not any([check_trend, check_seasonal]):
                    check = False
            else:
                j = 2

            # If this is NULL, then this was a short pool and we checked everything
            if not j:
                j = len(pool_small)
            if j > len(pool_small):
                check = False
        
        # Prepare a bigger pool based on the small one
        model_type_dict['models_pool'] = list(set(
            models_tested + 
            [e + t + s for e in pool_errors 
                for t in pool_trends 
                for s in pool_seasonals]
        ))
        j = len(models_tested)

    else:
        j = 0
        results = [None] * len(model_type_dict['models_pool'])

    models_number = len(model_type_dict['models_pool'])
    # Run the full pool of models
    if not silent:
        print("Estimation progress:    ", end="")

    # Start loop of models
    while j < models_number:
        j += 1
        if not silent:
            if j == 1:
                print("\b", end="")
            print("\b" * (len(str(round((j-1)/models_number * 100))) + 1), end="")
            print(f"{round(j/models_number * 100)}%", end="")

        model_current = model_type_dict['models_pool'][j-1]
        #print(model_current)

        # temporary copies for estimations
        model_type_dict_temp = model_type_dict.copy()
        phi_dict_temp = phi_dict.copy()

        model_type_dict_temp['error_type'] = model_current[0]
        model_type_dict_temp['trend_type'] = model_current[1]
        if len(model_current) == 4:
            phi_dict_temp['phi'] = 0.95
            model_type_dict_temp['season_type'] = model_current[3]
            phi_dict_temp['phi_estimate'] = True
        else:
            phi_dict_temp['phi'] = 1
            model_type_dict_temp['season_type'] = model_current[2]
            phi_dict_temp['phi_estimate'] = False

        results[j-1] = {}
        #print(lags_dict)
        #print(model_type_dict_temp)
        results[j-1]['adam_estimated'] = estimator(
                    general_dict= general_dict,
                    model_type_dict = model_type_dict_temp,
                    lags_dict = lags_dict,
                    observations_dict = observations_dict,
                    arima_dict=arima_dict,
                    constant_dict=constant_dict,
                    explanatory_dict=explanatory_dict,
                    profiles_recent_table= profiles_recent_table,
                    profiles_recent_provided= profiles_recent_provided,
                    persistence_dict=persistence_results,
                    initials_dict=initials_results,
                    occurrence_dict=occurrence_dict,
                    phi_dict=phi_dict_temp,
                    components_dict=components_dict,
                )
        #print(results[j-1]['adam_estimated'])
        # this need further itteration on how to return outputs
        results[j-1]["IC"] = ic_function(general_dict['ic'],loglik=results[j-1]['adam_estimated']["log_lik_adam_value"])
        results[j-1]['model_type_dict'] = model_type_dict_temp
        results[j-1]['phi_dict'] = phi_dict_temp
        results[j-1]['model'] = model_current

        #if phi_dict_temp['phi_estimate']:
        #    results[j-1]['phi_dict']["phi"] = results[j-1]["B"].get("phi")
        #else:
        #    results[j-1]['phi_dict']["phi"] = 1


    if not silent:
        print("... Done!")

    # Extract ICs and find the best
    ic_selection = [None] * models_number
    for j in range(models_number):
        ic_selection[j] = results[j]["IC"]

    # Set names for ic_selection
    ic_selection_dict = dict(zip(model_type_dict['models_pool'], ic_selection))

    # Replace NaN values with large number
    ic_selection = [1e100 if math.isnan(x) else x for x in ic_selection]

    return {"results": results, "ic_selection": ic_selection_dict}

def preparator_old(
    
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
    #if general_dict.get("model_do") != "use":
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
    if phi_dict["phi_estimate"]:
        phi_dict["phi"] = adam_estimated['B'],[next(i for i,v in enumerate(B.keys()) if v=="phi")]

    # Write down the initials in the recent profile
    profiles_dict["profiles_recent_table"][:] = matrices_dict['mat_vt'][:, :lags_dict["lags_model_max"]]
    profiles_dict["profiles_recent_initial"] = matrices_dict['mat_vt'][:, :lags_dict["lags_model_max"]].copy()
    

    # Convert pandas Series/DataFrames to numpy arrays
    y_in_sample = np.asarray(observations_dict['y_in_sample'].values.flatten(), dtype=np.float64)
    ot = np.asarray(observations_dict['ot'].values.flatten(), dtype=np.float64)
    
    mat_vt = np.asfortranarray(matrices_dict['mat_vt'], dtype=np.float64)
    mat_wt = np.asfortranarray(matrices_dict['mat_wt'], dtype=np.float64)
    mat_f = np.asfortranarray(matrices_dict['mat_f'], dtype=np.float64)
    vec_g = np.asfortranarray(matrices_dict['vec_g'], dtype=np.float64) # Make sure it's a 1D array
    lags_model_all = np.asfortranarray(lags_dict['lags_model_all'], dtype=np.uint64).reshape(-1,1)  # Make sure it's a 1D array
    index_lookup_table = np.asfortranarray(profiles_dict['index_lookup_table'], dtype=np.uint64)
    profiles_recent_table = np.asfortranarray(profiles_dict['profiles_recent_table'], dtype=np.float64)


    # Print debug information
    # print('mat_vt:', mat_vt)
    # print('mat_wt:', mat_wt)
    # print('dim mat_wt:', mat_wt.shape)
    # print('mat_f:', mat_f)
    # print('vec_g:', vec_g)
    # print('lags_model_all:', lags_model_all)
    # print('index_lookup_table:', index_lookup_table)
    # print('profiles_recent_table:', profiles_recent_table)
    # print('error_type:', model_type_dict['error_type'])
    # print('trend_type:', model_type_dict['trend_type'])
    # print('season_type:', model_type_dict['season_type'])
    # print('components_number_ets:', components_dict['components_number_ets'])
    # print('components_number_ets_seasonal:', components_dict['components_number_ets_seasonal'])
    # print('components_number_arima:', components_dict['components_number_arima'])
    # print('xreg_number:', explanatory_checked['xreg_number'])
    # print('constant_required:', constants_checked['constant_required'])
    # print('y_in_sample:', y_in_sample)
    # print('ot:', ot)
    # print('double checking', y_in_sample)


    # Fit the model to the data
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
    #print(adam_fitted)

    #print('adam_fitted')
    #print(adam_fitted)
    matrices_dict['mat_vt'][:] = adam_fitted["matVt"]
    profiles_dict["profiles_recent_table"] = adam_fitted["profile"]

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

    # Check what was returned in the end
    if np.any(np.isnan(y_fitted)) or np.any(pd.isna(y_fitted)):
        warnings.warn("Something went wrong in the estimation of the model and NaNs were produced. "
                     "If this is a mixed model, consider using the pure ones instead.")

    if occurrence_dict["occurrence_model"]:
        y_fitted[:] = y_fitted * occurrence_dict["p_fitted"]

    # Fix the cases, when we have zeroes in the provided occurrence
    if occurrence_dict["occurrence"] == "provided":
        y_fitted[~occurrence_dict["ot_logical"]] = y_fitted[~occurrence_dict["ot_logical"]] * occurrence_dict["p_fitted"][~occurrence_dict["ot_logical"]]

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
        
        
        
        
        # print(observations_dict["obs_in_sample"])
        #index_lookup_table = np.asfortranarray(profiles_dict['index_lookup_table'], dtype=np.uint64)
        #profiles_recent_table = np.asfortranarray(profiles_dict['profiles_recent_table'], dtype=np.float64)
        # index_lookup_table = np.asfortranarray(
        #     profiles_dict['index_lookup_table'][:, lags_dict["lags_model_max"] + observations_dict["obs_in_sample"] + np.arange(general_dict["h"])],
        #     dtype=np.uint64
        # ).copy()
        # profiles_recent_table = np.asfortranarray(profiles_dict['profiles_recent_table'], dtype=np.float64).copy()

        # mat_wt = np.asfortranarray(matrices_dict['mat_wt'][-general_dict["h"]:].copy(), dtype=np.float64).copy()
        # mat_f = np.asfortranarray(matrices_dict['mat_f'], dtype=np.float64).copy()
        # lags_model_all = np.asfortranarray(lags_dict["lags_model_all"], dtype=np.uint64).reshape(-1,1).copy()
            
        # print('index_lookup_table', index_lookup_table)
        # print('profiles_recent_table', profiles_recent_table)
        # print('mat_wt', mat_wt, mat_wt.shape)
        # print('mat_f', mat_f)
        # print('lags_model_all', lags_model_all)
        # print('model_type_dict', model_type_dict)
        # print('check', flush=True)
        
        # y_forecast[:] = adam_forecaster(
        #      matrixWt=mat_wt,
        #      matrixF=mat_f,
        #     lags=lags_model_all,
        #     indexLookupTable=index_lookup_table,
        #     profilesRecent=profiles_recent_table,
        #     E=model_type_dict["error_type"],
        #     T=model_type_dict["trend_type"],
        #     S=model_type_dict["season_type"],
        #     nNonSeasonal=components_dict["components_number_ets"],
        #     nSeasonal=components_dict["components_number_ets_seasonal"],
        #     nArima=components_dict.get("components_number_arima", 0),
        #     nXreg=explanatory_checked["xreg_number"],
        #     constant=constants_checked["constant_required"],
        #     horizon=general_dict["h"]
        # ).flatten()
        
        
        y_forecast[:] = adam_forecaster(
            matrixWt=matrices_dict['mat_wt'][-general_dict["h"]:],
            matrixF=matrices_dict['mat_f'],
            lags=lags_dict["lags_model_all"],
            indexLookupTable=profiles_dict["index_lookup_table"],
            profilesRecent=profiles_dict["profiles_recent_table"],
            E=model_type_dict["error_type"],
            T=model_type_dict["trend_type"],
            S=model_type_dict["season_type"],
            nNonSeasonal=components_dict["components_number_ets"],
            nSeasonal=components_dict["components_number_ets_seasonal"],
            nArima=components_dict.get("components_number_arima", 0),
            nXreg=explanatory_checked["xreg_number"],
            constant=constants_checked["constant_required"],
            horizon=general_dict["h"]
        ).flatten()

        # Make safety checks
        # If there are NaN values
        if np.any(np.isnan(y_forecast)):
            y_forecast[np.isnan(y_forecast)] = 0

        # Amend forecasts, multiplying by probability

        # skiping for now we dont have the occurence yet
        # if occurrence_dict["occurrence_model"] and not occurrence_dict["occurrence_model_provided"]:
        #     y_forecast[:] = y_forecast * np.array(forecast(occurrence_dict["oes_model"], h=general_dict["horizon"]).mean)
        # elif (occurrence_dict["occurrence_model"] and occurrence_dict["occurrence_model_provided"]) or occurrence_dict["occurrence"] == "provided":
        #     y_forecast[:] = y_forecast * occurrence_dict["p_forecast"]

    else:
        if any(observations_dict.get("y_classes", []) == "ts"):
            y_forecast = pd.Series([np.nan], 
                                 index=pd.date_range(start=observations_dict["y_forecast_start"], 
                                                   periods=1, 
                                                   freq=observations_dict["y_frequency"]))
        else:
            y_forecast = pd.Series(np.full(general_dict["horizon"], np.nan), 
                                 index=observations_dict["y_forecast_index"])

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
                start_idx = lags_dict["lags_model_max"] - lags_dict["lags_model"][i][0]
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

    # Get persistence values
    persistence = np.array(matrices_dict['vec_g']).flatten()

    # I have no names for the matrix
    #persistence = {name: value for name, value in zip(matrices_dict['vec_g'].index, persistence)}

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

    # Amend the class of state matrix
    if not isinstance(observations_dict.get("y_in_sample"), pd.Series):
        mat_vt = pd.Series(matrices_dict['mat_vt'].T.flatten(),
                          index=pd.date_range(start=observations_dict["y_forecast_start"], 
                                            periods=len(matrices_dict['mat_vt'].T),
                                            freq=observations_dict["frequency"]))
    else:
        mat_vt = pd.Series(matrices_dict['mat_vt'].T,
                          index=observations_dict["y_forecast_index"])

    # Update parameters number
    # There is an issue here that I need to fix with the parameters number
    general_dict["parameters_number"][0][2] = np.sum(general_dict["parameters_number"][0][:2])
    
    
    return {
            "model": model_type_dict["model"],
            "time_elapsed": None, # here will count the time
            #"data": np.column_stack((None, explanatory_checked["xreg_data"])),
            "holdout": general_dict["holdout"],
            "fitted": y_fitted,
            "residuals": errors,
            "forecast": y_forecast,
            "states": mat_vt,
            "profile": profiles_dict["profiles_recent_table"],
            "profile_initial": profiles_dict["profiles_recent_initial"],
            "persistence": persistence,
            "phi": phi_dict["phi"],
            "transition": matrices_dict['mat_f'],
            "measurement": matrices_dict['mat_wt'],
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