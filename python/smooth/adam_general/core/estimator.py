import numpy as np
from adam_profile import architector
from python.smooth.adam_general.core.creator import creator
import nlopt
from python.smooth.adam_general.core.utils.ic import ic_function
import pandas as pd





def estimator(ets_model, e_type, t_type, s_type, lags, lags_model_seasonal, lags_model_arima,
              obs_states, obs_in_sample,
              y_in_sample, persistence, persistence_estimate,
              persistence_level, persistence_level_estimate,
              persistence_trend, persistence_trend_estimate,
              persistence_seasonal, persistence_seasonal_estimate,
              persistence_xreg, persistence_xreg_estimate, persistence_xreg_provided,
              phi, phi_estimate,
              initial_type, initial_level, initial_trend, initial_seasonal,
              initial_arima, initial_estimate,
              initial_level_estimate, initial_trend_estimate, initial_seasonal_estimate,
              initial_arima_estimate, initial_xreg_estimate, initial_xreg_provided,
              arima_model, ar_required, i_required, ma_required, arma_parameters,
              components_number_arima, components_names_arima,
              formula, xreg_model, xreg_model_initials, xreg_data, xreg_number, xreg_names, regressors,
              xreg_parameters_missing, xreg_parameters_included,
              xreg_parameters_estimated, xreg_parameters_persistence,
              constant_required, constant_estimate, constant_value, constant_name,
              ot, ot_logical, occurrence_model, p_fitted,
              bounds, loss, loss_function, distribution,
              horizon, multisteps, other, other_parameter_estimate, lambda_param):


    # Create the basic variables
    adam_architect = architector(
        ets_model, e_type, t_type, s_type, lags, lags_model_seasonal,
        xreg_number, obs_in_sample, initial_type,
        arima_model, lags_model_arima, xreg_model, constant_required,
        profiles_recent_table, profiles_recent_provided
    )

    # Create the matrices for the specific ETS model
    adam_created = creator(
        ets_model, e_type, t_type, s_type, model_is_trendy, model_is_seasonal,
        lags, lags_model, lags_model_arima, lags_model_all, lags_model_max,
        profiles_recent_table, profiles_recent_provided,
        obs_states, obs_in_sample, obs_all, components_number_ets, components_number_ets_seasonal,
        components_names_ets, ot_logical, y_in_sample,
        persistence, persistence_estimate,
        persistence_level, persistence_level_estimate, persistence_trend, persistence_trend_estimate,
        persistence_seasonal, persistence_seasonal_estimate,
        persistence_xreg, persistence_xreg_estimate, persistence_xreg_provided,
        phi,
        initial_type, initial_estimate,
        initial_level, initial_level_estimate, initial_trend, initial_trend_estimate,
        initial_seasonal, initial_seasonal_estimate,
        initial_arima, initial_arima_estimate, initial_arima_number,
        initial_xreg_estimate, initial_xreg_provided,
        arima_model, ar_required, i_required, ma_required, arma_parameters,
        ar_orders, i_orders, ma_orders,
        components_number_arima, components_names_arima,
        xreg_model, xreg_model_initials, xreg_data, xreg_number, xreg_names,
        xreg_parameters_persistence,
        constant_required, constant_estimate, constant_value, constant_name
    )

    # Initialize B
    b_values = initialiser(
        ets_model, e_type, t_type, s_type, model_is_trendy, model_is_seasonal,
        components_number_ets_non_seasonal, components_number_ets_seasonal, components_number_ets,
        lags, lags_model, lags_model_seasonal, lags_model_arima, lags_model_max,
        adam_created['mat_vt'],
        persistence_estimate, persistence_level_estimate, persistence_trend_estimate,
        persistence_seasonal_estimate, persistence_xreg_estimate,
        phi_estimate, initial_type, initial_estimate,
        initial_level_estimate, initial_trend_estimate, initial_seasonal_estimate,
        initial_arima_estimate, initial_xreg_estimate,
        arima_model, ar_required, ma_required, ar_estimate, ma_estimate, ar_orders, ma_orders,
        components_number_arima, components_names_arima, initial_arima_number,
        xreg_model, xreg_number,
        xreg_parameters_estimated, xreg_parameters_persistence,
        constant_estimate, constant_name, other_parameter_estimate
    )
    
    if B is not None:
        if isinstance(B, dict):
            B = {k: v for k, v in B.items() if k in b_values['B']}
            b_values['B'].update(B)
        else:
            b_values['B'][:] = B
            B = dict(zip(b_values['B'].keys(), B))

    # Continue with the rest of the function...
    # Preheat the initial state of ARIMA. Do this only for optimal initials and if B is not provided
    if arima_model and initial_type == "optimal" and initial_arima_estimate and B is None:
        adam_created_arima = filler(
            b_values['B'],
            ets_model, e_type, t_type, s_type, model_is_trendy, model_is_seasonal,
            components_number_ets, components_number_ets_non_seasonal,
            components_number_ets_seasonal, components_number_arima,
            lags, lags_model, lags_model_max,
            adam_created['mat_vt'], adam_created['mat_wt'], adam_created['mat_f'], adam_created['vec_g'],
            persistence_estimate, persistence_level_estimate, persistence_trend_estimate,
            persistence_seasonal_estimate, persistence_xreg_estimate,
            phi_estimate,
            initial_type, initial_estimate,
            initial_level_estimate, initial_trend_estimate, initial_seasonal_estimate,
            initial_arima_estimate, initial_xreg_estimate,
            arima_model, ar_estimate, ma_estimate, ar_orders, i_orders, ma_orders,
            ar_required, ma_required, arma_parameters,
            non_zero_ari, non_zero_ma, adam_created['arima_polynomials'],
            xreg_model, xreg_number,
            xreg_parameters_missing, xreg_parameters_included,
            xreg_parameters_estimated, xreg_parameters_persistence, constant_estimate
        )

        # Write down the initials in the recent profile
        profiles_recent_table[:] = adam_created_arima['mat_vt'][:, :lags_model_max]

        # Do initial fit to get the state values from the backcasting
        adam_fitted = adam_fitter_wrap(
            adam_created_arima['mat_vt'], adam_created_arima['mat_wt'], adam_created_arima['mat_f'], adam_created_arima['vec_g'],
            lags_model_all, index_lookup_table, profiles_recent_table,
            e_type, t_type, s_type, components_number_ets, components_number_ets_seasonal,
            components_number_arima, xreg_number, constant_required,
            y_in_sample, ot, True
        )

        adam_created['mat_vt'][:, :lags_model_max] = adam_fitted['mat_vt'][:, :lags_model_max]
        # Produce new initials
        b_values_new = initialiser(
            ets_model, e_type, t_type, s_type, model_is_trendy, model_is_seasonal,
            components_number_ets_non_seasonal, components_number_ets_seasonal, components_number_ets,
            lags, lags_model, lags_model_seasonal, lags_model_arima, lags_model_max,
            adam_created['mat_vt'],
            persistence_estimate, persistence_level_estimate, persistence_trend_estimate,
            persistence_seasonal_estimate, persistence_xreg_estimate,
            phi_estimate, initial_type, initial_estimate,
            initial_level_estimate, initial_trend_estimate, initial_seasonal_estimate,
            initial_arima_estimate, initial_xreg_estimate,
            arima_model, ar_required, ma_required, ar_estimate, ma_estimate, ar_orders, ma_orders,
            components_number_arima, components_names_arima, initial_arima_number,
            xreg_model, xreg_number,
            xreg_parameters_estimated, xreg_parameters_persistence,
            constant_estimate, constant_name, other_parameter_estimate
        )
        B = b_values_new['B']
        # Failsafe, just in case if the initial values contain NA / NaN
        B[np.isnan(B)] = b_values['B'][np.isnan(B)]



        # Fix for mixed ETS models producing negative values
        if (e_type == "M" and any(t in ["A", "Ad"] for t in [t_type, s_type]) or
            t_type == "M" and any(t in ["A", "Ad"] for t in [e_type, s_type]) or
            s_type == "M" and any(t in ["A", "Ad"] for t in [e_type, t_type])):
            if e_type == "M" and ("level" in B) and (B["level"] <= 0):
                B["level"] = y_in_sample[0]
            if t_type == "M" and ("trend" in B) and (B["trend"] <= 0):
                B["trend"] = 1
            seasonal_params = [p for p in B.keys() if p.startswith("seasonal")]
            if s_type == "M" and any(B[p] <= 0 for p in seasonal_params):
                for p in seasonal_params:
                    if B[p] <= 0:
                        B[p] = 1

        # Create the vector of initials for the optimisation
        if B is None:
            B = b_values['B']
        if lb is None:
            lb = b_values['Bl']
        if ub is None:
            ub = b_values['Bu']

        # Companion matrices for the polynomials calculation -> stationarity / stability checks
        if arima_model:
            # AR polynomials
            ar_polynomial_matrix = np.zeros((np.sum(ar_orders) * lags, np.sum(ar_orders) * lags))
            if ar_polynomial_matrix.shape[0] > 1:
                ar_polynomial_matrix[1:, :-1] = np.eye(ar_polynomial_matrix.shape[0] - 1)
            # MA polynomials
            ma_polynomial_matrix = np.zeros((np.sum(ma_orders) * lags, np.sum(ma_orders) * lags))
            if ma_polynomial_matrix.shape[0] > 1:
                ma_polynomial_matrix[1:, :-1] = np.eye(ma_polynomial_matrix.shape[0] - 1)
        else:
            ma_polynomial_matrix = ar_polynomial_matrix = None

        # If the distribution is default, change it according to the error term
        if distribution == "default":
            if loss == "likelihood":
                distribution_new = "dnorm" if e_type == "A" else "dgamma"
            elif loss in ["MAEh", "MACE", "MAE"]:
                distribution_new = "dlaplace"
            elif loss in ["HAMh", "CHAM", "HAM"]:
                distribution_new = "ds"
            else:
                distribution_new = "dnorm"
        else:
            distribution_new = distribution


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
        opt = nlopt.opt(nlopt.LD_SLSQP, len(B))  # Use SLSQP algorithm to match R code
        
        # Set bounds
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        
        # Set stopping criteria
        opt.set_ftol_rel(ftol_rel)
        opt.set_xtol_rel(xtol_rel)
        opt.set_xtol_abs(xtol_abs)
        opt.set_ftol_abs(ftol_abs)
        opt.set_maxeval(maxeval_used)
        if maxtime is not None:
            opt.set_maxtime(maxtime)
        
        # Define objective function wrapper since nlopt expects different signature
        def objective_wrapper(x, grad):
            return CF(x,
                ets_model, e_type, t_type, s_type, model_is_trendy, model_is_seasonal, y_in_sample,
                ot, ot_logical, occurrence_model, obs_in_sample,
                components_number_ets, components_number_ets_seasonal, components_number_ets_non_seasonal,
                components_number_arima,
                lags, lags_model, lags_model_all, lags_model_max,
                index_lookup_table, profiles_recent_table,
                adam_created['mat_vt'], adam_created['mat_wt'], adam_created['mat_f'], adam_created['vec_g'],
                persistence_estimate, persistence_level_estimate, persistence_trend_estimate,
                persistence_seasonal_estimate, persistence_xreg_estimate,
                phi_estimate, initial_type, initial_estimate, initial_level_estimate,
                initial_trend_estimate, initial_seasonal_estimate,
                initial_arima_estimate, initial_xreg_estimate,
                arima_model, non_zero_ari, non_zero_ma, adam_created['arima_polynomials'],
                ar_estimate, ma_estimate,
                ar_orders, i_orders, ma_orders,
                ar_required, ma_required, arma_parameters,
                xreg_model, xreg_number,
                xreg_parameters_missing, xreg_parameters_included,
                xreg_parameters_estimated, xreg_parameters_persistence,
                constant_required, constant_estimate,
                bounds, loss, loss_function, distribution_new,
                horizon, multisteps,
                denominator, y_denominator,
                other, other_parameter_estimate, lambda_,
                ar_polynomial_matrix, ma_polynomial_matrix)
        
        # Set objective function
        opt.set_min_objective(objective_wrapper)
        
        try:
            # Run optimization
            x = opt.optimize(B)
            res_fun = opt.last_optimum_value()
            res = type('OptimizeResult', (), {
                'x': x,
                'fun': res_fun,
                'success': True
            })
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            res = type('OptimizeResult', (), {
                'x': B,
                'fun': 1e+300,
                'success': False
            })

        # If optimization failed, try again with modified initial values
        if np.isinf(res.fun) or res.fun == 1e+300:
            # Reset initial values
            if ets_model:
                B[:components_number_ets] = 0
            if arima_model:
                start_idx = components_number_ets + persistence_xreg_estimate * xreg_number
                end_idx = start_idx + sum(np.array(ar_orders) * ar_estimate + np.array(ma_orders) * ma_estimate)
                B[start_idx:end_idx] = 0.01
                
            try:
                # Try optimization again
                x = opt.optimize(B)
                res_fun = opt.last_optimum_value()
                res = type('OptimizeResult', (), {
                    'x': x,
                    'fun': res_fun,
                    'success': True
                })
            except Exception as e:
                print(f"Second optimization attempt failed: {str(e)}")
                res = type('OptimizeResult', (), {
                    'x': B,
                    'fun': 1e+300,
                    'success': False
                })

        if print_level_hidden > 0:
            print(res)

        # Check the obtained parameters and the loss value and remove redundant parameters
        # Cases to consider:
        # 1. Some smoothing parameters are zero or one;
        # 2. The cost function value is -Inf (due to no variability in the sample);

        # Prepare the values to return
        B[:] = res.x
        CF_value = res.fun
        # A fix for the special case of LASSO/RIDGE with lambda==1
        if any(loss == loss_type for loss_type in ["LASSO", "RIDGE"]) and lambda_ == 1:
            CF_value = 0
        n_param_estimated = len(B)

    
        # Return a proper logLik class equivalent
        logLikADAMValue = logLikReturn(
            B,
            ets_model, e_type, t_type, s_type, model_is_trendy, model_is_seasonal, y_in_sample,
            ot, ot_logical, occurrence_model, p_fitted, obs_in_sample,
            components_number_ets, components_number_ets_seasonal, components_number_ets_non_seasonal,
            components_number_arima,
            lags, lags_model, lags_model_all, lags_model_max,
            index_lookup_table, profiles_recent_table,
            adam_created['mat_vt'], adam_created['mat_wt'], adam_created['mat_f'], adam_created['vec_g'],
            persistence_estimate, persistence_level_estimate, persistence_trend_estimate,
            persistence_seasonal_estimate, persistence_xreg_estimate,
            phi_estimate, initial_type, initial_estimate,
            initial_level_estimate, initial_trend_estimate, initial_seasonal_estimate,
            initial_arima_estimate, initial_xreg_estimate,
            arima_model, non_zero_ari, non_zero_ma, ar_estimate, ma_estimate,
            adam_created['arima_polynomials'],
            ar_orders, i_orders, ma_orders, ar_required, ma_required, arma_parameters,
            xreg_model, xreg_number,
            xreg_parameters_missing, xreg_parameters_included,
            xreg_parameters_estimated, xreg_parameters_persistence,
            constant_required, constant_estimate,
            bounds, loss, loss_function, distribution_new, horizon, multisteps,
            denominator, y_denominator, other, other_parameter_estimate, lambda_,
            ar_polynomial_matrix, ma_polynomial_matrix
        )

        # In case of likelihood, we typically have one more parameter to estimate - scale.
        logLikADAMValue = {
            'value': logLikADAMValue,
            'nobs': obs_in_sample,
            'df': n_param_estimated + (1 if loss == "likelihood" else 0)
        }

        # Here I will add regressors when I have olm
        # line 3032 - 3322

        return {
            'B': B,
            'CF_value': CF_value,
            'n_param_estimated': n_param_estimated,
            'logLikADAMValue': logLikADAMValue,
            'xreg_model': xreg_model,
            'xreg_data': xreg_data,
            'xreg_number': xreg_number,
            'xreg_names': xreg_names,
            'xreg_model_initials': xreg_model_initials,
            'formula': formula,
            'initial_xreg_estimate': initial_xreg_estimate,
            'persistence_xreg_estimate': persistence_xreg_estimate,
            'xreg_parameters_missing': xreg_parameters_missing,
            'xreg_parameters_included': xreg_parameters_included,
            'xreg_parameters_estimated': xreg_parameters_estimated,
            'xreg_parameters_persistence': xreg_parameters_persistence,
            'arima_polynomials': adam_created['arima_polynomials']
        }
    

def selector(model, models_pool, allow_multiplicative,
             ets_model, e_type, t_type, s_type, damped, lags,
             lags_model_seasonal, lags_model_arima,
             obs_states, obs_in_sample,
             y_in_sample, persistence, persistence_estimate,
             persistence_level, persistence_level_estimate,
             persistence_trend, persistence_trend_estimate,
             persistence_seasonal, persistence_seasonal_estimate,
             persistence_xreg, persistence_xreg_estimate, persistence_xreg_provided,
             phi, phi_estimate,
             initial_type, initial_level, initial_trend, initial_seasonal,
             initial_arima, initial_estimate,
             initial_level_estimate, initial_trend_estimate, initial_seasonal_estimate,
             initial_arima_estimate, initial_xreg_estimate, initial_xreg_provided,
             arima_model, ar_required, i_required, ma_required, arma_parameters,
             components_number_arima, components_names_arima,
             xreg_model, xreg_model_initials, xreg_data, xreg_number, xreg_names, regressors,
             xreg_parameters_missing, xreg_parameters_included,
             xreg_parameters_estimated, xreg_parameters_persistence,
             constant_required, constant_estimate, constant_value, constant_name,
             ot, ot_logical, occurrence_model, p_fitted, ic_function,
             bounds, loss, loss_function, distribution,
             horizon, multisteps, other, other_parameter_estimate, lambda_):
    """Creates a pool of models and selects the best of them"""
    

    # Check if the pool was provided. In case of "no", form the big and the small ones
    if models_pool is None:
        # The variable saying that the pool was not provided.
        if not silent:
            print("Forming the pool of models based on... ", end="")

        # Define the whole pool of errors
        if not allow_multiplicative:
            pool_errors = ["A"]
            pool_trends = ["N", "A", "Ad"] 
            pool_seasonals = ["N", "A"]
        else:
            pool_errors = ["A", "M"]
            pool_trends = ["N", "A", "Ad", "M", "Md"]
            pool_seasonals = ["N", "A", "M"]

        # Some preparation variables
        # If e_type is not Z, then check on additive errors
        if e_type != "Z":
            pool_errors = pool_errors_small = e_type
        else:
            pool_errors_small = "A"

        # If t_type is not Z, then create a pool with specified type
        if t_type != "Z":
            if t_type == "X":
                pool_trends_small = ["N", "A"]
                pool_trends = ["N", "A", "Ad"]
                check_trend = True
            elif t_type == "Y":
                pool_trends_small = ["N", "M"]
                pool_trends = ["N", "M", "Md"]
                check_trend = True
            else:
                if damped:
                    pool_trends = pool_trends_small = [t_type + "d"]
                else:
                    pool_trends = pool_trends_small = [t_type]
                check_trend = False
        else:
            pool_trends_small = ["N", "A"]
            check_trend = True

        # If s_type is not Z, then create specific pools
        if s_type != "Z":
            if s_type == "X":
                pool_seasonals = pool_seasonals_small = ["N", "A"]
                check_seasonal = True
            elif s_type == "Y":
                pool_seasonals_small = ["N", "M"]
                pool_seasonals = ["N", "M"]
                check_seasonal = True
            else:
                pool_seasonals_small = [s_type]
                pool_seasonals = [s_type]
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
        if any(model[2] == "M" for model in pool_small) and e_type not in ["A", "X"]:
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
            i += 1
            model_current = pool_small[j-1]
            
            e_type = model_current[0]
            t_type = model_current[1]
            
            if len(model_current) == 4:
                phi = 0.95
                phi_estimate = True
                s_type = model_current[3]
            else:
                phi = 1
                phi_estimate = False
                s_type = model_current[2]

            results[i-1] = estimator(
                ets_model, e_type, t_type, s_type, lags, lags_model_seasonal, lags_model_arima,
                obs_states, obs_in_sample,
                y_in_sample, persistence, persistence_estimate,
                persistence_level, persistence_level_estimate,
                persistence_trend, persistence_trend_estimate,
                persistence_seasonal, persistence_seasonal_estimate,
                persistence_xreg, persistence_xreg_estimate, persistence_xreg_provided,
                phi, phi_estimate,
                initial_type, initial_level, initial_trend, initial_seasonal,
                initial_arima, initial_estimate,
                initial_level_estimate, initial_trend_estimate, initial_seasonal_estimate,
                initial_arima_estimate, initial_xreg_estimate, initial_xreg_provided,
                arima_model, ar_required, i_required, ma_required, arma_parameters,
                components_number_arima, components_names_arima,
                formula, xreg_model, xreg_model_initials, xreg_data, xreg_number, xreg_names, regressors,
                xreg_parameters_missing, xreg_parameters_included,
                xreg_parameters_estimated, xreg_parameters_persistence,
                constant_required, constant_estimate, constant_value, constant_name,
                ot, ot_logical, occurrence_model, p_fitted,
                bounds, loss, loss_function, distribution,
                horizon, multisteps, other, other_parameter_estimate, lambda_param
            )

            results[i-1]["IC"] = ic_function(results[i-1]["logLikADAMValue"])
            results[i-1]["Etype"] = e_type
            results[i-1]["Ttype"] = t_type
            results[i-1]["Stype"] = s_type
            results[i-1]["phiEstimate"] = phi_estimate
            
            if phi_estimate:
                results[i-1]["phi"] = results[i-1]["B"].get("phi")
            else:
                results[i-1]["phi"] = 1
                
            results[i-1]["model"] = model_current

            if models_tested is None:
                models_tested = [model_current]
            else:
                models_tested.append(model_current)

            if j > 1:
                # If the first is better than the second, then choose first
                if results[best_i-1]["IC"] <= results[i-1]["IC"]:
                    # If Ttype is the same, then we check seasonality
                    if model_current[1] == pool_small[best_j-1][1]:
                        pool_seasonals = results[best_i-1]["Stype"]
                        check_seasonal = False
                        j = [k+1 for k in range(len(pool_small)) 
                                if pool_small[k] != pool_small[best_j-1] and 
                                pool_small[k][-1] == pool_seasonals]
                    # Otherwise we checked trend
                    else:
                        pool_trends = results[best_j-1]["Ttype"]
                        check_trend = False
                else:
                    # If the trend is the same
                    if model_current[1] == pool_small[best_i-1][1]:
                        pool_seasonals = [s for s in pool_seasonals if s != results[best_i-1]["Stype"]]
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
                        pool_trends = [t for t in pool_trends if t != results[best_j-1]["Ttype"]]
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
        models_pool = list(set(
            models_tested + 
            [e + t + s for e in pool_errors 
                for t in pool_trends 
                for s in pool_seasonals]
        ))
        j = len(models_tested)

    else:
        j = 0
        results = [None] * len(models_pool)

    models_number = len(models_pool)

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

        model_current = models_pool[j-1]
        # print(model_current)
        e_type = model_current[0]
        t_type = model_current[1]
        if len(model_current) == 4:
            phi = 0.95
            s_type = model_current[3]
            phi_estimate = True
        else:
            phi = 1
            s_type = model_current[2]
            phi_estimate = False


        results[j-1] = estimator(
            ets_model, e_type, t_type, s_type, lags, lags_model_seasonal, lags_model_arima,
            obs_states, obs_in_sample,
            y_in_sample, persistence, persistence_estimate,
            persistence_level, persistence_level_estimate,
            persistence_trend, persistence_trend_estimate,
            persistence_seasonal, persistence_seasonal_estimate,
            persistence_xreg, persistence_xreg_estimate, persistence_xreg_provided,
            phi, phi_estimate,
            initial_type, initial_level, initial_trend, initial_seasonal,
            initial_arima, initial_estimate,
            initial_level_estimate, initial_trend_estimate, initial_seasonal_estimate,
            initial_arima_estimate, initial_xreg_estimate, initial_xreg_provided,
            arima_model, ar_required, i_required, ma_required, arma_parameters,
            components_number_arima, components_names_arima,
            formula, xreg_model, xreg_model_initials, xreg_data, xreg_number, xreg_names, regressors,
            xreg_parameters_missing, xreg_parameters_included,
            xreg_parameters_estimated, xreg_parameters_persistence,
            constant_required, constant_estimate, constant_value, constant_name,
            ot, ot_logical, occurrence_model, p_fitted,
            bounds, loss, loss_function, distribution,
            horizon, multisteps, other, other_parameter_estimate, lambda_)

        results[j-1]["IC"] = ic_function(results[j-1]["logLikADAMValue"])
        results[j-1]["Etype"] = e_type
        results[j-1]["Ttype"] = t_type
        results[j-1]["Stype"] = s_type
        results[j-1]["phiEstimate"] = phi_estimate
        if phi_estimate:
            results[j-1]["phi"] = results[j-1]["B"][next(i for i,v in enumerate(results[j-1]["B"].keys()) if v=="phi")]
        else:
            results[j-1]["phi"] = 1
        results[j-1]["model"] = model_current


    if not silent:
        print("... Done!")

    # Extract ICs and find the best
    ic_selection = [None] * models_number
    for i in range(models_number):
        ic_selection[i] = results[i]["IC"]

    # Set names for ic_selection
    ic_selection_dict = dict(zip(models_pool, ic_selection))

    # Replace NaN values with large number
    ic_selection = [1e100 if math.isnan(x) else x for x in ic_selection]

    return {"results": results, "icSelection": ic_selection_dict}


def preparator(B, ets_model, e_type, t_type, s_type,
               lags_model, lags_model_max, lags_model_all,
               components_number_ets, components_number_ets_seasonal,
               xreg_number, distribution, loss,
               persistence_estimate, persistence_level_estimate, persistence_trend_estimate,
               persistence_seasonal_estimate, persistence_xreg_estimate,
               phi_estimate, other_parameter_estimate,
               initial_type, initial_estimate,
               initial_level_estimate, initial_trend_estimate, initial_seasonal_estimate,
               initial_arima_estimate, initial_xreg_estimate,
               mat_vt, mat_wt, mat_f, vec_g,
               occurrence_model, ot, oes_model,
               parameters_number, cf_value,
               arima_model, ar_required, ma_required,
               ar_estimate, ma_estimate, ar_orders, i_orders, ma_orders,
               non_zero_ari, non_zero_ma,
               arima_polynomials, arma_parameters,
               constant_required, constant_estimate):
    """Function prepares all the matrices and vectors for return"""
    

    if model_do != "use":
        # Fill in the matrices
        adam_elements = filler(
            B,
            ets_model, e_type, t_type, s_type, model_is_trendy, model_is_seasonal,
            components_number_ets, components_number_ets_non_seasonal,
            components_number_ets_seasonal, components_number_arima,
            lags, lags_model, lags_model_max,
            mat_vt, mat_wt, mat_f, vec_g,
            persistence_estimate, persistence_level_estimate, persistence_trend_estimate,
            persistence_seasonal_estimate, persistence_xreg_estimate,
            phi_estimate,
            initial_type, initial_estimate,
            initial_level_estimate, initial_trend_estimate, initial_seasonal_estimate,
            initial_arima_estimate, initial_xreg_estimate,
            arima_model, ar_estimate, ma_estimate, ar_orders, i_orders, ma_orders,
            ar_required, ma_required, arma_parameters,
            non_zero_ari, non_zero_ma, arima_polynomials,
            xreg_model, xreg_number,
            xreg_parameters_missing, xreg_parameters_included,
            xreg_parameters_estimated, xreg_parameters_persistence, constant_estimate
        )


    # Write down phi
    if phi_estimate:
        phi[:] = B[next(i for i,v in enumerate(B.keys()) if v=="phi")]

    # Write down the initials in the recent profile
    profiles_recent_table[:] = mat_vt[:, :lags_model_max]
    profiles_recent_initial = mat_vt[:, :lags_model_max].copy()

    # Fit the model to the data
    adam_fitted = adam_fitter_wrap(
        mat_vt, mat_wt, mat_f, vec_g,
        lags_model_all, index_lookup_table, profiles_recent_table,
        e_type, t_type, s_type, components_number_ets, components_number_ets_seasonal,
        components_number_arima, xreg_number, constant_required,
        y_in_sample, ot, any(x in initial_type for x in ["complete", "backcasting"])
    )

    mat_vt[:] = adam_fitted["mat_vt"]

    # Write down the recent profile for future use
    profiles_recent_table = adam_fitted["profile"]


    # Make sure that there are no negative values in multiplicative components
    # This might appear in case of bounds="a"
    if t_type == "M" and (np.any(np.isnan(mat_vt[1,:])) or np.any(mat_vt[1,:] <= 0)):
        i = np.where(mat_vt[1,:] <= 0)[0]
        mat_vt[1,i] = 1e-6
        profiles_recent_table[1,i] = 1e-6

    if s_type == "M" and np.all(~np.isnan(mat_vt[components_number_ets_non_seasonal:components_number_ets_non_seasonal+components_number_ets_seasonal,:])) and \
        np.any(mat_vt[components_number_ets_non_seasonal:components_number_ets_non_seasonal+components_number_ets_seasonal,:] <= 0):
        i = np.where(mat_vt[components_number_ets_non_seasonal:components_number_ets_non_seasonal+components_number_ets_seasonal,:] <= 0)[0]
        mat_vt[components_number_ets_non_seasonal:components_number_ets_non_seasonal+components_number_ets_seasonal,i] = 1e-6
        i = np.where(profiles_recent_table[components_number_ets_non_seasonal:components_number_ets_non_seasonal+components_number_ets_seasonal,:] <= 0)[0]
        profiles_recent_table[components_number_ets_non_seasonal:components_number_ets_non_seasonal+components_number_ets_seasonal,i] = 1e-6

    # Prepare fitted and error with ts / zoo
    if any(y_classes == "ts"):
        y_fitted = pd.Series(np.full(obs_in_sample, np.nan), index=pd.date_range(start=y_start, periods=obs_in_sample, freq=y_frequency))
        errors = pd.Series(np.full(obs_in_sample, np.nan), index=pd.date_range(start=y_start, periods=obs_in_sample, freq=y_frequency))
    else:
        y_fitted = pd.Series(np.full(obs_in_sample, np.nan), index=y_in_sample_index)
        errors = pd.Series(np.full(obs_in_sample, np.nan), index=y_in_sample_index)

    errors[:] = adam_fitted["errors"]
    y_fitted[:] = adam_fitted["y_fitted"]
    # Check what was returned in the end
    if np.any(np.isnan(y_fitted)) or np.any(pd.isna(y_fitted)):
        warnings.warn("Something went wrong in the estimation of the model and NaNs were produced. "
                     "If this is a mixed model, consider using the pure ones instead.")

    if occurrence_model:
        y_fitted[:] = y_fitted * p_fitted

    # Fix the cases, when we have zeroes in the provided occurrence
    if occurrence == "provided":
        y_fitted[~ot_logical] = y_fitted[~ot_logical] * p_fitted[~ot_logical]

    # Produce forecasts if the horizon is non-zero
    if horizon > 0:
        if any(y_classes == "ts"):
            y_forecast = pd.Series(np.full(horizon, np.nan), 
                                    index=pd.date_range(start=y_forecast_start, periods=horizon, freq=y_frequency))
        else:
            y_forecast = pd.Series(np.full(horizon, np.nan), index=y_forecast_index)

        y_forecast[:] = adam_forecaster_wrap(
            mat_wt[-horizon:], mat_f,
            lags_model_all,
            index_lookup_table[:, lags_model_max + obs_in_sample + np.arange(horizon)],
            profiles_recent_table,
            e_type, t_type, s_type,
            components_number_ets, components_number_ets_seasonal,
            components_number_arima, xreg_number, constant_required,
            horizon
        )

        # Make safety checks
        # If there are NaN values
        if np.any(np.isnan(y_forecast)):
            y_forecast[np.isnan(y_forecast)] = 0

        # Amend forecasts, multiplying by probability
        if occurrence_model and not occurrence_model_provided:
            y_forecast[:] = y_forecast * np.array(forecast(oes_model, h=h).mean)
        elif (occurrence_model and occurrence_model_provided) or occurrence == "provided":
            y_forecast[:] = y_forecast * p_forecast

    else:
        if any(y_classes == "ts"):
            y_forecast = pd.Series([np.nan], index=pd.date_range(start=y_forecast_start, periods=1, freq=y_frequency))
        else:
            y_forecast = pd.Series(np.full(horizon, np.nan), index=y_forecast_index)

    # If the distribution is default, change it according to the error term
    if distribution == "default":
        if loss == "likelihood":
            if e_type == "A":
                distribution = "dnorm"
            elif e_type == "M":
                distribution = "dgamma"
        elif loss in ["MAEh", "MACE", "MAE"]:
            distribution = "dlaplace"
        elif loss in ["HAMh", "CHAM", "HAM"]:
            distribution = "ds"
        elif loss in ["MSEh", "MSCE", "MSE", "GPL"]:
            distribution = "dnorm"
        else:
            distribution = "dnorm"

    # Initial values to return
    initial_value = [None] * (ets_model * (1 + model_is_trendy + model_is_seasonal) + arima_model + xreg_model)
    initial_value_ets = [None] * (ets_model * len(lags_model))
    initial_value_names = [""] * (ets_model * (1 + model_is_trendy + model_is_seasonal) + arima_model + xreg_model)
    # The vector that defines what was estimated in the model
    initial_estimated = [False] * (ets_model * (1 + model_is_trendy + model_is_seasonal * components_number_ets_seasonal) + 
                                arima_model + xreg_model)
    
    # Write down the initials of ETS
    j = 0
    if ets_model:
        # Write down level, trend and seasonal
        for i in range(len(lags_model)):
            # In case of level / trend, we want to get the very first value
            if lags_model[i] == 1:
                initial_value_ets[i] = mat_vt[i, :lags_model_max][0]
            # In cases of seasonal components, they should be at the end of the pre-heat period
            else:
                initial_value_ets[i] = mat_vt[i, :lags_model_max][-lags_model[i]:]
        
        j = 0
        # Write down level in the final list
        initial_estimated[j] = initial_level_estimate
        initial_value[j] = initial_value_ets[j]
        initial_value_names[j] = "level"
        
        if model_is_trendy:
            j = 1
            initial_estimated[j] = initial_trend_estimate
            # Write down trend in the final list
            initial_value[j] = initial_value_ets[j]
            # Remove the trend from ETS list
            initial_value_ets[j] = None
            initial_value_names[j] = "trend"
        
        # Write down the initial seasonals
        if model_is_seasonal:
            initial_estimated[j + 1:j + 1 + components_number_ets_seasonal] = initial_seasonal_estimate
            # Remove the level from ETS list
            initial_value_ets[0] = None
            j += 1
            if len(initial_seasonal_estimate) > 1:
                initial_value[j] = [x for x in initial_value_ets if x is not None]
                initial_value_names[j] = "seasonal"
                for k in range(components_number_ets_seasonal):
                    initial_estimated[j + k] = f"seasonal{k+1}"
            else:
                initial_value[j] = next(x for x in initial_value_ets if x is not None)
                initial_value_names[j] = "seasonal"
                initial_estimated[j] = "seasonal"

    # Write down the ARIMA initials
    if arima_model:
        j += 1
        initial_estimated[j] = initial_arima_estimate
        if initial_arima_estimate:
            initial_value[j] = mat_vt[components_number_ets + components_number_arima - 1, :initial_arima_number]
        else:
            initial_value[j] = initial_arima
        initial_value_names[j] = "arima"
        initial_estimated[j] = "arima"


    # Set names for initial values
    initial_value = {name: value for name, value in zip(initial_value_names, initial_value)}

    # Get persistence values
    persistence = np.array(vec_g).flatten()
    persistence = {name: value for name, value in zip(vec_g.index, persistence)}

    # Remove xreg persistence if needed
    if xreg_model and regressors != "adapt":
        regressors = "use"
    elif not xreg_model:
        regressors = None

    # Handle ARMA parameters
    if arima_model:
        arma_parameters_list = {}
        j = 0
        if ar_required and phi_estimate:
            # Avoid damping parameter phi by checking name length > 3
            arma_parameters_list["ar"] = [b for name, b in B.items() if len(name) > 3 and name.startswith("phi")]
            j += 1
        elif ar_required and not phi_estimate:
            # Avoid damping parameter phi
            arma_parameters_list["ar"] = [p for name, p in arma_parameters.items() if name.startswith("phi")]
            j += 1
        
        if ma_required and ma_estimate:
            arma_parameters_list["ma"] = [b for name, b in B.items() if name.startswith("theta")]
        elif ma_required and not ma_estimate:
            arma_parameters_list["ma"] = [p for name, p in arma_parameters.items() if name.startswith("theta")]
    else:
        arma_parameters_list = None


    # Handle distribution parameters
    if distribution in ["dalaplace", "dgnorm", "dlgnorm", "dt"] and other_parameter_estimate:
        other = abs(B[-1])

    # Calculate scale parameter using scaler function
    # which() equivalent is just boolean indexing in numpy
    scale = scaler(distribution, Etype, errors[ot_logical], y_fitted[ot_logical], obs_in_sample, other)

    # Record constant if estimated
    if constant_estimate:
        constant_value = B[constant_name]

    # Prepare distribution parameters to return
    other_returned = {}
    
    # Write down parameters for distribution (always positive)
    if other_parameter_estimate:
        param_value = abs(B[-1])
    else:
        param_value = other

    # Set parameter name based on distribution
    if distribution == "dalaplace":
        other_returned["alpha"] = param_value
    elif distribution in ["dgnorm", "dlgnorm"]:
        other_returned["shape"] = param_value
    elif distribution == "dt":
        other_returned["nu"] = param_value

    # Add LASSO/RIDGE lambda if applicable
    if loss in ["LASSO", "RIDGE"]:
        other_returned["lambda"] = lambda_

    # Return ARIMA polynomials and indices for persistence and transition
    if arima_model:
        other_returned["polynomial"] = arima_polynomials
        other_returned["ARIMA_indices"] = {"nonZeroARI": non_zero_ari, "nonZeroMA": non_zero_ma}
        other_returned["ar_polynomial_matrix"] = np.zeros((ar_orders @ lags, ar_orders @ lags))
        
        if other_returned["ar_polynomial_matrix"].shape[0] > 1:
            # Set diagonal elements to 1 except first row/col
            other_returned["ar_polynomial_matrix"][1:-1, 2:] = np.eye(other_returned["ar_polynomial_matrix"].shape[0]-2)
            
            if ar_required:
                other_returned["ar_polynomial_matrix"][:, 0] = -arima_polynomials["ar_polynomial"][1:]
                
        other_returned["arma_parameters"] = arma_parameters

    # Amend the class of state matrix
    if "ts" in y_classes:
        mat_vt = pd.Series(
            mat_vt.T,
            index=pd.date_range(
                start=y.index[0] - pd.Timedelta(lags_model_max/y.index.freq),
                periods=len(mat_vt.T),
                freq=y.index.freq
            )
        )
    else:
        y_states_index = y_in_sample_index[0] - lags_model_max * np.diff(y_in_sample_index[-2:]) + \
                        np.arange(lags_model_max) * np.diff(y_in_sample_index[-2:])
        y_states_index = np.concatenate([y_states_index, y_in_sample_index])
        mat_vt = pd.Series(mat_vt.T, index=y_states_index)

    parameters_number[1, 4] = np.sum(parameters_number[1, :4])


    return {
        "model": None,
        "time_elapsed": None,
        "data": np.column_stack((None, xreg_data)),
        "holdout": None,
        "fitted": y_fitted,
        "residuals": errors,
        "forecast": y_forecast,
        "states": mat_vt,
        "profile": profiles_recent_table,
        "profile_initial": profiles_recent_initial,
        "persistence": persistence,
        "phi": phi,
        "transition": mat_f,
        "measurement": mat_wt,
        "initial": initial_value,
        "initial_type": initial_type,
        "initial_estimated": initial_estimated,
        "orders": orders,
        "arma": arma_parameters_list,
        "constant": constant_value,
        "n_param": parameters_number,
        "occurrence": oes_model,
        "formula": formula,
        "regressors": regressors,
        "loss": loss,
        "loss_value": cf_value,
        "log_lik": log_lik_adam_value,
        "distribution": distribution,
        "scale": scale,
        "other": other_returned,
        "B": B,
        "lags": lags,
        "lags_all": lags_model_all,
        "FI": fi
    }