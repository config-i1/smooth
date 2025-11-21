import numpy as np
import nlopt
import pandas as pd
import warnings
import math

from smooth.adam_general.core.utils.ic import ic_function
from smooth.adam_general.core.creator import creator, initialiser, architector, filler
from smooth.adam_general.core.utils.cost_functions import CF, log_Lik_ADAM
from smooth.adam_general.core.utils.utils import scaler
from smooth.adam_general._adam_general import adam_fitter, adam_forecaster







def _setup_arima_polynomials(model_type_dict, arima_dict, lags_dict):
    """
    Set up companion matrices for ARIMA polynomials.

    Parameters
    ----------
    model_type_dict : dict
        Model type specification
    arima_dict : dict
        ARIMA components specification
    lags_dict : dict
        Lags information

    Returns
    -------
    tuple
        AR and MA polynomial matrices
    """
    if model_type_dict["arima_model"]:
        # AR polynomials
        ar_polynomial_matrix = np.zeros(
            (
                np.sum(arima_dict["ar_orders"]) * lags_dict["lags"],
                np.sum(arima_dict["ar_orders"]) * lags_dict["lags"],
            )
        )
        if ar_polynomial_matrix.shape[0] > 1:
            ar_polynomial_matrix[1:, :-1] = np.eye(ar_polynomial_matrix.shape[0] - 1)

        # MA polynomials
        ma_polynomial_matrix = np.zeros(
            (
                np.sum(arima_dict["ma_orders"]) * lags_dict["lags"],
                np.sum(arima_dict["ma_orders"]) * lags_dict["lags"],
            )
        )
        if ma_polynomial_matrix.shape[0] > 1:
            ma_polynomial_matrix[1:, :-1] = np.eye(ma_polynomial_matrix.shape[0] - 1)

        return ar_polynomial_matrix, ma_polynomial_matrix
    else:
        return None, None


def _set_distribution(general_dict, model_type_dict):
    """
    Set distribution based on error term and loss function.

    Parameters
    ----------
    general_dict : dict
        General model parameters
    model_type_dict : dict
        Model type specification

    Returns
    -------
    dict
        Updated general_dict with appropriate distribution
    """
    general_dict_updated = general_dict.copy()

    if general_dict["distribution"] == "default":
        if general_dict["loss"] == "likelihood":
            general_dict_updated["distribution_new"] = (
                "dnorm" if model_type_dict["error_type"] == "A" else "dgamma"
            )
        elif general_dict["loss"] in ["MAEh", "MACE", "MAE"]:
            general_dict_updated["distribution_new"] = "dlaplace"
        elif general_dict["loss"] in ["HAMh", "CHAM", "HAM"]:
            general_dict_updated["distribution_new"] = "ds"
        else:
            general_dict_updated["distribution_new"] = "dnorm"
    else:
        general_dict_updated["distribution_new"] = general_dict["distribution"]

    return general_dict_updated


def _setup_optimization_parameters(
    general_dict,
    explanatory_dict,
    B,
    maxeval,
    adam_created,
    observations_dict,
    multisteps,
):
    """
    Set up parameters for optimization.

    Parameters
    ----------
    general_dict : dict
        General model parameters
    explanatory_dict : dict
        Explanatory variables specification
    B : array-like
        Initial parameter vector
    maxeval : int or None
        Maximum number of evaluations
    adam_created : dict
        Model matrices created by creator
    observations_dict : dict
        Observations information
    multisteps : bool
        Whether to use multi-step estimation

    Returns
    -------
    tuple
        maxeval_used, updated general_dict
    """
    general_dict_updated = general_dict.copy()

    # Set maxeval based on parameters
    maxeval_used = maxeval
    if maxeval is None:
        maxeval_used = len(B) * 200

        # If xreg model, do more iterations
        if explanatory_dict["xreg_model"]:
            maxeval_used = len(B) * 150
            maxeval_used = max(1500, maxeval_used)

    # Handle LASSO/RIDGE denominator calculation
    if general_dict["loss"] in ["LASSO", "RIDGE"]:
        if explanatory_dict["xreg_number"] > 0:
            # Calculate standard deviation for each column of matWt
            general_dict_updated["denominator"] = np.std(adam_created["mat_wt"], axis=0)
            # Replace infinite values with 1
            general_dict_updated["denominator"][
                np.isinf(general_dict_updated["denominator"])
            ] = 1
        else:
            general_dict_updated["denominator"] = None
        # Calculate denominator for y values
        general_dict_updated["y_denominator"] = max(
            np.std(np.diff(observations_dict["y_in_sample"])), 1
        )
    else:
        general_dict_updated["denominator"] = None
        general_dict_updated["y_denominator"] = None

    general_dict_updated["multisteps"] = multisteps

    return maxeval_used, general_dict_updated


def _configure_optimizer(opt, lb, ub, maxeval_used, maxtime, B, explanatory_dict, maxeval=None):
    """
    Configure NLopt optimizer with appropriate settings.

    Parameters
    ----------
    opt : nlopt.opt
        NLopt optimizer object
    lb : array-like
        Lower bounds
    ub : array-like
        Upper bounds
    maxeval_used : int
        Maximum number of evaluations
    maxtime : float or None
        Maximum time for optimization

    Returns
    -------
    nlopt.opt
        Configured optimizer
    """
    # Set bounds

    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    # Set tolerances
    opt.set_xtol_rel(1e-6)  # Match R's tolerance
    opt.set_ftol_rel(1e-8)  # Match R's tolerance
    opt.set_ftol_abs(0)  # Match R's tolerance
    opt.set_xtol_abs(1e-8)  # Match R's tolerance

    # Set maximum evaluations
    opt.set_maxeval(maxeval_used)

    # Increase maxeval to match or exceed R's value
    if maxeval is None:
        # Increase the default multiplier to ensure we run at least as many iterations as R
        maxeval_used = len(B) * 40  # Increased from 120 to 200
        
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
    return opt


def _create_objective_function(
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
    profile_dict,
    general_dict,
):
    """
    Create objective function for optimization.

    Parameters
    ----------
    model_type_dict : dict
        Model type specification
    components_dict : dict
        Components information
    lags_dict : dict
        Lags information
    adam_created : dict
        Model matrices created by creator
    persistence_dict : dict
        Persistence parameters
    initials_dict : dict
        Initial values
    arima_dict : dict
        ARIMA components specification
    explanatory_dict : dict
        Explanatory variables specification
    phi_dict : dict
        Damping parameter information
    constant_dict : dict
        Constant term specification
    observations_dict : dict
        Observations information
    profile_dict : dict
        Profiles information
    general_dict : dict
        General model parameters

    Returns
    -------
    function
        Objective function for optimizer
    """
    iteration_count = [0]
    def objective_wrapper(x, grad):
        """
        Wrapper for the objective function.
        """
        iteration_count[0] += 1
        # Calculate the cost function
        cf_value = CF(
            B=x,
            model_type_dict=model_type_dict,
            components_dict=components_dict,
            lags_dict=lags_dict,
            matrices_dict=adam_created,
            persistence_checked=persistence_dict,
            initials_checked=initials_dict,
            arima_checked=arima_dict,
            explanatory_checked=explanatory_dict,
            phi_dict=phi_dict,
            constants_checked=constant_dict,
            observations_dict=observations_dict,
            profile_dict=profile_dict,
            general=general_dict,
            bounds="usual",
        )
        

        # Limit extreme values to prevent numerical instability
        if not np.isfinite(cf_value) or cf_value > 1e10:
            return 1e10
        return cf_value

    return objective_wrapper


def _run_optimization(opt, B):
    """
    Run optimization and handle potential errors.

    Parameters
    ----------
    opt : nlopt.opt
        Configured optimizer
    B : array-like
        Initial parameter vector

    Returns
    -------
    object
        Optimization result object with x, fun, and success attributes
    """
    try:
        # Run optimization
        x = opt.optimize(B)
        #print(x)
        res_fun = opt.last_optimum_value()
        res = type("OptimizeResult", (), {"x": x, "fun": res_fun, "success": True})
    except Exception as e:
        print(e)
        # Log error if needed, but don't use the variable 'e' if not needed
        res = type("OptimizeResult", (), {"x": B, "fun": 1e300, "success": False})

    return res


def _calculate_loglik(
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
    multisteps,
    n_param_estimated,
):
    """
    Calculate log-likelihood for the estimated model.

    Parameters
    ----------
    B : array-like
        Parameter vector
    model_type_dict : dict
        Model type specification
    components_dict : dict
        Components information
    lags_dict : dict
        Lags information
    adam_created : dict
        Model matrices created by creator
    persistence_dict : dict
        Persistence parameters
    initials_dict : dict
        Initial values
    arima_dict : dict
        ARIMA components specification
    explanatory_dict : dict
        Explanatory variables specification
    phi_dict : dict
        Damping parameter information
    constant_dict : dict
        Constant term specification
    observations_dict : dict
        Observations information
    occurrence_dict : dict
        Occurrence model information
    general_dict : dict
        General model parameters
    profile_dict : dict
        Profiles information
    multisteps : bool
        Whether to use multi-step estimation
    n_param_estimated : int
        Number of estimated parameters

    Returns
    -------
    dict
        Log-likelihood value and information
    """
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
        multisteps=multisteps,
    )

    # In case of likelihood, we typically have one more parameter to estimate - scale.
    return {
        "value": log_lik_adam_value,
        "nobs": observations_dict["obs_in_sample"],
        "df": n_param_estimated + (1 if general_dict["loss"] == "likelihood" else 0),
    }


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
    multisteps=False,
    lb=None,
    ub=None,
    maxtime=None,
    print_level=1,  # 1 or 0
    maxeval=None,
    B_initial=None,
    return_matrices=False,
):
    """
    Estimate parameters for ADAM model.

    This function coordinates the process of estimating optimal parameters
    for an ADAM model by setting up the model structure, defining optimization
    parameters, and executing the optimization process.

    Parameters
    ----------
    general_dict : dict
        General model configuration parameters
    model_type_dict : dict
        Model type specification parameters
    lags_dict : dict
        Information about model lags
    observations_dict : dict
        Observed data information
    arima_dict : dict
        ARIMA component specification
    constant_dict : dict
        Constant term specification
    explanatory_dict : dict
        External regressors specification
    profiles_recent_table : array-like
        Recent profiles table
    profiles_recent_provided : bool
        Whether profiles were provided by user
    persistence_dict : dict
        Persistence parameters specification
    initials_dict : dict
        Initial values specification
    phi_dict : dict
        Damping parameter specification
    components_dict : dict
        Model components information
    occurrence_dict : dict
        Occurrence model information
    multisteps : bool, optional
        Whether to use multi-step estimation
    lb : array-like, optional
        Lower bounds for parameters
    ub : array-like, optional
        Upper bounds for parameters
    maxtime : float, optional
        Maximum optimization time
    print_level : int, optional
        Verbosity level (0 or 1)
    maxeval : int, optional
        Maximum number of evaluations
    B_initial : array-like, optional
        Initial parameter vector to start optimization from.
        If provided, it overrides the default initialization logic.

    Returns
    -------
    dict
        Dictionary containing estimated parameters and model information
    """
    # Step 1: Set up model structure
    # Simple call of the architector
    model_type_dict, components_dict, lags_dict, observations_dict, profile_dict = (
        architector(
            model_type_dict,
            lags_dict,
            observations_dict,
            arima_dict,
            constant_dict,
            explanatory_dict,
            profiles_recent_table,
            profiles_recent_provided,
        )
    )


    # Step 2: Create model matrices
    # Simple call of the creator
    adam_created = creator(
        model_type_dict,
        lags_dict,
        profile_dict,
        observations_dict,
        persistence_dict,
        initials_dict,
        arima_dict,
        constant_dict,
        phi_dict,
        components_dict,
        explanatory_dict,
    )
    # Step 3: Initialize parameters
    b_values = initialiser(
        model_type_dict=model_type_dict,
        components_dict=components_dict,
        lags_dict=lags_dict,
        adam_created=adam_created,
        persistence_checked=persistence_dict,
        initials_checked=initials_dict,
        arima_checked=arima_dict,
        constants_checked=constant_dict,
        explanatory_checked=explanatory_dict,
        observations_dict=observations_dict,
        bounds=general_dict["bounds"],
        phi_dict=phi_dict,
        profile_dict=profile_dict,
    )
    # Get initial parameter vector and bounds
    if B_initial is not None:
        B = B_initial
    else:
        B = b_values["B"]
    if lb is None:
        lb = b_values["Bl"]
    if ub is None:
        ub = b_values["Bu"]
    
    # Ensure bounds are compatible with B
    if B_initial is not None:
        # Extend bounds if necessary
        if len(lb) != len(B):
            # This shouldn't happen if B_initial has correct length, but safety first
            if len(lb) < len(B):
                lb = np.pad(lb, (0, len(B) - len(lb)), 'constant', constant_values=-np.inf)
                ub = np.pad(ub, (0, len(B) - len(ub)), 'constant', constant_values=np.inf)
        
        # Check compatibility
        if np.any(B < lb) or np.any(B > ub):
            # Adjust bounds to accommodate B
            lb = np.minimum(lb, B)
            ub = np.maximum(ub, B)
            # Maybe relax bounds slightly
            lb[B < lb] -= 1e-5
            ub[B > ub] += 1e-5

    # Step 4: Set up ARIMA polynomials if needed
    ar_polynomial_matrix, ma_polynomial_matrix = _setup_arima_polynomials(
        model_type_dict, arima_dict, lags_dict
    )

    # Step 5: Set appropriate distribution
    general_dict = _set_distribution(general_dict, model_type_dict)


    # Step 6: Configure optimization parameters
    print_level_hidden = print_level
    if print_level == 1:
        print_level = 0

    maxeval_used, general_dict = _setup_optimization_parameters(
        general_dict,
        explanatory_dict,
        B,
        maxeval,
        adam_created,
        observations_dict,
        multisteps,
    )

    # Step 7: Create and configure optimizer
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, len(B))
    opt = _configure_optimizer(opt, lb, ub, maxeval_used, maxtime, B, explanatory_dict)
    
    # start counting
    iteration_count = [0]  
    # Step 8: Create objective function
    objective_wrapper = _create_objective_function(
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
        profile_dict,
        general_dict,
    )

    # Set objective function
    opt.set_min_objective(objective_wrapper)
    # Step 9: Run optimization
    res = _run_optimization(opt, B)
    #print(res.fun)
    # Step 10: Process results
    B[:] = res.x
    CF_value = res.fun
    # A fix for the special case of LASSO/RIDGE with lambda==1
    if (
        any(general_dict["loss"] == loss_type for loss_type in ["LASSO", "RIDGE"])
        and general_dict["lambda_"] == 1
    ):
        CF_value = 0

    n_param_estimated = len(B)

    # Step 11: Calculate log-likelihood
    log_lik_adam_value = _calculate_loglik(
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
        multisteps,
        n_param_estimated,
    )

    # Step 12: Prepare and return results
    result = {
        "B": B,
        "CF_value": CF_value,
        "n_param_estimated": n_param_estimated,
        "log_lik_adam_value": log_lik_adam_value,
        "arima_polynomials": adam_created["arima_polynomials"],
    }

    if return_matrices:
        # Ensure matrices are updated with final B values and backcasted states
        # The CF function uses copies, so we need to update the originals
        from smooth.adam_general.core.utils.cost_functions import log_Lik_ADAM
        from smooth.adam_general._adam_general import adam_fitter

        # Fill matrices with final B
        filler(
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
        )

        # Run adam_fitter with backcasting to update states
        if initials_dict["initial_type"] in ["complete", "backcasting"]:
            mat_vt = np.asfortranarray(adam_created["mat_vt"], dtype=np.float64)
            mat_wt = np.asfortranarray(adam_created["mat_wt"], dtype=np.float64)
            mat_f = np.asfortranarray(adam_created["mat_f"], dtype=np.float64)
            vec_g = np.asfortranarray(adam_created["vec_g"], dtype=np.float64)
            lags_model_all = np.asfortranarray(lags_dict["lags_model_all"], dtype=np.uint64).reshape(-1, 1)
            index_lookup_table = np.asfortranarray(profile_dict["index_lookup_table"], dtype=np.uint64)
            profiles_recent_table = np.asfortranarray(profile_dict["profiles_recent_table"], dtype=np.float64)
            y_in_sample = np.asfortranarray(observations_dict["y_in_sample"], dtype=np.float64)
            ot = np.asfortranarray(observations_dict["ot"], dtype=np.float64)

            adam_fitter(
                matrixVt=mat_vt,
                matrixWt=mat_wt,
                matrixF=mat_f,
                vectorG=vec_g,
                lags=lags_model_all,
                indexLookupTable=index_lookup_table,
                profilesRecent=profiles_recent_table,
                E=model_type_dict["error_type"],
                T=model_type_dict["trend_type"],
                S=model_type_dict["season_type"],
                nNonSeasonal=components_dict["components_number_ets"] - components_dict["components_number_ets_seasonal"],
                nSeasonal=components_dict["components_number_ets_seasonal"],
                nArima=components_dict.get("components_number_arima", 0),
                nXreg=explanatory_dict["xreg_number"],
                constant=constant_dict["constant_required"],
                vectorYt=y_in_sample,
                vectorOt=ot,
                backcast=True,
                nIterations=initials_dict.get("n_iterations", 2) or 2,
                refineHead=not arima_dict["arima_model"],
                adamETS=False
            )

            # Update original matrices
            adam_created["mat_vt"][:] = mat_vt[:]

        result["matrices"] = adam_created
        result["lags_dict"] = lags_dict
        result["profile_dict"] = profile_dict
        result["components_dict"] = components_dict

    return result


# Helper functions for selector


def _form_model_pool(model_type_dict, silent=False):
    """
    Form a pool of models based on model type specifications.

    Parameters
    ----------
    model_type_dict : dict
        Model type specification
    silent : bool, optional
        Whether to suppress progress messages

    Returns
    -------
    tuple
        pool_small, pool_errors, pool_trends, pool_seasonals, check_trend, check_seasonal
    """
    # Check if the pool was provided
    if model_type_dict["models_pool"] is not None:
        return model_type_dict["models_pool"], None, None, None, None, None

    # Print status if not silent
    if not silent:
        print("Forming the pool of models based on... ", end="")

    # Define the whole pool of errors, trends, and seasonals
    if not model_type_dict["allow_multiplicative"]:
        pool_errors = ["A"]
        pool_trends = ["N", "A", "Ad"]
        pool_seasonals = ["N", "A"]
    else:
        pool_errors = ["A", "M"]
        pool_trends = ["N", "A", "Ad", "M", "Md"]
        pool_seasonals = ["N", "A", "M"]

    # Prepare error type
    if model_type_dict["error_type"] != "Z":
        pool_errors = model_type_dict["error_type"]
        pool_errors_small = model_type_dict["error_type"]
    else:
        pool_errors_small = "A"

    # Prepare trend type
    if model_type_dict["trend_type"] != "Z":
        if model_type_dict["trend_type"] == "X":
            pool_trends_small = ["N", "A"]
            pool_trends = ["N", "A", "Ad"]
            check_trend = True
        elif model_type_dict["trend_type"] == "Y":
            pool_trends_small = ["N", "M"]
            pool_trends = ["N", "M", "Md"]
            check_trend = True
        else:
            if model_type_dict["damped"]:
                pool_trends = pool_trends_small = [model_type_dict["trend_type"] + "d"]
            else:
                pool_trends = pool_trends_small = [model_type_dict["trend_type"]]
            check_trend = False
    else:
        pool_trends_small = ["N", "A"]
        check_trend = True

    # Prepare seasonal type
    if model_type_dict["season_type"] != "Z":
        if model_type_dict["season_type"] == "X":
            pool_seasonals = pool_seasonals_small = ["N", "A"]
            check_seasonal = True
        elif model_type_dict["season_type"] == "Y":
            pool_seasonals_small = ["N", "M"]
            pool_seasonals = ["N", "M"]
            check_seasonal = True
        else:
            pool_seasonals_small = [model_type_dict["season_type"]]
            pool_seasonals = [model_type_dict["season_type"]]
            check_seasonal = False
    else:
        pool_seasonals_small = ["N", "A", "M"]
        check_seasonal = True

    # Create the small pool
    pool_small = []
    for error in pool_errors_small:
        for trend in pool_trends_small:
            for seasonal in pool_seasonals_small:
                pool_small.append(error + trend + seasonal)

    # Align error and seasonality, if the error was not forced to be additive
    if any(model[2] == "M" for model in pool_small) and model_type_dict[
        "error_type"
    ] not in ["A", "X"]:
        for i, model in enumerate(pool_small):
            if model[2] == "M":
                pool_small[i] = "M" + model[1:]

    return (
        pool_small,
        pool_errors,
        pool_trends,
        pool_seasonals,
        check_trend,
        check_seasonal,
    )


def _estimate_model(
    model_type_dict_temp,
    phi_dict_temp,
    general_dict,
    lags_dict,
    observations_dict,
    arima_dict,
    constant_dict,
    explanatory_dict,
    profiles_recent_table,
    profiles_recent_provided,
    persistence_results,
    initials_results,
    occurrence_dict,
    components_dict,
):
    """
    Estimate a single model and calculate its information criterion.

    Parameters
    ----------
    model_type_dict_temp : dict
        Temporary model type dictionary for this model
    phi_dict_temp : dict
        Temporary phi dictionary for this model
    general_dict : dict
        General model parameters
    lags_dict : dict
        Lags information
    observations_dict : dict
        Observations information
    arima_dict : dict
        ARIMA components specification
    constant_dict : dict
        Constant term specification
    explanatory_dict : dict
        Explanatory variables specification
    profiles_recent_table : array-like
        Recent profiles table
    profiles_recent_provided : bool
        Whether profiles were provided by user
    persistence_results : dict
        Persistence parameters
    initials_results : dict
        Initial values
    occurrence_dict : dict
        Occurrence model information
    components_dict : dict
        Components information

    Returns
    -------
    dict
        Dictionary containing estimation results and model information
    """
    # Estimate the model
    adam_estimated = estimator(
        general_dict=general_dict,
        model_type_dict=model_type_dict_temp,
        lags_dict=lags_dict,
        observations_dict=observations_dict,
        arima_dict=arima_dict,
        constant_dict=constant_dict,
        explanatory_dict=explanatory_dict,
        profiles_recent_table=profiles_recent_table,
        profiles_recent_provided=profiles_recent_provided,
        persistence_dict=persistence_results,
        initials_dict=initials_results,
        occurrence_dict=occurrence_dict,
        phi_dict=phi_dict_temp,
        components_dict=components_dict,
    )

    # Calculate information criterion
    IC = ic_function(general_dict["ic"], loglik=adam_estimated["log_lik_adam_value"])

    # Return results
    return {
        "adam_estimated": adam_estimated,
        "IC": IC,
        "model_type_dict": model_type_dict_temp,
        "phi_dict": phi_dict_temp,
        "model": model_type_dict_temp["model"],
    }


def _run_branch_and_bound(
    pool_small,
    model_type_dict,
    phi_dict,
    general_dict,
    lags_dict,
    observations_dict,
    arima_dict,
    constant_dict,
    explanatory_dict,
    profiles_recent_table,
    profiles_recent_provided,
    persistence_results,
    initials_results,
    occurrence_dict,
    components_dict,
    pool_seasonals,
    pool_trends,
    check_seasonal,
    check_trend,
):
    """
    Run branch and bound algorithm to efficiently search model space.

    Parameters
    ----------
    pool_small : list
        Small pool of models to test
    model_type_dict : dict
        Model type specification
    phi_dict : dict
        Damping parameter specification
    general_dict : dict
        General model parameters
    lags_dict : dict
        Lags information
    observations_dict : dict
        Observations information
    arima_dict : dict
        ARIMA components specification
    constant_dict : dict
        Constant term specification
    explanatory_dict : dict
        Explanatory variables specification
    profiles_recent_table : array-like
        Recent profiles table
    profiles_recent_provided : bool
        Whether profiles were provided by user
    persistence_results : dict
        Persistence parameters
    initials_results : dict
        Initial values
    occurrence_dict : dict
        Occurrence model information
    components_dict : dict
        Components information
    pool_seasonals : list
        Pool of possible seasonal components
    pool_trends : list
        Pool of possible trend components
    check_seasonal : bool
        Whether to check seasonal components
    check_trend : bool
        Whether to check trend components

    Returns
    -------
    tuple
        results, models_tested
    """
    # Initialize variables
    models_tested = []
    results = [None] * len(pool_small)
    j = 1
    i = 0
    check = True
    best_i = best_j = 1

    # Branch and bound algorithm
    while check:
        i += 1
        model_current = pool_small[j - 1]

        # Create temporary copies for this model
        model_type_dict_temp = model_type_dict.copy()
        model_type_dict_temp["model"] = model_current
        phi_dict_temp = phi_dict.copy()

        # Set model parameters based on current model
        model_type_dict_temp["error_type"] = model_current[0]
        model_type_dict_temp["trend_type"] = model_current[1]

        if len(model_current) == 4:
            phi_dict_temp["phi"] = 0.95
            phi_dict_temp["phi_estimate"] = True
            model_type_dict_temp["season_type"] = model_current[3]
        else:
            phi_dict_temp["phi"] = 1
            phi_dict_temp["phi_estimate"] = False
            model_type_dict_temp["season_type"] = model_current[2]

        # Estimate the model
        results[i - 1] = _estimate_model(
            model_type_dict_temp,
            phi_dict_temp,
            general_dict,
            lags_dict,
            observations_dict,
            arima_dict,
            constant_dict,
            explanatory_dict,
            profiles_recent_table,
            profiles_recent_provided,
            persistence_results,
            initials_results,
            occurrence_dict,
            components_dict,
        )

        # Update phi value if it was estimated
        if phi_dict_temp["phi_estimate"]:
            results[i - 1]["phi_dict"]["phi"] = results[i - 1]["B"].get("phi")
        else:
            results[i - 1]["phi_dict"]["phi"] = 1

        # Add to models tested
        models_tested.append(model_current)

        # Branch and bound decision logic
        if j > 1:
            # If the first is better than the second, then choose first
            if results[best_i - 1]["IC"] <= results[i - 1]["IC"]:
                # If Ttype is the same, then we check seasonality
                if model_current[1] == pool_small[best_j - 1][1]:
                    pool_seasonals = results[best_i - 1]["model_type_dict"][
                        "season_type"
                    ]
                    check_seasonal = False
                    j = [
                        k + 1
                        for k in range(len(pool_small))
                        if pool_small[k] != pool_small[best_j - 1]
                        and pool_small[k][-1] == pool_seasonals
                    ]
                # Otherwise we checked trend
                else:
                    pool_trends = results[best_j - 1]["model_type_dict"]["trend_type"]
                    check_trend = False
            else:
                # If the trend is the same
                if model_current[1] == pool_small[best_i - 1][1]:
                    pool_seasonals = [
                        s
                        for s in pool_seasonals
                        if s != model_type_dict_temp["season_type"]
                    ]
                    if len(pool_seasonals) > 1:
                        # Select another seasonal model, not from previous iteration and not current
                        best_j = j
                        best_i = i
                        j = 3
                    else:
                        best_j = j
                        best_i = i
                        # Move to checking the trend
                        j = [
                            k + 1
                            for k in range(len(pool_small))
                            if pool_small[k][-1] == pool_seasonals[0]
                            and pool_small[k][1] != model_current[1]
                        ]
                        check_seasonal = False
                else:
                    pool_trends = [
                        t
                        for t in pool_trends
                        if t != model_type_dict_temp["trend_type"]
                    ]
                    best_i = i
                    best_j = j
                    check_trend = False

            if not any([check_trend, check_seasonal]):
                check = False
        else:
            j = 2

        # If j is None or exceeds the pool size, we're done
        if not j:
            j = len(pool_small)
        if j > len(pool_small):
            check = False

    return results, models_tested


def _estimate_all_models(
    models_pool,
    model_type_dict,
    phi_dict,
    general_dict,
    lags_dict,
    observations_dict,
    arima_dict,
    constant_dict,
    explanatory_dict,
    profiles_recent_table,
    profiles_recent_provided,
    persistence_results,
    initials_results,
    occurrence_dict,
    components_dict,
    silent=False,
):
    """
    Estimate all models in the provided pool.

    Parameters
    ----------
    models_pool : list
        List of models to estimate
    model_type_dict : dict
        Model type specification
    phi_dict : dict
        Damping parameter specification
    general_dict : dict
        General model parameters
    lags_dict : dict
        Lags information
    observations_dict : dict
        Observations information
    arima_dict : dict
        ARIMA components specification
    constant_dict : dict
        Constant term specification
    explanatory_dict : dict
        Explanatory variables specification
    profiles_recent_table : array-like
        Recent profiles table
    profiles_recent_provided : bool
        Whether profiles were provided by user
    persistence_results : dict
        Persistence parameters
    initials_results : dict
        Initial values
    occurrence_dict : dict
        Occurrence model information
    components_dict : dict
        Components information
    silent : bool, optional
        Whether to suppress progress messages

    Returns
    -------
    list
        List of results for each model
    """
    models_number = len(models_pool)
    results = [None] * models_number

    # Print progress message if not silent
    if not silent:
        print("Estimation progress:    ", end="")

    # Estimate each model
    for j in range(models_number):
        if not silent:
            if j == 0:
                print("\b", end="")
            print("\b" * (len(str(round((j) / models_number * 100))) + 1), end="")
            print(f"{round((j+1)/models_number * 100)}%", end="")

        model_current = models_pool[j]
            # Create copies for this model
        model_type_dict_temp = model_type_dict.copy()
        phi_dict_temp = phi_dict.copy()

        # Set model parameters
        model_type_dict_temp["error_type"] = model_current[0]
        model_type_dict_temp["trend_type"] = model_current[1]

        if len(model_current) == 4:
            phi_dict_temp["phi"] = 0.95
            model_type_dict_temp["season_type"] = model_current[3]
            phi_dict_temp["phi_estimate"] = True
        else:
            phi_dict_temp["phi"] = 1
            model_type_dict_temp["season_type"] = model_current[2]
            phi_dict_temp["phi_estimate"] = False
        
        # Estimate the model
        results[j] = {}
        results[j]['adam_estimated'] = estimator(
                general_dict=general_dict,
                model_type_dict=model_type_dict_temp,
                lags_dict=lags_dict,
                observations_dict=observations_dict,
                arima_dict=arima_dict,
                constant_dict=constant_dict,
                explanatory_dict=explanatory_dict,
                profiles_recent_table=profiles_recent_table,
                profiles_recent_provided=profiles_recent_provided,
                persistence_dict=persistence_results,
                initials_dict=initials_results,
                occurrence_dict=occurrence_dict,
                phi_dict=phi_dict_temp,
                components_dict=components_dict,
            )
        results[j]["IC"] = ic_function(general_dict['ic'],loglik=results[j]['adam_estimated']["log_lik_adam_value"])
        results[j]['model_type_dict'] = model_type_dict_temp
        results[j]['phi_dict'] = phi_dict_temp
        results[j]['model'] = model_current
    if not silent:
        print("... Done!")

    return results


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
    criterion="AICc",
    silent=False,
):
    """
    Create a pool of models and select the best one based on information criteria.

    Parameters
    ----------
    model_type_dict : dict
        Model type specification parameters
    phi_dict : dict
        Damping parameter specification
    general_dict : dict
        General model configuration parameters
    lags_dict : dict
        Information about model lags
    observations_dict : dict
        Observed data information
    arima_dict : dict
        ARIMA component specification
    constant_dict : dict
        Constant term specification
    explanatory_dict : dict
        External regressors specification
    occurrence_dict : dict
        Occurrence model information
    components_dict : dict
        Model components information
    profiles_recent_table : array-like
        Recent profiles table
    profiles_recent_provided : bool
        Whether profiles were provided by user
    persistence_results : dict
        Persistence parameters specification
    initials_results : dict
        Initial values specification
    criterion : str, optional
        Information criterion to use for model selection
    silent : bool, optional
        Whether to suppress progress messages

    Returns
    -------
    dict
        Dictionary containing results and model selection information
    """
    # Set the information criterion in general_dict
    general_dict["ic"] = criterion
    # Step 1: Form the model pool
    (
        pool_small,
        pool_errors,
        pool_trends,
        pool_seasonals,
        check_trend,
        check_seasonal,
    ) = _form_model_pool(model_type_dict, silent)
    # Step 2: Run branch and bound if pool was not provided

    if model_type_dict["models_pool"] is None:
        # Run branch and bound to select models
        results, models_tested = _run_branch_and_bound(
            pool_small,
            model_type_dict,
            phi_dict,
            general_dict,
            lags_dict,
            observations_dict,
            arima_dict,
            constant_dict,
            explanatory_dict,
            profiles_recent_table,
            profiles_recent_provided,
            persistence_results,
            initials_results,
            occurrence_dict,
            components_dict,
            pool_seasonals,
            pool_trends,
            check_seasonal,
            check_trend,
        )

        # Prepare a bigger pool based on the small one
        model_type_dict["models_pool"] = list(
            set(
                models_tested
                + [
                    e + t + s
                    for e in pool_errors
                    for t in pool_trends
                    for s in pool_seasonals
                ]
            )
        )

    # Step 3: Estimate all models in the pool
    results = _estimate_all_models(
        model_type_dict["models_pool"],
        model_type_dict,
        phi_dict,
        general_dict,
        lags_dict,
        observations_dict,
        arima_dict,
        constant_dict,
        explanatory_dict,
        profiles_recent_table,
        profiles_recent_provided,
        persistence_results,
        initials_results,
        occurrence_dict,
        components_dict,
        silent,
    )
    #print(results)

    # Step 4: Extract ICs and find the best model
    models_number = len(model_type_dict["models_pool"])
    ic_selection = [results[j]["IC"] for j in range(models_number)]
    # Create dictionary with model names and ICs
    ic_selection_dict = dict(zip(model_type_dict["models_pool"], ic_selection))
    # Replace NaN values with large number
    ic_selection = [1e100 if math.isnan(x) else x for x in ic_selection]

    return {"results": results, "ic_selection": ic_selection_dict}


def _generate_forecasts(
    general_dict,
    observations_dict,
    matrices_dict,
    lags_dict,
    profiles_dict,
    model_type_dict,
    components_dict,
    explanatory_checked,
    constants_checked,
):
    """
    Generate forecasts if the horizon is non-zero.

    Parameters
    ----------
    general_dict : dict
        General model parameters
    observations_dict : dict
        Observations information
    matrices_dict : dict
        Model matrices
    lags_dict : dict
        Lags information
    profiles_dict : dict
        Profiles information
    model_type_dict : dict
        Model type specification
    components_dict : dict
        Components information
    explanatory_checked : dict
        Explanatory variables specification
    constants_checked : dict
        Constant term specification

    Returns
    -------
    pandas.Series
        Forecasted values
    """
    # If horizon is zero, return an empty series
    if general_dict["h"] <= 0:
        if any(observations_dict.get("y_classes", []) == "ts"):
            return pd.Series(
                [np.nan],
                index=pd.date_range(
                    start=observations_dict["y_forecast_start"],
                    periods=1,
                    freq=observations_dict["y_frequency"],
                ),
            )
        else:
            return pd.Series(
                np.full(general_dict["horizon"], np.nan),
                index=observations_dict["y_forecast_index"],
            )

    # Create forecast Series
    if not isinstance(observations_dict.get("y_in_sample"), pd.Series):
        y_forecast = pd.Series(
            np.full(general_dict["h"], np.nan),
            index=pd.date_range(
                start=observations_dict["y_forecast_start"],
                periods=general_dict["h"],
                freq=observations_dict["frequency"],
            ),
        )
    else:
        y_forecast = pd.Series(
            np.full(general_dict["h"], np.nan),
            index=observations_dict["y_forecast_index"],
        )

    # Generate forecasts
    forecasts = adam_forecaster(
        matrixWt=matrices_dict["mat_wt"][-general_dict["h"] :],
        matrixF=matrices_dict["mat_f"],
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
        horizon=general_dict["h"],
    ).flatten()

    # Fill in forecast values
    y_forecast[:] = forecasts

    # Replace NaNs with zeros
    if np.any(np.isnan(y_forecast)):
        y_forecast[np.isnan(y_forecast)] = 0

    return y_forecast


def _update_distribution(general_dict, model_type_dict):
    """
    Update distribution based on error term and loss function.

    Parameters
    ----------
    general_dict : dict
        General model parameters
    model_type_dict : dict
        Model type specification

    Returns
    -------
    dict
        Updated general_dict
    """
    general_dict_updated = general_dict.copy()

    if general_dict["distribution"] == "default":
        if general_dict["loss"] == "likelihood":
            if model_type_dict["error_type"] == "A":
                general_dict_updated["distribution"] = "dnorm"
            elif model_type_dict["error_type"] == "M":
                general_dict_updated["distribution"] = "dgamma"
        elif general_dict["loss"] in ["MAEh", "MACE", "MAE"]:
            general_dict_updated["distribution"] = "dlaplace"
        elif general_dict["loss"] in ["HAMh", "CHAM", "HAM"]:
            general_dict_updated["distribution"] = "ds"
        elif general_dict["loss"] in ["MSEh", "MSCE", "MSE", "GPL"]:
            general_dict_updated["distribution"] = "dnorm"
        else:
            general_dict_updated["distribution"] = "dnorm"

    return general_dict_updated


def _process_initial_values(
    model_type_dict,
    lags_dict,
    matrices_dict,
    initials_checked,
    arima_checked,
    explanatory_checked,
    components_dict,
):
    """
    Process initial values for the model.

    Parameters
    ----------
    model_type_dict : dict
        Model type specification
    lags_dict : dict
        Lags information
    matrices_dict : dict
        Model matrices
    initials_checked : dict
        Initial values
    arima_checked : dict
        ARIMA components specification
    explanatory_checked : dict
        Explanatory variables specification
    components_dict : dict
        Components information

    Returns
    -------
    tuple
        initial_value, initial_value_ets, initial_value_names, initial_estimated
    """
    # Initial values to return
    initial_value = [None] * (
        model_type_dict["ets_model"]
        * (
            1
            + model_type_dict["model_is_trendy"]
            + model_type_dict["model_is_seasonal"]
        )
        + arima_checked["arima_model"]
        + explanatory_checked["xreg_model"]
    )
    initial_value_ets = [None] * (
        model_type_dict["ets_model"] * len(lags_dict["lags_model"])
    )
    initial_value_names = [""] * (
        model_type_dict["ets_model"]
        * (
            1
            + model_type_dict["model_is_trendy"]
            + model_type_dict["model_is_seasonal"]
        )
        + arima_checked["arima_model"]
        + explanatory_checked["xreg_model"]
    )

    # The vector that defines what was estimated in the model
    initial_estimated = [False] * (
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

    # Write down the initials of ETS
    if model_type_dict["ets_model"]:
        # Write down level, trend and seasonal
        for i in range(len(lags_dict["lags_model"])):
            # In case of level / trend, we want to get the very first value
            if lags_dict["lags_model"][i] == 1:
                initial_value_ets[i] = matrices_dict["mat_vt"][
                    i, : lags_dict["lags_model_max"]
                ][0]
            # In cases of seasonal components, they should be at the end of the pre-heat period
            # Only extract lag-1 values (the last one is the normalized value computed from others)
            else:
                start_idx = lags_dict["lags_model_max"] - lags_dict["lags_model"][i]
                seasonal_full = matrices_dict["mat_vt"][
                    i, start_idx : lags_dict["lags_model_max"]
                ].copy()

                # Renormalise seasonal initials to match R implementation
                if np.any(~np.isnan(seasonal_full)):
                    if model_type_dict["season_type"] == "A":
                        seasonal_full = seasonal_full - np.nanmean(seasonal_full)
                    elif model_type_dict["season_type"] == "M":
                        positive_vals = seasonal_full.copy()
                        positive_vals[positive_vals <= 0] = np.nan
                        if np.all(np.isnan(positive_vals)):
                            geo_mean = 1.0
                        else:
                            geo_mean = np.exp(np.nanmean(np.log(positive_vals)))
                            if np.isnan(geo_mean) or geo_mean == 0:
                                geo_mean = 1.0
                        seasonal_full = seasonal_full / geo_mean

                initial_value_ets[i] = seasonal_full[:-1]

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

                initial_estimated[
                    j + 1 : j + 1 + components_dict["components_number_ets_seasonal"]
                ] = seasonal_estimate_list
                # Remove the level from ETS list
                initial_value_ets[0] = None
                j += 1
                if len(seasonal_estimate_list) > 1:
                    initial_value[j] = [x for x in initial_value_ets if x is not None]
                    initial_value_names[j] = "seasonal"
                    for k in range(components_dict["components_number_ets_seasonal"]):
                        initial_estimated[j + k] = f"seasonal{k+1}"
                else:
                    initial_value[j] = next(
                        x for x in initial_value_ets if x is not None
                    )
                    initial_value_names[j] = "seasonal"
                    initial_estimated[j] = "seasonal"

    # Write down the ARIMA initials
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

    # Set names for initial values
    initial_value = {
        name: value for name, value in zip(initial_value_names, initial_value) if name
    }

    return initial_value, initial_value_ets, initial_value_names, initial_estimated


def _process_arma_parameters(B, arima_checked):
    """
    Process ARMA parameters from the estimates.

    Parameters
    ----------
    B : dict
        Parameter estimates
    arima_checked : dict
        ARIMA components specification

    Returns
    -------
    dict or None
        ARMA parameters list
    """
    if not arima_checked["arima_model"]:
        return None

    arma_parameters_list = {}

    # Process AR parameters
    if arima_checked["ar_required"] and arima_checked["ar_estimate"]:
        # Avoid damping parameter phi by checking name length > 3
        arma_parameters_list["ar"] = [
            b for name, b in B.items() if len(name) > 3 and name.startswith("phi")
        ]
    elif arima_checked["ar_required"] and not arima_checked["ar_estimate"]:
        # Avoid damping parameter phi
        arma_parameters_list["ar"] = [
            p
            for name, p in arima_checked["arma_parameters"].items()
            if name.startswith("phi")
        ]

    # Process MA parameters
    if arima_checked["ma_required"] and arima_checked["ma_estimate"]:
        arma_parameters_list["ma"] = [
            b for name, b in B.items() if name.startswith("theta")
        ]
    elif arima_checked["ma_required"] and not arima_checked["ma_estimate"]:
        arma_parameters_list["ma"] = [
            p
            for name, p in arima_checked["arma_parameters"].items()
            if name.startswith("theta")
        ]

    return arma_parameters_list


def _calculate_scale(
    general_dict, model_type_dict, errors, y_fitted, observations_dict, other
):
    """
    Calculate scale parameter using scaler function.

    Parameters
    ----------
    general_dict : dict
        General model parameters
    model_type_dict : dict
        Model type specification
    errors : pandas.Series
        Model errors
    y_fitted : pandas.Series
        Fitted values
    observations_dict : dict
        Observations information
    other : dict or None
        Other parameters

    Returns
    -------
    float
        Scale parameter
    """
    return scaler(
        general_dict["distribution_new"],
        model_type_dict["error_type"],
        errors[observations_dict["ot_logical"]],
        y_fitted[observations_dict["ot_logical"]],
        observations_dict["obs_in_sample"],
        other,
    )


def _process_other_parameters(
    constants_checked, adam_estimated, general_dict, arima_checked, lags_dict
):
    """
    Process other parameters including constants and ARIMA polynomials.

    Parameters
    ----------
    constants_checked : dict
        Constant term specification
    adam_estimated : dict
        Estimated model parameters
    general_dict : dict
        General model parameters
    arima_checked : dict
        ARIMA components specification
    lags_dict : dict
        Lags information

    Returns
    -------
    tuple
        constant_value, other_returned
    """
    # Record constant if estimated
    if constants_checked["constant_estimate"]:
        constant_value = adam_estimated["B"][constants_checked["constant_name"]]
    else:
        constant_value = None

    # Prepare distribution parameters to return
    other_returned = {}

    # Add LASSO/RIDGE lambda if applicable
    if general_dict["loss"] in ["LASSO", "RIDGE"]:
        other_returned["lambda"] = general_dict["lambda_"]

    # Return ARIMA polynomials and indices for persistence and transition
    if arima_checked["arima_model"]:
        other_returned["polynomial"] = adam_estimated["arima_polynomials"]
        other_returned["ARIMA_indices"] = {
            "nonZeroARI": arima_checked["non_zero_ari"],
            "nonZeroMA": arima_checked["non_zero_ma"],
        }
        other_returned["ar_polynomial_matrix"] = np.zeros(
            (
                sum(arima_checked["ar_orders"]) * lags_dict["lags"],
                sum(arima_checked["ar_orders"]) * lags_dict["lags"],
            )
        )

        if other_returned["ar_polynomial_matrix"].shape[0] > 1:
            # Set diagonal elements to 1 except first row/col
            other_returned["ar_polynomial_matrix"][1:-1, 2:] = np.eye(
                other_returned["ar_polynomial_matrix"].shape[0] - 2
            )

            if arima_checked["ar_required"]:
                other_returned["ar_polynomial_matrix"][:, 0] = -arima_polynomials[
                    "ar_polynomial"
                ][1:]

        other_returned["arma_parameters"] = arima_checked["arma_parameters"]

    return constant_value, other_returned


def _format_output(
    model_type_dict,
    y_fitted,
    errors,
    y_forecast,
    matrices_dict,
    profiles_dict,
    persistence,
    phi_dict,
    initial_value,
    initials_checked,
    initial_estimated,
    general_dict,
    arma_parameters_list,
    constant_value,
    occurrence_dict,
    explanatory_checked,
    scale,
    other_returned,
    adam_estimated,
    lags_dict,
    observations_dict,
):
    """
    Format the final output for the model.

    Parameters
    ----------
    model_type_dict : dict
        Model type specification
    y_fitted : pandas.Series
        Fitted values
    errors : pandas.Series
        Model errors
    y_forecast : pandas.Series
        Forecasted values
    matrices_dict : dict
        Model matrices
    profiles_dict : dict
        Profiles information
    persistence : array
        Persistence vector
    phi_dict : dict
        Damping parameter information
    initial_value : dict
        Initial values
    initials_checked : dict
        Initial values specification
    initial_estimated : list
        Which initials were estimated
    general_dict : dict
        General model parameters
    arma_parameters_list : dict or None
        ARMA parameters
    constant_value : float or None
        Constant term value
    occurrence_dict : dict
        Occurrence model information
    explanatory_checked : dict
        Explanatory variables specification
    scale : float
        Scale parameter
    other_returned : dict
        Other parameters
    adam_estimated : dict
        Estimated model parameters
    lags_dict : dict
        Lags information
    observations_dict : dict
        Observations information

    Returns
    -------
    dict
        Formatted model output
    """
    # Amend the class of state matrix
    if not isinstance(observations_dict.get("y_in_sample"), pd.Series):
        mat_vt = pd.Series(
            matrices_dict["mat_vt"].T.flatten(),
            index=pd.date_range(
                start=observations_dict["y_forecast_start"],
                periods=len(matrices_dict["mat_vt"].T),
                freq=observations_dict["frequency"],
            ),
        )
    else:
        mat_vt = pd.Series(
            matrices_dict["mat_vt"].T, index=observations_dict["y_forecast_index"]
        )

    # Update parameters number
    general_dict["parameters_number"][0][2] = np.sum(
        general_dict["parameters_number"][0][:2]
    )

    # Return the formatted output
    return {
        "model": model_type_dict["model"],
        "time_elapsed": None,  # here will count the time
        "holdout": general_dict["holdout"],
        "fitted": y_fitted,
        "residuals": errors,
        "forecast": y_forecast,
        "states": mat_vt,
        "profile": profiles_dict["profiles_recent_table"],
        "profile_initial": profiles_dict["profiles_recent_initial"],
        "persistence": persistence,
        "phi": phi_dict["phi"],
        "transition": matrices_dict["mat_f"],
        "measurement": matrices_dict["mat_wt"],
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
    Prepare final model output after estimation.

    This function takes the estimated model parameters and prepares
    the fitted values, forecasts, and other model components for output.

    Parameters
    ----------
    model_type_dict : dict
        Model type specification parameters
    components_dict : dict
        Model components information
    lags_dict : dict
        Information about model lags
    matrices_dict : dict
        Model matrices from creator
    persistence_checked : dict
        Processed persistence parameters
    initials_checked : dict
        Processed initial values
    arima_checked : dict
        Processed ARIMA component information
    explanatory_checked : dict
        Processed external regressors information
    phi_dict : dict
        Damping parameter information
    constants_checked : dict
        Processed constant term information
    observations_dict : dict
        Observed data information
    occurrence_dict : dict
        Occurrence model information
    general_dict : dict
        General model configuration parameters
    profiles_dict : dict
        Profiles information
    adam_estimated : dict
        Parameter estimates from optimization
    bounds : str, optional
        Type of bounds used
    other : dict, optional
        Additional parameters

    Returns
    -------
    dict
        Dictionary containing fitted model and forecasts
    """
    # Step 1: Fill in the matrices with estimated parameters
    matrices_dict = _fill_matrices(
        adam_estimated=adam_estimated,
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

    # Step 2: Update profiles and phi
    profiles_dict, phi_dict = _update_profiles(
        matrices_dict=matrices_dict,
        profiles_dict=profiles_dict,
        lags_dict=lags_dict,
        phi_dict=phi_dict,
        adam_estimated=adam_estimated,
    )

    # Step 3: Prepare arrays for model fitting
    (
        y_in_sample,
        ot,
        mat_vt,
        mat_wt,
        mat_f,
        vec_g,
        lags_model_all,
        index_lookup_table,
        profiles_recent_table,
    ) = _prepare_arrays(
        matrices_dict=matrices_dict,
        observations_dict=observations_dict,
        profiles_dict=profiles_dict,
        lags_dict=lags_dict,
    )

    # Step 4: Fit the model
    adam_fitted = _fit_model(
        mat_vt=mat_vt,
        mat_wt=mat_wt,
        mat_f=mat_f,
        vec_g=vec_g,
        lags_model_all=lags_model_all,
        index_lookup_table=index_lookup_table,
        profiles_recent_table=profiles_recent_table,
        model_type_dict=model_type_dict,
        components_dict=components_dict,
        explanatory_checked=explanatory_checked,
        constants_checked=constants_checked,
        y_in_sample=y_in_sample,
        ot=ot,
        initials_checked=initials_checked,
    )

    # Step 5: Update matrices with fitted values
    matrices_dict, profiles_dict = _update_matrices(
        matrices_dict=matrices_dict,
        profiles_dict=profiles_dict,
        adam_fitted=adam_fitted,
        model_type_dict=model_type_dict,
        components_dict=components_dict,
    )

    # Step 6: Prepare fitted values and errors
    y_fitted, errors = _prepare_fitted_values(
        observations_dict=observations_dict,
        adam_fitted=adam_fitted,
        occurrence_dict=occurrence_dict,
    )

    # Step 7: Generate forecasts
    y_forecast = _generate_forecasts(
        general_dict=general_dict,
        observations_dict=observations_dict,
        matrices_dict=matrices_dict,
        lags_dict=lags_dict,
        profiles_dict=profiles_dict,
        model_type_dict=model_type_dict,
        components_dict=components_dict,
        explanatory_checked=explanatory_checked,
        constants_checked=constants_checked,
    )

    # Step 8: Update distribution
    general_dict = _update_distribution(
        general_dict=general_dict, model_type_dict=model_type_dict
    )

    # Step 9: Process initial values
    initial_value, initial_value_ets, initial_value_names, initial_estimated = (
        _process_initial_values(
            model_type_dict=model_type_dict,
            lags_dict=lags_dict,
            matrices_dict=matrices_dict,
            initials_checked=initials_checked,
            arima_checked=arima_checked,
            explanatory_checked=explanatory_checked,
            components_dict=components_dict,
        )
    )

    # Step 10: Get persistence values
    persistence = np.array(matrices_dict["vec_g"]).flatten()

    # Step 11: Handle explanatory variables
    if (
        explanatory_checked["xreg_model"]
        and explanatory_checked.get("regressors") != "adapt"
    ):
        explanatory_checked["regressors"] = "use"
    elif not explanatory_checked["xreg_model"]:
        explanatory_checked["regressors"] = None

    # Step 12: Process ARMA parameters
    arma_parameters_list = _process_arma_parameters(
        B=adam_estimated["B"], arima_checked=arima_checked
    )

    # Step 13: Calculate scale parameter
    scale = _calculate_scale(
        general_dict=general_dict,
        model_type_dict=model_type_dict,
        errors=errors,
        y_fitted=y_fitted,
        observations_dict=observations_dict,
        other=other,
    )

    # Step 14: Process other parameters
    constant_value, other_returned = _process_other_parameters(
        constants_checked=constants_checked,
        adam_estimated=adam_estimated,
        general_dict=general_dict,
        arima_checked=arima_checked,
        lags_dict=lags_dict,
    )

    # Step 15: Format and return final output
    return _format_output(
        model_type_dict=model_type_dict,
        y_fitted=y_fitted,
        errors=errors,
        y_forecast=y_forecast,
        matrices_dict=matrices_dict,
        profiles_dict=profiles_dict,
        persistence=persistence,
        phi_dict=phi_dict,
        initial_value=initial_value,
        initials_checked=initials_checked,
        initial_estimated=initial_estimated,
        general_dict=general_dict,
        arma_parameters_list=arma_parameters_list,
        constant_value=constant_value,
        occurrence_dict=occurrence_dict,
        explanatory_checked=explanatory_checked,
        scale=scale,
        other_returned=other_returned,
        adam_estimated=adam_estimated,
        lags_dict=lags_dict,
        observations_dict=observations_dict,
    )
