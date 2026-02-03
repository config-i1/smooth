import numpy as np

from smooth.adam_general.core.utils.cost_functions import CF, log_Lik_ADAM

# Note: adam_cpp instance is passed to functions that need C++ integration


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
        elif general_dict["loss"] in ["MAE", "MAEh", "TMAE", "GTMAE", "MACE"]:
            general_dict_updated["distribution_new"] = "dlaplace"
        elif general_dict["loss"] in ["HAM", "HAMh", "THAM", "GTHAM", "CHAM"]:
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
    components_dict,
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
    components_dict : dict
        Component counts (components_number_ets, components_number_arima)

    Returns
    -------
    tuple
        maxeval_used, updated general_dict
    """
    general_dict_updated = general_dict.copy()

    # Set maxeval based on parameters - match R's defaults exactly
    maxeval_used = maxeval
    if maxeval is None:
        maxeval_used = len(B) * 40  # R's default: length(B) * 40

        # If xreg model, do more iterations (R: max(1000, length(B) * 100))
        if explanatory_dict["xreg_model"]:
            maxeval_used = len(B) * 100
            maxeval_used = max(1000, maxeval_used)

    # Handle LASSO/RIDGE denominator calculation
    if general_dict["loss"] in ["LASSO", "RIDGE"]:
        if explanatory_dict["xreg_number"] > 0:
            # Get component counts for slicing xreg columns only
            components_number_ets = components_dict["components_number_ets"]
            components_number_arima = components_dict.get("components_number_arima", 0)
            xreg_number = explanatory_dict["xreg_number"]

            # Slice only xreg columns from mat_wt (after ETS and ARIMA components)
            xreg_start = components_number_ets + components_number_arima
            xreg_end = xreg_start + xreg_number
            mat_wt_xreg = adam_created["mat_wt"][:, xreg_start:xreg_end]

            # Calculate standard deviation for each xreg column
            # Use ddof=1 to match R's sd() which uses sample std (n-1 denominator)
            general_dict_updated["denominator"] = np.std(mat_wt_xreg, axis=0, ddof=1)
            # Replace infinite values with 1
            general_dict_updated["denominator"][
                np.isinf(general_dict_updated["denominator"])
            ] = 1
        else:
            general_dict_updated["denominator"] = None

        # Calculate denominator for y values
        # Use ddof=1 to match R's sd() which uses sample std (n-1 denominator)
        y_diff = np.diff(observations_dict["y_in_sample"])
        y_std = np.std(y_diff, ddof=1)
        general_dict_updated["y_denominator"] = max(y_std, 1)
    else:
        general_dict_updated["denominator"] = None
        general_dict_updated["y_denominator"] = None

    general_dict_updated["multisteps"] = multisteps

    return maxeval_used, general_dict_updated


def _configure_optimizer(
    opt,
    lb,
    ub,
    maxeval_used,
    maxtime,
    xtol_rel=1e-6,
    xtol_abs=1e-8,
    ftol_rel=1e-8,
    ftol_abs=0,
):
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
        Maximum number of evaluations (already computed by
        _setup_optimization_parameters)
    maxtime : float or None
        Maximum time for optimization
    xtol_rel : float, default=1e-6
        Relative tolerance on optimization parameters
    xtol_abs : float, default=1e-8
        Absolute tolerance on optimization parameters
    ftol_rel : float, default=1e-8
        Relative tolerance on function value
    ftol_abs : float, default=0
        Absolute tolerance on function value

    Returns
    -------
    nlopt.opt
        Configured optimizer
    """
    # Set bounds
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    # Set tolerances (configurable)
    opt.set_xtol_rel(xtol_rel)
    opt.set_ftol_rel(ftol_rel)
    opt.set_ftol_abs(ftol_abs)
    opt.set_xtol_abs(xtol_abs)

    # Set maximum evaluations - use the value computed by _setup_optimization_parameters
    # which matches R's defaults: len(B)*40 standard, max(1000, len(B)*100) for xreg
    opt.set_maxeval(maxeval_used)

    # Set timeout if specified, otherwise use long default
    if maxtime is not None:
        opt.set_maxtime(maxtime)
    else:
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
    adam_cpp,
    print_level,
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
    best_cf = [float("inf")]

    def objective_wrapper(x, grad):
        """
        Wrapper for the objective function.
        """
        # Calculate the cost function with exception handling
        try:
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
                adam_cpp=adam_cpp,
                bounds="usual",
            )
        except Exception:
            cf_value = 1e100

        # Increment iteration counter
        iteration_count[0] += 1

        # Print optimization progress if print_level > 0
        if print_level > 0:
            # Track best CF
            if cf_value < best_cf[0]:
                best_cf[0] = cf_value

            # Print every iteration
            param_str = ", ".join([f"{val:.4f}" for val in x])
            print(f"Iter {iteration_count[0]:3d}: B=[{param_str}] -> CF={cf_value:.6f}")

        # Limit extreme values to prevent numerical instability
        if not np.isfinite(cf_value) or cf_value > 1e10:
            return 1e10
        return cf_value

    return objective_wrapper


def _run_optimization(opt, B):
    """
    Run optimization.

    Parameters
    ----------
    opt : nlopt.opt
        Configured optimizer
    B : array-like
        Initial parameter vector

    Returns
    -------
    numpy.ndarray
        Optimized parameter vector
    """
    # Run optimization - nlopt updates B in-place and returns optimized values
    # Any nlopt termination (including RoundoffLimited) still returns valid B
    try:
        x = opt.optimize(B)
    except Exception:
        # If any exception, B has still been updated in-place
        x = B.copy()
    return x


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
    adam_cpp,
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
        adam_cpp,
        multisteps=multisteps,
    )

    # In case of likelihood, we typically have one more parameter to estimate - scale.
    return {
        "value": log_lik_adam_value,
        "nobs": observations_dict["obs_in_sample"],
        "df": n_param_estimated + (1 if general_dict["loss"] == "likelihood" else 0),
    }
