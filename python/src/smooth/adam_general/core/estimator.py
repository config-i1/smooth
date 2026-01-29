import math

import nlopt
import numpy as np

from smooth.adam_general.core.creator import architector, creator, filler, initialiser
from smooth.adam_general.core.utils.cost_functions import CF, log_Lik_ADAM
from smooth.adam_general.core.utils.ic import ic_function

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
        elif general_dict["loss"] in [
            "MAE", "MAEh", "TMAE", "GTMAE", "MACE"
        ]:
            general_dict_updated["distribution_new"] = "dlaplace"
        elif general_dict["loss"] in [
            "HAM", "HAMh", "THAM", "GTHAM", "CHAM"
        ]:
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


def _configure_optimizer(
    opt, lb, ub, maxeval_used, maxtime,
    xtol_rel=1e-6, xtol_abs=1e-8, ftol_rel=1e-8, ftol_abs=0):
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
        Maximum number of evaluations (already computed by _setup_optimization_parameters)
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
    print_level
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
    best_cf = [float('inf')]

    def objective_wrapper(x, grad):
        """
        Wrapper for the objective function.
        """
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
            adam_cpp=adam_cpp,
            bounds="usual",
        )

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
    except:
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


def _run_two_stage_estimator(
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
    print_level=0,
    maxeval=None,
    return_matrices=False,
    xtol_rel=1e-6,
    xtol_abs=1e-8,
    ftol_rel=1e-8,
    ftol_abs=0,
    algorithm="NLOPT_LN_NELDERMEAD",
    smoother="lowess",
):
    """
    Internal function to handle two-stage initialization within estimator.

    This implements the two-stage procedure:
    1. Stage 1: Run with initial_type="complete" (backcasting) to get initial parameters
    2. Extract initial states from Stage 1's fitted mat_vt
    3. Stage 2: Run optimization with original "two-stage" using B_initial from Stage 1
    """
    # Stage 1: Run with "complete" to get backcasted parameters
    stage1_initials = initials_dict.copy()
    stage1_initials["initial_type"] = "complete"
    # For Stage 1 (backcasting): use user's value if provided, otherwise default to 2
    if initials_dict.get("n_iterations_provided", False):
        stage1_initials["n_iterations"] = initials_dict["n_iterations"]
    else:
        stage1_initials["n_iterations"] = 2

    adam_estimated_s1 = estimator(
        general_dict=general_dict,
        model_type_dict=model_type_dict,
        lags_dict=lags_dict,
        observations_dict=observations_dict,
        arima_dict=arima_dict,
        constant_dict=constant_dict,
        explanatory_dict=explanatory_dict,
        profiles_recent_table=profiles_recent_table,
        profiles_recent_provided=profiles_recent_provided,
        persistence_dict=persistence_dict,
        initials_dict=stage1_initials,
        phi_dict=phi_dict,
        components_dict=components_dict,
        occurrence_dict=occurrence_dict,
        multisteps=multisteps,
        maxtime=maxtime,
        print_level=print_level,
        maxeval=maxeval,
        return_matrices=True,  # Need matrices to extract initial states
        xtol_rel=xtol_rel,
        xtol_abs=xtol_abs,
        ftol_rel=ftol_rel,
        ftol_abs=ftol_abs,
        algorithm=algorithm,
        smoother=smoother,
    )

    # Get B vector structure with "two-stage" (includes initial states in B)
    # We need to call architector/creator/initialiser to get proper B structure
    model_type_dict_s2, components_dict_s2, lags_dict_s2, observations_dict_s2, profile_dict_s2, _ = (
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

    adam_created_s2 = creator(
        model_type_dict_s2,
        lags_dict_s2,
        profile_dict_s2,
        observations_dict_s2,
        persistence_dict,
        initials_dict,  # Original "two-stage"
        arima_dict,
        constant_dict,
        phi_dict,
        components_dict_s2,
        explanatory_dict,
        smoother=smoother,
    )

    b_values = initialiser(
        model_type_dict=model_type_dict_s2,
        components_dict=components_dict_s2,
        lags_dict=lags_dict_s2,
        adam_created=adam_created_s2,
        persistence_checked=persistence_dict,
        initials_checked=initials_dict,  # Original "two-stage"
        arima_checked=arima_dict,
        constants_checked=constant_dict,
        explanatory_checked=explanatory_dict,
        observations_dict=observations_dict_s2,
        bounds=general_dict["bounds"],
        phi_dict=phi_dict,
        profile_dict=profile_dict_s2,
    )

    B = b_values["B"].copy()
    Bl = b_values["Bl"].copy()
    Bu = b_values["Bu"].copy()

    # Extract results from Stage 1
    B_stage1 = adam_estimated_s1["B"]

    # Calculate nParametersBack: number of persistence, phi, and ARMA parameters
    # (excluding initials, constant, shape) - matching R's adam.R lines 2518-2522
    # IMPORTANT: Use model_type_dict_s2 (from architector) which has correct model_is_trendy
    # and model_is_seasonal flags, not the original model_type_dict which may have stale values
    # from the parent "ZXZ" model during model selection.
    n_params_back = 0
    if model_type_dict_s2.get("ets_model", False):
        if persistence_dict.get("persistence_level_estimate", False):
            n_params_back += 1
        if model_type_dict_s2.get("model_is_trendy", False) and persistence_dict.get("persistence_trend_estimate", False):
            n_params_back += 1
        if model_type_dict_s2.get("model_is_seasonal", False):
            persistence_seasonal_estimate = persistence_dict.get("persistence_seasonal_estimate", [])
            if isinstance(persistence_seasonal_estimate, list):
                n_params_back += sum(persistence_seasonal_estimate)
            elif persistence_seasonal_estimate:
                n_params_back += 1
        if phi_dict.get("phi_estimate", False):
            n_params_back += 1

    if explanatory_dict.get("xreg_model", False) and persistence_dict.get("persistence_xreg_estimate", False):
        xreg_parameters_persistence = explanatory_dict.get("xreg_parameters_persistence", [0])
        n_params_back += max(xreg_parameters_persistence) if xreg_parameters_persistence else 0

    if arima_dict.get("arima_model", False):
        ar_orders = arima_dict.get("ar_orders", [])
        ma_orders = arima_dict.get("ma_orders", [])
        if arima_dict.get("ar_estimate", False):
            n_params_back += sum(ar_orders) if ar_orders else 0
        if arima_dict.get("ma_estimate", False):
            n_params_back += sum(ma_orders) if ma_orders else 0

    # Copy persistence/phi/ARMA from Stage 1 to B
    if n_params_back > 0:
        B[:n_params_back] = B_stage1[:n_params_back]

    # Extract initial states from Stage 1's fitted mat_vt
    mat_vt_s1 = adam_estimated_s1["matrices"]["mat_vt"]
    lags_dict_s1 = adam_estimated_s1["lags_dict"]
    lags_model_s1 = lags_dict_s1["lags_model"]
    lags_model_max_s1 = lags_dict_s1["lags_model_max"]
    components_dict_s1 = adam_estimated_s1["components_dict"]

    initial_states = []
    current_row = 0

    # Level
    if initials_dict.get("initial_level_estimate", False):
        initial_states.append(mat_vt_s1[current_row, 0])
        current_row += 1
    elif model_type_dict.get("ets_model", False):
        current_row += 1

    # Trend
    if model_type_dict.get("model_is_trendy", False):
        if initials_dict.get("initial_trend_estimate", False):
            initial_states.append(mat_vt_s1[current_row, 0])
        current_row += 1

    # Seasonal
    if model_type_dict.get("model_is_seasonal", False):
        n_seasonal = components_dict_s1.get("components_number_ets_seasonal", 0)
        seasonal_estimate = initials_dict.get("initial_seasonal_estimate", [False] * n_seasonal)
        if not isinstance(seasonal_estimate, list):
            seasonal_estimate = [seasonal_estimate] * n_seasonal

        for i in range(n_seasonal):
            lag = lags_model_s1[current_row] if current_row < len(lags_model_s1) else 1
            if seasonal_estimate[i] if i < len(seasonal_estimate) else False:
                start_idx = lags_model_max_s1 - lag
                full_seasonal = mat_vt_s1[current_row, start_idx:lags_model_max_s1].copy()

                # Renormalize
                season_type = model_type_dict.get("season_type", "N")
                if season_type == "A":
                    full_seasonal = full_seasonal - np.mean(full_seasonal)
                elif season_type == "M":
                    if np.all(full_seasonal > 0):
                        geo_mean = np.exp(np.mean(np.log(full_seasonal)))
                        full_seasonal = full_seasonal / geo_mean

                # Truncate to m-1
                initial_states.extend(full_seasonal[:-1].tolist())
            current_row += 1

    # ARIMA initials
    if arima_dict.get("arima_model", False):
        n_arima = initials_dict.get("initial_arima_number", 0)
        if initials_dict.get("initial_arima_estimate", False) and n_arima > 0:
            for i in range(n_arima):
                initial_states.append(mat_vt_s1[current_row + i, lags_model_max_s1 - 1])
            current_row += n_arima

    # xreg initials
    if explanatory_dict.get("xreg_model", False):
        xreg_number = explanatory_dict.get("xreg_number", 0)
        xreg_params_estimated = explanatory_dict.get("xreg_parameters_estimated", [])
        if initials_dict.get("initial_xreg_estimate", False) and xreg_number > 0:
            n_ets = components_dict_s1.get("components_number_ets", 0)
            n_arima_components = components_dict_s1.get("components_number_arima", 0)
            xreg_start_row = n_ets + n_arima_components
            xreg_initials_all = mat_vt_s1[xreg_start_row:xreg_start_row + xreg_number, lags_model_max_s1 - 1]
            if xreg_params_estimated is not None and len(xreg_params_estimated) > 0:
                xreg_params_estimated_arr = np.array(xreg_params_estimated)
                xreg_initials_filtered = xreg_initials_all[xreg_params_estimated_arr == 1]
                initial_states.extend(xreg_initials_filtered.tolist())

    # Put extracted initials into B
    if len(initial_states) > 0:
        B[n_params_back:n_params_back + len(initial_states)] = initial_states

    # Handle constant
    if constant_dict.get("constant_estimate", False):
        n_ets = components_dict_s1.get("components_number_ets", 0)
        n_arima_components = components_dict_s1.get("components_number_arima", 0)
        xreg_number = explanatory_dict.get("xreg_number", 0) if explanatory_dict.get("xreg_model", False) else 0
        constant_idx = n_params_back + len(initial_states)
        if constant_idx < len(B):
            constant_row = n_ets + n_arima_components + xreg_number
            if constant_row < mat_vt_s1.shape[0]:
                constant_value = mat_vt_s1[constant_row, lags_model_max_s1 - 1]
                if not np.isnan(constant_value):
                    B[constant_idx] = constant_value

    # Handle other parameters like shape
    if general_dict.get("other_parameter_estimate", False):
        if len(B_stage1) > 0:
            B[-1] = abs(B_stage1[-1])

    # Adjust bounds
    Bl[np.isnan(Bl)] = -np.inf
    Bu[np.isnan(Bu)] = np.inf
    Bl[Bl > B] = -np.inf
    Bu[Bu < B] = np.inf

    # Stage 2: Run with original "two-stage" and B_initial from Stage 1
    return estimator(
        general_dict=general_dict,
        model_type_dict=model_type_dict,
        lags_dict=lags_dict,
        observations_dict=observations_dict,
        arima_dict=arima_dict,
        constant_dict=constant_dict,
        explanatory_dict=explanatory_dict,
        profiles_recent_table=profiles_recent_table,
        profiles_recent_provided=profiles_recent_provided,
        persistence_dict=persistence_dict,
        initials_dict=initials_dict,  # Original "two-stage"
        phi_dict=phi_dict,
        components_dict=components_dict,
        occurrence_dict=occurrence_dict,
        multisteps=multisteps,
        lb=Bl,
        ub=Bu,
        maxtime=maxtime,
        print_level=print_level,
        maxeval=maxeval,
        B_initial=B,  # Use B populated from Stage 1
        return_matrices=return_matrices,
        xtol_rel=xtol_rel,
        xtol_abs=xtol_abs,
        ftol_rel=ftol_rel,
        ftol_abs=ftol_abs,
        algorithm=algorithm,
        smoother=smoother,
    )


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
    print_level=0,  # 1 or 0
    maxeval=None,
    B_initial=None,
    return_matrices=False,
    # NLopt parameters
    xtol_rel=1e-6,
    xtol_abs=1e-8,
    ftol_rel=1e-8,
    ftol_abs=0,
    algorithm="NLOPT_LN_NELDERMEAD",
    smoother="lowess",
):
    """
    Estimate parameters for ADAM model using non-linear optimization.

    This function coordinates the complete parameter estimation process for an ADAM
    (Augmented Dynamic Adaptive Model) by setting up the state-space model structure,
    initializing parameters, and executing optimization via the NLopt library. The
    estimation minimizes a cost function (likelihood, MSE, MAE, etc.) to find optimal
    parameter values.

    The ADAM model is represented in state-space form as:

    .. math::

        y_t &= o_t(w(v_{t-l}) + h(x_t, a_{t-1}) + r(v_{t-l})\\epsilon_t)

        v_t &= f(v_{t-l}, a_{t-1}) + g(v_{t-l}, a_{t-1}, x_t)\\epsilon_t

    where:

    - :math:`y_t` is the observed value at time t
    - :math:`v_t` is the state vector
    - :math:`o_t` is the occurrence indicator (for intermittent data)
    - :math:`w(\\cdot)` is the measurement function
    - :math:`h(\\cdot)` is the exogenous variables function
    - :math:`r(\\cdot)` is the error function
    - :math:`f(\\cdot)` is the transition function
    - :math:`g(\\cdot)` is the persistence function
    - :math:`\\epsilon_t` is the error term

    **Estimation Algorithm**:

    1. **Architecture Setup**: Call ``architector()`` to define model structure, determine
       component counts, and set up lag structures
    2. **Matrix Creation**: Call ``creator()`` to build initial state-space matrices
       (measurement, transition, persistence)
    3. **Parameter Initialization**: Call ``initialiser()`` to construct the initial
       parameter vector B and bounds (lower/upper limits)
    4. **Distribution Selection**: Map loss function to appropriate error distribution
       (e.g., MSE → Normal, MAE → Laplace)
    5. **Optimization Setup**: Configure NLopt with Nelder-Mead algorithm, set tolerances
       and iteration limits
    6. **Objective Function**: Create wrapper for ``CF()`` cost function
    7. **Optimization Execution**: Run NLopt to minimize cost function
    8. **Log-likelihood Calculation**: Compute final log-likelihood using ``log_Lik_ADAM()``
    9. **Results Assembly**: Package estimated parameters, matrices, and diagnostics

    **Optimization Method**:

    Uses NLopt's Nelder-Mead (LN_NELDERMEAD) algorithm:

    - Gradient-free local optimization suitable for non-smooth cost functions
    - Tolerances: xtol_rel=1e-6, ftol_rel=1e-8, xtol_abs=1e-8
    - Maximum evaluations: 40 × len(B) for standard models, 150 × len(B) for models
      with external regressors
    - Default timeout: 30 minutes

    **Parameter Vector Structure**:

    The optimization parameter vector B contains (in order):

    1. **ETS Persistence**: α (level), β (trend), γ (seasonal)
    2. **Damping**: φ (if damped trend is present)
    3. **Initial States**: l₀ (level), b₀ (trend), s₀ (seasonal), ARIMA initial states
    4. **ARIMA Parameters**: AR coefficients (φ₁, φ₂, ...), MA coefficients (θ₁, θ₂, ...)
    5. **Regression Coefficients**: Weights for external regressors
    6. **Constant Term**: Intercept (if included)
    7. **Distribution Parameters**: Shape parameters for certain distributions

    Parameters
    ----------
    general_dict : dict
        General model configuration containing:

        - 'loss': Loss function ('likelihood', 'MSE', 'MAE', 'HAM', 'LASSO', 'RIDGE')
        - 'distribution': Error distribution specification
        - 'bounds': Parameter bounds type ('usual', 'admissible', 'none')
        - 'ic': Information criterion for model selection ('AIC', 'AICc', 'BIC', 'BICc')
        - 'h': Forecast horizon
        - 'holdout': Whether holdout sample is used

    model_type_dict : dict
        Model type specification containing:

        - 'model': Model string (e.g., "ANN", "AAA", "MAM")
        - 'error_type': Error type ('A' for additive, 'M' for multiplicative)
        - 'trend_type': Trend type ('N', 'A', 'Ad', 'M', 'Md')
        - 'season_type': Seasonality type ('N', 'A', 'M')
        - 'ets_model': Whether ETS components are present
        - 'arima_model': Whether ARIMA components are present
        - 'xreg_model': Whether external regressors are present

    lags_dict : dict
        Lag structure information containing:

        - 'lags': Primary lag vector (e.g., [1, 12] for monthly data with annual seasonality)
        - 'lags_model': Lags for each state component
        - 'lags_model_all': Complete lag specification for all components
        - 'lags_model_max': Maximum lag value (defines pre-sample period)

    observations_dict : dict
        Time series data containing:

        - 'y_in_sample': Observed values for estimation
        - 'y_holdout': Holdout sample (if applicable)
        - 'obs_in_sample': Number of in-sample observations
        - 'ot': Occurrence vector (1 for non-zero, 0 for zero observations)
        - 'ot_logical': Boolean mask for non-zero observations

    arima_dict : dict
        ARIMA specification containing:

        - 'arima_model': Whether ARIMA components exist
        - 'ar_orders': AR order for each lag (e.g., [1, 1] for SARIMA)
        - 'i_orders': Integration order for each lag
        - 'ma_orders': MA order for each lag
        - 'ar_estimate': Whether to estimate AR parameters
        - 'ma_estimate': Whether to estimate MA parameters
        - 'ar_required': Whether AR is included
        - 'ma_required': Whether MA is included

    constant_dict : dict
        Constant term specification containing:

        - 'constant_required': Whether model includes a constant
        - 'constant_estimate': Whether to estimate the constant
        - 'constant_name': Variable name for constant in B vector

    explanatory_dict : dict
        External regressors specification containing:

        - 'xreg_model': Whether regressors are present
        - 'xreg_number': Number of regressors
        - 'xreg_names': Names of regressor variables
        - 'mat_xt': Regressor data matrix

    profiles_recent_table : numpy.ndarray
        Recent profile values for time-varying parameters (advanced feature)
    profiles_recent_provided : bool
        Whether user provided custom profile values
    persistence_dict : dict
        Persistence parameters containing:

        - 'persistence_estimate': Whether to estimate persistence
        - 'persistence_level_estimate': Estimate α (level smoothing)
        - 'persistence_trend_estimate': Estimate β (trend smoothing)
        - 'persistence_seasonal_estimate': Estimate γ (seasonal smoothing)
        - 'persistence_level': Fixed value for α (if not estimated)
        - 'persistence_trend': Fixed value for β (if not estimated)
        - 'persistence_seasonal': Fixed value for γ (if not estimated)

    initials_dict : dict
        Initial states specification containing:

        - 'initial_type': Initialization method:

          * 'backcasting': Use backcasting with refinement
          * 'optimal': Optimize initial states
          * 'two-stage': First backcasting then optimisation
          * 'complete': Full backcasting without refinement
          * 'provided': User-provided initial values

        - 'initial_level': Fixed level initial (if provided)
        - 'initial_trend': Fixed trend initial (if provided)
        - 'initial_seasonal': Fixed seasonal initials (if provided)
        - 'initial_level_estimate': Whether to optimize level initial
        - 'initial_trend_estimate': Whether to optimize trend initial
        - 'initial_seasonal_estimate': Whether to optimize seasonal initials
        - 'n_iterations': Backcasting iteration count (typically 2)

    phi_dict : dict
        Damping parameter specification containing:

        - 'phi_estimate': Whether to estimate damping parameter
        - 'phi': Fixed damping value (if not estimated, typically 1.0 for undamped)

    components_dict : dict
        Component counts containing:

        - 'components_number_ets': Total ETS components
        - 'components_number_ets_seasonal': Number of seasonal components
        - 'components_number_ets_non_seasonal': Number of non-seasonal ETS components
        - 'components_number_arima': Number of ARIMA state components
        - 'components_number_all': Total state dimension

    occurrence_dict : dict
        Intermittent data specification containing:

        - 'occurrence_model': Whether intermittent demand model is used
        - 'occurrence_type': Type of occurrence model
        - 'ot': Occurrence vector

    multisteps : bool, default=False
        Whether to use multi-step-ahead cost function (e.g., for MSEh, MAEh).
        If True, errors are computed over h-step-ahead forecasts rather than
        one-step-ahead.
    lb : numpy.ndarray, optional
        Lower bounds for parameters. If None, computed by ``initialiser()``.
        Shape must match B.
    ub : numpy.ndarray, optional
        Upper bounds for parameters. If None, computed by ``initialiser()``.
        Shape must match B.
    maxtime : float, optional
        Maximum optimization time in seconds. If None, defaults to 1800 seconds (30 min).
    print_level : int, default=1
        Verbosity level:

        - 0: Silent (no output)
        - 1: Minimal output (currently suppressed in favor of general_dict['silent'])

    maxeval : int, optional
        Maximum number of cost function evaluations. If None, computed as:

        - Standard models: 40 × len(B)
        - Models with regressors: max(1500, 150 × len(B))

    B_initial : numpy.ndarray, optional
        Initial parameter vector to start optimization from. If provided, it overrides
        the default initialization computed by ``initialiser()``. Useful for:

        - Two-stage initialization (backcasting → optimal)
        - Warm-starting from previous estimates
        - Custom starting values

        Shape must match the parameter vector structure.
    return_matrices : bool, default=False
        Whether to return state-space matrices in the result dictionary. Useful for
        two-stage initialization where backcasted states are needed. If True, returns:

        - 'matrices': Updated state-space matrices with backcasted/optimized states
        - 'lags_dict': Updated lags dictionary
        - 'profile_dict': Updated profile dictionary
        - 'components_dict': Components information

    smoother : str, default="lowess"
        Smoother type for time series decomposition used in initial state estimation.

        - "lowess": Uses LOWESS for both trend and seasonal extraction
        - "ma": Uses moving average for both
        - "global": Uses lowess for trend and "ma" for seasonality

    Returns
    -------
    dict
        Dictionary containing estimation results:

        - **'B'** (numpy.ndarray): Optimized parameter vector
        - **'CF_value'** (float): Final cost function value at optimum
        - **'n_param_estimated'** (int): Number of estimated parameters
        - **'log_lik_adam_value'** (dict): Log-likelihood information with keys:

          * 'value': Log-likelihood value
          * 'nobs': Number of observations
          * 'df': Degrees of freedom (parameters + scale)

        - **'arima_polynomials'** (dict): AR and MA polynomial coefficients (if ARIMA present)

        If `return_matrices=True`, additionally includes:

        - **'matrices'** (dict): Updated state-space matrices (mat_vt, mat_wt, mat_f, vec_g)
        - **'lags_dict'** (dict): Lags information
        - **'profile_dict'** (dict): Profile matrices
        - **'components_dict'** (dict): Component counts

    Raises
    ------
    RuntimeError
        If optimization fails to converge or encounters numerical errors

    Notes
    -----
    **Special Cases**:

    1. **LASSO/RIDGE with λ=1**: Parameters are preset to zero, only initials are estimated
       using MSE
    2. **Two-stage initialization**: When initial_type='two-stage', the function is called
       twice:

       - Stage 1: initial_type='complete' (backcasting)
       - Stage 2: initial_type='optimal' using Stage 1 results as B_initial

    **Optimization Troubleshooting**:

    - If CF returns 1e100, parameter constraints were violated
    - If CF returns 1e300, NaN was encountered during computation
    - Increase maxeval if optimization terminates prematurely
    - Check bounds if estimated parameters are at boundaries

    **Performance Considerations**:

    - C++ ``adam_fitter()`` is called at each iteration (major computational cost)
    - Larger models (many seasonalities, high ARIMA orders) require more iterations
    - Backcasting (initial_type='backcasting' or 'complete') adds overhead but improves
      initial state estimates

    See Also
    --------
    architector : Set up model architecture
    creator : Create state-space matrices
    initialiser : Initialize parameter vector and bounds
    CF : Cost function for optimization
    log_Lik_ADAM : Calculate log-likelihood
    selector : Automatic model selection

    References
    ----------
    .. [1] Svetunkov, I. (2023). "Smooth forecasting with the smooth package in R".
           arXiv:2301.01790.
    .. [2] Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008).
           "Forecasting with Exponential Smoothing: The State Space Approach".
           Springer-Verlag.
    .. [3] Johnson, S.G. The NLopt nonlinear-optimization package.
           https://nlopt.readthedocs.io/

    Examples
    --------
    Estimate parameters for an ETS(A,A,A) model::

        >>> adam_estimated = estimator(
        ...     general_dict={'loss': 'likelihood', 'distribution': 'default', 'bounds': 'usual', ...},
        ...     model_type_dict={'model': 'AAA', 'error_type': 'A', 'trend_type': 'A', 'season_type': 'A', ...},
        ...     lags_dict={'lags': np.array([1, 12]), ...},
        ...     observations_dict={'y_in_sample': y_data, 'obs_in_sample': len(y_data), ...},
        ...     arima_dict={'arima_model': False, ...},
        ...     constant_dict={'constant_required': False, ...},
        ...     explanatory_dict={'xreg_model': False, ...},
        ...     profiles_recent_table=None,
        ...     profiles_recent_provided=False,
        ...     persistence_dict={'persistence_estimate': True, ...},
        ...     initials_dict={'initial_type': 'backcasting', ...},
        ...     phi_dict={'phi_estimate': False, 'phi': 1.0},
        ...     components_dict={...},
        ...     occurrence_dict={'occurrence_model': False, ...}
        ... )
        >>> print(f"Estimated parameters: {adam_estimated['B']}")
        >>> print(f"Log-likelihood: {adam_estimated['log_lik_adam_value']['value']}")

    Two-stage initialization example::

        >>> # Stage 1: Backcasting
        >>> initials_dict['initial_type'] = 'complete'
        >>> stage1 = estimator(..., return_matrices=True)
        >>> # Stage 2: Optimal
        >>> initials_dict['initial_type'] = 'optimal'
        >>> B_initial = np.concatenate([stage1['B'], extracted_initials_from_stage1])
        >>> stage2 = estimator(..., B_initial=B_initial)
    """
    # Handle two-stage initialization internally
    if initials_dict.get("initial_type") == "two-stage" and B_initial is None:
        return _run_two_stage_estimator(
            general_dict=general_dict,
            model_type_dict=model_type_dict,
            lags_dict=lags_dict,
            observations_dict=observations_dict,
            arima_dict=arima_dict,
            constant_dict=constant_dict,
            explanatory_dict=explanatory_dict,
            profiles_recent_table=profiles_recent_table,
            profiles_recent_provided=profiles_recent_provided,
            persistence_dict=persistence_dict,
            initials_dict=initials_dict,
            phi_dict=phi_dict,
            components_dict=components_dict,
            occurrence_dict=occurrence_dict,
            multisteps=multisteps,
            lb=lb,
            ub=ub,
            maxtime=maxtime,
            print_level=print_level,
            maxeval=maxeval,
            return_matrices=return_matrices,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            algorithm=algorithm,
            smoother=smoother,
        )

    # Step 1: Set up model structure
    # Simple call of the architector - also creates adam_cpp object
    model_type_dict, components_dict, lags_dict, observations_dict, profile_dict, adam_cpp = (
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
        smoother=smoother,
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
    # Convert algorithm string to nlopt constant
    nlopt_algorithm = getattr(nlopt, algorithm.replace("NLOPT_", ""), nlopt.LN_NELDERMEAD)
    opt = nlopt.opt(nlopt_algorithm, len(B))
    opt = _configure_optimizer(
        opt, lb, ub, maxeval_used, maxtime,
        xtol_rel=xtol_rel, xtol_abs=xtol_abs, ftol_rel=ftol_rel, ftol_abs=ftol_abs
    )

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
        adam_cpp,
        print_level
    )

    # Set objective function
    opt.set_min_objective(objective_wrapper)

    # Step 9: Run optimization
    B[:] = _run_optimization(opt, B)

    # Step 10: Extract the solution and the loss value
    CF_value = opt.last_optimum_value()
    
    # Step 10a: Retry optimization with zero smoothing parameters if initial optimization failed
    # This matches R's behavior (lines 2717-2768 in adam.R)
    # R checks for is.infinite(res$objective) || res$objective==1e+300
    # Python's objective wrapper caps at 1e10, so we check >= 1e10
    if not np.isfinite(CF_value) or CF_value >= 1e10:
        # Calculate number of ETS persistence parameters (alpha, beta, gamma)
        components_number_ets = 0
        if model_type_dict["ets_model"]:
            # Build persistence estimate vector with proper seasonal expansion
            persistence_estimate_vector = [
                persistence_dict['persistence_level_estimate'],
                model_type_dict["model_is_trendy"] and persistence_dict['persistence_trend_estimate'],
            ]
            if model_type_dict["model_is_seasonal"]:
                persistence_estimate_vector.extend(persistence_dict['persistence_seasonal_estimate'])
            components_number_ets = sum(persistence_estimate_vector)
            if components_number_ets > 0:
                B[:components_number_ets] = 0
        
        if arima_dict["arima_model"]:
            # Calculate starting index for ARIMA parameters
            # Match R's calculation exactly: componentsNumberETS + persistenceXregEstimate*xregNumber
            # Note: R's retry code doesn't account for phi, so we match that behavior
            ar_ma_start = components_number_ets
            if explanatory_dict['xreg_model'] and persistence_dict['persistence_xreg_estimate']:
                ar_ma_start += max(explanatory_dict['xreg_parameters_persistence'] or [0])
            
            # Calculate number of ARIMA parameters
            ar_orders = arima_dict.get('ar_orders', [])
            ma_orders = arima_dict.get('ma_orders', [])
            ar_estimate = arima_dict.get('ar_estimate', False)
            ma_estimate = arima_dict.get('ma_estimate', False)
            
            ar_count = sum(ar_orders) if ar_estimate else 0
            ma_count = sum(ma_orders) if ma_estimate else 0
            ar_ma_count = ar_count + ma_count
            
            if ar_ma_count > 0:
                B[ar_ma_start:ar_ma_start + ar_ma_count] = 0.01
        
        # Retry optimization with reset parameters
        opt2 = nlopt.opt(nlopt_algorithm, len(B))
        opt2 = _configure_optimizer(
            opt2, lb, ub, maxeval_used, maxtime,
            xtol_rel=xtol_rel, xtol_abs=xtol_abs, ftol_rel=ftol_rel, ftol_abs=ftol_abs
        )
        opt2.set_min_objective(objective_wrapper)
        B[:] = _run_optimization(opt2, B)
        CF_value = opt2.last_optimum_value()
    
    # Step 10: Calculate CF_value using optimized B
    # CF_value = CF(
    #     B=B,
    #     model_type_dict=model_type_dict,
    #     components_dict=components_dict,
    #     lags_dict=lags_dict,
    #     matrices_dict=adam_created,
    #     persistence_checked=persistence_dict,
    #     initials_checked=initials_dict,
    #     arima_checked=arima_dict,
    #     explanatory_checked=explanatory_dict,
    #     phi_dict=phi_dict,
    #     constants_checked=constant_dict,
    #     observations_dict=observations_dict,
    #     profile_dict=profile_dict,
    #     general=general_dict,
    #     adam_cpp=adam_cpp,
    #     bounds="usual",
    # )

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
        adam_cpp,
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
        "adam_cpp": adam_cpp,  # Always return adam_cpp for forecasting
    }

    if return_matrices:
        # Ensure matrices are updated with final B values and backcasted states
        # The CF function uses copies, so we need to update the originals

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

        # Run adam_cpp.fit() with backcasting to update states
        if initials_dict["initial_type"] in ["complete", "backcasting"]:
            mat_vt = np.asfortranarray(adam_created["mat_vt"], dtype=np.float64)
            mat_wt = np.asfortranarray(adam_created["mat_wt"], dtype=np.float64)
            mat_f = np.asfortranarray(adam_created["mat_f"], dtype=np.float64)
            vec_g = np.asfortranarray(adam_created["vec_g"], dtype=np.float64)
            index_lookup_table = np.asfortranarray(profile_dict["index_lookup_table"], dtype=np.uint64)
            profiles_recent_table = np.asfortranarray(profile_dict["profiles_recent_table"], dtype=np.float64)
            y_in_sample = np.asfortranarray(observations_dict["y_in_sample"], dtype=np.float64)
            ot = np.asfortranarray(observations_dict["ot"], dtype=np.float64)

            adam_cpp.fit(
                matrixVt=mat_vt,
                matrixWt=mat_wt,
                matrixF=mat_f,
                vectorG=vec_g,
                indexLookupTable=index_lookup_table,
                profilesRecent=profiles_recent_table,
                vectorYt=y_in_sample,
                vectorOt=ot,
                backcast=True,
                nIterations=initials_dict.get("n_iterations", 2) or 2,
                refineHead=True,  # Always True (fixed backcasting issue)
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
        if model_type_dict["error_type"] == "X":
            pool_errors = ["A"]
            pool_errors_small = "A"
        elif model_type_dict["error_type"] == "Y":
            pool_errors = ["M"]
            pool_errors_small = "M"
        else:
            pool_errors = [model_type_dict["error_type"]]
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
    # For 3-character models, the seasonal is at index 2 (model[2])
    # For 4-character models, the seasonal is the last character (model[-1])
    # Since pool_small only contains 3-character models at this stage, use model[-1] for safety
    if any(model[-1] == "M" for model in pool_small) and model_type_dict[
        "error_type"
    ] not in ["A", "X"]:
        for i, model in enumerate(pool_small):
            if model[-1] == "M":
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
    # NLopt parameters
    print_level=0,
    xtol_rel=1e-6,
    xtol_abs=1e-8,
    ftol_rel=1e-8,
    ftol_abs=0,
    algorithm="NLOPT_LN_NELDERMEAD",
    smoother="lowess",
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
        print_level=print_level,
        xtol_rel=xtol_rel,
        xtol_abs=xtol_abs,
        ftol_rel=ftol_rel,
        ftol_abs=ftol_abs,
        algorithm=algorithm,
        smoother=smoother,
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
    # NLopt parameters
    print_level=0,
    xtol_rel=1e-6,
    xtol_abs=1e-8,
    ftol_rel=1e-8,
    ftol_abs=0,
    algorithm="NLOPT_LN_NELDERMEAD",
    smoother="lowess",
    silent=False,
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
        (results, models_tested, pool_seasonals, pool_trends)
        - results: list of estimation results for tested models
        - models_tested: list of model strings that were tested
        - pool_seasonals: updated list of seasonal components to include in final pool
        - pool_trends: updated list of trend components to include in final pool
    """
    # Helper function to get seasonal type from model string (last character)
    def get_seasonal(model_str):
        return model_str[-1]

    # Helper function to get trend type from model string (second character)
    def get_trend(model_str):
        return model_str[1]

    # Helper function to find model index in pool_small
    def find_model_index(pool, seasonal=None, trend=None, error=None, exclude=None):
        """Find first model matching criteria, returns 1-indexed position or None."""
        if exclude is None:
            exclude = []
        for k, model in enumerate(pool):
            if model in exclude:
                continue
            if seasonal is not None and get_seasonal(model) != seasonal:
                continue
            if trend is not None and get_trend(model) != trend:
                continue
            if error is not None and model[0] != error:
                continue
            return k + 1  # 1-indexed
        return None

    # Helper function to estimate a model and store results
    def estimate_and_store(model_str, results_list, results_dict):
        """Estimate a model if not already tested, return results index."""
        if model_str in results_dict:
            return results_dict[model_str]

        idx = len([r for r in results_list if r is not None])

        model_type_dict_temp = model_type_dict.copy()
        model_type_dict_temp["model"] = model_str
        phi_dict_temp = phi_dict.copy()

        e_type = model_str[0]
        t_type = model_str[1]
        s_type = model_str[-1]

        model_type_dict_temp["error_type"] = e_type
        model_type_dict_temp["trend_type"] = t_type
        model_type_dict_temp["season_type"] = s_type

        if len(model_str) == 4:
            phi_dict_temp["phi"] = 0.95
            phi_dict_temp["phi_estimate"] = True
        else:
            phi_dict_temp["phi"] = 1
            phi_dict_temp["phi_estimate"] = False

        result = _estimate_model(
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
            print_level=print_level,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            algorithm=algorithm,
            smoother=smoother,
        )

        result["Etype"] = e_type
        result["Ttype"] = t_type
        result["Stype"] = s_type
        result["model"] = model_str

        if phi_dict_temp["phi_estimate"]:
            result["phi_dict"]["phi"] = result["adam_estimated"]["B"].get("phi", 0.95)
        else:
            result["phi_dict"]["phi"] = 1

        results_list.append(result)
        results_dict[model_str] = idx
        return idx

    # Ensure pool_seasonals and pool_trends are lists
    if isinstance(pool_seasonals, str):
        pool_seasonals = [pool_seasonals]
    if isinstance(pool_trends, str):
        pool_trends = [pool_trends]

    # Make copies to avoid modifying the original lists
    pool_seasonals = list(pool_seasonals)
    pool_trends = list(pool_trends)

    # Initialize
    results = []
    results_dict = {}  # model_str -> index in results
    models_tested = []

    # Branch and bound algorithm
    # Step 1: Estimate baseline model (ANN or equivalent)
    baseline_model = pool_small[0]  # Typically "ANN"
    baseline_idx = estimate_and_store(baseline_model, results, results_dict)
    models_tested.append(baseline_model)
    best_ic = results[baseline_idx]["IC"]
    best_model = baseline_model

    # Step 2: Check seasonality (if check_seasonal is True)
    if check_seasonal and len(pool_seasonals) > 1:
        # Find model with additive seasonality and same trend as baseline
        seasonal_model_a = find_model_index(
            pool_small, seasonal="A", trend=get_trend(baseline_model), exclude=models_tested
        )

        if seasonal_model_a is not None:
            model_a = pool_small[seasonal_model_a - 1]
            idx_a = estimate_and_store(model_a, results, results_dict)
            models_tested.append(model_a)
            ic_a = results[idx_a]["IC"]

            if ic_a < best_ic:
                # Seasonality helps - remove "N" from pool, check multiplicative
                pool_seasonals = [s for s in pool_seasonals if s != "N"]
                best_ic = ic_a
                best_model = model_a

                # Check multiplicative seasonality if available
                if "M" in pool_seasonals:
                    seasonal_model_m = find_model_index(
                        pool_small, seasonal="M", trend=get_trend(baseline_model), exclude=models_tested
                    )
                    if seasonal_model_m is not None:
                        model_m = pool_small[seasonal_model_m - 1]
                        idx_m = estimate_and_store(model_m, results, results_dict)
                        models_tested.append(model_m)
                        ic_m = results[idx_m]["IC"]

                        if ic_m < best_ic:
                            # Multiplicative is better
                            pool_seasonals = ["M"]
                            best_ic = ic_m
                            best_model = model_m
                        else:
                            # Additive is better
                            pool_seasonals = ["A"]
                else:
                    pool_seasonals = ["A"]

                # Now check trend with the selected seasonal
                if check_trend and len(pool_trends) > 1:
                    trend_model = find_model_index(
                        pool_small, seasonal=pool_seasonals[0], trend="A", exclude=models_tested
                    )
                    if trend_model is not None:
                        model_t = pool_small[trend_model - 1]
                        idx_t = estimate_and_store(model_t, results, results_dict)
                        models_tested.append(model_t)
                        ic_t = results[idx_t]["IC"]

                        if ic_t < best_ic:
                            # Trend helps - keep all trend options
                            pool_trends = [t for t in pool_trends if t != "N"]
                        else:
                            # No trend needed
                            pool_trends = ["N"]
            else:
                # No seasonality - pool_seasonals = ["N"]
                pool_seasonals = ["N"]

                # Check trend: AAN vs ANN
                if check_trend and len(pool_trends) > 1:
                    trend_model = find_model_index(
                        pool_small, seasonal="N", trend="A", exclude=models_tested
                    )
                    if trend_model is not None:
                        model_t = pool_small[trend_model - 1]
                        idx_t = estimate_and_store(model_t, results, results_dict)
                        models_tested.append(model_t)
                        ic_t = results[idx_t]["IC"]

                        if ic_t < best_ic:
                            # Trend helps - keep all trend options
                            pool_trends = [t for t in pool_trends if t != "N"]
                            best_ic = ic_t
                            best_model = model_t
                        else:
                            # No trend helps - only check MNN for error type
                            pool_trends = ["N"]

                            # Check MNN if multiplicative error is allowed
                            if model_type_dict.get("allow_multiplicative", True):
                                error_model = find_model_index(
                                    pool_small, seasonal="N", trend="N", error="M", exclude=models_tested
                                )
                                if error_model is not None:
                                    model_e = pool_small[error_model - 1]
                                    idx_e = estimate_and_store(model_e, results, results_dict)
                                    models_tested.append(model_e)
                                    # IC comparison for error type will be done in full pool estimation
        else:
            # No seasonal model to test, keep checking trend
            pool_seasonals = ["N"]
    elif not check_seasonal:
        # Seasonality already determined, check trend
        if check_trend and len(pool_trends) > 1:
            trend_model = find_model_index(
                pool_small, seasonal=pool_seasonals[0], trend="A", exclude=models_tested
            )
            if trend_model is not None:
                model_t = pool_small[trend_model - 1]
                idx_t = estimate_and_store(model_t, results, results_dict)
                models_tested.append(model_t)
                ic_t = results[idx_t]["IC"]

                if ic_t < best_ic:
                    pool_trends = [t for t in pool_trends if t != "N"]
                else:
                    pool_trends = ["N"]

    return results, models_tested, pool_seasonals, pool_trends


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
    # NLopt parameters
    print_level=0,
    xtol_rel=1e-6,
    xtol_abs=1e-8,
    ftol_rel=1e-8,
    ftol_abs=0,
    algorithm="NLOPT_LN_NELDERMEAD",
    # Pre-computed results from branch-and-bound
    precomputed_results=None,
    precomputed_models=None,
    smoother="lowess",
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
    precomputed_results : list, optional
        Results from branch-and-bound for models already estimated
    precomputed_models : list, optional
        List of model strings already estimated in branch-and-bound

    Returns
    -------
    list
        List of results for each model
    """
    # Build a dictionary of precomputed results
    precomputed_dict = {}
    if precomputed_results is not None and precomputed_models is not None:
        for model_str, result in zip(precomputed_models, precomputed_results):
            if result is not None:
                precomputed_dict[model_str] = result

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

        # Check if this model was already estimated in branch-and-bound
        if model_current in precomputed_dict:
            results[j] = precomputed_dict[model_current]
            continue

        # Create copies for this model
        model_type_dict_temp = model_type_dict.copy()
        phi_dict_temp = phi_dict.copy()

        # Set model parameters
        model_type_dict_temp["error_type"] = model_current[0]
        model_type_dict_temp["trend_type"] = model_current[1]

        if len(model_current) == 4:
            # 4-character model means damped (e.g., "AAdN")
            phi_dict_temp["phi"] = 0.95
            model_type_dict_temp["season_type"] = model_current[3]
            model_type_dict_temp["damped"] = True
            phi_dict_temp["phi_estimate"] = True
        else:
            # 3-character model means not damped (e.g., "AAN")
            phi_dict_temp["phi"] = 1
            model_type_dict_temp["season_type"] = model_current[2]
            model_type_dict_temp["damped"] = False
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
                print_level=print_level,
                xtol_rel=xtol_rel,
                xtol_abs=xtol_abs,
                ftol_rel=ftol_rel,
                ftol_abs=ftol_abs,
                algorithm=algorithm,
                smoother=smoother,
            )
        results[j]["IC"] = ic_function(general_dict['ic'], loglik=results[j]['adam_estimated']["log_lik_adam_value"])
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
    # NLopt parameters
    print_level=0,
    xtol_rel=1e-6,
    xtol_abs=1e-8,
    ftol_rel=1e-8,
    ftol_abs=0,
    algorithm="NLOPT_LN_NELDERMEAD",
    smoother="lowess",
):
    """
    Automatic model selection for ADAM using information criteria and Branch & Bound.

    This function implements the automatic model selection procedure for ADAM models
    by creating a pool of candidate models and selecting the best one based on an
    information criterion (AIC, AICc, BIC, or BICc). The selection process uses a
    **Branch and Bound algorithm** to efficiently explore the model space without
    estimating every possible combination.

    The function supports several selection modes triggered by special codes in the
    model specification string:

    - **"ZZZ"**: Select from all possible models using Branch and Bound
    - **"XXX"**: Select only additive components (A, N)
    - **"YYY"**: Select only multiplicative components (M, N)
    - **"PPP"**: Select between pure additive (AAA) and pure multiplicative (MMM)
    - **"FFF"**: Full search across all 30 ETS model types
    - **"CCC"**: Combine models using IC weights (not fully implemented in Python yet)
    - Mixed codes like **"ZXZ"**: Auto-select error and seasonality, only additive trend

    **Selection Algorithm**:

    1. **Pool Formation**: Create initial model pool based on specification:

       - Identify which components (error, trend, seasonality) need selection
       - Generate small pool for Branch and Bound exploration
       - Generate full pool for final estimation

    2. **Branch and Bound** (if applicable):

       - Start with simplest model (typically "ANN")
       - Iteratively explore promising branches (add/change components)
       - Prune branches that cannot improve upon best IC found
       - Significantly faster than exhaustive search

    3. **Full Estimation**: Estimate all models in the final pool

    4. **Selection**: Choose model with best (lowest) information criterion

    **Model Pool Generation**:

    The pool size depends on the specification:

    - **"ZZZ"**: Up to 30 models (5 errors × 3 trends × 2 seasonalities)
    - **"XXX"**: Up to 6 models (additive-only: A × {N,A,Ad} × {N,A})
    - **"YYY"**: Up to 10 models (multiplicative-focused)
    - **"FFF"**: All 30 models estimated exhaustively
    - Custom pool: Use provided list (e.g., ["ANN", "AAN", "AAA"])

    Parameters
    ----------
    model_type_dict : dict
        Model specification containing:

        - 'model': Model string with selection codes (e.g., "ZXZ", "FFF")
        - 'error_type': Error component ('Z' for auto-select, or 'A'/'M' for fixed)
        - 'trend_type': Trend component ('Z', 'X', 'Y', or specific like 'A', 'Ad', 'N')
        - 'season_type': Seasonality component ('Z', 'X', 'Y', or 'N', 'A', 'M')
        - 'models_pool': List of model strings (if pre-specified pool)
        - 'allow_multiplicative': Whether multiplicative models are allowed (data-dependent)
        - 'model_do': Action type ('select', 'combine', or 'estimate')

    phi_dict : dict
        Damping parameter specification containing:

        - 'phi_estimate': Whether to estimate damping
        - 'phi': Fixed damping value (if not estimated)

    general_dict : dict
        General configuration containing:

        - 'loss': Loss function ('likelihood', 'MSE', 'MAE', etc.)
        - 'distribution': Error distribution
        - 'bounds': Parameter bounds type
        - 'h': Forecast horizon
        - 'holdout': Whether holdout validation is used

    lags_dict : dict
        Lag structure containing:

        - 'lags': Primary lag vector (e.g., [1, 12])
        - 'lags_model': Lags for each component
        - 'lags_model_max': Maximum lag

    observations_dict : dict
        Time series data containing:

        - 'y_in_sample': Observed values
        - 'obs_in_sample': Number of observations
        - 'ot': Occurrence vector (for intermittent data)

    arima_dict : dict
        ARIMA specification containing:

        - 'arima_model': Whether ARIMA components exist
        - 'ar_orders': AR orders
        - 'ma_orders': MA orders
        - 'i_orders': Integration orders

    constant_dict : dict
        Constant term specification containing:

        - 'constant_required': Whether constant is included
        - 'constant_estimate': Whether to estimate constant

    explanatory_dict : dict
        External regressors specification containing:

        - 'xreg_model': Whether regressors are present
        - 'xreg_number': Number of regressors

    occurrence_dict : dict
        Intermittent demand specification containing:

        - 'occurrence_model': Whether occurrence model is used
        - 'occurrence_type': Type of occurrence model

    components_dict : dict
        Component counts (passed through to estimator)

    profiles_recent_table : numpy.ndarray
        Recent profile values for time-varying parameters

    profiles_recent_provided : bool
        Whether user provided custom profiles

    persistence_results : dict
        Persistence parameters containing:

        - 'persistence_estimate': Whether to estimate smoothing parameters
        - Fixed values for non-estimated parameters

    initials_results : dict
        Initial states specification containing:

        - 'initial_type': Initialization method
        - Flags for which initials to estimate

    criterion : str, default="AICc"
        Information criterion for model selection:

        - **"AIC"**: Akaike Information Criterion (penalizes complexity moderately)
        - **"AICc"**: Corrected AIC (recommended for small samples, **default**)
        - **"BIC"**: Bayesian Information Criterion (more parsimonious than AIC)
        - **"BICc"**: Corrected BIC

        Lower IC values indicate better models. AICc is default as it performs
        well across sample sizes and is the standard in forecast package.

    silent : bool, default=False
        Whether to suppress progress messages during model estimation.
        If False, prints which models are being estimated.

    Returns
    -------
    dict
        Dictionary containing selection results:

        - **'results'** (list of dict): Estimation results for each model in pool.
          Each dict contains:

          * 'model': Model string (e.g., "ANN", "MAM")
          * 'IC': Information criterion value
          * 'loglik': Log-likelihood
          * 'n_param': Number of parameters
          * 'B': Estimated parameter vector
          * Additional estimation outputs

        - **'ic_selection'** (dict): Mapping of model names to IC values.
          Keys are model strings, values are IC scores. NaN values are replaced
          with 1e100 for comparison purposes.

    Notes
    -----
    **Branch and Bound Algorithm**:

    The Branch and Bound method is a heuristic that exploits the nested structure
    of ETS models. It works by:

    1. Starting with the simplest model (no trend, no seasonality)
    2. "Branching" by adding/changing one component at a time
    3. "Bounding" by not exploring branches worse than current best + tolerance

    This can reduce the search from 30 models to ~10-15 models in typical cases,
    with minimal risk of missing the global optimum.

    **Performance Considerations**:

    - Branch and Bound: Estimates ~10-15 models typically
    - Full pool ("FFF"): Estimates all 30 models (slower but exhaustive)
    - Custom pool: Estimates only specified models (fastest)

    Estimation time is roughly proportional to number of models × optimization time
    per model. For large datasets or complex models, consider using a smaller pool.

    **Multiplicative Model Restrictions**:

    Multiplicative error and seasonality require strictly positive data. If data
    contains zeros or negatives, `allow_multiplicative=False` is automatically set,
    restricting the pool to additive models only.

    **Model Combination (CCC)**:

    The "CCC" option for combining model forecasts using IC weights is mentioned in
    R documentation but not fully implemented in Python version yet. Use "ZZZ" for
    selection instead.

    **Equivalent to R's auto.adam()**:

    This function implements Python equivalent of R's `auto.adam()` from smooth package.

    See Also
    --------
    estimator : Estimates a single model (called by selector for each candidate)
    _form_model_pool : Creates the pool of candidate models
    _run_branch_and_bound : Implements Branch & Bound algorithm
    _estimate_all_models : Estimates all models in the pool

    Examples
    --------
    Automatic selection with default AICc::

        >>> results = selector(
        ...     model_type_dict={'model': 'ZZZ', 'error_type': 'Z',
        ...                      'trend_type': 'Z', 'season_type': 'Z',
        ...                      'models_pool': None, ...},
        ...     general_dict={'loss': 'likelihood', 'h': 10, ...},
        ...     lags_dict={'lags': np.array([1, 12]), ...},
        ...     observations_dict={'y_in_sample': data, ...},
        ...     criterion='AICc',
        ...     ...
        ... )
        >>> best_model = min(results['ic_selection'], key=results['ic_selection'].get)
        >>> print(f"Best model: {best_model}, AICc: {results['ic_selection'][best_model]}")

    Select from additive models only::

        >>> results = selector(
        ...     model_type_dict={'model': 'XXX', 'error_type': 'X',
        ...                      'trend_type': 'X', 'season_type': 'X', ...},
        ...     criterion='BIC',  # Use BIC for more parsimonious selection
        ...     ...
        ... )
        >>> # Will only consider: ANN, AAN, AAdN, ANA, AAA, AAdA

    Use custom pool::

        >>> results = selector(
        ...     model_type_dict={'model': 'ZZZ',
        ...                      'models_pool': ['ANN', 'AAN', 'AAdN', 'ANA', 'AAA']},
        ...     criterion='AICc',
        ...     ...
        ... )
        >>> # Estimates only the 5 specified models
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

    # Initialize variables for precomputed results
    bb_results = None
    bb_models_tested = None

    if model_type_dict["models_pool"] is None:
        # Run branch and bound to select models
        bb_results, bb_models_tested, pool_seasonals, pool_trends = _run_branch_and_bound(
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
            print_level=print_level,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            algorithm=algorithm,
            smoother=smoother,
            silent=silent,
        )

        # Prepare a bigger pool based on the small one
        # Ensure pool_seasonals and pool_trends are lists
        if isinstance(pool_seasonals, str):
            pool_seasonals = [pool_seasonals]
        if isinstance(pool_trends, str):
            pool_trends = [pool_trends]

        # Generate models in the same order as R:
        # paste0(rep(poolErrors, each=len(poolTrends)*len(poolSeasonals)),
        #        poolTrends,
        #        rep(poolSeasonals, each=len(poolTrends)))
        # This is: for e in errors: for s in seasonals: for t in trends
        generated_models = [
            e + t + s
            for e in pool_errors
            for s in pool_seasonals  # outer loop (slower varying)
            for t in pool_trends     # inner loop (faster varying)
        ]

        # Use dict.fromkeys() to preserve insertion order while removing duplicates
        # (mimics R's unique(c(...)) behavior)
        combined_models = bb_models_tested + generated_models
        model_type_dict["models_pool"] = list(dict.fromkeys(combined_models))

    # Step 3: Estimate all models in the pool (skip already tested models)
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
        print_level=print_level,
        xtol_rel=xtol_rel,
        xtol_abs=xtol_abs,
        ftol_rel=ftol_rel,
        ftol_abs=ftol_abs,
        algorithm=algorithm,
        precomputed_results=bb_results,
        precomputed_models=bb_models_tested,
        smoother=smoother,
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


# def _generate_forecasts(
#     general_dict,
#     observations_dict,
#     matrices_dict,
#     lags_dict,
#     profiles_dict,
#     model_type_dict,
#     components_dict,
#     explanatory_checked,
#     constants_checked,
# ):
#     """
#     Generate forecasts if the horizon is non-zero.
#
#     Parameters
#     ----------
#     general_dict : dict
#         General model parameters
#     observations_dict : dict
#         Observations information
#     matrices_dict : dict
#         Model matrices
#     lags_dict : dict
#         Lags information
#     profiles_dict : dict
#         Profiles information
#     model_type_dict : dict
#         Model type specification
#     components_dict : dict
#         Components information
#     explanatory_checked : dict
#         Explanatory variables specification
#     constants_checked : dict
#         Constant term specification
#
#     Returns
#     -------
#     pandas.Series
#         Forecasted values
#     """
#     # If horizon is zero, return an empty series
#     if general_dict["h"] <= 0:
#         if any(observations_dict.get("y_classes", []) == "ts"):
#             return pd.Series(
#                 [np.nan],
#                 index=pd.date_range(
#                     start=observations_dict["y_forecast_start"],
#                     periods=1,
#                     freq=observations_dict["y_frequency"],
#                 ),
#             )
#         else:
#             return pd.Series(
#                 np.full(general_dict["horizon"], np.nan),
#                 index=observations_dict["y_forecast_index"],
#             )
#
#     # Create forecast Series
#     if not isinstance(observations_dict.get("y_in_sample"), pd.Series):
#         y_forecast = pd.Series(
#             np.full(general_dict["h"], np.nan),
#             index=pd.date_range(
#                 start=observations_dict["y_forecast_start"],
#                 periods=general_dict["h"],
#                 freq=observations_dict["frequency"],
#             ),
#         )
#     else:
#         y_forecast = pd.Series(
#             np.full(general_dict["h"], np.nan),
#             index=observations_dict["y_forecast_index"],
#         )
#
#     # Generate forecasts
#     forecasts = adam_forecaster(
#         matrixWt=matrices_dict["mat_wt"][-general_dict["h"] :],
#         matrixF=matrices_dict["mat_f"],
#         lags=lags_dict["lags_model_all"],
#         indexLookupTable=profiles_dict["index_lookup_table"],
#         profilesRecent=profiles_dict["profiles_recent_table"],
#         E=model_type_dict["error_type"],
#         T=model_type_dict["trend_type"],
#         S=model_type_dict["season_type"],
#         nNonSeasonal=components_dict["components_number_ets"],
#         nSeasonal=components_dict["components_number_ets_seasonal"],
#         nArima=components_dict.get("components_number_arima", 0),
#         nXreg=explanatory_checked["xreg_number"],
#         constant=constants_checked["constant_required"],
#         horizon=general_dict["h"],
#     ).flatten()
#
#     # Fill in forecast values
#     y_forecast[:] = forecasts
#
#     # Replace NaNs with zeros
#     if np.any(np.isnan(y_forecast)):
#         y_forecast[np.isnan(y_forecast)] = 0
#
#     return y_forecast
#
#
# def _update_distribution(general_dict, model_type_dict):
#     """
#     Update distribution based on error term and loss function.
#
#     Parameters
#     ----------
#     general_dict : dict
#         General model parameters
#     model_type_dict : dict
#         Model type specification
#
#     Returns
#     -------
#     dict
#         Updated general_dict
#     """
#     general_dict_updated = general_dict.copy()
#
#     if general_dict["distribution"] == "default":
#         if general_dict["loss"] == "likelihood":
#             if model_type_dict["error_type"] == "A":
#                 general_dict_updated["distribution"] = "dnorm"
#             elif model_type_dict["error_type"] == "M":
#                 general_dict_updated["distribution"] = "dgamma"
#         elif general_dict["loss"] in ["MAEh", "MACE", "MAE"]:
#             general_dict_updated["distribution"] = "dlaplace"
#         elif general_dict["loss"] in ["HAMh", "CHAM", "HAM"]:
#             general_dict_updated["distribution"] = "ds"
#         elif general_dict["loss"] in ["MSEh", "MSCE", "MSE", "GPL"]:
#             general_dict_updated["distribution"] = "dnorm"
#         else:
#             general_dict_updated["distribution"] = "dnorm"
#
#     return general_dict_updated
#
#
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
#
#
# def _process_arma_parameters(B, arima_checked):
#     """
#     Process ARMA parameters from the estimates.
#
#     Parameters
#     ----------
#     B : dict
#         Parameter estimates
#     arima_checked : dict
#         ARIMA components specification
#
#     Returns
#     -------
#     dict or None
#         ARMA parameters list
#     """
#     if not arima_checked["arima_model"]:
#         return None
#
#     arma_parameters_list = {}
#
#     # Process AR parameters
#     if arima_checked["ar_required"] and arima_checked["ar_estimate"]:
#         # Avoid damping parameter phi by checking name length > 3
#         arma_parameters_list["ar"] = [
#             b for name, b in B.items() if len(name) > 3 and name.startswith("phi")
#         ]
#     elif arima_checked["ar_required"] and not arima_checked["ar_estimate"]:
#         # Avoid damping parameter phi
#         arma_parameters_list["ar"] = [
#             p
#             for name, p in arima_checked["arma_parameters"].items()
#             if name.startswith("phi")
#         ]
#
#     # Process MA parameters
#     if arima_checked["ma_required"] and arima_checked["ma_estimate"]:
#         arma_parameters_list["ma"] = [
#             b for name, b in B.items() if name.startswith("theta")
#         ]
#     elif arima_checked["ma_required"] and not arima_checked["ma_estimate"]:
#         arma_parameters_list["ma"] = [
#             p
#             for name, p in arima_checked["arma_parameters"].items()
#             if name.startswith("theta")
#         ]
#
#     return arma_parameters_list
#
#
# def _calculate_scale(
#     general_dict, model_type_dict, errors, y_fitted, observations_dict, other
# ):
#     """
#     Calculate scale parameter using scaler function.
#
#     Parameters
#     ----------
#     general_dict : dict
#         General model parameters
#     model_type_dict : dict
#         Model type specification
#     errors : pandas.Series
#         Model errors
#     y_fitted : pandas.Series
#         Fitted values
#     observations_dict : dict
#         Observations information
#     other : dict or None
#         Other parameters
#
#     Returns
#     -------
#     float
#         Scale parameter
#     """
#     return scaler(
#         general_dict["distribution_new"],
#         model_type_dict["error_type"],
#         errors[observations_dict["ot_logical"]],
#         y_fitted[observations_dict["ot_logical"]],
#         observations_dict["obs_in_sample"],
#         other,
#     )
#
#
# def _process_other_parameters(
#     constants_checked, adam_estimated, general_dict, arima_checked, lags_dict
# ):
#     """
#     Process other parameters including constants and ARIMA polynomials.
#
#     Parameters
#     ----------
#     constants_checked : dict
#         Constant term specification
#     adam_estimated : dict
#         Estimated model parameters
#     general_dict : dict
#         General model parameters
#     arima_checked : dict
#         ARIMA components specification
#     lags_dict : dict
#         Lags information
#
#     Returns
#     -------
#     tuple
#         constant_value, other_returned
#     """
#     # Record constant if estimated
#     if constants_checked["constant_estimate"]:
#         constant_value = adam_estimated["B"][constants_checked["constant_name"]]
#     else:
#         constant_value = None
#
#     # Prepare distribution parameters to return
#     other_returned = {}
#
#     # Add LASSO/RIDGE lambda if applicable
#     if general_dict["loss"] in ["LASSO", "RIDGE"]:
#         other_returned["lambda"] = general_dict["lambda_"]
#
#     # Return ARIMA polynomials and indices for persistence and transition
#     if arima_checked["arima_model"]:
#         other_returned["polynomial"] = adam_estimated["arima_polynomials"]
#         other_returned["ARIMA_indices"] = {
#             "nonZeroARI": arima_checked["non_zero_ari"],
#             "nonZeroMA": arima_checked["non_zero_ma"],
#         }
#         other_returned["ar_polynomial_matrix"] = np.zeros(
#             (
#                 sum(arima_checked["ar_orders"]) * lags_dict["lags"],
#                 sum(arima_checked["ar_orders"]) * lags_dict["lags"],
#             )
#         )
#
#         if other_returned["ar_polynomial_matrix"].shape[0] > 1:
#             # Set diagonal elements to 1 except first row/col
#             other_returned["ar_polynomial_matrix"][1:-1, 2:] = np.eye(
#                 other_returned["ar_polynomial_matrix"].shape[0] - 2
#             )
#
#             if arima_checked["ar_required"]:
#                 other_returned["ar_polynomial_matrix"][:, 0] = -arima_polynomials[
#                     "ar_polynomial"
#                 ][1:]
#
#         other_returned["arma_parameters"] = arima_checked["arma_parameters"]
#
#     return constant_value, other_returned
#
#
# def _format_output(
#     model_type_dict,
#     y_fitted,
#     errors,
#     y_forecast,
#     matrices_dict,
#     profiles_dict,
#     persistence,
#     phi_dict,
#     initial_value,
#     initials_checked,
#     initial_estimated,
#     general_dict,
#     arma_parameters_list,
#     constant_value,
#     occurrence_dict,
#     explanatory_checked,
#     scale,
#     other_returned,
#     adam_estimated,
#     lags_dict,
#     observations_dict,
# ):
#     """
#     Format the final output for the model.
#
#     Parameters
#     ----------
#     model_type_dict : dict
#         Model type specification
#     y_fitted : pandas.Series
#         Fitted values
#     errors : pandas.Series
#         Model errors
#     y_forecast : pandas.Series
#         Forecasted values
#     matrices_dict : dict
#         Model matrices
#     profiles_dict : dict
#         Profiles information
#     persistence : array
#         Persistence vector
#     phi_dict : dict
#         Damping parameter information
#     initial_value : dict
#         Initial values
#     initials_checked : dict
#         Initial values specification
#     initial_estimated : list
#         Which initials were estimated
#     general_dict : dict
#         General model parameters
#     arma_parameters_list : dict or None
#         ARMA parameters
#     constant_value : float or None
#         Constant term value
#     occurrence_dict : dict
#         Occurrence model information
#     explanatory_checked : dict
#         Explanatory variables specification
#     scale : float
#         Scale parameter
#     other_returned : dict
#         Other parameters
#     adam_estimated : dict
#         Estimated model parameters
#     lags_dict : dict
#         Lags information
#     observations_dict : dict
#         Observations information
#
#     Returns
#     -------
#     dict
#         Formatted model output
#     """
#     # Amend the class of state matrix
#     if not isinstance(observations_dict.get("y_in_sample"), pd.Series):
#         mat_vt = pd.Series(
#             matrices_dict["mat_vt"].T.flatten(),
#             index=pd.date_range(
#                 start=observations_dict["y_forecast_start"],
#                 periods=len(matrices_dict["mat_vt"].T),
#                 freq=observations_dict["frequency"],
#             ),
#         )
#     else:
#         mat_vt = pd.Series(
#             matrices_dict["mat_vt"].T, index=observations_dict["y_forecast_index"]
#         )
#
#     # Update parameters number
#     general_dict["parameters_number"][0][2] = np.sum(
#         general_dict["parameters_number"][0][:2]
#     )
#
#     # Return the formatted output
#     return {
#         "model": model_type_dict["model"],
#         "time_elapsed": None,  # here will count the time
#         "holdout": general_dict["holdout"],
#         "fitted": y_fitted,
#         "residuals": errors,
#         "forecast": y_forecast,
#         "states": mat_vt,
#         "profile": profiles_dict["profiles_recent_table"],
#         "profile_initial": profiles_dict["profiles_recent_initial"],
#         "persistence": persistence,
#         "phi": phi_dict["phi"],
#         "transition": matrices_dict["mat_f"],
#         "measurement": matrices_dict["mat_wt"],
#         "initial": initial_value,
#         "initial_type": initials_checked["initial_type"],
#         "initial_estimated": initial_estimated,
#         "orders": general_dict.get("orders"),
#         "arma": arma_parameters_list,
#         "constant": constant_value,
#         "n_param": general_dict["parameters_number"],
#         "occurrence": occurrence_dict["oes_model"],
#         "formula": explanatory_checked.get("formula"),
#         "regressors": explanatory_checked.get("regressors"),
#         "loss": general_dict["loss"],
#         "loss_value": adam_estimated["CF_value"],
#         "log_lik": adam_estimated["log_lik_adam_value"],
#         "distribution": general_dict["distribution"],
#         "scale": scale,
#         "other": other_returned,
#         "B": adam_estimated["B"],
#         "lags": lags_dict["lags"],
#         "lags_all": lags_dict["lags_model_all"],
#         "FI": general_dict.get("fi"),
#     }


# def preparator(
#     # Model type info
#     model_type_dict,
#     # Components info
#     components_dict,
#     # Lags info
#     lags_dict,
#     # Matrices from creator
#     matrices_dict,
#     # Parameter dictionaries
#     persistence_checked,
#     initials_checked,
#     arima_checked,
#     explanatory_checked,
#     phi_dict,
#     constants_checked,
#     # Other parameters
#     observations_dict,
#     occurrence_dict,
#     general_dict,
#     profiles_dict,
#     # The parameter vector
#     adam_estimated,
#     # Optional parameters
#     bounds="usual",
#     other=None,
# ):
#     """
#     Prepare final model output after estimation.
#
#     This function takes the estimated model parameters and prepares
#     the fitted values, forecasts, and other model components for output.
#
#     Parameters
#     ----------
#     model_type_dict : dict
#         Model type specification parameters
#     components_dict : dict
#         Model components information
#     lags_dict : dict
#         Information about model lags
#     matrices_dict : dict
#         Model matrices from creator
#     persistence_checked : dict
#         Processed persistence parameters
#     initials_checked : dict
#         Processed initial values
#     arima_checked : dict
#         Processed ARIMA component information
#     explanatory_checked : dict
#         Processed external regressors information
#     phi_dict : dict
#         Damping parameter information
#     constants_checked : dict
#         Processed constant term information
#     observations_dict : dict
#         Observed data information
#     occurrence_dict : dict
#         Occurrence model information
#     general_dict : dict
#         General model configuration parameters
#     profiles_dict : dict
#         Profiles information
#     adam_estimated : dict
#         Parameter estimates from optimization
#     bounds : str, optional
#         Type of bounds used
#     other : dict, optional
#         Additional parameters
#
#     Returns
#     -------
#     dict
#         Dictionary containing fitted model and forecasts
#     """
#     # Step 1: Fill in the matrices with estimated parameters
#     matrices_dict = _fill_matrices(
#         adam_estimated=adam_estimated,
#         model_type_dict=model_type_dict,
#         components_dict=components_dict,
#         lags_dict=lags_dict,
#         matrices_dict=matrices_dict,
#         persistence_checked=persistence_checked,
#         initials_checked=initials_checked,
#         arima_checked=arima_checked,
#         explanatory_checked=explanatory_checked,
#         phi_dict=phi_dict,
#         constants_checked=constants_checked,
#     )
#
#     # Step 2: Update profiles and phi
#     profiles_dict, phi_dict = _update_profiles(
#         matrices_dict=matrices_dict,
#         profiles_dict=profiles_dict,
#         lags_dict=lags_dict,
#         phi_dict=phi_dict,
#         adam_estimated=adam_estimated,
#     )
#
#     # Step 3: Prepare arrays for model fitting
#     (
#         y_in_sample,
#         ot,
#         mat_vt,
#         mat_wt,
#         mat_f,
#         vec_g,
#         lags_model_all,
#         index_lookup_table,
#         profiles_recent_table,
#     ) = _prepare_arrays(
#         matrices_dict=matrices_dict,
#         observations_dict=observations_dict,
#         profiles_dict=profiles_dict,
#         lags_dict=lags_dict,
#     )
#
#     # Step 4: Fit the model
#     adam_fitted = _fit_model(
#         mat_vt=mat_vt,
#         mat_wt=mat_wt,
#         mat_f=mat_f,
#         vec_g=vec_g,
#         lags_model_all=lags_model_all,
#         index_lookup_table=index_lookup_table,
#         profiles_recent_table=profiles_recent_table,
#         model_type_dict=model_type_dict,
#         components_dict=components_dict,
#         explanatory_checked=explanatory_checked,
#         constants_checked=constants_checked,
#         y_in_sample=y_in_sample,
#         ot=ot,
#         initials_checked=initials_checked,
#     )
#
#     # Step 5: Update matrices with fitted values
#     matrices_dict, profiles_dict = _update_matrices(
#         matrices_dict=matrices_dict,
#         profiles_dict=profiles_dict,
#         adam_fitted=adam_fitted,
#         model_type_dict=model_type_dict,
#         components_dict=components_dict,
#     )
#
#     # Step 6: Prepare fitted values and errors
#     y_fitted, errors = _prepare_fitted_values(
#         observations_dict=observations_dict,
#         adam_fitted=adam_fitted,
#         occurrence_dict=occurrence_dict,
#     )
#
#     # Step 7: Generate forecasts
#     y_forecast = _generate_forecasts(
#         general_dict=general_dict,
#         observations_dict=observations_dict,
#         matrices_dict=matrices_dict,
#         lags_dict=lags_dict,
#         profiles_dict=profiles_dict,
#         model_type_dict=model_type_dict,
#         components_dict=components_dict,
#         explanatory_checked=explanatory_checked,
#         constants_checked=constants_checked,
#     )
#
#     # Step 8: Update distribution
#     general_dict = _update_distribution(
#         general_dict=general_dict, model_type_dict=model_type_dict
#     )
#
#     # Step 9: Process initial values
#     initial_value, initial_value_ets, initial_value_names, initial_estimated = (
#         _process_initial_values(
#             model_type_dict=model_type_dict,
#             lags_dict=lags_dict,
#             matrices_dict=matrices_dict,
#             initials_checked=initials_checked,
#             arima_checked=arima_checked,
#             explanatory_checked=explanatory_checked,
#             components_dict=components_dict,
#         )
#     )
#
#     # Step 10: Get persistence values
#     persistence = np.array(matrices_dict["vec_g"]).flatten()
#
#     # Step 11: Handle explanatory variables
#     if (
#         explanatory_checked["xreg_model"]
#         and explanatory_checked.get("regressors") != "adapt"
#     ):
#         explanatory_checked["regressors"] = "use"
#     elif not explanatory_checked["xreg_model"]:
#         explanatory_checked["regressors"] = None
#
#     # Step 12: Process ARMA parameters
#     arma_parameters_list = _process_arma_parameters(
#         B=adam_estimated["B"], arima_checked=arima_checked
#     )
#
#     # Step 13: Calculate scale parameter
#     scale = _calculate_scale(
#         general_dict=general_dict,
#         model_type_dict=model_type_dict,
#         errors=errors,
#         y_fitted=y_fitted,
#         observations_dict=observations_dict,
#         other=other,
#     )
#
#     # Step 14: Process other parameters
#     constant_value, other_returned = _process_other_parameters(
#         constants_checked=constants_checked,
#         adam_estimated=adam_estimated,
#         general_dict=general_dict,
#         arima_checked=arima_checked,
#         lags_dict=lags_dict,
#     )
#
#     # Step 15: Format and return final output
#     return _format_output(
#         model_type_dict=model_type_dict,
#         y_fitted=y_fitted,
#         errors=errors,
#         y_forecast=y_forecast,
#         matrices_dict=matrices_dict,
#         profiles_dict=profiles_dict,
#         persistence=persistence,
#         phi_dict=phi_dict,
#         initial_value=initial_value,
#         initials_checked=initials_checked,
#         initial_estimated=initial_estimated,
#         general_dict=general_dict,
#         arma_parameters_list=arma_parameters_list,
#         constant_value=constant_value,
#         occurrence_dict=occurrence_dict,
#         explanatory_checked=explanatory_checked,
#         scale=scale,
#         other_returned=other_returned,
#         adam_estimated=adam_estimated,
#         lags_dict=lags_dict,
#         observations_dict=observations_dict,
#     )
