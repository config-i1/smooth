import numpy as np
from scipy import stats
from scipy.special import gamma

from smooth.adam_general.core.utils.distributions import (
    generate_errors,
    normalize_errors,
)
from smooth.adam_general.core.utils.var_covar import (
    covar_anal,
    sigma,
    var_anal,
)

from ._helpers import _prepare_lookup_table, _prepare_matrices_for_forecast


def ensure_level_format(level, side):
    # Fix just in case user used 95 etc instead of 0.95
    level = level / 100 if level > 1 else level

    # Handle different interval sides
    if side == "both":
        level_low = round((1 - level) / 2, 3)
        level_up = round((1 + level) / 2, 3)

    elif side == "upper":
        # level_low = np.zeros_like(level)
        level_up = level
    else:
        level_low = 1 - level

    return level_low, level_up


def generate_prediction_interval(
    predictions,
    prepared_model,
    general,
    observations_dict,
    model_type_dict,
    lags_dict,
    params_info,
    level,
):
    mat_vt, mat_wt, vec_g, mat_f = _prepare_matrices_for_forecast(
        prepared_model, observations_dict, lags_dict, general
    )

    # stimate sigma
    s2 = sigma(observations_dict, params_info, general, prepared_model) ** 2

    # lines 8015 to 8022
    # line 8404 -> I dont get the (is.scale(object$scale))
    # Skipping for now.
    # Will ask Ivan what this is

    # Check if model is ETS and has certain distributions with multiplicative errors
    if (
        model_type_dict["ets_model"]
        and general["distribution"]
        in ["dinvgauss", "dgamma", "dlnorm", "dllaplace", "dls", "dlgnorm"]
        and model_type_dict["error_type"] == "M"
    ):
        # again scale object
        # lines 8425 8428

        v_voc_multi = var_anal(
            lags_dict["lags_model_all"], general["h"], mat_wt[0], mat_f, vec_g, s2
        )

        # Lines 8429-8433 in R/adam.R
        # If distribution is one of the log-based ones, transform the variance
        if general["distribution"] in ["dlnorm", "dls", "dllaplace", "dlgnorm"]:
            v_voc_multi = np.log(1 + v_voc_multi)

        # Lines 8435-8437 in R/adam.R
        # We don't do correct cumulatives in this case...
        if general.get("cumulative", False):
            v_voc_multi = np.sum(v_voc_multi)
    else:
        # Lines 8439-8441 in R/adam.R
        v_voc_multi = covar_anal(
            lags_dict["lags_model_all"], general["h"], mat_wt, mat_f, vec_g, s2
        )

        # Skipping the is.scale check (lines 8442-8445)

        # Lines 8447-8453 in R/adam.R
        # Do either the variance of sum, or a diagonal
        if general.get("cumulative", False):
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
    e_type = model_type_dict["error_type"]  # "A" or "M"

    distribution = general["distribution"]
    other_params = general.get(
        "other", {}
    )  # Handle cases where 'other' might be missing

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
        scale = (v_voc_multi / 120) ** 0.25
        loc = 1 if e_type == "M" else 0
        try:
            # Check if stats.s_dist exists before calling
            if hasattr(stats, "s_dist") and hasattr(stats.s_dist, "ppf"):
                y_lower[:] = stats.s_dist.ppf(level_low, loc=loc, scale=scale)
                y_upper[:] = stats.s_dist.ppf(level_up, loc=loc, scale=scale)
            else:
                print(
                    "Warning: stats.s_dist not found. "
                    "Cannot calculate intervals for 'ds'."
                )
                y_lower[:], y_upper[:] = np.nan, np.nan
        except Exception as e:
            print(f"Error calculating 'ds' interval: {e}")
            y_lower[:], y_upper[:] = np.nan, np.nan

    elif distribution == "dgnorm":
        # stats.gennorm.ppf(q, beta, loc=0, scale=1)
        shape_beta = other_params.get("shape")
        if shape_beta is not None:
            # Handle potential division by zero or issues with gamma function
            # if shape is invalid
            try:
                scale = np.sqrt(
                    v_voc_multi * (gamma(1 / shape_beta) / gamma(3 / shape_beta))
                )
                loc = 1 if e_type == "M" else 0
                y_lower[:] = stats.gennorm.ppf(
                    level_low, beta=shape_beta, loc=loc, scale=scale
                )
                y_upper[:] = stats.gennorm.ppf(
                    level_up, beta=shape_beta, loc=loc, scale=scale
                )
            except (ValueError, ZeroDivisionError) as e:
                print(
                    f"Warning: Could not calculate scale for dgnorm "
                    f"(shape={shape_beta}). Error: {e}"
                )
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
        df = observations_dict["obs_in_sample"] - params_info["n_param"]
        if df <= 0:
            print(
                f"Warning: Degrees of freedom ({df}) non-positive for dt "
                f"distribution. Setting intervals to NaN."
            )
            y_lower[:], y_upper[:] = np.nan, np.nan
        else:
            scale = np.sqrt(v_voc_multi)
            if e_type == "A":
                y_lower[:] = scale * stats.t.ppf(level_low, df)
                y_upper[:] = scale * stats.t.ppf(level_up, df)
            else:  # Etype == "M"
                y_lower[:] = 1 + scale * stats.t.ppf(level_low, df)
                y_upper[:] = 1 + scale * stats.t.ppf(level_up, df)

    elif distribution == "dalaplace":
        # Assuming stats.alaplace exists: ppf(q, loc, scale, alpha or kappa)
        alpha = other_params.get("alpha")
        if alpha is not None and 0 < alpha < 1:
            try:
                # Scale parameter from R code
                scale = np.sqrt(
                    v_voc_multi
                    * alpha**2
                    * (1 - alpha) ** 2
                    / (alpha**2 + (1 - alpha) ** 2)
                )
                loc = 1 if e_type == "M" else 0
                # Assuming the third parameter is alpha/kappa
                # Check if stats.alaplace exists before calling
                if hasattr(stats, "alaplace") and hasattr(stats.alaplace, "ppf"):
                    # SciPy <= 1.8 used 'kappa', >= 1.9 uses 'alpha'
                    try:
                        y_lower[:] = stats.alaplace.ppf(
                            level_low, loc=loc, scale=scale, alpha=alpha
                        )
                        y_upper[:] = stats.alaplace.ppf(
                            level_up, loc=loc, scale=scale, alpha=alpha
                        )
                    except TypeError:  # Try kappa for older SciPy versions
                        y_lower[:] = stats.alaplace.ppf(
                            level_low, loc=loc, scale=scale, kappa=alpha
                        )
                        y_upper[:] = stats.alaplace.ppf(
                            level_up, loc=loc, scale=scale, kappa=alpha
                        )
                else:
                    print(
                        "Warning: stats.alaplace not found. "
                        "Cannot calculate intervals for 'dalaplace'."
                    )
                    y_lower[:], y_upper[:] = np.nan, np.nan
            except (ValueError, ZeroDivisionError) as e:
                print(
                    f"Warning: Could not calculate scale for dalaplace "
                    f"(alpha={alpha}). Error: {e}"
                )
                y_lower[:], y_upper[:] = np.nan, np.nan
        else:
            print(
                f"Warning: Alpha parameter ({alpha}) invalid or not found "
                f"for dalaplace."
            )
            y_lower[:], y_upper[:] = np.nan, np.nan

    # Log-Distributions (handling depends on whether v_voc_multi is variance
    # of log). Assuming v_voc_multi IS the variance of the log error based on
    # R lines 8429-8433 if Etype=='M'. For Etype=='A', R calculates these as
    # if M and then adjusts. Python code does this too.

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
        scale_log = (v_voc_multi / 120) ** 0.25
        # Calculate quantiles of (1+error) multiplier
        try:
            # Check if stats.s_dist exists before calling
            if hasattr(stats, "s_dist") and hasattr(stats.s_dist, "ppf"):
                y_lower_mult = np.exp(
                    stats.s_dist.ppf(level_low, loc=0, scale=scale_log)
                )
                y_upper_mult = np.exp(
                    stats.s_dist.ppf(level_up, loc=0, scale=scale_log)
                )
            else:
                print(
                    "Warning: stats.s_dist not found. "
                    "Cannot calculate intervals for 'dls'."
                )
                y_lower_mult, y_upper_mult = np.nan, np.nan
        except Exception as e:
            print(f"Error calculating 'dls' interval: {e}")
            y_lower_mult, y_upper_mult = np.nan, np.nan
        # Final adjustment depends on Etype

    elif distribution == "dlgnorm":
        # Corresponds to exp(GenNorm(0, scale_log, beta))
        shape_beta = other_params.get("shape")
        if shape_beta is not None:
            try:
                scale_log = np.sqrt(
                    v_voc_multi * (gamma(1 / shape_beta) / gamma(3 / shape_beta))
                )
                # Calculate quantiles of (1+error) multiplier
                y_lower_mult = np.exp(
                    stats.gennorm.ppf(
                        level_low, beta=shape_beta, loc=0, scale=scale_log
                    )
                )
                y_upper_mult = np.exp(
                    stats.gennorm.ppf(level_up, beta=shape_beta, loc=0, scale=scale_log)
                )
            except (ValueError, ZeroDivisionError) as e:
                print(
                    f"Warning: Could not calculate scale for dlgnorm "
                    f"(shape={shape_beta}). Error: {e}"
                )
                y_lower_mult, y_upper_mult = np.nan, np.nan
        else:
            print("Warning: Shape parameter 'beta' not found for dlgnorm.")
            y_lower_mult, y_upper_mult = np.nan, np.nan
        # Final adjustment depends on Etype

    # Distributions naturally multiplicative (or treated as such for intervals)
    elif distribution == "dinvgauss":
        # stats.invgauss.ppf(q, mu, loc=0, scale=1). mu is shape parameter.
        # R: qinvgauss(p, mean=1, dispersion=vcovMulti) -> lambda = 1/vcovMulti
        # Map (mean=1, lambda=1/vcovMulti) to scipy's mu. Tentative:
        # mu = 1/vcovMulti? Variance = mean^3 / lambda. If mean=1, Var =
        # 1/lambda. If vcovMulti=Var -> lambda=1/vcovMulti. Let's try
        # mu = 1 / vcovMulti as the shape parameter `mu` for scipy
        if np.any(v_voc_multi <= 0):
            print(
                "Warning: Non-positive variance for dinvgauss. "
                "Setting intervals to NaN."
            )
            y_lower[:], y_upper[:] = np.nan, np.nan
        else:
            mu_shape = 1.0 / v_voc_multi  # Tentative mapping
            # Calculate quantiles of (1+error) multiplier (mean should be 1)
            y_lower_mult = stats.invgauss.ppf(
                level_low, mu=mu_shape, loc=0, scale=1
            )  # loc=0, scale=1 for standard form around mu
            y_upper_mult = stats.invgauss.ppf(level_up, mu=mu_shape, loc=0, scale=1)
            # Need to rescale ppf output? Let's assume R's mean=1 implies the
            # output is already centered around 1. Needs verification.

    elif distribution == "dgamma":
        # stats.gamma.ppf(q, a, loc=0, scale=1). a=shape.
        # R: qgamma(p, shape=1/vcovMulti, scale=vcovMulti) -> Mean =
        # shape*scale = 1. Variance = shape*scale^2 = vcovMulti.
        if np.any(v_voc_multi <= 0):
            print(
                "Warning: Non-positive variance for dgamma. Setting intervals to NaN."
            )
            y_lower[:], y_upper[:] = np.nan, np.nan
        else:
            shape_a = 1.0 / v_voc_multi
            scale_param = v_voc_multi
            # Calculate quantiles of (1+error) multiplier (mean is 1)
            y_lower_mult = stats.gamma.ppf(
                level_low, a=shape_a, loc=0, scale=scale_param
            )
            y_upper_mult = stats.gamma.ppf(
                level_up, a=shape_a, loc=0, scale=scale_param
            )

    else:
        print(
            f"Warning: Distribution '{distribution}' not recognized "
            f"for interval calculation."
        )
        y_lower[:], y_upper[:] = np.nan, np.nan

    # Final adjustments based on Etype (as done in R lines 8632-8640)
    # This part should come *after* the above block in your main script
    needs_etype_A_adjustment = distribution in [
        "dlnorm",
        "dllaplace",
        "dls",
        "dlgnorm",
        "dinvgauss",
        "dgamma",
    ]

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
        zero_lower_mask = level_low == 0
        if np.any(zero_lower_mask):
            if e_type == "A":
                y_lower_final[zero_lower_mask] = -np.inf
            else:  # e_type == "M"
                y_lower_final[zero_lower_mask] = 0.0

        # Check level_up for 100% quantile
        one_upper_mask = level_up == 1
        if np.any(one_upper_mask):
            y_upper_final[one_upper_mask] = np.inf
    else:  # cumulative = True (Dealing with a single value)
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
    else:  # e_type == "M"
        # y_lower/upper_final currently hold multipliers, multiply forecast
        y_lower_final = y_forecast * y_lower_final
        y_upper_final = y_forecast * y_upper_final

    return y_lower_final, y_upper_final


def generate_simulation_interval(
    predictions,
    prepared_model,
    general_dict,
    observations_dict,
    model_type_dict,
    lags_dict,
    components_dict,
    explanatory_checked,
    constants_checked,
    params_info,
    adam_cpp,
    level,
    nsim=10000,
    external_errors=None,
):
    """
    Generate prediction intervals using simulation.

    This implements the simulation-based intervals from R's forecast.adam()
    (lines 8317-8412 in R/adam.R).

    Parameters
    ----------
    predictions : np.ndarray
        Point forecasts.
    prepared_model : dict
        Dictionary with the prepared model.
    general_dict : dict
        Dictionary with general model parameters.
    observations_dict : dict
        Dictionary with observation data.
    model_type_dict : dict
        Dictionary with model type information.
    lags_dict : dict
        Dictionary with lag-related information.
    components_dict : dict
        Dictionary with model components information.
    explanatory_checked : dict
        Dictionary with external regressors information.
    constants_checked : dict
        Dictionary with information about constants.
    params_info : dict
        Dictionary with parameter information.
    level : float
        Confidence level for prediction intervals.
    nsim : int
        Number of simulations to run.
    external_errors : np.ndarray, optional
        Pre-generated error matrix of shape (h, nsim) for deterministic testing.
        If provided, these errors are used instead of generating new ones.
        This allows 100% reproducibility between R and Python by using
        the same random errors.

    Returns
    -------
    tuple
        (y_lower, y_upper, y_simulated) where y_simulated is the raw (h, nsim)
        simulation matrix when ``general_dict["scenarios"]`` is True, else None.
    """
    h = general_dict["h"]
    lags_model_max = lags_dict["lags_model_max"]

    # Get number of components
    n_components = (
        components_dict["components_number_ets"]
        + components_dict.get("components_number_arima", 0)
        + explanatory_checked["xreg_number"]
        + int(constants_checked["constant_required"])
    )

    # 1. Create 3D state array: [components, h+lags_max, nsim]
    arr_vt = np.zeros((n_components, h + lags_model_max, nsim), order="F")

    # Initialize with current states (replicated across nsim)
    mat_vt = prepared_model["states"][
        :,
        observations_dict["obs_states"] - lags_model_max : observations_dict[
            "obs_states"
        ]
        + 1,
    ]
    for i in range(nsim):
        arr_vt[:, :lags_model_max, i] = mat_vt[:, :lags_model_max]

    # 2. Calculate degrees of freedom for de-biasing
    # For variance calculation, use n_param_all - n_param_scale
    # Check if we have the new n_param table structure
    n_param = None
    if "n_param" in general_dict and general_dict["n_param"] is not None:
        n_param = general_dict["n_param"].n_param_for_variance
    elif params_info and params_info[0]:
        # Legacy: params_info[0][-1] is n_param_all, params_info[0][3] is n_param_scale
        n_param_all = (
            params_info[0][-1] if len(params_info[0]) > 4 else params_info[0][0]
        )
        n_param_scale = params_info[0][3] if len(params_info[0]) > 3 else 0
        n_param = n_param_all - n_param_scale
    else:
        n_param = 0

    df = observations_dict["obs_in_sample"] - n_param
    if df <= 0:
        df = observations_dict["obs_in_sample"]

    # 3. Get and de-bias scale
    scale_value = prepared_model["scale"] * observations_dict["obs_in_sample"] / df

    # 4. Generate random errors or use external errors
    if external_errors is not None:
        # Use externally provided errors for deterministic testing
        mat_errors = external_errors
        if mat_errors.shape != (h, nsim):
            raise ValueError(
                f"external_errors shape {mat_errors.shape} "
                f"does not match (h={h}, nsim={nsim})"
            )
        distribution = general_dict["distribution"]
    else:
        distribution = general_dict["distribution"]
        other_params = general_dict.get("other", {})

        # Generate h*nsim errors and reshape to (h, nsim)
        errors_flat = generate_errors(
            distribution=distribution,
            n=h * nsim,
            scale=scale_value,
            obs_in_sample=observations_dict["obs_in_sample"],
            n_param=n_param,
            shape=other_params.get("shape"),
            alpha=other_params.get("alpha"),
        )
        mat_errors = errors_flat.reshape((h, nsim), order="F")

    # 5. Normalize errors if nsim <= 500
    e_type = model_type_dict["error_type"]
    if nsim <= 500:
        mat_errors = normalize_errors(mat_errors, e_type)

    # 6. Determine modified error type for additive models with log-distributions
    e_type_modified = e_type
    if e_type == "A" and distribution in [
        "dlnorm",
        "dinvgauss",
        "dgamma",
        "dls",
        "dllaplace",
        "dlgnorm",
    ]:
        e_type_modified = "M"

    # 7. Prepare matrices for simulator
    mat_vt_prep, mat_wt, vec_g, mat_f = _prepare_matrices_for_forecast(
        prepared_model, observations_dict, lags_dict, general_dict
    )

    # Prepare lookup table
    lookup = _prepare_lookup_table(lags_dict, observations_dict, general_dict)

    # Create 3D arrays for F and G (replicated for each simulation)
    arr_f = np.zeros((mat_f.shape[0], mat_f.shape[1], nsim), order="F")
    for i in range(nsim):
        arr_f[:, :, i] = mat_f

    # G matrix: [n_components, nsim]
    mat_g = np.zeros((n_components, nsim), order="F")
    for i in range(nsim):
        mat_g[:, i] = vec_g.flatten()

    # Occurrence matrix (all ones for now - no occurrence model)
    mat_ot = np.ones((h, nsim), order="F")

    # Profiles recent table - expand to 3D cube (nComponents, lagsModelMax, nsim)
    profiles_recent_2d = prepared_model["profiles_recent_table"]
    profiles_recent = np.zeros(
        (profiles_recent_2d.shape[0], profiles_recent_2d.shape[1], nsim), order="F"
    )
    for i in range(nsim):
        profiles_recent[:, :, i] = profiles_recent_2d
    profiles_recent = np.asfortranarray(profiles_recent, dtype=np.float64)

    # Prepare inputs for C++ simulator
    arr_vt_f = np.asfortranarray(arr_vt, dtype=np.float64)
    mat_errors_f = np.asfortranarray(mat_errors, dtype=np.float64)
    mat_ot_f = np.asfortranarray(mat_ot, dtype=np.float64)
    arr_f_f = np.asfortranarray(arr_f, dtype=np.float64)
    mat_wt_f = np.asfortranarray(mat_wt, dtype=np.float64)
    mat_g_f = np.asfortranarray(mat_g, dtype=np.float64)
    lookup_f = np.asfortranarray(lookup, dtype=np.uint64)

    # 8. Call adam_cpp.simulate() with the prepared inputs
    # Note: E, T, S, nNonSeasonal, nSeasonal, nArima, nXreg, constant are set
    # during adamCore construction
    sim_result = adam_cpp.simulate(
        matrixErrors=mat_errors_f,
        matrixOt=mat_ot_f,
        arrayVt=arr_vt_f,
        matrixWt=mat_wt_f,
        arrayF=arr_f_f,
        matrixG=mat_g_f,
        indexLookupTable=lookup_f,
        profilesRecent=profiles_recent,
        E=e_type_modified,
    )

    y_simulated = sim_result.data  # Shape: (h, nsim)

    # 9. Handle cumulative forecasts
    if general_dict.get("cumulative", False):
        y_forecast_sim = np.mean(np.sum(y_simulated, axis=0))
        level_low = (1 - level) / 2
        level_up = 1 - level_low
        y_lower = np.array([np.quantile(np.sum(y_simulated, axis=0), level_low)])
        y_upper = np.array([np.quantile(np.sum(y_simulated, axis=0), level_up)])
    else:
        # 10. Calculate quantiles for each horizon
        level_low = (1 - level) / 2
        level_up = 1 - level_low

        y_lower = np.zeros(h)
        y_upper = np.zeros(h)
        y_forecast_sim = np.zeros(h)

        for i in range(h):
            # For multiplicative trend/seasonal, use trimmed mean
            if model_type_dict["trend_type"] == "M" or (
                model_type_dict["season_type"] == "M"
                and h > lags_dict.get("lags_model_min", 1)
            ):
                # Trim 1% on each side
                y_forecast_sim[i] = stats.trim_mean(y_simulated[i, :], 0.01)
            else:
                y_forecast_sim[i] = np.mean(y_simulated[i, :])

            # Use R's type=7 quantile (linear interpolation)
            y_lower[i] = np.quantile(
                y_simulated[i, :], level_low, interpolation="linear"
            )
            y_upper[i] = np.quantile(
                y_simulated[i, :], level_up, interpolation="linear"
            )

    # 11. Convert to relative form (like parametric intervals)
    # R uses the same yForecast for both conversion and final combination:
    # - For additive models: yForecast = point forecast (adamForecast)
    # - For multiplicative: yForecast = simulation mean (overwritten in loop)
    if e_type == "A":
        # For additive: use point forecast for both operations (like R)
        y_lower = y_lower - predictions
        y_upper = y_upper - predictions
    else:
        # For multiplicative: use simulation mean for both operations (like R)
        # Avoid division by zero
        y_lower = np.where(y_forecast_sim != 0, y_lower / y_forecast_sim, 0)
        y_upper = np.where(y_forecast_sim != 0, y_upper / y_forecast_sim, 0)

    # 12. Final combination with forecasts (same as parametric)
    y_lower_final = y_lower.copy()
    y_upper_final = y_upper.copy()

    # Handle NaNs
    nan_lower_mask = np.isnan(y_lower_final)
    if np.any(nan_lower_mask):
        replace_val = 0.0 if e_type == "A" else 1.0
        y_lower_final[nan_lower_mask] = replace_val

    nan_upper_mask = np.isnan(y_upper_final)
    if np.any(nan_upper_mask):
        replace_val = 0.0 if e_type == "A" else 1.0
        y_upper_final[nan_upper_mask] = replace_val

    # Combine with forecasts using the SAME value as used for relative form
    if e_type == "A":
        # For additive: use point forecast (same as subtraction above)
        y_lower_final = predictions + y_lower_final
        y_upper_final = predictions + y_upper_final
    else:
        # For multiplicative: use simulation mean (same as division above)
        y_lower_final = y_forecast_sim * y_lower_final
        y_upper_final = y_forecast_sim * y_upper_final

    scenarios_out = y_simulated if general_dict.get("scenarios", False) else None
    return y_lower_final, y_upper_final, scenarios_out
