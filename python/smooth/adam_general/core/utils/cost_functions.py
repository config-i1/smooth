import numpy as np
from numpy.linalg import eigvals
from smooth.adam_general.core.creator import filler
from smooth.adam_general.core.utils.utils import measurement_inverter, scaler, calculate_likelihood, calculate_entropy, calculate_multistep_loss
import numpy as np


def CF(B,
       model_type_dict,
       components_dict,
       lags_dict,
       matrices_dict,
       persistence_checked,
       initials_checked,
       arima_checked,
       explanatory_checked,
       phi_dict,
       constants_checked,
       observations_dict,
       profile_dict,
       general,
       adam_cpp,
       bounds = "usual",
       other=None, otherParameterEstimate=False,
       arPolynomialMatrix=None, maPolynomialMatrix=None,
       regressors=None):
    """
    Cost Function for ADAM model parameter estimation.

    This function calculates the value of the cost function (CF) for given parameters
    during the optimization process. The CF is minimized to find optimal parameter values
    for the ADAM model. The function implements various loss functions (likelihood, MSE, MAE,
    HAM, LASSO, RIDGE) and enforces parameter constraints (bounds).

    The cost function is evaluated as follows:

    1. **Parameter Filling**: Fill model matrices with current parameter values from vector B
    2. **Bounds Checking**: Apply parameter constraints based on the 'bounds' setting:

       - **"usual"**: Classical restrictions on smoothing parameters:

         * ETS smoothing parameters: :math:`0 \\leq \\alpha, \\beta, \\gamma \\leq 1`
         * Trend constraint: :math:`\\beta \\leq \\alpha`
         * Seasonal constraint: :math:`\\gamma \\leq 1 - \\alpha`
         * Damping constraint: :math:`0 \\leq \\phi \\leq 1`
         * ARIMA stationarity: AR and MA polynomial roots outside unit circle

       - **"admissible"**: Check eigenvalues of the state transition matrix to ensure stability
       - **None**: No bounds checking

    3. **Model Fitting**: Call C++ fitter to compute fitted values and errors
    4. **Loss Calculation**: Compute loss based on specified loss function:

       - **"likelihood"**: Negative log-likelihood for specified distribution
       - **"MSE"**: Mean Squared Error
       - **"MAE"**: Mean Absolute Error
       - **"HAM"**: Half Absolute Moment (square root of absolute errors)
       - **"LASSO"** / **"RIDGE"**: Regularized loss with L1/L2 penalties

    Parameters
    ----------
    B : numpy.ndarray
        Parameter vector containing (in order):

        - ETS persistence parameters (α, β, γ)
        - Damping parameter (φ)
        - Initial states
        - ARIMA parameters (AR, MA coefficients)
        - Regression coefficients
        - Constant term
        - Distribution parameters (if estimated)

    model_type_dict : dict
        Model type specification containing:

        - 'error_type': Error type ('A' for additive, 'M' for multiplicative)
        - 'trend_type': Trend type ('N', 'A', 'Ad', 'M', 'Md')
        - 'season_type': Seasonality type ('N', 'A', 'M')
        - 'ets_model': Whether ETS components are present
        - 'arima_model': Whether ARIMA components are present
        - 'model_is_trendy': Whether trend is present
        - 'model_is_seasonal': Whether seasonality is present

    components_dict : dict
        Components information containing:

        - 'components_number_ets': Total number of ETS components
        - 'components_number_ets_seasonal': Number of seasonal ETS components
        - 'components_number_ets_non_seasonal': Number of non-seasonal ETS components
        - 'components_number_arima': Number of ARIMA components

    lags_dict : dict
        Lag structure information containing:

        - 'lags': Vector of lags for each seasonal component
        - 'lags_model': Lags for each model component
        - 'lags_model_all': Complete lag specification
        - 'lags_model_max': Maximum lag value

    matrices_dict : dict
        State-space matrices from creator(), modified in-place:

        - 'mat_vt': State vector matrix
        - 'mat_wt': Measurement matrix
        - 'mat_f': Transition matrix
        - 'vec_g': Persistence vector

    persistence_checked : dict
        Persistence parameters specification from checker()
    initials_checked : dict
        Initial values specification containing:

        - 'initial_type': Initialization method ('optimal', 'backcasting', 'complete')
        - 'n_iterations': Number of backcasting iterations

    arima_checked : dict
        ARIMA specification containing:

        - 'arima_model': Whether ARIMA is present
        - 'ar_estimate': Whether to estimate AR parameters
        - 'ma_estimate': Whether to estimate MA parameters
        - 'ar_required': Whether AR is required
        - 'ma_required': Whether MA is required
        - 'ar_orders': AR orders for each lag
        - 'ma_orders': MA orders for each lag

    explanatory_checked : dict
        External regressors specification containing:

        - 'xreg_model': Whether external regressors are present
        - 'xreg_number': Number of external regressors

    phi_dict : dict
        Damping parameter specification containing:

        - 'phi_estimate': Whether to estimate damping parameter
        - 'phi': Current damping parameter value

    constants_checked : dict
        Constant term specification containing:

        - 'constant_required': Whether a constant is included
        - 'constant_estimate': Whether to estimate the constant

    observations_dict : dict
        Observations information containing:

        - 'y_in_sample': In-sample time series values
        - 'ot': Occurrence variable (for intermittent data)
        - 'ot_logical': Boolean mask for non-zero observations
        - 'obs_in_sample': Number of in-sample observations
        - 'obs_zero': Number of zero observations
        - 'occurrence_model': Whether occurrence model is present

    profile_dict : dict
        Profile matrices for time-varying parameters containing:

        - 'profiles_recent_table': Recent values for profile initialization
        - 'index_lookup_table': Index lookup for profile access

    general : dict
        General model configuration containing:

        - 'loss': Loss function ('likelihood', 'MSE', 'MAE', 'HAM', 'LASSO', 'RIDGE')
        - 'distribution_new': Error distribution ('dnorm', 'dlaplace', 'ds', 'dgnorm', 'dlnorm', 'dgamma', 'dinvgauss')
        - 'multisteps': Whether to use multistep loss
        - 'lambda': Regularization parameter for LASSO/RIDGE
        - 'denominator': Scaling denominator for LASSO/RIDGE
        - 'y_denominator': Y-value scaling for LASSO/RIDGE

    bounds : str, optional
        Type of bounds to enforce:

        - "usual": Classical parameter restrictions (default)
        - "admissible": Admissibility constraints based on eigenvalues
        - None: No bounds checking

    other : float, optional
        Additional distribution parameters (e.g., shape parameter for generalized normal)
    otherParameterEstimate : bool, optional
        Whether to estimate distribution parameters from B vector
    arPolynomialMatrix : numpy.ndarray, optional
        Companion matrix for AR polynomial (for bounds checking)
    maPolynomialMatrix : numpy.ndarray, optional
        Companion matrix for MA polynomial (for bounds checking)
    regressors : str, optional
        Regressor handling method ('use', 'select', 'adapt')

    Returns
    -------
    float
        Cost function value. Returns a large penalty (1e100 or 1e300) if constraints are
        violated or computation fails. Otherwise returns the computed loss value based on
        the specified loss function.

    Notes
    -----
    The function is called repeatedly during optimization by NLopt. It performs the following:

    1. Fills model matrices with current parameter values using ``filler()``
    2. Checks parameter bounds and returns penalty if violated
    3. Calls C++ ``adam_fitter()`` to compute fitted values and errors
    4. Calculates and returns the appropriate loss function value

    **Important Implementation Details**:

    - Matrices are passed to C++ by reference and may be modified
    - Arrays must be in Fortran order for C++ compatibility
    - Copies are made of matrices to avoid cross-iteration contamination
    - NaN values in CF result in a large penalty (1e300)

    **Parameter Constraints**:

    For "usual" bounds, violations return penalty 1e100:

    - ETS smoothing parameters outside [0,1]
    - Trend parameter β > α (violates smoothness)
    - Seasonal parameter γ > 1-α
    - Damping φ outside [0,1]
    - ARIMA polynomial roots inside unit circle (non-stationary)

    For "admissible" bounds, violations return penalty 1e100 × max(eigenvalue):

    - Eigenvalues of state transition matrix > 1 (unstable system)

    See Also
    --------
    log_Lik_ADAM : Calculate log-likelihood for fitted model
    filler : Fill model matrices with parameter values
    adam_fitter : C++ function for model fitting

    References
    ----------
    .. [1] Svetunkov, I. (2023). "Smooth forecasting with the smooth package in R".
           arXiv:2301.01790.
    .. [2] Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008).
           "Forecasting with Exponential Smoothing: The State Space Approach".
           Springer-Verlag.

    Examples
    --------
    This function is typically called internally during optimization::

        >>> # During optimization, NLopt calls CF repeatedly
        >>> cf_value = CF(B=initial_params, model_type_dict=..., components_dict=..., ...)
        >>> # If cf_value is large (1e100), constraints were violated
    """

    # Fill in the matrices
    adamElements = filler(B,
                        model_type_dict,
                        components_dict,
                        lags_dict,
                        matrices_dict,
                        persistence_checked,
                        initials_checked,
                        arima_checked,
                        explanatory_checked,
                        phi_dict,
                        constants_checked,
                        adam_cpp)
    # If we estimate parameters of distribution, take it from the B vector
    if otherParameterEstimate:
        
        other = abs(B[-1])
        if general['distribution_new'] in ["dgnorm", "dlgnorm"] and other < 0.25:
            # MODIFIED: reduced penalty value
            return 1e5 / other
    # Check the bounds, classical restrictions
    #print(components_dict['components_number_ets_non_seasonal'])
    
    if bounds == "usual":
        
        if arima_checked['arima_model'] and any([arima_checked['ar_estimate'], arima_checked['ma_estimate']]):
            if arima_checked['ar_estimate'] and sum(-adamElements['arimaPolynomials']['ar_polynomial'][1:]) >= 1:
                arPolynomialMatrix[:, 0] = -adamElements['arimaPolynomials']['ar_polynomial'][1:]
                arPolyroots = np.abs(eigvals(arPolynomialMatrix))
                # Strict constraint enforcement like in R
                if any(arPolyroots > 1):
                    # Return a large penalty value
                    return 1e100
            
            if arima_checked['ma_estimate'] and sum(adamElements['arimaPolynomials']['ma_polynomial'][1:]) >= 1:
                maPolynomialMatrix[:, 0] = adamElements['arimaPolynomials']['ma_polynomial'][1:]
                maPolyroots = np.abs(eigvals(maPolynomialMatrix))
                # Strict constraint enforcement like in R
                if any(maPolyroots > 1):
                    # Return a large penalty value
                    return 1e100
        
        if model_type_dict['ets_model']:
            # Strict constraint enforcement like in R
            # Check if any smoothing parameters are outside the [0,1] bounds
            if any(adamElements['vec_g'][:components_dict['components_number_ets']] > 1) or any(adamElements['vec_g'][:components_dict['components_number_ets']] < 0):
                
                return 1e100
            if model_type_dict['model_is_trendy']:
                # Strict constraint enforcement like in R
                if adamElements['vec_g'][1] > adamElements['vec_g'][0]:
                    return 1e100
                if model_type_dict['model_is_seasonal'] and \
                    any(adamElements['vec_g'][components_dict['components_number_ets_non_seasonal']:
                                    components_dict['components_number_ets_non_seasonal'] + 
                                    components_dict['components_number_ets_seasonal']] > (1 - adamElements['vec_g'][0])):
                    
                    return 1e100
            
            elif model_type_dict['model_is_seasonal'] and \
                    any(adamElements['vec_g'][components_dict['components_number_ets_non_seasonal']:
                                components_dict['components_number_ets_non_seasonal'] + 
                                components_dict['components_number_ets_seasonal']] > (1 - adamElements['vec_g'][0])):
                    
                    return 1e100

            # Strict constraint enforcement like in R
            if phi_dict['phi_estimate'] and (adamElements['mat_f'][1, 1] > 1 or adamElements['mat_f'][1, 1] < 0):
                return 1e100
        
        # Not supporting regression model now
        # if explanatory_checked['xreg_model'] and regressors == "adapt":
        #     if any(adamElements['vec_g'][components_dict['components_number_ets'] + 
        #                               components_dict['components_number_arima']:
        #                               components_dict['components_number_ets'] + 
        #                               components_dict['components_number_arima'] + 
        #                               explanatory_checked['xreg_number']] > 1) or \
        #        any(adamElements['vec_g'][components_dict['components_number_ets'] + 
        #                               components_dict['components_number_arima']:
        #                               components_dict['components_number_ets'] + 
        #                               components_dict['components_number_arima'] + 
        #                               explanatory_checked['xreg_number']] < 0):
        #         return 1e100 * np.max(np.abs(adamElements['vec_g'][components_dict['components_number_ets'] + 
        #                                                          components_dict['components_number_arima']:
        #                                                          components_dict['components_number_ets'] + 
        #                                                          components_dict['components_number_arima'] + 
        #                                                          explanatory_checked['xreg_number']] - 0.5))

    elif bounds == "admissible":
        if arima_checked['arima_model']:
            if arima_checked['ar_estimate'] and (sum(-adamElements['arimaPolynomials']['ar_polynomial'][1:]) >= 1 or sum(-adamElements['arimaPolynomials']['ar_polynomial'][1:]) < 0):
                arPolynomialMatrix[:, 0] = -adamElements['arimaPolynomials']['ar_polynomial'][1:]
                eigenValues = np.abs(eigvals(arPolynomialMatrix))
                if any(eigenValues > 1):
                    return 1e100 * np.max(eigenValues)

        if model_type_dict['ets_model'] or arima_checked['arima_model']:
            if explanatory_checked['xreg_model']:
                if regressors == "adapt":
                    eigenValues = np.abs(eigvals(
                        adamElements['mat_f'] -
                        np.diag(adamElements['vec_g'].flatten()) @
                        measurement_inverter(adamElements['mat_wt'][:observations_dict['obs_in_sample']]).T @
                        adamElements['mat_wt'][:observations_dict['obs_in_sample']] / observations_dict['obs_in_sample']
                    ))
                else:
                    indices = np.arange(components_dict['components_number_ets'] + components_dict['components_number_arima'])
                    eigenValues = np.abs(eigvals(
                        adamElements['mat_f'][np.ix_(indices, indices)] -
                        adamElements['vec_g'][indices] @
                        adamElements['mat_wt'][observations_dict['obs_in_sample']-1, indices]
                    ))
            else:
                if model_type_dict['ets_model'] or (arima_checked['arima_model'] and arima_checked['ma_estimate'] and (sum(adamElements['arimaPolynomials']['ma_polynomial'][1:]) >= 1 or sum(adamElements['arimaPolynomials']['ma_polynomial'][1:]) < 0)):
                    eigenValues = np.abs(eigvals(
                        adamElements['mat_f'] -
                        adamElements['vec_g'] @ adamElements['mat_wt'][observations_dict['obs_in_sample']-1]
                    ))
                else:
                    eigenValues = np.array([0])

            if any(eigenValues > 1 + 1e-50):
                return 1e100 * np.max(eigenValues)

    # Write down the initials in the recent profile
    profile_dict['profiles_recent_table'][:] = adamElements['mat_vt'][:, :lags_dict['lags_model_max']]
    # Convert pandas Series/DataFrames to numpy arrays
    y_in_sample = np.asarray(observations_dict['y_in_sample'], dtype=np.float64)
    ot = np.asarray(observations_dict['ot'], dtype=np.float64)
    # CRITICAL FIX: C++ adamFitter takes matrixVt by reference and modifies it!
    # We must pass a COPY to avoid polluting adamElements across optimization iterations
    mat_vt = np.asfortranarray(adamElements['mat_vt'].copy(), dtype=np.float64)
    mat_wt = np.asfortranarray(adamElements['mat_wt'], dtype=np.float64)
    mat_f = np.asfortranarray(adamElements['mat_f'].copy(), dtype=np.float64)  # Also copy mat_f since it's passed by reference
    vec_g = np.asfortranarray(adamElements['vec_g'], dtype=np.float64) # Make sure it's a 1D array
    lags_model_all = np.asfortranarray(lags_dict['lags_model_all'], dtype=np.uint64).reshape(-1,1)  # Make sure it's a 1D array
    index_lookup_table = np.asfortranarray(profile_dict['index_lookup_table'], dtype=np.uint64)
    profiles_recent_table = np.asfortranarray(profile_dict['profiles_recent_table'].copy(), dtype=np.float64)

    # Print detailed debug information
    # print('mat_vt shape:', mat_vt.shape, 'dtype:', mat_vt.dtype)
    # print('mat_vt:', mat_vt)
    # print('mat_wt shape:', mat_wt.shape, 'dtype:', mat_wt.dtype)
    # print('mat_wt:', mat_wt)
    # print('mat_f shape:', mat_f.shape, 'dtype:', mat_f.dtype)
    # print('mat_f:', mat_f)
    # print('vec_g shape:', vec_g.shape, 'dtype:', vec_g.dtype)
    # print('vec_g:', vec_g)
    # print('lags_model_all shape:', lags_model_all.shape, 'dtype:', lags_model_all.dtype)
    # print('lags_model_all:', lags_model_all)
    #print('index_lookup_table shape:', index_lookup_table.shape, 'dtype:', index_lookup_table)
    # print('profiles_recent_table shape:', profiles_recent_table.shape, 'dtype:', profiles_recent_table)
    # print('error_type:', model_type_dict['error_type'])
    # print('trend_type:', model_type_dict['trend_type'])
    # print('season_type:', model_type_dict['season_type'])
    # print('components_number_ets:', components_dict['components_number_ets'])
    # print('components_number_ets_seasonal:', components_dict['components_number_ets_seasonal'])
    # print('components_number_arima:', components_dict['components_number_arima'])
    # print('xreg_number:', explanatory_checked['xreg_number'])
    # print('constant_required:', constants_checked['constant_required'])
    # print('y_in_sample shape:', y_in_sample.shape, 'dtype:', y_in_sample.dtype)
    # print('y_in_sample:', y_in_sample)
    # print('ot shape:', ot.shape, 'dtype:', ot.dtype)
    # print('ot:', ot)

    # refineHead should always be True (fixed backcasting issue)
    refine_head = True
    # Use conventional ETS for now (adamETS=False)
    adam_ets = False

    # Check if initial_type is a list or string and compute backcast correctly
    if isinstance(initials_checked['initial_type'], list):
        backcast_value = any([t == "complete" or t == "backcasting" for t in initials_checked['initial_type']])
    else:
        backcast_value = initials_checked['initial_type'] in ["complete", "backcasting"]

    # Call adam_cpp.fit() using the new class-based interface
    # Parameters that were passed to adam_fitter are now stored in adam_cpp (E, T, S, etc.)
    adam_fitted = adam_cpp.fit(
        matrixVt=mat_vt,
        matrixWt=mat_wt,
        matrixF=mat_f,
        vectorG=vec_g,
        indexLookupTable=index_lookup_table,
        profilesRecent=profiles_recent_table,
        vectorYt=y_in_sample,
        vectorOt=ot,
        backcast=backcast_value,
        nIterations=initials_checked['n_iterations'],
        refineHead=refine_head
    )


    #adam_fitted.errors = np.repeat()

    #print('adam_fitted')
    #print(adam_fitted)
    if not general['multisteps']:
        if general['loss'] == "likelihood":
            
            scale = scaler(general['distribution_new'], 
                            model_type_dict['error_type'], 
                            adam_fitted.errors[observations_dict['ot_logical']],
                            adam_fitted.fitted[observations_dict['ot_logical']], 
                            observations_dict['obs_in_sample'], 
                            other)
            #print(adam_fitted.errors)
            # Calculate the likelihood
            CFValue = -np.sum(calculate_likelihood(general['distribution_new'], 
                                                    model_type_dict['error_type'], 
                                                    observations_dict['y_in_sample'][observations_dict['ot_logical']],
                                                    adam_fitted.fitted[observations_dict['ot_logical']], 
                                                    scale, 
                                                    other))
            #print(CFValue)
            # Differential entropy for the logLik of occurrence model
            if observations_dict.get('occurrence_model', False) or any(~observations_dict['ot_logical']):
                CFValueEntropy = calculate_entropy(general['distribution_new'], 
                                                scale, 
                                                other, 
                                                observations_dict['obs_zero'],
                                                adam_fitted.fitted[~observations_dict['ot_logical']])
                if np.isnan(CFValueEntropy) or CFValueEntropy < 0:
                    CFValueEntropy = np.inf
                CFValue += CFValueEntropy

        elif general['loss'] == "MSE":
            CFValue = np.sum(adam_fitted.errors**2) / observations_dict['obs_in_sample']
        elif general['loss'] == "MAE":
            CFValue = np.sum(np.abs(adam_fitted.errors)) / observations_dict['obs_in_sample']
        elif general['loss'] == "HAM":
            CFValue = np.sum(np.sqrt(np.abs(adam_fitted.errors))) / observations_dict['obs_in_sample']
        elif general['loss'] in ["LASSO", "RIDGE"]:
            persistenceToSkip = (components_dict['components_number_ets'] + 
                                persistence_checked['persistence_xreg_estimate'] * explanatory_checked['xreg_number'] + 
                                phi_dict['phi_estimate'] + 
                                sum(arima_checked['ar_orders']) + 
                                sum(arima_checked['ma_orders']))

            if phi_dict['phi_estimate']:
                B[components_dict['components_number_ets'] + 
                    persistence_checked['persistence_xreg_estimate'] * explanatory_checked['xreg_number']] = \
                    1 - B[components_dict['components_number_ets'] + 
                            persistence_checked['persistence_xreg_estimate'] * explanatory_checked['xreg_number']]

            j = (components_dict['components_number_ets'] + 
                    persistence_checked['persistence_xreg_estimate'] * explanatory_checked['xreg_number'] + 
                    phi_dict['phi_estimate'])

            if arima_checked['arima_model'] and (sum(arima_checked['ma_orders']) > 0 or sum(arima_checked['ar_orders']) > 0):
                for i in range(len(lags_dict['lags'])):
                    B[j:j+arima_checked['ar_orders'][i]] = 1 - B[j:j+arima_checked['ar_orders'][i]]
                    j += arima_checked['ar_orders'][i] + arima_checked['ma_orders'][i]

            if any([t == "optimal" or t == "backcasting" for t in initials_checked['initial_type']]):
                if explanatory_checked['xreg_number'] > 0:
                    B = np.concatenate([B[:persistenceToSkip],
                                        B[-explanatory_checked['xreg_number']:] / general['denominator'] 
                                        if model_type_dict['error_type'] == "A" 
                                        else B[-explanatory_checked['xreg_number']:]])
                else:
                    B = B[:persistenceToSkip]

            if model_type_dict['error_type'] == "A":
                CFValue = ((1 - general['lambda']) * 
                            np.sqrt(np.sum((adam_fitted.errors / general['y_denominator'])**2) / 
                                observations_dict['obs_in_sample']))
            else:  # "M"
                CFValue = ((1 - general['lambda']) * 
                            np.sqrt(np.sum(np.log(1 + adam_fitted.errors)**2) / 
                                observations_dict['obs_in_sample']))

            if general['loss'] == "LASSO":
                CFValue += general['lambda'] * np.sum(np.abs(B))
            else:  # "RIDGE"
                CFValue += general['lambda'] * np.sqrt(np.sum(B**2))

        elif general['loss'] == "custom":
            CFValue = general['loss_function'](actual=observations_dict['y_in_sample'], 
                                                fitted=adam_fitted.fitted, 
                                                B=B)
    #else:
    # currently no multistep loss function

        #adam_errors = adam_errorer_wrap(
        #    adam_fitted['matVt'], adamElements['matWt'], adamElements['matF'],
        #    lags_dict['lags_model_all'], index_lookup_table, profiles_recent_table,
        #    model_type_dict['error_type'], model_type_dict['trend_type'], model_type_dict['season_type'],
        #    components_dict['components_number_ets'], components_dict['components_number_ets_seasonal'],
        #    components_dict['components_number_arima'], explanatory_checked['xreg_number'], constants_checked['constant_required'], general['horizon'],
        #    observations_dict['y_in_sample'], observations_dict['ot'])

        #CFValue = calculate_multistep_loss(general['loss'], adamErrors, observations_dict['obs_in_sample'], general['horizon'])
    if np.isnan(CFValue):
        #print("CFValue is NaN")
        CFValue = 1e300
    return CFValue



def log_Lik_ADAM(
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
        multisteps = False
):
    """
    Calculate log-likelihood for the ADAM model.

    This function computes the log-likelihood value for an ADAM model with given parameters.
    The log-likelihood is used for model selection (via information criteria) and for
    computing confidence intervals. The function handles various loss functions and can
    compute both one-step-ahead and multi-step-ahead likelihoods.

    The log-likelihood is calculated as:

    .. math::

        \\ell = -\\text{CF}(B)

    where CF is the cost function value. For occurrence models (intermittent data), the
    log-likelihood is augmented with the log-probability of the occurrence process:

    .. math::

        \\ell_{\\text{total}} = \\ell + \\sum_{t \\in \\mathcal{D}} \\log p_t +
                                \\sum_{t \\notin \\mathcal{D}} \\log(1-p_t)

    where :math:`\\mathcal{D}` is the set of time points with non-zero demand, and
    :math:`p_t` is the probability of occurrence at time t.

    Parameters
    ----------
    B : numpy.ndarray
        Parameter vector containing (in order):

        - ETS persistence parameters (α, β, γ)
        - Damping parameter (φ)
        - Initial states
        - ARIMA parameters (AR, MA coefficients)
        - Regression coefficients
        - Constant term
        - Distribution parameters (if estimated)

    model_type_dict : dict
        Model type specification containing:

        - 'error_type': Error type ('A' for additive, 'M' for multiplicative)
        - 'trend_type': Trend type ('N', 'A', 'Ad', 'M', 'Md')
        - 'season_type': Seasonality type ('N', 'A', 'M')
        - 'ets_model': Whether ETS components are present
        - 'arima_model': Whether ARIMA components are present

    components_dict : dict
        Components information containing:

        - 'components_number_ets': Total number of ETS components
        - 'components_number_ets_seasonal': Number of seasonal ETS components
        - 'components_number_arima': Number of ARIMA components

    lags_dict : dict
        Lag structure information containing:

        - 'lags': Vector of lags for each seasonal component
        - 'lags_model': Lags for each model component
        - 'lags_model_all': Complete lag specification
        - 'lags_model_max': Maximum lag value

    adam_created : dict
        State-space matrices from creator():

        - 'mat_vt': State vector matrix
        - 'mat_wt': Measurement matrix
        - 'mat_f': Transition matrix
        - 'vec_g': Persistence vector

    persistence_dict : dict
        Persistence parameters specification from checker()
    initials_dict : dict
        Initial values specification containing:

        - 'initial_type': Initialization method ('optimal', 'backcasting', 'complete')
        - 'n_iterations': Number of backcasting iterations

    arima_dict : dict
        ARIMA specification containing:

        - 'arima_model': Whether ARIMA is present
        - 'ar_estimate': Whether to estimate AR parameters
        - 'ma_estimate': Whether to estimate MA parameters
        - 'ar_required': Whether AR is required
        - 'ma_required': Whether MA is required

    explanatory_dict : dict
        External regressors specification containing:

        - 'xreg_model': Whether external regressors are present
        - 'xreg_number': Number of external regressors

    phi_dict : dict
        Damping parameter specification containing:

        - 'phi_estimate': Whether to estimate damping parameter
        - 'phi': Current damping parameter value

    constant_dict : dict
        Constant term specification containing:

        - 'constant_required': Whether a constant is included
        - 'constant_estimate': Whether to estimate the constant

    observations_dict : dict
        Observations information containing:

        - 'y_in_sample': In-sample time series values
        - 'ot': Occurrence variable (for intermittent data)
        - 'ot_logical': Boolean mask for non-zero observations
        - 'obs_in_sample': Number of in-sample observations
        - 'obs_zero': Number of zero observations

    occurrence_dict : dict
        Occurrence model information containing:

        - 'occurrence_model': Whether occurrence model is present
        - 'p_fitted': Fitted probabilities of occurrence

    general_dict : dict
        General model configuration containing:

        - 'loss': Loss function ('likelihood', 'MSE', 'MAE', 'HAM', 'LASSO', 'RIDGE', multistep variants)
        - 'distribution_new': Error distribution ('dnorm', 'dlaplace', 'ds', etc.)
        - 'h': Forecast horizon (for multistep losses)

    profile_dict : dict
        Profile matrices for time-varying parameters containing:

        - 'profiles_recent_table': Recent values for profile initialization
        - 'index_lookup_table': Index lookup for profile access

    multisteps : bool, optional
        Whether to use multi-step-ahead likelihood calculation (default: False).
        If True, computes likelihood based on multi-step forecasts.

    Returns
    -------
    float or dict
        Log-likelihood value. For standard likelihood calculation, returns a float.
        The value represents the natural logarithm of the likelihood function,
        where higher (less negative) values indicate better fit.

        For LASSO/RIDGE losses, returns 0 as these do not have a proper likelihood.

    Notes
    -----
    **Distribution Mapping**:

    For non-likelihood loss functions, the function maps to appropriate distributions:

    - MSE → Normal distribution (dnorm)
    - MAE → Laplace distribution (dlaplace)
    - HAM → S distribution (ds)

    **Multi-step Likelihood**:

    For multi-step loss functions (MSEh, MAEh, HAMh, etc.), concentrated likelihoods are computed:

    - MSEh, TMSE, MSCE: :math:`-\\frac{T-h}{2}(\\log(2\\pi) + 1 + \\log(\\text{loss}))`
    - MAEh, TMAE, MACE: :math:`-(T-h)(\\log(2) + 1 + \\log(\\text{loss}))`
    - HAMh, THAM, CHAM: :math:`-(T-h)(\\log(4) + 2 + 2\\log(\\text{loss}))`

    where T is the sample size and h is the forecast horizon.

    **Occurrence Model**:

    For intermittent data with occurrence models, the total likelihood combines:

    1. Conditional likelihood given non-zero demand
    2. Probability of occurrence/non-occurrence

    **Multiplicative Models**:

    For multiplicative error models in multistep context, the likelihood is adjusted by
    the Jacobian term: :math:`-\\sum_t \\log|y_t|` to account for the log transformation.

    See Also
    --------
    CF : Cost function used during optimization
    ic_function : Calculate information criteria from log-likelihood

    References
    ----------
    .. [1] Svetunkov, I. (2023). "Smooth forecasting with the smooth package in R".
           arXiv:2301.01790.
    .. [2] Snyder, R.D., Ord, J.K., Koehler, A.B., McLaren, K.R., and Beaumont, A.N. (2017).
           "Forecasting compositional time series: A state space approach".
           International Journal of Forecasting, 33(2), 502-512.

    Examples
    --------
    Calculate log-likelihood for estimated parameters::

        >>> loglik = log_Lik_ADAM(
        ...     B=estimated_params,
        ...     model_type_dict=model_type,
        ...     components_dict=components,
        ...     lags_dict=lags,
        ...     adam_created=matrices,
        ...     persistence_dict=persistence,
        ...     initials_dict=initials,
        ...     arima_dict=arima,
        ...     explanatory_dict=explanatory,
        ...     phi_dict=phi,
        ...     constant_dict=constants,
        ...     observations_dict=observations,
        ...     occurrence_dict=occurrence,
        ...     general_dict=general,
        ...     profile_dict=profile
        ... )
        >>> print(f"Log-likelihood: {loglik}")

    For multi-step likelihood::

        >>> loglik_multistep = log_Lik_ADAM(
        ...     B=estimated_params,
        ...     ...,
        ...     multisteps=True
        ... )
    """

    if not multisteps:
        #print(profile_dict)
        if general_dict['loss'] in ["LASSO", "RIDGE"]:
            return 0
        else:
            general_dict['distribution_new'] = {
                "MSE": "dnorm",
                "MAE": "dlaplace",
                "HAM": "ds"
            }.get(general_dict['loss'], general_dict['distribution_new'])

            general_dict['loss_new'] = "likelihood" if general_dict['loss'] in ["MSE", "MAE", "HAM"] else general_dict['loss']

            # Call CF function with bounds="none"
            logLikReturn = -CF(B,  model_type_dict,
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
                                bounds = None)

            # Handle occurrence model
            if occurrence_dict['occurrence_model']:
                if np.isinf(logLikReturn):
                    logLikReturn = 0
                if any(1 - occurrence_dict['p_fitted'][~observations_dict['ot_logical']] == 0) or any(occurrence_dict['p_fitted'][observations_dict['ot_logical']] == 0):
                    pt_new = occurrence_dict['p_fitted'][(occurrence_dict['p_fitted'] != 0) & (occurrence_dict['p_fitted'] != 1)]
                    ot_new = observations_dict['ot'][(occurrence_dict['p_fitted'] != 0) & (occurrence_dict['p_fitted'] != 1)]
                    if len(pt_new) == 0:
                        return logLikReturn
                    else:
                        return logLikReturn + np.sum(np.log(pt_new[ot_new == 1])) + np.sum(np.log(1 - pt_new[ot_new == 0]))
                else:
                    return logLikReturn + np.sum(np.log(occurrence_dict['p_fitted'][observations_dict['ot_logical']])) + np.sum(np.log(1 - occurrence_dict['p_fitted'][~observations_dict['ot_logical']]))
            else:
                return logLikReturn
            
    else:
        # Call CF function with bounds="none"
        logLikReturn = CF(B,
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
                        bounds = None
                                )

        # Concentrated log-likelihoods for the multistep losses
        if general_dict['loss'] in ["MSEh", "aMSEh", "TMSE", "aTMSE", "MSCE", "aMSCE"]:
            # is horizon different than h?
            logLikReturn = -(observations_dict['obs_in_sample'] - general_dict['h']) / 2 * (np.log(2 * np.pi) + 1 + np.log(logLikReturn))
        elif general_dict['loss'] in ["GTMSE", "aGTMSE"]:
            logLikReturn = -(observations_dict['obs_in_sample'] - general_dict['h']) / 2 * (np.log(2 * np.pi) + 1 + logLikReturn)
        elif general_dict['loss'] in ["MAEh", "TMAE", "GTMAE", "MACE"]:
            logLikReturn = -(observations_dict['obs_in_sample'] - general_dict['h']) * (np.log(2) + 1 + np.log(logLikReturn))
        elif general_dict['loss'] in ["HAMh", "THAM", "GTHAM", "CHAM"]:
            logLikReturn = -(observations_dict['obs_in_sample'] - general_dict['h']) * (np.log(4) + 2 + 2 * np.log(logLikReturn))
        elif general_dict['loss'] in ["GPL", "aGPL"]:
            logLikReturn = -(observations_dict['obs_in_sample'] - general_dict['h']) / 2 * (general_dict['h'] * np.log(2 * np.pi) + general_dict['h'] + logLikReturn) / general_dict['h']

        # Make likelihood comparable
        logLikReturn = logLikReturn / (observations_dict['obs_in_sample'] - general_dict['h']) * observations_dict['obs_in_sample']

        # Handle multiplicative model
        if model_type_dict['ets_model'] and model_type_dict['error_type'] == "M":
            # Fill in the matrices
            adam_elements = filler(B,
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
                                    adam_cpp)

            # Write down the initials in the recent profile
            profile_dict['profiles_recent_table'][:] = adam_elements['mat_vt'][:, :lags_dict['lags_model_max']]

            # Fit the model again to extract the fitted values
            # refineHead should always be True (fixed backcasting issue)
            refine_head = True
            # Use conventional ETS for now (adamETS=False)
            adam_ets = False

            # Check if initial_type is a list or string and compute backcast correctly
            if isinstance(initials_dict['initial_type'], list):
                backcast_value_log = any([t == "complete" or t == "backcasting" for t in initials_dict['initial_type']])
            else:
                backcast_value_log = initials_dict['initial_type'] in ["complete", "backcasting"]

            adam_fitted = adam_cpp.fit(
                matrixVt=adam_elements['mat_vt'],
                matrixWt=adam_elements['mat_wt'],
                matrixF=adam_elements['mat_f'],
                vectorG=adam_elements['vec_g'],
                indexLookupTable=profile_dict['index_lookup_table'],
                profilesRecent=profile_dict['profiles_recent_table'],
                vectorYt=observations_dict['y_in_sample'],
                vectorOt=observations_dict['ot'],
                backcast=backcast_value_log,
                nIterations=initials_dict['n_iterations'],
                refineHead=refine_head
            )

            logLikReturn -= np.sum(np.log(np.abs(adam_fitted.fitted)))

        return logLikReturn