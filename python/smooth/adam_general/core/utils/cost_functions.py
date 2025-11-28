import numpy as np
from numpy.linalg import eigvals
from smooth.adam_general.core.creator import filler
from smooth.adam_general.core.utils.utils import measurement_inverter, scaler, calculate_likelihood, calculate_entropy, calculate_multistep_loss
import numpy as np
from smooth.adam_general._adam_general import adam_fitter, adam_forecaster


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
       bounds = "usual", 
       other=None, otherParameterEstimate=False, 
       arPolynomialMatrix=None, maPolynomialMatrix=None,
       regressors=None):
    
   
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
                        constants_checked)
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
            if arima_checked['ar_estimate'] and sum(-adamElements['arimaPolynomials']['arPolynomial'][1:]) >= 1:
                arPolynomialMatrix[:, 0] = -adamElements['arimaPolynomials']['arPolynomial'][1:]
                arPolyroots = np.abs(eigvals(arPolynomialMatrix))
                # Strict constraint enforcement like in R
                if any(arPolyroots > 1):
                    # Return a large penalty value
                    return 1e100
            
            if arima_checked['ma_estimate'] and sum(adamElements['arimaPolynomials']['maPolynomial'][1:]) >= 1:
                maPolynomialMatrix[:, 0] = adamElements['arimaPolynomials']['maPolynomial'][1:]
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
            if arima_checked['ar_estimate'] and (sum(-adamElements['arimaPolynomials']['arPolynomial'][1:]) >= 1 or sum(-adamElements['arimaPolynomials']['arPolynomial'][1:]) < 0):
                arPolynomialMatrix[:, 0] = -adamElements['arimaPolynomials']['arPolynomial'][1:]
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
                if model_type_dict['ets_model'] or (arima_checked['arima_model'] and arima_checked['ma_estimate'] and (sum(adamElements['arimaPolynomials']['maPolynomial'][1:]) >= 1 or sum(adamElements['arimaPolynomials']['maPolynomial'][1:]) < 0)):
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
    profiles_recent_table = np.asfortranarray(profile_dict['profiles_recent_table'], dtype=np.float64)

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
        nNonSeasonal=components_dict['components_number_ets'] - components_dict['components_number_ets_seasonal'],
        nSeasonal=components_dict['components_number_ets_seasonal'],
        nArima=components_dict['components_number_arima'],
        nXreg=explanatory_checked['xreg_number'],
        constant=constants_checked['constant_required'],
        vectorYt=y_in_sample,  # Ensure correct mapping
        vectorOt=ot,  # Ensure correct mapping
        backcast=backcast_value,
        nIterations=initials_checked['n_iterations'],
        refineHead=refine_head,
        adamETS=adam_ets
    )


    #adam_fitted['errors'] = np.repeat()

    #print('adam_fitted')
    #print(adam_fitted)
    if not general['multisteps']:
        if general['loss'] == "likelihood":
            
            scale = scaler(general['distribution_new'], 
                            model_type_dict['error_type'], 
                            adam_fitted['errors'][observations_dict['ot_logical']],
                            adam_fitted['yFitted'][observations_dict['ot_logical']], 
                            observations_dict['obs_in_sample'], 
                            other)
            #print(adam_fitted['errors'])
            # Calculate the likelihood
            CFValue = -np.sum(calculate_likelihood(general['distribution_new'], 
                                                    model_type_dict['error_type'], 
                                                    observations_dict['y_in_sample'][observations_dict['ot_logical']],
                                                    adam_fitted['yFitted'][observations_dict['ot_logical']], 
                                                    scale, 
                                                    other))
            #print(CFValue)
            # Differential entropy for the logLik of occurrence model
            if observations_dict.get('occurrence_model', False) or any(~observations_dict['ot_logical']):
                CFValueEntropy = calculate_entropy(general['distribution_new'], 
                                                scale, 
                                                other, 
                                                observations_dict['obs_zero'],
                                                adam_fitted['yFitted'][~observations_dict['ot_logical']])
                if np.isnan(CFValueEntropy) or CFValueEntropy < 0:
                    CFValueEntropy = np.inf
                CFValue += CFValueEntropy

        elif general['loss'] == "MSE":
            CFValue = np.sum(adam_fitted['errors']**2) / observations_dict['obs_in_sample']
        elif general['loss'] == "MAE":
            CFValue = np.sum(np.abs(adam_fitted['errors'])) / observations_dict['obs_in_sample']
        elif general['loss'] == "HAM":
            CFValue = np.sum(np.sqrt(np.abs(adam_fitted['errors']))) / observations_dict['obs_in_sample']
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
                            np.sqrt(np.sum((adam_fitted['errors'] / general['y_denominator'])**2) / 
                                observations_dict['obs_in_sample']))
            else:  # "M"
                CFValue = ((1 - general['lambda']) * 
                            np.sqrt(np.sum(np.log(1 + adam_fitted['errors'])**2) / 
                                observations_dict['obs_in_sample']))

            if general['loss'] == "LASSO":
                CFValue += general['lambda'] * np.sum(np.abs(B))
            else:  # "RIDGE"
                CFValue += general['lambda'] * np.sqrt(np.sum(B**2))

        elif general['loss'] == "custom":
            CFValue = general['loss_function'](actual=observations_dict['y_in_sample'], 
                                                fitted=adam_fitted['yFitted'], 
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
        multisteps = False
):
    

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
                                    constant_dict)

            # Write down the initials in the recent profile
            profile_dict['profiles_recent_table'][:] = adam_elements['matVt'][:, :lags_dict['lags_model_max']]

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

            adam_fitted = adam_fitter(adam_elements['mat_vt'],
                              adam_elements['mat_wt'],
                              adam_elements['mat_f'],
                              adam_elements['vec_g'],
                              lags_dict['lags_model_all'],
                              profile_dict['index_lookup_table'],
                              profile_dict['profiles_recent_table'],
                              model_type_dict['error_type'],
                              model_type_dict['trend_type'],
                              model_type_dict['season_type'],
                              components_dict['components_number_ets'],
                              components_dict['components_number_ets_seasonal'],
                              components_dict['components_number_arima'],
                              explanatory_dict['xreg_number'],
                              constant_dict['constant_required'],
                              observations_dict['y_in_sample'],
                              observations_dict['ot'],
                              backcast_value_log,
                              initials_dict['n_iterations'],
                              refine_head,
                              adam_ets)
            
            logLikReturn -= np.sum(np.log(np.abs(adam_fitted['y_fitted'])))

        return logLikReturn