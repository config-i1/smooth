import numpy as np
from numpy.linalg import eigvals
from core.creator import filler
from core.utils.utils import measurement_inverter, scaler, calculate_likelihood, calculate_entropy, calculate_multistep_loss
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
            return 1e10 / other


    # Check the bounds, classical restrictions
    if bounds == "usual":
        if arima_checked['arima_model'] and any([arima_checked['ar_estimate'], arima_checked['ma_estimate']]):
            if arima_checked['ar_estimate'] and sum(-adamElements['arimaPolynomials']['arPolynomial'][1:]) >= 1:
                arPolynomialMatrix[:, 0] = -adamElements['arimaPolynomials']['arPolynomial'][1:]
                arPolyroots = np.abs(eigvals(arPolynomialMatrix))
                if any(arPolyroots > 1):
                    return 1e100 * np.max(arPolyroots)
            
            if arima_checked['ma_estimate'] and sum(adamElements['arimaPolynomials']['maPolynomial'][1:]) >= 1:
                maPolynomialMatrix[:, 0] = adamElements['arimaPolynomials']['maPolynomial'][1:]
                maPolyroots = np.abs(eigvals(maPolynomialMatrix))
                if any(maPolyroots > 1):
                    return 1e100 * np.max(np.abs(maPolyroots))

        if model_type_dict['ets_model']:
            if any(adamElements['vec_g'][:components_dict['components_number_ets']] > 1) or \
               any(adamElements['vec_g'][:components_dict['components_number_ets']] < 0):
                return 1e300
            if model_type_dict['model_is_trendy']:
                if adamElements['vec_g'][1] > adamElements['vec_g'][0]:
                    return 1e300
                if model_type_dict['model_is_seasonal'] and \
                   any(adamElements['vec_g'][components_dict['components_number_ets_non_seasonal']:
                                          components_dict['components_number_ets_non_seasonal'] + 
                                          components_dict['components_number_ets_seasonal']] > (1 - adamElements['vec_g'][0])):
                    return 1e300
            elif model_type_dict['model_is_seasonal'] and \
                 any(adamElements['vec_g'][components_dict['components_number_ets_non_seasonal']:
                                        components_dict['components_number_ets_non_seasonal'] + 
                                        components_dict['components_number_ets_seasonal']] > (1 - adamElements['vec_g'][0])):
                return 1e300

            if phi_dict['phi_estimate'] and (adamElements['mat_f'][1, 1] > 1 or adamElements['mat_f'][1, 1] < 0):
                return 1e300

        if explanatory_checked['xreg_model'] and regressors == "adapt":
            if any(adamElements['vec_g'][components_dict['components_number_ets'] + 
                                      components_dict['components_number_arima']:
                                      components_dict['components_number_ets'] + 
                                      components_dict['components_number_arima'] + 
                                      explanatory_checked['xreg_number']] > 1) or \
               any(adamElements['vec_g'][components_dict['components_number_ets'] + 
                                      components_dict['components_number_arima']:
                                      components_dict['components_number_ets'] + 
                                      components_dict['components_number_arima'] + 
                                      explanatory_checked['xreg_number']] < 0):
                return 1e100 * np.max(np.abs(adamElements['vec_g'][components_dict['components_number_ets'] + 
                                                                 components_dict['components_number_arima']:
                                                                 components_dict['components_number_ets'] + 
                                                                 components_dict['components_number_arima'] + 
                                                                 explanatory_checked['xreg_number']] - 0.5))

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

    # Fitter and the losses calculation
    adam_fitted = adam_fitter(adamElements['mat_vt'], 
                              adamElements['mat_wt'], 
                              adamElements['mat_f'], 
                              adamElements['vec_g'],
                              lags_dict['lags_model_all'], 
                              profile_dict['index_lookup_table'], 
                              profile_dict['profiles_recent_table'],
                              model_type_dict['error_type'], 
                              model_type_dict['trend_type'], 
                              model_type_dict['season_type'], 
                              components_dict['components_number_ets'], 
                              components_dict['components_number_ets_seasonal'],
                              components_dict['components_number_arima'], 
                              explanatory_checked['xreg_number'], 
                              constants_checked['constant_required'],
                              observations_dict['y_in_sample'], 
                              observations_dict['ot'], 
                              any([t == "complete" or t == "backcasting" for t in initials_checked['initial_type']]))

    if not general['multisteps']:
        if general['loss'] == "likelihood":
            scale = scaler(general['distribution_new'], 
                         model_type_dict['error_type'], 
                         adam_fitted['errors'][observations_dict['ot_logical']],
                         adam_fitted['yFitted'][observations_dict['ot_logical']], 
                         observations_dict['obs_in_sample'], 
                         other)

            # Calculate the likelihood
            CFValue = -np.sum(calculate_likelihood(general['distribution_new'], 
                                                 model_type_dict['error_type'], 
                                                 observations_dict['y_in_sample'][observations_dict['ot_logical']],
                                                 adam_fitted['yFitted'][observations_dict['ot_logical']], 
                                                 scale, 
                                                 other))

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
                              any([t == "complete" or t == "backcasting" for t in initials_dict['initial_type']]))
            
            logLikReturn -= np.sum(np.log(np.abs(adam_fitted['y_fitted'])))

        return logLikReturn