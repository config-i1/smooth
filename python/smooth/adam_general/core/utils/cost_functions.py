import numpy as np
from numpy.linalg import eigvals
from python.smooth.adam_general.core.creator import filler
from python.smooth.adam_general.core.utils.utils import measurement_inverter, scaler, calculate_likelihood, calculate_entropy, calculate_multistep_loss
import numpy as np




def CF(B, etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
       ot, otLogical, occurrenceModel, obsInSample,
       componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
       componentsNumberARIMA,
       lags, lagsModel, lagsModelAll, lagsModelMax,
       indexLookupTable, profilesRecentTable,
       matVt, matWt, matF, vecG,
       persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
       persistenceSeasonalEstimate, persistenceXregEstimate, phiEstimate,
       initialType, initialEstimate,
       initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
       initialArimaEstimate, initialXregEstimate,
       arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate, arimaPolynomials,
       arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
       xregModel, xregNumber,
       xregParametersMissing, xregParametersIncluded,
       xregParametersEstimated, xregParametersPersistence,
       constantRequired, constantEstimate,
       bounds, loss, lossFunction, distribution, horizon, multisteps,
       denominator=None, yDenominator=None,
       other=None, otherParameterEstimate=False, lambda_param=None,
       arPolynomialMatrix=None, maPolynomialMatrix=None,
       regressors=None):  # Add regressors parameter here
    
    # Fill in the matrices
    adamElements = filler(B, etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                          componentsNumberETS, componentsNumberETSNonSeasonal,
                          componentsNumberETSSeasonal, componentsNumberARIMA,
                          lags, lagsModel, lagsModelMax,
                          matVt, matWt, matF, vecG,
                          persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                          persistenceSeasonalEstimate, persistenceXregEstimate,
                          phiEstimate,
                          initialType, initialEstimate,
                          initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                          initialArimaEstimate, initialXregEstimate,
                          arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                          arRequired, maRequired, armaParameters,
                          nonZeroARI, nonZeroMA, arimaPolynomials,
                          xregModel, xregNumber,
                          xregParametersMissing, xregParametersIncluded,
                          xregParametersEstimated, xregParametersPersistence, constantEstimate)

    # If we estimate parameters of distribution, take it from the B vector
    if otherParameterEstimate:
        other = abs(B[-1])
        if distribution in ["dgnorm", "dlgnorm"] and other < 0.25:
            return 1e10 / other

    # Check the bounds, classical restrictions
    if bounds == "usual":
        if arimaModel and any([arEstimate, maEstimate]):
            if arEstimate and sum(-adamElements['arimaPolynomials']['arPolynomial'][1:]) >= 1:
                arPolynomialMatrix[:, 0] = -adamElements['arimaPolynomials']['arPolynomial'][1:]
                arPolyroots = np.abs(eigvals(arPolynomialMatrix))
                if any(arPolyroots > 1):
                    return 1e100 * np.max(arPolyroots)
            
            if maEstimate and sum(adamElements['arimaPolynomials']['maPolynomial'][1:]) >= 1:
                maPolynomialMatrix[:, 0] = adamElements['arimaPolynomials']['maPolynomial'][1:]
                maPolyroots = np.abs(eigvals(maPolynomialMatrix))
                if any(maPolyroots > 1):
                    return 1e100 * np.max(np.abs(maPolyroots))

        if etsModel:
            if any(adamElements['vecG'][:componentsNumberETS] > 1) or any(adamElements['vecG'][:componentsNumberETS] < 0):
                return 1e300
            if modelIsTrendy:
                if adamElements['vecG'][1] > adamElements['vecG'][0]:
                    return 1e300
                if modelIsSeasonal and any(adamElements['vecG'][componentsNumberETSNonSeasonal:componentsNumberETSNonSeasonal+componentsNumberETSSeasonal] > (1 - adamElements['vecG'][0])):
                    return 1e300
            elif modelIsSeasonal and any(adamElements['vecG'][componentsNumberETSNonSeasonal:componentsNumberETSNonSeasonal+componentsNumberETSSeasonal] > (1 - adamElements['vecG'][0])):
                return 1e300

            if phiEstimate and (adamElements['matF'][1, 1] > 1 or adamElements['matF'][1, 1] < 0):
                return 1e300

        if xregModel and regressors == "adapt":
            if any(adamElements['vecG'][componentsNumberETS+componentsNumberARIMA:componentsNumberETS+componentsNumberARIMA+xregNumber] > 1) or \
               any(adamElements['vecG'][componentsNumberETS+componentsNumberARIMA:componentsNumberETS+componentsNumberARIMA+xregNumber] < 0):
                return 1e100 * np.max(np.abs(adamElements['vecG'][componentsNumberETS+componentsNumberARIMA:componentsNumberETS+componentsNumberARIMA+xregNumber] - 0.5))

    elif bounds == "admissible":
        if arimaModel:
            if arEstimate and (sum(-adamElements['arimaPolynomials']['arPolynomial'][1:]) >= 1 or sum(-adamElements['arimaPolynomials']['arPolynomial'][1:]) < 0):
                arPolynomialMatrix[:, 0] = -adamElements['arimaPolynomials']['arPolynomial'][1:]
                eigenValues = np.abs(eigvals(arPolynomialMatrix))
                if any(eigenValues > 1):
                    return 1e100 * np.max(eigenValues)

        if etsModel or arimaModel:
            if xregModel:
                if regressors == "adapt":
                    eigenValues = np.abs(eigvals(
                        adamElements['matF'] -
                        np.diag(adamElements['vecG'].flatten()) @
                        measurement_inverter(adamElements['matWt'][:obsInSample]).T @
                        adamElements['matWt'][:obsInSample] / obsInSample
                    ))
                else:
                    indices = np.arange(componentsNumberETS + componentsNumberARIMA)
                    eigenValues = np.abs(eigvals(
                        adamElements['matF'][np.ix_(indices, indices)] -
                        adamElements['vecG'][indices] @
                        adamElements['matWt'][obsInSample-1, indices]
                    ))
            else:
                if etsModel or (arimaModel and maEstimate and (sum(adamElements['arimaPolynomials']['maPolynomial'][1:]) >= 1 or sum(adamElements['arimaPolynomials']['maPolynomial'][1:]) < 0)):
                    eigenValues = np.abs(eigvals(
                        adamElements['matF'] -
                        adamElements['vecG'] @ adamElements['matWt'][obsInSample-1]
                    ))
                else:
                    eigenValues = np.array([0])

            if any(eigenValues > 1 + 1e-50):
                return 1e100 * np.max(eigenValues)

    # Write down the initials in the recent profile
    profilesRecentTable[:] = adamElements['matVt'][:, :lagsModelMax]

    # Fitter and the losses calculation
    adamFitted = adamFitterWrap(adamElements['matVt'], adamElements['matWt'], adamElements['matF'], adamElements['vecG'],
                                lagsModelAll, indexLookupTable, profilesRecentTable,
                                Etype, Ttype, Stype, componentsNumberETS, componentsNumberETSSeasonal,
                                componentsNumberARIMA, xregNumber, constantRequired,
                                yInSample, ot, any([t == "complete" or t == "backcasting" for t in initialType]))

    if not multisteps:
        if loss == "likelihood":
            scale = scaler(distribution, Etype, adamFitted['errors'][otLogical],
                           adamFitted['yFitted'][otLogical], obsInSample, other)

            # Calculate the likelihood
            CFValue = -np.sum(calculate_likelihood(distribution, Etype, yInSample[otLogical],
                                                   adamFitted['yFitted'][otLogical], scale, other))

            # Differential entropy for the logLik of occurrence model
            if occurrenceModel or any(~otLogical):
                CFValueEntropy = calculate_entropy(distribution, scale, other, obsZero,
                                                   adamFitted['yFitted'][~otLogical])
                if np.isnan(CFValueEntropy) or CFValueEntropy < 0:
                    CFValueEntropy = np.inf
                CFValue += CFValueEntropy

        elif loss == "MSE":
            CFValue = np.sum(adamFitted['errors']**2) / obsInSample
        elif loss == "MAE":
            CFValue = np.sum(np.abs(adamFitted['errors'])) / obsInSample
        elif loss == "HAM":
            CFValue = np.sum(np.sqrt(np.abs(adamFitted['errors']))) / obsInSample
        elif loss in ["LASSO", "RIDGE"]:
            persistenceToSkip = componentsNumberETS + persistenceXregEstimate * xregNumber + \
                                phiEstimate + sum(arOrders) + sum(maOrders)

            if phiEstimate:
                B[componentsNumberETS + persistenceXregEstimate * xregNumber] = \
                    1 - B[componentsNumberETS + persistenceXregEstimate * xregNumber]

            j = componentsNumberETS + persistenceXregEstimate * xregNumber + phiEstimate

            if arimaModel and (sum(maOrders) > 0 or sum(arOrders) > 0):
                for i in range(len(lags)):
                    B[j:j+arOrders[i]] = 1 - B[j:j+arOrders[i]]
                    j += arOrders[i] + maOrders[i]

            if any([t == "optimal" or t == "backcasting" for t in initialType]):
                if xregNumber > 0:
                    B = np.concatenate([B[:persistenceToSkip],
                                        B[-xregNumber:] / denominator if Etype == "A" else B[-xregNumber:]])
                else:
                    B = B[:persistenceToSkip]

            if Etype == "A":
                CFValue = (1 - lambda_param) * np.sqrt(np.sum((adamFitted['errors'] / yDenominator)**2) / obsInSample)
            else:  # "M"
                CFValue = (1 - lambda_param) * np.sqrt(np.sum(np.log(1 + adamFitted['errors'])**2) / obsInSample)

            if loss == "LASSO":
                CFValue += lambda_param * np.sum(np.abs(B))
            else:  # "RIDGE"
                CFValue += lambda_param * np.sqrt(np.sum(B**2))

        elif loss == "custom":
            CFValue = lossFunction(actual=yInSample, fitted=adamFitted['yFitted'], B=B)
    else:
        adamErrors = adamErrorerWrap(
            adamFitted['matVt'], adamElements['matWt'], adamElements['matF'],
            lagsModelAll, indexLookupTable, profilesRecentTable,
            Etype, Ttype, Stype,
            componentsNumberETS, componentsNumberETSSeasonal,
            componentsNumberARIMA, xregNumber, constantRequired, horizon,
            yInSample, ot
        )

        CFValue = calculate_multistep_loss(loss, adamErrors, obsInSample, horizon)

    if np.isnan(CFValue):
        CFValue = 1e300

    return CFValue

def logLikADAM(B, etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
               ot, otLogical, occurrenceModel, pFitted, obsInSample,
               componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
               componentsNumberARIMA, lags, lagsModel, lagsModelAll, lagsModelMax,
               indexLookupTable, profilesRecentTable, matVt, matWt, matF, vecG,
               persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
               persistenceSeasonalEstimate, persistenceXregEstimate, phiEstimate,
               initialType, initialEstimate, initialLevelEstimate, initialTrendEstimate,
               initialSeasonalEstimate, initialArimaEstimate, initialXregEstimate,
               arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate, arimaPolynomials,
               arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
               xregModel, xregNumber, xregParametersMissing, xregParametersIncluded,
               xregParametersEstimated, xregParametersPersistence, constantRequired,
               constantEstimate, bounds, loss, lossFunction, distribution, horizon,
               multisteps, denominator=None, yDenominator=None, other=None,
               otherParameterEstimate=False, lambda_param=None, arPolynomialMatrix=None,
               maPolynomialMatrix=None, hessianCalculation=False):
    

    if not multisteps:
        if loss in ["LASSO", "RIDGE"]:
            return 0
        else:
            distributionNew = {
                "MSE": "dnorm",
                "MAE": "dlaplace",
                "HAM": "ds"
            }.get(loss, distribution)

            lossNew = "likelihood" if loss in ["MSE", "MAE", "HAM"] else loss

            # Call CF function with bounds="none"
            logLikReturn = -CF(B, etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                            yInSample, ot, otLogical, occurrenceModel, obsInSample,
                            componentsNumberETS, componentsNumberETSSeasonal,
                            componentsNumberETSNonSeasonal, componentsNumberARIMA,
                            lags, lagsModel, lagsModelAll, lagsModelMax,
                            indexLookupTable, profilesRecentTable, matVt, matWt, matF, vecG,
                            persistenceEstimate, persistenceLevelEstimate,
                            persistenceTrendEstimate, persistenceSeasonalEstimate,
                            persistenceXregEstimate, phiEstimate, initialType,
                            initialEstimate, initialLevelEstimate, initialTrendEstimate,
                            initialSeasonalEstimate, initialArimaEstimate,
                            initialXregEstimate, arimaModel, nonZeroARI, nonZeroMA,
                            arEstimate, maEstimate, arimaPolynomials, arOrders, iOrders,
                            maOrders, arRequired, maRequired, armaParameters, xregModel,
                            xregNumber, xregParametersMissing, xregParametersIncluded,
                            xregParametersEstimated, xregParametersPersistence,
                            constantRequired, constantEstimate, bounds="none", loss=lossNew,
                            lossFunction=lossFunction, distribution=distributionNew,
                            horizon=horizon, multisteps=multisteps, denominator=denominator,
                            yDenominator=yDenominator, other=other,
                            otherParameterEstimate=otherParameterEstimate,
                            lambda_param=lambda_param, arPolynomialMatrix=arPolynomialMatrix,
                            maPolynomialMatrix=maPolynomialMatrix)

            # Handle occurrence model
            if occurrenceModel:
                if np.isinf(logLikReturn):
                    logLikReturn = 0
                if any(1 - pFitted[~otLogical] == 0) or any(pFitted[otLogical] == 0):
                    ptNew = pFitted[(pFitted != 0) & (pFitted != 1)]
                    otNew = ot[(pFitted != 0) & (pFitted != 1)]
                    if len(ptNew) == 0:
                        return logLikReturn
                    else:
                        return logLikReturn + np.sum(np.log(ptNew[otNew == 1])) + np.sum(np.log(1 - ptNew[otNew == 0]))
                else:
                    return logLikReturn + np.sum(np.log(pFitted[otLogical])) + np.sum(np.log(1 - pFitted[~otLogical]))
            else:
                return logLikReturn
            
    else:
        # Call CF function with bounds="none"
        logLikReturn = CF(B, etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                          yInSample, ot, otLogical, occurrenceModel, obsInSample,
                          componentsNumberETS, componentsNumberETSSeasonal,
                          componentsNumberETSNonSeasonal, componentsNumberARIMA,
                          lags, lagsModel, lagsModelAll, lagsModelMax,
                          indexLookupTable, profilesRecentTable, matVt, matWt, matF, vecG,
                          persistenceEstimate, persistenceLevelEstimate,
                          persistenceTrendEstimate, persistenceSeasonalEstimate,
                          persistenceXregEstimate, phiEstimate, initialType,
                          initialEstimate, initialLevelEstimate, initialTrendEstimate,
                          initialSeasonalEstimate, initialArimaEstimate,
                          initialXregEstimate, arimaModel, nonZeroARI, nonZeroMA,
                          arEstimate, maEstimate, arimaPolynomials, arOrders, iOrders,
                          maOrders, arRequired, maRequired, armaParameters, xregModel,
                          xregNumber, xregParametersMissing, xregParametersIncluded,
                          xregParametersEstimated, xregParametersPersistence,
                          constantRequired, constantEstimate, bounds="none", loss=loss,
                          lossFunction=lossFunction, distribution=distribution,
                          horizon=horizon, multisteps=multisteps, denominator=denominator,
                          yDenominator=yDenominator, other=other,
                          otherParameterEstimate=otherParameterEstimate,
                          lambda_param=lambda_param, arPolynomialMatrix=arPolynomialMatrix,
                          maPolynomialMatrix=maPolynomialMatrix)

        # Concentrated log-likelihoods for the multistep losses
        if loss in ["MSEh", "aMSEh", "TMSE", "aTMSE", "MSCE", "aMSCE"]:
            logLikReturn = -(obsInSample - horizon) / 2 * (np.log(2 * np.pi) + 1 + np.log(logLikReturn))
        elif loss in ["GTMSE", "aGTMSE"]:
            logLikReturn = -(obsInSample - horizon) / 2 * (np.log(2 * np.pi) + 1 + logLikReturn)
        elif loss in ["MAEh", "TMAE", "GTMAE", "MACE"]:
            logLikReturn = -(obsInSample - horizon) * (np.log(2) + 1 + np.log(logLikReturn))
        elif loss in ["HAMh", "THAM", "GTHAM", "CHAM"]:
            logLikReturn = -(obsInSample - horizon) * (np.log(4) + 2 + 2 * np.log(logLikReturn))
        elif loss in ["GPL", "aGPL"]:
            logLikReturn = -(obsInSample - horizon) / 2 * (horizon * np.log(2 * np.pi) + horizon + logLikReturn) / horizon

        # Make likelihood comparable
        logLikReturn = logLikReturn / (obsInSample - horizon) * obsInSample

        # Handle multiplicative model
        if Etype == "M":
            # Fill in the matrices
            adamElements = filler(B, etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                  componentsNumberETS, componentsNumberETSNonSeasonal,
                                  componentsNumberETSSeasonal, componentsNumberARIMA,
                                  lags, lagsModel, lagsModelMax, matVt, matWt, matF, vecG,
                                  persistenceEstimate, persistenceLevelEstimate,
                                  persistenceTrendEstimate, persistenceSeasonalEstimate,
                                  persistenceXregEstimate, phiEstimate, initialType,
                                  initialEstimate, initialLevelEstimate, initialTrendEstimate,
                                  initialSeasonalEstimate, initialArimaEstimate,
                                  initialXregEstimate, arimaModel, arEstimate, maEstimate,
                                  arOrders, iOrders, maOrders, arRequired, maRequired,
                                  armaParameters, nonZeroARI, nonZeroMA, arimaPolynomials,
                                  xregModel, xregNumber, xregParametersMissing,
                                  xregParametersIncluded, xregParametersEstimated,
                                  xregParametersPersistence, constantEstimate)

            # Write down the initials in the recent profile
            profilesRecentTable[:] = adamElements['matVt'][:, :lagsModelMax]

            # Fit the model again to extract the fitted values
            adamFitted = adamFitterWrap(adamElements['matVt'], adamElements['matWt'],
                                        adamElements['matF'], adamElements['vecG'],
                                        lagsModelAll, indexLookupTable, profilesRecentTable,
                                        Etype, Ttype, Stype, componentsNumberETS,
                                        componentsNumberETSSeasonal, componentsNumberARIMA,
                                        xregNumber, constantRequired, yInSample, ot,
                                        any(t in ["complete", "backcasting"] for t in initialType))
            
            logLikReturn -= np.sum(np.log(np.abs(adamFitted['yFitted'])))

        return logLikReturn