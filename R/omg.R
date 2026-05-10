omgLinkFunction <- function(fittedA, fittedB, EtypeA, EtypeB) {
    if(EtypeA == "A" && EtypeB == "A") {
        return(1 / (1 + exp(fittedB - fittedA)))
    }
    else if(EtypeA == "M" && EtypeB == "M") {
        return(1 / (1 + fittedB / fittedA))
    }
    else if(EtypeA == "M" && EtypeB == "A") {
        return(1 / (1 + exp(fittedB - log(fittedA))))
    }
    else {
        return(1 / (1 + exp(log(fittedB) - fittedA)))
    }
}

#' General occurrence model
#'
#' Fits two parallel ETS occurrence models (A: odds-ratio, B: inverse-odds-ratio)
#' jointly using a shared Bernoulli log-likelihood.  The combined probability
#' at each time point is \eqn{p_t = p_{At} / (p_{At} + p_{Bt})}.
#'
#' @param data Binary time series (0/1), vector or data frame.
#' @param modelA ETS model string for model A (default \code{"MNN"}).
#' @param modelB ETS model string for model B (default \code{"MNN"}).
#' @param ordersA ARIMA orders list for model A.
#' @param ordersB ARIMA orders list for model B.
#' @param constantA Logical, include constant in model A.
#' @param constantB Logical, include constant in model B.
#' @param formulaA Formula for exogenous variables in model A.
#' @param formulaB Formula for exogenous variables in model B.
#' @param regressorsA How to handle regressors in model A.
#' @param regressorsB How to handle regressors in model B.
#' @param persistenceA Persistence vector for model A.
#' @param persistenceB Persistence vector for model B.
#' @param phiA Damping parameter for model A.
#' @param phiB Damping parameter for model B.
#' @param armaA ARMA parameters for model A.
#' @param armaB ARMA parameters for model B.
#' @param etsA ETS variant for model A (\code{"conventional"} or \code{"adam"}).
#' @param etsB ETS variant for model B.
#' @param lags Seasonal lags (shared).
#' @param h Forecast horizon.
#' @param holdout If \code{TRUE}, hold out the last \code{h} observations.
#' @param initial Initialisation method (shared).
#' @param loss Loss function (shared).
#' @param ic Information criterion (shared).
#' @param bounds Parameter bounds type (shared).
#' @param silent If \code{TRUE}, suppress output.
#' @param ... Additional arguments passed to the optimiser.
#'
#' @return An object of class \code{c("omg","om","smooth")}.
#'
#' @seealso \link{om}, \link{forecast.omg}
#'
#' @examples
#' set.seed(41)
#' y <- rpois(100, 0.5)
#' m <- omg(y)
#' forecast(m, h=10)
#'
#' @export
omg <- function(data,
                modelA = "MNN", modelB = modelA,
                ordersA = list(ar=c(0), i=c(0), ma=c(0), select=FALSE),
                ordersB = ordersA,
                constantA = FALSE, constantB = constantA,
                formulaA = NULL, formulaB = formulaA,
                regressorsA = c("use","select","adapt"),
                regressorsB = regressorsA,
                persistenceA = NULL, persistenceB = persistenceA,
                phiA = NULL, phiB = phiA,
                armaA = NULL, armaB = armaA,
                etsA = c("conventional","adam"),
                etsB = etsA,
                lags = c(frequency(data)),
                h = 0, holdout = FALSE,
                initial = c("backcasting","optimal","two-stage","complete"),
                loss = c("likelihood","MSE","MAE","HAM","LASSO","RIDGE"),
                ic = c("AICc","AIC","BIC","BICc"),
                bounds = c("usual","admissible","none"),
                silent = TRUE, ...) {

    startTime <- Sys.time()
    cl        <- match.call()

    loss       <- match.arg(loss)
    ic         <- match.arg(ic)
    bounds     <- match.arg(bounds)
    initial    <- match.arg(initial)
    regressorsA <- match.arg(regressorsA)
    regressorsB <- match.arg(regressorsB)
    etsA       <- match.arg(etsA)
    etsB       <- match.arg(etsB, c("conventional","adam"))
    ellipsis   <- list(...)

    adamETSA <- (etsA == "adam")
    adamETSB <- (etsB == "adam")

    #### Shared data preparation ####
    if(is.data.frame(data)) {
        yName <- colnames(data)[1]
    } else {
        yName <- paste0(deparse(substitute(data)), collapse="")
        if(length(yName) == 0 || is.null(yName)) { yName <- "y" }
    }
    modelDo <- "estimate"

    # Convert one-sided formulas to two-sided (e.g. ~x -> y~x)
    makeTwoSided <- function(f, rName) {
        if(!is.null(f) && length(f) == 2) {
            return(as.formula(paste0(rName, "~", deparse(f[[2]]))))
        }
        return(f)
    }
    formulaA <- makeTwoSided(formulaA, yName)
    formulaB <- makeTwoSided(formulaB, yName)

    dataChecked <- adam_checkData(data, lags, h, holdout, yName, modelDo, formulaA)
    list2env(dataChecked, envir=environment())

    lossArg <- if(loss == "likelihood") "likelihood" else loss

    # Pre-select ETS model when wildcard characters (Z/X/Y) are present
    if(grepl("[ZXYFPS]", modelA)) {
        preA  <- om(data=data, model=modelA, lags=lags, orders=ordersA,
                    constant=constantA, formula=formulaA, regressors=regressorsA,
                    occurrence="odds-ratio", loss=loss, h=0, holdout=FALSE,
                    persistence=persistenceA, phi=phiA, initial=initial,
                    arma=armaA, ic=ic, bounds=bounds, ets=etsA, silent=TRUE, ...)
        modelA <- modelType(preA)
    }
    if(grepl("[ZXYFPS]", modelB)) {
        preB  <- om(data=data, model=modelB, lags=lags, orders=ordersB,
                    constant=constantB, formula=formulaB, regressors=regressorsB,
                    occurrence="inverse-odds-ratio", loss=loss, h=0, holdout=FALSE,
                    persistence=persistenceB, phi=phiB, initial=initial,
                    arma=armaB, ic=ic, bounds=bounds, ets=etsB, silent=TRUE, ...)
        modelB <- modelType(preB)
    }

    #### parametersChecker for A and B ####
    checkerA <- parametersChecker(data=data, model=modelA, lags=lags,
                                  formulaToUse=formulaA, orders=ordersA,
                                  constant=constantA, arma=armaA,
                                  persistence=persistenceA, phi=phiA,
                                  initial=initial, distribution="dnorm",
                                  loss=lossArg,
                                  h=h, holdout=holdout, occurrence="odds-ratio",
                                  ic=ic, bounds=bounds, regressors=regressorsA,
                                  yName=yName, silent=silent, modelDo=modelDo,
                                  ellipsis=ellipsis, fast=FALSE)

    checkerB <- parametersChecker(data=data, model=modelB, lags=lags,
                                  formulaToUse=formulaB, orders=ordersB,
                                  constant=constantB, arma=armaB,
                                  persistence=persistenceB, phi=phiB,
                                  initial=initial, distribution="dnorm",
                                  loss=lossArg,
                                  h=h, holdout=holdout, occurrence="inverse-odds-ratio",
                                  ic=ic, bounds=bounds, regressors=regressorsB,
                                  yName=yName, silent=silent, modelDo=modelDo,
                                  ellipsis=ellipsis, fast=FALSE)

    if(isTRUE(checkerA$select)) {
        stop("ARIMA order selection is not supported in omg(). Specify fixed orders.", call.=FALSE)
    }
    if(isTRUE(checkerB$select)) {
        stop("ARIMA order selection is not supported in omg(). Specify fixed orders.", call.=FALSE)
    }

    #### Shared binary indicators (same data for A and B) ####
    ot         <- checkerA$ot
    otLogical  <- checkerA$otLogical
    oInSample  <- matrix(as.numeric(ot), ncol=1)
    if(holdout) {
        oHoldout <- matrix(as.numeric(yHoldout != 0), ncol=1)
    }

    occurrenceModel <- FALSE
    yFitted         <- matrix(rep(mean(oInSample), obsInSample), ncol=1)
    refineHead      <- TRUE

    #### Optimiser settings ####
    optimSettings <- adam_checkOptimizer(ellipsis=ellipsis, loss=loss,
                                         distribution="dnorm",
                                         initialType=checkerA$initialType,
                                         lags=lags,
                                         arimaModel=checkerA$arimaModel)
    list2env(optimSettings, envir=environment())

    otLogicalInternal    <- otLogical
    otLogicalInternal[]  <- TRUE

    #### Inner joint estimator ####
    omgEstimator <- function() {

        omgCF_local <- function(B,
                                # A-side
                                etsModelA, EtypeA, TtypeA, StypeA,
                                modelIsTrendyA, modelIsSeasonalA,
                                componentsNumberETSA, componentsNumberETSNonSeasonalA,
                                componentsNumberETSSeasonalA, componentsNumberARIMAA,
                                lags, lagsModelA, lagsModelMaxA, lagsModelAllA,
                                indexLookupTableA, profilesRecentTableA,
                                matVtA, matWtA, matFA, vecGA,
                                persistenceEstimateA, persistenceLevelEstimateA,
                                persistenceTrendEstimateA, persistenceSeasonalEstimateA,
                                persistenceXregEstimateA, phiEstimateA,
                                initialTypeA, initialEstimateA,
                                initialLevelEstimateA, initialTrendEstimateA,
                                initialSeasonalEstimateA, initialArimaEstimateA,
                                initialXregEstimateA, initialArimaNumberA,
                                arimaModelA, arEstimateA, maEstimateA,
                                arOrdersA, iOrdersA, maOrdersA,
                                arRequiredA, maRequiredA, armaParametersA,
                                nonZeroARIA, nonZeroMAA, arimaPolynomialsA,
                                arPolynomialMatrixA, maPolynomialMatrixA,
                                xregModelA, xregNumberA,
                                xregParametersMissingA, xregParametersIncludedA,
                                xregParametersEstimatedA, xregParametersPersistenceA,
                                constantRequiredA, constantEstimateA,
                                adamCppA,
                                # B-side
                                etsModelB, EtypeB, TtypeB, StypeB,
                                modelIsTrendyB, modelIsSeasonalB,
                                componentsNumberETSB, componentsNumberETSNonSeasonalB,
                                componentsNumberETSSeasonalB, componentsNumberARIMAB,
                                lagsModelB, lagsModelMaxB, lagsModelAllB,
                                indexLookupTableB, profilesRecentTableB,
                                matVtB, matWtB, matFB, vecGB,
                                persistenceEstimateB, persistenceLevelEstimateB,
                                persistenceTrendEstimateB, persistenceSeasonalEstimateB,
                                persistenceXregEstimateB, phiEstimateB,
                                initialTypeB, initialEstimateB,
                                initialLevelEstimateB, initialTrendEstimateB,
                                initialSeasonalEstimateB, initialArimaEstimateB,
                                initialXregEstimateB, initialArimaNumberB,
                                arimaModelB, arEstimateB, maEstimateB,
                                arOrdersB, iOrdersB, maOrdersB,
                                arRequiredB, maRequiredB, armaParametersB,
                                nonZeroARIB, nonZeroMAB, arimaPolynomialsB,
                                arPolynomialMatrixB, maPolynomialMatrixB,
                                xregModelB, xregNumberB,
                                xregParametersMissingB, xregParametersIncludedB,
                                xregParametersEstimatedB, xregParametersPersistenceB,
                                constantRequiredB, constantEstimateB,
                                adamCppB,
                                # B-side scalars for omfitGeneral
                                nNonSeasonalB, nSeasonalB, nETSB,
                                nArimaB, nXregB, nComponentsB, adamETSB_flag,
                                # Shared
                                bounds, regressors,
                                ot, otLogical, obsInSample,
                                nIterations, refineHead, nParamsA) {

            B_A <- B[seq_len(nParamsA)]
            B_B <- B[seq_len(length(B) - nParamsA) + nParamsA]

            elemA <- adam_filler(B_A,
                                 etsModelA, EtypeA, TtypeA, StypeA,
                                 modelIsTrendyA, modelIsSeasonalA,
                                 componentsNumberETSA, componentsNumberETSNonSeasonalA,
                                 componentsNumberETSSeasonalA, componentsNumberARIMAA,
                                 lags, lagsModelA, lagsModelMaxA,
                                 matVtA, matWtA, matFA, vecGA,
                                 persistenceEstimateA, persistenceLevelEstimateA,
                                 persistenceTrendEstimateA, persistenceSeasonalEstimateA,
                                 persistenceXregEstimateA, phiEstimateA,
                                 initialTypeA, initialEstimateA,
                                 initialLevelEstimateA, initialTrendEstimateA,
                                 initialSeasonalEstimateA, initialArimaEstimateA,
                                 initialXregEstimateA,
                                 arimaModelA, arEstimateA, maEstimateA,
                                 arOrdersA, iOrdersA, maOrdersA,
                                 arRequiredA, maRequiredA, armaParametersA,
                                 nonZeroARIA, nonZeroMAA, arimaPolynomialsA,
                                 xregModelA, xregNumberA,
                                 xregParametersMissingA, xregParametersIncludedA,
                                 xregParametersEstimatedA, xregParametersPersistenceA,
                                 constantEstimateA, adamCppA,
                                 constantRequiredA, initialArimaNumberA)

            elemB <- adam_filler(B_B,
                                 etsModelB, EtypeB, TtypeB, StypeB,
                                 modelIsTrendyB, modelIsSeasonalB,
                                 componentsNumberETSB, componentsNumberETSNonSeasonalB,
                                 componentsNumberETSSeasonalB, componentsNumberARIMAB,
                                 lags, lagsModelB, lagsModelMaxB,
                                 matVtB, matWtB, matFB, vecGB,
                                 persistenceEstimateB, persistenceLevelEstimateB,
                                 persistenceTrendEstimateB, persistenceSeasonalEstimateB,
                                 persistenceXregEstimateB, phiEstimateB,
                                 initialTypeB, initialEstimateB,
                                 initialLevelEstimateB, initialTrendEstimateB,
                                 initialSeasonalEstimateB, initialArimaEstimateB,
                                 initialXregEstimateB,
                                 arimaModelB, arEstimateB, maEstimateB,
                                 arOrdersB, iOrdersB, maOrdersB,
                                 arRequiredB, maRequiredB, armaParametersB,
                                 nonZeroARIB, nonZeroMAB, arimaPolynomialsB,
                                 xregModelB, xregNumberB,
                                 xregParametersMissingB, xregParametersIncludedB,
                                 xregParametersEstimatedB, xregParametersPersistenceB,
                                 constantEstimateB, adamCppB,
                                 constantRequiredB, initialArimaNumberB)

            penaltyA <- adam_bounds_checker(elemA, elemA$arimaPolynomials, bounds,
                                            etsModelA, modelIsTrendyA, modelIsSeasonalA,
                                            componentsNumberETSA, componentsNumberETSNonSeasonalA,
                                            componentsNumberETSSeasonalA,
                                            arimaModelA, arEstimateA, maEstimateA,
                                            xregModelA, regressors, xregNumberA, componentsNumberARIMAA,
                                            lagsModelAllA, obsInSample,
                                            arPolynomialMatrixA, maPolynomialMatrixA, phiEstimateA)

            penaltyB <- adam_bounds_checker(elemB, elemB$arimaPolynomials, bounds,
                                            etsModelB, modelIsTrendyB, modelIsSeasonalB,
                                            componentsNumberETSB, componentsNumberETSNonSeasonalB,
                                            componentsNumberETSSeasonalB,
                                            arimaModelB, arEstimateB, maEstimateB,
                                            xregModelB, regressors, xregNumberB, componentsNumberARIMAB,
                                            lagsModelAllB, obsInSample,
                                            arPolynomialMatrixB, maPolynomialMatrixB, phiEstimateB)

            if(penaltyA + penaltyB > 0) { return(1e+300) }

            profilesRecentTableA[] <- elemA$matVt[, seq_len(lagsModelMaxA)]
            profilesRecentTableB[] <- elemB$matVt[, seq_len(lagsModelMaxB)]

            res <- adamCppA$omfitGeneral(
                elemA$matVt, elemA$matWt, elemA$matF, elemA$vecG,
                indexLookupTableA, profilesRecentTableA,
                EtypeB, TtypeB, StypeB,
                nNonSeasonalB, nSeasonalB, nETSB,
                nArimaB, nXregB, nComponentsB,
                constantRequiredB, adamETSB_flag,
                elemB$matVt, elemB$matWt, elemB$matF, elemB$vecG,
                indexLookupTableB, profilesRecentTableB,
                as.numeric(ot),
                any(initialTypeA == c("complete","backcasting")),
                nIterations, refineHead)

            pCombined <- omgLinkFunction(res$fittedA, res$fittedB, EtypeA, EtypeB)

            if(any(is.nan(pCombined)) || any(pCombined <= 0) || any(pCombined >= 1)) {
                return(1e+300)
            }

            return(-(sum(log(pCombined[otLogical])) +
                         sum(log(1 - pCombined[!otLogical]))))
        }

        # Architecture for A
        adamArchitectA <- adam_architector(
            checkerA$etsModel, checkerA$Etype, checkerA$Ttype, checkerA$Stype,
            lags, checkerA$lagsModelSeasonal,
            checkerA$xregNumber, obsInSample, checkerA$initialType,
            checkerA$arimaModel, checkerA$lagsModelARIMA,
            checkerA$xregModel, checkerA$constantRequired,
            checkerA$componentsNumberARIMA, obsAll, yIndexAll, yClasses, adamETSA)

        # Architecture for B — pad obsAll so B's lookup table covers A's loop bound
        lagsModelMaxA  <- adamArchitectA$lagsModelMax
        obsAllB_opt    <- max(obsAll, obsInSample + lagsModelMaxA)
        adamArchitectB <- adam_architector(
            checkerB$etsModel, checkerB$Etype, checkerB$Ttype, checkerB$Stype,
            lags, checkerB$lagsModelSeasonal,
            checkerB$xregNumber, obsInSample, checkerB$initialType,
            checkerB$arimaModel, checkerB$lagsModelARIMA,
            checkerB$xregModel, checkerB$constantRequired,
            checkerB$componentsNumberARIMA, obsAllB_opt, yIndexAll, yClasses, adamETSB)
        obsStatesB_opt <- max(adamArchitectB$obsStates, obsInSample + lagsModelMaxA)

        adamCppA <- adamArchitectA$adamCpp
        adamCppB <- adamArchitectB$adamCpp

        # Creator for A (Etype forced to "A")
        adamCreatedA <- adam_creator(
            checkerA$etsModel, Etype="A",
            Ttype=switch(checkerA$Ttype, "N"="N", "A"),
            Stype="A",
            adamArchitectA$modelIsTrendy, adamArchitectA$modelIsSeasonal,
            lags, adamArchitectA$lagsModel, checkerA$lagsModelARIMA,
            adamArchitectA$lagsModelAll, adamArchitectA$lagsModelMax,
            adamArchitectA$profilesRecentTable, FALSE,
            adamArchitectA$obsStates, obsInSample, obsAll,
            adamArchitectA$componentsNumberETS,
            adamArchitectA$componentsNumberETSSeasonal,
            adamArchitectA$componentsNamesETS, otLogicalInternal, ot,
            checkerA$persistence, checkerA$persistenceEstimate,
            checkerA$persistenceLevel, checkerA$persistenceLevelEstimate,
            checkerA$persistenceTrend, checkerA$persistenceTrendEstimate,
            checkerA$persistenceSeasonal, checkerA$persistenceSeasonalEstimate,
            checkerA$persistenceXreg, checkerA$persistenceXregEstimate,
            checkerA$persistenceXregProvided,
            checkerA$phi,
            checkerA$initialType, checkerA$initialEstimate,
            checkerA$initialLevel, checkerA$initialLevelEstimate,
            checkerA$initialTrend, checkerA$initialTrendEstimate,
            checkerA$initialSeasonal, checkerA$initialSeasonalEstimate,
            checkerA$initialArima, checkerA$initialArimaEstimate,
            checkerA$initialArimaNumber,
            checkerA$initialXregEstimate, checkerA$initialXregProvided,
            checkerA$arimaModel, checkerA$arRequired, checkerA$iRequired,
            checkerA$maRequired, checkerA$armaParameters,
            checkerA$arOrders, checkerA$iOrders, checkerA$maOrders,
            checkerA$componentsNumberARIMA, checkerA$componentsNamesARIMA,
            checkerA$xregModel, checkerA$xregModelInitials, checkerA$xregData,
            checkerA$xregNumber, checkerA$xregNames,
            checkerA$xregParametersPersistence,
            checkerA$constantRequired, checkerA$constantEstimate,
            checkerA$constantValue, checkerA$constantName,
            adamCppA,
            checkerA$arEstimate, checkerA$maEstimate, smoother,
            checkerA$nonZeroARI, checkerA$nonZeroMA)

        # Creator for B (Etype forced to "A")
        adamCreatedB <- adam_creator(
            checkerB$etsModel, Etype="A",
            Ttype=switch(checkerB$Ttype, "N"="N", "A"),
            Stype="A",
            adamArchitectB$modelIsTrendy, adamArchitectB$modelIsSeasonal,
            lags, adamArchitectB$lagsModel, checkerB$lagsModelARIMA,
            adamArchitectB$lagsModelAll, adamArchitectB$lagsModelMax,
            adamArchitectB$profilesRecentTable, FALSE,
            obsStatesB_opt, obsInSample, obsAll,
            adamArchitectB$componentsNumberETS,
            adamArchitectB$componentsNumberETSSeasonal,
            adamArchitectB$componentsNamesETS, otLogicalInternal, ot,
            checkerB$persistence, checkerB$persistenceEstimate,
            checkerB$persistenceLevel, checkerB$persistenceLevelEstimate,
            checkerB$persistenceTrend, checkerB$persistenceTrendEstimate,
            checkerB$persistenceSeasonal, checkerB$persistenceSeasonalEstimate,
            checkerB$persistenceXreg, checkerB$persistenceXregEstimate,
            checkerB$persistenceXregProvided,
            checkerB$phi,
            checkerB$initialType, checkerB$initialEstimate,
            checkerB$initialLevel, checkerB$initialLevelEstimate,
            checkerB$initialTrend, checkerB$initialTrendEstimate,
            checkerB$initialSeasonal, checkerB$initialSeasonalEstimate,
            checkerB$initialArima, checkerB$initialArimaEstimate,
            checkerB$initialArimaNumber,
            checkerB$initialXregEstimate, checkerB$initialXregProvided,
            checkerB$arimaModel, checkerB$arRequired, checkerB$iRequired,
            checkerB$maRequired, checkerB$armaParameters,
            checkerB$arOrders, checkerB$iOrders, checkerB$maOrders,
            checkerB$componentsNumberARIMA, checkerB$componentsNamesARIMA,
            checkerB$xregModel, checkerB$xregModelInitials, checkerB$xregData,
            checkerB$xregNumber, checkerB$xregNames,
            checkerB$xregParametersPersistence,
            checkerB$constantRequired, checkerB$constantEstimate,
            checkerB$constantValue, checkerB$constantName,
            adamCppB,
            checkerB$arEstimate, checkerB$maEstimate, smoother,
            checkerB$nonZeroARI, checkerB$nonZeroMA)

        # Initial transforms
        adamCreatedA$matVt <- om_initial_transform(
            adamCreatedA$matVt, "odds-ratio", checkerA$Etype,
            checkerA$Ttype, checkerA$Stype, checkerA$etsModel,
            adamArchitectA$modelIsTrendy, adamArchitectA$modelIsSeasonal,
            checkerA$initialLevelEstimate, checkerA$initialTrendEstimate,
            checkerA$initialSeasonalEstimate,
            adamArchitectA$componentsNumberETS,
            adamArchitectA$componentsNumberETSNonSeasonal,
            adamArchitectA$componentsNumberETSSeasonal,
            adamArchitectA$lagsModel, adamArchitectA$lagsModelMax,
            checkerA$lagsModelSeasonal, obsInSample, ot,
            checkerA$arimaModel, checkerA$componentsNumberARIMA,
            checkerA$initialArimaEstimate, checkerA$initialArimaNumber,
            checkerA$xregModel, checkerA$xregNumber,
            checkerA$initialXregEstimate,
            checkerA$constantRequired, checkerA$constantEstimate)

        adamCreatedB$matVt <- om_initial_transform(
            adamCreatedB$matVt, "inverse-odds-ratio", checkerB$Etype,
            checkerB$Ttype, checkerB$Stype, checkerB$etsModel,
            adamArchitectB$modelIsTrendy, adamArchitectB$modelIsSeasonal,
            checkerB$initialLevelEstimate, checkerB$initialTrendEstimate,
            checkerB$initialSeasonalEstimate,
            adamArchitectB$componentsNumberETS,
            adamArchitectB$componentsNumberETSNonSeasonal,
            adamArchitectB$componentsNumberETSSeasonal,
            adamArchitectB$lagsModel, adamArchitectB$lagsModelMax,
            checkerB$lagsModelSeasonal, obsInSample, ot,
            checkerB$arimaModel, checkerB$componentsNumberARIMA,
            checkerB$initialArimaEstimate, checkerB$initialArimaNumber,
            checkerB$xregModel, checkerB$xregNumber,
            checkerB$initialXregEstimate,
            checkerB$constantRequired, checkerB$constantEstimate)

        # Initial B vectors
        BValuesA <- adam_initialiser(
            checkerA$etsModel, checkerA$Etype, checkerA$Ttype, checkerA$Stype,
            adamArchitectA$modelIsTrendy, adamArchitectA$modelIsSeasonal,
            adamArchitectA$componentsNumberETSNonSeasonal,
            adamArchitectA$componentsNumberETSSeasonal,
            adamArchitectA$componentsNumberETS,
            lags, adamArchitectA$lagsModel, checkerA$lagsModelSeasonal,
            checkerA$lagsModelARIMA, adamArchitectA$lagsModelMax,
            adamCreatedA$matVt,
            checkerA$persistenceEstimate, checkerA$persistenceLevelEstimate,
            checkerA$persistenceTrendEstimate,
            checkerA$persistenceSeasonalEstimate,
            checkerA$persistenceXregEstimate,
            checkerA$phiEstimate, checkerA$initialType, checkerA$initialEstimate,
            checkerA$initialLevelEstimate, checkerA$initialTrendEstimate,
            checkerA$initialSeasonalEstimate,
            checkerA$initialArimaEstimate, checkerA$initialXregEstimate,
            checkerA$arimaModel, checkerA$arRequired, checkerA$maRequired,
            checkerA$arEstimate, checkerA$maEstimate,
            checkerA$arOrders, checkerA$maOrders,
            checkerA$componentsNumberARIMA, checkerA$componentsNamesARIMA,
            checkerA$initialArimaNumber,
            checkerA$xregModel, checkerA$xregNumber,
            checkerA$xregParametersEstimated, checkerA$xregParametersPersistence,
            checkerA$constantEstimate, checkerA$constantName,
            checkerA$otherParameterEstimate,
            adamCppA,
            etsA, bounds, ot, otLogicalInternal,
            checkerA$iOrders, checkerA$armaParameters, checkerA$other)

        BValuesB <- adam_initialiser(
            checkerB$etsModel, checkerB$Etype, checkerB$Ttype, checkerB$Stype,
            adamArchitectB$modelIsTrendy, adamArchitectB$modelIsSeasonal,
            adamArchitectB$componentsNumberETSNonSeasonal,
            adamArchitectB$componentsNumberETSSeasonal,
            adamArchitectB$componentsNumberETS,
            lags, adamArchitectB$lagsModel, checkerB$lagsModelSeasonal,
            checkerB$lagsModelARIMA, adamArchitectB$lagsModelMax,
            adamCreatedB$matVt,
            checkerB$persistenceEstimate, checkerB$persistenceLevelEstimate,
            checkerB$persistenceTrendEstimate,
            checkerB$persistenceSeasonalEstimate,
            checkerB$persistenceXregEstimate,
            checkerB$phiEstimate, checkerB$initialType, checkerB$initialEstimate,
            checkerB$initialLevelEstimate, checkerB$initialTrendEstimate,
            checkerB$initialSeasonalEstimate,
            checkerB$initialArimaEstimate, checkerB$initialXregEstimate,
            checkerB$arimaModel, checkerB$arRequired, checkerB$maRequired,
            checkerB$arEstimate, checkerB$maEstimate,
            checkerB$arOrders, checkerB$maOrders,
            checkerB$componentsNumberARIMA, checkerB$componentsNamesARIMA,
            checkerB$initialArimaNumber,
            checkerB$xregModel, checkerB$xregNumber,
            checkerB$xregParametersEstimated, checkerB$xregParametersPersistence,
            checkerB$constantEstimate, checkerB$constantName,
            checkerB$otherParameterEstimate,
            adamCppB,
            etsB, bounds, ot, otLogicalInternal,
            checkerB$iOrders, checkerB$armaParameters, checkerB$other)

        B_A <- BValuesA$B
        B_B <- BValuesB$B
        nParamsA <- length(B_A)
        B_used  <- c(B_A, B_B)
        lb      <- c(BValuesA$Bl, BValuesB$Bl)
        ub      <- c(BValuesA$Bu, BValuesB$Bu)

        # Mixed model checks
        EtypeA <- checkerA$Etype; TtypeA <- checkerA$Ttype; StypeA <- checkerA$Stype
        EtypeB <- checkerB$Etype; TtypeB <- checkerB$Ttype; StypeB <- checkerB$Stype

        if((EtypeA=="A" && TtypeA=="A" && StypeA=="M") ||
           (EtypeA=="A" && TtypeA=="M" && StypeA=="A") ||
           (EtypeA=="M" && TtypeA=="A" && StypeA=="A") ||
           (EtypeA=="A" && TtypeA=="M" && StypeA=="N") ||
           (EtypeA=="M" && TtypeA=="M" && StypeA=="A") ||
           (EtypeA=="M" && TtypeA=="N" && StypeA=="A") ||
           (EtypeA=="A" && TtypeA=="N" && StypeA=="M")) {
            B_used[seq_len(nParamsA)] <- 0
        }
        if((EtypeB=="A" && TtypeB=="A" && StypeB=="M") ||
           (EtypeB=="A" && TtypeB=="M" && StypeB=="A") ||
           (EtypeB=="M" && TtypeB=="A" && StypeB=="A") ||
           (EtypeB=="A" && TtypeB=="M" && StypeB=="N") ||
           (EtypeB=="M" && TtypeB=="M" && StypeB=="A") ||
           (EtypeB=="M" && TtypeB=="N" && StypeB=="A") ||
           (EtypeB=="A" && TtypeB=="N" && StypeB=="M")) {
            B_used[seq_len(length(B_B)) + nParamsA] <- 0
        }

        # ARIMA companion matrices for A
        if(checkerA$arimaModel) {
            arPolynomialMatrixA <- matrix(0, checkerA$arOrders %*% lags,
                                          checkerA$arOrders %*% lags)
            if(nrow(arPolynomialMatrixA) > 1) {
                arPolynomialMatrixA[2:nrow(arPolynomialMatrixA)-1,
                                    2:nrow(arPolynomialMatrixA)] <-
                    diag(nrow(arPolynomialMatrixA) - 1)
            }
            maPolynomialMatrixA <- matrix(0, checkerA$maOrders %*% lags,
                                          checkerA$maOrders %*% lags)
            if(nrow(maPolynomialMatrixA) > 1) {
                maPolynomialMatrixA[2:nrow(maPolynomialMatrixA)-1,
                                    2:nrow(maPolynomialMatrixA)] <-
                    diag(nrow(maPolynomialMatrixA) - 1)
            }
        } else {
            arPolynomialMatrixA <- maPolynomialMatrixA <- NULL
        }

        # ARIMA companion matrices for B
        if(checkerB$arimaModel) {
            arPolynomialMatrixB <- matrix(0, checkerB$arOrders %*% lags,
                                          checkerB$arOrders %*% lags)
            if(nrow(arPolynomialMatrixB) > 1) {
                arPolynomialMatrixB[2:nrow(arPolynomialMatrixB)-1,
                                    2:nrow(arPolynomialMatrixB)] <-
                    diag(nrow(arPolynomialMatrixB) - 1)
            }
            maPolynomialMatrixB <- matrix(0, checkerB$maOrders %*% lags,
                                          checkerB$maOrders %*% lags)
            if(nrow(maPolynomialMatrixB) > 1) {
                maPolynomialMatrixB[2:nrow(maPolynomialMatrixB)-1,
                                    2:nrow(maPolynomialMatrixB)] <-
                    diag(nrow(maPolynomialMatrixB) - 1)
            }
        } else {
            arPolynomialMatrixB <- maPolynomialMatrixB <- NULL
        }

        # B-side scalars for omfitGeneral
        nNonSeasonalB <- adamArchitectB$componentsNumberETSNonSeasonal
        nSeasonalB    <- adamArchitectB$componentsNumberETSSeasonal
        nETSB         <- adamArchitectB$componentsNumberETS
        nArimaB       <- checkerB$componentsNumberARIMA
        nXregB        <- checkerB$xregNumber
        nComponentsB  <- length(adamArchitectB$lagsModelAll)

        # Explicit nloptr args
        nloptrArgs <- list(
            # A-side
            etsModelA=checkerA$etsModel, EtypeA=EtypeA, TtypeA=TtypeA, StypeA=StypeA,
            modelIsTrendyA=adamArchitectA$modelIsTrendy,
            modelIsSeasonalA=adamArchitectA$modelIsSeasonal,
            componentsNumberETSA=adamArchitectA$componentsNumberETS,
            componentsNumberETSNonSeasonalA=adamArchitectA$componentsNumberETSNonSeasonal,
            componentsNumberETSSeasonalA=adamArchitectA$componentsNumberETSSeasonal,
            componentsNumberARIMAA=checkerA$componentsNumberARIMA,
            lags=lags,
            lagsModelA=adamArchitectA$lagsModel,
            lagsModelMaxA=adamArchitectA$lagsModelMax,
            lagsModelAllA=adamArchitectA$lagsModelAll,
            indexLookupTableA=adamArchitectA$indexLookupTable,
            profilesRecentTableA=adamArchitectA$profilesRecentTable,
            matVtA=adamCreatedA$matVt, matWtA=adamCreatedA$matWt,
            matFA=adamCreatedA$matF, vecGA=adamCreatedA$vecG,
            persistenceEstimateA=checkerA$persistenceEstimate,
            persistenceLevelEstimateA=checkerA$persistenceLevelEstimate,
            persistenceTrendEstimateA=checkerA$persistenceTrendEstimate,
            persistenceSeasonalEstimateA=checkerA$persistenceSeasonalEstimate,
            persistenceXregEstimateA=checkerA$persistenceXregEstimate,
            phiEstimateA=checkerA$phiEstimate,
            initialTypeA=checkerA$initialType, initialEstimateA=checkerA$initialEstimate,
            initialLevelEstimateA=checkerA$initialLevelEstimate,
            initialTrendEstimateA=checkerA$initialTrendEstimate,
            initialSeasonalEstimateA=checkerA$initialSeasonalEstimate,
            initialArimaEstimateA=checkerA$initialArimaEstimate,
            initialXregEstimateA=checkerA$initialXregEstimate,
            initialArimaNumberA=checkerA$initialArimaNumber,
            arimaModelA=checkerA$arimaModel,
            arEstimateA=checkerA$arEstimate, maEstimateA=checkerA$maEstimate,
            arOrdersA=checkerA$arOrders, iOrdersA=checkerA$iOrders,
            maOrdersA=checkerA$maOrders,
            arRequiredA=checkerA$arRequired, maRequiredA=checkerA$maRequired,
            armaParametersA=checkerA$armaParameters,
            nonZeroARIA=checkerA$nonZeroARI, nonZeroMAA=checkerA$nonZeroMA,
            arimaPolynomialsA=adamCreatedA$arimaPolynomials,
            arPolynomialMatrixA=arPolynomialMatrixA,
            maPolynomialMatrixA=maPolynomialMatrixA,
            xregModelA=checkerA$xregModel, xregNumberA=checkerA$xregNumber,
            xregParametersMissingA=checkerA$xregParametersMissing,
            xregParametersIncludedA=checkerA$xregParametersIncluded,
            xregParametersEstimatedA=checkerA$xregParametersEstimated,
            xregParametersPersistenceA=checkerA$xregParametersPersistence,
            constantRequiredA=checkerA$constantRequired,
            constantEstimateA=checkerA$constantEstimate,
            adamCppA=adamCppA,
            # B-side
            etsModelB=checkerB$etsModel, EtypeB=EtypeB, TtypeB=TtypeB, StypeB=StypeB,
            modelIsTrendyB=adamArchitectB$modelIsTrendy,
            modelIsSeasonalB=adamArchitectB$modelIsSeasonal,
            componentsNumberETSB=adamArchitectB$componentsNumberETS,
            componentsNumberETSNonSeasonalB=adamArchitectB$componentsNumberETSNonSeasonal,
            componentsNumberETSSeasonalB=adamArchitectB$componentsNumberETSSeasonal,
            componentsNumberARIMAB=checkerB$componentsNumberARIMA,
            lagsModelB=adamArchitectB$lagsModel,
            lagsModelMaxB=adamArchitectB$lagsModelMax,
            lagsModelAllB=adamArchitectB$lagsModelAll,
            indexLookupTableB=adamArchitectB$indexLookupTable,
            profilesRecentTableB=adamArchitectB$profilesRecentTable,
            matVtB=adamCreatedB$matVt, matWtB=adamCreatedB$matWt,
            matFB=adamCreatedB$matF, vecGB=adamCreatedB$vecG,
            persistenceEstimateB=checkerB$persistenceEstimate,
            persistenceLevelEstimateB=checkerB$persistenceLevelEstimate,
            persistenceTrendEstimateB=checkerB$persistenceTrendEstimate,
            persistenceSeasonalEstimateB=checkerB$persistenceSeasonalEstimate,
            persistenceXregEstimateB=checkerB$persistenceXregEstimate,
            phiEstimateB=checkerB$phiEstimate,
            initialTypeB=checkerB$initialType, initialEstimateB=checkerB$initialEstimate,
            initialLevelEstimateB=checkerB$initialLevelEstimate,
            initialTrendEstimateB=checkerB$initialTrendEstimate,
            initialSeasonalEstimateB=checkerB$initialSeasonalEstimate,
            initialArimaEstimateB=checkerB$initialArimaEstimate,
            initialXregEstimateB=checkerB$initialXregEstimate,
            initialArimaNumberB=checkerB$initialArimaNumber,
            arimaModelB=checkerB$arimaModel,
            arEstimateB=checkerB$arEstimate, maEstimateB=checkerB$maEstimate,
            arOrdersB=checkerB$arOrders, iOrdersB=checkerB$iOrders,
            maOrdersB=checkerB$maOrders,
            arRequiredB=checkerB$arRequired, maRequiredB=checkerB$maRequired,
            armaParametersB=checkerB$armaParameters,
            nonZeroARIB=checkerB$nonZeroARI, nonZeroMAB=checkerB$nonZeroMA,
            arimaPolynomialsB=adamCreatedB$arimaPolynomials,
            arPolynomialMatrixB=arPolynomialMatrixB,
            maPolynomialMatrixB=maPolynomialMatrixB,
            xregModelB=checkerB$xregModel, xregNumberB=checkerB$xregNumber,
            xregParametersMissingB=checkerB$xregParametersMissing,
            xregParametersIncludedB=checkerB$xregParametersIncluded,
            xregParametersEstimatedB=checkerB$xregParametersEstimated,
            xregParametersPersistenceB=checkerB$xregParametersPersistence,
            constantRequiredB=checkerB$constantRequired,
            constantEstimateB=checkerB$constantEstimate,
            adamCppB=adamCppB,
            # B-side scalars for omfitGeneral
            nNonSeasonalB=nNonSeasonalB, nSeasonalB=nSeasonalB, nETSB=nETSB,
            nArimaB=nArimaB, nXregB=nXregB, nComponentsB=nComponentsB,
            adamETSB_flag=adamETSB,
            # Shared
            bounds=bounds, regressors=regressorsA,
            ot=ot, otLogical=otLogical, obsInSample=obsInSample,
            nIterations=nIterations, refineHead=refineHead,
            nParamsA=nParamsA)

        if(length(B_used) == 0){
            res <- list(solution=B_used,
                        objective=do.call(omgCF_local, c(list(B=B_used), nloptrArgs)));
        } else {
            maxevalUsed <- if(is.null(maxeval)) length(B_used) * 40L else maxeval
            res <- suppressWarnings(
                do.call(nloptr,
                        c(list(x0=B_used, eval_f=omgCF_local, lb=lb, ub=ub,
                               opts=list(algorithm=algorithm, xtol_rel=xtol_rel, xtol_abs=xtol_abs,
                                         ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                         maxeval=maxevalUsed, maxtime=maxtime,
                                         print_level=print_level)),
                          nloptrArgs)))
            res$call <- quote(nloptr(x0=B_used, eval_f=omgCF_local, lb=lb, ub=ub, opts=opts));

            if(is.infinite(res$objective) || res$objective == 1e+300) {
                B_used[] <- c(BValuesA$B, BValuesB$B)
                res <- suppressWarnings(
                    do.call(nloptr,
                            c(list(x0=B_used, eval_f=omgCF_local, lb=lb, ub=ub,
                                   opts=list(algorithm=algorithm, xtol_rel=xtol_rel, xtol_abs=xtol_abs,
                                             ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                             maxeval=maxevalUsed, maxtime=maxtime,
                                             print_level=print_level)),
                              nloptrArgs)))
                res$call <- quote(nloptr(x0=B_used, eval_f=omgCF_local, lb=lb, ub=ub, opts=opts));
            }
        }

        B_joint <- res$solution
        names(B_joint) <- c(names(BValuesA$B), names(BValuesB$B))
        CFValue <- res$objective

        return(list(
            B_A       = B_joint[seq_len(nParamsA)],
            B_B       = B_joint[seq_len(length(B_joint) - nParamsA) + nParamsA],
            CFValue   = CFValue,
            logLikValue = -CFValue,
            nParamsA  = nParamsA,
            nParamsB  = length(B_joint) - nParamsA,
            # Pass through model info needed by omgFinalFitA/B
            adamArchitectA = adamArchitectA,
            adamArchitectB = adamArchitectB,
            adamCppA = adamCppA,
            adamCppB = adamCppB))
    }

    #### Final fit closures ####

    omgFinalFitA <- function(resA, hLocal=0) {
        otLogicalInternal <- otLogical
        otLogicalInternal[] <- TRUE

        adamArchitect <- adam_architector(
            checkerA$etsModel, checkerA$Etype, checkerA$Ttype, checkerA$Stype,
            lags, checkerA$lagsModelSeasonal,
            checkerA$xregNumber, obsInSample, checkerA$initialType,
            checkerA$arimaModel, checkerA$lagsModelARIMA,
            checkerA$xregModel, checkerA$constantRequired,
            checkerA$componentsNumberARIMA, obsAll, yIndexAll, yClasses, adamETSA)

        adamCreated <- adam_creator(
            checkerA$etsModel, checkerA$Etype, checkerA$Ttype, checkerA$Stype,
            resA$modelIsTrendy, resA$modelIsSeasonal,
            lags, adamArchitect$lagsModel, checkerA$lagsModelARIMA,
            adamArchitect$lagsModelAll, adamArchitect$lagsModelMax,
            adamArchitect$profilesRecentTable, FALSE,
            adamArchitect$obsStates, obsInSample, obsAll,
            adamArchitect$componentsNumberETS, adamArchitect$componentsNumberETSSeasonal,
            adamArchitect$componentsNamesETS, otLogicalInternal, ot,
            checkerA$persistence, resA$persistenceEstimate,
            checkerA$persistenceLevel, resA$persistenceLevelEstimate,
            checkerA$persistenceTrend, resA$persistenceTrendEstimate,
            checkerA$persistenceSeasonal, resA$persistenceSeasonalEstimate,
            checkerA$persistenceXreg, resA$persistenceXregEstimate,
            checkerA$persistenceXregProvided,
            checkerA$phi,
            checkerA$initialType, resA$initialEstimate,
            checkerA$initialLevel, resA$initialLevelEstimate,
            checkerA$initialTrend, resA$initialTrendEstimate,
            checkerA$initialSeasonal, resA$initialSeasonalEstimate,
            checkerA$initialArima, resA$initialArimaEstimate,
            checkerA$initialArimaNumber,
            resA$initialXregEstimate, checkerA$initialXregProvided,
            resA$arimaModel, resA$arRequired, resA$iRequired, resA$maRequired,
            checkerA$armaParameters,
            resA$arOrders, resA$iOrders, resA$maOrders,
            checkerA$componentsNumberARIMA, checkerA$componentsNamesARIMA,
            checkerA$xregModel, checkerA$xregModelInitials, checkerA$xregData,
            checkerA$xregNumber, checkerA$xregNames,
            checkerA$xregParametersPersistence,
            checkerA$constantRequired, checkerA$constantEstimate,
            checkerA$constantValue, checkerA$constantName,
            adamArchitect$adamCpp,
            resA$arEstimate, resA$maEstimate, smoother,
            checkerA$nonZeroARI, checkerA$nonZeroMA)

        adamCreated$matVt <- om_initial_transform(
            adamCreated$matVt, "odds-ratio", resA$Etype, resA$Ttype, resA$Stype,
            resA$etsModel,
            resA$modelIsTrendy, resA$modelIsSeasonal,
            resA$initialLevelEstimate, resA$initialTrendEstimate,
            resA$initialSeasonalEstimate,
            adamArchitect$componentsNumberETS,
            adamArchitect$componentsNumberETSNonSeasonal,
            adamArchitect$componentsNumberETSSeasonal,
            adamArchitect$lagsModel, adamArchitect$lagsModelMax,
            checkerA$lagsModelSeasonal, obsInSample, ot,
            resA$arimaModel, checkerA$componentsNumberARIMA,
            resA$initialArimaEstimate, checkerA$initialArimaNumber,
            checkerA$xregModel, checkerA$xregNumber, resA$initialXregEstimate,
            checkerA$constantRequired, checkerA$constantEstimate)

        adamFilled <- adam_filler(resA$B,
                                  resA$etsModel, resA$Etype, resA$Ttype, resA$Stype,
                                  resA$modelIsTrendy, resA$modelIsSeasonal,
                                  adamArchitect$componentsNumberETS,
                                  adamArchitect$componentsNumberETSNonSeasonal,
                                  adamArchitect$componentsNumberETSSeasonal,
                                  checkerA$componentsNumberARIMA,
                                  lags, adamArchitect$lagsModel, adamArchitect$lagsModelMax,
                                  adamCreated$matVt, adamCreated$matWt, adamCreated$matF, adamCreated$vecG,
                                  resA$persistenceEstimate, resA$persistenceLevelEstimate,
                                  resA$persistenceTrendEstimate, resA$persistenceSeasonalEstimate,
                                  resA$persistenceXregEstimate, resA$phiEstimate,
                                  checkerA$initialType, resA$initialEstimate,
                                  resA$initialLevelEstimate, resA$initialTrendEstimate,
                                  resA$initialSeasonalEstimate, resA$initialArimaEstimate,
                                  resA$initialXregEstimate,
                                  resA$arimaModel, resA$arEstimate, resA$maEstimate,
                                  resA$arOrders, resA$iOrders, resA$maOrders,
                                  resA$arRequired, resA$maRequired, checkerA$armaParameters,
                                  checkerA$nonZeroARI, checkerA$nonZeroMA, adamCreated$arimaPolynomials,
                                  checkerA$xregModel, checkerA$xregNumber,
                                  checkerA$xregParametersMissing, checkerA$xregParametersIncluded,
                                  checkerA$xregParametersEstimated, checkerA$xregParametersPersistence,
                                  checkerA$constantEstimate, adamArchitect$adamCpp,
                                  checkerA$constantRequired, checkerA$initialArimaNumber)

        prof <- adamFilled$matVt[, seq_len(adamArchitect$lagsModelMax), drop=FALSE]
        adamFitted <- adamArchitect$adamCpp$fit(
            adamFilled$matVt, adamFilled$matWt, adamFilled$matF, adamFilled$vecG,
            adamArchitect$indexLookupTable, prof,
            as.numeric(ot), as.numeric(ot),
            any(checkerA$initialType == c("complete","backcasting")),
            nIterations, refineHead, "o")

        yFitted <- adamFitted$fitted

        if(is.null(resA$logLikADAMValue)) {
            pFitted  <- omLinkFunction(as.numeric(yFitted), resA$Etype, "odds-ratio")
            ot_vec   <- as.numeric(oInSample)
            ll <- sum(ot_vec * log(pmax(pFitted, 1e-15)) +
                          (1 - ot_vec) * log(pmax(1 - pFitted, 1e-15)))
            resA$logLikADAMValue <- ll
            resA$CFValue <- -ll
        }

        if(hLocal > 0) {
            yForecast <- adamArchitect$adamCpp$forecast(
                tail(adamFilled$matWt, hLocal),
                adamFilled$matF,
                adamArchitect$indexLookupTable[,
                                               adamArchitect$lagsModelMax + obsInSample +
                                                   seq_len(hLocal), drop=FALSE],
                adamFitted$profile, hLocal)$forecast
        }

        # States
        statesRaw <- adamFitted$states[,
                                       (adamArchitect$lagsModelMax+1):ncol(adamFitted$states), drop=FALSE]
        compNames <- rownames(adamCreated$matVt)
        if(!is.null(compNames)) rownames(statesRaw) <- compNames

        if(any(yClasses == "ts")) {
            yFitted <- ts(yFitted, start=yStart, frequency=yFrequency)
            errors  <- ts(as.numeric(oInSample) - yFitted, start=yStart, frequency=yFrequency)
            matVt   <- ts(t(statesRaw), start=yStart, frequency=yFrequency)
        } else {
            yFitted <- zoo(yFitted, order.by=yInSampleIndex)
            errors  <- zoo(as.numeric(oInSample) - yFitted, order.by=yInSampleIndex)
            matVt   <- zoo(t(statesRaw), order.by=yInSampleIndex)
        }

        if(hLocal > 0 && !is.null(yForecast)) {
            if(any(yClasses == "ts")) {
                yForecast <- ts(yForecast, start=yForecastStart, frequency=yFrequency)
            } else {
                yForecast <- zoo(yForecast, order.by=yForecastIndex)
            }
        } else {
            yForecast <- if(any(yClasses=="ts")) {
                ts(NA, start=yForecastStart, frequency=yFrequency)
            } else {
                zoo(NA, order.by=yForecastIndex[1])
            }
        }

        modelStr  <- paste0(resA$Etype, resA$Ttype,
                            "d"[resA$phiEstimate], resA$Stype)
        modelName <- adam_model_name(
            resA$etsModel, modelStr,
            checkerA$xregModel, resA$arimaModel,
            resA$arOrders, resA$iOrders, resA$maOrders, lags,
            regressorsA, checkerA$constantRequired, checkerA$constantName,
            "odds-ratio", adamArchitect$componentsNumberETSSeasonal,
            prefix = "o")

        vecGFinal <- adamFilled$vecG
        if(adamArchitect$componentsNumberETS > 0) {
            persistenceVec <- as.vector(vecGFinal)[seq_len(adamArchitect$componentsNumberETS)]
            names(persistenceVec) <- rownames(vecGFinal)[seq_len(adamArchitect$componentsNumberETS)]
        } else {
            persistenceVec <- numeric(0)
        }

        initialCollected <- adam_initial_collector(
            adamFitted$states[, seq_len(adamArchitect$lagsModelMax), drop=FALSE],
            resA$etsModel, resA$modelIsTrendy, resA$modelIsSeasonal,
            adamArchitect$lagsModel, adamArchitect$lagsModelMax,
            resA$initialLevelEstimate, resA$initialTrendEstimate,
            resA$initialSeasonalEstimate,
            adamArchitect$componentsNumberETSSeasonal,
            resA$arimaModel, resA$initialArimaEstimate,
            checkerA$initialArima, checkerA$initialArimaNumber,
            adamArchitect$componentsNumberETS, checkerA$componentsNumberARIMA,
            adamFilled$arimaPolynomials, resA$Etype,
            checkerA$xregModel, resA$initialXregEstimate, checkerA$xregNumber)

        if(resA$arimaModel && (resA$arRequired || resA$maRequired)) {
            armaParametersList <- vector("list", resA$arRequired + resA$maRequired)
            j <- 1L
            if(resA$arRequired && resA$arEstimate) {
                armaParametersList[[j]] <- resA$B[nchar(names(resA$B))>3 &
                                                      substr(names(resA$B),1,3)=="phi"]
                names(armaParametersList)[j] <- "ar"; j <- j + 1L
            } else if(resA$arRequired) {
                armaParametersList[[j]] <- checkerA$armaParameters[
                    substr(names(checkerA$armaParameters),1,3)=="phi"]
                names(armaParametersList)[j] <- "ar"; j <- j + 1L
            }
            if(resA$maRequired && resA$maEstimate) {
                armaParametersList[[j]] <- resA$B[substr(names(resA$B),1,5)=="theta"]
                names(armaParametersList)[j] <- "ma"
            } else if(resA$maRequired) {
                armaParametersList[[j]] <- checkerA$armaParameters[
                    substr(names(checkerA$armaParameters),1,5)=="theta"]
                names(armaParametersList)[j] <- "ma"
            }
        } else {
            armaParametersList <- NULL
        }

        parNum <- checkerA$parametersNumber
        parNum[1,1] <- resA$nParamEstimated
        parNum[1,5] <- sum(parNum[1,1:4])
        parNum[2,5] <- sum(parNum[2,1:4])

        subModel <- list(
            model       = modelName,
            timeElapsed = Sys.time() - startTime,
            data        = yInSample,
            fitted      = yFitted,
            residuals   = errors,
            forecast    = yForecast,
            states      = matVt,
            profile     = adamFitted$profile,
            profileInitial = NULL,
            persistence = persistenceVec,
            phi         = if(resA$phiEstimate) resA$B["phi"] else checkerA$phi,
            transition  = adamFilled$matF,
            measurement = adamFilled$matWt,
            initial     = initialCollected$initialValue,
            initialType = checkerA$initialType,
            initialEstimated = initialCollected$initialEstimated,
            orders      = list(ar=resA$arOrders, i=resA$iOrders, ma=resA$maOrders),
            arma        = armaParametersList,
            constant    = if(checkerA$constantRequired) {
                if(checkerA$constantEstimate) resA$B[checkerA$constantName] else checkerA$constantValue
            } else NULL,
            nParam      = parNum,
            occurrence  = "odds-ratio",
            formula     = checkerA$formula,
            regressors  = regressorsA,
            loss        = loss,
            lossValue   = resA$CFValue,
            lossFunction = NULL,
            logLik      = resA$logLikADAMValue,
            distribution = "plogis",
            scale       = NA,
            other       = NULL,
            B           = resA$B,
            lags        = lags,
            lagsAll     = adamArchitect$lagsModelAll,
            ets         = resA$etsModel,
            res         = resA,
            FI          = NULL,
            adamCpp     = adamArchitect$adamCpp,
            bounds      = bounds,
            call        = cl)

        if(holdout) {
            subModel$holdout  <- oHoldout
            subModel$accuracy <- measures(as.vector(oHoldout), yForecast,
                                          as.vector(oInSample))
        }

        class(subModel) <- c("om","adam","smooth")
        return(subModel)
    }

    omgFinalFitB <- function(resB, hLocal=0) {
        otLogicalInternal <- otLogical
        otLogicalInternal[] <- TRUE

        adamArchitect <- adam_architector(
            checkerB$etsModel, checkerB$Etype, checkerB$Ttype, checkerB$Stype,
            lags, checkerB$lagsModelSeasonal,
            checkerB$xregNumber, obsInSample, checkerB$initialType,
            checkerB$arimaModel, checkerB$lagsModelARIMA,
            checkerB$xregModel, checkerB$constantRequired,
            checkerB$componentsNumberARIMA, obsAll, yIndexAll, yClasses, adamETSB)

        adamCreated <- adam_creator(
            checkerB$etsModel, checkerB$Etype, checkerB$Ttype, checkerB$Stype,
            resB$modelIsTrendy, resB$modelIsSeasonal,
            lags, adamArchitect$lagsModel, checkerB$lagsModelARIMA,
            adamArchitect$lagsModelAll, adamArchitect$lagsModelMax,
            adamArchitect$profilesRecentTable, FALSE,
            adamArchitect$obsStates, obsInSample, obsAll,
            adamArchitect$componentsNumberETS, adamArchitect$componentsNumberETSSeasonal,
            adamArchitect$componentsNamesETS, otLogicalInternal, ot,
            checkerB$persistence, resB$persistenceEstimate,
            checkerB$persistenceLevel, resB$persistenceLevelEstimate,
            checkerB$persistenceTrend, resB$persistenceTrendEstimate,
            checkerB$persistenceSeasonal, resB$persistenceSeasonalEstimate,
            checkerB$persistenceXreg, resB$persistenceXregEstimate,
            checkerB$persistenceXregProvided,
            checkerB$phi,
            checkerB$initialType, resB$initialEstimate,
            checkerB$initialLevel, resB$initialLevelEstimate,
            checkerB$initialTrend, resB$initialTrendEstimate,
            checkerB$initialSeasonal, resB$initialSeasonalEstimate,
            checkerB$initialArima, resB$initialArimaEstimate,
            checkerB$initialArimaNumber,
            resB$initialXregEstimate, checkerB$initialXregProvided,
            resB$arimaModel, resB$arRequired, resB$iRequired, resB$maRequired,
            checkerB$armaParameters,
            resB$arOrders, resB$iOrders, resB$maOrders,
            checkerB$componentsNumberARIMA, checkerB$componentsNamesARIMA,
            checkerB$xregModel, checkerB$xregModelInitials, checkerB$xregData,
            checkerB$xregNumber, checkerB$xregNames,
            checkerB$xregParametersPersistence,
            checkerB$constantRequired, checkerB$constantEstimate,
            checkerB$constantValue, checkerB$constantName,
            adamArchitect$adamCpp,
            resB$arEstimate, resB$maEstimate, smoother,
            checkerB$nonZeroARI, checkerB$nonZeroMA)

        adamCreated$matVt <- om_initial_transform(
            adamCreated$matVt, "inverse-odds-ratio", resB$Etype, resB$Ttype, resB$Stype,
            resB$etsModel,
            resB$modelIsTrendy, resB$modelIsSeasonal,
            resB$initialLevelEstimate, resB$initialTrendEstimate,
            resB$initialSeasonalEstimate,
            adamArchitect$componentsNumberETS,
            adamArchitect$componentsNumberETSNonSeasonal,
            adamArchitect$componentsNumberETSSeasonal,
            adamArchitect$lagsModel, adamArchitect$lagsModelMax,
            checkerB$lagsModelSeasonal, obsInSample, ot,
            resB$arimaModel, checkerB$componentsNumberARIMA,
            resB$initialArimaEstimate, checkerB$initialArimaNumber,
            checkerB$xregModel, checkerB$xregNumber, resB$initialXregEstimate,
            checkerB$constantRequired, checkerB$constantEstimate)

        adamFilled <- adam_filler(resB$B,
                                  resB$etsModel, resB$Etype, resB$Ttype, resB$Stype,
                                  resB$modelIsTrendy, resB$modelIsSeasonal,
                                  adamArchitect$componentsNumberETS,
                                  adamArchitect$componentsNumberETSNonSeasonal,
                                  adamArchitect$componentsNumberETSSeasonal,
                                  checkerB$componentsNumberARIMA,
                                  lags, adamArchitect$lagsModel, adamArchitect$lagsModelMax,
                                  adamCreated$matVt, adamCreated$matWt, adamCreated$matF, adamCreated$vecG,
                                  resB$persistenceEstimate, resB$persistenceLevelEstimate,
                                  resB$persistenceTrendEstimate, resB$persistenceSeasonalEstimate,
                                  resB$persistenceXregEstimate, resB$phiEstimate,
                                  checkerB$initialType, resB$initialEstimate,
                                  resB$initialLevelEstimate, resB$initialTrendEstimate,
                                  resB$initialSeasonalEstimate, resB$initialArimaEstimate,
                                  resB$initialXregEstimate,
                                  resB$arimaModel, resB$arEstimate, resB$maEstimate,
                                  resB$arOrders, resB$iOrders, resB$maOrders,
                                  resB$arRequired, resB$maRequired, checkerB$armaParameters,
                                  checkerB$nonZeroARI, checkerB$nonZeroMA, adamCreated$arimaPolynomials,
                                  checkerB$xregModel, checkerB$xregNumber,
                                  checkerB$xregParametersMissing, checkerB$xregParametersIncluded,
                                  checkerB$xregParametersEstimated, checkerB$xregParametersPersistence,
                                  checkerB$constantEstimate, adamArchitect$adamCpp,
                                  checkerB$constantRequired, checkerB$initialArimaNumber)

        prof <- adamFilled$matVt[, seq_len(adamArchitect$lagsModelMax), drop=FALSE]
        adamFitted <- adamArchitect$adamCpp$fit(
            adamFilled$matVt, adamFilled$matWt, adamFilled$matF, adamFilled$vecG,
            adamArchitect$indexLookupTable, prof,
            as.numeric(ot), as.numeric(ot),
            any(checkerB$initialType == c("complete","backcasting")),
            nIterations, refineHead, "i")

        yFitted <- adamFitted$fitted

        if(is.null(resB$logLikADAMValue)) {
            pFitted  <- omLinkFunction(as.numeric(yFitted), resB$Etype, "inverse-odds-ratio")
            ot_vec   <- as.numeric(oInSample)
            ll <- sum(ot_vec * log(pmax(pFitted, 1e-15)) +
                          (1 - ot_vec) * log(pmax(1 - pFitted, 1e-15)))
            resB$logLikADAMValue <- ll
            resB$CFValue <- -ll
        }

        if(hLocal > 0) {
            yForecast <- adamArchitect$adamCpp$forecast(
                tail(adamFilled$matWt, hLocal),
                adamFilled$matF,
                adamArchitect$indexLookupTable[,
                                               adamArchitect$lagsModelMax + obsInSample + seq_len(hLocal), drop=FALSE],
                adamFitted$profile, hLocal)$forecast
        }

        statesRaw <- adamFitted$states[,
                                       (adamArchitect$lagsModelMax+1):ncol(adamFitted$states), drop=FALSE]
        compNames <- rownames(adamCreated$matVt)
        if(!is.null(compNames)) rownames(statesRaw) <- compNames

        if(any(yClasses == "ts")) {
            yFitted <- ts(yFitted, start=yStart, frequency=yFrequency)
            errors  <- ts(as.numeric(oInSample) - yFitted, start=yStart, frequency=yFrequency)
            matVt   <- ts(t(statesRaw), start=yStart, frequency=yFrequency)
        } else {
            yFitted <- zoo(yFitted, order.by=yInSampleIndex)
            errors  <- zoo(as.numeric(oInSample) - yFitted, order.by=yInSampleIndex)
            matVt   <- zoo(t(statesRaw), order.by=yInSampleIndex)
        }

        if(hLocal > 0 && !is.null(yForecast)) {
            if(any(yClasses == "ts")) {
                yForecast <- ts(yForecast, start=yForecastStart, frequency=yFrequency)
            } else {
                yForecast <- zoo(yForecast, order.by=yForecastIndex)
            }
        } else {
            yForecast <- if(any(yClasses=="ts")) {
                ts(NA, start=yForecastStart, frequency=yFrequency)
            } else {
                zoo(NA, order.by=yForecastIndex[1])
            }
        }

        modelStr  <- paste0(resB$Etype, resB$Ttype,
                            "d"[resB$phiEstimate], resB$Stype)
        modelName <- adam_model_name(
            resB$etsModel, modelStr,
            checkerB$xregModel, resB$arimaModel,
            resB$arOrders, resB$iOrders, resB$maOrders, lags,
            regressorsB, checkerB$constantRequired, checkerB$constantName,
            "inverse-odds-ratio", adamArchitect$componentsNumberETSSeasonal,
            prefix = "o")

        vecGFinal <- adamFilled$vecG
        if(adamArchitect$componentsNumberETS > 0) {
            persistenceVec <- as.vector(vecGFinal)[seq_len(adamArchitect$componentsNumberETS)]
            names(persistenceVec) <- rownames(vecGFinal)[seq_len(adamArchitect$componentsNumberETS)]
        } else {
            persistenceVec <- numeric(0)
        }

        initialCollected <- adam_initial_collector(
            adamFitted$states[, seq_len(adamArchitect$lagsModelMax), drop=FALSE],
            resB$etsModel, resB$modelIsTrendy, resB$modelIsSeasonal,
            adamArchitect$lagsModel, adamArchitect$lagsModelMax,
            resB$initialLevelEstimate, resB$initialTrendEstimate,
            resB$initialSeasonalEstimate,
            adamArchitect$componentsNumberETSSeasonal,
            resB$arimaModel, resB$initialArimaEstimate,
            checkerB$initialArima, checkerB$initialArimaNumber,
            adamArchitect$componentsNumberETS, checkerB$componentsNumberARIMA,
            adamFilled$arimaPolynomials, resB$Etype,
            checkerB$xregModel, resB$initialXregEstimate, checkerB$xregNumber)

        if(resB$arimaModel && (resB$arRequired || resB$maRequired)) {
            armaParametersList <- vector("list", resB$arRequired + resB$maRequired)
            j <- 1L
            if(resB$arRequired && resB$arEstimate) {
                armaParametersList[[j]] <- resB$B[nchar(names(resB$B))>3 &
                                                      substr(names(resB$B),1,3)=="phi"]
                names(armaParametersList)[j] <- "ar"; j <- j + 1L
            } else if(resB$arRequired) {
                armaParametersList[[j]] <- checkerB$armaParameters[
                    substr(names(checkerB$armaParameters),1,3)=="phi"]
                names(armaParametersList)[j] <- "ar"; j <- j + 1L
            }
            if(resB$maRequired && resB$maEstimate) {
                armaParametersList[[j]] <- resB$B[substr(names(resB$B),1,5)=="theta"]
                names(armaParametersList)[j] <- "ma"
            } else if(resB$maRequired) {
                armaParametersList[[j]] <- checkerB$armaParameters[
                    substr(names(checkerB$armaParameters),1,5)=="theta"]
                names(armaParametersList)[j] <- "ma"
            }
        } else {
            armaParametersList <- NULL
        }

        parNum <- checkerB$parametersNumber
        parNum[1,1] <- resB$nParamEstimated
        parNum[1,5] <- sum(parNum[1,1:4])
        parNum[2,5] <- sum(parNum[2,1:4])

        subModel <- list(
            model       = modelName,
            timeElapsed = Sys.time() - startTime,
            data        = yInSample,
            fitted      = yFitted,
            residuals   = errors,
            forecast    = yForecast,
            states      = matVt,
            profile     = adamFitted$profile,
            profileInitial = NULL,
            persistence = persistenceVec,
            phi         = if(resB$phiEstimate) resB$B["phi"] else checkerB$phi,
            transition  = adamFilled$matF,
            measurement = adamFilled$matWt,
            initial     = initialCollected$initialValue,
            initialType = checkerB$initialType,
            initialEstimated = initialCollected$initialEstimated,
            orders      = list(ar=resB$arOrders, i=resB$iOrders, ma=resB$maOrders),
            arma        = armaParametersList,
            constant    = if(checkerB$constantRequired) {
                if(checkerB$constantEstimate) resB$B[checkerB$constantName] else checkerB$constantValue
            } else NULL,
            nParam      = parNum,
            occurrence  = "inverse-odds-ratio",
            formula     = checkerB$formula,
            regressors  = regressorsB,
            loss        = loss,
            lossValue   = resB$CFValue,
            lossFunction = NULL,
            logLik      = resB$logLikADAMValue,
            distribution = "plogis",
            scale       = NA,
            other       = NULL,
            B           = resB$B,
            lags        = lags,
            lagsAll     = adamArchitect$lagsModelAll,
            ets         = resB$etsModel,
            res         = resB,
            FI          = NULL,
            adamCpp     = adamArchitect$adamCpp,
            bounds      = bounds,
            call        = cl)

        if(holdout) {
            subModel$holdout  <- oHoldout
            subModel$accuracy <- measures(as.vector(oHoldout), yForecast,
                                          as.vector(oInSample))
        }

        class(subModel) <- c("om","adam","smooth")
        return(subModel)
    }

    #### Run estimation ####
    jointResult <- omgEstimator()

    # Build result structs for sub-models
    resA <- list(
        B                        = jointResult$B_A,
        nParamEstimated          = jointResult$nParamsA,
        logLikADAMValue          = NULL,
        CFValue                  = 0,
        etsModel                 = checkerA$etsModel,
        Etype                    = checkerA$Etype,
        Ttype                    = checkerA$Ttype,
        Stype                    = checkerA$Stype,
        modelIsTrendy            = checkerA$modelIsTrendy,
        modelIsSeasonal          = checkerA$modelIsSeasonal,
        arimaModel               = checkerA$arimaModel,
        arOrders                 = checkerA$arOrders,
        iOrders                  = checkerA$iOrders,
        maOrders                 = checkerA$maOrders,
        arRequired               = checkerA$arRequired,
        iRequired                = checkerA$iRequired,
        maRequired               = checkerA$maRequired,
        arEstimate               = checkerA$arEstimate,
        maEstimate               = checkerA$maEstimate,
        phiEstimate              = checkerA$phiEstimate,
        persistenceEstimate      = checkerA$persistenceEstimate,
        persistenceLevelEstimate = checkerA$persistenceLevelEstimate,
        persistenceTrendEstimate = checkerA$persistenceTrendEstimate,
        persistenceSeasonalEstimate = checkerA$persistenceSeasonalEstimate,
        persistenceXregEstimate  = checkerA$persistenceXregEstimate,
        initialEstimate          = checkerA$initialEstimate,
        initialLevelEstimate     = checkerA$initialLevelEstimate,
        initialTrendEstimate     = checkerA$initialTrendEstimate,
        initialSeasonalEstimate  = checkerA$initialSeasonalEstimate,
        initialArimaEstimate     = checkerA$initialArimaEstimate,
        initialXregEstimate      = checkerA$initialXregEstimate,
        FI                       = NULL)

    resB <- list(
        B                        = jointResult$B_B,
        nParamEstimated          = jointResult$nParamsB,
        logLikADAMValue          = NULL,
        CFValue                  = 0,
        etsModel                 = checkerB$etsModel,
        Etype                    = checkerB$Etype,
        Ttype                    = checkerB$Ttype,
        Stype                    = checkerB$Stype,
        modelIsTrendy            = checkerB$modelIsTrendy,
        modelIsSeasonal          = checkerB$modelIsSeasonal,
        arimaModel               = checkerB$arimaModel,
        arOrders                 = checkerB$arOrders,
        iOrders                  = checkerB$iOrders,
        maOrders                 = checkerB$maOrders,
        arRequired               = checkerB$arRequired,
        iRequired                = checkerB$iRequired,
        maRequired               = checkerB$maRequired,
        arEstimate               = checkerB$arEstimate,
        maEstimate               = checkerB$maEstimate,
        phiEstimate              = checkerB$phiEstimate,
        persistenceEstimate      = checkerB$persistenceEstimate,
        persistenceLevelEstimate = checkerB$persistenceLevelEstimate,
        persistenceTrendEstimate = checkerB$persistenceTrendEstimate,
        persistenceSeasonalEstimate = checkerB$persistenceSeasonalEstimate,
        persistenceXregEstimate  = checkerB$persistenceXregEstimate,
        initialEstimate          = checkerB$initialEstimate,
        initialLevelEstimate     = checkerB$initialLevelEstimate,
        initialTrendEstimate     = checkerB$initialTrendEstimate,
        initialSeasonalEstimate  = checkerB$initialSeasonalEstimate,
        initialArimaEstimate     = checkerB$initialArimaEstimate,
        initialXregEstimate      = checkerB$initialXregEstimate,
        FI                       = NULL)

    # Build sub-model objects using jointly-optimized parameters
    modelA <- omgFinalFitA(resA, hLocal=h)
    modelB <- omgFinalFitB(resB, hLocal=h)

    # Combined fitted and forecast using raw sub-model values via omgLinkFunction
    EtypeA <- errorType(modelA)
    EtypeB <- errorType(modelB)

    yFittedA  <- as.vector(modelA$fitted)
    yFittedB  <- as.vector(modelB$fitted)
    yFitted   <- modelA$fitted
    yFitted[] <- omgLinkFunction(yFittedA, yFittedB, EtypeA, EtypeB)

    yForecast <- NULL
    if(h > 0) {
        yForecastA  <- as.vector(modelA$forecast)
        yForecastB  <- as.vector(modelB$forecast)
        yForecast   <- modelA$forecast
        yForecast[] <- omgLinkFunction(yForecastA, yForecastB, EtypeA, EtypeB)
    }

    modelName <- paste0("oETS[G](", modelType(modelA), ")(", modelType(modelB), ")")

    result <- list(
        model       = modelName,
        modelA      = modelA,
        modelB      = modelB,
        fitted      = yFitted,
        forecast    = yForecast,
        occurrence  = "general",
        lags        = lags,
        lossValue   = jointResult$CFValue,
        logLik      = jointResult$logLikValue,
        nParam      = {
            nParamMat <- matrix(0, 2, 5,
                                dimnames=list(c("Estimated","Provided"),
                                              c("nParamInternal","nParamXreg","nParamOccurrence",
                                                "nParamScale","nParamAll")))
            nParamMat[1,1] <- jointResult$nParamsA + jointResult$nParamsB
            nParamMat[1,5] <- nParamMat[1,1]
            nParamMat[2,1:4] <- modelA$nParam[2,1:4] + modelB$nParam[2,1:4]
            nParamMat[2,5]   <- sum(nParamMat[2,1:4])
            nParamMat
        },
        distribution = "plogis",
        loss        = "likelihood",
        call        = cl,
        timeElapsed = Sys.time() - startTime)

    if(holdout) {
        result$holdout  <- oHoldout
        result$accuracy <- measures(as.vector(oHoldout), yForecast,
                                    as.vector(yFitted))
    }

    class(result) <- c("omg","om","smooth")
    return(result)
}

#' @export
forecast.omg <- function(object, h=NULL, ...) {
    if(is.null(h)) { h <- length(object$forecast) }
    fcA <- forecast.adam(object$modelA, h=h, interval="none",
                         level=0.95, side="both", cumulative=FALSE, ...)
    fcB <- forecast.adam(object$modelB, h=h, interval="none",
                         level=0.95, side="both", cumulative=FALSE, ...)
    EtypeA      <- errorType(object$modelA)
    EtypeB      <- errorType(object$modelB)
    yForecastA  <- as.vector(fcA$mean)
    yForecastB  <- as.vector(fcB$mean)
    yForecast   <- fcA$mean
    yForecast[] <- omgLinkFunction(yForecastA, yForecastB, EtypeA, EtypeB)
    return(structure(
        list(mean=yForecast, lower=fcA$lower, upper=fcA$upper,
             model=object, level=fcA$level, interval=fcA$interval,
             side=fcA$side, cumulative=fcA$cumulative, h=h,
             scenarios=fcA$scenarios),
        class=c("adam.forecast","smooth.forecast","forecast")))
}

#' @export
actuals.omg <- function(object, ...) { actuals.om(object$modelA, ...) }

#' @export
print.omg <- function(x, digits=4, ...) {
    cat("Time elapsed:", round(as.numeric(x$timeElapsed, units="secs"), 2), "seconds")
    cat(paste0("\nGeneral occurrence model: ", x$model))
    stripModel <- function(m) sub("^o", "", sub("\\[.*\\]$", "", m))
    cat(paste0("\nModel A: ", stripModel(x$modelA$model)))
    cat(paste0("\nModel B: ", stripModel(x$modelB$model)))

    distrib <- switch(x$distribution,
                      "dnorm"     = "Normal",
                      "dlaplace"  = "Laplace",
                      "ds"        = "S",
                      "dgnorm"    = paste0("Generalised Normal with shape=", round(x$other$shape, digits)),
                      "dlogis"    = "Logistic",
                      "plogis"    = "Cumulative Logistic",
                      "dt"        = paste0("Student t with df=", round(x$other$nu, digits)),
                      "dalaplace" = paste0("Asymmetric Laplace with alpha=", round(x$other$alpha, digits)),
                      "dlnorm"    = "Log-Normal",
                      "dllaplace" = "Log-Laplace",
                      "dls"       = "Log-S",
                      "dlgnorm"   = paste0("Log-Generalised Normal with shape=", round(x$other$shape, digits)),
                      "dinvgauss" = "Inverse Gaussian",
                      "dgamma"    = "Gamma")
    cat(paste0("\n\nDistribution assumed in the model: ", distrib))
    cat(paste0("\nLoss function type: ", x$loss))
    if(!is.null(x$lossValue)) {
        cat(paste0("; Loss function value: ", round(x$lossValue, digits)))
    }

    cat(paste0("\n\nSample size: ", nobs(x)))
    cat(paste0("\nNumber of estimated parameters: ", round(nparam(x), digits)))
    cat(paste0("\nNumber of degrees of freedom: ", nobs(x) - round(nparam(x), digits)))
    if(x$nParam[2,5] > 0) {
        cat(paste0("\nNumber of provided parameters: ", x$nParam[2,5]))
    }
    ICs <- c(AIC=AIC(x), AICc=AICc(x), BIC=BIC(x), BICc=BICc(x))
    cat("\nInformation criteria:\n")
    print(round(ICs, digits))
    return(invisible(x))
}

#' @export
summary.omg <- function(object, ...) { return(invisible(object)) }

#' @export
plot.omg <- function(x, which=c(1,2,4,6), ...) {
    plotObj        <- x$modelA
    plotObj$fitted <- x$fitted
    plot.adam(plotObj, which=which, ...)
}

#' @importFrom stats rstandard
#' @export
rstandard.omg <- function(model, ...){
    obs <- nobs(model);
    df  <- obs - nparam(model);
    p   <- as.numeric(model$fitted);
    e   <- as.numeric(actuals(model)) - p;
    return(e / sqrt(p * (1 - p)) * sqrt(obs / df));
}

#' @importFrom stats rstudent
#' @export
rstudent.omg <- function(model, ...){
    obs <- nobs(model);
    df  <- obs - nparam(model) - 1;
    p   <- as.numeric(model$fitted);
    e   <- as.numeric(actuals(model)) - p;
    return(e / sqrt(p * (1 - p)) * sqrt(obs / df));
}
