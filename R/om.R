#' Occurrence Model
#'
#' Function returns the occurrence part of the ADAM model with the specified
#' probability update and model types.
#'
#' The function estimates probability of demand occurrence, using the selected
#' ADAM state space model. It supports ETS, ARIMA and explanatory variables,
#' also allowing to have multiple frequencies and doing variables selection.
#' It is an ADAM analogue for the binary occurrence variable modelling.
#'
#' For the details about the model and its implementation, see the respective
#' vignette: \code{vignette("om","smooth")}
#'
#' @param data Numeric vector, time series, or data frame. Non-binary input is
#'   automatically binarised: any non-zero value becomes 1.
#' @param model Three-letter ETS specification such as \code{"MNN"} or
#'   \code{"AAN"}. Automatic selection with \code{"Z"} / \code{"X"} /
#'   \code{"Y"} wildcards is supported.
#' @param lags Vector of seasonal lags. Defaults to \code{frequency(data)}.
#' @param orders ARIMA orders list: \code{list(ar, i, ma, select)}.
#' @param constant Logical; whether to include a constant term.
#' @param formula Optional formula for external regressors.
#' @param regressors How to handle regressors: \code{"use"},
#'   \code{"select"}, or \code{"adapt"}.
#' @param occurrence Type of link function mapping state to probability:
#'   \code{"fixed"} (constant), \code{"odds-ratio"}, \code{"inverse-odds-ratio"},
#'   or \code{"direct"}.
#' @param loss Loss function: \code{"likelihood"} (Bernoulli) or \code{"MSE"}.
#' @param h Forecast horizon.
#' @param holdout If \code{TRUE}, a holdout sample of size \code{h} is withheld.
#' @param persistence Optional persistence (smoothing) parameter vector.
#' @param phi Optional damping parameter.
#' @param initial Initialisation method: \code{"backcasting"}, \code{"optimal"},
#'   \code{"two-stage"}, or \code{"complete"}.
#' @param arma Optional fixed ARMA parameters.
#' @param ic Information criterion for model selection.
#' @param bounds Parameter bounds type.
#' @param ets Type of ETS model: \code{"conventional"} or \code{"adam"}.
#' @param silent If \code{TRUE}, suppresses output and plot.
#' @param ... Additional arguments passed to the optimiser (\code{maxeval},
#'   \code{xtol_rel}, \code{algorithm}, \code{print_level}).
#'
#' @return An object of class \code{c("om","adam","smooth")}.
#'
#' @seealso \link{forecast.om}, \link{adam}
#'
#' @examples
#' set.seed(42)
#' y <- rbinom(120, 1, 0.6)
#' m <- om(y, model="MNN", occurrence="odds-ratio")
#' forecast(m, h=12)
#'
#' @rdname om
#' @export
om <- function(data,
               model = "ZXZ",
               lags  = c(frequency(data)),
               orders = list(ar=c(0), i=c(0), ma=c(0), select=FALSE),
               constant = FALSE,
               formula  = NULL,
               regressors = c("use","select","adapt"),
               occurrence = c("auto","fixed","odds-ratio","inverse-odds-ratio","direct","general"),
               loss = c("likelihood","MSE","MAE","HAM","LASSO","RIDGE"),
               h = 0, holdout = FALSE,
               persistence = NULL, phi = NULL,
               initial = c("backcasting","optimal","two-stage","complete"),
               arma = NULL,
               ic = c("AICc","AIC","BIC","BICc"),
               bounds = c("usual","admissible","none"),
               ets = c("conventional","adam"),
               silent = TRUE, ...){

    startTime <- Sys.time();
    cl <- match.call();

    occurrence <- match.arg(occurrence);
    if(occurrence == "auto") {
        result <- auto.om(data=data, model=model, lags=lags, orders=orders,
                          formula=formula, regressors=regressors,
                          h=h, holdout=holdout,
                          persistence=persistence, phi=phi,
                          initial=initial, arma=arma,
                          ic=ic, bounds=bounds, silent=silent, ets=ets,
                          occurrence=c("fixed","general","odds-ratio","inverse-odds-ratio","direct"),
                          constant=constant, loss=loss, ...)
        result$call <- match.call()
        return(result)
    }
    if(occurrence == "general") {
        result <- omg(data=data, modelA=model, modelB=model,
                      ordersA=orders, ordersB=orders,
                      constantA=constant, constantB=constant,
                      formulaA=formula, formulaB=formula,
                      regressorsA=regressors, regressorsB=regressors,
                      persistenceA=persistence, persistenceB=persistence,
                      phiA=phi, phiB=phi,
                      armaA=arma, armaB=arma,
                      etsA=ets, etsB=ets,
                      lags=lags, h=h, holdout=holdout,
                      initial=initial, loss=loss, ic=ic,
                      bounds=bounds, silent=silent, ...)
        result$call <- match.call()
        return(result)
    }
    occurrenceType <- occurrence;
    loss <- match.arg(loss);
    ic <- match.arg(ic);
    bounds <- match.arg(bounds);
    regressors <- match.arg(regressors);
    ets <- match.arg(ets);
    ellipsis <- list(...);

    occurrenceChar <- switch(occurrence,
                             "odds-ratio"          = "o",
                             "inverse-odds-ratio"  = "i",
                             "direct"              = "d",
                             "fixed"               = "f",
                             "n");

    #### Data preparation ####
    yName <- paste0(deparse(substitute(data)), collapse="");
    if(length(yName)==0 || is.null(yName)){
        yName <- "y";
    }
    modelDo <- "estimate";

    dataChecked <- adam_checkData(data, lags, h, holdout, yName, modelDo, formula);
    list2env(dataChecked, envir=environment());

    #### Force ETS(A,N,N) with persistence=0 for "fixed" occurrence ####
    if(occurrence == "fixed"){
        model <- "ANN";
        persistence <- 0;
        initial <- "optimal";
        modelDo <- "use";
    }

    #### Call parametersChecker ####
    checkerReturn <- parametersChecker(data=data, model=model, lags=lags,
                                       formulaToUse=formula, orders=orders,
                                       constant=constant, arma=arma,
                                       persistence=persistence, phi=phi,
                                       initial=initial,
                                       distribution="dnorm",
                                       loss=if(loss=="likelihood") "likelihood" else loss,
                                       h=h, holdout=holdout,
                                       occurrence=occurrence,
                                       ic=ic, bounds=bounds, regressors=regressors,
                                       yName=yName,
                                       silent=silent, modelDo=modelDo,
                                       ellipsis=ellipsis, fast=FALSE);
    list2env(checkerReturn, envir=environment());

    # ARIMA order selection is not supported in om()
    if(isTRUE(select)){
        stop("ARIMA order selection (orders$select=TRUE) is not supported in om(). ",
             "Specify fixed orders instead.", call.=FALSE);
    }

    occurrence <- occurrenceType;

    # For "fixed": set initial level analytically and disable estimation
    if(occurrenceType == "fixed"){
        if(initialLevelEstimate){
            initialLevel <- mean(ot);
        }
        initialType <- "provided";
        initialLevelEstimate <- FALSE;
        initialEstimate <- FALSE;
        persistenceEstimate <- FALSE;
        persistenceLevelEstimate <- FALSE;
        occurrenceChar <- "d";
        nParamEstimated <- 1;
        modelDo <- "use";
    }

    # Binary indicators (ot from checker is already binary when occurrence != "none")
    oInSample <- matrix(as.numeric(ot), ncol=1);
    if(holdout){
        oHoldout <- matrix(as.numeric(yHoldout != 0), ncol=1);
    }

    # Override occurrence-related flags set by checker
    occurrenceModel <- FALSE;
    oesModel <- NULL;
    yFitted <- matrix(rep(mean(oInSample), obsInSample), ncol=1);
    refineHead <- TRUE;
    adamETS <- (ets == "adam");

    #### Optimiser settings ####
    optimSettings <- adam_checkOptimizer(ellipsis=ellipsis, loss=loss, distribution="dnorm",
                                         initialType=initialType, lags=lags,
                                         arimaModel=arimaModel);
    list2env(optimSettings, envir=environment());

    # This is the internal variable, which should be equal to 1 everywhere
    # This is to create all necessary objects correctly.
    otLogicalInternal <- otLogical;
    otLogicalInternal[] <- TRUE;

    #### Inner estimator for a single ETS/ARIMA model ####
    omEstimator <- function(etsModel, Etype, Ttype, Stype, lags,
                            lagsModelSeasonal, lagsModelARIMA,
                            obsStates, obsInSample,
                            yInSample, persistence, persistenceEstimate,
                            persistenceLevel, persistenceLevelEstimate,
                            persistenceTrend, persistenceTrendEstimate,
                            persistenceSeasonal, persistenceSeasonalEstimate,
                            persistenceXreg, persistenceXregEstimate,
                            persistenceXregProvided,
                            phi, phiEstimate,
                            initialType, initialLevel, initialTrend,
                            initialSeasonal, initialArima, initialEstimate,
                            initialLevelEstimate, initialTrendEstimate,
                            initialSeasonalEstimate, initialArimaEstimate,
                            initialXregEstimate, initialXregProvided,
                            arimaModel, arRequired, iRequired, maRequired,
                            armaParameters,
                            componentsNumberARIMA, componentsNamesARIMA,
                            formula, xregModel, xregModelInitials, xregData,
                            xregNumber, xregNames, regressors,
                            xregParametersMissing, xregParametersIncluded,
                            xregParametersEstimated, xregParametersPersistence,
                            constantRequired, constantEstimate, constantValue,
                            constantName,
                            ot, otLogical, occurrenceModel, yFitted,
                            bounds, loss, lossFunction, distribution,
                            horizon, multisteps, other, otherParameterEstimate,
                            lambda, B){

        omCF_local <- function(B,
                               etsModel, Etype, Ttype, Stype,
                               modelIsTrendy, modelIsSeasonal,
                               componentsNumberETS, componentsNumberETSNonSeasonal,
                               componentsNumberETSSeasonal, componentsNumberARIMA,
                               lags, lagsModel, lagsModelMax, lagsModelAll,
                               indexLookupTable, profilesRecentTable,
                               matVt, matWt, matF, vecG,
                               persistenceEstimate, persistenceLevelEstimate,
                               persistenceTrendEstimate, persistenceSeasonalEstimate,
                               persistenceXregEstimate, phiEstimate,
                               initialType, initialEstimate,
                               initialLevelEstimate, initialTrendEstimate,
                               initialSeasonalEstimate, initialArimaEstimate,
                               initialXregEstimate, initialArimaNumber,
                               arimaModel, arEstimate, maEstimate,
                               arOrders, iOrders, maOrders,
                               arRequired, maRequired, armaParameters,
                               nonZeroARI, nonZeroMA, arimaPolynomials,
                               arPolynomialMatrix, maPolynomialMatrix,
                               xregModel, xregNumber,
                               xregParametersMissing, xregParametersIncluded,
                               xregParametersEstimated, xregParametersPersistence,
                               constantRequired, constantEstimate,
                               bounds, regressors, loss,
                               ot, otLogical, obsInSample,
                               nIterations, refineHead,
                               occurrence, occurrenceChar,
                               adamCpp){
            adamElements <- adam_filler(B,
                                        etsModel, Etype, Ttype, Stype,
                                        modelIsTrendy, modelIsSeasonal,
                                        componentsNumberETS, componentsNumberETSNonSeasonal,
                                        componentsNumberETSSeasonal, componentsNumberARIMA,
                                        lags, lagsModel, lagsModelMax,
                                        matVt, matWt, matF, vecG,
                                        persistenceEstimate, persistenceLevelEstimate,
                                        persistenceTrendEstimate, persistenceSeasonalEstimate,
                                        persistenceXregEstimate, phiEstimate,
                                        initialType, initialEstimate,
                                        initialLevelEstimate, initialTrendEstimate,
                                        initialSeasonalEstimate, initialArimaEstimate,
                                        initialXregEstimate,
                                        arimaModel, arEstimate, maEstimate,
                                        arOrders, iOrders, maOrders,
                                        arRequired, maRequired, armaParameters,
                                        nonZeroARI, nonZeroMA, arimaPolynomials,
                                        xregModel, xregNumber,
                                        xregParametersMissing, xregParametersIncluded,
                                        xregParametersEstimated, xregParametersPersistence,
                                        constantEstimate, adamCpp,
                                        constantRequired, initialArimaNumber);
            penalty <- adam_bounds_checker(adamElements, adamElements$arimaPolynomials,
                                           bounds,
                                           etsModel, modelIsTrendy, modelIsSeasonal,
                                           componentsNumberETS, componentsNumberETSNonSeasonal,
                                           componentsNumberETSSeasonal,
                                           arimaModel, arEstimate, maEstimate,
                                           xregModel, regressors, xregNumber,
                                           componentsNumberARIMA,
                                           lagsModelAll, obsInSample,
                                           arPolynomialMatrix, maPolynomialMatrix,
                                           phiEstimate);
            if(penalty != 0){
                return(penalty);
            }
            profilesRecentTable[] <- adamElements$matVt[, 1:lagsModelMax];
            adamFitted <- adamCpp$fit(adamElements$matVt, adamElements$matWt,
                                      adamElements$matF, adamElements$vecG,
                                      indexLookupTable, profilesRecentTable,
                                      as.numeric(ot), as.numeric(ot),
                                      any(initialType == c("complete","backcasting")),
                                      nIterations, refineHead, occurrenceChar);
            yFitted <- omLinkFunction(adamFitted$fitted, Etype, occurrence);
            if(any(is.nan(yFitted)) || any(yFitted<0) || any(yFitted>1)){
                return(1e+300);
            }
            if(loss == "likelihood"){
                CFValue <- -(sum(log(yFitted[otLogical])) + sum(log(1 - yFitted[!otLogical])));
            }
            else{
                CFValue <- mean((as.numeric(ot) - yFitted)^2);
            }
            return(CFValue)
        }

        adamArchitect <- adam_architector(etsModel, Etype, Ttype, Stype, lags,
                                          lagsModelSeasonal,
                                          xregNumber, obsInSample, initialType,
                                          arimaModel, lagsModelARIMA, xregModel,
                                          constantRequired,
                                          componentsNumberARIMA,
                                          obsAll, yIndexAll, yClasses, adamETS);
        list2env(adamArchitect, environment());

        # Etype="A" is needed for the decomposition to work in case of 0/1 data
        adamCreated <- adam_creator(etsModel, Etype="A", Ttype=switch(Ttype, "N"="N", "A"), Stype="A",
                                    modelIsTrendy, modelIsSeasonal,
                                    lags, lagsModel, lagsModelARIMA, lagsModelAll,
                                    lagsModelMax,
                                    profilesRecentTable, FALSE,
                                    obsStates, obsInSample,
                                    obsAll,
                                    componentsNumberETS, componentsNumberETSSeasonal,
                                    componentsNamesETS, otLogicalInternal, ot,
                                    persistence, persistenceEstimate,
                                    persistenceLevel, persistenceLevelEstimate,
                                    persistenceTrend, persistenceTrendEstimate,
                                    persistenceSeasonal, persistenceSeasonalEstimate,
                                    persistenceXreg, persistenceXregEstimate,
                                    persistenceXregProvided,
                                    phi,
                                    initialType, initialEstimate,
                                    initialLevel, initialLevelEstimate,
                                    initialTrend, initialTrendEstimate,
                                    initialSeasonal, initialSeasonalEstimate,
                                    initialArima, initialArimaEstimate,
                                    initialArimaNumber,
                                    initialXregEstimate, initialXregProvided,
                                    arimaModel, arRequired, iRequired, maRequired,
                                    armaParameters,
                                    arOrders, iOrders, maOrders,
                                    componentsNumberARIMA, componentsNamesARIMA,
                                    xregModel, xregModelInitials, xregData,
                                    xregNumber, xregNames,
                                    xregParametersPersistence,
                                    constantRequired, constantEstimate,
                                    constantValue, constantName,
                                    adamCpp,
                                    arEstimate, maEstimate, smoother,
                                    nonZeroARI, nonZeroMA);

        adamCreated$matVt <- om_initial_transform(
            adamCreated$matVt, occurrence, Etype, Ttype, Stype,
            etsModel,
            modelIsTrendy, modelIsSeasonal,
            initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
            componentsNumberETS,
            componentsNumberETSNonSeasonal, componentsNumberETSSeasonal,
            lagsModel, lagsModelMax, lagsModelSeasonal,
            obsInSample, ot,
            arimaModel, componentsNumberARIMA,
            initialArimaEstimate, initialArimaNumber,
            xregModel, xregNumber, initialXregEstimate,
            constantRequired, constantEstimate);

        BValues <- adam_initialiser(etsModel, Etype, Ttype, Stype,
                                    modelIsTrendy, modelIsSeasonal,
                                    componentsNumberETSNonSeasonal,
                                    componentsNumberETSSeasonal,
                                    componentsNumberETS,
                                    lags, lagsModel, lagsModelSeasonal,
                                    lagsModelARIMA, lagsModelMax,
                                    adamCreated$matVt,
                                    persistenceEstimate, persistenceLevelEstimate,
                                    persistenceTrendEstimate,
                                    persistenceSeasonalEstimate,
                                    persistenceXregEstimate,
                                    phiEstimate, initialType, initialEstimate,
                                    initialLevelEstimate, initialTrendEstimate,
                                    initialSeasonalEstimate,
                                    initialArimaEstimate, initialXregEstimate,
                                    arimaModel, arRequired, maRequired,
                                    arEstimate, maEstimate,
                                    arOrders, maOrders,
                                    componentsNumberARIMA, componentsNamesARIMA,
                                    initialArimaNumber,
                                    xregModel, xregNumber,
                                    xregParametersEstimated, xregParametersPersistence,
                                    constantEstimate, constantName,
                                    otherParameterEstimate,
                                    adamCpp,
                                    ets, bounds, ot, otLogicalInternal,
                                    iOrders, armaParameters, other);

        B_used <- BValues$B;

        lb <- BValues$Bl;
        ub <- BValues$Bu;

        # Treat the dangerous mixed models
        if((Etype=="A" && Ttype=="A" && Stype=="M") ||
           (Etype=="A" && Ttype=="M" && Stype=="A") ||
           (Etype=="M" && Ttype=="A" && Stype=="A") ||
           (Etype=="A" && Ttype=="M" && Stype=="N") ||
           (Etype=="M" && Ttype=="M" && Stype=="A") ||
           (Etype=="M" && Ttype=="N" && Stype=="A") ||
           (Etype=="A" && Ttype=="N" && Stype=="M")){
            B_used[] <- 0;
        }

        # ARIMA companion matrices for bounds checking
        if(arimaModel){
            arPolynomialMatrix <- matrix(0, arOrders %*% lags, arOrders %*% lags);
            if(nrow(arPolynomialMatrix) > 1){
                arPolynomialMatrix[2:nrow(arPolynomialMatrix)-1, 2:nrow(arPolynomialMatrix)] <-
                    diag(nrow(arPolynomialMatrix) - 1);
            }
            maPolynomialMatrix <- matrix(0, maOrders %*% lags, maOrders %*% lags);
            if(nrow(maPolynomialMatrix) > 1){
                maPolynomialMatrix[2:nrow(maPolynomialMatrix)-1, 2:nrow(maPolynomialMatrix)] <-
                    diag(nrow(maPolynomialMatrix) - 1);
            }
        } else {
            arPolynomialMatrix <- maPolynomialMatrix <- NULL;
        }

        # All arguments needed by omCF_local are passed explicitly to nloptr
        # below, so the cost function never reads them from the surrounding
        # closure (which is shared across the model-pool loop and could leak
        # the original model's flags into a different submodel's evaluation).
        nloptrArgs <- list(
            etsModel=etsModel, Etype=Etype, Ttype=Ttype, Stype=Stype,
            modelIsTrendy=modelIsTrendy, modelIsSeasonal=modelIsSeasonal,
            componentsNumberETS=componentsNumberETS,
            componentsNumberETSNonSeasonal=componentsNumberETSNonSeasonal,
            componentsNumberETSSeasonal=componentsNumberETSSeasonal,
            componentsNumberARIMA=componentsNumberARIMA,
            lags=lags, lagsModel=lagsModel, lagsModelMax=lagsModelMax,
            lagsModelAll=lagsModelAll,
            indexLookupTable=indexLookupTable,
            profilesRecentTable=profilesRecentTable,
            matVt=adamCreated$matVt, matWt=adamCreated$matWt,
            matF=adamCreated$matF, vecG=adamCreated$vecG,
            persistenceEstimate=persistenceEstimate,
            persistenceLevelEstimate=persistenceLevelEstimate,
            persistenceTrendEstimate=persistenceTrendEstimate,
            persistenceSeasonalEstimate=persistenceSeasonalEstimate,
            persistenceXregEstimate=persistenceXregEstimate,
            phiEstimate=phiEstimate,
            initialType=initialType, initialEstimate=initialEstimate,
            initialLevelEstimate=initialLevelEstimate,
            initialTrendEstimate=initialTrendEstimate,
            initialSeasonalEstimate=initialSeasonalEstimate,
            initialArimaEstimate=initialArimaEstimate,
            initialXregEstimate=initialXregEstimate,
            initialArimaNumber=initialArimaNumber,
            arimaModel=arimaModel, arEstimate=arEstimate, maEstimate=maEstimate,
            arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
            arRequired=arRequired, maRequired=maRequired,
            armaParameters=armaParameters,
            nonZeroARI=nonZeroARI, nonZeroMA=nonZeroMA,
            arimaPolynomials=adamCreated$arimaPolynomials,
            arPolynomialMatrix=arPolynomialMatrix,
            maPolynomialMatrix=maPolynomialMatrix,
            xregModel=xregModel, xregNumber=xregNumber,
            xregParametersMissing=xregParametersMissing,
            xregParametersIncluded=xregParametersIncluded,
            xregParametersEstimated=xregParametersEstimated,
            xregParametersPersistence=xregParametersPersistence,
            constantRequired=constantRequired,
            constantEstimate=constantEstimate,
            bounds=bounds, regressors=regressors, loss=loss,
            ot=ot, otLogical=otLogical, obsInSample=obsInSample,
            nIterations=nIterations, refineHead=refineHead,
            occurrence=occurrence, occurrenceChar=occurrenceChar,
            adamCpp=adamCpp);

        res <- suppressWarnings(do.call(nloptr,
                                        c(list(x0=B_used, eval_f=omCF_local,
                                               lb=lb, ub=ub,
                                               opts=list(algorithm=algorithm,
                                                         xtol_rel=xtol_rel,
                                                         maxeval=maxeval,
                                                         print_level=print_level)),
                                          nloptrArgs)));

        if(is.infinite(res$objective) || res$objective == 1e+300){
            B_used[] <- BValues$B;
            res <- suppressWarnings(do.call(nloptr,
                                            c(list(x0=B_used, eval_f=omCF_local,
                                                   lb=lb, ub=ub,
                                                   opts=list(algorithm=algorithm,
                                                             xtol_rel=xtol_rel,
                                                             maxeval=maxeval,
                                                             print_level=print_level)),
                                              nloptrArgs)));
        }

        B_used <- res$solution;
        names(B_used) <- names(BValues$B);
        CFValue <- res$objective;
        nParamEstimated <- length(B_used);
        logLikValue <- -CFValue;

        #### Fisher Information ####
        # Negative log-likelihood Hessian at the optimum gives FI directly
        # because omCF_local already returns -logLik. Wrap omCF_local in a
        # try() so numerical perturbations that violate bounds get a finite
        # large penalty instead of a hard error.
        if(isTRUE(FI)){
            CFAtOptimum <- omCF_local(B_used);
            omCF_FI <- function(B){
                names(B) <- names(B_used);
                val <- tryCatch(suppressWarnings(omCF_local(B)),
                                error = function(e) CFAtOptimum + 1e6);
                if(!is.finite(val)){
                    val <- CFAtOptimum + 1e6;
                }
                return(val);
            }
            FIMatrix <- try(suppressWarnings(pracma::hessian(omCF_FI, B_used, h=stepSize)),
                            silent=TRUE);
            if(inherits(FIMatrix, "try-error") || any(!is.finite(FIMatrix))){
                FIMatrix <- NULL;
            } else {
                colnames(FIMatrix) <- names(B_used);
                rownames(FIMatrix) <- names(B_used);
            }
        } else {
            FIMatrix <- NULL;
        }

        return(list(B=B_used, CFValue=CFValue, nParamEstimated=nParamEstimated,
                    logLikADAMValue=logLikValue,
                    xregModel=xregModel, xregData=xregData, xregNumber=xregNumber,
                    xregNames=xregNames, xregModelInitials=xregModelInitials,
                    formula=formula,
                    initialXregEstimate=initialXregEstimate,
                    persistenceXregEstimate=persistenceXregEstimate,
                    xregParametersMissing=xregParametersMissing,
                    xregParametersIncluded=xregParametersIncluded,
                    xregParametersEstimated=xregParametersEstimated,
                    xregParametersPersistence=xregParametersPersistence,
                    arimaPolynomials=adamCreated$arimaPolynomials,
                    res=res, FI=FIMatrix, adamCpp=adamCpp,
                    etsModel=etsModel, Etype=Etype, Ttype=Ttype, Stype=Stype,
                    arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
                    modelIsTrendy=modelIsTrendy, modelIsSeasonal=modelIsSeasonal,
                    # Per-submodel "Estimate" flags so omFinalFit() never
                    # falls back to outer om()'s lexical scope (which
                    # carries the originally-requested model's flags, not
                    # the variant currently being assembled).
                    phiEstimate=phiEstimate,
                    persistenceEstimate=persistenceEstimate,
                    persistenceLevelEstimate=persistenceLevelEstimate,
                    persistenceTrendEstimate=persistenceTrendEstimate,
                    persistenceSeasonalEstimate=persistenceSeasonalEstimate,
                    initialEstimate=initialEstimate,
                    initialLevelEstimate=initialLevelEstimate,
                    initialTrendEstimate=initialTrendEstimate,
                    initialSeasonalEstimate=initialSeasonalEstimate,
                    initialArimaEstimate=initialArimaEstimate,
                    arimaModel=arimaModel,
                    arRequired=arRequired, iRequired=iRequired, maRequired=maRequired,
                    arEstimate=arEstimate, maEstimate=maEstimate,
                    adamArchitect=adamArchitect, adamCreated=adamCreated));
    }

    #### IC function with shared environment for nParam ####
    .icEnv <- new.env(parent=emptyenv());
    .icEnv$nP <- 0L;

    icFunction <- function(ll){
        nP <- .icEnv$nP;
        llObj <- structure(ll, nobs=obsInSample, df=nP, class="logLik");
        return(switch(ic,
                      "AIC"  = AIC(llObj),
                      "AICc" = AICc(llObj),
                      "BIC"  = BIC(llObj),
                      "BICc" = BICc(llObj)));
    };
    icFunctionWrap <- function(ll, nP, obsIS=obsInSample){
        .icEnv$nP <- nP;
        return(icFunction(ll));
    };

    #### Helper: build a full om object from an estimator result ####
    omFinalFit <- function(res, hLocal=0, fullObject=FALSE,
                           yFittedOverride=NULL, yForecastOverride=NULL){
        otLogicalInternal <- otLogical;
        otLogicalInternal[] <- TRUE;

        if(!is.null(res$adamArchitect)){
            # Reuse objects built before nloptr — avoids any mismatch between the
            # matrices the optimiser saw and those rebuilt here from scratch.
            adamArchitect <- res$adamArchitect;
            adamCreated   <- res$adamCreated;
        } else {
            adamArchitect <- adam_architector(res$etsModel, res$Etype, res$Ttype, res$Stype,
                                              lags, lagsModelSeasonal,
                                              xregNumber, obsInSample, initialType,
                                              res$arimaModel, lagsModelARIMA,
                                              xregModel, constantRequired,
                                              componentsNumberARIMA,
                                              obsAll, yIndexAll, yClasses, adamETS);
            adamCreated <- adam_creator(res$etsModel, res$Etype, res$Ttype, res$Stype,
                                        res$modelIsTrendy, res$modelIsSeasonal,
                                        lags, adamArchitect$lagsModel, lagsModelARIMA,
                                        adamArchitect$lagsModelAll, adamArchitect$lagsModelMax,
                                        adamArchitect$profilesRecentTable, FALSE,
                                        adamArchitect$obsStates, obsInSample,
                                        obsAll,
                                        adamArchitect$componentsNumberETS,
                                        adamArchitect$componentsNumberETSSeasonal,
                                        adamArchitect$componentsNamesETS, otLogicalInternal, ot,
                                        persistence, res$persistenceEstimate,
                                        persistenceLevel, res$persistenceLevelEstimate,
                                        persistenceTrend, res$persistenceTrendEstimate,
                                        persistenceSeasonal, res$persistenceSeasonalEstimate,
                                        persistenceXreg, res$persistenceXregEstimate,
                                        persistenceXregProvided,
                                        phi,
                                        initialType, res$initialEstimate,
                                        initialLevel, res$initialLevelEstimate,
                                        initialTrend, res$initialTrendEstimate,
                                        initialSeasonal, res$initialSeasonalEstimate,
                                        initialArima, res$initialArimaEstimate,
                                        initialArimaNumber,
                                        res$initialXregEstimate, initialXregProvided,
                                        res$arimaModel, res$arRequired, res$iRequired, res$maRequired,
                                        armaParameters,
                                        res$arOrders, res$iOrders, res$maOrders,
                                        componentsNumberARIMA, componentsNamesARIMA,
                                        xregModel, xregModelInitials, xregData,
                                        xregNumber, xregNames,
                                        xregParametersPersistence,
                                        constantRequired, constantEstimate,
                                        constantValue, constantName,
                                        res$adamCpp,
                                        res$arEstimate, res$maEstimate, smoother,
                                        nonZeroARI, nonZeroMA);
            adamCreated$matVt <- om_initial_transform(
                adamCreated$matVt, occurrence, res$Etype, res$Ttype, res$Stype,
                res$etsModel,
                res$modelIsTrendy, res$modelIsSeasonal,
                res$initialLevelEstimate, res$initialTrendEstimate, res$initialSeasonalEstimate,
                adamArchitect$componentsNumberETS,
                adamArchitect$componentsNumberETSNonSeasonal,
                adamArchitect$componentsNumberETSSeasonal,
                adamArchitect$lagsModel, adamArchitect$lagsModelMax, lagsModelSeasonal,
                obsInSample, ot,
                res$arimaModel, componentsNumberARIMA,
                res$initialArimaEstimate, initialArimaNumber,
                xregModel, xregNumber, res$initialXregEstimate,
                constantRequired, constantEstimate);
        }

        adamFilled <- adam_filler(res$B,
                                  res$etsModel, res$Etype, res$Ttype, res$Stype,
                                  res$modelIsTrendy, res$modelIsSeasonal,
                                  adamArchitect$componentsNumberETS, adamArchitect$componentsNumberETSNonSeasonal,
                                  adamArchitect$componentsNumberETSSeasonal, componentsNumberARIMA,
                                  lags, adamArchitect$lagsModel, adamArchitect$lagsModelMax,
                                  adamCreated$matVt, adamCreated$matWt, adamCreated$matF, adamCreated$vecG,
                                  res$persistenceEstimate, res$persistenceLevelEstimate,
                                  res$persistenceTrendEstimate, res$persistenceSeasonalEstimate,
                                  res$persistenceXregEstimate, res$phiEstimate,
                                  initialType, res$initialEstimate,
                                  res$initialLevelEstimate, res$initialTrendEstimate,
                                  res$initialSeasonalEstimate, res$initialArimaEstimate,
                                  res$initialXregEstimate,
                                  res$arimaModel, res$arEstimate, res$maEstimate,
                                  res$arOrders, res$iOrders, res$maOrders,
                                  res$arRequired, res$maRequired, armaParameters,
                                  nonZeroARI, nonZeroMA, adamCreated$arimaPolynomials,
                                  xregModel, xregNumber,
                                  xregParametersMissing, xregParametersIncluded,
                                  xregParametersEstimated, xregParametersPersistence,
                                  constantEstimate, adamArchitect$adamCpp,
                                  constantRequired, initialArimaNumber);
        prof <- adamFilled$matVt[, 1:adamArchitect$lagsModelMax, drop=FALSE];
        adamFitted <- adamArchitect$adamCpp$fit(adamFilled$matVt, adamFilled$matWt, adamFilled$matF, adamFilled$vecG,
                                                adamArchitect$indexLookupTable, prof,
                                                as.numeric(ot), as.numeric(ot),
                                                any(initialType == c("complete","backcasting")),
                                                nIterations, refineHead, occurrenceChar);
        yFitted <- omLinkFunction(adamFitted$fitted, res$Etype, occurrence);

        # For "fixed" occurrence the optimizer never ran, so logLikADAMValue is absent.
        # Compute the Bernoulli log-likelihood from the constant fitted probability.
        if(is.null(res$logLikADAMValue)){
            ot_vec   <- as.numeric(oInSample);
            yfit_vec <- as.numeric(yFitted);
            ll <- sum(ot_vec   * log(pmax(yfit_vec,     1e-15)) +
                      (1 - ot_vec) * log(pmax(1 - yfit_vec, 1e-15)));
            res$logLikADAMValue <- ll;
            res$CFValue <- -ll;
        }

        # Forecast
        if(hLocal > 0){
            yForecast <- adamArchitect$adamCpp$forecast(tail(adamFilled$matWt, hLocal),
                                                        adamFilled$matF,
                                                        adamArchitect$indexLookupTable[, adamArchitect$lagsModelMax +
                                                                                           obsInSample + 1:hLocal,
                                                                                       drop=FALSE],
                                                        adamFitted$profile, hLocal)$forecast;
            yForecast <- omLinkFunction(yForecast, res$Etype, occurrence);
            yForecast[is.nan(yForecast)] <- 0;
        }

        if(!fullObject){
            if(hLocal == 0){
                return(yFitted);
            }
            return(list(fitted=yFitted, forecast=as.vector(yForecast)));
        }

        # States
        statesRaw <- adamFitted$states[, (adamArchitect$lagsModelMax+1):ncol(adamFitted$states), drop=FALSE];
        compNames <- rownames(adamCreated$matVt);
        if(!is.null(compNames)){
            rownames(statesRaw) <- compNames;
        }

        # Wrap as ts/zoo
        if(any(yClasses == "ts")){
            yFitted    <- ts(yFitted, start=yStart, frequency=yFrequency);
            errors <- ts(as.numeric(oInSample) - yFitted, start=yStart, frequency=yFrequency);
            matVt    <- ts(t(statesRaw), start=yStart, frequency=yFrequency);
        } else {
            yFitted    <- zoo(yFitted, order.by=yInSampleIndex);
            errors <- zoo(as.numeric(oInSample) - yFitted, order.by=yInSampleIndex);
            matVt    <- zoo(t(statesRaw), order.by=yInSampleIndex);
        }

        # Forecast ts
        if(hLocal > 0 && !is.null(yForecast)){
            if(any(yClasses == "ts")){
                yForecast <- ts(yForecast, start=yForecastStart, frequency=yFrequency);
            } else {
                yForecast <- zoo(yForecast, order.by=yForecastIndex);
            }
        } else {
            yForecast <- if(any(yClasses=="ts")){
                ts(NA, start=yForecastStart, frequency=yFrequency);
            } else {
                zoo(NA, order.by=yForecastIndex[1]);
            }
        }

        # Use overrides for combination, otherwise leave model-fitted values in place
        if(!is.null(yFittedOverride)){
            yFitted[] <- yFittedOverride;
        }
        if(!is.null(yForecastOverride)){
            yForecast[] <- yForecastOverride;
        }

        # Model name
        modelStr <- paste0(res$Etype, res$Ttype, "d"[res$phiEstimate], res$Stype);
        modelName <- adam_model_name(res$etsModel, modelStr, xregModel, res$arimaModel,
                                     res$arOrders, res$iOrders, res$maOrders, lags,
                                     regressors, constantRequired, constantName,
                                     occurrenceType, adamArchitect$componentsNumberETSSeasonal,
                                     prefix = "o");

        # Persistence vector
        vecGFinal <- adamFilled$vecG;
        if(adamArchitect$componentsNumberETS > 0){
            persistenceVec <- as.vector(vecGFinal)[1:adamArchitect$componentsNumberETS];
            names(persistenceVec) <- rownames(vecGFinal)[1:adamArchitect$componentsNumberETS];
        } else {
            persistenceVec <- numeric(0);
        }

        # Initial values
        initialCollected <- adam_initial_collector(
            adamFitted$states[, 1:adamArchitect$lagsModelMax, drop=FALSE],
            res$etsModel, res$modelIsTrendy, res$modelIsSeasonal,
            adamArchitect$lagsModel, adamArchitect$lagsModelMax,
            res$initialLevelEstimate, res$initialTrendEstimate, res$initialSeasonalEstimate,
            adamArchitect$componentsNumberETSSeasonal,
            res$arimaModel, res$initialArimaEstimate, initialArima, initialArimaNumber,
            adamArchitect$componentsNumberETS, componentsNumberARIMA,
            adamFilled$arimaPolynomials, res$Etype,
            xregModel, res$initialXregEstimate, xregNumber);

        # ARMA parameters
        if(res$arimaModel && (res$arRequired || res$maRequired)){
            armaParametersList <- vector("list", res$arRequired + res$maRequired);
            j <- 1L;
            if(res$arRequired && res$arEstimate){
                armaParametersList[[j]] <- res$B[nchar(names(res$B))>3 &
                                                     substr(names(res$B),1,3)=="phi"];
                names(armaParametersList)[j] <- "ar";
                j <- j + 1L;
            } else if(res$arRequired){
                armaParametersList[[j]] <- armaParameters[substr(names(armaParameters),1,3)=="phi"];
                names(armaParametersList)[j] <- "ar";
                j <- j + 1L;
            }
            if(res$maRequired && res$maEstimate){
                armaParametersList[[j]] <- res$B[substr(names(res$B),1,5)=="theta"];
                names(armaParametersList)[j] <- "ma";
            } else if(res$maRequired){
                armaParametersList[[j]] <- armaParameters[substr(names(armaParameters),1,5)=="theta"];
                names(armaParametersList)[j] <- "ma";
            }
        } else {
            armaParametersList <- NULL;
        }

        # Parameter counts
        parNum <- parametersNumber;
        parNum[1,1] <- res$nParamEstimated;
        parNum[1,5] <- sum(parNum[1,1:4]);
        parNum[2,5] <- sum(parNum[2,1:4]);

        subModel <- list(
            model = modelName,
            timeElapsed = Sys.time() - startTime,
            data = yInSample,
            fitted = yFitted,
            residuals = errors,
            forecast = yForecast,
            states = matVt,
            profile = adamFitted$profile,
            profileInitial = if(exists("profilesRecentInitial", inherits=FALSE)) {
                profilesRecentInitial
            } else NULL,
            persistence = persistenceVec,
            phi = if(res$phiEstimate) res$B["phi"] else phi,
            transition = adamFilled$matF,
            measurement = adamFilled$matWt,
            initial = initialCollected$initialValue,
            initialType = initialType,
            initialEstimated = initialCollected$initialEstimated,
            orders = list(ar=res$arOrders, i=res$iOrders, ma=res$maOrders),
            arma = armaParametersList,
            constant = if(constantRequired) {
                if(constantEstimate) res$B[constantName] else constantValue
            } else NULL,
            nParam = parNum,
            occurrence = occurrenceType,
            formula = formula,
            regressors = regressors,
            loss = loss,
            lossValue = res$CFValue,
            lossFunction = lossFunction,
            logLik = res$logLikADAMValue,
            distribution = "plogis",
            scale = NA,
            other = if(exists("otherReturned", inherits=FALSE)) otherReturned else NULL,
            B = res$B,
            lags = lags,
            lagsAll = adamArchitect$lagsModelAll,
            ets = res$etsModel,
            res = res$res,
            FI = res$FI,
            adamCpp = adamArchitect$adamCpp,
            bounds = bounds,
            call = cl
        );

        if(holdout){
            subModel$holdout <- oHoldout;
            subModel$accuracy <- measures(as.vector(oHoldout), yForecast,
                                          as.vector(oInSample));
        }

        class(subModel) <- c("om","adam","smooth");
        return(subModel);
    };

    #### Model selection or combination ####
    if(modelDo %in% c("select","combine")){
        omEstimatorWrapper <- function(etsModel, Etype, Ttype, Stype, lags,
                                       lagsModelSeasonal, lagsModelARIMA,
                                       obsStates, obsInSample,
                                       yInSample, persistence, persistenceEstimate,
                                       persistenceLevel, persistenceLevelEstimate,
                                       persistenceTrend, persistenceTrendEstimate,
                                       persistenceSeasonal, persistenceSeasonalEstimate,
                                       persistenceXreg, persistenceXregEstimate,
                                       persistenceXregProvided,
                                       phi, phiEstimate,
                                       initialType, initialLevel, initialTrend,
                                       initialSeasonal, initialArima, initialEstimate,
                                       initialLevelEstimate, initialTrendEstimate,
                                       initialSeasonalEstimate, initialArimaEstimate,
                                       initialXregEstimate, initialXregProvided,
                                       arimaModel, arRequired, iRequired, maRequired,
                                       armaParameters,
                                       componentsNumberARIMA, componentsNamesARIMA,
                                       formula, xregModel, xregModelInitials, xregData,
                                       xregNumber, xregNames, regressors,
                                       xregParametersMissing, xregParametersIncluded,
                                       xregParametersEstimated, xregParametersPersistence,
                                       constantRequired, constantEstimate, constantValue,
                                       constantName,
                                       ot, otLogical, occurrenceModel, yFitted,
                                       bounds, loss, lossFunction, distribution,
                                       horizon, multisteps, other, otherParameterEstimate,
                                       lambda, B){
            res <- omEstimator(etsModel, Etype, Ttype, Stype, lags,
                               lagsModelSeasonal, lagsModelARIMA,
                               obsStates, obsInSample,
                               yInSample, persistence, persistenceEstimate,
                               persistenceLevel, persistenceLevelEstimate,
                               persistenceTrend, persistenceTrendEstimate,
                               persistenceSeasonal, persistenceSeasonalEstimate,
                               persistenceXreg, persistenceXregEstimate,
                               persistenceXregProvided,
                               phi, phiEstimate,
                               initialType, initialLevel, initialTrend,
                               initialSeasonal, initialArima, initialEstimate,
                               initialLevelEstimate, initialTrendEstimate,
                               initialSeasonalEstimate, initialArimaEstimate,
                               initialXregEstimate, initialXregProvided,
                               arimaModel, arRequired, iRequired, maRequired,
                               armaParameters,
                               componentsNumberARIMA, componentsNamesARIMA,
                               formula, xregModel, xregModelInitials, xregData,
                               xregNumber, xregNames, regressors,
                               xregParametersMissing, xregParametersIncluded,
                               xregParametersEstimated, xregParametersPersistence,
                               constantRequired, constantEstimate, constantValue,
                               constantName,
                               ot, otLogical, occurrenceModel, yFitted,
                               bounds, loss, lossFunction, "dnorm",
                               horizon, multisteps, other, otherParameterEstimate,
                               lambda, B);
            .icEnv$nP <- res$nParamEstimated;
            res$IC <- icFunction(res$logLikADAMValue);
            return(res);
        }

        adamSelected <- adam_selector(omEstimatorWrapper,
                                      model, modelsPool, allowMultiplicative,
                                      modelDo=modelDo,
                                      etsModel, Etype, Ttype, Stype, damped, lags,
                                      lagsModelSeasonal, lagsModelARIMA,
                                      obsStates, obsInSample,
                                      yInSample, persistence, persistenceEstimate,
                                      persistenceLevel, persistenceLevelEstimate,
                                      persistenceTrend, persistenceTrendEstimate,
                                      persistenceSeasonal, persistenceSeasonalEstimate,
                                      persistenceXreg, persistenceXregEstimate,
                                      persistenceXregProvided,
                                      phi, phiEstimate,
                                      initialType, initialLevel, initialTrend, initialSeasonal,
                                      initialArima, initialEstimate,
                                      initialLevelEstimate, initialTrendEstimate,
                                      initialSeasonalEstimate,
                                      initialArimaEstimate, initialXregEstimate,
                                      initialXregProvided,
                                      arimaModel, arRequired, iRequired, maRequired,
                                      armaParameters,
                                      componentsNumberARIMA, componentsNamesARIMA,
                                      formula, xregModel, xregModelInitials, xregData,
                                      xregNumber, xregNames, regressors,
                                      xregParametersMissing, xregParametersIncluded,
                                      xregParametersEstimated, xregParametersPersistence,
                                      constantRequired, constantEstimate, constantValue,
                                      constantName,
                                      ot, otLogical, occurrenceModel, yFitted,
                                      icFunction,
                                      bounds, loss, lossFunction, "dnorm",
                                      horizon, multisteps, other, otherParameterEstimate,
                                      lambda, silent, B);

        icSelection <- adamSelected$icSelection;
        bestIdx <- which.min(icSelection)[1];
        modelOriginal <- model;
        estimatorResult <- adamSelected$results[[bestIdx]];
        list2env(estimatorResult, environment());

        if(modelDo == "combine"){
            icWeights <- adam_ic_weights(icSelection);
            # temporary weights, dropping the very small ones
            icWeightsTemp <- icWeights
            icWeightsTemp[icWeightsTemp<1e-5] <- 0;
            # Calculate sensible weights
            wSum <- sum(icWeightsTemp);
            # Amend the weights if they don't add up to 1
            if(wSum > 0 && wSum!=1){
                icWeights <- icWeights / wSum;
            }

            yFittedCombined <- matrix(0, obsInSample, 1);
            yForecastCombined <- if(h > 0) numeric(h) else NULL;
            individualModels <- vector("list", length(icWeights));
            for(i in seq_along(icWeights)){
                subModel <- tryCatch(omFinalFit(adamSelected$results[[i]],
                                                hLocal=h, fullObject=TRUE),
                                     error=function(e){
                                         message("om(): combine: model ", i, " fitter failed (",
                                                 conditionMessage(e), "), dropping from average.");
                                         icWeights[i] <<- 0;
                                         NULL;
                                     });
                if(is.null(subModel)){
                    next;
                }
                individualModels[[i]] <- subModel;
                # Don't add thingy if the weight is low or there are NaNs
                if(icWeights[i] >= 1e-5 && !any(is.nan(subModel$fitted))){
                    yFittedCombined[] <- yFittedCombined +
                        icWeights[i] * subModel$fitted;
                    if(h > 0){
                        yForecastCombined[] <- yForecastCombined +
                            icWeights[i] * subModel$forecast;
                    }
                }
            }
            names(individualModels) <- if(!is.null(names(icSelection))) names(icSelection) else
                paste0("model", seq_along(individualModels));
        }

        adamArchitect <- adam_architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                          xregNumber, obsInSample, initialType,
                                          arimaModel, lagsModelARIMA, xregModel, constantRequired,
                                          componentsNumberARIMA,
                                          obsAll, yIndexAll, yClasses, adamETS);
        list2env(adamArchitect, environment());
    }
    else if(modelDo == "use"){
        # No estimation needed — parameters fully specified (e.g. "fixed" occurrence)
        estimatorResult <- list(
            B=numeric(0), nParamEstimated=nParamEstimated,
            etsModel=etsModel, Etype=Etype, Ttype=Ttype, Stype=Stype,
            modelIsTrendy=modelIsTrendy, modelIsSeasonal=modelIsSeasonal,
            res=list(objective=0), FI=NULL,
            arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
            # Per-submodel "Estimate" flags expected by omFinalFit()
            phiEstimate=phiEstimate,
            persistenceEstimate=persistenceEstimate,
            persistenceLevelEstimate=persistenceLevelEstimate,
            persistenceTrendEstimate=persistenceTrendEstimate,
            persistenceSeasonalEstimate=persistenceSeasonalEstimate,
            persistenceXregEstimate=persistenceXregEstimate,
            initialEstimate=initialEstimate,
            initialLevelEstimate=initialLevelEstimate,
            initialTrendEstimate=initialTrendEstimate,
            initialSeasonalEstimate=initialSeasonalEstimate,
            initialArimaEstimate=initialArimaEstimate,
            initialXregEstimate=initialXregEstimate,
            arimaModel=arimaModel,
            arRequired=arRequired, iRequired=iRequired, maRequired=maRequired,
            arEstimate=arEstimate, maEstimate=maEstimate);
    }
    else{
        estimatorResult <- omEstimator(etsModel, Etype, Ttype, Stype, lags,
                                       lagsModelSeasonal, lagsModelARIMA,
                                       obsStates, obsInSample,
                                       yInSample, persistence, persistenceEstimate,
                                       persistenceLevel, persistenceLevelEstimate,
                                       persistenceTrend, persistenceTrendEstimate,
                                       persistenceSeasonal, persistenceSeasonalEstimate,
                                       persistenceXreg, persistenceXregEstimate,
                                       persistenceXregProvided,
                                       phi, phiEstimate,
                                       initialType, initialLevel, initialTrend,
                                       initialSeasonal, initialArima, initialEstimate,
                                       initialLevelEstimate, initialTrendEstimate,
                                       initialSeasonalEstimate,
                                       initialArimaEstimate, initialXregEstimate,
                                       initialXregProvided,
                                       arimaModel, arRequired, iRequired, maRequired,
                                       armaParameters,
                                       componentsNumberARIMA, componentsNamesARIMA,
                                       formula, xregModel, xregModelInitials, xregData,
                                       xregNumber, xregNames, regressors,
                                       xregParametersMissing, xregParametersIncluded,
                                       xregParametersEstimated, xregParametersPersistence,
                                       constantRequired, constantEstimate, constantValue,
                                       constantName,
                                       ot, otLogical, occurrenceModel, yFitted,
                                       bounds, loss, lossFunction, "dnorm",
                                       horizon, multisteps, other, otherParameterEstimate,
                                       lambda, B);
        list2env(estimatorResult, environment());
    }

    #### Build return object via omFinalFit ####
    if(modelDo == "combine"){
        modelReturned <- omFinalFit(estimatorResult, hLocal=h, fullObject=TRUE,
                                    yFittedOverride=yFittedCombined,
                                    yForecastOverride=yForecastCombined);
        modelReturned$model <- adam_model_name(etsModel, modelOriginal, xregModel, arimaModel,
                                               arOrders, iOrders, maOrders, lags,
                                               regressors, constantRequired, constantName,
                                               occurrenceType, componentsNumberETSSeasonal,
                                               prefix = "o");
        modelReturned$models <- individualModels;
        modelReturned$ICw <- icWeights;
        modelReturned$ICs <- icSelection;
        # Weighted estimated parameters across the combined pool (mirrors adam())
        nParamMat <- modelReturned$nParam;
        nParamMat[1,1] <- sum(sapply(individualModels,
                                     function(x) if(is.null(x)) 0 else nparam(x)) *
                                  icWeights);
        nParamMat[1,5] <- sum(nParamMat[1,1:4]);
        nParamMat[2,5] <- sum(nParamMat[2,1:4]);
        modelReturned$nParam <- nParamMat;
        class(modelReturned) <- c("omCombined","om","adam","smooth");
    } else {
        modelReturned <- omFinalFit(estimatorResult, hLocal=h, fullObject=TRUE);
        if(modelDo == "select"){
            modelReturned$ICs <- icSelection;
        }
    }

    if(!silent){
        plot(modelReturned, 7);
    }

    return(modelReturned);
}

# Transform initial state values for occurrence models.
# Overwrites matVt level, trend, and seasonal rows with values on the
# correct occurrence-model scale.  Only touches components whose
# corresponding *Estimate flag is TRUE (i.e. not user-provided).
# Follows the pattern of oesInitialiser in oes.R lines 288-368.
om_initial_transform <- function(matVt, occurrence, Etype, Ttype, Stype,
                                 etsModel,
                                 modelIsTrendy, modelIsSeasonal,
                                 initialLevelEstimate, initialTrendEstimate,
                                 initialSeasonalEstimate,
                                 componentsNumberETS,
                                 componentsNumberETSNonSeasonal,
                                 componentsNumberETSSeasonal,
                                 lagsModel, lagsModelMax, lagsModelSeasonal,
                                 obsInSample, ot,
                                 arimaModel, componentsNumberARIMA,
                                 initialArimaEstimate, initialArimaNumber,
                                 xregModel, xregNumber, initialXregEstimate,
                                 constantRequired, constantEstimate){

    occurrenceTransformer <- function(value){
        value <- switch(occurrence,
                        "odds-ratio"         = value / (1 - value),
                        "inverse-odds-ratio" = (1 - value) / value,
                        "fixed"              =,
                        "direct"             = value,
                        value);
        if(Etype == "A" && occurrence %in% c("odds-ratio","inverse-odds-ratio")){
            value <- log(value);
        }
        return(value);
    }

    # j tracks the number of ETS rows already consumed in matVt.
    # When etsModel is FALSE (e.g. pure ARIMA), there is no level row at index 1.
    j <- 0;
    levelOriginal <- if(etsModel) matVt[1, 1] else NA_real_;

    if(etsModel){
        #### Level ####
        if(initialLevelEstimate){
            # Failsafe in case of negative level
            if(matVt[1, 1]<0 || matVt[1, 1]>1){
                levelOriginal <- mean(ot);
                matVt[1, 1:lagsModelMax] <- levelOriginal;
            }
            matVt[1, 1:lagsModelMax] <- occurrenceTransformer(matVt[1, 1:lagsModelMax]);
        }
        j <- 1;

        #### Trend ####
        if(modelIsTrendy){
            if(initialTrendEstimate){
                # levels <- switch(Ttype,
                #                  "A"=levelOriginal + c(0, matVt[j+1, 1]),
                #                  "M"=levelOriginal * c(1, matVt[j+1, 1]));
                # levels[] <- occurrenceTransformer(levels);
                # if(Ttype=="A"){
                #     matVt[j+1, 1:lagsModelMax] <- diff(levels);
                # }
                # else{
                #     matVt[j+1, 1:lagsModelMax] <- levels[2]/levels[1];
                # }
                if(Ttype=="A"){
                    matVt[j+1, 1:lagsModelMax] <- 0;
                }
                else{
                    matVt[j+1, 1:lagsModelMax] <- 1;
                }
            }
            j[] <- j + 1;
        }

        #### Seasonal ####
        if(modelIsSeasonal){
            if(any(initialSeasonalEstimate)){
                for(i in 1:componentsNumberETSSeasonal){
                    seasonalOcc <- matVt[j+i, 1:lagsModelSeasonal[i]];
                    if(Stype=="M"){
                        # Transform this into the "multiplicative" seasonality
                        seasonalOcc[] <- seasonalOcc / levelOriginal + 1;
                    }
                    else{
                        # If additive, normalise
                        seasonalOcc[] <- seasonalOcc - mean(seasonalOcc);
                    }
                    matVt[j+i, 1:lagsModelSeasonal[i]] <- seasonalOcc;
                }
            }
            j[] <- j + componentsNumberETSSeasonal;
        }
    }

    #### ARIMA ####
    if(arimaModel){
        if(initialArimaEstimate && componentsNumberARIMA > 0){
            arimaRows <- (j+1):(j+componentsNumberARIMA);
            matVt[arimaRows, 1:initialArimaNumber] <-
                occurrenceTransformer(matVt[arimaRows, 1:initialArimaNumber, drop=FALSE]);
        }
        j[] <- j + componentsNumberARIMA;
    }

    #### Xreg ####
    if(xregModel){
        if(initialXregEstimate){
            for(k in seq_len(xregNumber)){
                matVt[j + k, 1:lagsModelMax] <-
                    occurrenceTransformer(matVt[j + k, 1:lagsModelMax]);
            }
        }
        j[] <- j + xregNumber;
    }

    #### Constant ####
    if(constantRequired && constantEstimate){
        matVt[j + 1, ] <- occurrenceTransformer(matVt[j + 1, 1]);
    }

    return(matVt);
}


omLinkFunction <- function(x, Etype, occurrence){
    switch(occurrence,
           "odds-ratio"         = switch(Etype,
                                         "M" = x / (1 + x),
                                         "A" = exp(x) / (1 + exp(x)),
                                         x / (1 + x)),
           "inverse-odds-ratio" = switch(Etype,
                                         "M" = 1 / (1 + x),
                                         "A" = 1 / (1 + exp(x)),
                                         1 / (1 + x)),
           "fixed"              =,
           "direct"             = pmin(pmax(x, 0), 1),
           x);
}

#' Forecast from an occurrence model
#'
#' Wraps \code{forecast.adam()} and applies the occurrence link function to
#' convert state-space forecasts to probabilities.
#'
#' @param object An object of class \code{om}.
#' @param h Forecast horizon. If \code{NULL}, uses \code{object$h}.
#' @param interval Type of prediction interval.
#' @param ... Additional arguments passed to \code{forecast.adam()}.
#'
#' @return An object of class \code{forecast.smooth} with probability forecasts.
#'
#' @export
forecast.om <- function(object, h=NULL, ...){
    # Intervals on the probability scale are not implemented yet, so the
    # underlying forecast.adam() call is forced to interval="none". The
    # remaining slots (level, side, cumulative) are set to their defaults so
    # the returned structure is shape-compatible with forecast.adam().
    fc <- forecast.adam(object, h=h, interval="none",
                        level=0.95, side="both", cumulative=FALSE, ...);

    Etype <- errorType(object);
    occurrence <- object$occurrence;
    fc$mean[] <- omLinkFunction(fc$mean, Etype, occurrence);
    if(occurrence %in% c("odds-ratio","inverse-odds-ratio") && Etype == "A"){
        fc$mean[is.nan(fc$mean)] <- 1;
    }

    return(fc);
}

#' @export
actuals.om <- function(object, ...){
    if(is.null(object$data)){
        return(NULL);
    }
    yObs <- if(is.data.frame(object$data) || is.matrix(object$data)) object$data[,1] else object$data;
    return(as.numeric(yObs != 0));
}

#' @export
print.om <- function(x, ...){
    cat("Occurrence model\n");
    print.adam(x, ...);
}

#' @export
summary.om <- function(object, ...){
    cat("Occurrence model\n");
    summary.adam(object, ...);
}

#' @export
print.omCombined <- function(x, ...){
    cat("Occurrence model\n");
    print.adamCombined(x, ...);
}

#' @export
summary.omCombined <- function(object, ...){
    cat("Occurrence model\n");
    summary.adamCombined(object, ...);
}

#' @export
forecast.omCombined <- function(object, h=NULL, ...){
    # Interval-related slots are fixed internally so the returned structure
    # matches forecast.adamCombined's field set; intervals on the probability
    # scale are not implemented yet.
    interval   <- "none";
    level      <- 0.95;
    side       <- "both";
    cumulative <- FALSE;

    obsInSample <- nobs(object);
    yClasses <- class(actuals(object));
    if(is.null(h)){
        h <- length(object$forecast);
    }
    nLevels <- length(level);

    if(any(yClasses == "ts")){
        yForecastStart <- time(actuals(object))[obsInSample] + deltat(actuals(object));
        yFrequency <- frequency(actuals(object));
        yForecast <- ts(rep(0, h), start=yForecastStart, frequency=yFrequency);
        yLower <- yUpper <- ts(matrix(NA_real_, h, nLevels),
                               start=yForecastStart, frequency=yFrequency);
    } else {
        yIndex <- time(actuals(object));
        yForecastIndex <- yIndex[obsInSample] + diff(tail(yIndex, 2)) * (1:h);
        yForecast <- zoo(rep(0, h), order.by=yForecastIndex);
        yLower <- yUpper <- zoo(matrix(NA_real_, h, nLevels),
                                order.by=yForecastIndex);
    }

    object$ICw[object$ICw < 1e-2] <- 0;
    object$ICw[] <- object$ICw / sum(object$ICw);

    for(i in seq_along(object$models)){
        if(object$ICw[i] == 0){
            next;
        }
        fc_i <- forecast.adam(object$models[[i]], h=h, interval=interval,
                              level=level, side=side, cumulative=cumulative, ...);
        Etype_i <- errorType(object$models[[i]]);
        fc_i$mean[] <- omLinkFunction(fc_i$mean, Etype_i, object$occurrence);
        yForecast[] <- yForecast + fc_i$mean * object$ICw[i];
    }

    object$models <- NULL;
    return(structure(list(mean=yForecast,
                          lower=yLower, upper=yUpper,
                          model=object,
                          level=level, interval=interval,
                          side=side, cumulative=cumulative, h=h),
                     class=c("adam.forecast","smooth.forecast","forecast")));
}
