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
                                 lagsModel, lagsModelMax,
                                 obsInSample, ot,
                                 arimaModel, componentsNumberARIMA,
                                 initialArimaEstimate, initialArimaNumber,
                                 xregModel, xregNumber, initialXregEstimate,
                                 constantRequired, constantEstimate){

    occurrenceTransformer <- function(value){
        value <- switch(occurrence,
            "odds-ratio"         = value / (1 - value),
            "inverse-odds-ratio" = (1 - value) / value,
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
                levels <- levelOriginal + c(0, matVt[j+1, 1]);
                levels[] <- occurrenceTransformer(levels);
                if(Ttype=="A"){
                    matVt[j+1, 1:lagsModelMax] <- diff(levels);
                }
                else{
                    matVt[j+1, 1:lagsModelMax] <- levels[2]/levels[1];
                }
            }
            j[] <- j + 1;
        }

        #### Seasonal ####
        if(modelIsSeasonal){
            if(any(initialSeasonalEstimate)){
                for(i in 1:componentsNumberETSNonSeasonal){
                    seasonalOcc <- matVt[1, j+i];
                    # Transform this into the "multiplicative" seasonality
                    seasonalOcc[] <- seasonalOcc / levelOriginal + 1;
                    # If additive, transform via logs and normalise
                    if(Stype=="A"){
                        seasonalOcc[] <- log(seasonalOcc);
                        seasonalOcc[] <- seasonalOcc - mean(seasonalOcc);
                    }
                    matVt[1, j+i] <- seasonalOcc;
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

#' Occurrence Model
#'
#' Fits a state-space occurrence (probability) model to binary time series
#' data using ADAM's C++ infrastructure with Bernoulli log-likelihood.
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
#' @export
om <- function(data,
               model = "ZXZ",
               lags  = c(frequency(data)),
               orders = list(ar=c(0), i=c(0), ma=c(0), select=FALSE),
               constant = FALSE,
               formula  = NULL,
               regressors = c("use","select","adapt"),
               occurrence = c("fixed","odds-ratio","inverse-odds-ratio","direct"),
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

    #### Handle "fixed" occurrence separately ####
    if(occurrence == "fixed"){
        oInSample <- matrix(as.numeric(yInSample != 0), ncol=1);
        if(holdout){
            oHoldout <- matrix(as.numeric(yHoldout != 0), ncol=1);
        }
        iprob <- mean(oInSample);
        pFitted <- rep(iprob, obsInSample);
        pForecast <- if(h > 0) rep(iprob, h) else numeric(0);
        logLikValue <- sum(log(pmax(pFitted[as.logical(oInSample)], 1e-10))) +
            sum(log(pmax(1 - pFitted[!as.logical(oInSample)], 1e-10)));
        if(any(yClasses=="ts")){
            fittedTS    <- ts(pFitted, start=yStart, frequency=yFrequency);
            residualsTS <- ts(as.numeric(oInSample) - pFitted, start=yStart, frequency=yFrequency);
            forecastTS  <- if(h > 0) ts(pForecast, start=yForecastStart, frequency=yFrequency) else
                           ts(NA, start=yForecastStart, frequency=yFrequency);
        } else {
            fittedTS    <- zoo(pFitted, order.by=yInSampleIndex);
            residualsTS <- zoo(as.numeric(oInSample) - pFitted, order.by=yInSampleIndex);
            forecastTS  <- if(h > 0) zoo(pForecast, order.by=yForecastIndex) else
                           zoo(NA, order.by=yForecastIndex[1]);
        }
        parametersNumber <- matrix(0, 2, 5,
                                   dimnames=list(c("Estimated","Provided"),
                                                 c("nParamInternal","nParamXreg",
                                                   "nParamOccurrence","nParamScale","nParamAll")));
        parametersNumber[1,3] <- 1;
        parametersNumber[1,5] <- 1;
        modelReturned <- list(
            model = "oETS[F](MNN)",
            occurrence = occurrenceType,
            loss = loss,
            distribution = "plogis",
            timeElapsed = Sys.time() - startTime,
            fitted = fittedTS,
            residuals = residualsTS,
            forecast = forecastTS,
            states = NULL,
            B = numeric(0),
            persistence = numeric(0),
            phi = 1,
            lags = lags, lagsAll = lags,
            orders = orders,
            constant = NULL,
            ets = FALSE,
            ICs = NULL,
            FI = NULL,
            logLik = logLikValue,
            nParam = parametersNumber,
            scale = NA,
            call = cl
        );
        if(holdout){
            modelReturned$holdout <- oHoldout;
            modelReturned$accuracy <- measures(as.vector(oHoldout), forecastTS,
                                               as.vector(oInSample));
        }
        class(modelReturned) <- c("om","adam","smooth");
        return(modelReturned);
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
    occurrence <- occurrenceType;

    # Binary indicators (ot from checker is already binary when occurrence != "none")
    oInSample <- matrix(as.numeric(ot), ncol=1);
    if(holdout){
        oHoldout <- matrix(as.numeric(yHoldout != 0), ncol=1);
    }

    # Override occurrence-related flags set by checker
    occurrenceModel <- FALSE;
    oesModel <- NULL;
    pFitted <- matrix(rep(mean(oInSample), obsInSample), ncol=1);
    refineHead <- TRUE;
    adamETS <- (ets == "adam");

    #### Optimiser settings ####
    optimSettings <- adam_checkOptimizer(ellipsis=ellipsis, loss=loss, distribution="dnorm",
                                         initialType=initialType, lags=lags,
                                         arimaModel=arimaModel);
    list2env(optimSettings, envir=environment());

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
                             ot, otLogical, occurrenceModel, pFitted,
                             bounds, loss, lossFunction, distribution,
                             horizon, multisteps, other, otherParameterEstimate,
                             lambda, B){
        adamArchitect <- adam_architector(etsModel, Etype, Ttype, Stype, lags,
                                          lagsModelSeasonal,
                                          xregNumber, obsInSample, initialType,
                                          arimaModel, lagsModelARIMA, xregModel,
                                          constantRequired,
                                          componentsNumberARIMA,
                                          obsAll, yIndexAll, yClasses, adamETS);
        list2env(adamArchitect, environment());

        # Etype="A" is needed for the decomposition to work in case of 0/1 data
        adamCreated <- adam_creator(etsModel, Etype="A", Ttype, Stype,
                                    modelIsTrendy, modelIsSeasonal,
                                    lags, lagsModel, lagsModelARIMA, lagsModelAll,
                                    lagsModelMax,
                                    profilesRecentTable, FALSE,
                                    obsStates, obsInSample,
                                    obsAll,
                                    componentsNumberETS, componentsNumberETSSeasonal,
                                    componentsNamesETS, otLogical, yInSample,
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
            lagsModel, lagsModelMax, obsInSample, ot,
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
                                    ets, bounds, yInSample, otLogical,
                                    iOrders, armaParameters, other);

        B_used <- BValues$B;

        lb <- BValues$Bl;
        ub <- BValues$Bu;

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

        omCF_local <- function(B){
            adamElems <- adam_filler(B,
                                     etsModel, Etype, Ttype, Stype,
                                     modelIsTrendy, modelIsSeasonal,
                                     componentsNumberETS, componentsNumberETSNonSeasonal,
                                     componentsNumberETSSeasonal, componentsNumberARIMA,
                                     lags, lagsModel, lagsModelMax,
                                     adamCreated$matVt, adamCreated$matWt,
                                     adamCreated$matF, adamCreated$vecG,
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
                                     nonZeroARI,
                                     nonZeroMA,
                                     adamCreated$arimaPolynomials,
                                     xregModel, xregNumber,
                                     xregParametersMissing, xregParametersIncluded,
                                     xregParametersEstimated, xregParametersPersistence,
                                     constantEstimate,
                                     adamCpp,
                                     constantRequired, initialArimaNumber);
            penalty <- adam_bounds_checker(adamElems, adamElems$arimaPolynomials,
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
            profilesRecentTable[] <- adamElems$matVt[, 1:lagsModelMax];
            adamFitted <- adamCpp$fit(adamElems$matVt, adamElems$matWt,
                                      adamElems$matF, adamElems$vecG,
                                      indexLookupTable, profilesRecentTable,
                                      as.numeric(ot), as.numeric(ot),
                                      any(initialType == c("complete","backcasting")),
                                      nIterations, refineHead, occurrenceChar);
            p <- pmin(pmax(adamFitted$fitted, 1e-10), 1 - 1e-10);
            if(loss == "likelihood"){
                return(-(sum(log(p[otLogical])) + sum(log(1 - p[!otLogical]))));
            }
            return(mean((as.numeric(ot) - p)^2));
        }

        maxeval <- if(is.null(ellipsis$maxeval)) length(B_used)*40 else ellipsis$maxeval;
        res <- suppressWarnings(nloptr(B_used, omCF_local, lb=lb, ub=ub,
                                       opts=list(algorithm=algorithm, xtol_rel=xtol_rel,
                                                 maxeval=maxeval,
                                                 print_level=print_level)));

        if(is.infinite(res$objective) || res$objective == 1e+300){
            B_used[] <- BValues$B;
            res <- suppressWarnings(nloptr(B_used, omCF_local, lb=lb, ub=ub,
                                           opts=list(algorithm=algorithm, xtol_rel=xtol_rel,
                                                     maxeval=maxeval,
                                                     print_level=print_level)));
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
                    modelIsTrendy=modelIsTrendy, modelIsSeasonal=modelIsSeasonal));
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

    #### Helper: re-run final fit for a given estimator result (used in combination) ####
    omFinalFit <- function(res, hLocal=0){
        arch <- adam_architector(res$etsModel, res$Etype, res$Ttype, res$Stype,
                                  lags, lagsModelSeasonal,
                                  xregNumber, obsInSample, initialType,
                                  arimaModel, lagsModelARIMA,
                                  xregModel, constantRequired,
                                  componentsNumberARIMA,
                                  obsAll, yIndexAll, yClasses, adamETS);
        cr <- adam_creator(res$etsModel, res$Etype, res$Ttype, res$Stype,
                           res$modelIsTrendy, res$modelIsSeasonal,
                           lags, arch$lagsModel, lagsModelARIMA,
                           arch$lagsModelAll, arch$lagsModelMax,
                           arch$profilesRecentTable, FALSE,
                           arch$obsStates, obsInSample,
                           obsAll,
                           arch$componentsNumberETS, arch$componentsNumberETSSeasonal,
                           arch$componentsNamesETS, otLogical, yInSample,
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
                           arch$adamCpp,
                           arEstimate, maEstimate, smoother,
                           nonZeroARI, nonZeroMA);
        fill <- adam_filler(res$B,
                            res$etsModel, res$Etype, res$Ttype, res$Stype,
                            res$modelIsTrendy, res$modelIsSeasonal,
                            arch$componentsNumberETS, arch$componentsNumberETSNonSeasonal,
                            arch$componentsNumberETSSeasonal, componentsNumberARIMA,
                            lags, arch$lagsModel, arch$lagsModelMax,
                            cr$matVt, cr$matWt, cr$matF, cr$vecG,
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
                            nonZeroARI, nonZeroMA, cr$arimaPolynomials,
                            xregModel, xregNumber,
                            xregParametersMissing, xregParametersIncluded,
                            xregParametersEstimated, xregParametersPersistence,
                            constantEstimate, arch$adamCpp,
                            constantRequired, initialArimaNumber);
        prof <- fill$matVt[, 1:arch$lagsModelMax, drop=FALSE];
        fit <- arch$adamCpp$fit(fill$matVt, fill$matWt, fill$matF, fill$vecG,
                                arch$indexLookupTable, prof,
                                as.numeric(ot), as.numeric(ot),
                                any(initialType == c("complete","backcasting")),
                                nIterations, refineHead, occurrenceChar);
        fitted <- pmin(pmax(fit$fitted, 1e-10), 1 - 1e-10);
        if(hLocal == 0){
            return(fitted);
        }
        # Compute forecast on the occurrence-probability scale
        profForecast <- fit$profile;
        fc <- arch$adamCpp$forecast(tail(fill$matWt, hLocal), fill$matF,
                                    arch$indexLookupTable[, arch$lagsModelMax + obsInSample + 1:hLocal,
                                                          drop=FALSE],
                                    profForecast, hLocal)$forecast;
        fc[is.nan(fc)] <- 0;
        fc <- pmin(pmax(fc, 1e-10), 1 - 1e-10);
        return(list(fitted=fitted, forecast=as.vector(fc)));
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
                                        ot, otLogical, occurrenceModel, pFitted,
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
                               ot, otLogical, occurrenceModel, pFitted,
                               bounds, loss, lossFunction, "dnorm",
                               horizon, multisteps, other, otherParameterEstimate,
                               lambda, B);
            .icEnv$nP <- res$nParamEstimated;
            res$IC <- icFunction(res$logLikADAMValue);
            return(res);
        }

        adamSelected <- adam_selector(omEstimatorWrapper,
                                      model, modelsPool, allowMultiplicative,
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
                                      ot, otLogical, occurrenceModel, pFitted,
                                      icFunction,
                                      bounds, loss, lossFunction, "dnorm",
                                      horizon, multisteps, other, otherParameterEstimate,
                                      lambda, silent, B);

        icSelection <- adamSelected$icSelection;
        bestIdx <- which.min(icSelection)[1];
        list2env(adamSelected$results[[bestIdx]], environment());

        if(modelDo == "combine"){
            icWeights <- adam_ic_weights(icSelection);
            pFittedCombined <- matrix(0, obsInSample, 1);
            pForecastCombined <- if(h > 0) numeric(h) else NULL;
            individualModels <- vector("list", length(icWeights));
            for(i in seq_along(icWeights)){
                individualModels[[i]] <- adamSelected$results[[i]];
                if(icWeights[i] < 1e-5){
                    next;
                }
                fitOut <- tryCatch(omFinalFit(adamSelected$results[[i]], hLocal=h),
                                   error=function(e){
                                       message("om(): combine: model ", i, " fitter failed (",
                                               conditionMessage(e), "), dropping from average.");
                                       icWeights[i] <<- 0;
                                       NULL;
                                   });
                if(is.null(fitOut)){
                    next;
                }
                if(h > 0){
                    pFittedCombined <- pFittedCombined + icWeights[i] * fitOut$fitted;
                    pForecastCombined <- pForecastCombined + icWeights[i] * fitOut$forecast;
                } else {
                    pFittedCombined <- pFittedCombined + icWeights[i] * fitOut;
                }
            }
            wSum <- sum(icWeights);
            if(wSum > 0 && abs(wSum - 1) > 1e-10){
                # Renormalise after dropping failed models
                icWeights <- icWeights / wSum;
                pFittedCombined <- pFittedCombined / wSum;
                if(h > 0 && !is.null(pForecastCombined)){
                    pForecastCombined <- pForecastCombined / wSum;
                }
            }
            nParamEstimated <- round(sum(icWeights *
                sapply(adamSelected$results, function(x) x$nParamEstimated)));
            names(individualModels) <- if(!is.null(names(icSelection))) names(icSelection) else
                paste0("model", seq_along(individualModels));
        }

        adamArchitect <- adam_architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                          xregNumber, obsInSample, initialType,
                                          arimaModel, lagsModelARIMA, xregModel, constantRequired,
                                          componentsNumberARIMA,
                                          obsAll, yIndexAll, yClasses, adamETS);
        list2env(adamArchitect, environment());
    } else {
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
                                        ot, otLogical, occurrenceModel, pFitted,
                                        bounds, loss, lossFunction, "dnorm",
                                        horizon, multisteps, other, otherParameterEstimate,
                                        lambda, B);
        list2env(estimatorResult, environment());

        adamArchitect <- adam_architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                          xregNumber, obsInSample, initialType,
                                          arimaModel, lagsModelARIMA, xregModel, constantRequired,
                                          componentsNumberARIMA,
                                          obsAll, yIndexAll, yClasses, adamETS);
        list2env(adamArchitect, environment());
    }

    #### Final pass: fitted values and states for best/single model ####
    adamCreatedFinal <- adam_creator(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                     lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                                     profilesRecentTable, FALSE,
                                     obsStates, obsInSample,
                                     obsAll,
                                     componentsNumberETS, componentsNumberETSSeasonal,
                                     componentsNamesETS, otLogical, yInSample,
                                     persistence, persistenceEstimate,
                                     persistenceLevel, persistenceLevelEstimate,
                                     persistenceTrend, persistenceTrendEstimate,
                                     persistenceSeasonal, persistenceSeasonalEstimate,
                                     persistenceXreg, persistenceXregEstimate,
                                     persistenceXregProvided,
                                     phi, initialType, initialEstimate,
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
                                     constantRequired, constantEstimate, constantValue,
                                     constantName, adamCpp,
                                     arEstimate, maEstimate, smoother,
                                     nonZeroARI, nonZeroMA);

    adamFilledFinal <- adam_filler(B,
                                   etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                   componentsNumberETS, componentsNumberETSNonSeasonal,
                                   componentsNumberETSSeasonal, componentsNumberARIMA,
                                   lags, lagsModel, lagsModelMax,
                                   adamCreatedFinal$matVt, adamCreatedFinal$matWt,
                                   adamCreatedFinal$matF, adamCreatedFinal$vecG,
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
                                   nonZeroARI,
                                   nonZeroMA,
                                   adamCreatedFinal$arimaPolynomials,
                                   xregModel, xregNumber,
                                   xregParametersMissing, xregParametersIncluded,
                                   xregParametersEstimated, xregParametersPersistence,
                                   constantEstimate, adamCpp,
                                   constantRequired, initialArimaNumber);
    profilesRecentTable[] <- adamFilledFinal$matVt[, 1:lagsModelMax];

    adamFittedFinal <- adamCpp$fit(adamFilledFinal$matVt, adamFilledFinal$matWt,
                                   adamFilledFinal$matF, adamFilledFinal$vecG,
                                   indexLookupTable, profilesRecentTable,
                                   as.numeric(ot), as.numeric(ot),
                                   any(initialType == c("complete","backcasting")),
                                   nIterations, refineHead, occurrenceChar);
    profilesRecentTable <- adamFittedFinal$profile;

    # For combination: use IC-weighted fitted; for single/selection: use final-pass fitted
    if(modelDo == "combine"){
        pFittedFinal <- pFittedCombined;
    } else {
        pFittedFinal <- pmin(pmax(adamFittedFinal$fitted, 1e-10), 1 - 1e-10);
    }
    logLikValue <- sum(log(pFittedFinal[otLogical])) + sum(log(1 - pFittedFinal[!otLogical]));

    #### Wrap output as ts/zoo ####
    allComponentNames <- rownames(adamCreatedFinal$matVt);
    statesRaw <- adamFittedFinal$states[, (lagsModelMax+1):ncol(adamFittedFinal$states), drop=FALSE];
    if(!is.null(allComponentNames)){
        rownames(statesRaw) <- allComponentNames;
    }
    if(any(yClasses == "ts")){
        fittedTS    <- ts(pFittedFinal, start=yStart, frequency=yFrequency);
        residualsTS <- ts(as.numeric(oInSample) - pFittedFinal, start=yStart, frequency=yFrequency);
        statesTS    <- ts(t(statesRaw), start=yStart, frequency=yFrequency);
    } else {
        fittedTS    <- zoo(pFittedFinal, order.by=yInSampleIndex);
        residualsTS <- zoo(as.numeric(oInSample) - pFittedFinal, order.by=yInSampleIndex);
        statesTS    <- zoo(t(statesRaw), order.by=yInSampleIndex);
    }

    #### Parameter counts ####
    parametersNumber[1,1] <- nParamEstimated;
    parametersNumber[1,5] <- sum(parametersNumber[1,1:4]);
    parametersNumber[2,5] <- sum(parametersNumber[2,1:4]);

    #### Model name ####
    modelName <- adam_model_name(etsModel, model, xregModel, arimaModel,
                                  arOrders, iOrders, maOrders, lags,
                                  regressors, constantRequired, constantName,
                                  occurrenceType, componentsNumberETSSeasonal);

    #### ARMA parameter list ####
    if(arimaModel && (arRequired || maRequired)){
        armaParametersList <- vector("list", arRequired + maRequired);
        j <- 1L;
        if(arRequired && arEstimate){
            armaParametersList[[j]] <- B[nchar(names(B))>3 & substr(names(B),1,3)=="phi"];
            names(armaParametersList)[j] <- "ar";
            j <- j + 1L;
        } else if(arRequired){
            armaParametersList[[j]] <- armaParameters[substr(names(armaParameters),1,3)=="phi"];
            names(armaParametersList)[j] <- "ar";
            j <- j + 1L;
        }
        if(maRequired && maEstimate){
            armaParametersList[[j]] <- B[substr(names(B),1,5)=="theta"];
            names(armaParametersList)[j] <- "ma";
        } else if(maRequired){
            armaParametersList[[j]] <- armaParameters[substr(names(armaParameters),1,5)=="theta"];
            names(armaParametersList)[j] <- "ma";
        }
    } else {
        armaParametersList <- NULL;
    }

    #### Persistence vector ####
    vecGFinal <- adamFilledFinal$vecG;
    if(componentsNumberETS > 0){
        persistenceVec <- as.vector(vecGFinal)[1:componentsNumberETS];
        names(persistenceVec) <- rownames(vecGFinal)[1:componentsNumberETS];
    } else {
        persistenceVec <- numeric(0);
    }

    #### Initial values to return ####
    initialCollected <- adam_initial_collector(
        adamFittedFinal$states[, 1:lagsModelMax, drop=FALSE],
        etsModel, modelIsTrendy, modelIsSeasonal,
        lagsModel, lagsModelMax,
        initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
        componentsNumberETSSeasonal,
        arimaModel, initialArimaEstimate, initialArima, initialArimaNumber,
        componentsNumberETS, componentsNumberARIMA,
        adamFilledFinal$arimaPolynomials, Etype,
        xregModel, initialXregEstimate, xregNumber);
    initialValue <- initialCollected$initialValue;
    initialEstimated <- initialCollected$initialEstimated;

    #### Placeholders for fields filled in later layers ####
    if(!exists("profilesRecentInitial", inherits=FALSE)){
        profilesRecentInitial <- NULL;
    }
    if(!exists("otherReturned", inherits=FALSE)){
        otherReturned <- NULL;
    }
    if(!exists("FI", inherits=FALSE)){
        FI <- NULL;
    }
    if(h > 0){
        if(any(yClasses == "ts")){
            forecastTS <- ts(rep(NA, h), start=yForecastStart, frequency=yFrequency);
        } else {
            forecastTS <- zoo(rep(NA, h), order.by=yForecastIndex);
        }
        if(modelDo == "combine" && exists("pForecastCombined", inherits=FALSE) &&
           !is.null(pForecastCombined)){
            forecastTS[] <- pForecastCombined;
        } else {
            forecastValues <- adamCpp$forecast(
                tail(adamFilledFinal$matWt, h),
                adamFilledFinal$matF,
                indexLookupTable[, lagsModelMax + obsInSample + 1:h, drop=FALSE],
                profilesRecentTable, h)$forecast;
            forecastValues[is.nan(forecastValues)] <- 0;
            forecastTS[] <- pmin(pmax(forecastValues, 1e-10), 1 - 1e-10);
        }
    } else {
        forecastTS <- if(any(yClasses=="ts")){
            ts(NA, start=yForecastStart, frequency=yFrequency);
        } else {
            zoo(NA, order.by=yForecastIndex[1]);
        }
    }

    #### Construct return object ####
    modelReturned <- list(
        model = modelName,
        timeElapsed = Sys.time() - startTime,
        data = yInSample,
        fitted = fittedTS,
        residuals = residualsTS,
        forecast = forecastTS,
        states = statesTS,
        profile = profilesRecentTable,
        profileInitial = profilesRecentInitial,
        persistence = persistenceVec,
        phi = if(phiEstimate) B[names(B)=="phi"] else phi,
        transition = adamFilledFinal$matF,
        measurement = adamFilledFinal$matWt,
        initial = initialValue,
        initialType = initialType,
        initialEstimated = initialEstimated,
        orders = list(ar=arOrders, i=iOrders, ma=maOrders),
        arma = armaParametersList,
        constant = if(constantRequired) {
            if(constantEstimate) B[names(B)==constantName] else constantValue
        } else NULL,
        nParam = parametersNumber,
        occurrence = occurrenceType,
        formula = formula,
        regressors = regressors,
        loss = loss,
        lossValue = -logLikValue,
        lossFunction = lossFunction,
        logLik = logLikValue,
        distribution = "plogis",
        scale = NA,
        other = otherReturned,
        B = B,
        lags = lags,
        lagsAll = lagsModelAll,
        ets = etsModel,
        res = res,
        FI = FI,
        adamCpp = adamCpp,
        bounds = bounds,
        call = cl
    );

    if(holdout){
        modelReturned$holdout <- oHoldout;
        modelReturned$accuracy <- measures(as.vector(oHoldout), modelReturned$forecast,
                                           as.vector(oInSample));
    }

    if(modelDo == "combine"){
        modelReturned$models <- individualModels;
        modelReturned$ICw <- icWeights;
        class(modelReturned) <- c("omCombined","om","adam","smooth");
    } else {
        class(modelReturned) <- c("om","adam","smooth");
    }

    if(!silent){
        plot(modelReturned, 7);
    }

    return(modelReturned);
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
forecast.om <- function(object, h=NULL, interval="none", ...){
    if(object$occurrence == "fixed"){
        iprob <- as.numeric(object$fitted)[1];
        fc <- list(mean = ts(rep(iprob, if(is.null(h)) 1 else h),
                             start=tsp(object$fitted)[2] + 1/tsp(object$fitted)[3],
                             frequency=tsp(object$fitted)[3]),
                   lower = NULL, upper = NULL,
                   level = NULL, interval = interval,
                   model = object);
        class(fc) <- c("forecast.smooth","forecast");
        return(fc);
    }

    fc <- forecast.adam(object, h=h, interval=interval, ...);
    Etype <- substr(object$model, regexpr("\\(",object$model)+1, regexpr("\\(",object$model)+1);
    occurrence <- object$occurrence;

    .link <- function(x){
        switch(occurrence,
               "odds-ratio"         = switch(Etype,
                                             "M" = x / (1 + x),
                                             "A" = exp(x) / (1 + exp(x)),
                                             x / (1 + x)),
               "inverse-odds-ratio" = switch(Etype,
                                             "M" = 1 / (1 + x),
                                             "A" = 1 / (1 + exp(x)),
                                             1 / (1 + x)),
               "direct"             = pmin(pmax(x, 0), 1),
               x);
    }

    fc$mean[] <- .link(fc$mean);

    if(occurrence %in% c("odds-ratio","inverse-odds-ratio") && Etype == "A"){
        fc$mean[is.nan(fc$mean)] <- 1;
    }

    # TODO: implement intervals for om()
    fc$lower <- NULL;
    fc$upper <- NULL;
    fc$level <- NULL;

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
