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
#' @param silent If \code{TRUE}, suppresses output.
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
               initial = c("optimal","backcasting","two-stage","complete"),
               arma = NULL,
               ic = c("AICc","AIC","BIC","BICc"),
               bounds = c("usual","admissible","none"),
               silent = TRUE, ...){

    startTime <- Sys.time();
    cl <- match.call();

    occurrence <- match.arg(occurrence);
    loss <- match.arg(loss);
    ic <- match.arg(ic);
    bounds <- match.arg(bounds);
    regressors <- match.arg(regressors);
    initial <- match.arg(initial);
    ellipsis <- list(...);

    # Single character used in C++ fit() call
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

    # Convert to binary
    ot <- matrix(as.integer(yInSample != 0), ncol=1);
    if(any(yClasses=="ts")){
        ot <- ts(ot, start=yStart, frequency=yFrequency);
    }
    else{
        ot <- ts(ot, start=c(0,0), frequency=max(lags));
    }
    otLogical <- as.logical(ot);
    obsNonzero <- sum(otLogical);
    obsZero <- obsInSample - obsNonzero;

    # For binary data, all obs are "demand" observations
    allowMultiplicative <- TRUE;

    #### Handle "fixed" occurrence separately ####
    if(occurrence == "fixed"){
        iprob <- mean(ot[1:obsInSample]);
        pFitted <- rep(iprob, obsInSample);
        pForecast <- if(h > 0) rep(iprob, h) else numeric(0);
        logLikValue <- sum(log(pmax(pFitted[otLogical], 1e-10))) +
            sum(log(pmax(1 - pFitted[!otLogical], 1e-10)));
        if(any(yClasses=="ts")){
            fittedTS <- ts(pFitted, start=yStart, frequency=yFrequency);
            residualsTS <- ts(as.numeric(ot) - pFitted, start=yStart, frequency=yFrequency);
            if(h > 0){
                forecastTS <- ts(pForecast, start=yForecastStart, frequency=yFrequency);
            }
            else{
                forecastTS <- ts(NA, start=yForecastStart, frequency=yFrequency);
            }
        }
        else{
            fittedTS <- pFitted;
            residualsTS <- as.numeric(ot) - pFitted;
            forecastTS <- pForecast;
        }
        parametersNumber <- matrix(0, 2, 5,
                                   dimnames=list(c("Estimated","Provided"),
                                                 c("nParamInternal","nParamXreg",
                                                   "nParamOccurrence","nParamScale","nParamAll")));
        parametersNumber[1,1] <- 0;
        parametersNumber[1,5] <- 0;
        modelReturned <- list(
            model = "iConstant[F]",
            occurrence = occurrence,
            loss = loss,
            distribution = "dbinom",
            timeElapsed = Sys.time() - startTime,
            fitted = fittedTS,
            residuals = residualsTS,
            forecast = forecastTS,
            states = NULL,
            B = numeric(0),
            persistence = numeric(0),
            phi = 1,
            lags = lags,
            lagsAll = lags,
            orders = orders,
            logLik = logLikValue,
            nParam = parametersNumber,
            scale = NA,
            iprob = iprob,
            y = ot[1:obsInSample],
            call = cl
        );
        if(holdout){
            modelReturned$holdout <- yHoldout;
            modelReturned$accuracy <- measures(yHoldout, forecastTS, ot[1:obsInSample]);
        }
        class(modelReturned) <- c("om","adam","smooth");
        return(modelReturned);
    }

    occurrenceSaved <- occurrence;

    #### Call parametersChecker to set up ETS/ARIMA/xreg structure ####
    # Pass occurrence="none" so the checker doesn't try to fit a separate oes() model.
    # We override ot/otLogical/allowMultiplicative afterwards.
    checkerReturn <- parametersChecker(data=data, model=model, lags=lags,
                                       formulaToUse=formula, orders=orders,
                                       constant=constant, arma=arma,
                                       persistence=persistence, phi=phi,
                                       initial=initial,
                                       distribution="dnorm",
                                       loss=if(loss=="likelihood") "likelihood" else loss,
                                       h=h, holdout=holdout,
                                       occurrence="none",
                                       ic=ic, bounds=bounds, regressors=regressors,
                                       yName=yName,
                                       silent=silent, modelDo=modelDo,
                                       ellipsis=ellipsis, fast=FALSE);
    list2env(checkerReturn, envir=environment());
    # Restore occurrence: checkerReturn overwrites it with "none" since we passed occurrence="none"
    occurrence <- occurrenceSaved;

    # Override occurrence-related variables after checker
    ot <- matrix(as.integer(yInSample != 0), ncol=1);
    if(any(yClasses=="ts")){
        ot <- ts(ot, start=yStart, frequency=yFrequency);
    }
    else{
        ot <- ts(ot, start=c(0,0), frequency=lagsModelMax);
    }
    otLogical <- as.logical(ot);
    allowMultiplicative[] <- TRUE;
    occurrenceModel <- FALSE;
    oesModel <- NULL;
    pFitted <- matrix(rep(mean(ot), obsInSample), ncol=1);
    pForecast <- rep(mean(ot), max(h, 1));

    refineHead <- TRUE;
    adamETS <- TRUE;

    #### Optimiser settings ####
    optimSettings <- adam_checkOptimizer(ellipsis=ellipsis, loss=loss, distribution="dnorm",
                                         initialType=initialType, lags=lags,
                                         arimaModel=arimaModel);
    list2env(optimSettings, envir=environment());

    #### Inner estimator for a single model ####
    omEstimator <- function(etsModelE, EtypeE, TtypeE, StypeE, lagsE,
                             lagsModelSeasonalE, lagsModelARIMAE,
                             obsStatesE, obsInSampleE,
                             yInSampleE, persistenceE, persistenceEstimateE,
                             persistenceLevelE, persistenceLevelEstimateE,
                             persistenceTrendE, persistenceTrendEstimateE,
                             persistenceSeasonalE, persistenceSeasonalEstimateE,
                             persistenceXregE, persistenceXregEstimateE,
                             persistenceXregProvidedE,
                             phiE, phiEstimateE,
                             initialTypeE, initialLevelE, initialTrendE,
                             initialSeasonalE, initialArimaE, initialEstimateE,
                             initialLevelEstimateE, initialTrendEstimateE,
                             initialSeasonalEstimateE, initialArimaEstimate,
                             initialXregEstimateE, initialXregProvidedE,
                             arimaModelE, arRequiredE, iRequiredE, maRequiredE,
                             armaParametersE,
                             componentsNumberARIMAE, componentsNamesARIMAE,
                             formulaE, xregModelE, xregModelInitialsE, xregDataE,
                             xregNumberE, xregNamesE, regressorsE,
                             xregParametersMissingE, xregParametersIncludedE,
                             xregParametersEstimatedE, xregParametersPersistenceE,
                             constantRequiredE, constantEstimateE, constantValueE,
                             constantNameE,
                             otE, otLogicalE, occurrenceModelE, pFittedE,
                             boundsE, lossE, lossFunctionE, distributionE,
                             horizonE, multistepsE, otherE, otherParameterEstimateE,
                             lambdaE, BE){
        # Architecture and state space matrices
        adamArchitectE <- adam_architector(etsModelE, EtypeE, TtypeE, StypeE, lagsE,
                                           lagsModelSeasonalE,
                                           xregNumberE, obsInSampleE, initialTypeE,
                                           arimaModelE, lagsModelARIMAE, xregModelE,
                                           constantRequiredE,
                                           componentsNumberARIMA,
                                           obsAll, yIndexAll, yClasses, adamETS);
        modelIsTrendyE              <- adamArchitectE$modelIsTrendy;
        modelIsSeasonalE            <- adamArchitectE$modelIsSeasonal;
        lagsModelE                  <- adamArchitectE$lagsModel;
        lagsModelMaxE               <- adamArchitectE$lagsModelMax;
        lagsModelAllE               <- adamArchitectE$lagsModelAll;
        obsStatesE                  <- adamArchitectE$obsStates;
        indexLookupTableE           <- adamArchitectE$indexLookupTable;
        profilesRecentTableE        <- adamArchitectE$profilesRecentTable;
        adamCppE                    <- adamArchitectE$adamCpp;
        componentsNumberETSE        <- adamArchitectE$componentsNumberETS;
        componentsNumberETSSeasonalE <- adamArchitectE$componentsNumberETSSeasonal;
        componentsNumberETSNonSeasonalE <- adamArchitectE$componentsNumberETSNonSeasonal;
        componentsNamesETSE         <- adamArchitectE$componentsNamesETS;

        adamCreatedE <- adam_creator(etsModelE, EtypeE, TtypeE, StypeE,
                                     modelIsTrendyE, modelIsSeasonalE,
                                     lagsE, lagsModelE, lagsModelARIMAE, lagsModelAllE,
                                     lagsModelMaxE,
                                     profilesRecentTableE, FALSE,
                                     obsStatesE, obsInSampleE,
                                     obsInSampleE + lagsModelMaxE,
                                     componentsNumberETSE, componentsNumberETSSeasonalE,
                                     componentsNamesETSE, otLogicalE, yInSampleE,
                                     persistenceE, persistenceEstimateE,
                                     persistenceLevelE, persistenceLevelEstimateE,
                                     persistenceTrendE, persistenceTrendEstimateE,
                                     persistenceSeasonalE, persistenceSeasonalEstimateE,
                                     persistenceXregE, persistenceXregEstimateE,
                                     persistenceXregProvidedE,
                                     phiE,
                                     initialTypeE, initialEstimateE,
                                     initialLevelE, initialLevelEstimateE,
                                     initialTrendE, initialTrendEstimateE,
                                     initialSeasonalE, initialSeasonalEstimateE,
                                     initialArimaE, initialArimaEstimate,
                                     initialArimaNumber,
                                     initialXregEstimateE, initialXregProvidedE,
                                     arimaModelE, arRequiredE, iRequiredE, maRequiredE,
                                     armaParametersE,
                                     arOrders, iOrders, maOrders,
                                     componentsNumberARIMAE, componentsNamesARIMAE,
                                     xregModelE, xregModelInitialsE, xregDataE,
                                     xregNumberE, xregNamesE,
                                     xregParametersPersistenceE,
                                     constantRequiredE, constantEstimateE,
                                     constantValueE, constantNameE,
                                     adamCppE,
                                     arEstimate, maEstimate, smoother,
                                     nonZeroARI, nonZeroMA);

        # Initialise B
        BValuesE <- adam_initialiser(etsModelE, EtypeE, TtypeE, StypeE,
                                     modelIsTrendyE, modelIsSeasonalE,
                                     componentsNumberETSNonSeasonalE,
                                     componentsNumberETSSeasonalE,
                                     componentsNumberETSE,
                                     lagsE, lagsModelE, lagsModelSeasonalE,
                                     lagsModelARIMAE, lagsModelMaxE,
                                     adamCreatedE$matVt,
                                     persistenceEstimateE, persistenceLevelEstimateE,
                                     persistenceTrendEstimateE,
                                     persistenceSeasonalEstimateE,
                                     persistenceXregEstimateE,
                                     phiEstimateE, initialTypeE, initialEstimateE,
                                     initialLevelEstimateE, initialTrendEstimateE,
                                     initialSeasonalEstimateE,
                                     initialArimaEstimate, initialXregEstimateE,
                                     arimaModelE, arRequiredE, maRequiredE,
                                     arEstimate, maEstimate,
                                     arOrders, maOrders,
                                     componentsNumberARIMAE, componentsNamesARIMAE,
                                     initialArimaNumber,
                                     xregModelE, xregNumberE,
                                     xregParametersEstimatedE, xregParametersPersistenceE,
                                     constantEstimateE, constantNameE,
                                     otherParameterEstimateE,
                                     adamCppE,
                                     "adam", boundsE, yInSampleE, otLogicalE,
                                     iOrders, armaParametersE, otherE);

        BE_used <- BValuesE$B;

        # Override initial level with oes-style transform
        p0 <- mean(as.numeric(otE));
        p0 <- max(1e-4, min(1 - 1e-4, p0));
        levelInitIdx <- which(names(BE_used) %in% c("level","l"));
        if(length(levelInitIdx) > 0){
            BE_used[levelInitIdx[1]] <- switch(occurrence,
                "odds-ratio"         = p0 / (1 - p0),
                "inverse-odds-ratio" = (1 - p0) / p0,
                "direct"             = p0,
                p0);
            # For additive ETS + odds-ratio/inverse: transform to log scale
            if(EtypeE == "A" && occurrence %in% c("odds-ratio","inverse-odds-ratio")){
                BE_used[levelInitIdx[1]] <- log(max(1e-4, BE_used[levelInitIdx[1]]));
            }
        }

        lbE <- BValuesE$Bl;
        ubE <- BValuesE$Bu;

        # ARIMA companion matrices for bounds checking
        if(arimaModelE){
            arPolynomialMatrixE <- matrix(0, arOrders %*% lagsE, arOrders %*% lagsE);
            if(nrow(arPolynomialMatrixE) > 1){
                arPolynomialMatrixE[2:nrow(arPolynomialMatrixE)-1, 2:nrow(arPolynomialMatrixE)] <-
                    diag(nrow(arPolynomialMatrixE) - 1);
            }
            maPolynomialMatrixE <- matrix(0, maOrders %*% lagsE, maOrders %*% lagsE);
            if(nrow(maPolynomialMatrixE) > 1){
                maPolynomialMatrixE[2:nrow(maPolynomialMatrixE)-1, 2:nrow(maPolynomialMatrixE)] <-
                    diag(nrow(maPolynomialMatrixE) - 1);
            }
        }
        else{
            arPolynomialMatrixE <- maPolynomialMatrixE <- NULL;
        }

        # Cost function for this model
        omCF_local <- function(B){
            adamElemsE <- adam_filler(B,
                                      etsModelE, EtypeE, TtypeE, StypeE,
                                      modelIsTrendyE, modelIsSeasonalE,
                                      componentsNumberETSE, componentsNumberETSNonSeasonalE,
                                      componentsNumberETSSeasonalE, componentsNumberARIMAE,
                                      lagsE, lagsModelE, lagsModelMaxE,
                                      adamCreatedE$matVt, adamCreatedE$matWt,
                                      adamCreatedE$matF, adamCreatedE$vecG,
                                      persistenceEstimateE, persistenceLevelEstimateE,
                                      persistenceTrendEstimateE, persistenceSeasonalEstimateE,
                                      persistenceXregEstimateE, phiEstimateE,
                                      initialTypeE, initialEstimateE,
                                      initialLevelEstimateE, initialTrendEstimateE,
                                      initialSeasonalEstimateE, initialArimaEstimate,
                                      initialXregEstimateE,
                                      arimaModelE, arEstimate, maEstimate,
                                      arOrders, iOrders, maOrders,
                                      arRequiredE, maRequiredE, armaParametersE,
                                      nonZeroARI,
                                      nonZeroMA,
                                      adamCreatedE$arimaPolynomials,
                                      xregModelE, xregNumberE,
                                      xregParametersMissingE, xregParametersIncludedE,
                                      xregParametersEstimatedE, xregParametersPersistenceE,
                                      constantEstimateE,
                                      adamCppE,
                                      constantRequiredE, initialArimaNumber);
            penaltyE <- adam_bounds_checker(adamElemsE, adamElemsE$arimaPolynomials,
                                            boundsE,
                                            etsModelE, modelIsTrendyE, modelIsSeasonalE,
                                            componentsNumberETSE, componentsNumberETSNonSeasonalE,
                                            componentsNumberETSSeasonalE,
                                            arimaModelE, arEstimate, maEstimate,
                                            xregModelE, regressorsE, xregNumberE,
                                            componentsNumberARIMAE,
                                            lagsModelAllE, obsInSampleE,
                                            arPolynomialMatrixE, maPolynomialMatrixE,
                                            phiEstimateE);
            if(penaltyE != 0){
                return(penaltyE);
            }
            profilesRecentTableE[] <- adamElemsE$matVt[, 1:lagsModelMaxE];
            adamFittedE <- adamCppE$fit(adamElemsE$matVt, adamElemsE$matWt,
                                        adamElemsE$matF, adamElemsE$vecG,
                                        indexLookupTableE, profilesRecentTableE,
                                        as.numeric(otE), as.numeric(otE),
                                        any(initialTypeE == c("complete","backcasting")),
                                        nIterations, refineHead, occurrenceChar);
            p <- pmin(pmax(adamFittedE$fitted, 1e-10), 1 - 1e-10);
            if(lossE == "likelihood"){
                return(-(sum(log(p[otLogicalE])) + sum(log(1 - p[!otLogicalE]))));
            }
            return(mean((as.numeric(otE) - p)^2));
        }

        maxevalE <- if(is.null(ellipsis$maxeval)) length(BE_used)*40 else ellipsis$maxeval;
        resE <- suppressWarnings(nloptr(BE_used, omCF_local, lb=lbE, ub=ubE,
                                        opts=list(algorithm=algorithm, xtol_rel=xtol_rel,
                                                  maxeval=maxevalE,
                                                  print_level=print_level)));

        if(is.infinite(resE$objective) || resE$objective == 1e+300){
            BE_used[] <- BValuesE$B;
            resE <- suppressWarnings(nloptr(BE_used, omCF_local, lb=lbE, ub=ubE,
                                            opts=list(algorithm=algorithm, xtol_rel=xtol_rel,
                                                      maxeval=maxevalE,
                                                      print_level=print_level)));
        }

        BE_used <- resE$solution;
        CFValueE <- resE$objective;

        # Count parameters
        nParamEstimatedE <- (etsModelE*(persistenceLevelEstimateE +
                                            modelIsTrendyE*persistenceTrendEstimateE +
                                            modelIsSeasonalE*sum(persistenceSeasonalEstimateE) +
                                            phiEstimateE) +
                                 xregModelE*persistenceXregEstimateE*
                                 max(xregParametersPersistenceE) +
                                 arimaModelE*(arEstimate*sum(arOrders)+
                                              maEstimate*sum(maOrders)) +
                                 etsModelE*all(initialTypeE!=c("complete","backcasting"))*
                                 (initialLevelEstimateE +
                                      modelIsTrendyE*initialTrendEstimateE +
                                      modelIsSeasonalE*sum(initialSeasonalEstimateE*
                                                           (lagsModelSeasonalE-1))) +
                                 all(initialTypeE!=c("complete","backcasting"))*
                                 arimaModelE*initialArimaNumber*initialArimaEstimate +
                                 xregModelE*initialXregEstimateE*
                                 sum(xregParametersEstimatedE) +
                                 constantEstimateE);
        logLikE <- -CFValueE;

        return(list(B=BE_used, CFValue=CFValueE, nParamEstimated=nParamEstimatedE,
                    logLikADAMValue=logLikE,
                    xregModel=xregModelE, xregData=xregDataE, xregNumber=xregNumberE,
                    xregNames=xregNamesE, xregModelInitials=xregModelInitialsE,
                    formula=formulaE,
                    initialXregEstimate=initialXregEstimateE,
                    persistenceXregEstimate=persistenceXregEstimateE,
                    xregParametersMissing=xregParametersMissingE,
                    xregParametersIncluded=xregParametersIncludedE,
                    xregParametersEstimated=xregParametersEstimatedE,
                    xregParametersPersistence=xregParametersPersistenceE,
                    arimaPolynomials=adamCreatedE$arimaPolynomials,
                    res=resE, adamCpp=adamCppE,
                    etsModel=etsModelE, Etype=EtypeE, Ttype=TtypeE, Stype=StypeE,
                    arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
                    modelIsTrendy=modelIsTrendyE, modelIsSeasonal=modelIsSeasonalE));
    }

    #### Model selection or combination ####
    # Shared env so omEstimatorWrapper can communicate nP to icFunction per model
    .icEnv <- new.env(parent=emptyenv());
    .icEnv$nP <- 0L;

    icFunction <- function(ll){
        nP <- .icEnv$nP;
        switch(ic,
            "AICc" = -2*ll + 2*nP + 2*nP*(nP+1)/max(obsInSample-nP-1, 1),
            "AIC"  = -2*ll + 2*nP,
            "BIC"  = -2*ll + log(obsInSample)*nP,
            "BICc" = -2*ll + log(obsInSample)*nP + nP*(nP+1)/max(obsInSample-nP-1, 1)
        );
    };
    icFunctionWrap <- function(ll, nP, obsIS=obsInSample){
        .icEnv$nP <- nP;
        icFunction(ll);
    };

    if(modelDo %in% c("select","combine")){
        # Wrap omEstimator to match adam_selector's expected signature
        omEstimatorWrapper <- function(etsModelE, EtypeE, TtypeE, StypeE, lagsE,
                                        lagsModelSeasonalE, lagsModelARIMAE,
                                        obsStatesE, obsInSampleE,
                                        yInSampleE, persistenceE, persistenceEstimateE,
                                        persistenceLevelE, persistenceLevelEstimateE,
                                        persistenceTrendE, persistenceTrendEstimateE,
                                        persistenceSeasonalE, persistenceSeasonalEstimateE,
                                        persistenceXregE, persistenceXregEstimateE,
                                        persistenceXregProvidedE,
                                        phiE, phiEstimateE,
                                        initialTypeE, initialLevelE, initialTrendE,
                                        initialSeasonalE, initialArimaE, initialEstimateE,
                                        initialLevelEstimateE, initialTrendEstimateE,
                                        initialSeasonalEstimateE, initialArimaEstimate,
                                        initialXregEstimateE, initialXregProvidedE,
                                        arimaModelE, arRequiredE, iRequiredE, maRequiredE,
                                        armaParametersE,
                                        componentsNumberARIMAE, componentsNamesARIMAE,
                                        formulaE, xregModelE, xregModelInitialsE, xregDataE,
                                        xregNumberE, xregNamesE, regressorsE,
                                        xregParametersMissingE, xregParametersIncludedE,
                                        xregParametersEstimatedE, xregParametersPersistenceE,
                                        constantRequiredE, constantEstimateE, constantValueE,
                                        constantNameE,
                                        otE, otLogicalE, occurrenceModelE, pFittedE,
                                        boundsE, lossE, lossFunctionE, distributionE,
                                        horizonE, multistepsE, otherE, otherParameterEstimateE,
                                        lambdaE, BE){
            res <- omEstimator(etsModelE, EtypeE, TtypeE, StypeE, lagsE,
                                lagsModelSeasonalE, lagsModelARIMAE,
                                obsStatesE, obsInSampleE,
                                yInSampleE, persistenceE, persistenceEstimateE,
                                persistenceLevelE, persistenceLevelEstimateE,
                                persistenceTrendE, persistenceTrendEstimateE,
                                persistenceSeasonalE, persistenceSeasonalEstimateE,
                                persistenceXregE, persistenceXregEstimateE,
                                persistenceXregProvidedE,
                                phiE, phiEstimateE,
                                initialTypeE, initialLevelE, initialTrendE,
                                initialSeasonalE, initialArimaE, initialEstimateE,
                                initialLevelEstimateE, initialTrendEstimateE,
                                initialSeasonalEstimateE, initialArimaEstimate,
                                initialXregEstimateE, initialXregProvidedE,
                                arimaModelE, arRequiredE, iRequiredE, maRequiredE,
                                armaParametersE,
                                componentsNumberARIMAE, componentsNamesARIMAE,
                                formulaE, xregModelE, xregModelInitialsE, xregDataE,
                                xregNumberE, xregNamesE, regressorsE,
                                xregParametersMissingE, xregParametersIncludedE,
                                xregParametersEstimatedE, xregParametersPersistenceE,
                                constantRequiredE, constantEstimateE, constantValueE,
                                constantNameE,
                                ot, otLogical, occurrenceModel, pFitted,
                                boundsE, loss, lossFunction, "dnorm",
                                horizon, multisteps, other, otherParameterEstimate,
                                lambda, BE);
            # Set shared nP then compute IC via icFunction
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

        # Re-run architect for best model
        adamArchitect <- adam_architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                          xregNumber, obsInSample, initialType,
                                          arimaModel, lagsModelARIMA, xregModel, constantRequired,
                                          componentsNumberARIMA,
                                          obsAll, yIndexAll, yClasses, adamETS);
        list2env(adamArchitect, environment());
    }
    else{
        # Single fixed model: just call omEstimator
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

    #### Final pass: get fitted values and states ####
    adamCreatedFinal <- adam_creator(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                     lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                                     profilesRecentTable, FALSE,
                                     obsStates, obsInSample,
                                     obsInSample + lagsModelMax,
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

    pFittedFinal <- pmin(pmax(adamFittedFinal$fitted, 1e-10), 1 - 1e-10);
    logLikValue <- sum(log(pFittedFinal[otLogical])) + sum(log(1 - pFittedFinal[!otLogical]));

    #### Wrap as ts ####
    # matVt from adam_creator has proper row names (all components: ETS + ARIMA + xreg + constant)
    allComponentNames <- rownames(adamCreatedFinal$matVt);
    statesRaw <- adamFittedFinal$states[, (lagsModelMax+1):ncol(adamFittedFinal$states), drop=FALSE];
    if(!is.null(allComponentNames)){
        rownames(statesRaw) <- allComponentNames;
    }
    if(any(yClasses == "ts")){
        fittedTS <- ts(pFittedFinal, start=yStart, frequency=yFrequency);
        residualsTS <- ts(as.numeric(ot) - pFittedFinal, start=yStart, frequency=yFrequency);
        statesTS <- ts(t(statesRaw), start=yStart, frequency=yFrequency);
    }
    else{
        fittedTS <- pFittedFinal;
        residualsTS <- as.numeric(ot) - pFittedFinal;
        statesTS <- t(statesRaw);
    }

    #### Parameter counts ####
    nParamEstimated <- (etsModel*(persistenceLevelEstimate +
                                      modelIsTrendy*persistenceTrendEstimate +
                                      modelIsSeasonal*sum(persistenceSeasonalEstimate) +
                                      phiEstimate) +
                            xregModel*persistenceXregEstimate*max(xregParametersPersistence) +
                            arimaModel*(arEstimate*sum(arOrders)+maEstimate*sum(maOrders)) +
                            etsModel*all(initialType!=c("complete","backcasting"))*
                            (initialLevelEstimate +
                                 modelIsTrendy*initialTrendEstimate +
                                 modelIsSeasonal*sum(initialSeasonalEstimate*(lagsModelSeasonal-1))) +
                            all(initialType!=c("complete","backcasting"))*
                            arimaModel*initialArimaNumber*initialArimaEstimate +
                            xregModel*initialXregEstimate*sum(xregParametersEstimated) +
                            constantEstimate);
    parametersNumber[1,1] <- nParamEstimated;
    parametersNumber[1,5] <- sum(parametersNumber[1,1:4]);
    parametersNumber[2,5] <- sum(parametersNumber[2,1:4]);

    #### Model name ####
    modelName <- adam_model_name(etsModel, model, xregModel, arimaModel,
                                  arOrders, iOrders, maOrders, lags,
                                  regressors, constantRequired, constantName,
                                  occurrence, componentsNumberETSSeasonal);

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
    } else{
        armaParametersList <- NULL;
    }

    #### Persistence vector ####
    vecGFinal <- adamFilledFinal$vecG;
    persistenceVec <- vecGFinal[1:componentsNumberETS];

    #### Construct return object ####
    modelReturned <- list(
        model = modelName,
        timeElapsed = Sys.time() - startTime,
        fitted = fittedTS,
        residuals = residualsTS,
        forecast = if(h > 0) ts(rep(NA, h), start=yForecastStart, frequency=yFrequency) else ts(NA),
        states = statesTS,
        persistence = persistenceVec,
        phi = if(phiEstimate) B[names(B)=="phi"] else phi,
        transition = adamFilledFinal$matF,
        measurement = adamFilledFinal$matWt,
        initial = list(level=adamFittedFinal$states[1, lagsModelMax]),
        initialType = initialType,
        lags = lags,
        lagsAll = lagsModelAll,
        orders = list(ar=arOrders, i=iOrders, ma=maOrders),
        arma = armaParametersList,
        loss = loss,
        lossValue = -logLikValue,
        lossFunction = lossFunction,
        logLik = logLikValue,
        nParam = parametersNumber,
        scale = NA,
        iprob = mean(as.numeric(ot)),
        distribution = "dbinom",
        occurrence = NULL,
        occurrenceType = occurrence,
        B = B,
        bounds = bounds,
        formula = formula,
        profile = profilesRecentTable,
        data = yInSample,
        y = ot[1:obsInSample],
        call = cl,
        regressors = regressors,
        adamCpp = adamCpp
    );

    if(holdout){
        modelReturned$holdout <- yHoldout;
        modelReturned$accuracy <- measures(yHoldout, modelReturned$forecast, ot[1:obsInSample]);
    }

    class(modelReturned) <- c("om","adam","smooth");
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
    if(object$occurrenceType == "fixed"){
        iprob <- object$iprob;
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
    occurrence <- object$occurrenceType;

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

    # Replace NaN (from exp(large)) with 1
    if(occurrence %in% c("odds-ratio","inverse-odds-ratio") && Etype == "A"){
        fc$mean[is.nan(fc$mean)] <- 1;
    }

    if(interval != "none" && !is.null(fc$lower)){
        fc$lower[] <- .link(fc$lower);
        fc$upper[] <- .link(fc$upper);
        if(occurrence %in% c("odds-ratio","inverse-odds-ratio") && Etype == "A"){
            fc$lower[is.nan(fc$lower)] <- 1;
            fc$upper[is.nan(fc$upper)] <- 1;
        }
    }

    return(fc);
}

#' @export
actuals.om <- function(object, ...){
    return(object$y);
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
