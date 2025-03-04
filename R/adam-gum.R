utils::globalVariables(c("xregData","xregModel","xregNumber","initialXregEstimate","xregNames",
                         "otLogical","yFrequency","yIndex",
                         "persistenceXreg","yHoldout","distribution"));

#' Generalised Univariate Model
#'
#' Function constructs Generalised Univariate Model, estimating matrices F, w,
#' vector g and initial parameters.
#'
#' The function estimates the Single Source of Error state space model of the
#' following type:
#'
#' \deqn{y_{t} = w_t' v_{t-l} + \epsilon_{t}}
#'
#' \deqn{v_{t} = F v_{t-l} + g_{t} \epsilon_{t}}
#'
#' where \eqn{v_{t}} is the state vector (defined using \code{orders}) and
#' \eqn{l} is the vector of \code{lags}, \eqn{w_t} is the \code{measurement}
#' vector (which includes fixed elements and explanatory variables),
#' \eqn{F} is the \code{transition} matrix, \eqn{g_t} is the \code{persistence}
#' vector (includes explanatory variables as well if provided), finally,
#' \eqn{\epsilon_{t}} is the error term.
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("gum","smooth")}
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ADAMDataFormulaRegLossSilentHHoldout
#'
#' @template smoothRef
#' @template ssGeneralRef
#'
#' @param orders Order of the model. Specified as vector of number of states
#' with different lags. For example, \code{orders=c(1,1)} means that there are
#' two states: one of the first lag type, the second of the second type.
#' In case of \code{auto.gum()}, this parameters is the value of the max order
#' to check.
#' @param lags Defines lags for the corresponding orders. If, for example,
#' \code{orders=c(1,1)} and lags are defined as \code{lags=c(1,12)}, then the
#' model will have two states: the first will have lag 1 and the second will
#' have lag 12. The length of \code{lags} must correspond to the length of
#' \code{orders}. In case of the \code{auto.gum()}, the value of the maximum
#' lag to check. This should usually be a maximum frequency of the data.
#' @param type Type of model. Can either be \code{"additive"} or
#' \code{"multiplicative"}. The latter means that the GUM is fitted on
#' log-transformed data. In case of \code{auto.gum()}, can also be \code{"select"},
#' implying automatic selection of the type.
#' @param initial Can be either character or a vector of initial states. If it
#' is character, then it can be \code{"optimal"}, meaning that the initial
#' states are optimised, \code{"backcasting"}, meaning that the initials are
#' produced using backcasting procedure (still estimating initials for explanatory
#' variables), or \code{"complete"}, meaning backcasting for all states.
#' @param persistence Persistence vector \eqn{g}, containing smoothing
#' parameters. If \code{NULL}, then estimated.
#' @param transition Transition matrix \eqn{F}. Can be provided as a vector.
#' Matrix will be formed using the default \code{matrix(transition,nc,nc)},
#' where \code{nc} is the number of components in the state vector. If
#' \code{NULL}, then estimated.
#' @param measurement Measurement vector \eqn{w}. If \code{NULL}, then
#' estimated.
#' @param bounds The type of bounds for the persistence to use in the model
#' estimation. Can be either \code{admissible} - guaranteeing the stability of the
#' model, or \code{none} - no restrictions (potentially dangerous).
#' @param model A previously estimated GUM model, if provided, the function
#' will not estimate anything and will use all its parameters.
#' @param ...  Other non-documented parameters. See \link[smooth]{adam} for
#' details
#'
#' @return Object of class "adam" is returned with similar elements to the
#' \link[smooth]{adam} function.
#'
#' @seealso \code{\link[smooth]{adam}, \link[smooth]{es}, \link[smooth]{ces}}
#'
#' @examples
#' gum(BJsales, h=8, holdout=TRUE)
#'
#' \donttest{ourModel <- gum(rnorm(118,100,3), orders=c(2,1), lags=c(1,4), h=18, holdout=TRUE)}
#'
#' # Redo previous model on a new data and produce prediction interval
#' \donttest{gum(rnorm(118,100,3), model=ourModel, h=18)}
#'
#' # Produce something crazy with optimal initials (not recommended)
#' \donttest{gum(rnorm(118,100,3), orders=c(1,1,1), lags=c(1,3,5), h=18, holdout=TRUE, initial="o")}
#'
#' # Simpler model estimated using trace forecast error loss function and its analytical analogue
#' \donttest{gum(rnorm(118,100,3), orders=c(1), lags=c(1), h=18, holdout=TRUE, bounds="n", loss="TMSE")}
#'
#' @rdname gum
#' @export
gum <- function(data, orders=c(1,1), lags=c(1,frequency(data)), type=c("additive","multiplicative"),
                formula=NULL, regressors=c("use","select","adapt","integrate"),
                initial=c("optimal","backcasting","complete"),
                persistence=NULL, transition=NULL, measurement=rep(1,sum(orders)),
                loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                h=0, holdout=FALSE, bounds=c("admissible","none"), silent=TRUE,
                model=NULL, ...){
# General Univariate Model function. Paper to follow... at some point... maybe.
#
#    Copyright (C) 2016 - Inf Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();
    cl <- match.call();
    # Record the parental environment. Needed for optimal initialisation
    env <- parent.frame();

    ellipsis <- list(...);

    # Check seasonality and loss
    type <- match.arg(type);
    loss <- match.arg(loss);

    # paste0() is needed in order to get rid of potential issues with names
    yName <- paste0(deparse(substitute(data)),collapse="");

    # Assume that the model is not provided
    profilesRecentProvided <- FALSE;
    profilesRecentTable <- NULL;

    # If a previous model provided as a model, write down the variables
    if(!is.null(model)){
        if(is.null(model$model)){
            stop("The provided model is not GUM.",call.=FALSE);
        }
        else if(smoothType(model)!="GUM"){
            stop("The provided model is not GUM.",call.=FALSE);
        }
        # This needs to be fixed to align properly in case of various seasonals
        profilesRecentInitial <- profilesRecentTable <- model$profileInitial;
        profilesRecentProvided[] <- TRUE;
        # This is needed to save initials and to avoid the standard checks
        initialValueProvided <- model$initial;
        initialOriginal <- initial <- model$initialType;
        seasonality <- model$seasonality;
        # matVt <- t(model$states);
        measurement <- model$measurement;
        transition <- model$transition;
        persistenceOriginal <- model$persistence;
        ellipsis$B <- coef(model);
        lags <- lags(model);
        orders <- orders(model);

        model <- model$model;
        model <- NULL;
        modelDo <- modelDoOriginal <- "use";
    }
    else{
        modelDo <- modelDoOriginal <- "estimate";
        initialOriginal <- initial;
        initialValueProvided <- NULL;
        persistenceOriginal <- persistence;
    }

    # If this is Mcomp data, then take the frequency from it
    if(any(class(data)=="Mdata") && all(lags %in% c(1,frequency(data)))){
        lags <- c(1,frequency(data$x));
    }

    orders <- orders[order(lags)];
    lags <- sort(lags);
    # Remove redundant lags (if present)
    lags <- lags[!is.na(orders)];
    # Remove NAs (if lags are longer than orders)
    orders <- orders[!is.na(orders)];

    # GUM is "checked" as ARIMA
    model <- "NNN";
    ordersOriginal <- orders;
    lagsOriginal <- lags;

    # Specific checks for orders and lags
    if(any(is.complex(c(orders,lags)))){
        stop("Complex values? Right! Come on! Be real!", call.=FALSE);
    }
    if(any(c(orders)<0)){
        stop("Funny guy! How am I gonna construct a model with negative orders?", call.=FALSE);
    }
    if(any(c(lags)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?", call.=FALSE);
    }

    # If there are zero lags, drop them
    if(any(lags==0)){
        orders <- orders[lags!=0];
        lags <- lags[lags!=0];
    }
    # If zeroes are defined for some orders, drop them.
    if(any(orders==0)){
        lags <- lags[orders!=0];
        orders <- orders[orders!=0];
    }

    # Get rid of duplicates in lags
    if(length(unique(lags))!=length(lags)){
        lagsNew <- unique(lags);
        ordersNew <- lagsNew;
        for(i in 1:length(lagsNew)){
            ordersNew[i] <- max(orders[which(lags==lagsNew[i])]);
        }
        orders <- ordersNew;
        lags <- lagsNew;
    }

    # Check whether the multiplicative model is applicable
    if(type=="multiplicative"){
        if(any(yInSample<=0)){
            warning("Multiplicative model can only be used on positive data. Switching to the additive one.",
                    call.=FALSE);
            modelIsMultiplicative <- FALSE;
            type <- "additive";
        }
        else{
            yInSample <- log(yInSample);
            modelIsMultiplicative <- TRUE;
        }
    }
    else{
        modelIsMultiplicative <- FALSE;
    }

    # If initial was provided, trick parametersChecker
    if(!is.character(initial)){
        initialValueProvided <- initial;
        initial <- "optimal";
    }
    else{
        initial <- match.arg(initial);
    }

    # Hack parametersChecker if initial="integrate"
    regressorsIntegrate <- FALSE;
    regressors <- match.arg(regressors);
    if(regressors=="integrate"){
        regressorsIntegrate <- TRUE;
        regressors <- "adapt";
    }

    ##### Set environment for ssInput and make all the checks #####
    checkerReturn <- parametersChecker(data=data, model, lags, formulaToUse=formula,
                                       orders=list(ar=c(orders),i=c(0),ma=c(0),select=FALSE),
                                       constant=FALSE, arma=NULL,
                                       outliers="ignore", level=0.99,
                                       persistence=NULL, phi=NULL, initial,
                                       distribution="dnorm", loss, h, holdout, occurrence="none",
                                       # This is not needed by the gum() function
                                       ic="AICc", bounds=bounds[1],
                                       regressors=regressors, yName=yName,
                                       silent, modelDo, ParentEnvironment=environment(), ellipsis, fast=FALSE);

    # This is the variable needed for the C++ code to determine whether the head of data needs to be
    # refined. GUM doesn't need that.
    refineHead <- TRUE;

    ##### Elements of GUM #####
    filler <- function(B, vt, matF, vecG, matWt){

        nCoefficients <- 0;
        if(persistenceEstimate){
            vecG[1:componentsNumberAll,] <- B[nCoefficients+(1:componentsNumberAll)];
            nCoefficients[] <- nCoefficients + componentsNumberAll;
        }

        if(transitionEstimate){
            matF[1:componentsNumberAll,1:componentsNumberAll] <- B[nCoefficients+(1:(componentsNumberAll^2))];
            nCoefficients[] <- nCoefficients + componentsNumberAll^2;
        }

        if(measurementEstimate){
            matWt[,1:componentsNumber] <- B[nCoefficients+(1:componentsNumber)];
            nCoefficients[] <- nCoefficients + componentsNumber;
        }

        if(initialType=="optimal"){
            for(i in 1:componentsNumber){
                vt[i,1:lagsModelMax] <- rep(B[nCoefficients+(1:lagsModel[i])], lagsModelMax)[1:lagsModelMax];
                nCoefficients[] <- nCoefficients + lagsModel[i];
            }
        }

        # In case of backcasting, we still estimate initials of xreg
        if(xregModel && initialXregEstimate && initialType!="complete"){
            vt[componentsNumber+c(1:xregNumber),1:lagsModelMax] <- B[nCoefficients+(1:xregNumber)];
        }

        return(list(matWt=matWt,matF=matF,vecG=vecG,vt=vt));
    }

    ##### Function returns scale parameter for the provided parameters #####
    scaler <- function(errors, obsInSample){
        return(sqrt(sum(errors^2)/obsInSample));
    }

    ##### Cost function for CES #####
    CF <- function(B, matVt, matF, vecG, matWt){
        # Obtain the elements of CES
        elements <- filler(B, matVt[,1:lagsModelMax,drop=FALSE], matF, vecG, matWt);

        if(xregModel){
            # We drop the X parts from matrices
            indices <- c(1:componentsNumber)
            eigenValues <- abs(eigen(elements$matF[indices,indices,drop=FALSE] -
                                         elements$vecG[indices,,drop=FALSE] %*%
                                         matWt[obsInSample,indices,drop=FALSE],
                                     symmetric=FALSE, only.values=TRUE)$values);
        }
        else{
            eigenValues <- abs(eigen(elements$matF -
                                         elements$vecG %*% matWt[obsInSample,,drop=FALSE],
                                     symmetric=FALSE, only.values=TRUE)$values);
        }
        if(any(eigenValues>1+1E-50)){
            return(1E+100*max(eigenValues));
        }

        # Write down the initials in the recent profile
        matVt[,1:lagsModelMax] <- profilesRecentTable[] <- elements$vt;

        adamFitted <- adamFitterWrap(matVt, elements$matWt, elements$matF, elements$vecG,
                                     lagsModelAll, indexLookupTable, profilesRecentTable,
                                     Etype, Ttype, Stype, componentsNumberETS, componentsNumberETSSeasonal,
                                     componentsNumberARIMA, xregNumber, FALSE,
                                     yInSample, ot, any(initialType==c("complete","backcasting")),
                                     nIterations, refineHead);

        if(!multisteps){
            if(loss=="likelihood"){
                # Scale for different functions
                scale <- scaler(adamFitted$errors[otLogical], obsInSample);

                # Calculate the likelihood
                CFValue <- -sum(dnorm(x=yInSample[otLogical],
                                      mean=adamFitted$yFitted[otLogical],
                                      sd=scale, log=TRUE));
            }
            else if(loss=="MSE"){
                CFValue <- sum(adamFitted$errors^2)/obsInSample;
            }
            else if(loss=="MAE"){
                CFValue <- sum(abs(adamFitted$errors))/obsInSample;
            }
            else if(loss=="HAM"){
                CFValue <- sum(sqrt(abs(adamFitted$errors)))/obsInSample;
            }
            else if(loss=="custom"){
                CFValue <- lossFunction(actual=yInSample,fitted=adamFitted$yFitted,B=B);
            }
        }
        else{
            # Call for the Rcpp function to produce a matrix of multistep errors
            adamErrors <- adamErrorerWrap(adamFitted$matVt, elements$matWt, elements$matF,
                                          lagsModelAll, indexLookupTable, profilesRecentTable,
                                          Etype, Ttype, Stype,
                                          componentsNumberETS, componentsNumberETSSeasonal,
                                          componentsNumberARIMA, xregNumber, constantRequired, h,
                                          yInSample, ot);

            # Not done yet: "aMSEh","aTMSE","aGTMSE","aMSCE","aGPL"
            CFValue <- switch(loss,
                              "MSEh"=sum(adamErrors[,h]^2)/(obsInSample-h),
                              "TMSE"=sum(colSums(adamErrors^2)/(obsInSample-h)),
                              "GTMSE"=sum(log(colSums(adamErrors^2)/(obsInSample-h))),
                              "MSCE"=sum(rowSums(adamErrors)^2)/(obsInSample-h),
                              "MAEh"=sum(abs(adamErrors[,h]))/(obsInSample-h),
                              "TMAE"=sum(colSums(abs(adamErrors))/(obsInSample-h)),
                              "GTMAE"=sum(log(colSums(abs(adamErrors))/(obsInSample-h))),
                              "MACE"=sum(abs(rowSums(adamErrors)))/(obsInSample-h),
                              "HAMh"=sum(sqrt(abs(adamErrors[,h])))/(obsInSample-h),
                              "THAM"=sum(colSums(sqrt(abs(adamErrors)))/(obsInSample-h)),
                              "GTHAM"=sum(log(colSums(sqrt(abs(adamErrors)))/(obsInSample-h))),
                              "CHAM"=sum(sqrt(abs(rowSums(adamErrors))))/(obsInSample-h),
                              "GPL"=log(det(t(adamErrors) %*% adamErrors/(obsInSample-h))),
                              0);
        }

        if(is.na(CFValue) || is.nan(CFValue)){
            CFValue[] <- 1e+300;
        }

        return(CFValue);
    }

    #### Likelihood function ####
    logLikFunction <- function(B, matVt, matF, vecG, matWt){
        return(-CF(B, matVt=matVt, matF=matF, vecG=vecG, matWt=matWt));
    }

    initialValue <- initialValueProvided;
    initial <- initialOriginal;
    persistence <- persistenceOriginal;

    orders <- ordersOriginal;
    lags <- lagsOriginal;

    # Fixes to get correct number of components and lags
    # lagsModel is the lags of GUM
    lagsModel <- matrix(rep(lags, times=orders),ncol=1);
    # lagsModelAll is all the lags, GUM+X
    if(xregModel){
        lagsModelAll <- c(lagsModel, rep(1, xregNumber));
    }
    else{
        lagsModelAll <- lagsModel;
    }
    lagsModelMax <- max(lagsModelAll);
    obsStates[] <- obsInSample + lagsModelMax;

    # The reversed lags to fill in values in the state vector
    # lagsModelRev <- lagsModelMax - lagsModel + 1;
    componentsNumberARIMA <- componentsNumber <- sum(orders);

    # componentsNumberAll is used to fill in all matrices
    componentsNumberAll <- componentsNumber
    if(regressorsIntegrate){
        regressors <- "integrate";
        componentsNumberAll <- componentsNumber+xregNumber;
    }

    matF <- diag(componentsNumber+xregNumber);
    vecG <- matrix(0,componentsNumber+xregNumber,1);
    if(xregModel){
        rownames(vecG) <- c(paste0("g",1:componentsNumber),
                            paste0("delta",1:xregNumber));
    }
    else{
        rownames(vecG) <- paste0("g",1:componentsNumber);
    }
    matWt <- matrix(1, obsInSample, componentsNumber+xregNumber);
    matVt <- matrix(0, componentsNumber+xregNumber, obsStates,
                    dimnames=list(c(paste0("Component ",1:(componentsNumber)), xregNames), NULL));

    # Fixes for what to estimate
    persistenceEstimate <- is.null(persistence);
    transitionEstimate <- is.null(transition);
    measurementEstimate <- is.null(measurement);
    initialEstimate <- is.null(initialValueProvided);

    # Provided measurement should be just a vector for the dynamic elements
    if(!measurementEstimate){
        matWt[,1:componentsNumber] <- matrix(measurement, obsInSample, componentsNumber, byrow=TRUE);
    }
    if(!transitionEstimate){
        matF[1:componentsNumberAll,1:componentsNumberAll] <- transition;
    }
    if(!persistenceEstimate){
        vecG[1:componentsNumberAll,] <- persistence;
    }
    if(!initialEstimate){
        matVt[1:componentsNumber,1:lagsModelMax] <- initialValue$endogenous;
        if(xregModel){
            matVt[componentsNumber+1:xregNumber,1:lagsModelMax] <- initialValue$xreg;
        }
        initialType <- "provided";
    }
    else{
        if(initialType!="complete"){
            slope <- (cov(yInSample[1:min(max(12,lagsModelMax),obsInSample),],c(1:min(max(12,lagsModelMax),obsInSample)))/
                          var(c(1:min(max(12,lagsModelMax),obsInSample))));
            intercept <- (sum(yInSample[1:min(max(12,lagsModelMax),obsInSample),])/min(max(12,lagsModelMax),obsInSample) -
                              slope * (sum(c(1:min(max(12,lagsModelMax),obsInSample)))/
                                           min(max(12,lagsModelMax),obsInSample) - 1));

            vtvalues <- vector("numeric", orders %*% lags);
            nCoefficients <- 0;
            if(any(lags==1) && length(orders[lags==1])>=1){
                vtvalues[nCoefficients+1] <- intercept;
                nCoefficients[] <- nCoefficients + 1;
            }
            if(any(lags==1) && length(orders[lags==1])>1){
                vtvalues[nCoefficients+1] <- slope;
                nCoefficients[] <- nCoefficients + 1;
            }
            if((orders %*% lags)>2){
                vtvalues[nCoefficients + 1:(orders %*% lags - nCoefficients)] <- yInSample[1:(orders %*% lags - nCoefficients),];
            }

            nCoefficients[] <- 0;
            for(i in 1:componentsNumber){
                matVt[i,1:lagsModel[i]] <- vtvalues[nCoefficients+(1:lagsModel[i])];
                nCoefficients[] <- nCoefficients + lagsModel[i];
            }
        }
    }

    # Add parameters for the X
    if(xregModel){
        matWt[,componentsNumber+c(1:xregNumber)] <- xregData[1:obsInSample,];
        if(initialXregEstimate && initialType!="complete"){
            matVt[componentsNumber+c(1:xregNumber),1] <- xregModelInitials[[1]][[1]];
        }
    }

    # A hack in case all the parameters were provided
    if(modelDoOriginal=="use"){
        modelDo <- modelDoOriginal;
    }

    ##### Pre-set yFitted, yForecast, errors and basic parameters #####
    # Prepare fitted and error with ts / zoo
    if(any(yClasses=="ts")){
        yFitted <- ts(rep(NA,obsInSample), start=yStart, frequency=yFrequency);
        errors <- ts(rep(NA,obsInSample), start=yStart, frequency=yFrequency);
    }
    else{
        yFitted <- zoo(rep(NA,obsInSample), order.by=yInSampleIndex);
        errors <- zoo(rep(NA,obsInSample), order.by=yInSampleIndex);
    }
    yForecast <- rep(NA, h);

    # Values for occurrence. No longer supported in ces()
    parametersNumber[1,3] <- parametersNumber[2,3] <- 0;
    # Xreg parameters
    parametersNumber[1,2] <- xregNumber + sum(persistenceXreg);
    # Scale value
    parametersNumber[1,4] <- 1;

    #### If we need to estimate the model ####
    if(modelDo=="estimate"){
        # Create ADAM profiles for correct treatment of seasonality
        adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                           lags=lags, yIndex=yIndexAll, yClasses=yClasses);
        profilesRecentTable <- adamProfiles$recent;
        indexLookupTable <- adamProfiles$lookup;

        if(is.null(B)){
            B <- vector("numeric", persistenceEstimate*componentsNumberAll +
                            transitionEstimate*componentsNumberAll^2 +
                            measurementEstimate*componentsNumber +
                            initialEstimate*(initialType=="optimal")*sum(orders %*% lags) +
                            xregNumber*initialXregEstimate*(initialType!="complete"));
            names(B) <- c(paste0("g",1:componentsNumberAll)[persistenceEstimate*(1:componentsNumberAll)],
                          paste0("F",paste0(rep(1:componentsNumberAll,each=componentsNumberAll),
                                            rep(1:componentsNumberAll,times=componentsNumberAll))
                          )[transitionEstimate*(1:(componentsNumberAll^2))],
                          paste0("w",1:componentsNumber)[measurementEstimate*(1:componentsNumber)],
                          paste0("vt",1:sum(orders %*% lags))[initialEstimate*(initialType=="optimal")*(1:sum(orders %*% lags))],
                          xregNames[(1:xregNumber)*initialXregEstimate*(initialType!="complete")]);

            nCoefficients <- 0;
            if(persistenceEstimate){
                B[nCoefficients+1:componentsNumberAll] <- rep(0.1, componentsNumberAll);
                nCoefficients[] <- nCoefficients + componentsNumberAll;
            }

            if(transitionEstimate){
                B[nCoefficients+1:(componentsNumberAll^2)] <- as.numeric(matF[1:componentsNumberAll,1:componentsNumberAll]);
                nCoefficients[] <- nCoefficients + componentsNumberAll^2;
            }

            if(measurementEstimate){
                B[nCoefficients+1:componentsNumber] <- rep(1, componentsNumber);
                nCoefficients[] <- nCoefficients + componentsNumber;
            }

            # In case of optimal, get some initials from backcasting
            if(initialEstimate && (initialType=="optimal")){
                clNew <- cl;
                # If environment is provided, use it
                if(!is.null(ellipsis$environment)){
                    env <- ellipsis$environment;
                }
                # Use complete backcasting
                clNew$initial <- "complete";
                # Shut things up
                clNew$silent <- TRUE;
                # Switch off regressors selection
                if(!is.null(clNew$regressors) && clNew$regressors=="select"){
                    clNew$regressors <- "use";
                }

                # Call for GUM with backcasting
                gumBack <- suppressWarnings(eval(clNew, envir=env));
                B[1:nCoefficients] <- gumBack$B;

                # B <- c(B, gumBack$initial$endogenous);
                for(i in 1:componentsNumber){
                    # B[nCoefficients+(1:lagsModel[i])] <- matVt[i,lagsModelRev[i]:lagsModelMax];
                    B[nCoefficients+(1:lagsModel[i])] <- gumBack$initial$endogenous[i,1:lagsModel[i]];
                    nCoefficients[] <- nCoefficients + lagsModel[i];
                }
            }

            if(xregModel && initialXregEstimate && initialType!="complete"){
                B[nCoefficients+1:xregNumber] <- matVt[-c(1:componentsNumber),lagsModelMax];
            }
        }


        # Print level defined
        print_level_hidden <- print_level;
        if(print_level==41){
            cat("Initial parameters:",B,"\n");
            print_level[] <- 0;
        }

        # maxeval based on what was provided
        maxevalUsed <- maxeval;
        if(is.null(maxeval)){
            maxevalUsed <- length(B) * 40;
            if(xregModel && initialType!="complete"){
                maxevalUsed[] <- length(B) * 100;
                maxevalUsed[] <- max(1000,maxevalUsed);
            }
        }

        res <- suppressWarnings(nloptr(B, CF,
                                       opts=list(algorithm=algorithm, xtol_rel=xtol_rel, xtol_abs=xtol_abs,
                                                 ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                                 maxeval=maxevalUsed, maxtime=maxtime, print_level=print_level),
                                       matVt=matVt, matF=matF, vecG=vecG, matWt=matWt));

        if(print_level_hidden>0){
            print(res);
        }

        B[] <- res$solution;
        CFValue <- res$objective;

        # Parameters estimated + variance
        nParamEstimated <- length(B) + (loss=="likelihood")*1;

        # Prepare for fitting
        elements <- filler(B, matVt[,1:lagsModelMax,drop=FALSE], matF, vecG, matWt);
        matF[] <- elements$matF;
        vecG[] <- elements$vecG;
        matVt[,1:lagsModelMax] <- elements$vt;
        matWt[] <- elements$matWt;

        # Write down the initials in the recent profile
        profilesRecentInitial <- profilesRecentTable[] <- matVt[,1:lagsModelMax,drop=FALSE];
        parametersNumber[1,1] <- length(B);
    }
    #### If we just use the provided values ####
    else{
        # Create index lookup table
        indexLookupTable <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                           lags=lags, yIndex=yIndexAll, yClasses=yClasses)$lookup;
        if(initialType=="optimal"){
            initialType <- "provided";
        }
        initialValue <- profilesRecentTable;
        initialXregEstimateOriginal <- initialXregEstimate;
        initialXregEstimate <- FALSE;

        CFValue <- CF(B, matVt, matF, vecG, matWt);
        res <- NULL;

        # Only variance is estimated
        nParamEstimated <- 1;

        initialType <- initialOriginal;
        initialXregEstimate <- initialXregEstimateOriginal;
    }

    #### Fisher Information ####
    if(FI){
        # Substitute values to get hessian
        if(any(substr(names(B),1,1)=="g")){
            persistenceEstimateOriginal <- persistenceEstimate;
            persistenceEstimate <- TRUE;
        }
        if(any(substr(names(B),1,1)=="F")){
            transitionEstimateOriginal <- transitionEstimate;
            transitionEstimate <- TRUE;
        }
        if(any(substr(names(B),1,1)=="w")){
            measurementEstimateOriginal <- measurementEstimate;
            measurementEstimate <- TRUE;
        }
        initialTypeOriginal <- initialType;
        initialType <- "optimal";
        if(!is.null(initialValueProvided$xreg) && initialOriginal!="complete"){
            initialXregEstimateOriginal <- initialXregEstimate;
            initialXregEstimate <- TRUE;
        }

        FI <- -hessian(logLikFunction, B, h=stepSize, matVt=matVt, matF=matF, vecG=vecG, matWt=matWt);
        colnames(FI) <- rownames(FI) <- names(B);

        if(any(substr(names(B),1,1)=="g")){
            persistenceEstimate <- persistenceEstimateOriginal;
        }
        if(any(substr(names(B),1,1)=="F")){
            transitionEstimate <- transitionEstimateOriginal;
        }
        if(any(substr(names(B),1,1)=="w")){
            measurementEstimate <- measurementEstimateOriginal;
        }
        initialType <- initialTypeOriginal;
        if(!is.null(initialValueProvided$xreg) && initialOriginal!="complete"){
            initialXregEstimate <- initialXregEstimateOriginal;
        }
    }
    else{
        FI <- NA;
    }

    # In case of likelihood, we typically have one more parameter to estimate - scale.
    logLikValue <- structure(logLikFunction(B, matVt=matVt, matF=matF, vecG=vecG, matWt=matWt),
                             nobs=obsInSample, df=nParamEstimated, class="logLik");

    adamFitted <- adamFitterWrap(matVt, matWt, matF, vecG,
                                 lagsModelAll, indexLookupTable, profilesRecentTable,
                                 Etype, Ttype, Stype, componentsNumberETS, componentsNumberETSSeasonal,
                                 componentsNumberARIMA, xregNumber, FALSE,
                                 yInSample, ot, any(initialType==c("complete","backcasting")),
                                 nIterations, refineHead);

    errors[] <- adamFitted$errors;
    yFitted[] <- adamFitted$yFitted;
    # Write down the recent profile for future use
    profilesRecentTable <- adamFitted$profile;
    matVt[] <- adamFitted$matVt;

    scale <- scaler(adamFitted$errors[otLogical], obsInSample);

    if(any(yClasses=="ts")){
        yForecast <- ts(rep(NA, max(1,h)), start=yForecastStart, frequency=yFrequency);
    }
    else{
        yForecast <- zoo(rep(NA, max(1,h)), order.by=yForecastIndex);
    }
    if(h>0){
        yForecast[] <- adamForecasterWrap(tail(matWt,h), matF,
                                          lagsModelAll,
                                          indexLookupTable[,lagsModelMax+obsInSample+c(1:h),drop=FALSE],
                                          profilesRecentTable,
                                          Etype, Ttype, Stype,
                                          componentsNumberETS, componentsNumberETSSeasonal,
                                          componentsNumberARIMA, xregNumber, FALSE,
                                          h);
    }
    else{
        yForecast[] <- NA;
    }


    ##### Do final check and make some preparations for output #####
    # Write down initials of states vector and exogenous
    if(initialType!="provided"){
        initialValue <- list(endogenous=matVt[1:componentsNumber,1:lagsModelMax,drop=FALSE]);
        # initialValue <- vector("list", 1*(seasonality!="simple") + 1*(seasonality!="none") + xregModel);
        # if(seasonality=="none"){
        #     names(initialValue) <- c("nonseasonal","xreg")[c(TRUE,xregModel)]
        #     initialValue$nonseasonal <- matVt[1:2,1];
        # }
        # else if(seasonality=="simple"){
        #     names(initialValue) <- c("seasonal","xreg")[c(TRUE,xregModel)]
        #     initialValue$seasonal <- matVt[1:2,1:lagsModelMax];
        # }
        # else{
        #     names(initialValue) <- c("nonseasonal","seasonal","xreg")[c(TRUE,TRUE,xregModel)]
        #     initialValue$nonseasonal <- matVt[1:2,1];
        #     initialValue$seasonal <- matVt[lagsModelAll!=1,1:lagsModelMax];
        # }

        # if(initialType=="optimal"){
        #     parametersNumber[1,1] <- (parametersNumber[1,1] + orders %*% lags);
        # }
    }
    if(xregModel){
        initialValue$xreg <- matVt[componentsNumber+1:xregNumber,1];
    }
    parametersNumber[1,5] <- sum(parametersNumber[1,])

    # Right down the smoothing parameters
    nCoefficients <- 0;

    modelname <- "GUM";
    if(xregModel){
        modelname[] <- paste0(modelname,"X");
    }
    modelname[] <- paste0(modelname,"(",paste(orders,"[",lags,"]",collapse=",",sep=""),")");

    if(all(occurrence!=c("n","none"))){
        modelname[] <- paste0("i",modelname);
    }

    parametersNumber[1,5] <- sum(parametersNumber[1,1:4]);
    parametersNumber[2,5] <- sum(parametersNumber[2,1:4]);

    ##### Deal with the holdout sample #####
    if(holdout && h>0){
        errormeasures <- measures(yHoldout,yForecast,yInSample);
    }
    else{
        errormeasures <- NULL;
    }

    # Amend the class of state matrix
    if(any(yClasses=="ts")){
        matVt <- ts(t(matVt), start=(yIndex[1]-(yIndex[2]-yIndex[1])*lagsModelMax), frequency=yFrequency);
    }
    else{
        yStatesIndex <- yInSampleIndex[1] - lagsModelMax*diff(tail(yInSampleIndex,2)) +
            c(1:lagsModelMax-1)*diff(tail(yInSampleIndex,2));
        yStatesIndex <- c(yStatesIndex, yInSampleIndex);
        matVt <- zoo(t(matVt), order.by=yStatesIndex);
    }

    ##### Print output #####
    # if(!silent){
    #     if(any(abs(eigen(matF - vecG %*% matWt, only.values=TRUE)$values)>(1 + 1E-10))){
    #         if(bounds!="a"){
    #             warning("Unstable model was estimated! Use bounds='admissible' to address this issue!",call.=FALSE);
    #         }
    #         else{
    #             warning("Something went wrong in optimiser - unstable model was estimated! Please report this error to the maintainer.",
    #                     call.=FALSE);
    #         }
    #     }
    # }

    ##### Make a plot #####
    if(!silent){
        graphmaker(actuals=y,forecast=yForecast,fitted=yFitted,
                   legend=FALSE,main=modelname);
    }

    # Transform everything into appropriate classes
    if(any(yClasses=="ts")){
        yInSample <- ts(yInSample,start=yStart, frequency=yFrequency);
        if(holdout){
            yHoldout <- ts(as.matrix(yHoldout), start=yForecastStart, frequency=yFrequency);
        }
    }
    else{
        yInSample <- zoo(yInSample, order.by=yInSampleIndex);
        if(holdout){
            yHoldout <- zoo(as.matrix(yHoldout), order.by=yForecastIndex);
        }
    }

    ##### Return values #####
    modelReturned <- list(model=modelname, timeElapsed=Sys.time()-startTime,
                          call=cl, orders=orders, lags=lags,
                          data=yInSample, holdout=yHoldout, fitted=yFitted, residuals=errors,
                          forecast=yForecast, states=matVt, accuracy=errormeasures,
                          profile=profilesRecentTable, profileInitial=profilesRecentInitial,
                          persistence=vecG[,1], transition=matF,
                          measurement=matWt, initial=initialValue, initialType=initialType,
                          nParam=parametersNumber,
                          formula=formula, regressors=regressors,
                          loss=loss, lossValue=CFValue, lossFunction=lossFunction, logLik=logLikValue,
                          ICs=setNames(c(AIC(logLikValue), AICc(logLikValue), BIC(logLikValue), BICc(logLikValue)),
                                       c("AIC","AICc","BIC","BICc")),
                          distribution=distribution, bounds=bounds,
                          scale=scale, B=B, lags=lags, lagsAll=lagsModelAll, res=res, FI=FI);

    # Fix data and holdout if we had explanatory variables
    if(!is.null(xregData) && !is.null(ncol(data))){
        # Remove redundant columns from the data
        modelReturned$data <- data[1:obsInSample,,drop=FALSE];
        if(holdout){
            modelReturned$holdout <- data[obsInSample+c(1:h),,drop=FALSE];
        }
        # Fix the ts class, which is destroyed during subsetting
        if(all(yClasses!="zoo")){
            if(is.data.frame(data)){
                modelReturned$data[,responseName] <- ts(modelReturned$data[,responseName],
                                                        start=yStart, frequency=yFrequency);
                if(holdout){
                    modelReturned$holdout[,responseName] <- ts(modelReturned$holdout[,responseName],
                                                               start=yForecastStart, frequency=yFrequency);
                }
            }
            else{
                modelReturned$data <- ts(modelReturned$data, start=yStart, frequency=yFrequency);
                if(holdout){
                    modelReturned$holdout <- ts(modelReturned$holdout, start=yForecastStart, frequency=yFrequency);
                }
            }
        }
    }

    return(structure(modelReturned,class=c("adam","smooth")));
}
