utils::globalVariables(c("xregData","xregModel","xregNumber","initialXregEstimate","xregNames",
                         "otLogical","yFrequency","yIndex",
                         "persistenceXreg","yHoldout","distribution"));

#' Complex Exponential Smoothing
#'
#' Function estimates CES in state space form with information potential equal
#' to errors and returns several variables.
#'
#' The function estimates Complex Exponential Smoothing in the state space form
#' described in Svetunkov et al. (2022) with the information potential
#' equal to the approximation error.
#'
#' The \code{auto.ces()} function implements the automatic seasonal component
#' selection based on information criteria.
#'
#' \code{ces_old()} is the old implementation of the model and will be discontinued
#' starting from smooth v4.5.0.
#'
#' \code{ces()} uses two optimisers to get good estimates of parameters. By default
#' these are BOBYQA and then Nelder-Mead. This can be regulated via \code{...} - see
#' details below.
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("ces","smooth")}
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ADAMDataFormulaRegLossSilentHHoldout
#'
#' @template smoothRef
#' @template ssADAMRef
#' @template ssGeneralRef
#' @template ssCESRef
#'
#' @param data Vector, containing data needed to be forecasted. If a matrix (or
#' data.frame / data.table) is provided, then the first column is used as a
#' response variable, while the rest of the matrix is used as a set of explanatory
#' variables. \code{formula} can be used in the latter case in order to define what
#' relation to have.
#' @param seasonality The type of seasonality used in CES. Can be: \code{none}
#' - No seasonality; \code{simple} - Simple seasonality, using lagged CES
#' (based on \code{t-m} observation, where \code{m} is the seasonality lag);
#' \code{partial} - Partial seasonality with the real seasonal component
#' (equivalent to additive seasonality); \code{full} - Full seasonality with
#' complex seasonal component (can do both multiplicative and additive
#' seasonality, depending on the data). First letter can be used instead of
#' full words.
#'
#' In case of the \code{auto.ces()} function, this parameter defines which models
#' to try.
#' @param lags Vector of lags to use in the model. Allows defining multiple seasonal models.
#' @param formula Formula to use in case of explanatory variables. If \code{NULL},
#' then all the variables are used as is. Can also include \code{trend}, which would add
#' the global trend. Only needed if \code{data} is a matrix or if \code{trend} is provided.
#' @param initial Should be a character, which can be \code{"optimal"},
#' meaning that all initial states are optimised, or \code{"backcasting"},
#' meaning that the initials of dynamic part of the model are produced using
#' backcasting procedure (advised for data with high frequency). In the latter
#' case, the parameters of the explanatory variables are optimised. This is
#' recommended for CESX. Alternatively, you can set \code{initial="complete"}
#' backcasting, which means that all states (including explanatory variables)
#' are initialised via backcasting.
#' @param a First complex smoothing parameter. Should be a complex number.
#'
#' NOTE! CES is very sensitive to a and b values so it is advised either to
#' leave them alone, or to use values from previously estimated model.
#' @param b Second complex smoothing parameter. Can be real if
#' \code{seasonality="partial"}. In case of \code{seasonality="full"} must be
#' complex number.
#' @param bounds The type of bounds for the persistence to use in the model
#' estimation. Can be either \code{admissible} - guaranteeing the stability of the
#' model, or \code{none} - no restrictions (potentially dangerous).
#' @param model A previously estimated GUM model, if provided, the function
#' will not estimate anything and will use all its parameters.
#' @param ...  Other non-documented parameters. See \link[smooth]{adam} for
#' details. However, there are several unique parameters passed to the optimiser
#' in comparison with \code{adam}:
#' 1. \code{algorithm0}, which defines what algorithm to use in nloptr for the initial
#' optimisation. By default, this is "NLOPT_LN_BOBYQA".
#' 2. \code{algorithm} determines the second optimiser. By default this is
#' "NLOPT_LN_NELDERMEAD".
#' 3. maxeval0 and maxeval, that determine the number of iterations for the two
#' optimisers. By default, \code{maxeval0=1000}, \code{maxeval=40*k}, where
#' k is the number of estimated parameters.
#' 4. xtol_rel0 and xtol_rel, which are 1e-8 and 1e-6 respectively.
#' There are also ftol_rel0, ftol_rel, ftol_abs0 and ftol_abs, which by default
#' are set to values explained in the \code{nloptr.print.options()} function.
#'
#' @return Object of class "adam" is returned with similar elements to the
#' \link[smooth]{adam} function.
#'
#' @seealso \code{\link[smooth]{adam}, \link[smooth]{es}}
#'
#' @examples
#' y <- rnorm(100,10,3)
#' ces(y, h=20, holdout=FALSE)
#'
#' y <- 500 - c(1:100)*0.5 + rnorm(100,10,3)
#' ces(y, h=20, holdout=TRUE)
#'
#' ces(BJsales, h=8, holdout=TRUE)
#'
#' \donttest{ces(AirPassengers, h=18, holdout=TRUE, seasonality="s")
#' ces(AirPassengers, h=18, holdout=TRUE, seasonality="p")
#' ces(AirPassengers, h=18, holdout=TRUE, seasonality="f")}

#' @rdname ces
#' @export
ces <- function(data, seasonality=c("none","simple","partial","full"), lags=c(frequency(data)),
                formula=NULL, regressors=c("use","select","adapt"),
                initial=c("backcasting","optimal","complete"), a=NULL, b=NULL,
                loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                h=0, holdout=FALSE, bounds=c("admissible","none"), silent=TRUE,
                model=NULL, ...){
# Function estimates CES in state space form with sigma = error
# and returns complex smoothing parameter value, fitted values,
# residuals, point and interval forecasts, matrix of CES components and values of
# information criteria.
#
#    Copyright (C) 2015 - Inf  Ivan Svetunkov


# Start measuring the time of calculations
    startTime <- Sys.time();
    cl <- match.call();
    # Record the parental environment. Needed for optimal initialisation
    env <- parent.frame();

    ellipsis <- list(...);

    # Check seasonality and loss
    seasonality <- match.arg(seasonality);
    loss <- match.arg(loss);

    # paste0() is needed in order to get rid of potential issues with names
    yName <- paste0(deparse(substitute(data)),collapse="");

    # Assume that the model is not provided
    profilesRecentProvided <- FALSE;
    profilesRecentTable <- NULL;

    # If a previous model provided as a model, write down the variables
    if(!is.null(model)){
        if(is.null(model$model)){
            stop("The provided model is not CES.",call.=FALSE);
        }
        else if(smoothType(model)!="CES"){
            stop("The provided model is not CES.",call.=FALSE);
        }
        # This needs to be fixed to align properly in case of various seasonals
        profilesRecentInitial <- profilesRecentTable <- model$profileInitial;
        profilesRecentProvided[] <- TRUE;
        # This is needed to save initials and to avoid the standard checks
        initialValueProvided <- model$initial;
        initialOriginal <- initial <- model$initialType;
        a <- model$parameters$a;
        b <- model$parameters$b;
        seasonality <- model$seasonality;
        matVt <- t(model$states);
        matWt <- model$measurement;
        matF <- model$transition;
        vecG <- as.matrix(model$persistence);
        ellipsis$B <- coef(model);
        lags <- lags(model);

        model <- model$model;
        model <- NULL;
        modelDo <- modelDoOriginal <- "use";
    }
    else{
        modelDo <- modelDoOriginal <- "estimate";
        initialValueProvided <- NULL;
        initialOriginal <- initial;
    }

    a <- list(value=a);
    b <- list(value=b);

    if(is.null(a$value)){
        a$estimate <- TRUE;
    }
    else{
        a$estimate <- FALSE;
    }
    if(is.null(b$value) && any(seasonality==c("partial","full"))){
        b$estimate <- TRUE;
    }
    else{
        b$estimate <- FALSE;
    }

    if(seasonality=="partial"){
        b$number <- 1;
    }
    else if(seasonality=="full"){
        b$number <- 2;
    }
    else{
        b$number <- 0;
    }

    # Make it look like ANN/ANA/AAA (to get correct lagsModelAll)
    model <- switch(seasonality,
                    "none"="AAN",
                    "partial"="AAA",
                    "ANA");

    # If initial was provided, trick parametersChecker
    if(!is.character(initial)){
        initialValueProvided <- initial;
        initial <- "optimal";
    }
    else{
        initial <- match.arg(initial);
    }

    ##### Set environment for ssInput and make all the checks #####
    checkerReturn <- parametersChecker(data=data, model, lags, formulaToUse=formula,
                                       orders=list(ar=c(0),i=c(0),ma=c(0),select=FALSE),
                                       constant=FALSE, arma=NULL,
                                       outliers="ignore", level=0.99,
                                       persistence=NULL, phi=NULL, initial,
                                       distribution="dnorm", loss, h, holdout, occurrence="none",
                                       # This is not needed by the function
                                       ic="AICc", bounds=bounds[1],
                                       regressors=regressors, yName=yName,
                                       silent, modelDo, ParentEnvironment=environment(), ellipsis, fast=FALSE);

    # This is the variable needed for the C++ code to determine whether the head of data needs to be
    # refined. GUM doesn't need that.
    refineHead <- FALSE;

    # Values for the preliminary optimiser
    if(is.null(ellipsis$algorithm0)){
        algorithm0 <- "NLOPT_LN_BOBYQA";
    }
    else{
        algorithm0 <- ellipsis$algorithm0;
    }
    if(is.null(ellipsis$maxeval0)){
        maxeval0 <- 1000;
    }
    else{
        maxeval0 <- ellipsis$maxeval0;
    }
    if(is.null(ellipsis$maxtime0)){
        maxtime0 <- -1;
    }
    else{
        maxtime0 <- ellipsis$maxtime0;
    }
    if(is.null(ellipsis$xtol_rel0)){
        xtol_rel0 <- 1e-8;
    }
    else{
        xtol_rel0 <- ellipsis$xtol_rel0;
    }
    if(is.null(ellipsis$xtol_abs0)){
        xtol_abs0 <- 0;
    }
    else{
        xtol_abs0 <- ellipsis$xtol_abs0;
    }
    if(is.null(ellipsis$ftol_rel0)){
        ftol_rel0 <- 0;
    }
    else{
        ftol_rel0 <- ellipsis$ftol_rel0;
    }
    if(is.null(ellipsis$ftol_abs0)){
        ftol_abs0 <- 0;
    }
    else{
        ftol_abs0 <- ellipsis$ftol_abs0;
    }


    # Fix lagsModel and Ttype for CES. This is needed because the function drops duplicate seasonal lags
    # if(seasonality=="simple"){
    #     # Build our own lags, no non-seasonal ones
    #     lagsModelSeasonal <- lags <- rep(lags[lags!=1], each=2);
    #     lagsModelAll <- lagsModel <- matrix(lags,ncol=1);
    # }
    # else if(seasonality=="full"){
    #     # Remove unit lags
    #     lagsModelSeasonal <- lags <- rep(lags[lags!=1], each=2);
    #     # Build our own
    #     lags <- c(1,1,lags);
    #     lagsModelAll <- lagsModel <- matrix(lags,ncol=1);
    # }
    # else{
    #     Ttype <- "N";
    #     model <- "ANN";
    # }

    ##### Elements of CES #####
    filler <- function(B, matVt, matF, vecG, a, b){

        nCoefficients <- 0;
        # No seasonality or Simple seasonality, lagged CES
        if(a$estimate){
            matF[1,2] <- B[2]-1;
            matF[2,2] <- 1-B[1];
            vecG[1:2,] <- c(B[1]-B[2],
                            B[1]+B[2]);
            nCoefficients[] <- nCoefficients + 2;
        }
        else{
            matF[1,2] <- Im(a$value)-1;
            matF[2,2] <- 1-Re(a$value);
            vecG[1:2,] <- c(Re(a$value)-Im(a$value),
                            Re(a$value)+Im(a$value));
        }

        if(seasonality=="partial"){
            # Partial seasonality with a real part only
            if(b$estimate){
                vecG[3,] <- B[nCoefficients+1];
                nCoefficients[] <- nCoefficients + 1;
            }
            else{
                vecG[3,] <- b$value;
            }
        }
        else if(seasonality=="full"){
            # Full seasonality with both real and imaginary parts
            if(b$estimate){
                matF[3,4] <- B[nCoefficients+2]-1;
                matF[4,4] <- 1-B[nCoefficients+1];
                vecG[3:4,] <- c(B[nCoefficients+1]-B[nCoefficients+2],
                                B[nCoefficients+1]+B[nCoefficients+2]);
                nCoefficients[] <- nCoefficients + 2;
            }
            else{
                matF[3,4] <- Im(b$value)-1;
                matF[4,4] <- 1-Re(b$value);
                vecG[3:4,] <- c(Re(b$value)-Im(b$value),
                                Re(b$value)+Im(b$value));
            }
        }

        vt <- matVt[,1:lagsModelMax,drop=FALSE];
        j <- 0;
        if(initialType=="optimal"){
            if(any(seasonality==c("none","simple"))){
                vt[1:2,1:lagsModelMax] <- B[nCoefficients+(1:(2*lagsModelMax))];
                nCoefficients[] <- nCoefficients + lagsModelMax*2;
                j <- j+2;
            }
            else if(seasonality=="partial"){
                vt[1:2,] <- B[nCoefficients+(1:2)];
                nCoefficients[] <- nCoefficients + 2;
                vt[3,1:lagsModelMax] <- B[nCoefficients+(1:lagsModelMax)];
                nCoefficients[] <- nCoefficients + lagsModelMax;
                j <- j+3;
            }
            else if(seasonality=="full"){
                vt[1:2,] <- B[nCoefficients+(1:2)];
                nCoefficients[] <- nCoefficients + 2;
                vt[3:4,1:lagsModelMax] <- B[nCoefficients+(1:(lagsModelMax*2))];
                nCoefficients[] <- nCoefficients + lagsModelMax*2;
                j <- j+4;
            }

            # If exogenous are included
            if(xregModel && initialXregEstimate && initialType!="complete"){
                vt[j+(1:xregNumber),] <- B[nCoefficients+(1:xregNumber)];
                nCoefficients[] <- nCoefficients + xregNumber;
            }
        }
        else if(initialType=="provided"){
            vt[,1:lagsModelMax] <- initialValue;
        }

        return(list(matF=matF,vecG=vecG,vt=vt));
    }

    ##### Function returns scale parameter for the provided parameters #####
    scaler <- function(errors, obsInSample){
        return(sqrt(sum(errors^2)/obsInSample));
    }

    ##### Cost function for CES #####
    CF <- function(B, matVt, matF, vecG, a, b){
        # Obtain the elements of CES
        elements <- filler(B, matVt, matF, vecG, a, b);

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

        matVt[,1:lagsModelMax] <- elements$vt;
        # Write down the initials in the recent profile
        profilesRecentTable[] <- elements$vt;

        adamFitted <- adamFitterWrap(matVt, matWt, elements$matF, elements$vecG,
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
    logLikFunction <- function(B, matVt, matF, vecG, a, b){
        return(-CF(B, matVt=matVt, matF=matF, vecG=vecG, a=a, b=b));
    }


    #### ! In order for CES to work on the ADAM engine, pretend that it is ARIMA ####
    # So, componentsNumberETS should be zero and componentsNumberARIMA = componentsNumber

    # Create all the necessary matrices and vectors
    componentsNumberARIMA <- componentsNumber <- switch(seasonality,
                                                        "none"=,
                                                        "simple"=2,
                                                        "partial"=3,
                                                        "full"=4);

    componentsNumberETS <- componentsNumberETSSeasonal <- 0;

    lagsModelAll <- lagsModel <- matrix(c(switch(seasonality,
                                               "none"=c(1,1),
                                               "simple"=c(lagsModelMax,lagsModelMax),
                                               "partial"=c(1,1,lagsModelMax),
                                               "full"=c(1,1,lagsModelMax,lagsModelMax)),
                                          rep(1, xregNumber)),
                                        ncol=1);

    Stype <- Ttype <- "N";
    model <- "ANN";
    # A hack in case the parameters were provided
    modelDo <- modelDoOriginal;

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

        matF <- diag(componentsNumber+xregNumber);
        matF[2,1] <- 1;
        vecG <- matrix(0,componentsNumber+xregNumber,1);
        matWt <- matrix(1, obsInSample, componentsNumber+xregNumber);
        matWt[,2] <- 0;
        matVt <- matrix(0, componentsNumber+xregNumber, obsStates);
        # Fix matrices for special cases
        if(seasonality=="full"){
            matF[4,3] <- 1;
            matWt[,4] <- 0;
            rownames(matVt) <- c("level", "potential", "seasonal 1", "seasonal 2", xregNames);
            matVt[1,1:lagsModelMax] <- mean(yInSample[1:lagsModelMax]);
            matVt[2,1:lagsModelMax] <- matVt[1,1:lagsModelMax]/1.1;
            matVt[3,1:lagsModelMax] <- msdecompose(yInSample, lags=lags[lags!=1],
                                                   type="additive")$seasonal[[1]][1:lagsModelMax];
            matVt[4,1:lagsModelMax] <- matVt[3,1:lagsModelMax]/1.1;
        }
        else if(seasonality=="partial"){
            rownames(matVt) <- c("level", "potential", "seasonal", xregNames);
            matVt[1,1:lagsModelMax] <- mean(yInSample[1:lagsModelMax]);
            matVt[2,1:lagsModelMax] <- matVt[1,1:lagsModelMax]/1.1;
            matVt[3,1:lagsModelMax] <- msdecompose(yInSample, lags=lags[lags!=1],
                                                   type="additive")$seasonal[[1]][1:lagsModelMax];
        }
        else if(seasonality=="simple"){
            rownames(matVt) <- c("level.s", "potential.s", xregNames);
            matVt[1,1:lagsModelMax] <- yInSample[1:lagsModelMax];
            matVt[2,1:lagsModelMax] <- matVt[1,1:lagsModelMax]/1.1;
        }
        else{
            rownames(matVt) <- c("level", "potential", xregNames);
            matVt[1:componentsNumber,1] <- c(mean(yInSample[1:min(max(10,yFrequency),obsInSample)]),
                                             mean(yInSample[1:min(max(10,yFrequency),obsInSample)])/1.1);
        }

        # Add parameters for the X
        if(xregModel){
            matVt[componentsNumber+c(1:xregNumber),1] <- xregModelInitials[[1]][[1]];
            matWt[,componentsNumber+c(1:xregNumber)] <- xregData[1:obsInSample,];
        }

        ##### Check number of observations vs number of max parameters #####
        # if(obsNonzero <= nParamMax){
        #     if(regressors=="select"){
        #         if(obsNonzero <= (nParamMax - nParamExo)){
        #             warning(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
        #                         nParamMax," while the number of observations is ",obsNonzero - nParamExo,"!"),call.=FALSE);
        #             tinySample <- TRUE;
        #         }
        #         else{
        #             warning(paste0("The potential number of exogenous variables is higher than the number of observations. ",
        #                            "This may cause problems in the estimation."),call.=FALSE);
        #         }
        #     }
        #     else{
        #         warning(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
        #                        nParamMax," while the number of observations is ",obsNonzero,"!"),call.=FALSE);
        #         tinySample <- TRUE;
        #     }
        # }
        # else{
        #     tinySample <- FALSE;
        # }

        # If this is tiny sample, use SES instead
        # if(tinySample){
        #     warning("Not enough observations to fit CES. Switching to ETS(A,N,N).",call.=FALSE);
        #     return(es(y,"ANN",initial=initial,loss=loss,
        #               h=h,holdout=holdout,cumulative=cumulative,
        #               interval=interval,level=level,
        #               occurrence=occurrence,
        #               oesmodel=oesmodel,
        #               bounds="u",
        #               silent=silent,
        #               xreg=xreg,regressors=regressors,initialX=initialX,
        #               updateX=updateX,persistenceX=persistenceX,transitionX=transitionX));
        # }

        # Initialisation before the optimiser
        # if(any(initialType=="optimal",a$estimate,b$estimate)){
        B <- NULL;
        # If we don't need to estimate a
        if(a$estimate){
            B <- c(1.3,1);
            names(B) <- c("alpha_0","alpha_1");
        }

        if(b$estimate){
            if(seasonality=="partial"){
                B <- c(B, setNames(0.1, "beta"));
            }
            else{
                B <- c(B,
                       setNames(c(1.3,1), c("beta_0","beta_1")));
            }
        }

        # In case of optimal, get some initials from backcasting
        if(initialType=="optimal"){
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

            # Call for CES with backcasting
            cesBack <- suppressWarnings(eval(clNew, envir=env));
            B <- cesBack$B;
            # Vector of initial estimates of parameters
            if(seasonality!="simple"){
                B <- c(B, cesBack$initial$nonseasonal);
            }
            if(seasonality!="none"){
                BSeasonal <- as.vector(cesBack$initial$seasonal);
                if(seasonality=="partial"){
                    names(BSeasonal) <- paste0("seasonal_", c(1:lagsModelMax));
                }
                else{
                    names(BSeasonal) <- paste0(rep(c("seasonal 1_","seasonal 2_"), times=lagsModelMax),
                                               rep(c(1:lagsModelMax), each=2))
                }
                B <- c(B, BSeasonal);
            }

            # if(any(seasonality==c("none","simple"))){
            #     B <- c(B,c(matVt[1:2,1:lagsModelMax]));
            # }
            # else if(seasonality=="partial"){
            #     B <- c(B,
            #            setNames(matVt[1:2,1], c("level","potential")));
            #     B <- c(B,
            #            setNames(matVt[3,1:lagsModelMax],
            #                     paste0("seasonal_", c(1:lagsModelMax))));
            # }
            # else{
            #     B <- c(B,
            #            setNames(matVt[1:2,1], c("level","potential")));
            #     B <- c(B,
            #            setNames(matVt[3:4,1:lagsModelMax],
            #                     paste0(rep(c("seasonal 1_","seasonal 2_"), each=lagsModelMax),
            #                            rep(c(1:lagsModelMax), times=2))));
            # }
        }

        if(xregModel){
            B <- c(B, setNames(matVt[-c(1:componentsNumber),1], xregNames));
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
            if(xregModel){
                maxevalUsed[] <- length(B) * 100;
                maxevalUsed[] <- max(1000,maxevalUsed);
            }
        }

        # First run of BOBYQA to get better values of B
        res <- nloptr(B, CF, opts=list(algorithm=algorithm0, xtol_rel=xtol_rel0, xtol_abs=xtol_abs0,
                                       ftol_rel=ftol_rel0, ftol_abs=ftol_abs0,
                                       maxeval=maxeval0, maxtime=maxtime0, print_level=print_level),
                      matVt=matVt, matF=matF, vecG=vecG, a=a, b=b);

        if(print_level_hidden>0){
            print(res);
        }

        B[] <- res$solution;

        # Tuning the best obtained values using Nelder-Mead
        res <- suppressWarnings(nloptr(B, CF,
                                       opts=list(algorithm=algorithm, xtol_rel=xtol_rel, xtol_abs=xtol_abs,
                                                 ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                                 maxeval=maxevalUsed, maxtime=maxtime, print_level=print_level),
                                       matVt=matVt, matF=matF, vecG=vecG, a=a, b=b));

        if(print_level_hidden>0){
            print(res);
        }

        B[] <- res$solution;
        CFValue <- res$objective;

        # Parameters estimated + variance
        nParamEstimated <- length(B) + (loss=="likelihood")*1;

        # Prepare for fitting
        elements <- filler(B, matVt, matF, vecG, a, b);
        matF <- elements$matF;
        vecG <- elements$vecG;
        matVt[,1:lagsModelMax] <- elements$vt;

        # Write down the initials in the recent profile
        profilesRecentInitial <- profilesRecentTable[] <- matVt[,1:lagsModelMax,drop=FALSE];
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

        CFValue <- CF(B, matVt, matF, vecG, a, b);
        res <- NULL;

        # Only variance is estimated
        nParamEstimated <- 1;

        initialType <- initialOriginal;
        initialXregEstimate <- initialXregEstimateOriginal;
    }

    #### Fisher Information ####
    if(FI){
        # Substitute values to get hessian
        if(any(substr(names(B),1,5)=="alpha")){
            a$estimateOriginal <- a$estimate;
            a$estimate <- TRUE;
        }
        if(any(substr(names(B),1,4)=="beta")){
            b$estimateOriginal <- b$estimate;
            b$estimate <- TRUE;
        }
        initialTypeOriginal <- initialType;
        initialType <- "optimal";
        if(!is.null(initialValueProvided$xreg) && initialOriginal!="complete"){
            initialXregEstimateOriginal <- initialXregEstimate;
            initialXregEstimate <- TRUE;
        }

        FI <- -hessian(logLikFunction, B, h=stepSize, matVt=matVt, matF=matF, vecG=vecG, a=a, b=b);
        colnames(FI) <- rownames(FI) <- names(B);

        if(any(substr(names(B),1,5)=="alpha")){
            a$estimate <- a$estimateOriginal;
        }
        if(any(substr(names(B),1,4)=="beta")){
            b$estimate <- b$estimateOriginal;
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
    logLikValue <- structure(logLikFunction(B, matVt=matVt, matF=matF, vecG=vecG, a=a, b=b),
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
        initialValue <- vector("list", 1*(seasonality!="simple") + 1*(seasonality!="none") + xregModel);
        if(seasonality=="none"){
            names(initialValue) <- c("nonseasonal","xreg")[c(TRUE,xregModel)]
            initialValue$nonseasonal <- matVt[1:2,1];
        }
        else if(seasonality=="simple"){
            names(initialValue) <- c("seasonal","xreg")[c(TRUE,xregModel)]
            initialValue$seasonal <- matVt[1:2,1:lagsModelMax];
        }
        else{
            names(initialValue) <- c("nonseasonal","seasonal","xreg")[c(TRUE,TRUE,xregModel)]
            initialValue$nonseasonal <- matVt[1:2,1];
            initialValue$seasonal <- matVt[lagsModelAll!=1,1:lagsModelMax];
        }

        if(initialType=="optimal"){
            parametersNumber[1,1] <- (parametersNumber[1,1] + 2*(seasonality!="simple") +
                                      lagsModelMax*(seasonality!="none") + lagsModelMax*any(seasonality==c("full","simple")));
        }
    }
    if(xregModel){
        initialValue$xreg <- matVt[componentsNumber+1:xregNumber,1];
    }
    parametersNumber[1,5] <- sum(parametersNumber[1,])

    # Right down the smoothing parameters
    nCoefficients <- 0;
    if(a$estimate){
        a$value <- complex(real=B[1],imaginary=B[2]);
        nCoefficients <- 2;
        parametersNumber[1,1] <- parametersNumber[1,1] + 2;
    }
    names(a$value) <- "a0+ia1";

    if(b$estimate){
        if(seasonality=="partial"){
            b$value <- B[nCoefficients+1];
            parametersNumber[1,1] <- parametersNumber[1,1] + 1;
        }
        else if(seasonality=="full"){
            b$value <- complex(real=B[nCoefficients+1], imaginary=B[nCoefficients+2]);
            parametersNumber[1,1] <- parametersNumber[1,1] + 2;
        }
    }
    if(b$number!=0){
        if(is.complex(b$value)){
            names(b$value) <- "b0+ib1";
        }
        else{
            names(b$value) <- "b";
        }
    }

    modelname <- "CES";
    if(xregModel){
        modelname[] <- paste0(modelname,"X");
    }
    modelname[] <- paste0(modelname,"(",seasonality,")");

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
                          call=cl, parameters=list(a=a$value, b=b$value), seasonality=seasonality,
                          data=yInSample, holdout=yHoldout, fitted=yFitted, residuals=errors,
                          forecast=yForecast, states=matVt, accuracy=errormeasures,
                          profile=profilesRecentTable, profileInitial=profilesRecentInitial,
                          persistence=vecG[,1], transition=matF,
                          measurement=matWt, initial=initialValue, initialType=initialType,
                          nParam=parametersNumber,
                          formula=formula, regressors=regressors,
                          loss=loss, lossValue=CFValue, lossFunction=lossFunction, logLik=logLikValue,
                          # ICs=setNames(c(AIC(logLikValue), AICc(logLikValue), BIC(logLikValue), BICc(logLikValue)),
                          #              c("AIC","AICc","BIC","BICc")),
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
