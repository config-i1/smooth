utils::globalVariables(c("silentText","silentGraph","silentLegend","initialType","yForecastStart",
                         # The following four should be deleted when the function is ready
                         "arma","persistence","phi","profilesRecentProvided"));

#' Complex Exponential Smoothing
#'
#' Function estimates CES in state space form with information potential equal
#' to errors and returns several variables.
#'
#' The function estimates Complex Exponential Smoothing in the state space 2
#' described in Svetunkov, Kourentzes (2017) with the information potential
#' equal to the approximation error.  The estimation of initial states of xt is
#' done using backcast.
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("ces","smooth")}
#'
#' @template ssBasicParam
#' @template ssAdvancedParam
#' @template ssIntervals
#' @template ssInitialParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssCESRef
#'
#' @param seasonality The type of seasonality used in CES. Can be: \code{none}
#' - No seasonality; \code{simple} - Simple seasonality, using lagged CES
#' (based on \code{t-m} observation, where \code{m} is the seasonality lag);
#' \code{partial} - Partial seasonality with real seasonal components
#' (equivalent to additive seasonality); \code{full} - Full seasonality with
#' complex seasonal components (can do both multiplicative and additive
#' seasonality, depending on the data). First letter can be used instead of
#' full words.  Any seasonal CES can only be constructed for time series
#' vectors.
#' @param lags Vector of seasonal lags.
#' @param a First complex smoothing parameter. Should be a complex number.
#'
#' NOTE! CES is very sensitive to a and b values so it is advised either to
#' leave them alone, or to use values from previously estimated model.
#' @param b Second complex smoothing parameter. Can be real if
#' \code{seasonality="partial"}. In case of \code{seasonality="full"} must be
#' complex number.
#' @param ...  Other non-documented parameters.  For example parameter
#' \code{model} can accept a previously estimated CES model and use all its
#' parameters.  \code{FI=TRUE} will make the function produce Fisher
#' Information matrix, which then can be used to calculated variances of
#' parameters of the model.
#' @return Object of class "smooth" is returned. It contains the list of the
#' following values: \itemize{
#' \item \code{model} - type of constructed model.
#' \item \code{timeElapsed} - time elapsed for the construction of the model.
#' \item \code{states} - the matrix of the components of CES. The included
#' minimum is "level" and "potential". In the case of seasonal model the
#' seasonal component is also included. In the case of exogenous variables the
#' estimated coefficients for the exogenous variables are also included.
#' \item \code{a} - complex smoothing parameter in the form a0 + ia1
#' \item \code{b} - smoothing parameter for the seasonal component. Can either
#' be real (if \code{seasonality="P"}) or complex (if \code{seasonality="F"})
#' in a form b0 + ib1.
#' \item \code{persistence} - persistence vector. This is the place, where
#' smoothing parameters live.
#' \item \code{transition} - transition matrix of the model.
#' \item \code{measurement} - measurement vector of the model.
#' \item \code{initialType} - Type of the initial values used.
#' \item \code{initial} - the initial values of the state vector (non-seasonal).
#' \item \code{nParam} - table with the number of estimated / provided parameters.
#' If a previous model was reused, then its initials are reused and the number of
#' provided parameters will take this into account.
#' \item \code{fitted} - the fitted values of CES.
#' \item \code{forecast} - the point forecast of CES.
#' \item \code{lower} - the lower bound of prediction interval. When
#' \code{interval="none"} then NA is returned.
#' \item \code{upper} - the upper bound of prediction interval. When
#' \code{interval="none"} then NA is returned.
#' \item \code{residuals} - the residuals of the estimated model.
#' \item \code{errors} - The matrix of 1 to h steps ahead errors. Only returned when the
#' multistep losses are used and semiparametric interval is needed.
#' \item \code{s2} - variance of the residuals (taking degrees of
#' freedom into account).
#' \item \code{interval} - type of interval asked by user.
#' \item \code{level} - confidence level for interval.
#' \item \code{cumulative} - whether the produced forecast was cumulative or not.
#' \item \code{y} - The data provided in the call of the function.
#' \item \code{holdout} - the holdout part of the original data.
#' \item \code{xreg} - provided vector or matrix of exogenous variables. If
#' \code{regressors="s"}, then this value will contain only selected exogenous
#' variables.
#' exogenous variables were estimated as well.
#' \item \code{initialX} - initial values for parameters of exogenous variables.
#' \item \code{ICs} - values of information criteria of the model. Includes
#' AIC, AICc, BIC and BICc.
#' \item \code{logLik} - log-likelihood of the function.
#' \item \code{lossValue} - Cost function value.
#' \item \code{loss} - Type of loss function used in the estimation.
#' \item \code{FI} - Fisher Information. Equal to NULL if \code{FI=FALSE}
#' or when \code{FI} is not provided at all.
#' \item \code{accuracy} - vector of accuracy measures for the holdout sample. In
#' case of non-intermittent data includes: MPE, MAPE, SMAPE, MASE, sMAE,
#' RelMAE, sMSE and Bias coefficient (based on complex numbers). In case of
#' intermittent data the set of errors will be: sMSE, sPIS, sCE (scaled
#' cumulative error) and Bias coefficient. This is available only when
#' \code{holdout=TRUE}.
#' \item \code{B} - the vector of all the estimated parameters.
#' }
#' @seealso \code{\link[smooth]{es}, \link[stats]{ts}, \link[smooth]{auto.ces}}
#'
#' @examples
#'
#' y <- rnorm(100,10,3)
#' ces(y,h=20,holdout=TRUE)
#' ces(y,h=20,holdout=FALSE)
#'
#' y <- 500 - c(1:100)*0.5 + rnorm(100,10,3)
#' ces(y,h=20,holdout=TRUE,interval="p",bounds="a")
#'
#' ces(BJsales,h=8,holdout=TRUE,seasonality="s",interval="sp",level=0.8)
#'
#' \donttest{ces(AirPassengers,h=18,holdout=TRUE,seasonality="s",interval="sp")
#' ces(AirPassengers,h=18,holdout=TRUE,seasonality="p",interval="np")
#' ces(AirPassengers,h=18,holdout=TRUE,seasonality="f",interval="p")}
#'
#' \donttest{ces(BJsales,holdout=TRUE,interval="np",xreg=BJsales.lead,loss="TMSE")}
#'
#' @rdname ces
# @export ces
ces_new <- function(y, seasonality=c("none","simple","partial","full"), lags=c(frequency(data)),
                initial=c("backcasting","optimal"), a=NULL, b=NULL, ic=c("AICc","AIC","BIC","BICc"),
                loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                h=10, holdout=FALSE, cumulative=FALSE,
                interval=c("none","parametric","likelihood","semiparametric","nonparametric"), level=0.95,
                bounds=c("admissible","none"),
                silent=c("all","graph","legend","output","none"),
                xreg=NULL, regressors=c("use","select"), initialX=NULL, ...){
# Function estimates CES in state space form with sigma = error
#  and returns complex smoothing parameter value, fitted values,
#  residuals, point and interval forecasts, matrix of CES components and values of
#  information criteria.
#
#    Copyright (C) 2015 - 2016i  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();
    cl <- match.call();
    ellipsis <- list(...);

    # If a previous model provided as a model, write down the variables
    if(!is.null(ellipsis$model)){
        if(is.null(ellipsis$model$model)){
            stop("The provided model is not CES.",call.=FALSE);
        }
        else if(smoothType(ellipsis$model)!="CES"){
            stop("The provided model is not CES.",call.=FALSE);
        }
        initial <- ellipsis$model$initial;
        a <- ellipsis$model$a;
        b <- ellipsis$model$b;
        if(is.null(xreg)){
            xreg <- ellipsis$model$xreg;
        }
        else{
            if(is.null(ellipsis$model$xreg)){
                xreg <- NULL;
            }
            else{
                if(ncol(xreg)!=ncol(ellipsis$model$xreg)){
                    xreg <- xreg[,colnames(ellipsis$model$xreg)];
                }
            }
        }
        model <- ellipsis$model$model;
        seasonality <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
        modelDo <- "use";
    }
    else{
        modelDo <- "";
    }

    # Prepare data in the adam() format
    if(!is.null(xreg) && is.numeric(y)){
        data <- cbind(y=as.data.frame(y),as.data.frame(xreg));
        data <- as.matrix(data)
        data <- ts(data, start=start(y), frequency=frequency(y));
        colnames(data)[1] <- "y";
        # Give name to the explanatory variables if they do not have them
        if(is.null(names(xreg))){
            if(!is.null(ncol(xreg))){
                colnames(data)[-1] <- paste0("x",c(1:ncol(xreg)));
            }
            else{
                colnames(data)[-1] <- "x";
            }
        }
    }
    else{
        data <- y;
    }

    a <- list(value=a);
    b <- list(value=b);

    if(is.null(a$value)){
        a$estimate <- TRUE;
    }
    else{
        a$estimate <- FALSE;
    }
    if(all(is.null(b$value),any(seasonality==c("p","f")))){
        b$estimate <- TRUE;
    }
    else{
        b$estimate <- FALSE;
    }

    ##### Set environment for ssInput and make all the checks #####
    # environment(ssInput) <- environment();
    # ssInput("ces",ParentEnvironment=environment());
    checkerReturn <- parametersChecker(data, model, lags, formula, orders, constant=FALSE, arma,
                                       outliers="ignore", level=0.99,
                                       persistence, phi, initial,
                                       distribution="dnorm", loss, h, holdout, occurrence="none", ic, bounds=bounds[1],
                                       regressors, yName="y",
                                       silent, modelDo, ParentEnvironment=environment(), ellipsis, fast=FALSE);

##### Elements of CES #####
filler <- function(B, matVt, matF, vecG, a, b){
    vt <- matVt[,1:lagsModelMax,drop=FALSE];
    nCoefficients <- 0;
    # No seasonality or Simple seasonality, lagged CES
    if(a$estimate){
        matF[1,2] <- B[2]-1;
        matF[2,2] <- 1-B[1];
        vecG[1:2,] <- c(B[1]-B[2],B[1]+B[2]);
        nCoefficients <- nCoefficients + 2;
    }
    else{
        matF[1,2] <- Im(a$value)-1;
        matF[2,2] <- 1-Re(a$value);
        vecG[1:2,] <- c(Re(a$value)-Im(a$value),Re(a$value)+Im(a$value));
    }

    if(seasonality=="p"){
    # Partial seasonality with a real part only
        if(b$estimate){
            vecG[3,] <- B[nCoefficients+1];
            nCoefficients <- nCoefficients + 1;
        }
        else{
            vecG[3,] <- b$value;
        }
    }
    else if(seasonality=="f"){
    # Full seasonality with both real and imaginary parts
        if(b$estimate){
            matF[3,4] <- B[nCoefficients+2]-1;
            matF[4,4] <- 1-B[nCoefficients+1];
            vecG[3:4,] <- c(B[nCoefficients+1]-B[nCoefficients+2],B[nCoefficients+1]+B[nCoefficients+2]);
            nCoefficients <- nCoefficients + 2;
        }
        else{
            matF[3,4] <- Im(b$value)-1;
            matF[4,4] <- 1-Re(b$value);
            vecG[3:4,] <- c(Re(b$value)-Im(b$value),Re(b$value)+Im(b$value));
        }
    }

    j <- 0;
    if(initialType=="o"){
        if(any(seasonality==c("n","s"))){
            vt[1:2,1:lagsModelMax] <- B[nCoefficients+(1:(2*lagsModelMax))];
            nCoefficients <- nCoefficients + lagsModelMax*2;
            j <- j+2;
        }
        else if(seasonality=="p"){
            vt[1:2,] <- B[nCoefficients+(1:2)];
            nCoefficients <- nCoefficients + 2;
            vt[3,1:lagsModelMax] <- B[nCoefficients+(1:lagsModelMax)];
            nCoefficients <- nCoefficients + lagsModelMax;
            j <- j+3;
        }
        else if(seasonality=="f"){
            vt[1:2,] <- B[nCoefficients+(1:2)];
            nCoefficients <- nCoefficients + 2;
            vt[3:4,1:lagsModelMax] <- B[nCoefficients+(1:(lagsModelMax*2))];
            nCoefficients <- nCoefficients + lagsModelMax*2;
            j <- j+4;
        }
    }
    else if(initialType=="p"){
        vt[,1:lagsModelMax] <- initialValue;
    }

# If exogenous are included
    if(xregEstimate){
        if(initialXEstimate){
            vt[j+(1:xregNumber),] <- B[nCoefficients+(1:xregNumber)];
            nCoefficients <- nCoefficients + xregNumber;
        }
    }

    return(list(matF=matF,vecG=vecG,vt=vt));
}

##### Cost function for CES #####
CF <- function(B){
# Obtain the elements of CES
    elements <- filler(B, matVt, matF, vecG, a, b);
    matVt[,1:lagsModelMax] <- elements$vt;

    adamFitted <- adamFitterWrap(matVt, matWt, elements$matF, elements$vecG,
                                 lagsModelAll, profilesObservedTable, profilesRecentTable,
                                 Etype, Ttype, Stype, componentsNumberETS, componentsNumberETSSeasonal,
                                 componentsNumberARIMA, xregNumber, constantRequired,
                                 yInSample, ot, initialType=="backcasting");
    #
    # cfRes <- costfunc(matVt, matF, matWt, yInSample, vecG,
    #                   h, lagsModel, Etype, Ttype, Stype,
    #                   multisteps, loss, normalizer, initialType,
    #                   matxt, matat, matFX, vecGX, ot,
    #                   bounds, 0);

    if(is.nan(cfRes) | is.na(cfRes)){
        cfRes <- 1e100;
    }
    return(cfRes);
}

##### Estimate ces or just use the provided values #####
CreatorCES <- function(initialType, a, b, initialXEstimate){
    # environment(likelihoodFunction) <- environment();
    # environment(ICFunction) <- environment();

    # Initialisation before the optimiser
    if(any(initialType=="o",a$estimate,b$estimate,initialXEstimate)){
        B <- NULL;
        # If we don't need to estimate a
        if(a$estimate){
            B <- c(1.3,1);
        }

        # Index for states
        j <- 0
        if(any(seasonality==c("n","s"))){
            if(initialType=="o"){
                B <- c(B,c(matVt[1:2,1:lagsModelMax]));
                j <- 2;
            }
        }
        else if(seasonality=="p"){
            if(b$estimate){
                B <- c(B,0.1);
            }
            if(initialType=="o"){
                B <- c(B,c(matVt[1:2,1]));
                B <- c(B,c(matVt[3,1:lagsModelMax]));
                j <- 3;
            }
        }
        else{
            if(b$estimate){
                B <- c(B,1.3,1);
            }
            if(initialType=="o"){
                B <- c(B,c(matVt[1:2,1]));
                B <- c(B,c(matVt[3:4,1:lagsModelMax]));
                j <- 4;
            }
        }

        if(xregEstimate){
            if(initialXEstimate){
                B <- c(B,matVt[j+c(1:xregNumber),1]);
            }
        }

        res <- nloptr(B, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
        B <- res$solution;

        #In cases of xreg the optimiser sometimes fails to find reasonable parameters
        if(!is.null(xreg)){
            res2 <- nloptr(B, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-8, "maxeval"=5000));
        }
        else{
            res2 <- nloptr(B, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-8, "maxeval"=1000));
        }
            # This condition is needed in order to make sure that we did not make the solution worse
        if(res2$objective <= res$objective){
            res <- res2;
        }

        B <- res$solution;
        cfObjective <- res$objective;

        # Parameters estimated + variance
        nParam <- length(B) + 1;
    }
    else{
        B <- c(a$value,b$value,initialValue,initialX,transitionX,persistenceX);
        cfObjective <- CF(B);

        # Only variance is estimated
        nParam <- 1;
    }

    ICValues <- ICFunction(nParam=nParam,nParamOccurrence=nParamOccurrence,
                           B=B,Etype=Etype);
    ICs <- ICValues$ICs;
    logLik <- ICValues$llikelihood;

    icBest <- ICs[ic];

    return(list(cfObjective=cfObjective,B=B,ICs=ICs,icBest=icBest,nParam=nParam,logLik=logLik));
}

##### Preset yFitted, yForecast, errors and basic parameters #####
    matVt <- matrix(NA,nComponents,obsStates);
    yFitted <- rep(NA,obsInSample);
    yForecast <- rep(NA,h);
    errors <- rep(NA,obsInSample);

    # Create ADAM profiles for correct treatment of seasonality
    adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                       lags=lags, yIndex=yIndexAll, yClasses=yClasses);
    if(profilesRecentProvided){
        profilesRecentTable <- profilesRecentTable[,1:lagsModelMax,drop=FALSE];
    }
    else{
        profilesRecentTable <- adamProfiles$recent;
    }
    profilesObservedTable <- adamProfiles$observed;

##### Define parameters for different seasonality types #####
    # Define "w" matrix, seasonal complex smoothing parameter, seasonality lag (if it is present).
    #   matVt - the matrix with the components, lags is the lags used in pFitted matrix.
    if(seasonality=="n"){
        # No seasonality
        matF <- matrix(1,2,2);
        vecG <- matrix(0,2);
        matWt <- matrix(c(1,0),1,2);
        matVt <- matrix(NA,obsStates,2);
        colnames(matVt) <- c("level","potential");
        matVt[1,] <- c(mean(yot[1:min(max(10,dataFreq),obsNonzero)]),mean(yot[1:min(max(10,dataFreq),obsNonzero)])/1.1);
    }
    else if(seasonality=="s"){
        # Simple seasonality, lagged CES
        matF <- matrix(1,2,2);
        vecG <- matrix(0,2);
        matWt <- matrix(c(1,0),1,2);
        matVt <- matrix(NA,obsStates,2);
        colnames(matVt) <- c("level.s","potential.s");
        matVt[1:lagsModelMax,1] <- yInSample[1:lagsModelMax];
        matVt[1:lagsModelMax,2] <- matVt[1:lagsModelMax,1]/1.1;
    }
    else if(seasonality=="p"){
        # Partial seasonality with a real part only
        matF <- diag(3);
        matF[2,1] <- 1;
        vecG <- matrix(0,3);
        matWt <- matrix(c(1,0,1),1,3);
        matVt <- matrix(NA,obsStates,3);
        colnames(matVt) <- c("level","potential","seasonal");
        matVt[1:lagsModelMax,1] <- mean(yInSample[1:lagsModelMax]);
        matVt[1:lagsModelMax,2] <- matVt[1:lagsModelMax,1]/1.1;
        matVt[1:lagsModelMax,3] <- decompose(ts(yInSample,frequency=lagsModelMax),type="additive")$figure;
    }
    else if(seasonality=="f"){
        # Full seasonality with both real and imaginary parts
        matF <- diag(4);
        matF[2,1] <- 1;
        matF[4,3] <- 1;
        vecG <- matrix(0,4);
        matWt <- matrix(c(1,0,1,0),1,4);
        matVt <- matrix(NA,obsStates,4);
        colnames(matVt) <- c("level","potential","seasonal 1", "seasonal 2");
        matVt[1:lagsModelMax,1] <- mean(yInSample[1:lagsModelMax]);
        matVt[1:lagsModelMax,2] <- matVt[1:lagsModelMax,1]/1.1;
        matVt[1:lagsModelMax,3] <- decompose(ts(yInSample,frequency=lagsModelMax),type="additive")$figure;
        matVt[1:lagsModelMax,4] <- matVt[1:lagsModelMax,3]/1.1;
    }

##### Prepare exogenous variables #####
    xregdata <- ssXreg(y=y, xreg=xreg, updateX=FALSE, ot=ot,
                       persistenceX=NULL, transitionX=NULL, initialX=initialX,
                       obsInSample=obsInSample, obsAll=obsAll, obsStates=obsStates,
                       lagsModelMax=lagsModelMax, h=h, regressors=regressors, silent=silentText);

    if(regressors=="u"){
        xregNumber <- xregdata$xregNumber;
        matxt <- xregdata$matxt;
        matat <- xregdata$matat;
        xregEstimate <- xregdata$xregEstimate;
        matFX <- xregdata$matFX;
        vecGX <- xregdata$vecGX;
        xregNames <- colnames(matxt);
    }
    else{
        xregNumber <- 1;
        xregNumberOriginal <- xregdata$xregNumber;
        matxtOriginal <- xregdata$matxt;
        matatOriginal <- xregdata$matat;
        xregEstimateOriginal <- xregdata$xregEstimate;
        matFXOriginal <- xregdata$matFX;
        vecGXOriginal <- xregdata$vecGX;

        matxt <- matrix(1,nrow(matxtOriginal),1);
        matat <- matrix(0,nrow(matatOriginal),1);
        xregEstimate <- FALSE;
        matFX <- matrix(1,1,1);
        vecGX <- matrix(0,1,1);
        xregNames <- NULL;
    }
    xreg <- xregdata$xreg;
    FXEstimate <- xregdata$FXEstimate;
    gXEstimate <- xregdata$gXEstimate;
    initialXEstimate <- xregdata$initialXEstimate;
    if(is.null(xreg)){
        regressors <- "u";
    }

    # These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

    # Check number of parameters vs data
    nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecGX) + initialXEstimate*ncol(matat);
    nParamOccurrence <- all(occurrence!=c("n","p"))*1;
    nParamMax <- nParamMax + nParamExo + nParamOccurrence;

    if(regressors=="u"){
        parametersNumber[1,2] <- nParamExo;
        # If transition is provided and not identity, and other things are provided, write them as "provided"
        parametersNumber[2,2] <- (length(matFX)*(!is.null(transitionX) & !all(matFX==diag(ncol(matat)))) +
                                      nrow(vecGX)*(!is.null(persistenceX)) +
                                      ncol(matat)*(!is.null(initialX)));
    }

##### Check number of observations vs number of max parameters #####
    if(obsNonzero <= nParamMax){
        if(regressors=="select"){
            if(obsNonzero <= (nParamMax - nParamExo)){
                warning(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                            nParamMax," while the number of observations is ",obsNonzero - nParamExo,"!"),call.=FALSE);
                tinySample <- TRUE;
            }
            else{
                warning(paste0("The potential number of exogenous variables is higher than the number of observations. ",
                               "This may cause problems in the estimation."),call.=FALSE);
            }
        }
        else{
            warning(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                           nParamMax," while the number of observations is ",obsNonzero,"!"),call.=FALSE);
            tinySample <- TRUE;
        }
    }
    else{
        tinySample <- FALSE;
    }

# If this is tiny sample, use SES instead
    if(tinySample){
        warning("Not enough observations to fit CES. Switching to ETS(A,N,N).",call.=FALSE);
        return(es(y,"ANN",initial=initial,loss=loss,
                  h=h,holdout=holdout,cumulative=cumulative,
                  interval=interval,level=level,
                  occurrence=occurrence,
                  oesmodel=oesmodel,
                  bounds="u",
                  silent=silent,
                  xreg=xreg,regressors=regressors,initialX=initialX,
                  updateX=updateX,persistenceX=persistenceX,transitionX=transitionX));
    }

##### Start doing things #####
    environment(intermittentParametersSetter) <- environment();
    environment(intermittentMaker) <- environment();
    environment(ssForecaster) <- environment();
    environment(ssFitter) <- environment();

##### If occurrence=="a", run a loop and select the best one #####
    if(occurrence=="a"){
        if(!silentText){
            cat("Selecting the best occurrence model...\n");
        }
        # First produce the auto model
        intermittentParametersSetter(occurrence="a",ParentEnvironment=environment());
        intermittentMaker(occurrence="a",ParentEnvironment=environment());
        intermittentModel <- CreatorCES(silent=silentText);
        occurrenceBest <- occurrence;
        occurrenceModelBest <- occurrenceModel;

        if(!silentText){
            cat("Comparing it with the best non-intermittent model...\n");
        }
        # Then fit the model without the occurrence part
        occurrence[] <- "n";
        intermittentParametersSetter(occurrence=occurrence,ParentEnvironment=environment());
        intermittentMaker(occurrence=occurrence,ParentEnvironment=environment());
        nonIntermittentModel <- CreatorCES(silent=silentText);

        # Compare the results and return the best
        if(nonIntermittentModel$icBest[ic] <= intermittentModel$icBest[ic]){
            cesValues <- nonIntermittentModel;
        }
        # If this is the "auto", then use the selected occurrence to reset the parameters
        else{
            cesValues <- intermittentModel;
            occurrenceModel <- occurrenceModelBest;
            occurrence[] <- occurrenceBest;
            intermittentParametersSetter(occurrence=occurrence,ParentEnvironment=environment());
            intermittentMaker(occurrence=occurrence,ParentEnvironment=environment());
        }
        rm(intermittentModel,nonIntermittentModel,occurrenceModelBest);
    }
    else{
        intermittentParametersSetter(occurrence=occurrence,ParentEnvironment=environment());
        intermittentMaker(occurrence=occurrence,ParentEnvironment=environment());

        cesValues <- CreatorCES(silent=silentText);
    }

    list2env(cesValues,environment());

#     if(regressors!="u"){
#         # Prepare for fitting
#         elements <- filler(B, matVt, matF, vecG, a, b);
#         matF <- elements$matF;
#         vecG <- elements$vecG;
#         matVt[1:lagsModelMax,] <- elements$vt;
#         matat[1:lagsModelMax,] <- elements$at;
#         matFX <- elements$matFX;
#         vecGX <- elements$vecGX;
#
#         # cesValues <- CreatorCES(silentText=TRUE);
#         ssFitter(ParentEnvironment=environment());
#
#         xregNames <- colnames(matxtOriginal);
#         xregNew <- cbind(errors,xreg[1:nrow(errors),]);
#         colnames(xregNew)[1] <- "errors";
#         colnames(xregNew)[-1] <- xregNames;
#         xregNew <- as.data.frame(xregNew);
#         xregResults <- stepwise(xregNew, ic=ic, silent=TRUE, df=nParam+nParamOccurrence-1);
#         xregNames <- names(coef(xregResults))[-1];
#         xregNumber <- length(xregNames);
#         if(xregNumber>0){
#             xregEstimate <- TRUE;
#             matxt <- as.data.frame(matxtOriginal)[,xregNames];
#             matat <- as.data.frame(matatOriginal)[,xregNames];
#             matFX <- diag(xregNumber);
#             vecGX <- matrix(0,xregNumber,1);
#
#             if(xregNumber==1){
#                 matxt <- matrix(matxt,ncol=1);
#                 matat <- matrix(matat,ncol=1);
#                 colnames(matxt) <- colnames(matat) <- xregNames;
#             }
#             else{
#                 matxt <- as.matrix(matxt);
#                 matat <- as.matrix(matat);
#             }
#         }
#         else{
#             xregNumber <- 1;
#             xreg <- NULL;
#         }
#
#         if(!is.null(xreg)){
#             cesValues <- CreatorCES(silentText=TRUE);
#             list2env(cesValues,environment());
#         }
#     }
#
#     if(!is.null(xreg)){
#         if(ncol(matat)==1){
#             colnames(matxt) <- colnames(matat) <- xregNames;
#         }
#         xreg <- matxt;
#         if(regressors=="s"){
#             nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecGX) + initialXEstimate*ncol(matat);
#             parametersNumber[1,2] <- nParamExo;
#         }
#     }

# Prepare for fitting
    elements <- filler(B, matVt, matF, vecG, a, b);
    matF <- elements$matF;
    vecG <- elements$vecG;
    matVt[1:lagsModelMax,] <- elements$vt;
    matat[1:lagsModelMax,] <- elements$at;
    matFX <- elements$matFX;
    vecGX <- elements$vecGX;

##### Fit simple model and produce forecast #####
    ssFitter(ParentEnvironment=environment());
    ssForecaster(ParentEnvironment=environment());

##### Do final check and make some preparations for output #####

# Write down initials of states vector and exogenous
    if(initialType!="p"){
        initialValue <- matVt[1:lagsModelMax,];
        if(initialType!="b"){
            parametersNumber[1,1] <- (parametersNumber[1,1] + 2*(seasonality!="s") +
                                      lagsModelMax*(seasonality!="n") + lagsModelMax*any(seasonality==c("f","s")));
        }
    }

    if(initialXEstimate){
        initialX <- matat[1,];
        names(initialX) <- colnames(matat);
    }

    # Make initialX NULL if all xreg were dropped
    if(length(initialX)==1){
        if(initialX==0){
            initialX <- NULL;
        }
    }

    if(gXEstimate){
        persistenceX <- vecGX;
    }

    if(FXEstimate){
        transitionX <- matFX;
    }

    # Add variance estimation
    parametersNumber[1,1] <- parametersNumber[1,1] + 1;

    # Write down the number of parameters of occurrence
    if(all(occurrence!=c("n","p")) & !occurrenceModelProvided){
        parametersNumber[1,3] <- nparam(occurrenceModel);
    }

    if(!is.null(xreg)){
        statenames <- c(colnames(matVt),colnames(matat));
        matVt <- cbind(matVt,matat);
        colnames(matVt) <- statenames;
        if(updateX){
            rownames(vecGX) <- xregNames;
            dimnames(matFX) <- list(xregNames,xregNames);
        }
    }

    # Right down the smoothing parameters
    nCoefficients <- 0;
    if(a$estimate){
        a$value <- complex(real=B[1],imaginary=B[2]);
        nCoefficients <- 2;
        parametersNumber[1,1] <- parametersNumber[1,1] + 2;
    }

    names(a$value) <- "a0+ia1";

    if(b$estimate){
        if(seasonality=="p"){
            b$value <- B[nCoefficients+1];
            parametersNumber[1,1] <- parametersNumber[1,1] + 1;
        }
        else if(seasonality=="f"){
            b$value <- complex(real=B[nCoefficients+1],imaginary=B[nCoefficients+2]);
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

    if(!is.null(xreg)){
        modelname <- "CESX";
    }
    else{
        modelname <- "CES";
    }
    modelname <- paste0(modelname,"(",seasonality,")");

    if(all(occurrence!=c("n","none"))){
        modelname <- paste0("i",modelname);
    }

    parametersNumber[1,4] <- sum(parametersNumber[1,1:3]);
    parametersNumber[2,4] <- sum(parametersNumber[2,1:3]);

    # Write down Fisher Information if needed
    if(FI & parametersNumber[1,4]>1){
        environment(likelihoodFunction) <- environment();
        FI <- -numDeriv::hessian(likelihoodFunction,B);
    }
    else{
        FI <- NA;
    }

    ##### Deal with the holdout sample #####
    if(holdout){
        yHoldout <- ts(y[(obsInSample+1):obsAll],start=yForecastStart,frequency=dataFreq);
        if(cumulative){
            errormeasures <- measures(sum(yHoldout),yForecast,h*yInSample);
        }
        else{
            errormeasures <- measures(yHoldout,yForecast,yInSample);
        }

        if(cumulative){
            yHoldout <- ts(sum(yHoldout),start=yForecastStart,frequency=dataFreq);
        }
    }
    else{
        yHoldout <- NA;
        errormeasures <- NA;
    }

    ##### Print output #####
    if(!silentText){
        if(any(abs(eigen(matF - vecG %*% matWt, only.values=TRUE)$values)>(1 + 1E-10))){
            if(bounds!="a"){
                warning("Unstable model was estimated! Use bounds='admissible' to address this issue!",call.=FALSE);
            }
            else{
                warning("Something went wrong in optimiser - unstable model was estimated! Please report this error to the maintainer.",
                        call.=FALSE);
            }
        }
    }

    ##### Make a plot #####
    if(!silentGraph){
        yForecastNew <- yForecast;
        yUpperNew <- yUpper;
        yLowerNew <- yLower;
        if(cumulative){
            yForecastNew <- ts(rep(yForecast/h,h),start=yForecastStart,frequency=dataFreq)
            if(interval){
                yUpperNew <- ts(rep(yUpper/h,h),start=yForecastStart,frequency=dataFreq)
                yLowerNew <- ts(rep(yLower/h,h),start=yForecastStart,frequency=dataFreq)
            }
        }

        if(interval){
            graphmaker(actuals=y,forecast=yForecastNew,fitted=yFitted, lower=yLowerNew,upper=yUpperNew,
                       level=level,legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
        else{
            graphmaker(actuals=y,forecast=yForecastNew,fitted=yFitted,
                       legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
    }

    ##### Return values #####
    model <- list(model=modelname,timeElapsed=Sys.time()-startTime,
                  states=matVt,a=a$value,b=b$value,
                  persistence=vecG,transition=matF,
                  measurement=matWt,
                  initialType=initialType,initial=initialValue,
                  nParam=parametersNumber,
                  fitted=yFitted,forecast=yForecast,lower=yLower,upper=yUpper,residuals=errors,
                  errors=errors.mat,s2=s2,interval=intervalType,level=level,cumulative=cumulative,
                  y=y,holdout=yHoldout,
                  xreg=xreg,initialX=initialX,
                  ICs=ICs,logLik=logLik,lossValue=cfObjective,loss=loss,FI=FI,accuracy=errormeasures,
                  B=B);
    return(structure(model,class="smooth"));
}
