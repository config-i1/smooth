utils::globalVariables(c("silentText","silentGraph","silentLegend","initialType","yForecastStart"));

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
#' \item \code{errors} - The matrix of 1 to h steps ahead errors.
#' \item \code{s2} - variance of the residuals (taking degrees of
#' freedom into account).
#' \item \code{interval} - type of interval asked by user.
#' \item \code{level} - confidence level for interval.
#' \item \code{cumulative} - whether the produced forecast was cumulative or not.
#' \item \code{y} - The data provided in the call of the function.
#' \item \code{holdout} - the holdout part of the original data.
#' \item \code{xreg} - provided vector or matrix of exogenous variables. If
#' \code{xregDo="s"}, then this value will contain only selected exogenous
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
#' @seealso \code{\link[forecast]{ets}, \link[forecast]{forecast},
#' \link[stats]{ts}, \link[smooth]{auto.ces}}
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
#' library("Mcomp")
#' y <- ts(c(M3$N0740$x,M3$N0740$xx),start=start(M3$N0740$x),frequency=frequency(M3$N0740$x))
#' ces(y,h=8,holdout=TRUE,seasonality="s",interval="sp",level=0.8)
#'
#' \dontrun{y <- ts(c(M3$N1683$x,M3$N1683$xx),start=start(M3$N1683$x),frequency=frequency(M3$N1683$x))
#' ces(y,h=18,holdout=TRUE,seasonality="s",interval="sp")
#' ces(y,h=18,holdout=TRUE,seasonality="p",interval="np")
#' ces(y,h=18,holdout=TRUE,seasonality="f",interval="p")}
#'
#' \dontrun{x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)))
#' ces(ts(c(M3$N1457$x,M3$N1457$xx),frequency=12),h=18,holdout=TRUE,
#'     interval="np",xreg=x,loss="TMSE")}
#'
#' @export ces
ces <- function(y, seasonality=c("none","simple","partial","full"),
                initial=c("backcasting","optimal"), a=NULL, b=NULL, ic=c("AICc","AIC","BIC","BICc"),
                loss=c("MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                h=10, holdout=FALSE, cumulative=FALSE,
                interval=c("none","parametric","likelihood","semiparametric","nonparametric"), level=0.95,
                bounds=c("admissible","none"),
                silent=c("all","graph","legend","output","none"),
                xreg=NULL, xregDo=c("use","select"), initialX=NULL, ...){
# Function estimates CES in state space form with sigma = error
#  and returns complex smoothing parameter value, fitted values,
#  residuals, point and interval forecasts, matrix of CES components and values of
#  information criteria.
#
#    Copyright (C) 2015 - 2016i  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

    ### Depricate the old parameters
    ellipsis <- list(...)
    ellipsis <- depricator(ellipsis, "occurrence", "es");
    ellipsis <- depricator(ellipsis, "oesmodel", "es");
    ellipsis <- depricator(ellipsis, "updateX", "es");
    ellipsis <- depricator(ellipsis, "persistenceX", "es");
    ellipsis <- depricator(ellipsis, "transitionX", "es");
    updateX <- FALSE;
    persistenceX <- transitionX <- NULL;
    occurrence <- "none";
    oesmodel <- "MNN";

# Add all the variables in ellipsis to current environment
    list2env(ellipsis,environment());

    # If a previous model provided as a model, write down the variables
    if(exists("model",inherits=FALSE)){
        if(is.null(model$model)){
            stop("The provided model is not CES.",call.=FALSE);
        }
        else if(smoothType(model)!="CES"){
            stop("The provided model is not CES.",call.=FALSE);
        }
        if(!is.null(model$occurrence)){
            occurrence <- model$occurrence;
        }
        initial <- model$initial;
        a <- model$a;
        b <- model$b;
        if(is.null(xreg)){
            xreg <- model$xreg;
        }
        else{
            if(is.null(model$xreg)){
                xreg <- NULL;
            }
            else{
                if(ncol(xreg)!=ncol(model$xreg)){
                    xreg <- xreg[,colnames(model$xreg)];
                }
            }
        }
        initialX <- model$initialX;
        persistenceX <- model$persistenceX;
        transitionX <- model$transitionX;
        if(any(c(persistenceX)!=0) | any((transitionX!=0)&(transitionX!=1))){
            updateX <- TRUE;
        }
        model <- model$model;
        seasonality <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
    }

##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- environment();
    ssInput("ces",ParentEnvironment=environment());

##### Elements of CES #####
ElementsCES <- function(B){
    vt <- matrix(matvt[1:lagsModelMax,],lagsModelMax);
    nCoefficients <- 0;
    # No seasonality or Simple seasonality, lagged CES
    if(a$estimate){
        matF[1,2] <- B[2]-1;
        matF[2,2] <- 1-B[1];
        vecg[1:2,] <- c(B[1]-B[2],B[1]+B[2]);
        nCoefficients <- nCoefficients + 2;
    }
    else{
        matF[1,2] <- Im(a$value)-1;
        matF[2,2] <- 1-Re(a$value);
        vecg[1:2,] <- c(Re(a$value)-Im(a$value),Re(a$value)+Im(a$value));
    }

    if(seasonality=="p"){
    # Partial seasonality with a real part only
        if(b$estimate){
            vecg[3,] <- B[nCoefficients+1];
            nCoefficients <- nCoefficients + 1;
        }
        else{
            vecg[3,] <- b$value;
        }
    }
    else if(seasonality=="f"){
    # Full seasonality with both real and imaginary parts
        if(b$estimate){
            matF[3,4] <- B[nCoefficients+2]-1;
            matF[4,4] <- 1-B[nCoefficients+1];
            vecg[3:4,] <- c(B[nCoefficients+1]-B[nCoefficients+2],B[nCoefficients+1]+B[nCoefficients+2]);
            nCoefficients <- nCoefficients + 2;
        }
        else{
            matF[3,4] <- Im(b$value)-1;
            matF[4,4] <- 1-Re(b$value);
            vecg[3:4,] <- c(Re(b$value)-Im(b$value),Re(b$value)+Im(b$value));
        }
    }

    if(initialType=="o"){
        if(any(seasonality==c("n","s"))){
            vt[1:lagsModelMax,] <- B[nCoefficients+(1:(2*lagsModelMax))];
            nCoefficients <- nCoefficients + lagsModelMax*2;
        }
        else if(seasonality=="p"){
            vt[,1:2] <- rep(B[nCoefficients+(1:2)],each=lagsModelMax);
            nCoefficients <- nCoefficients + 2;
            vt[1:lagsModelMax,3] <- B[nCoefficients+(1:lagsModelMax)];
            nCoefficients <- nCoefficients + lagsModelMax;
        }
        else if(seasonality=="f"){
            vt[,1:2] <- rep(B[nCoefficients+(1:2)],each=lagsModelMax);
            nCoefficients <- nCoefficients + 2;
            vt[1:lagsModelMax,3:4] <- B[nCoefficients+(1:(lagsModelMax*2))];
            nCoefficients <- nCoefficients + lagsModelMax*2;
        }
    }
    else if(initialType=="b"){
        vt[1:lagsModelMax,] <- matvt[1:lagsModelMax,];
    }
    else{
        vt[1:lagsModelMax,] <- initialValue;
    }

# If exogenous are included
    if(xregEstimate){
        at <- matrix(NA,lagsModelMax,nExovars);
        if(initialXEstimate){
            at[,] <- rep(B[nCoefficients+(1:nExovars)],each=lagsModelMax);
            nCoefficients <- nCoefficients + nExovars;
        }
        else{
            at <- matat[1:lagsModelMax,];
        }
        if(updateX){
            if(FXEstimate){
                matFX <- matrix(B[nCoefficients+(1:(nExovars^2))],nExovars,nExovars);
                nCoefficients <- nCoefficients + nExovars^2;
            }

            if(gXEstimate){
                vecgX <- matrix(B[nCoefficients+(1:nExovars)],nExovars,1);
                nCoefficients <- nCoefficients + nExovars;
            }
        }
    }
    else{
        at <- matrix(matat[1:lagsModelMax,],lagsModelMax,nExovars);
    }

    return(list(matF=matF,vecg=vecg,vt=vt,at=at,matFX=matFX,vecgX=vecgX));
}

##### Cost function for CES #####
CF <- function(B){
# Obtain the elements of CES
    elements <- ElementsCES(B);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:lagsModelMax,] <- elements$vt;
    matat[1:lagsModelMax,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

    cfRes <- costfunc(matvt, matF, matw, yInSample, vecg,
                      h, lagsModel, Etype, Ttype, Stype,
                      multisteps, loss, normalizer, initialType,
                      matxt, matat, matFX, vecgX, ot,
                      bounds, 0);

    if(is.nan(cfRes) | is.na(cfRes)){
        cfRes <- 1e100;
    }
    return(cfRes);
}

##### Estimate ces or just use the provided values #####
CreatorCES <- function(silentText=FALSE,...){
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();

    if(any(initialType=="o",a$estimate,b$estimate,initialXEstimate,FXEstimate,gXEstimate)){
        B <- NULL;
        # If we don't need to estimate a
        if(a$estimate){
            B <- c(1.3,1);
        }

        if(any(seasonality==c("n","s"))){
            if(initialType=="o"){
                B <- c(B,c(matvt[1:lagsModelMax,]));
            }
        }
        else if(seasonality=="p"){
            if(b$estimate){
                B <- c(B,0.1);
            }
            if(initialType=="o"){
                B <- c(B,c(matvt[1,1:2]));
                B <- c(B,c(matvt[1:lagsModelMax,3]));
            }
        }
        else{
            if(b$estimate){
                B <- c(B,1.3,1);
            }
            if(initialType=="o"){
                B <- c(B,c(matvt[1,1:2]));
                B <- c(B,c(matvt[1:lagsModelMax,3:4]));
            }
        }

        if(xregEstimate){
            if(initialXEstimate){
                B <- c(B,matat[lagsModelMax,]);
            }
            if(updateX){
                if(FXEstimate){
                    B <- c(B,c(diag(nExovars)));
                }
                if(gXEstimate){
                    B <- c(B,rep(0,nExovars));
                }
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
    matvt <- matrix(NA,nrow=obsStates,ncol=nComponents);
    yFitted <- rep(NA,obsInSample);
    yForecast <- rep(NA,h);
    errors <- rep(NA,obsInSample);

##### Define parameters for different seasonality types #####
    # Define "w" matrix, seasonal complex smoothing parameter, seasonality lag (if it is present).
    #   matvt - the matrix with the components, lags is the lags used in pFitted matrix.
    if(seasonality=="n"){
        # No seasonality
        matF <- matrix(1,2,2);
        vecg <- matrix(0,2);
        matw <- matrix(c(1,0),1,2);
        matvt <- matrix(NA,obsStates,2);
        colnames(matvt) <- c("level","potential");
        matvt[1,] <- c(mean(yot[1:min(max(10,dataFreq),obsNonzero)]),mean(yot[1:min(max(10,dataFreq),obsNonzero)])/1.1);
    }
    else if(seasonality=="s"){
        # Simple seasonality, lagged CES
        matF <- matrix(1,2,2);
        vecg <- matrix(0,2);
        matw <- matrix(c(1,0),1,2);
        matvt <- matrix(NA,obsStates,2);
        colnames(matvt) <- c("level.s","potential.s");
        matvt[1:lagsModelMax,1] <- yInSample[1:lagsModelMax];
        matvt[1:lagsModelMax,2] <- matvt[1:lagsModelMax,1]/1.1;
    }
    else if(seasonality=="p"){
        # Partial seasonality with a real part only
        matF <- diag(3);
        matF[2,1] <- 1;
        vecg <- matrix(0,3);
        matw <- matrix(c(1,0,1),1,3);
        matvt <- matrix(NA,obsStates,3);
        colnames(matvt) <- c("level","potential","seasonal");
        matvt[1:lagsModelMax,1] <- mean(yInSample[1:lagsModelMax]);
        matvt[1:lagsModelMax,2] <- matvt[1:lagsModelMax,1]/1.1;
        matvt[1:lagsModelMax,3] <- decompose(ts(yInSample,frequency=lagsModelMax),type="additive")$figure;
    }
    else if(seasonality=="f"){
        # Full seasonality with both real and imaginary parts
        matF <- diag(4);
        matF[2,1] <- 1;
        matF[4,3] <- 1;
        vecg <- matrix(0,4);
        matw <- matrix(c(1,0,1,0),1,4);
        matvt <- matrix(NA,obsStates,4);
        colnames(matvt) <- c("level","potential","seasonal 1", "seasonal 2");
        matvt[1:lagsModelMax,1] <- mean(yInSample[1:lagsModelMax]);
        matvt[1:lagsModelMax,2] <- matvt[1:lagsModelMax,1]/1.1;
        matvt[1:lagsModelMax,3] <- decompose(ts(yInSample,frequency=lagsModelMax),type="additive")$figure;
        matvt[1:lagsModelMax,4] <- matvt[1:lagsModelMax,3]/1.1;
    }

##### Prepare exogenous variables #####
    xregdata <- ssXreg(y=y, xreg=xreg, updateX=FALSE, ot=ot,
                       persistenceX=NULL, transitionX=NULL, initialX=initialX,
                       obsInSample=obsInSample, obsAll=obsAll, obsStates=obsStates,
                       lagsModelMax=lagsModelMax, h=h, xregDo=xregDo, silent=silentText);

    if(xregDo=="u"){
        nExovars <- xregdata$nExovars;
        matxt <- xregdata$matxt;
        matat <- xregdata$matat;
        xregEstimate <- xregdata$xregEstimate;
        matFX <- xregdata$matFX;
        vecgX <- xregdata$vecgX;
        xregNames <- colnames(matxt);
    }
    else{
        nExovars <- 1;
        nExovarsOriginal <- xregdata$nExovars;
        matxtOriginal <- xregdata$matxt;
        matatOriginal <- xregdata$matat;
        xregEstimateOriginal <- xregdata$xregEstimate;
        matFXOriginal <- xregdata$matFX;
        vecgXOriginal <- xregdata$vecgX;

        matxt <- matrix(1,nrow(matxtOriginal),1);
        matat <- matrix(0,nrow(matatOriginal),1);
        xregEstimate <- FALSE;
        matFX <- matrix(1,1,1);
        vecgX <- matrix(0,1,1);
        xregNames <- NULL;
    }
    xreg <- xregdata$xreg;
    FXEstimate <- xregdata$FXEstimate;
    gXEstimate <- xregdata$gXEstimate;
    initialXEstimate <- xregdata$initialXEstimate;
    if(is.null(xreg)){
        xregDo <- "u";
    }

    # These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

    # Check number of parameters vs data
    nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) + initialXEstimate*ncol(matat);
    nParamOccurrence <- all(occurrence!=c("n","p"))*1;
    nParamMax <- nParamMax + nParamExo + nParamOccurrence;

    if(xregDo=="u"){
        parametersNumber[1,2] <- nParamExo;
        # If transition is provided and not identity, and other things are provided, write them as "provided"
        parametersNumber[2,2] <- (length(matFX)*(!is.null(transitionX) & !all(matFX==diag(ncol(matat)))) +
                                      nrow(vecgX)*(!is.null(persistenceX)) +
                                      ncol(matat)*(!is.null(initialX)));
    }

##### Check number of observations vs number of max parameters #####
    if(obsNonzero <= nParamMax){
        if(xregDo=="select"){
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
                  xreg=xreg,xregDo=xregDo,initialX=initialX,
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

    if(xregDo!="u"){
        # Prepare for fitting
        elements <- ElementsCES(B);
        matF <- elements$matF;
        vecg <- elements$vecg;
        matvt[1:lagsModelMax,] <- elements$vt;
        matat[1:lagsModelMax,] <- elements$at;
        matFX <- elements$matFX;
        vecgX <- elements$vecgX;

        # cesValues <- CreatorCES(silentText=TRUE);
        ssFitter(ParentEnvironment=environment());

        xregNames <- colnames(matxtOriginal);
        xregNew <- cbind(errors,xreg[1:nrow(errors),]);
        colnames(xregNew)[1] <- "errors";
        colnames(xregNew)[-1] <- xregNames;
        xregNew <- as.data.frame(xregNew);
        xregResults <- stepwise(xregNew, ic=ic, silent=TRUE, df=nParam+nParamOccurrence-1);
        xregNames <- names(coef(xregResults))[-1];
        nExovars <- length(xregNames);
        if(nExovars>0){
            xregEstimate <- TRUE;
            matxt <- as.data.frame(matxtOriginal)[,xregNames];
            matat <- as.data.frame(matatOriginal)[,xregNames];
            matFX <- diag(nExovars);
            vecgX <- matrix(0,nExovars,1);

            if(nExovars==1){
                matxt <- matrix(matxt,ncol=1);
                matat <- matrix(matat,ncol=1);
                colnames(matxt) <- colnames(matat) <- xregNames;
            }
            else{
                matxt <- as.matrix(matxt);
                matat <- as.matrix(matat);
            }
        }
        else{
            nExovars <- 1;
            xreg <- NULL;
        }

        if(!is.null(xreg)){
            cesValues <- CreatorCES(silentText=TRUE);
            list2env(cesValues,environment());
        }
    }

    if(!is.null(xreg)){
        if(ncol(matat)==1){
            colnames(matxt) <- colnames(matat) <- xregNames;
        }
        xreg <- matxt;
        if(xregDo=="s"){
            nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) + initialXEstimate*ncol(matat);
            parametersNumber[1,2] <- nParamExo;
        }
    }

# Prepare for fitting
    elements <- ElementsCES(B);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:lagsModelMax,] <- elements$vt;
    matat[1:lagsModelMax,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

##### Fit simple model and produce forecast #####
    ssFitter(ParentEnvironment=environment());
    ssForecaster(ParentEnvironment=environment());

##### Do final check and make some preparations for output #####

# Write down initials of states vector and exogenous
    if(initialType!="p"){
        initialValue <- matvt[1:lagsModelMax,];
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
        persistenceX <- vecgX;
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
        statenames <- c(colnames(matvt),colnames(matat));
        matvt <- cbind(matvt,matat);
        colnames(matvt) <- statenames;
        if(updateX){
            rownames(vecgX) <- xregNames;
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
        if(any(abs(eigen(matF - vecg %*% matw)$values)>(1 + 1E-10))){
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
                  states=matvt,a=a$value,b=b$value,
                  persistence=vecg,transition=matF,
                  measurement=matw,
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
