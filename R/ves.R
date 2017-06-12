utils::globalVariables(c("damped","matG","initialEstimate","initialSeasonEstimate","xregEstimate","persistenceEstimate","phi",
                         "FXEstimate","gXEstimate","initialXEstimate","matGX","nParamMax"));

#' NOT AVAILABLE YET: Vector Exponential Smoothing in SSOE state-space model
#'
#' Function constructs vector ETS model and returns forecast, fitted values, errors
#' and matrix of states along with other useful variables. THIS IS CURRENTLY UNDER CONSTRUCTION!
#'
#' Function estimates vector ETS in a form of the Single Source of Error State-space
#' model of the following type:
#'
#' \deqn{
#' \mathbf{y}_{t} = \mathbf{o}_{t} (\mathbf{W} \mathbf{v}_{t-l} + \mathbf{x}_t
#' \mathbf{a}_{t-1} + \mathbf{\epsilon}_{t})
#' }{
#' y_{t} = o_{t} (W v_{t-l} + x_t a_{t-1} + \epsilon_{t})
#' }
#'
#' \deqn{
#' \mathbf{v}_{t} = \mathbf{F} \mathbf{v}_{t-l} + \mathbf{G}
#' \mathbf{\epsilon}_{t}
#' }{
#' v_{t} = F v_{t-l} + G \epsilon_{t}
#' }
#'
#' \deqn{\mathbf{a}_{t} = \mathbf{F_{X}} \mathbf{a}_{t-1} + \mathbf{G_{X}}
#' \mathbf{\epsilon}_{t} / \mathbf{x}_{t}}{a_{t} = F_{X} a_{t-1} + G_{X} \epsilon_{t}
#' / x_{t}}
#'
#' Where \eqn{y_{t}} is the vector of time series on observation \eqn{t}, \eqn{o_{t}}
#' is the vector of Bernoulli distributed random variable (in case of normal data it
#' becomes unit vector for all observations), \eqn{\mathbf{v}_{t}} is the matrix of
#' states and \eqn{l} is the matrix of lags, \eqn{\mathbf{x}_t} is the vector of
#' exogenous variables. \eqn{\mathbf{W}} is the measurement matrix, \eqn{\mathbf{F}}
#' is the transition matrix and \eqn{\mathbf{G}} is the persistence matrix.
#' Finally, \eqn{\epsilon_{t}} is the vector of error terms.
#'
#' Conventionally we formulate values as:
#'
#' \deqn{\mathbf{y}'_t = (y_{1,t}, y_{2,t}, \dots, y_{m,t})}{y_t = (y_{1,t}, y_{2,t},
#' \dots, y_{m,t}),}
#' where \eqn{m} is the number of series in the group.
#' \deqn{\mathbf{v}'_t = (v_{1,t}, v_{2,t}, \dots, v_{m,t})}{v'_t = (v_{1,t}, v_{2,t},
#' \dots, v_{m,t}),}
#' where \eqn{v_{i,t}} is vector of components for i-th time series.
#' \deqn{\mathbf{W}' = (w_{1}, \dots , 0;
#' \vdots , \ddots , \vdots;
#' 0 , \vdots , w_{m})}{W' = (w_{1}, ... , 0;
#' ... , ... , ...;
#' 0 , ... , w_{m})} is matrix of measurement vectors.
#'
#' For the details see Hyndman et al. (2008), chapter 17.
#'
#' @template vssBasicParam
#' @template vssAdvancedParam
#' @template ssAuthor
#' @template vssKeywords
#'
#' @template vssGeneralRef
#'
#' @param model The type of ETS model. Can consist of 3 or 4 chars: \code{ANN},
#' \code{AAN}, \code{AAdN}, \code{AAA}, \code{AAdA}, \code{MAdM} etc.
#' \code{ZZZ} means that the model will be selected based on the chosen
#' information criteria type. ATTENTION! NO MODEL SELECTION IS AVAILABLE AT
#' THIS STAGE!
#'
#' Also \code{model} can accept a previously estimated VES model and use all its
#' parameters.
#'
#' Keep in mind that model selection with "Z" components uses Branch and Bound
#' algorithm and may skip some models that could have slightly smaller
#' information criteria.

#' @param phi In cases of damped trend this parameter defines whether the \eqn{phi}
#' should be estimated separately for each series (\code{individual}) or for the whole
#' set (\code{group}). If vector or a value is provided here, then it is used by the
#' model.
#' @param initial Can be either character or a vector / matrix of initial states.
#' If it is character, then it can be \code{"individual"}, individual values of
#' the intial non-seasonal components are udes, or \code{"group"}, meaning that
#' the initials for all the time series are set to be equal to the same value.
#' If vector of states is provided, then it is automatically transformed into
#' a matrix, assuming that these values are provided for the whole group.
#' @param initialSeason Can be either character or a vector / matrix of initial
#' states. Treated the same way as \code{initial}. This means that different time
#' series may share the same initial seasonal component.
#' @param ...  Other non-documented parameters. For example \code{FI=TRUE} will
#' make the function also produce Fisher Information matrix, which then can be
#' used to calculated variances of smoothing parameters and initial states of
#' the model.
#' @return Object of class "smooth" is returned. It contains a list of
#' values.
#'
#' @seealso \code{\link[smooth]{es}, \link[forecast]{ets},
#' \link[forecast]{forecast}}
#'
#' @examples
#'
#' library(Mcomp)
#'
#' \dontrun{es(M3$N2568$x,model="MAM",h=18,holdout=TRUE)}
#'
#'
ves <- function(data, model="ANN", persistence=c("group","independent","dependent"),
                transition=c("group","independent","dependent"), phi=c("group","individual"),
                initial=c("group","individual"), initialSeason=c("group","individual"),
                cfType=c("likelihood","diagonal","trace"),
                ic=c("AICc","AIC","BIC"), h=10, holdout=FALSE,
                intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                intermittent=c("none","auto","fixed","tsb"),
                bounds=c("admissible","usual","none"),
                silent=c("none","all","graph","legend","output"), ...){
# Copyright (C) 2017 - Inf  Ivan Svetunkov

### This should be done as expanded es() function with matrix of states (rows - time, cols - states),
### large transition matrix and a persistence matrix. The returned value of the fit is vector.
### So the vfitter can be based on amended version fitter.

# Start measuring the time of calculations
    startTime <- Sys.time();

# If a previous model provided as a model, write down the variables
    # if(is.list(model)){
    #     if(gregexpr("ETS",model$model)==-1){
    #         stop("The provided model is not ETS.",call.=FALSE);
    #     }
    #     intermittent <- model$intermittent;
    #     if(any(intermittent==c("p","provided"))){
    #         warning("The provided model had predefined values of occurences for the holdout. We don't have them.",call.=FALSE);
    #         warning("Switching to intermittent='auto'.",call.=FALSE);
    #         intermittent <- "a";
    #     }
    #     persistence <- model$persistence;
    #     transition <- model$transition;
    #     measurement <- model$measurement;
    #     initial <- model$initial;
    #     initialSeason <- model$initialSeason;
    #     if(is.null(xreg)){
    #         xreg <- model$xreg;
    #     }
    #     initialX <- model$initialX;
    #     persistenceX <- model$persistenceX;
    #     transitionX <- model$transitionX;
    #     if(any(c(persistenceX)!=0) | any((transitionX!=0)&(transitionX!=1))){
    #         updateX <- TRUE;
    #     }
    #     model <- model$model;
    #     model <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
    #     if(any(unlist(gregexpr("C",model))!=-1)){
    #         initial <- "o";
    #     }
    # }

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

##### Set environment for vssInput and make all the checks #####
    environment(vssInput) <- environment();
    vssInput(modelType="ves",ParentEnvironment=environment());

##### Cost Function for VES #####
CF <- function(A){
    # The following function should be done in R in BasicInitialiserVES
    elements <- etsmatrices(matvt, matG, matrix(A,nrow=1), nComponentsAll,
                            modellags, initialType, Ttype, Stype, nExovars, matat,
                            persistenceEstimate, initialType=="o", initialSeasonEstimate, xregEstimate,
                            matFX, matGX, updateX, FXEstimate, gXEstimate, initialXEstimate);

    cfRes <- vOptimiserWrap(elements$matvt, elements$matF, elements$matw, y, elements$matG,
                      h, modellags, Etype, Ttype, Stype,
                      normalizer,
                      matxt, elements$matat, elements$matFX, elements$matGX, ot);
    # multisteps, cfType, initialType, bounds,

    if(is.nan(cfRes) | is.na(cfRes) | is.infinite(cfRes)){
        cfRes <- 1e+100;
    }

    return(cfRes);
}

##### A values for estimation #####
# Function constructs default bounds where A values should lie
AValues <- function(Ttype,Stype,maxlag,nComponentsAll,nComponentsNonSeasonal,nSeries){
    A <- NA;
    ALower <- NA;
    AUpper <- NA;

    ### Persistence matrix
    if(persistenceEstimate){
        if(persistenceType=="g"){
            persistenceLength <- nComponentsAll;
        }
        else if(persistenceType=="i"){
            persistenceLength <- nComponentsAll*nSeries;
        }
        else if(persistenceType=="d"){
            persistenceLength <- nComponentsAll*nSeries^2;
        }
        A <- c(A,rep(0.1,persistenceLength));
        ALower <- c(ALower,rep(-5,persistenceLength));
        AUpper <- c(AUpper,rep(5,persistenceLength));
    }

    ### Damping parameter
    if(dampedEstimate){
        if(dampedType=="g"){
            dampedLength <- 1;
        }
        else if(dampedType=="i"){
            dampedLength <- nSeries;
        }
        A <- c(A,rep(0.95,dampedLength));
        ALower <- c(ALower,rep(0,dampedLength));
        AUpper <- c(AUpper,rep(1,dampedLength));
    }

    ### Transition matrix
    if(transitionEstimate){
        if(transitionType=="d"){
             transitionLength <- (nSeries*nComponentsAll - nComponentsAll^2)*nSeries;
        }
        A <- c(A,rep(0.1,transitionLength));
        ALower <- c(ALower,rep(-1,transitionLength));
        AUpper <- c(AUpper,rep(1,transitionLength));
    }

    ### Vector of initials
    if(initialEstimate){
        if(initialType=="g"){
            initialLength <- nComponentsNonSeasonal;
        }
        else{
            initialLength <- nComponentsNonSeasonal*nSeries;
        }
        A <- c(A,initialValue);
        if(Ttype!="M"){
            ALower <- c(ALower,rep(-Inf,initialLength));
            AUpper <- c(AUpper,rep(Inf,initialLength));
        }
        else{
            ALower <- c(ALower,rep(c(0.1,0.01),initialLength/2));
            AUpper <- c(AUpper,rep(c(Inf,3),initialLength/2));
        }
    }

    ### Vector of initial seasonals
    if(initialSeasonEstimate){
        if(initialType=="g"){
            initialSeasonLength <- maxlag;
        }
        else{
            initialSeasonLength <- maxlag*nSeries;
        }
        A <- c(A,initialSeasonValue);
        if(Stype=="A"){
            ALower <- c(ALower,rep(-Inf,initialSeasonLength));
            AUpper <- c(AUpper,rep(Inf,initialSeasonLength));
        }
        else{
            ALower <- c(ALower,rep(-0.0001,initialSeasonLength));
            AUpper <- c(AUpper,rep(20,initialSeasonLength));
        }
    }

    A <- A[!is.na(A)];
    ALower <- ALower[!is.na(ALower)];
    AUpper <- AUpper[!is.na(AUpper)];

    return(list(A=A,ALower=ALower,AUpper=AUpper));
}

##### Basic matrices initialiser #####
# This thing returns matvt, matF, matG, matW, dampedValue, initialValue and initialSeasonValue if they are not provided + modellags
BasicMakerVES <- function(...){
    # ellipsis <- list(...);
    # ParentEnvironment <- ellipsis[['ParentEnvironment']];

    ### Persistence matrix
    matG <- matrix(0,nSeries*nComponentsAll,nSeries);
    if(!persistenceEstimate){
        matG <- persistenceValue;
    }

    ### Damping parameter
    if(!damped){
        dampedValue <- matrix(1,nSeries,1);
    }

    ### Transition matrix
    if(any(transitionType==c("g","i","d"))){
        if(Ttype=="N"){
            transitionValue <- matrix(1,1,1);
        }
        else if(Ttype!="N"){
            transitionValue <- matrix(c(1,0,dampedValue[1],dampedValue[1]),2,2);
        }
        if(Stype!="N"){
            transitionValue <- cbind(transitionValue,rep(0,nComponentsNonSeasonal));
            transitionValue <- rbind(transitionValue,c(rep(0,nComponentsNonSeasonal),1));
        }
        transitionValue <- matrix(transitionValue,nComponentsAll,nComponentsAll);
        transitionBuffer <- diag(nSeries*nComponentsAll);
        for(i in 1:nSeries){
            transitionBuffer[c(1:nComponentsAll)+nComponentsAll*(i-1),
                             c(1:nComponentsAll)+nComponentsAll*(i-1)] <- transitionValue;
        }
        if(any(transitionType==c("i","d")) & damped){
            for(i in 1:nSeries){
                transitionBuffer[c(1:nComponentsNonSeasonal)+nComponentsAll*(i-1),
                                 nComponentsNonSeasonal+nComponentsAll*(i-1)] <- dampedValue[i];
            }
        }
        transitionValue <- transitionBuffer;
    }
    if(transitionType=="d"){
        # Fill in the other values of F with some values
        for(i in 1:nSeries){
            transitionValue[c(1:nComponentsAll)+nComponentsAll*(i-1),
                            setdiff(c(1:nSeries*nComponentsAll),c(1:nComponentsAll)+nComponentsAll*(i-1))] <- 0.1;
        }
    }
    matF <- transitionValue;

    ### Measurement matrix
    matW <- matrix(0,nSeries,nSeries*nComponentsAll);
    for(i in 1:nSeries){
        matW[i,c(1:nComponentsAll)+nComponentsAll*(i-1)] <- 1;
    }
    if(damped){
        for(i in 1:nSeries){
            matW[i,nComponentsNonSeasonal+nComponentsAll*(i-1)] <- dampedValue[i];
        }
    }

    ### Vector of states
    matvt <- matrix(NA,nComponentsAll*nSeries,obsStates);
    ## Deal with non-seasonal part of the vector of states
    if(!initialEstimate){
        initialPlaces <- nComponentsAll*(c(1:nSeries)-1)+1;
        if(Ttype!="N"){
            initialPlaces <- c(initialPlaces,nComponentsAll*(c(1:nSeries)-1)+2);
            initialPlaces <- sort(initialPlaces);
        }
        matvt[initialPlaces,1:maxlag] <- rep(initialValue,maxlag);
    }
    else{
        XValues <- rbind(rep(1,obsInSample),c(1:obsInSample));
        if(Ttype!="M"){
            initialValue <- y %*% t(XValues) %*% solve(XValues %*% t(XValues));
        }
        else{
            initialValue <- log(y) %*% t(XValues) %*% solve(XValues %*% t(XValues));
        }
        if(Ttype=="N"){
            initialValue <- matrix(initialValue[,-2],nSeries,1);
        }
        if(initialType=="g"){
            initialValue <- colMeans(initialValue);
        }
        else{
            initialValue <- as.vector(t(initialValue));
        }
        initialValue <- matrix(initialValue,nComponentsNonSeasonal * nSeries,1);
    }

    ## Deal with seasonal part of the vector of states
    if(modelIsSeasonal){
        if(initialSeasonType=="p"){
            initialPlaces <- nComponentsAll*(c(1:nSeries)-1)+3;
            matvt[initialPlaces,1:maxlag] <- initialSeasonValue;
        }
        else{
            XValues <- matrix(rep(diag(maxlag),ceiling(obsInsSample/maxlag)),maxlag)[,1:obsInsSample];
            if(Stype=="A"){
                initialSeasonValue <- (Y-rowMeans(Y)) %*% t(XValues) %*% solve(XValues %*% t(XValues));
            }
            else{
                initialSeasonValue <- (log(Y)-rowMeans(log(Y))) %*% t(XValues) %*% solve(XValues %*% t(XValues));
            }
            if(initialType=="g"){
                initialSeasonValue <- colMeans(initialSeasonValue);
            }
            else{
                initialSeasonValue <- as.vector(t(initialSeasonValue));
            }
            initialSeasonValue <- matrix(initialSeasonValue,nSeries,maxlag);
        }
    }

    ### Modellags
    modelLags <- rep(1,nComponentsAll);
    if(modelIsSeasonal){
        modelLags[nComponentsAll] <- maxlag;
    }
    modelLags <- matrix(modelLags,nSeries*nComponentsAll,1);

    return(list(matvt=matvt, matF=matF, matG=matG, matW=matW, dampedValue=dampedValue,
                initialValue=initialValue, initialSeasonValue=initialSeasonValue, modelLags=modelLags));

    # list2env(list(matvt=matvt, matF=matF, matG=matG, matW=matW, dampedValue=dampedValue,
    #               initialValue=initialValue, initialSeasonValue=initialSeasonValue, modelLags=modelLags),
    #          ParentEnvironment);
}



##### Basic parameter propagator #####
# This thing fills in matvt, matF, matG and matW with values from A and returns the corrected values
BasicInitialiserVES <- function(matvt,matF,matG,matW,A){

    nCoefficients <- 0;
    ### Persistence matrix
    if(persistenceEstimate){
        persistenceBuffer <- matrix(0,nSeries*nComponentsAll,nSeries);
        # Grouped values
        if(persistenceType=="g"){
            persistenceValue <- A[1:nComponentsAll];
            nCoefficients <- nComponentsAll;
            for(i in 1:nSeries){
                persistenceBuffer[1:nComponentsAll+nComponentsAll*(i-1),i] <- persistenceValue;
            }
            persistenceValue <- persistenceBuffer;
        }
        # Independent values
        else if(persistenceType=="i"){
            persistenceValue <- A[1:(nComponentsAll*nSeries)];
            nCoefficients <- nComponentsAll*nSeries;
            for(i in 1:nSeries){
                persistenceBuffer[1:nComponentsAll+nComponentsAll*(i-1),i] <- persistenceValue[1:nComponentsAll+nComponentsAll*(i-1)];
            }
            persistenceValue <- persistenceBuffer;
        }
        # Dependent values
        else{
            persistenceValue <- A[1:(nComponentsAll*nSeries^2)];
            nCoefficients <- nComponentsAll*nSeries^2;
        }
        matG[,] <- persistenceValue;
    }

    ### Damping parameter
    if(damped){
        if(dampedType=="g"){
            dampedValue <- matrix(A[nCoefficients+1],nSeries,1);
            nCoefficients <- nCoefficients + 1;
        }
        else if(dampedType=="i"){
            dampedValue <- matrix(A[nCoefficients+(1:nSeries)],nSeries,1);
            nCoefficients <- nCoefficients + nSeries;
        }
    }

    ### Transition matrix
    if(any(transitionType==c("i","d","g")) & damped){
        for(i in 1:nSeries){
            matF[c(1:nComponentsNonSeasonal)+nComponentsAll*(i-1),
                 nComponentsNonSeasonal+nComponentsAll*(i-1)] <- dampedValue[i];
        }
    }
    if(transitionType=="d"){
        # Fill in the other values of F with some values
        nCoefficientsBuffer <- (nSeries-1)*nComponentsAll^2;
        for(i in 1:nSeries){
            matF[c(1:nComponentsAll)+nComponentsAll*(i-1),
                            setdiff(c(1:nSeries*nComponentsAll),
                                    c(1:nComponentsAll)+nComponentsAll*(i-1))] <- A[nCoefficients+c(1:nCoefficientsBuffer)];
            nCoefficients <- nCoefficients + nCoefficientsBuffer;
        }
    }

    ### Measurement matrix
    if(dampedEstimate){
        for(i in 1:nSeries){
            matW[i,nComponentsNonSeasonal+nComponentsAll*(i-1)] <- dampedValue[i];
        }
    }

    ### Vector of states
    ## Deal with non-seasonal part of the vector of states
    if(initialEstimate){
        initialPlaces <- nComponentsAll*(c(1:nSeries)-1)+1;
        if(Ttype!="N"){
            initialPlaces <- c(initialPlaces,nComponentsAll*(c(1:nSeries)-1)+2);
            initialPlaces <- sort(initialPlaces);
        }
        if(initialType=="i"){
            initialValue <- matrix(A[nCoefficients+c(1:(nComponentsNonSeasonal*nSeries))],nComponentsNonSeasonal * nSeries,1);
            nCoefficients <- nCoefficients + nComponentsNonSeasonal*nSeries;
        }
        else if(initialType=="g"){
            initialValue <- matrix(A[nCoefficients+c(1:nComponentsNonSeasonal)],nComponentsNonSeasonal * nSeries,1);
            nCoefficients <- nCoefficients + nComponentsNonSeasonal;
        }
        matvt[initialPlaces,1:maxlag] <- rep(initialValue,maxlag);
    }

    ## Deal with seasonal part of the vector of states
    if(modelIsSeasonal & initialSeasonEstimate){
        initialPlaces <- nComponentsAll*(c(1:nSeries)-1)+3;
        if(initialSeasonType=="i"){
            matvt[initialPlaces,1:maxlag] <- matrix(A[nCoefficients+c(1:(nSeries*maxlag))],nSeries,maxlag);
            nCoefficients <- nCoefficients + nSeries*maxlag;
        }
        else if(initialSeasonType=="g"){
            matvt[initialPlaces,1:maxlag] <- matrix(A[nCoefficients+c(1:maxlag)],nSeries,maxlag);
            nCoefficients <- nCoefficients + maxlag;
        }
    }

    return(list(matvt=matvt,matF=matF,matG=matG,matW=matW));
}



##### Basic estimation function for es() #####
EstimatorVES <- function(...){
    environment(BasicMakerVES) <- environment();
    environment(AValues) <- environment();
    # environment(likelihoodFunction) <- environment();
    # environment(ICFunction) <- environment();
    environment(CF) <- environment();
    BasicMakerVES(ParentEnvironment=environment());

    AList <- AValues(bounds,Ttype,Stype,matG,matvt,maxlag,nComponents,matat);

    # Parameters are chosen to speed up the optimisation process and have decent accuracy
    res <- nloptr(AList$A, CF, lb=AList$ALower, ub=AList$AUpper,
                  opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=500));
    A <- res$solution;

    # If the optimisation failed, then probably this is because of smoothing parameters in mixed models. Set them eqaul to zero.
    if(any(A==AList$A)){
        if(A[1]==AList$A[1]){
            A[1] <- 0;
        }
        if(Ttype!="N"){
            if(A[2]==AList$A[2]){
                A[2] <- 0;
            }
            if(Stype!="N"){
                if(A[3]==AList$A[3]){
                    A[3] <- 0;
                }
            }
        }
        else{
            if(Stype!="N"){
                if(A[2]==AList$A[2]){
                    A[2] <- 0;
                }
            }
        }
        res <- nloptr(A, CF, lb=AList$ALower, ub=AList$AUpper,
                      opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=500));
        A <- res$solution;
    }

    if(any((A>=AList$AUpper),(A<=AList$ALower))){
        A[A>=AList$AUpper] <- AList$AUpper[A>=AList$AUpper] * 0.999 - 0.001;
        A[A<=AList$ALower] <- AList$ALower[A<=AList$ALower] * 1.001 + 0.001;
    }

    res2 <- nloptr(A, CF, lb=AList$ALower, ub=AList$AUpper,
                  opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-6, "maxeval"=500));
    # This condition is needed in order to make sure that we did not make the solution worse
    if(res2$objective <= res$objective){
        res <- res2;
    }
    A <- res$solution;

    if(all(A==AList$A) & (initialType!="b")){
        if(any(persistenceEstimate,gXEstimate,FXEstimate)){
            warning(paste0("Failed to optimise the model ETS(", modelCurrent,
                           "). Try different initialisation maybe?\nAnd check all the messages and warnings...",
                           "If you did your best, but the optimiser still fails, report this to the maintainer, please."),
                    call.=FALSE);
        }
    }

    nParam <- 1 + nComponents + damped + (nComponents + (maxlag - 1) * (Stype!="N")) * initialEstimate + (!is.null(xreg)) * nExovars + (updateX)*(nExovars^2 + nExovars);

    # Change cfType for model selection
    if(multisteps){
        if(substring(cfType,1,1)=="a"){
            cfType <- "aTFL";
        }
        else{
            cfType <- "TFL";
        }
    }
    else{
        cfType <- "MSE";
    }

    IAValues <- ICFunction(nParam=nParam+nParamIntermittent,A=res$solution,Etype=Etype);
    ICs <- IAValues$ICs;
    logLik <- IAValues$llikelihood;

    # Change back
    cfType <- cfTypeOriginal;
    return(list(ICs=ICs,objective=res$objective,A=A,nParam=nParam,FI=FI,logLik=logLik));
}

##### Function constructs the VES function #####
CreatorVES <- function(silent=FALSE,...){
    if(modelDo=="estimate"){
        environment(EstimatorVES) <- environment();
        res <- EstimatorVES(ParentEnvironment=environment());
        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,
                             cfObjective=res$objective,A=res$A,ICs=res$ICs,icBest=res$ICs[ic],
                             nParam=res$nParam,FI=FI,logLik=res$logLik,xreg=xreg,
                             matFX=matFX,matGX=matGX,nExovars=nExovars);
        # if(xregDo!="u"){
        #     listToReturn <- XregSelector(listToReturn=listToReturn);
        # }

        return(listToReturn);
    }
    else{
        environment(CF) <- environment();
        environment(ICFunction) <- environment();
        environment(likelihoodFunction) <- environment();
        environment(BasicMakerVES) <- environment();
        BasicMakerVES(ParentEnvironment=environment());

        A <- c(matG);
        # if(damped){
        #     A <- c(A,phi);
        # }
        A <- c(A,initialValue,initialSeason);
        if(xregEstimate){
            A <- c(A,initialX);
            if(updateX){
                A <- c(A,transitionX,persistenceX);
            }
        }

        cfObjective <- CF(A);

        # Number of parameters
        nParam <- 1 + nComponents + damped + (nComponents + (maxlag-1) * (Stype!="N")) * (initialType!="b") + !is.null(xreg) * nExovars + (updateX)*(nExovars^2 + nExovars);

# Change cfType for model selection
        if(multisteps){
            if(substring(cfType,1,1)=="a"){
                cfType <- "aTFL";
            }
            else{
                cfType <- "TFL";
            }
        }
        else{
            cfType <- "MSE";
        }

        IAValues <- ICFunction(nParam=nParam+nParamIntermittent,A=A,Etype=Etype);
        logLik <- IAValues$llikelihood;
        ICs <- IAValues$ICs;
        icBest <- ICs[ic];
        # Change back
        cfType <- cfTypeOriginal;

        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,
                             cfObjective=cfObjective,A=A,ICs=ICs,icBest=icBest,
                             nParam=nParam,FI=FI,logLik=logLik,xreg=xreg,
                             matFX=matFX,matGX=matGX,nExovars=nExovars);
        return(listToReturn);
    }
}

##### Set initialstates, initialSesons and persistence vector #####
    # If initial values are provided, write them. If not, estimate them.
    # First two columns are needed for additive seasonality, the 3rd and 4th - for the multiplicative
    if(Ttype!="N"){
        if(initialType!="p"){
            initialstates <- matrix(NA,1,4);
            initialstates[1,2] <- cov(yot[1:min(12,obsNonzero)],c(1:min(12,obsNonzero)))/var(c(1:min(12,obsNonzero)));
            initialstates[1,1] <- mean(yot[1:min(12,obsNonzero)]) - initialstates[1,2] * mean(c(1:min(12,obsNonzero)));
            if(allowMultiplicative){
                initialstates[1,4] <- exp(cov(log(yot[1:min(12,obsNonzero)]),c(1:min(12,obsNonzero)))/var(c(1:min(12,obsNonzero))));
                initialstates[1,3] <- exp(mean(log(yot[1:min(12,obsNonzero)])) - log(initialstates[1,4]) * mean(c(1:min(12,obsNonzero))));
            }
        }
        else{
            initialstates <- matrix(rep(initialValue,2),nrow=1);
        }
    }
    else{
        if(initialType!="p"){
            initialstates <- matrix(rep(mean(yot[1:min(12,obsNonzero)]),4),nrow=1);
        }
        else{
            initialstates <- matrix(rep(initialValue,4),nrow=1);
        }
    }

    # Define matrix of seasonal coefficients. The first column consists of additive, the second - multiplicative elements
    # If the seasonal model is chosen and initials are provided, fill in the first "maxlag" values of seasonal component.
    if(Stype!="N"){
        if(is.null(initialSeason)){
            initialSeasonEstimate <- TRUE;
            seasonalCoefs <- decompose(ts(c(y),frequency=datafreq),type="additive")$seasonal[1:datafreq];
            seasonalCoefs <- cbind(seasonalCoefs,decompose(ts(c(y),frequency=datafreq),
                                                           type="multiplicative")$seasonal[1:datafreq]);
        }
        else{
            initialSeasonEstimate <- FALSE;
            seasonalCoefs <- cbind(initialSeason,initialSeason);
        }
    }
    else{
        initialSeasonEstimate <- FALSE;
        seasonalCoefs <- matrix(1,1,1);
    }

    # If the persistence vector is provided, use it
    if(!is.null(persistence)){
        smoothingParameters <- cbind(persistence,persistence);
    }
    else{
        # smoothingParameters <- cbind(c(0.2,0.1,0.05),rep(0.05,3));
        smoothingParameters <- cbind(c(0.3,0.2,0.1),c(0.1,0.05,0.01));
    }

##### Preset y.fit, y.for, errors and basic parameters #####
    y.fit <- rep(NA,obsInsample);
    y.for <- rep(NA,h);
    errors <- rep(NA,obsInsample);

    basicparams <- initparams(Ttype, Stype, datafreq, obsInsample, obsAll, y,
                              damped, smoothingParameters, initialstates, seasonalCoefs);

##### Define modelDo #####
    if(any(persistenceEstimate, transitionEstimate, dampedEstimate, initialEstimate, initialSeasonEstimate,
           FXEstimate, gXEstimate, initialXEstimate)){
        modelDo <- "estimate";
        modelCurrent <- model;
    }
    else{
        modelDo <- "nothing";
    }

##### Now do estimation and model selection #####
    # environment(intermittentParametersSetter) <- environment();
    # environment(intermittentMaker) <- environment();
    environment(BasicInitialiserVES) <- environment();
    environment(vssFitter) <- environment();
    environment(vssForecaster) <- environment();

    EtypeOriginal <- Etype;
    TtypeOriginal <- Ttype;
    StypeOriginal <- Stype;
# If auto intermittent, then estimate model with intermittent="n" first.
    # if(any(intermittent==c("a","n"))){
    #     intermittentParametersSetter(intermittent="n",ParentEnvironment=environment());
    #     if(intermittent=="a"){
    #         if(Etype=="M"){
    #             Etype <- "A";
    #         }
    #         if(Ttype=="M"){
    #             Ttype <- "A";
    #         }
    #         if(Stype=="M"){
    #             Stype <- "A";
    #         }
    #     }
    # }
    # else{
    #     intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
    #     intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    # }
    esValues <- CreatorVES(silent=silentText);

##### If intermittent=="a", run a loop and select the best one #####
#     if(intermittent=="a"){
#         Etype <- EtypeOriginal;
#         Ttype <- TtypeOriginal;
#         Stype <- StypeOriginal;
#         if(cfType!="MSE"){
#             warning(paste0("'",cfType,"' is used as cost function instead of 'MSE'. A wrong intermittent model may be selected"),call.=FALSE);
#         }
#         if(!silentText){
#             cat("Selecting appropriate type of intermittency... ");
#         }
# # Prepare stuff for intermittency selection
#         intermittentModelsPool <- c("n","f","c","t","s");
#         intermittentCFs <- intermittentICs <- rep(NA,length(intermittentModelsPool));
#         intermittentModelsList <- list(NA);
#         intermittentICs[1] <- esValues$icBest;
#         intermittentCFs[1] <- esValues$cfObjective;
#
#         for(i in 2:length(intermittentModelsPool)){
#             intermittentParametersSetter(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
#             intermittentMaker(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
#             intermittentModelsList[[i]] <- CreatorVES(silent=TRUE);
#             intermittentICs[i] <- intermittentModelsList[[i]]$icBest;
#             intermittentCFs[i] <- intermittentModelsList[[i]]$cfObjective;
#         }
#         intermittentICs[is.nan(intermittentICs) | is.na(intermittentICs)] <- 1e+100;
#         intermittentCFs[is.nan(intermittentCFs) | is.na(intermittentCFs)] <- 1e+100;
#         # In cases when the data is binary, choose between intermittent models only
#         if(any(intermittentCFs==0)){
#             if(all(intermittentCFs[2:length(intermittentModelsPool)]==0)){
#                 intermittentICs[1] <- Inf;
#             }
#         }
#         iBest <- which(intermittentICs==min(intermittentICs))[1];
#
#         if(!silentText){
#             cat("Done!\n");
#         }
#         if(iBest!=1){
#             intermittent <- intermittentModelsPool[iBest];
#             intermittentModel <- intermittentModelsList[[iBest]];
#             esValues <- intermittentModelsList[[iBest]];
#         }
#         else{
#             intermittent <- "n"
#         }
#
#         intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
#         intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
#     }

##### Fit the model and produce forecast #####
        list2env(esValues,environment());
        BasicMakerVES(ParentEnvironment=environment());

        # if(!is.null(xregNames)){
        #     matat <- as.matrix(matatOriginal[,xregNames]);
        #     matxt <- as.matrix(matxtOriginal[,xregNames]);
        #     if(ncol(matat)==1){
        #         colnames(matxt) <- xregNames;
        #     }
        #     xreg <- matxt;
        # }
        # else{
        xreg <- NULL;
        # }
        BasicInitialiserVES(ParentEnvironment=environment());

        if(damped){
            model <- paste0(Etype,Ttype,"d",Stype);
        }
        else{
            model <- paste0(Etype,Ttype,Stype);
        }

# Write down Fisher Information if needed
        if(FI){
            environment(likelihoodFunction) <- environment();
            FI <- numDeriv::hessian(likelihoodFunction,A);
        }

        ##### These functions need to be rewrittent #####
        vssFitter(ParentEnvironment=environment());
        vssForecaster(ParentEnvironment=environment());

        component.names <- "level";
        if(Ttype!="N"){
            component.names <- c(component.names,"trend");
        }
        if(Stype!="N"){
            component.names <- c(component.names,"seasonality");
        }

        colnames(matvt) <- c(component.names);

# Write down the initials. Done especially for Nikos and issue #10
        if(persistenceEstimate){
            persistence <- as.vector(matG);
        }
        if(Ttype!="N"){
            names(persistence) <- c("alpha","beta","gamma")[1:nComponents];
        }
        else{
            names(persistence) <- c("alpha","gamma")[1:nComponents];
        }

        if(initialType!="p"){
            initialValue <- matvt[maxlag,1:(nComponents - (Stype!="N"))];
        }

        if(initialXEstimate){
            initialX <- matat[1,];
            names(initialX) <- colnames(matat);
        }

        if(initialSeasonEstimate){
            if(Stype!="N"){
                initialSeason <- matvt[1:maxlag,nComponents];
                names(initialSeason) <- paste0("s",1:maxlag);
            }
        }

# Write down the formula of ETS
        # esFormula <- "l[t-1]";
        # if(Ttype=="A"){
        #     esFormula <- paste0(esFormula," + b[t-1]");
        # }
        # else if(Ttype=="M"){
        #     esFormula <- paste0(esFormula," * b[t-1]");
        # }
        # if(Stype=="A"){
        #     esFormula <- paste0(esFormula," + s[t-",maxlag,"]");
        # }
        # else if(Stype=="M"){
        #     if(Ttype=="A"){
        #         esFormula <- paste0("(",esFormula,")");
        #     }
        #     esFormula <- paste0(esFormula," * s[t-",maxlag,"]");
        # }
        # if(Etype=="A"){
        #     if(!is.null(xreg)){
        #         if(updateX){
        #             esFormula <- paste0(esFormula," + ",paste0(paste0("a",c(1:nExovars),"[t-1] * "),paste0(xregNames,"[t]"),collapse=" + "));
        #         }
        #         else{
        #             esFormula <- paste0(esFormula," + ",paste0(paste0("a",c(1:nExovars)," * "),paste0(xregNames,"[t]"),collapse=" + "));
        #         }
        #     }
        #     esFormula <- paste0(esFormula," + e[t]");
        # }
        # else{
        #     if(any(c(Ttype,Stype)=="A") & Stype!="M"){
        #         esFormula <- paste0("(",esFormula,")");
        #     }
        #     if(!is.null(xreg)){
        #         if(updateX){
        #             esFormula <- paste0(esFormula," * exp(",paste0(paste0("a",c(1:nExovars),"[t-1] * "),paste0(xregNames,"[t]"),collapse=" + "),")");
        #         }
        #         else{
        #             esFormula <- paste0(esFormula," * exp(",paste0(paste0("a",c(1:nExovars)," * "),paste0(xregNames,"[t]"),collapse=" + "),")");
        #         }
        #     }
        #     esFormula <- paste0(esFormula," * e[t]");
        # }
        # if(intermittent!="n"){
        #     esFormula <- paste0("o[t] * (",esFormula,")");
        # }
        # esFormula <- paste0("y[t] = ",esFormula);

##### Do final check and make some preparations for output #####

    # Write down the probabilities from intermittent models
    # pt <- ts(c(as.vector(pt),as.vector(pt.for)),start=start(data),frequency=datafreq);
    # if(intermittent=="f"){
    #     intermittent <- "fixed";
    # }
    # else if(intermittent=="c"){
    #     intermittent <- "croston";
    # }
    # else if(intermittent=="t"){
    #     intermittent <- "tsb";
    # }
    # else if(intermittent=="n"){
    #     intermittent <- "none";
    # }
    # else if(intermittent=="p"){
    #     intermittent <- "provided";
    # }

##### Now let's deal with the holdout #####
    if(holdout){
        y.holdout <- ts(data[(obsInsample+1):obsAll],start=start(y.for),frequency=frequency(data));
        errormeasures <- NA;
        # errormeasures <- errorMeasurer(y.holdout,y.for,y);

# Add PLS
        # errormeasuresNames <- names(errormeasures);
        # if(all(intermittent!=c("n","none"))){
        #     errormeasures <- c(errormeasures, suppressWarnings(pls(actuals=y.holdout, forecasts=y.for, Etype=Etype,
        #                                                            sigma=s2, trace=FALSE, iprob=pt[obsInsample+c(1:h)])));
        # }
        # else{
        #     if(multisteps){
        #         sigma <- t(errors.mat) %*% errors.mat / obsInsample;
        #     }
        #     else{
        #         sigma <- s2;
        #     }
        #     errormeasures <- c(errormeasures, suppressWarnings(pls(actuals=y.holdout, forecasts=y.for, Etype=Etype,
        #                                                            sigma=sigma, trace=multisteps, iprob=pt[obsInsample+c(1:h)])));
        # }
        # names(errormeasures) <- c(errormeasuresNames,"PLS");
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

    if(!is.null(xreg)){
        modelname <- "VESX";
    }
    else{
        modelname <- "VES";
    }
    modelname <- paste0(modelname,"(",model,")");
    # if(all(intermittent!=c("n","none"))){
    #     modelname <- paste0("i",modelname);
    # }

##### Print output #####
    if(!silentText){
        if(any(abs(eigen(matF - matG %*% matw)$values)>(1 + 1E-10))){
            warning(paste0("Model ETS(",model,") is unstable! Use a different value of 'bounds' parameter to address this issue!"),
                    call.=FALSE);
        }
    }

##### Make a plot #####
    # if(!silentGraph){
    #     if(intervals){
    #         graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
    #                    level=level,legend=!silentLegend,main=modelname);
    #     }
    #     else{
    #         graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
    #                 level=level,legend=!silentLegend,main=modelname);
    #     }
    # }

    ##### Return values #####
    model <- list(model=modelname,timeElapsed=Sys.time()-startTime,
                  states=matvt,persistence=persistence,
                  initialType=initialType,initial=initialValue,initialSeason=initialSeason,
                  nParam=nParam,
                  fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,
                  errors=errors.mat,s2=s2,intervals=intervalsType,level=level,
                  actuals=data,holdout=y.holdout,iprob=pt,intermittent=intermittent,
                  xreg=xreg,updateX=updateX,initialX=initialX,persistenceX=matGX,transitionX=matFX,
                  ICs=ICs,logLik=logLik,cf=cfObjective,cfType=cfType,FI=FI,accuracy=errormeasures);
    return(structure(model,class="smooth"));
}
