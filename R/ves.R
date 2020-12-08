utils::globalVariables(c("nParamMax","nComponentsAll","nComponentsNonSeasonal","nSeries","modelIsSeasonal",
                         "obsInSample","obsAll","lagsModel","persistenceEstimate","persistenceType",
                         "persistenceValue","damped","dampedEstimate","dampedType","transitionType",
                         "initialEstimate","initialSeasonEstimate","initialSeasonValue","initialSeasonType",
                         "modelIsMultiplicative","matG","matW","B","ub","lb", "maxeval", "algorithm1",
                         "algorithm2", "xtol_rel1", "xtol_rel2", "Sigma","yFitted","PI","dataDeltat",
                         "dataFreq","dataStart","otObs","dataNames","seasonalType"));

#' Vector Exponential Smoothing in SSOE state space model
#'
#' Function constructs vector ETS model and returns forecast, fitted values, errors
#' and matrix of states along with other useful variables.
#'
#' Function estimates vector ETS in a form of the Single Source of Error state space
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
#' For the details on the additive model see Hyndman et al. (2008),
#' chapter 17.
#'
#' In case of multiplicative model, instead of the vector y_t we use its logarithms.
#' As a result the multiplicative model is much easier to work with.
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("ves","smooth")}
#'
#' @template vssBasicParam
#' @template vssAdvancedParam
#' @template vssIntervals
#' @template ssAuthor
#' @template vssKeywords
#'
#' @template vssGeneralRef
#'
#' @param model The type of ETS model. Can consist of 3 or 4 chars: \code{ANN},
#' \code{AAN}, \code{AAdN}, \code{AAA}, \code{AAdA}, \code{MMdM} etc.
#' \code{ZZZ} means that the model will be selected based on the chosen
#' information criteria type.
#' ATTENTION! ONLY PURE ADDITIVE AND PURE MULTIPLICATIVE MODELS ARE CURRENTLY
#' AVAILABLE + NO MODEL SELECTION IS AVAILABLE AT THIS STAGE!
#' Pure multiplicative models are done as additive model applied to log(data).
#'
#' Also \code{model} can accept a previously estimated VES model and use all its
#' parameters.
#'
#' Keep in mind that model selection with "Z" components uses Branch and Bound
#' algorithm and may skip some models that could have slightly smaller
#' information criteria.

#' @param phi In cases of damped trend this parameter defines whether the \eqn{phi}
#' should be estimated separately for each series (\code{"individual"}) or for the whole
#' set (\code{"common"}). If vector or a value is provided here, then it is used by the
#' model.
#' @param initial Can be either character or a vector / matrix of initial states.
#' If it is character, then it can be \code{"individual"}, individual values of
#' the initial non-seasonal components are used, or \code{"common"}, meaning that
#' the initials for all the time series are set to be equal to the same value.
#' If vector of states is provided, then it is automatically transformed into
#' a matrix, assuming that these values are provided for the whole group.
#' @param initialSeason Can be either character or a vector / matrix of initial
#' states. Treated the same way as \code{initial}. This means that different time
#' series may share the same initial seasonal component.
#' @param seasonal The type of seasonal component across the series. Can be
#' \code{"individual"}, so that each series has its own component or \code{"common"},
#' so that the component is shared across the series.
#' @param weights The weights for the errors between the series with the common
#' seasonal component. Ignored if \code{seasonal="individual"}.
#' @param ...  Other non-documented parameters. For example \code{FI=TRUE} will
#' make the function also produce Fisher Information matrix, which then can be
#' used to calculated variances of smoothing parameters and initial states of
#' the model. The vector of initial parameter for the optimiser can be provided
#' here as the variable \code{B}. The upper bound for the optimiser is provided
#' via \code{ub}, while the lower one is \code{lb}. \code{maxeval=1000} is the
#' default number of iterations for both optimisers used in the function.
#' \code{algorithm1="NLOPT_LN_BOBYQA"} is the algorithm used in the first optimiser,
#' while \code{algorithm2="NLOPT_LN_NELDERMEAD"} is the second one. \code{xtol_rel1=1e-8}
#' is the relative tolerance in the first optimiser, while \code{xtol_rel2=1e-6} is for
#' the second one. All of this can be amended and passed in ellipsis for finer tuning.
#' @return Object of class "vsmooth" is returned. It contains the following list of
#' values:
#' \itemize{
#' \item \code{model} - The name of the fitted model;
#' \item \code{timeElapsed} - The time elapsed for the construction of the model;
#' \item \code{states} - The matrix of states with components in columns and time in rows;
#' \item \code{persistence} - The persistence matrix;
#' \item \code{transition} - The transition matrix;
#' \item \code{measurement} - The measurement matrix;
#' \item \code{phi} - The damping parameter value;
#' \item \code{B} - The vector of all the estimated coefficients;
#' \item \code{initial} - The initial values of the non-seasonal components;
#' \item \code{initialSeason} - The initial values of the seasonal components;
#' \item \code{nParam} - The number of estimated parameters;
#' \item \code{imodel} - The intermittent model estimated with VES;
#' \item \code{y} - The matrix with the original data;
#' \item \code{fitted} - The matrix of the fitted values;
#' \item \code{holdout} - The matrix with the holdout values (if \code{holdout=TRUE} in
#' the estimation);
#' \item \code{residuals} - The matrix of the residuals of the model;
#' \item \code{Sigma} - The covariance matrix of the errors (estimated with the correction
#' for the number of degrees of freedom);
#' \item \code{forecast} - The matrix of point forecasts;
#' \item \code{PI} - The bounds of the prediction interval;
#' \item \code{interval} - The type of the constructed prediction interval;
#' \item \code{level} - The level of the confidence for the prediction interval;
#' \item \code{ICs} - The values of the information criteria;
#' \item \code{logLik} - The log-likelihood function;
#' \item \code{lossValue} - The value of the loss function;
#' \item \code{loss} - The type of the used loss function;
#' \item \code{accuracy} - the values of the error measures. Currently not available.
#' \item \code{FI} - Fisher information if user asked for it using \code{FI=TRUE}.
#' }
#' @seealso \code{\link[smooth]{es}, \link[forecast]{ets}}
#'
#' @examples
#'
#' Y <- ts(cbind(rnorm(100,100,10),rnorm(100,75,8)),frequency=12)
#'
#' # The simplest model applied to the data with the default values
#' ves(Y,model="ANN",h=10,holdout=TRUE)
#'
#' # Damped trend model with the dependent persistence
#' ves(Y,model="AAdN",persistence="d",h=10,holdout=TRUE)
#'
#' # Multiplicative damped trend model with individual phi
#' ves(Y,model="MMdM",persistence="i",h=10,holdout=TRUE,initialSeason="c")
#'
#' Y <- cbind(c(rpois(25,0.1),rpois(25,0.5),rpois(25,1),rpois(25,5)),
#'            c(rpois(25,0.1),rpois(25,0.5),rpois(25,1),rpois(25,5)))
#'
#' # Intermittent VES with logistic probability
#' ves(Y,model="MNN",h=10,holdout=TRUE,intermittent="l")
#'
#' @export
ves <- function(y, model="ANN", persistence=c("common","individual","dependent","seasonal-common"),
                transition=c("common","individual","dependent"), phi=c("common","individual"),
                initial=c("individual","common"), initialSeason=c("common","individual"),
                seasonal=c("individual","common"), weights=rep(1/ncol(y),ncol(y)),
                loss=c("likelihood","diagonal","trace"),
                ic=c("AICc","AIC","BIC","BICc"), h=10, holdout=FALSE,
                interval=c("none","conditional","unconditional","individual","likelihood"), level=0.95,
                cumulative=FALSE,
                intermittent=c("none","fixed","logistic"), imodel="ANN",
                iprobability=c("dependent","independent"),
                bounds=c("admissible","usual","none"),
                silent=c("all","graph","output","none"), ...){
# Copyright (C) 2017 - Inf  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

    # Check if the old value of parameters are passed
    if(any(c("group","g") %in% c(persistence, transition, phi, initial, initialSeason))){
        warning("You are using the old value of the parameters. We now have 'common' instead of 'group'.");
        persistence[c("group","g") %in% persistence] <- "common";
        transition[c("group","g") %in% transition] <- "common";
        phi[c("group","g") %in% phi] <- "common";
        initial[c("group","g") %in% initial] <- "common";
        initialSeason[c("group","g") %in% initialSeason] <- "common";
    }
    if(any("independent" %in% c(persistence, transition, phi, initial, initialSeason))){
        warning("You are using the old value of the parameters. We now have 'common' instead of 'group'.");
        persistence["independent" %in% persistence] <- "individual";
        transition["independent" %in% transition] <- "individual";
        phi["independent" %in% phi] <- "individual";
        initial["independent" %in% initial] <- "individual";
        initialSeason["independent" %in% initialSeason] <- "individual";
        interval["independent" %in% interval] <- "individual";
    }
    ##### Up until here

# If a previous model provided as a model, write down the variables
    if(any(is.vsmooth(model))){
        if(smoothType(model)!="VES"){
            stop("The provided model is not VES.",call.=FALSE);
        }
        persistence <- model$persistence;
        transition <- model$transition;
        phi <- model$phi;
        measurement <- model$measurement;
        initial <- model$initial;
        initialSeason <- model$initialSeason;
        # nParamOriginal <- model$nParam;
        # if(is.null(xreg)){
        #     xreg <- model$xreg;
        # }
        # initialX <- model$initialX;
        # persistenceX <- model$persistenceX;
        # transitionX <- model$transitionX;
        # if(any(c(persistenceX)!=0) | any((transitionX!=0)&(transitionX!=1))){
        #     updateX <- TRUE;
        # }
        model <- modelType(model);
    }
    # else{
        # nParamOriginal <- NULL;
    # }

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

##### Set environment for vssInput and make all the checks #####
    environment(vssInput) <- environment();
    vssInput("ves",ParentEnvironment=environment());

##### Cost Function for VES #####
CF <- function(B){
    elements <- BasicInitialiserVES(matvt, matF, matG, matW, B);

    # Check the bounds
    if(bounds=="a"){
        eigenValues <- eigen(elements$matF - elements$matG %*% elements$matW, only.values=TRUE, symmetric=TRUE)$values;
        if(max(abs(eigenValues)>(1 + 1E-50))){
            return(max(abs(eigenValues))*1E+100);
        }
    }

    # Fit the model
    fitting <- vFitterWrap(yInSample, elements$matvt, elements$matF, elements$matW, elements$matG,
                           lagsModel, Etype, Ttype, Stype, ot);

    # Calculate the loss
    if(loss=="l"){
        cfRes <- suppressWarnings(log(det((fitting$errors / normalizer) %*% t(fitting$errors / normalizer) / otObs)) +
                                      nSeries * log(normalizer^2));
    }
    else if(loss=="d"){
        cfRes <- sum(log(apply(fitting$errors^2, 2, sum) / obsInSample));
    }
    else{
        cfRes <- sum(apply(fitting$errors^2, 2, sum) / obsInSample);
    }

    # cfRes <- vOptimiserWrap(yInSample, elements$matvt, elements$matF, elements$matW, elements$matG,
    #                         lagsModel, Etype, Ttype, Stype, loss, normalizer, bounds, ot, otObs);
    # multisteps, initialType, bounds,

    if(is.nan(cfRes) | is.na(cfRes) | is.infinite(cfRes)){
        cfRes <- 1e+100;
    }

    return(cfRes);
}

##### B values for estimation #####
# Function constructs default bounds where B values should lie
BValues <- function(Ttype,Stype,lagsModelMax,nComponentsAll,nComponentsNonSeasonal,nSeries){
    B <- NA;
    BLower <- NA;
    BUpper <- NA;
    BNames <- NA;

    if(seasonalType=="i"){
        #### Individual seasonality ####
        ### Persistence matrix
        if(persistenceEstimate){
            if(persistenceType=="c"){
                persistenceLength <- nComponentsAll;
            }
            else if(persistenceType=="i"){
                persistenceLength <- nComponentsAll*nSeries;
            }
            else if(persistenceType=="d"){
                persistenceLength <- nComponentsAll*nSeries^2;
            }
            else if(persistenceType=="s"){
                persistenceLength <- (nComponentsAll-1)*nSeries+1;
            }
            B <- c(B,rep(0.1,persistenceLength));
            if(bounds=="u"){
                BLower <- c(BLower,rep(0,persistenceLength));
                BUpper <- c(BUpper,rep(1,persistenceLength));
            }
            else{
                BLower <- c(BLower,rep(-5,persistenceLength));
                BUpper <- c(BUpper,rep(5,persistenceLength));
            }
            BNames <- c(BNames,paste0("Persistence",c(1:persistenceLength)));
        }

        ### Damping parameter
        if(dampedEstimate){
            if(dampedType=="c"){
                dampedLength <- 1;
            }
            else if(dampedType=="i"){
                dampedLength <- nSeries;
            }
            B <- c(B,rep(0.95,dampedLength));
            BLower <- c(BLower,rep(0,dampedLength));
            BUpper <- c(BUpper,rep(1,dampedLength));
            BNames <- c(BNames,paste0("phi",c(1:dampedLength)));
        }

        ### Transition matrix
        if(transitionEstimate){
            if(transitionType=="d"){
                transitionLength <- ((nSeries-1)*nComponentsAll^2)*nSeries;
            }
            B <- c(B,rep(0.1,transitionLength));
            BLower <- c(BLower,rep(-1,transitionLength));
            BUpper <- c(BUpper,rep(1,transitionLength));
            BNames <- c(BNames,paste0("transition",c(1:transitionLength)));
        }

        ### Vector of initials
        if(initialEstimate){
            if(initialType=="c"){
                initialLength <- nComponentsNonSeasonal;
            }
            else{
                initialLength <- nComponentsNonSeasonal*nSeries;
            }
            B <- c(B,initialValue);
            BNames <- c(BNames,paste0("initial",c(1:initialLength)));
            BLower <- c(BLower,rep(-Inf,initialLength));
            BUpper <- c(BUpper,rep(Inf,initialLength));
        }

        ### Vector of initial seasonals
        if(initialSeasonEstimate){
            if(initialSeasonType=="c"){
                initialSeasonLength <- lagsModelMax;
            }
            else{
                initialSeasonLength <- lagsModelMax*nSeries;
            }
            B <- c(B,initialSeasonValue);
            BNames <- c(BNames,paste0("initialSeason",c(1:initialSeasonLength)));
            # if(Stype=="A"){
            BLower <- c(BLower,rep(-Inf,initialSeasonLength));
            BUpper <- c(BUpper,rep(Inf,initialSeasonLength));
            # }
            # else{
            #     BLower <- c(BLower,rep(-0.0001,initialSeasonLength));
            #     BUpper <- c(BUpper,rep(20,initialSeasonLength));
            # }
        }
    }
    else{
        #### Common seasonality ####
        ### Persistence matrix
        if(persistenceEstimate){
            if(persistenceType=="c"){
                persistenceLength <- nComponentsAll;
            }
            else if(persistenceType=="i"){
                persistenceLength <- nComponentsNonSeasonal*nSeries+nSeries;
            }
            else if(persistenceType=="d"){
                persistenceLength <- nComponentsNonSeasonal*nSeries^2+nSeries;
            }
            else if(persistenceType=="s"){
                persistenceLength <- nComponentsNonSeasonal*nSeries+1;
            }
            B <- c(B,rep(0.1,persistenceLength));
            if(bounds=="u"){
                BLower <- c(BLower,rep(0,persistenceLength));
                BUpper <- c(BUpper,rep(1,persistenceLength));
            }
            else{
                BLower <- c(BLower,rep(-5,persistenceLength));
                BUpper <- c(BUpper,rep(5,persistenceLength));
            }
            BNames <- c(BNames,paste0("Persistence",c(1:persistenceLength)));
        }

        ### Damping parameter
        if(dampedEstimate){
            if(dampedType=="c"){
                dampedLength <- 1;
            }
            else if(dampedType=="i"){
                dampedLength <- nSeries;
            }
            B <- c(B,rep(0.95,dampedLength));
            BLower <- c(BLower,rep(0,dampedLength));
            BUpper <- c(BUpper,rep(1,dampedLength));
            BNames <- c(BNames,paste0("phi",c(1:dampedLength)));
        }

        ### Transition matrix
        if(transitionEstimate){
            if(transitionType=="d"){
                transitionLength <- ((nSeries-1)*nComponentsNonSeasonal^2)*nSeries;
            }
            B <- c(B,rep(0.1,transitionLength));
            BLower <- c(BLower,rep(-1,transitionLength));
            BUpper <- c(BUpper,rep(1,transitionLength));
            BNames <- c(BNames,paste0("transition",c(1:transitionLength)));
        }

        ### Vector of initials
        if(initialEstimate){
            if(initialType=="c"){
                initialLength <- nComponentsNonSeasonal;
            }
            else{
                initialLength <- nComponentsNonSeasonal*nSeries;
            }
            B <- c(B,initialValue);
            BNames <- c(BNames,paste0("initial",c(1:initialLength)));
            BLower <- c(BLower,rep(-Inf,initialLength));
            BUpper <- c(BUpper,rep(Inf,initialLength));
        }

        ### Vector of initial seasonals
        if(initialSeasonEstimate){
            initialSeasonLength <- lagsModelMax;
            B <- c(B,initialSeasonValue);
            BNames <- c(BNames,paste0("initialSeason",c(1:initialSeasonLength)));
            BLower <- c(BLower,rep(-Inf,initialSeasonLength));
            BUpper <- c(BUpper,rep(Inf,initialSeasonLength));
        }
    }

    B <- B[!is.na(B)];
    BLower <- BLower[!is.na(BLower)];
    BUpper <- BUpper[!is.na(BUpper)];
    BNames <- BNames[!is.na(BNames)];

    return(list(B=B,BLower=BLower,BUpper=BUpper,BNames=BNames));
}

##### Basic VES initialiser
### This function will accept Etype, Ttype, Stype and damped and would return:
# nComponentsNonSeasonal, nComponentsAll, lagsModelMax, modelIsSeasonal, obsStates
# This is needed for model selection

##### Basic matrices creator #####
    # This thing returns matvt, matF, matG, matW, dampedValue, initialValue
    # and initialSeasonValue if they are not provided + lagsModel
BasicMakerVES <- function(...){
    # ellipsis <- list(...);
    # ParentEnvironment <- ellipsis[['ParentEnvironment']];

    ### Persistence matrix
    matG <- switch(seasonalType,
                   "i" =  matrix(0,nSeries*nComponentsAll,nSeries),
                   "c" = matrix(0,nSeries*nComponentsNonSeasonal+1,nSeries));
    if(!persistenceEstimate){
        matG <- persistenceValue;
    }

    ### Damping parameter
    if(!damped){
        dampedValue <- matrix(1,nSeries,1);
    }

    ### Transition matrix
    if(any(transitionType==c("c","i","d"))){
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
    matF <- switch(seasonalType,
                   "i"=transitionValue,
                   "c"=rbind(cbind(transitionValue[-(c(1:nSeries)*nComponentsAll),
                                                   -(c(1:nSeries)*nComponentsAll)],
                                   0),
                             c(transitionValue[nComponentsAll*nSeries,
                                               -(c(1:nSeries)*nComponentsAll)],1)));

    ### Measurement matrix
    if(seasonalType=="i"){
        matW <- matrix(0,nSeries,nSeries*nComponentsAll);
        for(i in 1:nSeries){
            matW[i,c(1:nComponentsAll)+nComponentsAll*(i-1)] <- 1;
        }
        if(damped){
            for(i in 1:nSeries){
                matW[i,nComponentsNonSeasonal+nComponentsAll*(i-1)] <- dampedValue[i];
            }
        }
    }
    else{
        matW <- matrix(0,nSeries,nSeries*nComponentsNonSeasonal+1);
        for(i in 1:nSeries){
            matW[i,c(1:nComponentsNonSeasonal)+nComponentsNonSeasonal*(i-1)] <- 1;
        }
        matW[,nSeries*nComponentsNonSeasonal+1] <- 1;
        if(damped){
            for(i in 1:nSeries){
                matW[i,nComponentsNonSeasonal+nComponentsNonSeasonal*(i-1)] <- dampedValue[i];
            }
        }
    }

    ### Vector of states
    statesNames <- "level";
    if(Ttype!="N"){
        statesNames <- c(statesNames,"trend");
    }
    if(Stype!="N"){
        statesNames <- c(statesNames,"seasonal");
    }
    matvt <- matrix(NA, nComponentsAll*nSeries, obsStates,
                    dimnames=list(paste0(rep(dataNames,each=nComponentsAll),
                                         "_",statesNames),NULL));
    if(seasonalType=="c"){
        matvt <- rbind(matvt[-(c(1:nSeries)*nComponentsAll),,drop=F],
                       matvt[nComponentsAll,,drop=F]);
        rownames(matvt)[nComponentsNonSeasonal*nSeries+1] <- "seasonal";
    }

    ## Deal with non-seasonal part of the vector of states
    if(!initialEstimate){
        if(seasonalType=="i"){
            initialPlaces <- nComponentsAll*(c(1:nSeries)-1)+1;
            if(Ttype!="N"){
                initialPlaces <- c(initialPlaces,nComponentsAll*(c(1:nSeries)-1)+2);
                initialPlaces <- sort(initialPlaces);
            }
        }
        else{
            initialPlaces <- c(1:(nSeries*nComponentsNonSeasonal));
        }
        matvt[initialPlaces,1:lagsModelMax] <- rep(initialValue,lagsModelMax);
    }
    else{
        XValues <- rbind(rep(1,obsInSample),c(1:obsInSample));
        initialValue <- yInSample %*% t(XValues) %*% solve(XValues %*% t(XValues));
        if(Etype=="L"){
            initialValue[,1] <- (initialValue[,1] - 0.5) * 20;
        }

        if(Ttype=="N"){
            initialValue <- matrix(initialValue[,-2],nSeries,1);
        }
        if(initialType=="c"){
            initialValue <- matrix(colMeans(initialValue),nComponentsNonSeasonal,1);
        }
        else{
            initialValue <- matrix(as.vector(t(initialValue)),nComponentsNonSeasonal * nSeries,1);
        }
    }

    ## Deal with seasonal part of the vector of states
    if(modelIsSeasonal){
        if(initialSeasonType=="p"){
            if(seasonalType=="i"){
                initialPlaces <- nComponentsAll*(c(1:nSeries)-1)+nComponentsAll;
            }
            else{
                initialPlaces <- nSeries*nComponentsNonSeasonal+1;
            }
            matvt[initialPlaces,1:lagsModelMax] <- initialSeasonValue;
        }
        else{
            # Matrix of dummies for seasons
            XValues <- matrix(rep(diag(lagsModelMax),ceiling(obsInSample/lagsModelMax)),lagsModelMax)[,1:obsInSample];
            # if(Stype=="A"){
                initialSeasonValue <- (yInSample-rowMeans(yInSample)) %*% t(XValues) %*% solve(XValues %*% t(XValues));
            # }
            # else{
            #     initialSeasonValue <- (yInSample-rowMeans(yInSample)) %*% t(XValues) %*% solve(XValues %*% t(XValues));
            # }
            if(initialSeasonType=="c" || seasonalType=="c"){
                initialSeasonValue <- matrix(colMeans(initialSeasonValue),1,lagsModelMax);
            }
            else{
                initialSeasonValue <- matrix(as.vector(t(initialSeasonValue)),nSeries,lagsModelMax);
            }
        }
    }

    ### lagsModel
    if(seasonalType=="i"){
        lagsModel <- rep(1,nComponentsAll);
        if(modelIsSeasonal){
            lagsModel[nComponentsAll] <- lagsModelMax;
        }
        lagsModel <- matrix(lagsModel,nSeries*nComponentsAll,1);
    }
    else{
        lagsModel <- matrix(c(rep(1,nSeries*nComponentsNonSeasonal),lagsModelMax),
                            nSeries*nComponentsNonSeasonal+1,1);
    }

    return(list(matvt=matvt, matF=matF, matG=matG, matW=matW, dampedValue=dampedValue,
                initialValue=initialValue, initialSeasonValue=initialSeasonValue, lagsModel=lagsModel));
}

##### Basic matrices filler #####
# This thing fills in matvt, matF, matG and matW with values from B and returns the corrected values
BasicInitialiserVES <- function(matvt,matF,matG,matW,B){
    nCoefficients <- 0;
    ##### Individual seasonality #####
    if(seasonalType=="i"){
        ### Persistence matrix
        if(persistenceEstimate){
            persistenceBuffer <- matrix(0,nSeries*nComponentsAll,nSeries);
            # Grouped values
            if(persistenceType=="c"){
                persistenceValue <- B[1:nComponentsAll];
                nCoefficients <- nComponentsAll;
                for(i in 1:nSeries){
                    persistenceBuffer[1:nComponentsAll+nComponentsAll*(i-1),i] <- persistenceValue;
                }
                persistenceValue <- persistenceBuffer;
            }
            # Independent values
            else if(persistenceType=="i"){
                persistenceValue <- B[1:(nComponentsAll*nSeries)];
                nCoefficients <- nComponentsAll*nSeries;
                for(i in 1:nSeries){
                    persistenceBuffer[1:nComponentsAll+nComponentsAll*(i-1),
                                      i] <- persistenceValue[1:nComponentsAll+nComponentsAll*(i-1)];
                }
                persistenceValue <- persistenceBuffer;
            }
            # Dependent values
            else if(persistenceType=="d"){
                persistenceValue <- B[1:(nComponentsAll*nSeries^2)];
                nCoefficients <- nComponentsAll*nSeries^2;
            }
            # Grouped seasonal values
            else if(persistenceType=="s"){
                persistenceValue <- B[1:((nComponentsAll-1)*nSeries+1)];
                persistenceSeasonal <- persistenceValue[length(persistenceValue)];
                nCoefficients <- ((nComponentsAll-1)*nSeries+1);
                for(i in 1:nSeries){
                    persistenceBuffer[1:(nComponentsAll-1)+nComponentsAll*(i-1),
                                      i] <- persistenceValue[1:(nComponentsAll-1)+(nComponentsAll-1)*(i-1)];
                    persistenceBuffer[nComponentsAll+nComponentsAll*(i-1),i] <- persistenceSeasonal;
                }
                persistenceValue <- persistenceBuffer;
            }
            matG[,] <- persistenceValue;
        }

        ### Damping parameter
        if(damped){
            if(dampedType=="c"){
                dampedValue <- matrix(B[nCoefficients+1],nSeries,1);
                nCoefficients <- nCoefficients + 1;
            }
            else if(dampedType=="i"){
                dampedValue <- matrix(B[nCoefficients+(1:nSeries)],nSeries,1);
                nCoefficients <- nCoefficients + nSeries;
            }
        }

        ### Transition matrix
        if(any(transitionType==c("i","d","c")) & damped){
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
                     setdiff(c(1:(nSeries*nComponentsAll)),
                             c(1:nComponentsAll)+nComponentsAll*(i-1))] <- B[nCoefficients+c(1:nCoefficientsBuffer)];
                nCoefficients <- nCoefficients + nCoefficientsBuffer;
            }
        }

        ### Measurement matrix
        # Needs to be filled in with dampedValue even if dampedValue has been provided by a user
        if(damped){
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
                initialValue <- matrix(B[nCoefficients+c(1:(nComponentsNonSeasonal*nSeries))],
                                       nComponentsNonSeasonal * nSeries,1);
                nCoefficients <- nCoefficients + nComponentsNonSeasonal*nSeries;
            }
            else if(initialType=="c"){
                initialValue <- matrix(B[nCoefficients+c(1:nComponentsNonSeasonal)],
                                       nComponentsNonSeasonal * nSeries,1);
                nCoefficients <- nCoefficients + nComponentsNonSeasonal;
            }
            matvt[initialPlaces,1:lagsModelMax] <- rep(initialValue,lagsModelMax);
        }

        ## Deal with seasonal part of the vector of states
        if(modelIsSeasonal & initialSeasonEstimate){
            initialPlaces <- nComponentsAll*(c(1:nSeries)-1)+nComponentsAll;
            if(initialSeasonType=="i"){
                matvt[initialPlaces,1:lagsModelMax] <- matrix(B[nCoefficients+c(1:(nSeries*lagsModelMax))],
                                                        nSeries,lagsModelMax,byrow=TRUE);
                nCoefficients <- nCoefficients + nSeries*lagsModelMax;
            }
            else if(initialSeasonType=="c"){
                matvt[initialPlaces,1:lagsModelMax] <- matrix(B[nCoefficients+c(1:lagsModelMax)],nSeries,lagsModelMax,byrow=TRUE);
                nCoefficients <- nCoefficients + lagsModelMax;
            }
        }
    }
    ##### Common seasonality #####
    else{
        ### Persistence matrix
        if(persistenceEstimate){
            persistenceBuffer <- matrix(0,nSeries*nComponentsNonSeasonal+1,nSeries);
            # Grouped values
            if(persistenceType=="c"){
                persistenceValue <- B[1:nComponentsAll];
                nCoefficients <- nComponentsAll;
                for(i in 1:nSeries){
                    persistenceBuffer[1:nComponentsNonSeasonal+nComponentsNonSeasonal*(i-1),
                                      i] <- persistenceValue[1:nComponentsNonSeasonal];
                }
                persistenceBuffer[nSeries*nComponentsNonSeasonal+1,] <- weights*persistenceValue[nComponentsAll];
                persistenceValue <- persistenceBuffer;
            }
            # Independent values
            else if(persistenceType=="i"){
                persistenceValue <- B[1:(nComponentsNonSeasonal*nSeries+nSeries)];
                nCoefficients <- nComponentsNonSeasonal*nSeries+nSeries;
                for(i in 1:nSeries){
                    persistenceBuffer[1:nComponentsNonSeasonal+nComponentsNonSeasonal*(i-1),
                                      i] <- persistenceValue[1:nComponentsNonSeasonal+nComponentsNonSeasonal*(i-1)];
                }
                persistenceBuffer[nSeries*nComponentsNonSeasonal+1,
                                  ] <- weights*persistenceValue[nComponentsNonSeasonal*nSeries+c(1:nSeries)];
                persistenceValue <- persistenceBuffer;
            }
            # Dependent values
            else if(persistenceType=="d"){
                persistenceValue <- B[1:(nSeries^2*nComponentsNonSeasonal+nSeries)];
                nCoefficients <- nSeries^2*nComponentsNonSeasonal+nSeries;
            }
            # Grouped seasonal values
            else if(persistenceType=="s"){
                persistenceValue <- B[1:(nComponentsNonSeasonal*nSeries+1)];
                nCoefficients <- nComponentsNonSeasonal*nSeries+1;
                for(i in 1:nSeries){
                    persistenceBuffer[1:nComponentsNonSeasonal+nComponentsNonSeasonal*(i-1),
                                      i] <- persistenceValue[1:nComponentsNonSeasonal+nComponentsNonSeasonal*(i-1)];
                }
                persistenceBuffer[nSeries*nComponentsNonSeasonal+1,
                                  ] <- weights*persistenceValue[nComponentsNonSeasonal*nSeries+1];
                persistenceValue <- persistenceBuffer;
            }
            matG[,] <- persistenceValue;
        }

        ### Damping parameter
        if(damped){
            if(dampedType=="c"){
                dampedValue <- matrix(B[nCoefficients+1],nSeries,1);
                nCoefficients <- nCoefficients + 1;
            }
            else if(dampedType=="i"){
                dampedValue <- matrix(B[nCoefficients+(1:nSeries)],nSeries,1);
                nCoefficients <- nCoefficients + nSeries;
            }
        }

        ### Transition matrix
        if(any(transitionType==c("i","d","c")) & damped){
            for(i in 1:nSeries){
                matF[c(1:nComponentsNonSeasonal)+nComponentsNonSeasonal*(i-1),
                     nComponentsNonSeasonal+nComponentsNonSeasonal*(i-1)] <- dampedValue[i];
            }
        }
        if(transitionType=="d"){
            # Fill in the other values of F with some values
            nCoefficientsBuffer <- (nSeries-1)*nComponentsNonSeasonal^2;

            for(i in 1:nSeries){
                matF[c(1:nComponentsNonSeasonal)+nComponentsNonSeasonal*(i-1),
                     setdiff(c(1:(nSeries*nComponentsNonSeasonal)),
                             c(1:nComponentsNonSeasonal)+nComponentsNonSeasonal*(i-1))
                     ] <- B[nCoefficients+c(1:nCoefficientsBuffer)];
                nCoefficients <- nCoefficients + nCoefficientsBuffer;
            }
        }

        ### Measurement matrix
        # Needs to be filled in with dampedValue even if dampedValue has been provided by a user
        if(damped){
            for(i in 1:nSeries){
                matW[i,nComponentsNonSeasonal+nComponentsNonSeasonal*(i-1)] <- dampedValue[i];
            }
        }

        ### Vector of states
        ## Deal with non-seasonal part of the vector of states
        if(initialEstimate){
            initialPlaces <- nComponentsNonSeasonal*(c(1:nSeries)-1)+1;
            if(Ttype!="N"){
                initialPlaces <- c(initialPlaces,nComponentsNonSeasonal*(c(1:nSeries)-1)+2);
                initialPlaces <- sort(initialPlaces);
            }
            if(initialType=="i"){
                initialValue <- matrix(B[nCoefficients+c(1:(nComponentsNonSeasonal*nSeries))],
                                       nComponentsNonSeasonal * nSeries,1);
                nCoefficients <- nCoefficients + nComponentsNonSeasonal*nSeries;
            }
            else if(initialType=="c"){
                initialValue <- matrix(B[nCoefficients+c(1:nComponentsNonSeasonal)],nComponentsNonSeasonal * nSeries,1);
                nCoefficients <- nCoefficients + nComponentsNonSeasonal;
            }
            matvt[initialPlaces,1:lagsModelMax] <- rep(initialValue,lagsModelMax);
        }

        ## Deal with seasonal part of the vector of states
        if(modelIsSeasonal & initialSeasonEstimate){
            matvt[nComponentsNonSeasonal*nSeries+1,1:lagsModelMax] <- matrix(B[nCoefficients+c(1:lagsModelMax)],1,lagsModelMax,byrow=TRUE);
            nCoefficients <- nCoefficients + lagsModelMax;
        }
    }

    return(list(matvt=matvt,matF=matF,matG=matG,matW=matW,dampedValue=dampedValue));
}

##### Basic estimation function for ves() #####
EstimatorVES <- function(...){
    environment(BasicMakerVES) <- environment();
    environment(BValues) <- environment();
    environment(vLikelihoodFunction) <- environment();
    environment(vICFunction) <- environment();
    environment(CF) <- environment();
    elements <- BasicMakerVES();
    list2env(elements,environment());

    if(is.null(B) && is.null(ub) && is.null(lb)){
        BList <- BValues(Ttype,Stype,lagsModelMax,nComponentsAll,nComponentsNonSeasonal,nSeries);
        B <- BList$B;

        if(any((B>=BList$BUpper),(B<=BList$BLower))){
            B[B>=BList$BUpper] <- BList$BUpper[B>=BList$BUpper] - 0.01;
            B[B<=BList$BLower] <- BList$BLower[B<=BList$BLower] + 0.01;
        }
    }
    else{
        BList <- BValues(Ttype,Stype,lagsModelMax,nComponentsAll,nComponentsNonSeasonal,nSeries);
        if(is.null(B)){
            B <- BList$B;
        }
        if(!is.null(lb)){
            BList$Blower <- lb;
        }
        if(!is.null(ub)){
            BList$BUpper <- ub;
        }
    }

    normalizer <- sum(colMeans(abs(diff(t(yInSample))),na.rm=TRUE));

    # Parameters are chosen to speed up the optimisation process and have decent accuracy
    res <- nloptr(B, CF, lb=BList$BLower, ub=BList$BUpper,
                  opts=list(algorithm=algorithm1, xtol_rel=xtol_rel1, maxeval=maxeval));
    B <- res$solution;

    # This is just in case something went out of the bounds
    if(any((B>=BList$BUpper),(B<=BList$BLower))){
        BList$BUpper[B>=BList$BUpper] <- B[B>=BList$BUpper] + 1;
        BList$BLower[B<=BList$BLower] <- B[B<=BList$BLower] - 1;
    }

    res2 <- nloptr(B, CF, lb=BList$BLower, ub=BList$BUpper,
                  opts=list(algorithm=algorithm2, xtol_rel=xtol_rel2, maxeval=maxeval));
    # This condition is needed in order to make sure that we did not make the solution worse
    if(res2$objective <= res$objective){
        res <- res2;
    }
    B <- res$solution;

    if(all(B==BList$B) & modelDo=="estimate"){
        if(persistenceEstimate){
            warning(paste0("Failed to optimise the model ETS(", modelCurrent,
                           "). Try different initialisation maybe?\nAnd check all the messages and warnings...",
                           "If you did your best, but the optimiser still fails, report this to the maintainer, please."),
                    call.=FALSE);
        }
    }
    names(B) <- BList$BNames;

    # First part is for the covariance matrix
    if(loss=="l"){
        nParam <- nSeries * (nSeries + 1) / 2 + length(B);
    }
    else{
        nParam <- nSeries + length(B);
    }

    ICValues <- vICFunction(nParam=nParam,B=B,Etype=Etype);
    ICs <- ICValues$ICs;
    logLik <- ICValues$llikelihood;

    # Write down Fisher Information if needed
    if(FI){
        environment(vLikelihoodFunction) <- environment();
        FI <- -numDeriv::hessian(vLikelihoodFunction,B);
        rownames(FI) <- BList$BNames;
        colnames(FI) <- BList$BNames;
    }

    return(list(ICs=ICs,objective=res$objective,B=B,nParam=nParam,logLik=logLik,FI=FI));
}

##### Function constructs the VES function #####
CreatorVES <- function(silent=FALSE,...){
    if(modelDo=="estimate"){
        environment(EstimatorVES) <- environment();
        res <- EstimatorVES(ParentEnvironment=environment());
        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,
                             cfObjective=res$objective,B=res$B,ICs=res$ICs,icBest=res$ICs[ic],
                             nParam=res$nParam,logLik=res$logLik,FI=res$FI);

        return(listToReturn);
    }
    else{
        environment(CF) <- environment();
        environment(vICFunction) <- environment();
        environment(vLikelihoodFunction) <- environment();
        environment(BasicMakerVES) <- environment();
        elements <- BasicMakerVES();
        list2env(elements,environment());

        B <- c(persistenceValue);
        BNames <- paste0("Persistence",c(1:length(persistenceValue)));
        if(damped){
            B <- c(B,dampedValue);
            BNames <- c(BNames,paste0("phi",c(1:length(dampedValue))));
        }
        if(transitionType=="d"){
            transitionLength <- length(B);
            # Write values from the rest of transition matrix
            for(i in 1:nSeries){
                B <- c(B, c(transitionValue[c(1:nComponentsAll)+nComponentsAll*(i-1),
                                            setdiff(c(1:nSeries*nComponentsAll),
                                                    c(1:nComponentsAll)+nComponentsAll*(i-1))]));
            }
            transitionLength <- length(B) - transitionLength;
            BNames <- c(BNames,paste0("transition",c(1:transitionLength)));
        }
        B <- c(B,initialValue);
        BNames <- c(BNames,paste0("initial",c(1:length(initialValue))));
        if(Stype!="N"){
            B <- c(B,initialSeasonValue);
            BNames <- c(BNames,paste0("initialSeason",c(1:length(initialSeasonValue))));
        }
        names(B) <- BNames;

        cfObjective <- CF(B);

        # Number of parameters
        # First part is for the covariance matrix
        if(loss=="l"){
            nParam <- nSeries * (nSeries + 1) / 2;
        }
        else if(loss=="d"){
            nParam <- nSeries;
        }
        else{
            nParam <- nSeries;
        }

        ICValues <- vICFunction(nParam=nParam,B=B,Etype=Etype);
        logLik <- ICValues$llikelihood;
        ICs <- ICValues$ICs;
        icBest <- ICs[ic];

        # Write down Fisher Information if needed
        if(FI){
            environment(vLikelihoodFunction) <- environment();
            FI <- -numDeriv::hessian(vLikelihoodFunction,B);
            rownames(FI) <- BNames;
            colnames(FI) <- BNames;
        }

        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,
                             cfObjective=cfObjective,B=B,ICs=ICs,icBest=icBest,
                             nParam=nParam,logLik=logLik,FI=FI);
        return(listToReturn);
    }
}

##### Preset yFitted, yForecast, errors and basic parameters #####
    yFitted <- matrix(NA,nSeries,obsInSample);
    yForecast <- matrix(NA,nSeries,h);
    errors <- matrix(NA,nSeries,obsInSample);
    rownames(yFitted) <- rownames(yForecast) <- rownames(errors) <- dataNames;

##### Define modelDo #####
    if(any(persistenceEstimate, transitionEstimate, dampedEstimate, initialEstimate, initialSeasonEstimate)){
        modelDo <- "estimate";
        modelCurrent <- model;
    }
    else{
        modelDo <- "nothing";
        modelCurrent <- model;
        bounds <- "n";
    }

##### Now do estimation and model selection #####
    environment(BasicMakerVES) <- environment();
    environment(BasicInitialiserVES) <- environment();
    environment(vssFitter) <- environment();
    environment(vssForecaster) <- environment();

    vesValues <- CreatorVES(silent=silentText);

##### Fit the model and produce forecast #####
    list2env(vesValues,environment());
    list2env(BasicMakerVES(),environment());
    list2env(BasicInitialiserVES(matvt,matF,matG,matW,B),environment());

    if(Etype=="M"){
        cfObjective <- exp(cfObjective);
    }

    if(damped){
        model <- paste0(Etype,Ttype,"d",Stype);
    }
    else{
        model <- paste0(Etype,Ttype,Stype);
    }

    vssFitter(ParentEnvironment=environment());
    vssForecaster(ParentEnvironment=environment());

    ##### Write down persistence, transition, initials etc #####
# Write down the persistenceValue, transitionValue, initialValue, initialSeasonValue

    persistenceNames <- "level";
    if(Ttype!="N"){
        persistenceNames <- c(persistenceNames,"trend");
    }
    if(Stype!="N"){
        persistenceNames <- c(persistenceNames,"seasonal");
    }
    if(persistenceEstimate){
        persistenceValue <- matG;
        if(persistenceType=="c"){
            parametersNumber[1,1] <- parametersNumber[1,1] + nComponentsAll;
        }
        else if(persistenceType=="i"){
            if(seasonalType=="i"){
                parametersNumber[1,1] <- parametersNumber[1,1] + nSeries*nComponentsAll;
            }
            else{
                parametersNumber[1,1] <- parametersNumber[1,1] + nSeries*nComponentsNonSeasonal+nSeries;
            }
        }
        else if(persistenceType=="s"){
            if(seasonalType=="i"){
                parametersNumber[1,1] <- parametersNumber[1,1] + nSeries*(nComponentsAll-1)+1;
            }
            else{
                parametersNumber[1,1] <- parametersNumber[1,1] + nSeries*nComponentsNonSeasonal+1;
            }
        }
        else{
            parametersNumber[1,1] <- parametersNumber[1,1] + length(matG);
        }
    }
    if(seasonalType=="i"){
        rownames(persistenceValue) <- paste0(rep(dataNames,each=nComponentsAll), "_", persistenceNames);
    }
    else{
        rownames(persistenceValue) <- c(paste0(rep(dataNames,each=nComponentsNonSeasonal), "_",
                                               persistenceNames[-nComponentsAll]),
                                        persistenceNames[nComponentsAll]);
    }
    colnames(persistenceValue) <- dataNames;

# This is needed anyway for the reusability of the model
    transitionValue <- matF;
    if(transitionEstimate){
        if(seasonalType=="i"){
            parametersNumber[1,1] <- parametersNumber[1,1] + (nSeries-1)*nSeries*nComponentsAll^2;
        }
        else{
            parametersNumber[1,1] <- parametersNumber[1,1] + (nSeries-1)*nSeries*nComponentsNonSeasonal^2;
        }
    }
    colnames(transitionValue) <- rownames(persistenceValue);
    rownames(transitionValue) <- rownames(persistenceValue);

    if(damped){
        rownames(dampedValue) <- dataNames;
        if(dampedEstimate){
            parametersNumber[1,1] <- parametersNumber[1,1] + length(unique(as.vector(dampedValue)));
        }
    }

    rownames(matW) <- dataNames;
    colnames(matW) <- rownames(persistenceValue);

    if(seasonalType=="i"){
        initialPlaces <- nComponentsAll*(c(1:nSeries)-1)+1;
        initialNames <- "level";
        if(Ttype!="N"){
            initialPlaces <- c(initialPlaces,nComponentsAll*(c(1:nSeries)-1)+2);
            initialPlaces <- sort(initialPlaces);
            initialNames <- c(initialNames,"trend");
        }
        if(initialEstimate){
            initialValue <- matrix(matvt[initialPlaces,lagsModelMax],nComponentsNonSeasonal*nSeries,1);
            parametersNumber[1,1] <- parametersNumber[1,1] + length(unique(as.vector(initialValue)));
        }
    }
    else{
        initialNames <- "level";
        if(Ttype!="N"){
            initialNames <- c(initialNames,"trend");
        }
        if(initialEstimate){
            initialValue <- matrix(matvt[1:(nComponentsNonSeasonal*nSeries),lagsModelMax],
                                   nComponentsNonSeasonal*nSeries,1);
            parametersNumber[1,1] <- parametersNumber[1,1] + length(unique(as.vector(initialValue)));
        }
    }
    rownames(initialValue) <- paste0(rep(dataNames,each=nComponentsNonSeasonal), "_", initialNames);

    if(modelIsSeasonal){
        if(seasonalType=="i"){
            if(initialSeasonEstimate){
                initialPlaces <- nComponentsAll*(c(1:nSeries)-1)+nComponentsAll;
                initialSeasonValue <- matrix(matvt[initialPlaces,1:lagsModelMax],nSeries,lagsModelMax);
                parametersNumber[1,1] <- parametersNumber[1,1] + length(unique(as.vector(initialSeasonValue)));
            }
            rownames(initialSeasonValue) <- dataNames;
        }
        else{
            initialSeasonValue <- matrix(matvt[nComponentsNonSeasonal*nSeries+1,1:lagsModelMax],1,lagsModelMax);
            parametersNumber[1,1] <- parametersNumber[1,1] + lagsModelMax;
            rownames(initialSeasonValue) <- "Common";
        }
        colnames(initialSeasonValue) <- paste0("Seasonal",c(1:lagsModelMax));
    }

    matvt <- ts(t(matvt),start=(time(y)[1] - dataDeltat*lagsModelMax),frequency=dataFreq);
    yFitted <- ts(t(yFitted),start=dataStart,frequency=dataFreq);
    errors <- ts(t(errors),start=dataStart,frequency=dataFreq);

    yForecast <- ts(t(yForecast),start=yForecastStart,frequency=dataFreq);
    if(!is.matrix(yForecast)){
        yForecast <- as.matrix(yForecast,h,nSeries);
    }
    colnames(yForecast) <- dataNames;
    yForecastStart <- start(yForecast)
    if(any(intervalType==c("i","u","l"))){
        PI <-  ts(PI,start=yForecastStart,frequency=dataFreq);
    }

    if(loss=="l"){
        loss <- "likelihood";
        parametersNumber[1,1] <- parametersNumber[1,1] + nSeries * (nSeries + 1) / 2;
    }
    else if(loss=="d"){
        loss <- "diagonal";
        parametersNumber[1,1] <- parametersNumber[1,1] + nSeries;
    }
    else{
        loss <- "trace";
        parametersNumber[1,1] <- parametersNumber[1,1] + nSeries;
    }

    parametersNumber[1,4] <- sum(parametersNumber[1,1:3]);
    parametersNumber[2,4] <- sum(parametersNumber[2,1:3]);

##### Now let's deal with the holdout #####
    if(holdout){
        if(modelIsMultiplicative){
            yInSample[] <- exp(yInSample);
        }
        yHoldout <- ts(y[(obsInSample+1):obsAll,],start=yForecastStart,frequency=dataFreq);
        colnames(yHoldout) <- dataNames;

        measureFirst <- measures(yHoldout[,1],yForecast[,1],yInSample[1,]);
        errorMeasures <- matrix(NA,nSeries,length(measureFirst));
        rownames(errorMeasures) <- dataNames;
        colnames(errorMeasures) <- names(measureFirst);
        errorMeasures[1,] <- measureFirst;
        for(i in 2:nSeries){
            errorMeasures[i,] <- measures(yHoldout[,i],yForecast[,i],yInSample[i,]);
        }
    }
    else{
        yHoldout <- NA;
        errorMeasures <- NA;
    }

    modelname <- "VES";
    modelname <- paste0(modelname,"(",model,")");

    if(intermittent!="n"){
        modelname <- paste0("i",modelname);
    }

    if(modelIsSeasonal){
        submodelName <- "[";
        if(seasonalType=="c"){
            submodelName[] <- paste0(submodelName,"C");
        }
        else{
            submodelName[] <- paste0(submodelName,"I");
        }

        if(persistenceType=="i"){
            submodelName[] <- paste0(submodelName,"I");
        }
        else if(persistenceType=="c"){
            submodelName[] <- paste0(submodelName,"CA");
        }
        else if(persistenceType=="s"){
            submodelName[] <- paste0(submodelName,"CS");
        }
        else if(persistenceType=="d"){
            submodelName[] <- paste0(submodelName,"D");
        }

        if(initialSeasonType=="i"){
            submodelName[] <- paste0(submodelName,"I");
        }
        else{
            submodelName[] <- paste0(submodelName,"C");
        }
        submodelName[] <- paste0(submodelName,"]");
        modelname[] <- paste0(modelname,submodelName);
    }

##### Print output #####
    if(!silentText){
        if(any(abs(eigen(matF - matG %*% matW)$values)>(1 + 1E-10))){
            warning(paste0("Model VES(",model,") is unstable! ",
                           "Use a different value of 'bounds' parameter to address this issue!"),
                    call.=FALSE);
        }
    }

##### Make a plot #####
    # This is a temporary solution
    if(!silentGraph){
        pages <- ceiling(nSeries / 5);
        perPage <- ceiling(nSeries / pages);
        packs <- c(seq(1, nSeries+1, perPage));
        if(packs[length(packs)]<nSeries+1){
            packs <- c(packs,nSeries+1);
        }
        parDefault <- par(no.readonly=TRUE);
        for(j in 1:pages){
            par(mar=c(4,4,2,1),mfcol=c(perPage,1));
            for(i in packs[j]:(packs[j+1]-1)){
                if(any(intervalType==c("u","i","l"))){
                    plotRange <- range(min(y[,i],yForecast[,i],yFitted[,i],PI[,i*2-1]),
                                       max(y[,i],yForecast[,i],yFitted[,i],PI[,i*2]));
                }
                else{
                    plotRange <- range(min(y[,i],yForecast[,i],yFitted[,i]),
                                       max(y[,i],yForecast[,i],yFitted[,i]));
                }
                plot(y[,i],main=paste0(modelname," on ",dataNames[i]),ylab="Y",
                     ylim=plotRange, xlim=range(time(y[,i])[1],time(yForecast)[max(h,1)]),
                     type="l");
                lines(yFitted[,i],col="purple",lwd=2,lty=2);
                if(h>1){
                    if(any(intervalType==c("u","i","l"))){
                        lines(PI[,i*2-1],col="darkgrey",lwd=3,lty=2);
                        lines(PI[,i*2],col="darkgrey",lwd=3,lty=2);

                        polygon(c(seq(dataDeltat*(yForecastStart[2]-1)+yForecastStart[1],
                                      dataDeltat*(end(yForecast)[2]-1)+end(yForecast)[1],dataDeltat),
                                  rev(seq(dataDeltat*(yForecastStart[2]-1)+yForecastStart[1],
                                          dataDeltat*(end(yForecast)[2]-1)+end(yForecast)[1],dataDeltat))),
                                c(as.vector(PI[,i*2]), rev(as.vector(PI[,i*2-1]))), col="lightgray",
                                border=NA, density=10);
                    }
                    lines(yForecast[,i],col="blue",lwd=2);
                }
                else{
                    if(any(intervalType==c("u","i","l"))){
                        points(PI[,i*2-1],col="darkgrey",lwd=3,pch=4);
                        points(PI[,i*2],col="darkgrey",lwd=3,pch=4);
                    }
                    points(yForecast[,i],col="blue",lwd=2,pch=4);
                }
                abline(v=dataDeltat*(yForecastStart[2]-2)+yForecastStart[1],col="red",lwd=2);
            }
        }
        par(parDefault);
    }

    ##### Return values #####
    model <- list(model=modelname,timeElapsed=Sys.time()-startTime,
                  states=matvt,persistence=persistenceValue,transition=transitionValue,
                  measurement=matW, phi=dampedValue, B=B,
                  initialType=initialType,initial=initialValue,initialSeason=initialSeasonValue,
                  nParam=parametersNumber, imodel=imodel,
                  y=y,fitted=yFitted,holdout=yHoldout,residuals=errors,Sigma=Sigma,
                  forecast=yForecast,PI=PI,interval=intervalType,level=level,
                  ICs=ICs,logLik=logLik,lossValue=cfObjective,loss=loss,accuracy=errorMeasures,
                  FI=FI);
    return(structure(model,class=c("vsmooth","smooth")));
}
