#' Vector exponential smoothing model with Group Seasonal Indices
#'
#' Function constructs VES model with restrictions on seasonal indices
#'
#' Function estimates VES in a form of the Single Source of Error state space
#' model, restricting the seasonal indices. The model is based on \link[smooth]{ves}
#'
#' In case of multiplicative model, instead of the vector y_t we use its logarithms.
#' As a result the multiplicative model is much easier to work with.
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("ves","smooth")}
#'
#' @template ssAuthor
#' @template vssKeywords
#'
#' @template vssGeneralRef
#'
#' @param y The matrix with data, where series are in columns and
#' observations are in rows.
#' @param model The type of seasonal ETS model. Currently only "MMM" is available.
#' @param weights The vector of weights for seasonal indices of the length equal to
#' the number of time series in the model.
#' @param type Type of the GSI model. Can be "Model 1", "Model 2" or "Model 3".
#' @param h Length of forecasting horizon.
#' @param holdout If \code{TRUE}, holdout sample of size \code{h} is taken from
#' the end of the data.
#' @param ic The information criterion used in the model selection procedure.
#' @param interval Type of interval to construct. NOT AVAILABLE YET!
#'
#' This can be:
#'
#' \itemize{
#' \item \code{none}, aka \code{n} - do not produce prediction
#' interval.
#' \item \code{conditional}, \code{c} - produces multidimensional elliptic
#' interval for each step ahead forecast.
#' \item \code{unconditional}, \code{u} - produces separate bounds for each series
#' based on ellipses for each step ahead. These bounds correspond to min and max
#' values of the ellipse assuming that all the other series but one take values in
#' the centre of the ellipse. This leads to less accurate estimates of bounds
#' (wider interval than needed), but these could still be useful.
#' \item \code{independent}, \code{i} - produces interval based on variances of
#' each separate series. This does not take vector structure into account.
#' }
#' The parameter also accepts \code{TRUE} and \code{FALSE}. The former means that
#' conditional interval are constructed, while the latter is equivalent to
#' \code{none}.
#' @param level Confidence level. Defines width of prediction interval.
#' @param silent If \code{silent="none"}, then nothing is silent, everything is
#' printed out and drawn. \code{silent="all"} means that nothing is produced or
#' drawn (except for warnings). In case of \code{silent="graph"}, no graph is
#' produced. If \code{silent="legend"}, then legend of the graph is skipped.
#' And finally \code{silent="output"} means that nothing is printed out in the
#' console, but the graph is produced. \code{silent} also accepts \code{TRUE}
#' and \code{FALSE}. In this case \code{silent=TRUE} is equivalent to
#' \code{silent="all"}, while \code{silent=FALSE} is equivalent to
#' \code{silent="none"}. The parameter also accepts first letter of words ("n",
#' "a", "g", "l", "o").
#' @param loss Type of Cost Function used in optimization. \code{loss} can
#' be:
#' \itemize{
#' \item \code{likelihood} - which assumes the minimisation of the determinant
#' of the covariance matrix of errors between the series. This implies that the
#' series could be correlated;
#' \item \code{diagonal} - the covariance matrix is assumed to be diagonal with
#' zeros off the diagonal. The determinant of this matrix is just a product of
#' variances. This thing is minimised in this situation in logs.
#' \item \code{trace} - the trace of the covariance matrix. The sum of variances
#' is minimised in this case.
#' }
#' @param bounds What type of bounds to use in the model estimation. The first
#' letter can be used instead of the whole word. Currently only \code{"admissible"}
#' bounds are available.
#' @param ...  Other non-documented parameters. For example \code{FI=TRUE} will
#' make the function also produce Fisher Information matrix, which then can be
#' used to calculated variances of smoothing parameters and initial states of
#' the model.
#' @return Object of class "vsmooth" is returned. It contains the following list of
#' values:
#' \itemize{
#' \item \code{model} - The name of the fitted model;
#' \item \code{timeElapsed} - The time elapsed for the construction of the model;
#' \item \code{states} - The matrix of states with components in columns and time in rows;
#' \item \code{persistence} - The persistence matrix;
#' \item \code{coefficients} - The vector of all the estimated coefficients;
#' \item \code{initial} - The initial values of the non-seasonal components;
#' \item \code{initialSeason} - The initial values of the seasonal components;
#' \item \code{nParam} - The number of estimated parameters;
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
#' initialSeason <- runif(12,-1,1)
#' Y <- sim.ves("AAA", obs=120, nSeries=2, frequency=12, initial=c(10,0),
#'              initialSeason=initialSeason-mean(initialSeason),
#'              persistence=c(0.06,0.05,0.2), mean=0, sd=0.03)
#' Y$data <- exp(Y$data)
#'
#' # The simplest model applied to the data with the default values
#' gsi(Y, h=10, holdout=TRUE, interval="u")
#'
#' # An example with MASS package and correlated errors
#' \dontrun{library(MASS)}
#' \dontrun{Y <- sim.ves("AAA", obs=120, nSeries=2, frequency=12,
#'          initial=c(5,0), initialSeason=initialSeason-mean(initialSeason),
#'          persistence=c(0.02,0.01,0.1), randomizer="mvrnorm", mu=c(0,0),
#'          Sigma=matrix(c(0.2,0.1,0.1,0.1),2,2))}
#' \dontrun{Y$data <- exp(Y$data)}
#' \dontrun{gsi(Y, h=10, holdout=TRUE, interval="u", silent=FALSE)}
#'
#' @export
gsi <- function(y, model="MNM", weights=1/ncol(y),
                type=c(3,2,1),
                loss=c("likelihood","diagonal","trace"),
                ic=c("AICc","AIC","BIC","BICc"), h=10, holdout=FALSE,
                interval=c("none","conditional","unconditional","independent"), level=0.95,
                bounds=c("admissible","usual","none"),
                silent=c("all","graph","output","none"), ...){
# Copyright (C) 2018 - Inf  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

    ##### Check if data was used instead of y. Remove by 2.6.0 #####
    y <- depricator(y, list(...), "data");
    loss <- depricator(loss, list(...), "cfType");
    interval <- depricator(interval, list(...), "intervals");

# If a previous model provided as a model, write down the variables
    # if(any(class(model)=="vsmooth")){
    #     if(smoothType(model)!="GSI"){
    #         stop("The provided model is not GSI.",call.=FALSE);
    #     }
    #     # intermittent <- model$intermittent;
    #     # if(any(intermittent==c("p","provided"))){
    #     #     warning("The provided model had predefined values of occurences for the holdout. We don't have them.",call.=FALSE);
    #     #     warning("Switching to intermittent='auto'.",call.=FALSE);
    #     #     intermittent <- "a";
    #     # }
    #     persistence <- model$persistence;
    #     transition <- model$transition;
    #     phi <- model$phi;
    #     measurement <- model$measurement;
    #     initial <- model$initial;
    #     initialSeason <- model$initialSeason;
    #     # nParamOriginal <- model$nParam;
    #     # if(is.null(xreg)){
    #     #     xreg <- model$xreg;
    #     # }
    #     # initialX <- model$initialX;
    #     # persistenceX <- model$persistenceX;
    #     # transitionX <- model$transitionX;
    #     # if(any(c(persistenceX)!=0) | any((transitionX!=0)&(transitionX!=1))){
    #     #     updateX <- TRUE;
    #     # }
    #     model <- modelType(model);
    # }
    # # else{
    #     # nParamOriginal <- NULL;
    # # }

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

##### Set environment for vssInput and make all the checks #####
    # environment(vssInput) <- environment();
    # vssInput("gsi",ParentEnvironment=environment());

##### Cost Function for GSI #####
CF <- function(A){
    elements <- BasicInitialiserGSI(matvt,matF,matG,matW,A);

    cfRes <- vOptimiserWrap(yInSample, elements$matvt, elements$matF, elements$matW, elements$matG,
                            modelLags, "A", "A", "A", loss, normalizer, bounds, ot, otObs);
    # multisteps, initialType, bounds,

    if(is.nan(cfRes) | is.na(cfRes) | is.infinite(cfRes)){
        cfRes <- 1e+100;
    }

    return(cfRes);
}

##### A values for estimation #####
# Function constructs default bounds where A values should lie
AValues <- function(maxlag,nComponentsAll,nComponentsNonSeasonal,nSeries){
    A <- NA;
    ALower <- NA;
    AUpper <- NA;
    ANames <- NA;

    ### Persistence matrix
    persistenceLength <- nComponentsAll*nSeries+1;
    A <- c(A,rep(0.1,persistenceLength));
    if(bounds=="u"){
        ALower <- c(ALower,rep(0,persistenceLength));
        AUpper <- c(AUpper,rep(1,persistenceLength));
    }
    else{
        ALower <- c(ALower,rep(-5,persistenceLength));
        AUpper <- c(AUpper,rep(5,persistenceLength));
    }
    ANames <- c(ANames,paste0("Persistence",c(1:persistenceLength)));

    ### Vector of initials
    initialLength <- nComponentsNonSeasonal*nSeries;
    A <- c(A,initialValue);
    ANames <- c(ANames,paste0("initial",c(1:initialLength)));
    ALower <- c(ALower,rep(-Inf,initialLength));
    AUpper <- c(AUpper,rep(Inf,initialLength));

    ### Vector of initial seasonals
    initialSeasonLength <- maxlag;
    A <- c(A,initialSeasonValue);
    ANames <- c(ANames,paste0("initialSeason",c(1:initialSeasonLength)));
    ALower <- c(ALower,rep(-Inf,initialSeasonLength));
    AUpper <- c(AUpper,rep(Inf,initialSeasonLength));

    A <- A[!is.na(A)];
    ALower <- ALower[!is.na(ALower)];
    AUpper <- AUpper[!is.na(AUpper)];
    ANames <- ANames[!is.na(ANames)];

    return(list(A=A,ALower=ALower,AUpper=AUpper,ANames=ANames));
}


##### Basic matrices creator #####
# This thing returns matvt, matF, matG, matW, dampedValue, initialValue and initialSeasonValue if they are not provided + modelLags
BasicMakerGSI <- function(...){
    # ellipsis <- list(...);
    # ParentEnvironment <- ellipsis[['ParentEnvironment']];

    ### Persistence matrix
    matG <- matrix(0,nSeries*nComponentsAll+1,nSeries);

    ### Transition matrix
    # This is specific to the non-damped trend model
    transitionValue <- matrix(c(1,0,1,1),2,2);
    transitionBuffer <- diag(nSeries*nComponentsAll+1);
    for(i in 1:nSeries){
        transitionBuffer[c(1:nComponentsAll)+nComponentsAll*(i-1),
                         c(1:nComponentsAll)+nComponentsAll*(i-1)] <- transitionValue;
    }
    matF <- transitionBuffer;

    ### Measurement matrix
    matW <- matrix(0,nSeries,nSeries*nComponentsAll+1);
    # Fill in the values for non-seasonal parts
    for(i in 1:nSeries){
        matW[i,c(1:nComponentsAll)+nComponentsAll*(i-1)] <- 1;
    }
    # Add the seasonal one
    matW[,nSeries*nComponentsAll+1] <- 1;

    ### Vector of states
    statesNames <- c("level","trend");

    matvt <- matrix(NA, nComponentsAll*nSeries+1, obsStates,
                    dimnames=list(c(paste0(rep(dataNames,each=nComponentsAll),
                                         "_",statesNames),"seasonal"),NULL));
    ## Deal with non-seasonal part of the vector of states
    XValues <- rbind(rep(1,obsInSample),c(1:obsInSample));
    initialValue <- yInSample %*% t(XValues) %*% solve(XValues %*% t(XValues));
    initialValue <- matrix(as.vector(t(initialValue)),nComponentsNonSeasonal * nSeries,1);

    ## Deal with seasonal part of the vector of states
    # Matrix of dummies for seasons
    XValues <- matrix(rep(diag(maxlag),ceiling(obsInSample/maxlag)),maxlag)[,1:obsInSample];
    initialSeasonValue <- (yInSample-rowMeans(yInSample)) %*% t(XValues) %*% solve(XValues %*% t(XValues));
    initialSeasonValue <- matrix(colMeans(initialSeasonValue),1,maxlag);

    ### modelLags
    modelLags <- matrix(c(rep(1,nSeries*nComponentsAll),maxlag),nSeries*nComponentsAll+1,1);

    return(list(matvt=matvt, matF=matF, matG=matG, matW=matW,
                initialValue=initialValue, initialSeasonValue=initialSeasonValue, modelLags=modelLags));
}

##### Basic matrices filler #####
# This thing fills in matvt, matF, matG and matW with values from A and returns the corrected values
BasicInitialiserGSI <- function(matvt,matF,matG,matW,A){
    nCoefficients <- 0;

    ### Persistence matrix
    persistenceBuffer <- matrix(0,nSeries*nComponentsAll+1,nSeries);
    # Independent values
    persistenceValue <- A[1:(nComponentsAll*nSeries+1)];
    nCoefficients <- nComponentsAll*nSeries+1;
    for(i in 1:nSeries){
        persistenceBuffer[1:nComponentsAll+nComponentsAll*(i-1),i] <- persistenceValue[1:nComponentsAll+nComponentsAll*(i-1)];
    }
    persistenceBuffer[nSeries*nComponentsAll+1,] <- weights * persistenceValue[nComponentsAll*nSeries+1];
    matG <- persistenceValue <- persistenceBuffer;

    ### Vector of states
    ## Deal with non-seasonal part of the vector of states

    initialValue <- matrix(A[nCoefficients+c(1:(nComponentsNonSeasonal*nSeries))],nComponentsNonSeasonal * nSeries, 1);
    nCoefficients <- nCoefficients + nComponentsNonSeasonal*nSeries;
    matvt[1:(nComponentsNonSeasonal*nSeries),1:maxlag] <- rep(initialValue,maxlag);

    ## Deal with seasonal part of the vector of states
    initialPlaces <- nComponentsAll*nSeries+1;
    matvt[initialPlaces,1:maxlag] <- matrix(A[nCoefficients+c(1:maxlag)],1,maxlag,byrow=TRUE);
    nCoefficients <- nCoefficients + maxlag;

    return(list(matvt=matvt,matF=matF,matG=matG,matW=matW));
}

##### Basic estimation function for gsi() #####
EstimatorGSI <- function(...){
    environment(BasicMakerGSI) <- environment();
    environment(AValues) <- environment();
    environment(vLikelihoodFunction) <- environment();
    environment(vICFunction) <- environment();
    environment(CF) <- environment();
    elements <- BasicMakerGSI();
    list2env(elements,environment());

    AList <- AValues(maxlag,nComponentsAll,nComponentsNonSeasonal,nSeries);
    A <- AList$A;

    if(any((A>=AList$AUpper),(A<=AList$ALower))){
        A[A>=AList$AUpper] <- AList$AUpper[A>=AList$AUpper] * 0.999 - 0.001;
        A[A<=AList$ALower] <- AList$ALower[A<=AList$ALower] * 1.001 + 0.001;
    }

    # Parameters are chosen to speed up the optimisation process and have decent accuracy
    res <- nloptr(A, CF, lb=AList$ALower, ub=AList$AUpper,
                  opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
    A <- res$solution;

    if(any((A>=AList$AUpper),(A<=AList$ALower))){
        A[A>=AList$AUpper] <- AList$AUpper[A>=AList$AUpper] * 0.999 - 0.001;
        A[A<=AList$ALower] <- AList$ALower[A<=AList$ALower] * 1.001 + 0.001;
    }

    res2 <- nloptr(A, CF, lb=AList$ALower, ub=AList$AUpper,
                  opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-6, "maxeval"=1000));
    # This condition is needed in order to make sure that we did not make the solution worse
    if(res2$objective <= res$objective){
        res <- res2;
    }
    A <- res$solution;

    if(all(A==AList$A)){
        if(persistenceEstimate){
            warning(paste0("Failed to optimise the model. ",
                           "Try different initialisation maybe?\nAnd check all the messages and warnings...",
                           "If you did your best, but the optimiser still fails, report this to the maintainer, please."),
                    call.=FALSE);
        }
    }
    names(A) <- AList$ANames;

    # First part is for the covariance matrix
    if(loss=="l"){
        nParam <- nSeries * (nSeries + 1) / 2 + length(A);
    }
    else{
        nParam <- nSeries + length(A);
    }

    IAValues <- vICFunction(nParam=nParam,A=A,Etype="M");
    ICs <- IAValues$ICs;
    logLik <- IAValues$llikelihood;

    # Write down Fisher Information if needed
    if(FI){
        environment(vLikelihoodFunction) <- environment();
        FI <- -numDeriv::hessian(vLikelihoodFunction,A);
        rownames(FI) <- AList$ANames;
        colnames(FI) <- AList$ANames;
    }

    return(list(ICs=ICs,objective=res$objective,A=A,nParam=nParam,logLik=logLik,FI=FI));
}

##### Function constructs the GSI function #####
CreatorGSI <- function(silent=FALSE,...){
        environment(EstimatorGSI) <- environment();
        res <- EstimatorGSI(ParentEnvironment=environment());
        listToReturn <- list(cfObjective=res$objective,A=res$A,ICs=res$ICs,icBest=res$ICs[ic],
                             nParam=res$nParam,logLik=res$logLik,FI=res$FI);

        return(listToReturn);
}


##### silent #####
    silent <- silent[1];
    # Fix for cases with TRUE/FALSE.
    if(!is.logical(silent)){
        if(all(silent!=c("none","all","graph","legend","output","debugging","n","a","g","l","o","d"))){
            warning(paste0("Sorry, I have no idea what 'silent=",silent,"' means. Switching to 'none'."),call.=FALSE);
            silent <- "none";
        }
        silent <- substring(silent,1,1);
    }
    silentValue <- silent;

    if(silentValue==FALSE | silentValue=="n"){
        silentText <- FALSE;
        silentGraph <- FALSE;
        silentLegend <- FALSE;
    }
    else if(silentValue==TRUE | silentValue=="a"){
        silentText <- TRUE;
        silentGraph <- TRUE;
        silentLegend <- TRUE;
    }
    else if(silentValue=="g"){
        silentText <- FALSE;
        silentGraph <- TRUE;
        silentLegend <- TRUE;
    }
    else if(silentValue=="l"){
        silentText <- FALSE;
        silentGraph <- FALSE;
        silentLegend <- TRUE;
    }
    else if(silentValue=="o"){
        silentText <- TRUE;
        silentGraph <- FALSE;
        silentLegend <- FALSE;
    }
    else if(silentValue=="d"){
        silentText <- TRUE;
        silentGraph <- FALSE;
        silentLegend <- FALSE;
    }

#### Check data ####
    if(any(is.vsmooth.sim(y))){
        y <- y$data;
        if(length(dim(y))==3){
            warning("Simulated data contains several samples. Selecting a random one.",call.=FALSE);
            y <- ts(y[,,runif(1,1,dim(y)[3])]);
        }
    }

    if(!is.data.frame(y)){
        if(!is.numeric(y)){
            stop("The provided data is not a numeric matrix! Can't construct any model!", call.=FALSE);
        }
    }

    if(is.null(dim(y))){
        stop("The provided data is not a matrix or a data.frame! If it is a vector, please use es() function instead.", call.=FALSE);
    }

    if(is.data.frame(y)){
        y <- as.matrix(y);
    }

    # Number of series in the matrix
    nSeries <- ncol(y);

    if(is.null(ncol(y))){
        stop("The provided data is not a matrix! Use es() function instead!", call.=FALSE);
    }
    if(ncol(y)==1){
        stop("The provided data contains only one column. Use es() function instead!", call.=FALSE);
    }
    # Check the data for NAs
    if(any(is.na(y))){
        if(!silentText){
            warning("Data contains NAs. These observations will be substituted by zeroes.", call.=FALSE);
        }
        y[is.na(y)] <- 0;
    }

    # Define obs, the number of observations of in-sample
    obsInSample <- nrow(y) - holdout*h;

    # Define obsAll, the overal number of observations (in-sample + holdout)
    obsAll <- nrow(y) + (1 - holdout)*h;

    # If obsInSample is negative, this means that we can't do anything...
    if(obsInSample<=0){
        stop("Not enough observations in sample.", call.=FALSE);
    }
    # Define the actual values. Transpose the matrix!
    yInSample <- matrix(y[1:obsInSample,],nSeries,obsInSample,byrow=TRUE);
    dataFreq <- frequency(y);
    dataDeltat <- deltat(y);
    dataStart <- start(y);
    yForecastStart <- time(y)[obsInSample]+deltat(y);
    dataNames <- colnames(y);
    if(!is.null(dataNames)){
        dataNames <- gsub(" ", "_", dataNames, fixed = TRUE);
        dataNames <- gsub(":", "_", dataNames, fixed = TRUE);
        dataNames <- gsub("$", "_", dataNames, fixed = TRUE);
    }
    else{
        dataNames <- paste0("Series",c(1:nSeries));
    }

    if(all(yInSample>0)){
        yInSample <- log(yInSample);
    }
    else{
        stop("Cannot apply multiplicative model to the non-positive data", call.=FALSE);
    }

    # Number of parameters to estimate / provided
    parametersNumber <- matrix(0,2,4,
                               dimnames=list(c("Estimated","Provided"),
                                             c("nParamInternal","nParamXreg","nParamIntermittent","nParamAll")));

    if(any(is.null(weights))){
        warning("The weights are not provided. Substituting them by equal ones.", call.=FALSE);
        weights <- rep(1/nSeries, nSeries);
    }
    else if(any(is.na(weights))){
        warning("The weights are NAs. Substituting them by equal ones.", call.=FALSE);
        weights <- rep(1/nSeries, nSeries);
    }

##### Cost function type #####
    loss <- loss[1];
    if(!any(loss==c("likelihood","diagonal","trace","l","d","t"))){
        warning(paste0("Strange loss function specified: ",loss,". Switching to 'likelihood'."),call.=FALSE);
        loss <- "likelihood";
    }
    loss <- substr(loss,1,1);

    normalizer <- sum(colMeans(abs(diff(t(yInSample))),na.rm=TRUE));

##### Define the main variables #####
    # For now we only have level and trend. The seasonal component is common to all the series
    nComponentsNonSeasonal <- nComponentsAll <- 2;
    maxlag <- dataFreq;
    obsStates <- max(obsAll + maxlag, obsInSample + 2*maxlag);
    Etype <- "M";
    Ttype <- "M";
    Stype <- "M";
    modelIsMultiplicative <- TRUE;
    FI <- FALSE;

##### Non-intermittent model, please!
    ot <- matrix(1,nrow=nrow(yInSample),ncol=ncol(yInSample));
    otObs <- matrix(obsInSample,nSeries,nSeries);
    intermittent <- "n";
    imodel <- NULL;
    cumulative <- FALSE;


##### Information Criteria #####
    ic <- ic[1];
    if(all(ic!=c("AICc","AIC","BIC","BICc"))){
        warning(paste0("Strange type of information criteria defined: ",ic,". Switching to 'AICc'."),call.=FALSE);
        ic <- "AICc";
    }

##### interval, intervalType, level #####
    intervalType <- interval[1];
    # Check the provided type of interval

    if(is.logical(intervalType)){
        if(intervalType){
            intervalType <- "c";
        }
        else{
            intervalType <- "none";
        }
    }

    if(all(intervalType!=c("c","u","i","n","none","conditional","unconditional","independent"))){
        warning(paste0("Wrong type of interval: '",intervalType, "'. Switching to 'conditional'."),call.=FALSE);
        intervalType <- "c";
    }

    if(intervalType=="none"){
        intervalType <- "n";
        interval <- FALSE;
    }
    else if(intervalType=="conditional"){
        intervalType <- "c";
        interval <- TRUE;
    }
    else if(intervalType=="unconditional"){
        intervalType <- "u";
        interval <- TRUE;
    }
    else if(intervalType=="independent"){
        intervalType <- "i";
        interval <- TRUE;
    }
    else{
        interval <- TRUE;
    }

    if(level>1){
        level <- level / 100;
    }

##### bounds #####
    bounds <- substring(bounds[1],1,1);
    if(all(bounds!=c("u","a","n"))){
        warning("Strange bounds are defined. Switching to 'admissible'.",call.=FALSE);
        bounds <- "a";
    }



##### Preset yFitted, yForecast, errors and basic parameters #####
    yFitted <- matrix(NA,nSeries,obsInSample);
    yForecast <- matrix(NA,nSeries,h);
    errors <- matrix(NA,nSeries,obsInSample);
    rownames(yFitted) <- rownames(yForecast) <- rownames(errors) <- dataNames;


##### Now do estimation and model selection #####
    environment(BasicMakerGSI) <- environment();
    environment(BasicInitialiserGSI) <- environment();
    environment(vssFitter) <- environment();
    environment(vssForecaster) <- environment();

    gsiValues <- CreatorGSI(silent=silentText);

##### Fit the model and produce forecast #####
    list2env(gsiValues,environment());
    list2env(BasicMakerGSI(),environment());
    list2env(BasicInitialiserGSI(matvt,matF,matG,matW,A),environment());

    cfObjective <- exp(cfObjective);

    model <- "MMM";

    vssFitter(ParentEnvironment=environment());
    vssForecaster(ParentEnvironment=environment());

    ##### Write down persistence, transition, initials etc #####
# Write down the persistenceValue, transitionValue, initialValue, initialSeasonValue

    persistenceNames <- c("level","trend");
    persistenceValue <- matG;
    rownames(persistenceValue) <- c(1:(nSeries*nComponentsAll+1));
    rownames(persistenceValue)[1:(nSeries*nComponentsAll)] <- paste0(rep(dataNames,each=nComponentsAll), "_", persistenceNames);
    rownames(persistenceValue)[nSeries*nComponentsAll+1] <- "seasonal";
    colnames(persistenceValue) <- dataNames;
    parametersNumber[1,1] <- parametersNumber[1,1] + nSeries*nComponentsAll + 1;

    initialPlaces <- c(nComponentsAll*(c(1:nSeries)-1)+1, nComponentsAll*(c(1:nSeries)-1)+2);
    initialPlaces <- sort(initialPlaces);
    initialNames <- c("level","trend");
    initialValue <- matrix(matvt[initialPlaces,maxlag],nComponentsNonSeasonal*nSeries,1);
    rownames(initialValue) <- paste0(rep(dataNames,each=nComponentsNonSeasonal), "_", initialNames);
    parametersNumber[1,1] <- parametersNumber[1,1] + length(unique(as.vector(initialValue)));

    initialPlaces <- nComponentsAll*nSeries + 1;
    initialSeasonValue <- matrix(matvt[initialPlaces,1:maxlag],1,maxlag);
    colnames(initialSeasonValue) <- paste0("Seasonal",c(1:maxlag));
    parametersNumber[1,1] <- parametersNumber[1,1] + length(unique(as.vector(initialSeasonValue)));


    matvt <- ts(t(matvt),start=(time(y)[1] - dataDeltat*maxlag),frequency=dataFreq);
    yFitted <- ts(t(yFitted),start=dataStart,frequency=dataFreq);
    errors <- ts(t(errors),start=dataStart,frequency=dataFreq);

    yForecast <- ts(t(yForecast),start=yForecastStart,frequency=dataFreq);
    if(!is.matrix(yForecast)){
        yForecast <- as.matrix(yForecast,h,nSeries);
    }
    colnames(yForecast) <- dataNames;
    yForecastStart <- start(yForecast)
    if(any(intervalType==c("i","u"))){
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

    modelname <- "GSI";
    modelname <- paste0(modelname,"(",model,")");

    if(intermittent!="n"){
        modelname <- paste0("i",modelname);
    }

##### Print output #####
    if(!silentText){
        if(any(abs(eigen(matF - matG %*% matW)$values)>(1 + 1E-10))){
            warning(paste0("Model GSI(",model,") is unstable! Use a different value of 'bounds' parameter to address this issue!"),
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
                if(any(intervalType==c("u","i"))){
                    plotRange <- range(min(y[,i],yForecast[,i],yFitted[,i],PI[,i*2-1]),
                                       max(y[,i],yForecast[,i],yFitted[,i],PI[,i*2]));
                }
                else{
                    plotRange <- range(min(y[,i],yForecast[,i],yFitted[,i]),
                                       max(y[,i],yForecast[,i],yFitted[,i]));
                }
                plot(y[,i],main=paste0(modelname," ",dataNames[i]),ylab="Y",
                     ylim=plotRange, xlim=range(time(y[,i])[1],time(yForecast)[max(h,1)]),
                     type="l");
                lines(yFitted[,i],col="purple",lwd=2,lty=2);
                if(h>1){
                    if(any(intervalType==c("u","i"))){
                        lines(PI[,i*2-1],col="darkgrey",lwd=3,lty=2);
                        lines(PI[,i*2],col="darkgrey",lwd=3,lty=2);

                        polygon(c(seq(dataDeltat*(yForecastStart[2]-1)+yForecastStart[1],dataDeltat*(end(yForecast)[2]-1)+end(yForecast)[1],dataDeltat),
                                  rev(seq(dataDeltat*(yForecastStart[2]-1)+yForecastStart[1],dataDeltat*(end(yForecast)[2]-1)+end(yForecast)[1],dataDeltat))),
                                c(as.vector(PI[,i*2]), rev(as.vector(PI[,i*2-1]))), col = "lightgray", border=NA, density=10);
                    }
                    lines(yForecast[,i],col="blue",lwd=2);
                }
                else{
                    if(any(intervalType==c("u","i"))){
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
                  states=matvt, coefficients=A,
                  persistence=persistenceValue,
                  initial=initialValue, initialSeason=initialSeasonValue,
                  nParam=parametersNumber,
                  y=y,fitted=yFitted,holdout=yHoldout,residuals=errors,Sigma=Sigma,
                  forecast=yForecast,PI=PI,interval=intervalType,level=level,
                  ICs=ICs,logLik=logLik,lossValue=cfObjective,loss=loss,accuracy=errorMeasures,
                  FI=FI);
    return(structure(model,class=c("vsmooth","smooth")));
}
