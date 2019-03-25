utils::globalVariables(c("modelDo","initialValue","modelLagsMax"));

#' Occurrence ETS model
#'
#' Function returns the occurrence part of iETS model with the specified
#' probability update and model types.
#'
#' The function estimates probability of demand occurrence, using the selected
#' ETS state space models. Although the function accepts all types of ETS models,
#' only the pure multiplicative models make sense.
#'
#' @template ssIntermittentRef
#' @template ssInitialParam
#' @template ssPersistenceParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param data Either numeric vector or time series vector.
#' @param occurrence Type of model used in probability estimation. Can be
#' \code{"none"} - none,
#' \code{"fixed"} - constant probability,
#' \code{"general"} - the general Beta model with two parameters,
#' \code{"odds-ratio"} - the Odds-ratio model with b=1 in Beta distribution,
#' \code{"inverse-odds-ratio"} - the model with a=1 in Beta distribution,
#' \code{"probability"} - the TSB-like (Teunter et al., 2011) probability update
#' mechanism a+b=1,
#' \code{"auto"} - the automatically selected type of occurrence model.
#' @param ic Information criteria to use in case of model selection.
#' @param h Forecast horizon.
#' @param holdout If \code{TRUE}, holdout sample of size \code{h} is taken from
#' the end of the data.
#' @param intervals Type of intervals to construct. This can be:
#'
#' \itemize{
#' \item \code{none}, aka \code{n} - do not produce prediction
#' intervals.
#' \item \code{parametric}, \code{p} - use state-space structure of ETS. In
#' case of mixed models this is done using simulations, which may take longer
#' time than for the pure additive and pure multiplicative models.
#' \item \code{semiparametric}, \code{sp} - intervals based on covariance
#' matrix of 1 to h steps ahead errors and assumption of normal / log-normal
#' distribution (depending on error type).
#' \item \code{nonparametric}, \code{np} - intervals based on values from a
#' quantile regression on error matrix (see Taylor and Bunn, 1999). The model
#' used in this process is e[j] = a j^b, where j=1,..,h.
#' }
#' The parameter also accepts \code{TRUE} and \code{FALSE}. The former means that
#' parametric intervals are constructed, while the latter is equivalent to
#' \code{none}.
#' If the forecasts of the models were combined, then the intervals are combined
#' quantile-wise (Lichtendahl et al., 2013).
#' @param level Confidence level. Defines width of prediction interval.
#' @param bounds What type of bounds to use in the model estimation. The first
#' letter can be used instead of the whole word.
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
#' @param model Type of ETS model used for the estimation. Normally this should
#' be either \code{"MNN"}.
#' @param initialSeason Initial vector of seasonal components. If \code{NULL},
#' then it is estimated.
#' @param xreg Vector of matrix of exogenous variables, explaining some parts
#' of occurrence variable (probability).
#' @param xregDo Variable defines what to do with the provided xreg:
#' \code{"use"} means that all of the data should be used, while
#' \code{"select"} means that a selection using \code{ic} should be done.
#' \code{"combine"} will be available at some point in future...
#' @param updateX If \code{TRUE}, transition matrix for exogenous variables is
#' estimated, introducing non-linear interactions between parameters.
#' Prerequisite - non-NULL \code{xreg}.
#' @param ... The parameters passed to the optimiser, such as \code{maxeval},
#' \code{xtol_rel}, \code{algorithm} and \code{print_level}. The description of
#' these is printed out by \code{nloptr.print.options()} function from the \code{nloptr}
#' package. The default values in the oes function are \code{maxeval=500},
#' \code{xtol_rel=1E-8}, \code{algorithm="NLOPT_LN_SBPLX"} and \code{print_level=0}.
#' @return The object of class "occurrence" is returned. It contains following list of
#' values:
#'
#' \itemize{
#' \item \code{model} - the type of the estimated ETS model;
#' \item \code{fitted} - fitted values of the constructed model;
#' \item \code{forecast} - forecast for \code{h} observations ahead;
#' \item \code{states} - values of states (currently level only);
#' \item \code{logLik} - likelihood value for the model
#' \item \code{nParam} - number of parameters used in the model;
#' \item \code{residuals} - residuals of the model;
#' \item \code{actuals} - actual values of probabilities (zeros and ones).
#' \item \code{persistence} - the vector of smoothing parameters;
#' \item \code{initial} - initial values of the state vector;
#' \item \code{initialSeason} - the matrix of initials seasonal states;
#' \item \code{occurrence} - the type of the occurrence model.
#' }
#' @seealso \code{\link[forecast]{ets}, \link[forecast]{forecast},
#' \link[smooth]{es}}
#' @keywords iss intermittent demand intermittent demand state space model
#' exponential smoothing forecasting
#' @examples
#'
#' y <- rpois(100,0.1)
#' oes(y, occurrence="o")
#'
#' oes(y, occurrence="f")
#'
#' @export
oes <- function(data, model="MNN", persistence=NULL, initial="o", initialSeason=NULL,
                occurrence=c("general","fixed","odds-ratio","inverse-odds-ratio",
                                   "probability","auto","none"),
                ic=c("AICc","AIC","BIC","BICc"), h=10, holdout=FALSE,
                intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                bounds=c("usual","admissible","none"),
                silent=c("all","graph","legend","output","none"),
                xreg=NULL, xregDo=c("use","select"), updateX=FALSE, ...){
    # Function returns the occurrence part of the intermittent state space model

    # Options for the fitter and forecaster:
    # O: M / A odds-ratio - "odds-ratio"
    # I: - M / A inverse-odds-ratio - "inverse-odds-ratio"
    # G: - M / A general model - "general" <- This should rely on a vector-based model
    # P: - M Probability based (TSB like) - "probability"
    # Not in the fitter:
    # F: - fixed

    ##### Preparations #####
    occurrence <- substring(occurrence[1],1,1);
    if(all(occurrence!=c("g","f","o","i","p","a","n"))){
        warning(paste0("Unknown value of occurrence provided: '",occurrence,"'. Changing to 'fixed'"),call.=FALSE);
        occurrence <- "f";
    }

    if(is.smooth.sim(data)){
        data <- data$data;
    }

    # Add all the variables in ellipsis to current environment
    # list2env(list(...),environment());
    ellipsis <- list(...);

    # Parameters for the nloptr
    if(any(names(ellipsis)=="maxeval")){
        maxeval <- ellipsis$maxeval;
    }
    else{
        maxeval <- 500;
    }
    if(any(names(ellipsis)=="xtol_rel")){
        xtol_rel <- ellipsis$xtol_rel;
    }
    else{
        xtol_rel <- 1e-8;
    }
    if(any(names(ellipsis)=="algorithm")){
        algorithm <- ellipsis$algorithm;
    }
    else{
        algorithm <- "NLOPT_LN_SBPLX";
    }
    if(any(names(ellipsis)=="print_level")){
        print_level <- ellipsis$print_level;
    }
    else{
        print_level <- 0;
    }

    #### These are needed in order for ssInput to go forward
    cfType <- "MSE";
    imodel <- NULL;
    phi <- NULL;

    ##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- environment();
    ssInput("oes",ParentEnvironment=environment());

    if(Stype!="N"){
        initialSeasonEstimate <- TRUE;
    }
    else{
        initialSeasonEstimate <- FALSE;
    }

    ### Produce vectors with zeroes and ones, fixed probability and the number of ones.
    ot <- (y!=0)*1;
    otAll <- (data!=0)*1;
    iprob <- mean(ot);
    obsOnes <- sum(ot);

    if(all(ot==ot[1])){
        warning(paste0("There is no variability in the occurrence of the variable in-sample.\n",
                       "Switching to occurrence='none'."),call.=FALSE)
        occurrence <- "n";
    }

    ##### Prepare exogenous variables #####
    xregdata <- ssXreg(data=otAll, Etype=Etype, xreg=xreg, updateX=updateX, ot=rep(1,obsInsample),
                       persistenceX=NULL, transitionX=NULL, initialX=NULL,
                       obsInsample=obsInsample, obsAll=obsAll, obsStates=obsStates,
                       maxlag=1, h=h, xregDo=xregDo, silent=silentText,
                       allowMultiplicative=FALSE);

    nExovars <- xregdata$nExovars;
    matxt <- xregdata$matxt;
    matat <- t(xregdata$matat);
    xregEstimate <- xregdata$xregEstimate;
    matFX <- xregdata$matFX;
    vecgX <- xregdata$vecgX;
    xregNames <- colnames(matxt);
    xreg <- xregdata$xreg;

    if(any(occurrence==c("g","o","i","p"))){
        ##### Initialiser of oes #####
        # This creates the states, transition, persistence and measurement matrices
        oesInitialiser <- function(Etype, Ttype, Stype, damped, occurrence,
                                   dataFreq, obsInsample, obsAll, obsStates, ot,
                                   persistenceEstimate, persistence, initialType, initialValue,
                                   initialSeasonEstimate, initialSeason){
            # Define the lags of the model, number of components and max lag
            modelLags <- 1;
            statesNames <- "level";
            if(Ttype!="N"){
                modelLags <- c(modelLags, 1);
                statesNames <- c(statesNames, "trend");
            }
            nComponentsNonSeasonal <- length(modelLags);
            if(Stype!="N"){
                modelLags <- c(modelLags, dataFreq);
                statesNames <- c(statesNames, "seasonal");
            }
            nComponentsAll <- length(modelLags);
            modelLagsMax <- max(modelLags);

            # Transition matrix
            matF <- diag(nComponentsAll);
            if(Ttype!="N"){
                matF[1,2] <- 1;
            }

            # Persistence vector. The initials are set here!
            if(persistenceEstimate){
                vecg <- matrix(0.05, nComponentsAll, 1);
            }
            else{
                vecg <- matrix(persistence, nComponentsAll, 1);
            }
            rownames(vecg) <- statesNames;

            # Measurement vector
            matw <- matrix(1, 1, nComponentsAll, dimnames=list(NULL, statesNames));

            # The matrix of states
            matvt <- matrix(NA, nComponentsAll, obsStates, dimnames=list(statesNames, NULL));

            # Define initial states. The initials are set here!
            if(initialType!="p"){
                initialStates <- rep(0, nComponentsNonSeasonal);
                initialStates[1] <- mean(ot[1:max(dataFreq,12)]);
                if(Ttype!="N"){
                    initialStates[2] <- 1e-5;
                }
                if(occurrence=="o"){
                    initialStates[1] <- initialStates[1] / (1 - initialStates[1]);
                }
                else if(occurrence=="i"){
                    initialStates[1] <- (1-initialStates[1]) / initialStates[1];
                }
                if(Etype=="M"){
                    initialStates <- exp(initialStates);
                }

                matvt[1,1:modelLagsMax] <- initialStates[1];
                if(Ttype!="N"){
                    matvt[2,1:modelLagsMax] <- initialStates[2];
                }
            }
            else{
                matvt[1:nComponentsNonSeasonal,1:modelLagsMax] <- initialValue;
            }

            # Define the seasonals
            if(Stype!="N"){
                if(initialSeasonEstimate){
                    XValues <- matrix(rep(diag(modelLagsMax),ceiling(obsInsample/modelLagsMax)),modelLagsMax)[,1:obsInsample];
                    # The seasonal values should be between -1 and 1
                    initialSeasonValue <- solve(XValues %*% t(XValues)) %*% XValues %*% (ot - mean(ot));
                    # But make sure that it lies there
                    if(any(abs(initialSeasonValue)>1)){
                        initialSeasonValue <- initialSeasonValue / (max(abs(initialSeasonValue)) + 1E-10);
                    }

                    # Correct seasonals for the two models
                    if(any(occurrence==c("o","i"))){
                        # If there are some boundary values, move them a bit
                        if(any(abs(initialSeasonValue)==1)){
                            initialSeasonValue[initialSeasonValue==1] <- 1 - 1E-10;
                            initialSeasonValue[initialSeasonValue==-1] <- -1 + 1E-10;
                        }
                        # Transform this into the underlying scale
                        initialSeasonValue <- (initialSeasonValue + 1) / 2;
                        if(occurrence=="o"){
                            initialSeasonValue <- initialSeasonValue / (1 - initialSeasonValue);
                        }
                        else{
                            initialSeasonValue <- (1 - initialSeasonValue) / initialSeasonValue;
                        }

                        if(Stype=="A"){
                            initialSeasonValue <- log(initialSeasonValue);
                        }
                    }

                    # Write down the initial seasons into the state matrix
                    matvt[nComponentsAll,1:modelLagsMax] <- initialSeasonValue;
                }
                else{
                    matvt[nComponentsAll,1:modelLagsMax] <- initialSeason;
                }
            }

            return(list(nComponentsAll=nComponentsAll, nComponentsNonSeasonal=nComponentsNonSeasonal,
                        modelLagsMax=modelLagsMax, modelLags=modelLags,
                        matvt=matvt, vecg=vecg, matF=matF, matw=matw));
        }

        ##### Fill in the elements of oes #####
        # This takes the existing matrices and fills them in
        oesElements <- function(A, modelLags, Ttype, Stype,
                                nComponentsAll, nComponentsNonSeasonal, nExovars, modelLagsMax,
                                persistenceEstimate, initialType, initialSeasonEstimate, xregEstimate, updateX,
                                matvt, vecg, matF, matw, matat, matFX, vecgX){
            i <- 0;
            if(persistenceEstimate){
                vecg[] <- A[1:nComponentsAll];
                i[] <- nComponentsAll;
            }
            if(damped){
                i[] <- i + 1;
                matF[,nComponentsNonSeasonal] <- A[i];
                matw[,nComponentsNonSeasonal] <- A[i];
            }
            if(initialType=="o"){
                matvt[1:nComponentsNonSeasonal,modelLagsMax] <- A[i+c(1:nComponentsNonSeasonal)];
                i[] <- i + nComponentsNonSeasonal;
            }
            if(initialSeasonEstimate){
                matvt[nComponentsAll,1:modelLagsMax] <- A[i+modelLagsMax]
                i[] <- i + modelLagsMax;
            }
            if(xregEstimate){
                matat[,1] <- A[i+c(1:nExovars)];
                i[] <- i + nExovars;
                if(updateX){
                    matFX[] <- A[i+c(1:(nExovars^2))];
                    i[] <- i + nExovars^2;

                    vecgX[] <- A[i+c(1:nExovars)];
                }
            }

            return(list(vecg=vecg, matF=matF, matw=matw, matvt=matvt,
                        matat=matat, matFX=matFX, vecgX=vecgX));
        }

        ##### A values for estimation #####
        # Function constructs default bounds where A values should lie
        AValues <- function(bounds, Ttype, Stype,
                            modelLagsMax, nComponentsAll, nComponentsNonSeasonal,
                            vecg, matvt, matat){
            A <- NA;
            ALower <- NA;
            AUpper <- NA;

            #### Usual bounds ####
            if(bounds=="u"){
                # Smoothing parameters
                if(persistenceEstimate){
                    A <- c(A,vecg);
                    ALower <- c(ALower,rep(0,nComponentsAll));
                    AUpper <- c(AUpper,rep(1,nComponentsAll));
                }
                # Phi
                if(damped){
                    A <- c(A,0.95);
                    ALower <- c(ALower,0);
                    AUpper <- c(AUpper,1);
                }
                # Initial states
                if(initialType=="o"){
                    if(Etype=="A"){
                        A <- c(A,matvt[1:nComponentsNonSeasonal,modelLagsMax]);
                        ALower <- c(ALower,-Inf);
                        AUpper <- c(AUpper,Inf);
                    }
                    else{
                        if(Ttype=="A"){
                            # This is something like ETS(M,A,N), so set level to mean, trend to zero for stability
                            A <- c(A,mean(ot[1:min(dataFreq,obsInsample)]),1E-5);
                        }
                        else{
                            A <- c(A,abs(matvt[1:nComponentsNonSeasonal,modelLagsMax]));
                        }
                        ALower <- c(ALower,1E-10);
                        AUpper <- c(AUpper,Inf);
                    }
                    if(Ttype=="A"){
                        ALower <- c(ALower,-Inf);
                        AUpper <- c(AUpper,Inf);
                    }
                    else if(Ttype=="M"){
                        ALower <- c(ALower,1E-20);
                        AUpper <- c(AUpper,3);
                    }
                    # Initial seasonals
                    if(Stype!="N"){
                        if(initialSeasonEstimate){
                            A <- c(A,matvt[nComponentsAll,1:modelLagsMax]);
                            if(Stype=="A"){
                                ALower <- c(ALower,rep(-Inf,modelLagsMax));
                                AUpper <- c(AUpper,rep(Inf,modelLagsMax));
                            }
                            else{
                                ALower <- c(ALower,matvt[nComponentsAll,1:modelLagsMax]*0.1);
                                AUpper <- c(AUpper,matvt[nComponentsAll,1:modelLagsMax]*10);
                            }
                        }
                    }
                }
            }
            #### Admissible bounds ####
            else if(bounds=="a"){
                # Smoothing parameters
                if(persistenceEstimate){
                    A <- c(A,vecg);
                    ALower <- c(ALower,rep(-5,nComponentsAll));
                    AUpper <- c(AUpper,rep(5,nComponentsAll));
                }
                # Phi
                if(damped){
                    A <- c(A,0.95);
                    ALower <- c(ALower,0);
                    AUpper <- c(AUpper,1);
                }
                # Initial states
                if(initialType=="o"){
                    if(Etype=="A"){
                        A <- c(A,matvt[modelLagsMax,1:nComponentsNonSeasonal]);
                        ALower <- c(ALower,-Inf);
                        AUpper <- c(AUpper,Inf);
                    }
                    else{
                        if(Ttype=="A"){
                            # This is something like ETS(M,A,N), so set level to mean, trend to zero for stability
                            A <- c(A,mean(ot[1:min(dataFreq,obsInsample)]),1E-5);
                        }
                        else{
                            A <- c(A,abs(matvt[modelLagsMax,1:nComponentsNonSeasonal]));
                        }
                        ALower <- c(ALower,1E-10);
                        AUpper <- c(AUpper,Inf);
                    }
                    if(Ttype=="A"){
                        ALower <- c(ALower,-Inf);
                        AUpper <- c(AUpper,Inf);
                    }
                    else if(Ttype=="M"){
                        ALower <- c(ALower,1E-20);
                        AUpper <- c(AUpper,3);
                    }
                    # Initial seasonals
                    if(Stype!="N"){
                        if(initialSeasonEstimate){
                            A <- c(A,matvt[nComponentsAll,1:modelLagsMax]);
                            if(Stype=="A"){
                                ALower <- c(ALower,rep(-Inf,modelLagsMax));
                                AUpper <- c(AUpper,rep(Inf,modelLagsMax));
                            }
                            else{
                                ALower <- c(ALower,matvt[nComponentsAll,1:modelLagsMax]*0.1);
                                AUpper <- c(AUpper,matvt[nComponentsAll,1:modelLagsMax]*10);
                            }
                        }
                    }
                }
            }
            #### No bounds ####
            else{
                # Smoothing parameters
                if(persistenceEstimate){
                    A <- c(A,vecg);
                    ALower <- c(ALower,rep(-Inf,nComponentsAll));
                    AUpper <- c(AUpper,rep(Inf,nComponentsAll));
                }
                # Phi
                if(damped){
                    A <- c(A,0.95);
                    ALower <- c(ALower,-Inf);
                    AUpper <- c(AUpper,Inf);
                }
                # Initial states
                if(initialType=="o"){
                    if(Etype=="A"){
                        A <- c(A,matvt[modelLagsMax,1:nComponentsNonSeasonal]);
                        ALower <- c(ALower,-Inf);
                        AUpper <- c(AUpper,Inf);
                    }
                    else{
                        if(Ttype=="A"){
                            # This is something like ETS(M,A,N), so set level to mean, trend to zero for stability
                            A <- c(A,mean(ot[1:min(dataFreq,obsInsample)]),1E-5);
                        }
                        else{
                            A <- c(A,abs(matvt[modelLagsMax,1:nComponentsNonSeasonal]));
                        }
                        ALower <- c(ALower,1E-10);
                        AUpper <- c(AUpper,Inf);
                    }
                    if(Ttype=="A"){
                        ALower <- c(ALower,-Inf);
                        AUpper <- c(AUpper,Inf);
                    }
                    else if(Ttype=="M"){
                        ALower <- c(ALower,1E-20);
                        AUpper <- c(AUpper,3);
                    }
                    # Initial seasonals
                    if(Stype!="N"){
                        if(initialSeasonEstimate){
                            A <- c(A,matvt[nComponentsAll,1:modelLagsMax]);
                            if(Stype=="A"){
                                ALower <- c(ALower,rep(-Inf,modelLagsMax));
                                AUpper <- c(AUpper,rep(Inf,modelLagsMax));
                            }
                            else{
                                ALower <- c(ALower,matvt[nComponentsAll,1:modelLagsMax]*0.1);
                                AUpper <- c(AUpper,matvt[nComponentsAll,1:modelLagsMax]*10);
                            }
                        }
                    }
                }
            }

            # Explanatory variables
            if(xregEstimate){
                # Initial values of at
                A <- c(A,matat[xregNames,1]);
                ALower <- c(ALower,rep(-Inf,nExovars));
                AUpper <- c(AUpper,rep(Inf,nExovars));
                if(updateX){
                    # Initials for the transition matrix
                    A <- c(A,as.vector(matFX));
                    ALower <- c(ALower,rep(-Inf,nExovars^2));
                    AUpper <- c(AUpper,rep(Inf,nExovars^2));

                    # Initials for the persistence matrix
                    A <- c(A,as.vector(vecgX));
                    ALower <- c(ALower,rep(-Inf,nExovars));
                    AUpper <- c(AUpper,rep(Inf,nExovars));
                }
            }

            # Clean and remove NAs
            A <- A[!is.na(A)];
            ALower <- ALower[!is.na(ALower)];
            AUpper <- AUpper[!is.na(AUpper)];

            return(list(A=A,ALower=ALower,AUpper=AUpper));
        }

##### Cost Function for oes #####
        CF <- function(A, modelLags, Etype, Ttype, Stype, occurrence,
                       nComponentsAll, nComponentsNonSeasonal, nExovars, modelLagsMax,
                       persistenceEstimate, initialType, initialSeasonEstimate, xregEstimate, updateX,
                       matvt, vecg, matF, matw, matat, matFX, vecgX, matxt,
                       ot, bounds){

            elements <- oesElements(A, modelLags, Ttype, Stype,
                                    nComponentsAll, nComponentsNonSeasonal, nExovars, modelLagsMax,
                                    persistenceEstimate, initialType, initialSeasonEstimate, xregEstimate, updateX,
                                    matvt, vecg, matF, matw, matat, matFX, vecgX);

            cfRes <- occurrenceOptimizerWrap(elements$matvt, elements$matF, elements$matw, elements$vecg, ot,
                                             modelLags, Etype, Ttype, Stype, occurrence,
                                             matxt, elements$matat, elements$matFX, elements$vecgX,
                                             bounds);

            if(is.nan(cfRes) | is.na(cfRes) | is.infinite(cfRes)){
                cfRes <- 1e+500;
            }

            return(cfRes);
        }
    }

##### Fixed probability #####
    if(occurrence=="f"){
        if(initialType!="o"){
            pt <- ts(matrix(rep(initial,obsInsample),obsInsample,1), start=dataStart, frequency=dataFreq);
        }
        else{
            initial <- iprob;
            pt <- ts(matrix(rep(initial,obsInsample),obsInsample,1), start=dataStart, frequency=dataFreq);
        }
        names(initial) <- "level";
        pForecast <- ts(rep(pt[1],h), start=time(y)[obsInsample]+deltat(y), frequency=dataFreq);
        errors <- ts(ot-iprob, start=dataStart, frequency=dataFreq);

        parametersNumber[1,c(1,4)] <- 1;

        output <- list(fitted=pt, forecast=pForecast, states=pt,
                       nParam=parametersNumber, residuals=errors, actuals=otAll,
                       persistence=matrix(0,1,1,dimnames=list("level",NULL)),
                       initial=initial, initialSeason=NULL);
    }
##### Odds-ratio, inverse and probability models #####
    else if(any(occurrence==c("o","i","p"))){
        if(modelDo=="estimate"){
            # Initialise the model
            basicparams <- oesInitialiser(Etype, Ttype, Stype, damped, occurrence,
                                          dataFreq, obsInsample, obsAll, obsStates, ot,
                                          persistenceEstimate, persistence, initialType, initialValue,
                                          initialSeasonEstimate, initialSeason);
            list2env(basicparams, environment());

            if(damped){
                model <- paste0(Etype,Ttype,"d",Stype);
            }
            else{
                model <- paste0(Etype,Ttype,Stype);
            }

            # Prepare the parameters
            A <- AValues(bounds, Ttype, Stype,
                         modelLagsMax, nComponentsAll, nComponentsNonSeasonal,
                         vecg, matvt, matat);

            if(any(c(persistenceEstimate,initialType=="o",initialSeasonEstimate,xregEstimate))){
                # Run the optimisation
                res <- nloptr(A$A, CF, lb=A$ALower, ub=A$AUpper,
                              opts=list(algorithm=algorithm, xtol_rel=xtol_rel, maxeval=maxeval, print_level=print_level),
                              modelLags=modelLags, Etype=Etype, Ttype=Ttype, Stype=Stype, occurrence=occurrence,
                              nComponentsAll=nComponentsAll, nComponentsNonSeasonal=nComponentsNonSeasonal, nExovars=nExovars, modelLagsMax=modelLagsMax,
                              persistenceEstimate=persistenceEstimate, initialType=initialType, initialSeasonEstimate=initialSeasonEstimate,
                              xregEstimate=xregEstimate, updateX=updateX,
                              matvt=matvt, vecg=vecg, matF=matF, matw=matw, matat=matat, matFX=matFX, vecgX=vecgX, matxt=matxt,
                              ot=ot, bounds=bounds);
                A <- res$solution;
            }

            # Parameters estimated + variance
            parametersNumber[1,1] <- length(A) + 1;

            if(damped){
                phi <- A[nComponentsAll+1];
            }
            else{
                phi <- NA;
            }

            elements <- oesElements(A, modelLags, Ttype, Stype,
                                    nComponentsAll, nComponentsNonSeasonal, nExovars, modelLagsMax,
                                    persistenceEstimate, initialType, initialSeasonEstimate, xregEstimate, updateX,
                                    matvt, vecg, matF, matw, matat, matFX, vecgX);
            matF[] <- elements$matF;
            matw[] <- elements$matw;
            vecg[] <- elements$vecg;
            matFX[] <- elements$matFX;
            vecgX[] <- elements$vecgX;

            # Produce fitted values
            fitting <- occurenceFitterWrap(elements$matvt, matF, matw, vecg, ot,
                                           modelLags, Etype, Ttype, Stype, occurrence,
                                           matxt, elements$matat, matFX, vecgX);
            matvt[] <- fitting$matvt;
            matat[] <- fitting$matat;
            pFitted <- ts(fitting$pfit,start=dataStart,frequency=dataFreq);
            yFitted <- ts(fitting$yfit,start=dataStart,frequency=dataFreq);
            errors <- ts(fitting$errors,start=dataStart,frequency=dataFreq);

            yForecastStart <- time(data)[obsInsample]+deltat(data);

            # Produce forecasts
            if(h>0){
                # yForecast is the underlying forecast, while pForecast is the probability forecast
                pForecast <- yForecast <- as.vector(forecasterwrap(t(matvt[,(obsInsample+1):(obsInsample+modelLagsMax),drop=FALSE]),
                                                                   elements$matF, elements$matw, h, Etype, Ttype, Stype, modelLags,
                                                                   matxt[(obsAll-h+1):(obsAll),,drop=FALSE],
                                                                   t(matat[,(obsAll-h+1):(obsAll),drop=FALSE]), elements$matFX));

                if(Etype=="M" & any(yForecast<=0)){
                    pForecast[pForecast<=0] <- 1E-10;
                    warning(paste0("Negative values were produced in the forecast. ",
                                   "This is unreasonable for the model with the multiplicative error, so we trimmed them out."),
                            call.=FALSE);
                }

                pForecast[] <- switch(occurrence,
                                      "o" = switch(Etype,
                                                   "M"=pForecast/(1+pForecast),
                                                   "A"=exp(pForecast)/(1+exp(pForecast))),
                                      "i" = switch(Etype,
                                                   "M"=1/(1+pForecast),
                                                   "A"=1/(1+exp(pForecast))),
                                      "p" = sapply(sapply(as.vector(pForecast),min,1),max,0));
                pForecast <- ts(pForecast, start=yForecastStart, frequency=dataFreq);
                yForecast <- ts(yForecast, start=yForecastStart, frequency=dataFreq);
            }
            else{
                yForecast <- pForecast <- ts(NA,start=yForecastStart,frequency=dataFreq);
            }

            parametersNumber[1,4] <- sum(parametersNumber[1,1:3]);
            parametersNumber[2,4] <- sum(parametersNumber[2,1:3]);

            if(holdout){
                yHoldout <- ts(otAll[(obsInsample+1):obsAll],start=yForecastStart,frequency=dataFreq);
                errormeasures <- Accuracy(yHoldout,pForecast,ot);
            }
            else{
                yHoldout <- NA;
                errormeasures <- NA;
            }
        }
        else{
            stop("The model selection and combinations are not implemented in oes just yet", call.=FALSE);
        }

        output <- list(fitted=pFitted, forecast=pForecast, states=ts(t(matvt), start=(time(data)[1] - deltat(data)*modelLagsMax), frequency=dataFreq),
                       nParam=parametersNumber, residuals=errors, actuals=otAll,
                       persistence=vecg, phi=phi, initial=matvt[1:nComponentsNonSeasonal,1],
                       initialSeason=matvt[nComponentsAll,1:modelLagsMax], fittedBeta=yFitted, forecastBeta=yForecast);
    }
#### None ####
    else{
        pt <- ts(ot,start=dataStart,frequency=dataFreq);
        pForecast <- ts(rep(ot[obsInsample],h), start=time(y)[obsInsample]+deltat(y),frequency=dataFreq);
        errors <- ts(rep(0,obsInsample), start=dataStart, frequency=dataFreq);
        parametersNumber[] <- 0;
        output <- list(fitted=pt, forecast=pForecast, states=pt,
                       nParam=parametersNumber, residuals=errors, actuals=pt,
                       persistence=NULL, initial=NULL, initialSeason=NULL);
    }
    # Occurrence and model name
    if(!is.null(xreg)){
        modelname <- "oETSX";
    }
    else{
        modelname <- "oETS";
    }
    output$occurrence <- occurrence;
    output$model <- paste0(modelname,"(",model,")");

    ##### Make a plot #####
    if(!silentGraph){
        # if(intervals){
        #     graphmaker(actuals=otAll, forecast=yForecastNew, fitted=pFitted, lower=yLowerNew, upper=yUpperNew,
        #                level=level,legend=!silentLegend,main=output$model);
        # }
        # else{
            graphmaker(actuals=otAll,forecast=output$forecast,fitted=output$fitted,
                       legend=!silentLegend,main=output$model);
        # }
    }

    # Produce log likelihood. It's the same for all the models
    pt <- output$fitted;
    if(any(c(1-pt[ot==0]==0,pt[ot==1]==0))){
        ptNew <- pt[(pt!=0) & (pt!=1)];
        otNew <- ot[(pt!=0) & (pt!=1)];
        output$logLik <- sum(log(ptNew[otNew==1])) + sum(log(1-ptNew[otNew==0]));
    }
    else{
        output$logLik <- (sum(log(pt[ot!=0])) + sum(log(1-pt[ot==0])));
    }

    # The occurrence="none" should have unreasonable likelihood for security reasons
    if(occurrence=="n"){
        output$logLik <- -Inf;
    }
    return(structure(output,class=c("oes","smooth")));
}
