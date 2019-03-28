utils::globalVariables(c("modelDo","initialValue","modelLagsMax","updateX","xregDo"));

#' Occurrence ETS, general model
#'
#' Function returns the general occurrence model of the of iETS model.
#'
#' The function estimates probability of demand occurrence, based on the iETS_G
#' state-space model. It involves the estimation and modelling of the two
#' simultaneous state space equations. Thus two parts for the model type,
#' persistence, intials etc. Although the function accepts all types
#' of ETS models, only the pure multiplicative models make sense.
#'
#' The model is based on:
#'
#' \deqn{o_t \sim Bernoulli(p_t)}
#' \deqn{p_t = \frac{a_t}{a_t+b_t}},
#'
#' where a_t and b_t are the parameters of the Beta distribution and are modelled
#' using separate ETS models.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param data Either numeric vector or time series vector.
#' @param modelA The type of the ETS for the model A.
#' @param modelB The type of the ETS for the model B.
#' @param persistenceA The persistence vector \eqn{g}, containing smoothing
#' parameters used in the model A. If \code{NULL}, then estimated.
#' @param persistenceB The persistence vector \eqn{g}, containing smoothing
#' parameters used in the model B. If \code{NULL}, then estimated.
#' @param phiA The value of the dampening parameter in the model A. Used only
#' for damped-trend models.
#' @param phiB The value of the dampening parameter in the model B. Used only
#' for damped-trend models.
#' @param initialA Either \code{"o"} - optimal or the vector of initials for the
#' level and / or trend for the model A.
#' @param initialB Either \code{"o"} - optimal or the vector of initials for the
#' level and / or trend for the model B.
#' @param initialSeasonA The vector of the initial seasonal components for the
#' model A. If \code{NULL}, then it is estimated.
#' @param initialSeasonB The vector of the initial seasonal components for the
#' model B. If \code{NULL}, then it is estimated.
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
#' @param xregA The vector or the matrix of exogenous variables, explaining some parts
#' of occurrence variable of the model A.
#' @param xregB The vector or the matrix of exogenous variables, explaining some parts
#' of occurrence variable of the model B.
#' @param xregDoA Variable defines what to do with the provided \code{xregA}:
#' \code{"use"} means that all of the data should be used, while
#' \code{"select"} means that a selection using \code{ic} should be done.
#' @param xregDoB Similar to the \code{xregDoA}, but for the part B of the model.
#' @param initialXA The vector of initial parameters for exogenous variables in the model
#' A. Ignored if \code{xregA} is NULL.
#' @param initialXB The vector of initial parameters for exogenous variables in the model
#' B. Ignored if \code{xregB} is NULL.
#' @param updateXA If \code{TRUE}, transition matrix for exogenous variables is
#' estimated, introducing non-linear interactions between parameters.
#' Prerequisite - non-NULL \code{xregA}.
#' @param updateXB If \code{TRUE}, transition matrix for exogenous variables is
#' estimated, introducing non-linear interactions between parameters.
#' Prerequisite - non-NULL \code{xregB}.
#' @param persistenceXA The persistence vector \eqn{g_X}, containing smoothing
#' parameters for the exogenous variables of the model A. If \code{NULL}, then estimated.
#' Prerequisite - non-NULL \code{xregA}.
#' @param persistenceXB The persistence vector \eqn{g_X}, containing smoothing
#' parameters for the exogenous variables of the model B. If \code{NULL}, then estimated.
#' Prerequisite - non-NULL \code{xregB}.
#' @param transitionXA The transition matrix \eqn{F_x} for exogenous variables of the model A.
#' Can be provided as a vector. Matrix will be formed using the default
#' \code{matrix(transition,nc,nc)}, where \code{nc} is number of components in
#' state vector. If \code{NULL}, then estimated. Prerequisite - non-NULL
#' \code{xregA}.
#' @param transitionXB The transition matrix \eqn{F_x} for exogenous variables of the model B.
#' Similar to the \code{transitionXA}.
#' @param ... The parameters passed to the optimiser, such as \code{maxeval},
#' \code{xtol_rel}, \code{algorithm} and \code{print_level}. The description of
#' these is printed out by \code{nloptr.print.options()} function from the \code{nloptr}
#' package. The default values in the oes function are \code{maxeval=500},
#' \code{xtol_rel=1E-8}, \code{algorithm="NLOPT_LN_SBPLX"} and \code{print_level=0}.
#' @return The object of class "occurrence" is returned. It contains following list of
#' values:
#'
#' \itemize{
#' \item \code{modelA} - the model A of the class oesg, that contains the output similar
#' to the one from the \code{oes()} function;
#' \item \code{modelB} - the model B of the class oesg, that contains the output similar
#' to the one from the \code{oes()} function.
#' }
#' @seealso \code{\link[smooth]{es}, \link[smooth]{oes}}
#' @keywords iss intermittent demand intermittent demand state space model
#' exponential smoothing forecasting
#' @examples
#'
#' y <- rpois(100,0.1)
#' # oesg(y, occurrence="g")
#'
#'
oesg <- function(data, modelA="MNN", modelB="MNN", persistenceA=NULL, persistenceB=NULL,
                 phiA=NULL, phiB=NULL,
                 initialA="o", initialB="o", initialSeasonA=NULL, initialSeasonB=NULL,
                 ic=c("AICc","AIC","BIC","BICc"), h=10, holdout=FALSE,
                 intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                 bounds=c("usual","admissible","none"),
                 silent=c("all","graph","legend","output","none"),
                 xregA=NULL, xregB=NULL, initialXA=NULL, initialXB=NULL,
                 xregDoA=c("use","select"), xregDoB=c("use","select"),
                 updateXA=FALSE, updateXB=FALSE, transitionXA=NULL, transitionXB=NULL,
                 persistenceXA=NULL, persistenceXB=NULL,
                 ...){
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
    xregdata <- ssXreg(data=otAll, Etype="A", xreg=xreg, updateX=updateX, ot=rep(1,obsInsample),
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

    # The start time for the forecasts
    yForecastStart <- time(data)[obsInsample]+deltat(data);

    #### The functions for the O, I, and P models ####
    if(any(occurrence==c("o","i","p"))){
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
            if(modelIsSeasonal){
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
                vecg <- matrix(0.01, nComponentsAll, 1);
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
                initialStates[1] <- mean(ot);
                if(Ttype=="M"){
                    initialStates[2] <- 1;
                }
                else if(Ttype=="A"){
                    initialStates[2] <- 0;
                }
                if(occurrence=="o"){
                    initialStates[1] <- initialStates[1] / (1 - initialStates[1]);
                }
                else if(occurrence=="i"){
                    initialStates[1] <- (1-initialStates[1]) / initialStates[1];
                }
                # Initials specifically for ETS(A,M,N) and alike
                if(Etype=="A" && any(occurrence==c("o","i")) && Ttype=="M" && (initialStates[1]>1)){
                    initialStates[1] <- log(initialStates[1]);
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
            if(modelIsSeasonal){
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
                        # Transform this into the probability scale
                        initialSeasonValue <- (initialSeasonValue + 1) / 2;
                        if(occurrence=="o"){
                            initialSeasonValue <- initialSeasonValue / (1 - initialSeasonValue);
                        }
                        else{
                            initialSeasonValue <- (1 - initialSeasonValue) / initialSeasonValue;
                        }

                        # Transform to the adequate scale
                        if(Stype=="A"){
                            initialSeasonValue <- log(initialSeasonValue);
                        }
                    }
                    else{
                        if(Stype=="M"){
                            initialSeasonValue <- exp(initialSeasonValue);
                        }
                    }

                    # Normalise the initials
                    if(Stype=="A"){
                        initialSeasonValue <- initialSeasonValue - mean(initialSeasonValue);
                    }
                    else{
                        initialSeasonValue <- exp(log(initialSeasonValue) - mean(log(initialSeasonValue)));
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
                matvt[1:nComponentsNonSeasonal,1:modelLagsMax] <- A[i+c(1:nComponentsNonSeasonal)];
                i[] <- i + nComponentsNonSeasonal;
            }
            if(modelIsSeasonal && initialSeasonEstimate){
                matvt[nComponentsAll,1:modelLagsMax] <- A[i+c(1:modelLagsMax)];
                i[] <- i + modelLagsMax;
            }
            if(xregEstimate){
                matat[,1:modelLagsMax] <- A[i+c(1:nExovars)];
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
                    if(modelIsSeasonal){
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
                    if(modelIsSeasonal){
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
                    if(modelIsSeasonal){
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

            #### Explanatory variables ####
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
        model <- "MNN";
        if(initialType!="o"){
            pt <- ts(matrix(rep(initial,obsInsample),obsInsample,1), start=dataStart, frequency=dataFreq);
        }
        else{
            initial <- iprob;
            pt <- ts(matrix(rep(initial,obsInsample),obsInsample,1), start=dataStart, frequency=dataFreq);
        }
        names(initial) <- "level";
        pForecast <- ts(rep(pt[1],h), start=yForecastStart, frequency=dataFreq);
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

            #### Start the optimisation ####
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

            # Parameters estimated. The variance is not estimated, so not needed
                parametersNumber[1,1] <- length(A);
            }

            if(damped){
                phi <- A[nComponentsAll+1];
            }
            else{
                phi <- NA;
            }

            ##### Deal with the fitting and the forecasts #####
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

            if(fitting$warning){
                warning(paste0("Unreasonable values of states were produced in the estimation. ",
                               "So, we substituted them with the previous values.\nThis is because the model ETS(",model,") is unstable."),
                        call.=FALSE);
            }

            # Produce forecasts
            if(h>0){
                # yForecast is the underlying forecast, while pForecast is the probability forecast
                pForecast <- yForecast <- as.vector(forecasterwrap(t(matvt[,(obsInsample+1):(obsInsample+modelLagsMax),drop=FALSE]),
                                                                   elements$matF, elements$matw, h, Etype, Ttype, Stype, modelLags,
                                                                   matxt[(obsAll-h+1):(obsAll),,drop=FALSE],
                                                                   t(matat[,(obsAll-h+1):(obsAll),drop=FALSE]), elements$matFX));

                if(Etype=="M" && any(yForecast<0)){
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
                # This is usually due to the exp(big number), which corresponds to 1
                if(any(occurrence==c("i","o")) && Etype=="A" && any(is.nan(pForecast))){
                    pForecast[is.nan(pForecast)] <- 1;
                }

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

        # Merge states of vt and at if the xreg was provided
        if(!is.null(xreg)){
            matvt <- rbind(matvt,matat);
            xreg <- matxt;
        }

        output <- list(fitted=pFitted, forecast=pForecast, states=ts(t(matvt), start=(time(data)[1] - deltat(data)*modelLagsMax), frequency=dataFreq),
                       nParam=parametersNumber, residuals=errors, actuals=otAll,
                       persistence=vecg, phi=phi, initial=matvt[1:nComponentsNonSeasonal,1],
                       initialSeason=matvt[nComponentsAll,1:modelLagsMax], fittedBeta=yFitted, forecastBeta=yForecast,
                       initialX=matat[,1], xreg=xreg);
    }
#### Auto ####
    else if(occurrence=="a"){
        IC <- switch(ic,
                     "AIC"=AIC,
                     "AICc"=AICc,
                     "BIC"=BIC,
                     "BICc"=BICc);

        # occurrencePool <- c("f","o","i","p","g");
        occurrencePool <- c("f","o","i","p");
        occurrencePoolLength <- length(occurrencePool);
        occurrenceModels <- vector("list",occurrencePoolLength);
        for(i in 1:occurrencePoolLength){
            occurrenceModels[[i]] <- oes(data=data,model=model,occurrence=occurrencePool[i],
                                         ic=ic, h=h, holdout=holdout,
                                         intervals=intervals, level=level,
                                         bounds=bounds,
                                         silent=TRUE,
                                         xreg=xreg, xregDo=xregDo, updateX=updateX, ...);
        }
        ICBest <- which.min(sapply(occurrenceModels, IC))[1]
        occurrence <- occurrencePool[ICBest];

        if(!silentGraph){
            graphmaker(actuals=otAll,forecast=occurrenceModels[[ICBest]]$forecast,fitted=occurrenceModels[[ICBest]]$fitted,
                       legend=!silentLegend,main=paste0(occurrenceModels[[ICBest]]$model,"_",toupper(occurrence)));
        }
        return(occurrenceModels[[ICBest]]);

        # return(oes(data=data,model=modelType(occurrenceModels[[ICBest]]),occurrence=occurrence,
        #            ic=ic, h=h, holdout=holdout,
        #            intervals=intervals, level=level,
        #            bounds=bounds,
        #            silent=silent,
        #            xreg=xreg, xregDo=xregDo, updateX=updateX, ...));
    }
#### None ####
    else{
        pt <- ts(ot,start=dataStart,frequency=dataFreq);
        pForecast <- ts(rep(ot[obsInsample],h), start=yForecastStart, frequency=dataFreq);
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
                       legend=!silentLegend,main=paste0(output$model,"_",toupper(occurrence)));
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
