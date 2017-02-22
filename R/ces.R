utils::globalVariables(c("silentText","silentGraph","silentLegend","initialType"));



#' Complex Exponential Smoothing
#'
#' Function estimates CES in state-space form with information potential equal
#' to errors and returns several variables.
#'
#' The function estimates Complex Exponential Smoothing in the state-space 2
#' described in Svetunkov, Kourentzes (2015) with the information potential
#' equal to the approximation error.  The estimation of initial states of xt is
#' done using backcast.
#'
#' @param data Either numeric vector or time series vector.
#' @param seasonality The type of seasonality used in CES. Can be: \code{none}
#' - No seasonality; \code{simple} - Simple seasonality, using lagged CES
#' (based on \code{t-m} observation, where \code{m} is the seasonality lag);
#' \code{partial} - Partial seasonality with real seasonal components
#' (equivalent to additive seasonality); \code{full} - Full seasonality with
#' complex seasonal components (can do both multiplicative and additive
#' seasonality, depending on the data). First letter can be used instead of
#' full words.  Any seasonal CES can only be constructed for time series
#' vectors.
#' @param initial Can be either character or a vector of initial states. If it
#' is character, then it can be \code{"optimal"}, meaning that the initial
#' states are optimised, or \code{"backcasting"}, meaning that the initials are
#' produced using backcasting procedure (advised for data with high frequency).
#' @param A First complex smoothing parameter. Should be a complex number.
#'
#' NOTE! CES is very sensitive to A and B values so it is advised either to
#' leave them alone, or to use values from previously estimated model.
#' @param B Second complex smoothing parameter. Can be real if
#' \code{seasonality="partial"}. In case of \code{seasonality="full"} must be
#' complex number.
#' @param ic Information criterion to use in model selection.
#' @param cfType Type of Cost Function used in optimization. \code{cfType} can
#' be: \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error),
#' \code{HAM} (Half Absolute Moment), \code{MLSTFE} - Mean Log Squared Trace
#' Forecast Error, \code{MSTFE} - Mean Squared Trace Forecast Error and
#' \code{MSEh} - optimisation using only h-steps ahead error. If
#' \code{cfType!="MSE"}, then likelihood and model selection is done based on
#' equivalent \code{MSE}. Model selection in this cases becomes not optimal.
#'
#' There are also available analytical approximations for multistep functions:
#' \code{aMSEh}, \code{aMSTFE} and \code{aMLSTFE}. These can be useful in cases
#' of small samples.
#' @param h The forecasting horizon.
#' @param holdout If \code{TRUE}, holdout sample of size \code{h} is taken from
#' the end of the data.
#' @param intervals Type of intervals to construct. This can be:
#'
#' \itemize{
#' \item \code{none}, aka \code{n} - do not produce prediction intervals.
#' \item \code{parametric}, \code{p} - use state-space structure of ETS. In
#' case of mixed models this is done using simulations, which may take longer
#' time than for the pure additive and pure multiplicative models.
#' \item \code{semiparametric}, \code{sp} - intervals based on covariance
#' matrix of 1 to h steps ahead errors and assumption of normal / log-normal
#' distribution (depending on error type).
#' \item \code{nonparametric}, \code{np} - intervals based on values from a
#' quantile regression on error matrix (see Taylor and Bunn, 1999). The model
#' used in this process is e[j] = a j^b, where j=1,..,h.
#' %\item Finally \code{asymmetric} are based on half moment of distribution.
#' }
#'
#' The parameter also accepts \code{TRUE} and \code{FALSE}. Former means that
#' parametric intervals are constructed, while latter is equivalent to
#' \code{none}.
#' @param level Confidence level. Defines width of prediction interval.
#' @param intermittent Defines type of intermittent model used. Can be: 1.
#' \code{none}, meaning that the data should be considered as non-intermittent;
#' 2. \code{fixed}, taking into account constant Bernoulli distribution of
#' demand occurancies; 3. \code{croston}, based on Croston, 1972 method with
#' SBA correction; 4. \code{tsb}, based on Teunter et al., 2011 method. 5.
#' \code{auto} - automatic selection of intermittency type based on information
#' criteria. The first letter can be used instead. 6. \code{"sba"} -
#' Syntetos-Boylan Approximation for Croston's method (bias correction)
#' discussed in Syntetos and Boylan, 2005.
#' @param bounds What type of bounds to use for smoothing parameters
#' ("admissible" or "usual"). The first letter can be used instead of the whole
#' word.
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
#' @param xreg Vector (either numeric or time series) or matrix (or data.frame)
#' of exogenous variables that should be included in the model. If matrix
#' included than columns should contain variables and rows - observations. Note
#' that \code{xreg} should have number of observations equal either to
#' in-sample or to the whole series. If the number of observations in
#' \code{xreg} is equal to in-sample, then values for the holdout sample are
#' produced using Naive.
#' @param xregDo Variable defines what to do with the provided xreg:
#' \code{"use"} means that all of the data should be used, whilie
#' \code{"select"} means that a selection using \code{ic} should be done.
#' \code{"combine"} will be available at some point in future...
#' @param initialX Vector of initial parameters for exogenous variables.
#' Ignored if \code{xreg} is NULL.
#' @param updateX If \code{TRUE}, transition matrix for exogenous variables is
#' estimated, introducing non-linear interractions between parameters.
#' Prerequisite - non-NULL \code{xreg}.
#' @param persistenceX Persistence vector \eqn{g_X}, containing smoothing
#' parameters for exogenous variables. If \code{NULL}, then estimated.
#' Prerequisite - non-NULL \code{xreg}.
#' @param transitionX Transition matrix \eqn{F_x} for exogenous variables. Can
#' be provided as a vector. Matrix will be formed using the default
#' \code{matrix(transition,nc,nc)}, where \code{nc} is number of components in
#' state vector. If \code{NULL}, then estimated. Prerequisite - non-NULL
#' \code{xreg}.
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
#' \item \code{A} - complex smoothing parameter in the form a0 + ia1
#' \item \code{B} - smoothing parameter for the seasonal component. Can either
#' be real (if \code{seasonality="P"}) or complex (if \code{seasonality="F"})
#' in a form b0 + ib1.
#' \item \code{initialType} - Typetof initial values used.
#' \item \code{initial} - the intial values of the state vector (non-seasonal).
#' \item \code{nParam} - number of estimated parameters.
#' \item \code{fitted} - the fitted values of CES.
#' \item \code{forecast} - the point forecast of CES.
#' \item \code{lower} - the lower bound of prediction interval. When
#' \code{intervals="none"} then NA is returned.
#' \item \code{upper} - the upper bound of prediction interval. When
#' \code{intervals="none"} then NA is returned.
#' \item \code{residuals} - the residuals of the estimated model.
#' \item \code{errors} - The matrix of 1 to h steps ahead errors.
#' \item \code{s2} - variance of the residuals (taking degrees of
#' freedom into account).
#' \item \code{intervals} - type of intervals asked by user.
#' \item \code{level} - confidence level for intervals.
#' \item \code{actuals} - The data provided in the call of the function.
#' \item \code{holdout} - the holdout part of the original data.
#' \item \code{iprob} - the fitted and forecasted values of the probability
#' of demand occurrence.
#' \item \code{intermittent} - type of intermittent model fitted to the data.
#' \item \code{xreg} - provided vector or matrix of exogenous variables. If
#' \code{xregDo="s"}, then this value will contain only selected exogenous
#' variables.
#' \item \code{updateX} - boolean, defining, if the states of
#' exogenous variables were estimated as well.
#' \item \code{initialX} - initial values for parameters of exogenous variables.
#' \item \code{persistenceX} - persistence vector g for exogenous variables.
#' \item \code{transitionX} - transition matrix F for exogenous variables.
#' \item \code{ICs} - values of information criteria of the model. Includes
#' AIC, AICc, BIC and CIC (Complex IC).
#' \item \code{logLik} - log-likelihood of the function.
#' \item \code{cf} - Cost function value.
#' \item \code{cfType} - Type of cost function used in the estimation.
#' \item \code{FI} - Fisher Information. Equal to NULL if \code{FI=FALSE}
#' or when \code{FI} is not provided at all.
#' \item \code{accuracy} - vector of accuracy measures for the holdout sample. In
#' case of non-intermittent data includes: MPE, MAPE, SMAPE, MASE, sMAE,
#' RelMAE, sMSE and Bias coefficient (based on complex numbers). In case of
#' intermittent data the set of errors will be: sMSE, sPIS, sCE (scaled
#' cumulative error) and Bias coefficient. This is available only when
#' \code{holdout=TRUE}.
#' }
#' @author Ivan Svetunkov, \email{ivan@@svetunkov.ru}
#' @seealso \code{\link[forecast]{ets}, \link[forecast]{forecast},
#' \link[stats]{ts}, \link[smooth]{auto.ces}}
#' @references \itemize{
#' \item Svetunkov, I., Kourentzes, N. (February 2015). Complex exponential
#' smoothing. Working Paper of Department of Management Science, Lancaster
#' University 2015:1, 1-31.
#' \item Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008)
#' Forecasting with exponential smoothing: the state space approach,
#' Springer-Verlag. \url{http://www.exponentialsmoothing.net}.
#' \item Svetunkov S. (2012) Complex-Valued Modeling in Economics and Finance.
#' SpringerLink: Bucher. Springer.
#' \item Teunter R., Syntetos A., Babai Z. (2011). Intermittent demand:
#' Linking forecasting to inventory obsolescence. European Journal of
#' Operational Research, 214, 606-615.
#' \item Croston, J. (1972) Forecasting and stock control for intermittent
#' demands. Operational Research Quarterly, 23(3), 289-303.
#' \item Syntetos, A., Boylan J. (2005) The accuracy of intermittent demand
#' estimates. International Journal of Forecasting, 21(2),
#' 303-314.
#' }
#' @keywords ces complex exponential smoothing exponential smoothing
#' forecasting complex variables
#' @examples
#'
#' y <- rnorm(100,10,3)
#' ces(y,h=20,holdout=TRUE)
#' ces(y,h=20,holdout=FALSE)
#'
#' y <- 500 - c(1:100)*0.5 + rnorm(100,10,3)
#' ces(y,h=20,holdout=TRUE,intervals="p",bounds="a")
#'
#' library("Mcomp")
#' y <- ts(c(M3$N0740$x,M3$N0740$xx),start=start(M3$N0740$x),frequency=frequency(M3$N0740$x))
#' ces(y,h=8,holdout=TRUE,seasonality="s",intervals="sp",level=0.8)
#'
#' \dontrun{y <- ts(c(M3$N1683$x,M3$N1683$xx),start=start(M3$N1683$x),frequency=frequency(M3$N1683$x))
#' ces(y,h=18,holdout=TRUE,seasonality="s",intervals="sp")
#' ces(y,h=18,holdout=TRUE,seasonality="p",intervals="np")
#' ces(y,h=18,holdout=TRUE,seasonality="f",intervals="p")}
#'
#' \dontrun{x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)))
#' ces(ts(c(M3$N1457$x,M3$N1457$xx),frequency=12),h=18,holdout=TRUE,
#'     intervals="np",xreg=x,cfType="MSTFE")}
#'
#' # Exogenous variables in CES
#' \dontrun{x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)))
#' ces(ts(c(M3$N1457$x,M3$N1457$xx),frequency=12),h=18,holdout=TRUE,xreg=x)
#' ourModel <- ces(ts(c(M3$N1457$x,M3$N1457$xx),frequency=12),h=18,holdout=TRUE,xreg=x,updateX=TRUE)
#' # This will be the same model as in previous line but estimated on new portion of data
#' ces(ts(c(M3$N1457$x,M3$N1457$xx),frequency=12),model=ourModel,h=18,holdout=FALSE)}
#'
#' # Intermittent data example
#' x <- rpois(100,0.2)
#' # Best type of intermittent model based on iETS(Z,Z,N)
#' ourModel <- ces(x,intermittent="auto")
#'
#' summary(ourModel)
#' forecast(ourModel)
#' plot(forecast(ourModel))
#'
#' @export ces
ces <- function(data, seasonality=c("none","simple","partial","full"),
                initial=c("backcasting","optimal"), A=NULL, B=NULL, ic=c("AICc","AIC","BIC"),
                cfType=c("MSE","MAE","HAM","MLSTFE","MSTFE","MSEh"),
                h=10, holdout=FALSE,
                intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                intermittent=c("none","auto","fixed","croston","tsb","sba"),
                bounds=c("admissible","none"), silent=c("none","all","graph","legend","output"),
                xreg=NULL, xregDo=c("use","select"), initialX=NULL,
                updateX=FALSE, persistenceX=NULL, transitionX=NULL, ...){
# Function estimates CES in state-space form with sigma = error
#  and returns complex smoothing parameter value, fitted values,
#  residuals, point and interval forecasts, matrix of CES components and values of
#  information criteria.
#
#    Copyright (C) 2015 - 2016i  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

    # If a previous model provided as a model, write down the variables
    if(exists("model",inherits=FALSE)){
        if(is.null(model$model)){
            stop("The provided model is not CES.",call.=FALSE);
        }
        else if(gregexpr("ES",model$model)==-1){
            stop("The provided model is not CES.",call.=FALSE);
        }
        intermittent <- model$intermittent;
        if(any(intermittent==c("p","provided"))){
            warning("The provided model had predefined values of occurences for the holdout. We don't have them.",call.=FALSE);
            warning("Switching to intermittent='auto'.",call.=FALSE);
            intermittent <- "a";
        }
        initial <- model$initial;
        A <- model$A;
        B <- model$B;
        if(is.null(xreg)){
            xreg <- model$xreg;
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
    ssInput(modelType="ces",ParentEnvironment=environment());

##### Elements of CES #####
ElementsCES <- function(C){
    vt <- matrix(matvt[1:maxlag,],maxlag);
    n.coef <- 0;
    # No seasonality or Simple seasonality, lagged CES
    if(A$estimate){
        matF[1,2] <- C[2]-1;
        matF[2,2] <- 1-C[1];
        vecg[1:2,] <- c(C[1]-C[2],C[1]+C[2]);
        n.coef <- n.coef + 2;
    }
    else{
        matF[1,2] <- Im(A$value)-1;
        matF[2,2] <- 1-Re(A$value);
        vecg[1:2,] <- c(Re(A$value)-Im(A$value),Re(A$value)+Im(A$value));
    }

    if(seasonality=="p"){
    # Partial seasonality with a real part only
        if(B$estimate){
            vecg[3,] <- C[n.coef+1];
            n.coef <- n.coef + 1;
        }
        else{
            vecg[3,] <- B$value;
        }
    }
    else if(seasonality=="f"){
    # Full seasonality with both real and imaginary parts
        if(B$estimate){
            matF[3,4] <- C[n.coef+2]-1;
            matF[4,4] <- 1-C[n.coef+1];
            vecg[3:4,] <- c(C[n.coef+1]-C[n.coef+2],C[n.coef+1]+C[n.coef+2]);
            n.coef <- n.coef + 2;
        }
        else{
            matF[3,4] <- Im(B$value)-1;
            matF[4,4] <- 1-Re(B$value);
            vecg[3:4,] <- c(Re(B$value)-Im(B$value),Re(B$value)+Im(B$value));
        }
    }

    if(initialType=="o"){
        if(any(seasonality==c("n","s"))){
            vt[1:maxlag,] <- C[n.coef+(1:(2*maxlag))];
            n.coef <- n.coef + maxlag*2;
        }
        else if(seasonality=="p"){
            vt[,1:2] <- rep(C[n.coef+(1:2)],each=maxlag);
            n.coef <- n.coef + 2;
            vt[1:maxlag,3] <- C[n.coef+(1:maxlag)];
            n.coef <- n.coef + maxlag;
        }
        else if(seasonality=="f"){
            vt[,1:2] <- rep(C[n.coef+(1:2)],each=maxlag);
            n.coef <- n.coef + 2;
            vt[1:maxlag,3:4] <- C[n.coef+(1:(maxlag*2))];
            n.coef <- n.coef + maxlag*2;
        }
    }
    else if(initialType=="b"){
        vt[1:maxlag,] <- matvt[1:maxlag,];
    }
    else{
        vt[1:maxlag,] <- initialValue;
    }

# If exogenous are included
    if(xregEstimate){
        at <- matrix(NA,maxlag,nExovars);
        if(initialXEstimate){
            at[,] <- rep(C[n.coef+(1:nExovars)],each=maxlag);
            n.coef <- n.coef + nExovars;
        }
        else{
            at <- matat[1:maxlag,];
        }
        if(updateX){
            if(FXEstimate){
                matFX <- matrix(C[n.coef+(1:(nExovars^2))],nExovars,nExovars);
                n.coef <- n.coef + nExovars^2;
            }

            if(gXEstimate){
                vecgX <- matrix(C[n.coef+(1:nExovars)],nExovars,1);
                n.coef <- n.coef + nExovars;
            }
        }
    }
    else{
        at <- matrix(matat[1:maxlag,],maxlag,nExovars);
    }

    return(list(matF=matF,vecg=vecg,vt=vt,at=at,matFX=matFX,vecgX=vecgX));
}

##### Cost function for CES #####
CF <- function(C){
# Obtain the elements of CES
    elements <- ElementsCES(C);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

    cfRes <- costfunc(matvt, matF, matw, y, vecg,
                      h, modellags, Etype, Ttype, Stype,
                      multisteps, cfType, normalizer, initialType,
                      matxt, matat, matFX, vecgX, ot,
                      bounds);

    if(is.nan(cfRes) | is.na(cfRes)){
        cfRes <- 1e100;
    }
    return(cfRes);
}

##### Estimate ces or just use the provided values #####
CreatorCES <- function(silentText=FALSE,...){
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();

    nParam <- sum(modellags)*(initialType!="b") + A$number + B$number + (!is.null(xreg)) * nExovars + (updateX)*(nExovars^2 + nExovars) + 1;

    if(any(initialType=="o",A$estimate,B$estimate,initialXEstimate,FXEstimate,gXEstimate)){
        C <- NULL;
        # If we don't need to estimate A
        if(A$estimate){
            C <- c(1.3,1);
        }

        if(any(seasonality==c("n","s"))){
            if(initialType=="o"){
                C <- c(C,c(matvt[1:maxlag,]));
            }
        }
        else if(seasonality=="p"){
            if(B$estimate){
                C <- c(C,0.1);
            }
            if(initialType=="o"){
                C <- c(C,c(matvt[1,1:2]));
                C <- c(C,c(matvt[1:maxlag,3]));
            }
        }
        else{
            if(B$estimate){
                C <- c(C,1.3,1);
            }
            if(initialType=="o"){
                C <- c(C,c(matvt[1,1:2]));
                C <- c(C,c(matvt[1:maxlag,3:4]));
            }
        }

        if(xregEstimate){
            if(initialXEstimate){
                C <- c(C,matat[maxlag,]);
            }
            if(updateX){
                if(FXEstimate){
                    C <- c(C,c(diag(nExovars)));
                }
                if(gXEstimate){
                    C <- c(C,rep(0,nExovars));
                }
            }
        }

        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
        C <- res$solution;

        res2 <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-8, "maxeval"=1000));
            # This condition is needed in order to make sure that we did not make the solution worse
        if(res2$objective <= res$objective){
            res <- res2;
        }

        C <- res$solution;
        cfObjective <- res$objective;
    }
    else{
        C <- c(A$value,B$value,initialValue,initialX,transitionX,persistenceX);
        cfObjective <- CF(C);
    }
    if(multisteps){
        cfType <- "aTFL";
    }
    else{
        cfType <- "MSE";
    }
    ICValues <- ICFunction(nParam=nParam+nParamIntermittent,C=C,Etype=Etype);
    ICs <- ICValues$ICs;
    logLik <- ICValues$llikelihood;

    bestIC <- ICs["AICc"];

# Revert to the provided cost function
    cfType <- cfTypeOriginal;

    return(list(cfObjective=cfObjective,C=C,ICs=ICs,bestIC=bestIC,nParam=nParam,logLik=logLik));
}

##### Preset y.fit, y.for, errors and basic parameters #####
    matvt <- matrix(NA,nrow=obsStates,ncol=nComponents);
    y.fit <- rep(NA,obsInsample);
    y.for <- rep(NA,h);
    errors <- rep(NA,obsInsample);

##### Define parameters for different seasonality types #####
    # Define "w" matrix, seasonal complex smoothing parameter, seasonality lag (if it is present).
    #   matvt - the matrix with the components, lags is the lags used in pt matrix.
    if(seasonality=="n"){
        # No seasonality
        matF <- matrix(1,2,2);
        vecg <- matrix(0,2);
        matw <- matrix(c(1,0),1,2);
        matvt <- matrix(NA,obsStates,2);
        colnames(matvt) <- c("level","potential");
        matvt[1,] <- c(mean(yot[1:min(10,obsNonzero)]),mean(yot[1:min(10,obsNonzero)])/1.1);
    }
    else if(seasonality=="s"){
        # Simple seasonality, lagged CES
        matF <- matrix(1,2,2);
        vecg <- matrix(0,2);
        matw <- matrix(c(1,0),1,2);
        matvt <- matrix(NA,obsStates,2);
        colnames(matvt) <- c("level.s","potential.s");
        matvt[1:maxlag,1] <- y[1:maxlag];
        matvt[1:maxlag,2] <- matvt[1:maxlag,1]/1.1;
    }
    else if(seasonality=="p"){
        # Partial seasonality with a real part only
        matF <- diag(3);
        matF[2,1] <- 1;
        vecg <- matrix(0,3);
        matw <- matrix(c(1,0,1),1,3);
        matvt <- matrix(NA,obsStates,3);
        colnames(matvt) <- c("level","potential","seasonal");
        matvt[1:maxlag,1] <- mean(y[1:maxlag]);
        matvt[1:maxlag,2] <- matvt[1:maxlag,1]/1.1;
        matvt[1:maxlag,3] <- decompose(ts(y,frequency=maxlag),type="additive")$figure;
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
        matvt[1:maxlag,1] <- mean(y[1:maxlag]);
        matvt[1:maxlag,2] <- matvt[1:maxlag,1]/1.1;
        matvt[1:maxlag,3] <- decompose(ts(y,frequency=maxlag),type="additive")$figure;
        matvt[1:maxlag,4] <- matvt[1:maxlag,3]/1.1;
    }

##### Prepare exogenous variables #####
    xregdata <- ssXreg(data=data, xreg=xreg, updateX=updateX,
                       persistenceX=persistenceX, transitionX=transitionX, initialX=initialX,
                       obsInsample=obsInsample, obsAll=obsAll, obsStates=obsStates, maxlag=maxlag, h=h, silent=silentText);

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

    # These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

    # Check number of parameters vs data
    nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) + initialXEstimate*ncol(matat);
    nParamMax <- nParamMax + nParamExo + (intermittent!="n");

    ##### Check number of observations vs number of max parameters #####
    if(obsNonzero <= nParamMax){
        if(!silentText){
            message(paste0("Number of non-zero observations is ",obsNonzero,
                           ", while the number of parameters to estimate is ", nParamMax,"."));
        }
        stop("Can't fit the model you ask.",call.=FALSE);
    }

##### Start doing things #####
    environment(intermittentParametersSetter) <- environment();
    environment(intermittentMaker) <- environment();
    environment(ssForecaster) <- environment();
    environment(ssFitter) <- environment();

    # If auto intermittent, then estimate model with intermittent="n" first.
    if(any(intermittent==c("a","n"))){
        intermittentParametersSetter(intermittent="n",ParentEnvironment=environment());
    }
    else{
        intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
        intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    }

    cesValues <- CreatorCES(silentText=silentText);

##### If intermittent=="a", run a loop and select the best one #####
    if(intermittent=="a"){
        if(cfType!="MSE"){
            warning(paste0("'",cfType,"' is used as cost function instead of 'MSE'. A wrong intermittent model may be selected"),call.=FALSE);
        }
        if(!silentText){
            cat("Selecting appropriate type of intermittency... ");
        }
# Prepare stuff for intermittency selection
        intermittentModelsPool <- c("n","f","c","t","s");
        intermittentICs <- rep(1e+10,length(intermittentModelsPool));
        intermittentModelsList <- list(NA);
        intermittentICs <- cesValues$bestIC;

        for(i in 2:length(intermittentModelsPool)){
            intermittentParametersSetter(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentMaker(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentModelsList[[i]] <- CreatorCES(silentText=TRUE);
            intermittentICs[i] <- intermittentModelsList[[i]]$bestIC;
            if(intermittentICs[i]>intermittentICs[i-1]){
                break;
            }
        }
        intermittentICs[is.nan(intermittentICs)] <- 1e+100;
        intermittentICs[is.na(intermittentICs)] <- 1e+100;
        iBest <- which(intermittentICs==min(intermittentICs));

        if(!silentText){
            cat("Done!\n");
        }
        if(iBest!=1){
            intermittent <- intermittentModelsPool[iBest];
            intermittentModel <- intermittentModelsList[[iBest]];
            cesValues <- intermittentModelsList[[iBest]];
        }
        else{
            intermittent <- "n"
        }

        intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
        intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    }

    list2env(cesValues,environment());

    if(xregDo!="u"){
        # Prepare for fitting
        elements <- ElementsCES(C);
        matF <- elements$matF;
        vecg <- elements$vecg;
        matvt[1:maxlag,] <- elements$vt;
        matat[1:maxlag,] <- elements$at;
        matFX <- elements$matFX;
        vecgX <- elements$vecgX;

        # cesValues <- CreatorCES(silentText=TRUE);
        ssFitter(ParentEnvironment=environment());

        xregNames <- colnames(matxtOriginal);
        xregNew <- cbind(errors,xreg[1:nrow(errors),]);
        colnames(xregNew)[1] <- "errors";
        colnames(xregNew)[-1] <- xregNames;
        xregNew <- as.data.frame(xregNew);
        xregResults <- stepwise(xregNew, ic=ic, silent=TRUE, df=nParam+nParamIntermittent-1);
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
    }

# Prepare for fitting
    elements <- ElementsCES(C);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

# Write down Fisher Information if needed
    if(FI){
        environment(likelihoodFunction) <- environment();
        FI <- numDeriv::hessian(likelihoodFunction,C);
    }

##### Fit simple model and produce forecast #####
    ssFitter(ParentEnvironment=environment());
    ssForecaster(ParentEnvironment=environment());

##### Do final check and make some preparations for output #####

# Write down initials of states vector and exogenous
    if(initialType!="p"){
        initialValue <- matvt[1:maxlag,];
    }
    if(initialXEstimate){
        initialX <- matat[1,];
    }

    # Write down the probabilities from intermittent models
    pt <- ts(c(as.vector(pt),as.vector(pt.for)),start=start(data),frequency=datafreq);
    if(intermittent=="f"){
        intermittent <- "fixed";
    }
    else if(intermittent=="c"){
        intermittent <- "croston";
    }
    else if(intermittent=="t"){
        intermittent <- "tsb";
    }
    else if(intermittent=="n"){
        intermittent <- "none";
    }
    else if(intermittent=="p"){
        intermittent <- "provided";
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
    n.coef <- 0;
    if(A$estimate){
        A$value <- complex(real=C[1],imaginary=C[2]);
        n.coef <- 2;
    }

    names(A$value) <- "a0+ia1";

    if(B$estimate){
        if(seasonality=="p"){
            B$value <- C[n.coef+1];
        }
        else if(seasonality=="f"){
            B$value <- complex(real=C[n.coef+1],imaginary=C[n.coef+2]);
        }
    }
    if(B$number!=0){
        if(is.complex(B$value)){
            names(B$value) <- "b0+ib1";
        }
        else{
            names(B$value) <- "b";
        }
    }

    if(!is.null(xreg)){
        modelname <- "CESX";
    }
    else{
        modelname <- "CES";
    }
    modelname <- paste0(modelname,"(",seasonality,")");

    if(all(intermittent!=c("n","none"))){
        modelname <- paste0("i",modelname);
    }

    if(holdout){
        y.holdout <- ts(data[(obsInsample+1):obsAll],start=start(y.for),frequency=datafreq);
        errormeasures <- errorMeasurer(y.holdout,y.for,y);
    }
    else{
        y.holdout <- NA;
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

# Make plot
    if(!silentGraph){
        if(intervals){
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
                       level=level,legend=!silentLegend,main=modelname);
        }
        else{
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                    level=level,legend=!silentLegend,main=modelname);
        }
    }

    model <- list(model=modelname,timeElapsed=Sys.time()-startTime,
                  states=matvt,A=A$value,B=B$value,
                  initialType=initialType,initial=initialValue,
                  nParam=nParam,
                  fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,
                  errors=errors.mat,s2=s2,intervals=intervalsType,level=level,
                  actuals=data,holdout=y.holdout,iprob=pt,intermittent=intermittent,
                  xreg=xreg,updateX=updateX,initialX=initialX,persistenceX=vecgX,transitionX=matFX,
                  ICs=ICs,logLik=logLik,cf=cfObjective,cfType=cfType,FI=FI,accuracy=errormeasures);
    return(structure(model,class="smooth"));
}
