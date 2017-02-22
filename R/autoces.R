utils::globalVariables(c("silentText","silentGraph","silentLegend","initialType"));



#' Complex Exponential Smoothing Auto
#'
#' Function estimates CES in state-space form with information potential equal
#' to errors with different seasonality types and chooses the one with the
#' lowest IC value.
#'
#' The function estimates several Complex Exponential Smoothing in the
#' state-space 2 described in Svetunkov, Kourentzes (2015) with the information
#' potential equal to the approximation error using different types of
#' seasonality and chooses the one with the lowest value of information
#' criterion.
#'
#' @param data Either numeric vector or time series vector.
#' @param models The vector containing several types of seasonality that should
#' be used in CES selection. See \link[smooth]{ces} for more details about the
#' possible types of seasonal models.
#' @param initial Can be either character or a vector of initial states. If it
#' is character, then it can be \code{"optimal"}, meaning that the initial
#' states are optimised, or \code{"backcasting"}, meaning that the initials are
#' produced using backcasting procedure.
#' @param ic The information criterion used in the model selection procedure.
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
#' @param holdout If \code{TRUE}, the holdout sample of size h will be taken
#' from the data. If \code{FALSE}, no holdout is defined.
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
#' @param bounds What type of bounds to use for the smoothing parameters. The
#' first letter can be used instead of the whole word.
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
#' \code{"nothing"} means that all of the data should be used, whilie
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
#' @param ...  Other non-documented parameters.  For example \code{FI=TRUE}
#' will make the function produce Fisher Information matrix, which then can be
#' used to calculated variances of parameters of the model.
#' @return Object of class "smooth" is returned. See \link[smooth]{ces} for
#' details.
#' @author Ivan Svetunkov, \email{ivan@@svetunkov.ru}
#' @seealso \code{\link[smooth]{ces}, \link[forecast]{ets},
#' \link[forecast]{forecast}, \link[stats]{ts}}
#' @references \itemize{
#' \item Svetunkov, I., Kourentzes, N. (February 2015). Complex exponential
#' smoothing. Working Paper of Department of Management Science, Lancaster
#' University 2015:1, 1-31.
#' \item Svetunkov I., Kourentzes N. (2015) Complex Exponential Smoothing
#' for Time Series Forecasting. Not yet published.
#' \item Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008)
#' Forecasting with exponential smoothing: the state space approach,
#' Springer-Verlag. \url{http://www.exponentialsmoothing.net}.
#' \item Svetunkov S. (2012) Complex-Valued Modeling in Economics and
#' Finance. SpringerLink: Bucher. Springer.
#' \item Teunter R., Syntetos A., Babai Z. (2011). Intermittent demand:
#' Linking forecasting to inventory obsolescence. European Journal of
#' Operational Research, 214, 606-615.
#' \item Croston, J. (1972) Forecasting and stock control for intermittent
#' demands. Operational Research Quarterly, 23(3), 289-303.
#' \item Syntetos, A., Boylan J. (2005) The accuracy of intermittent
#' demand estimates. International Journal of Forecasting, 21(2), 303-314.
#' }
#' @keywords ces complex exponential smoothing exponential smoothing
#' forecasting complex variables
#' @examples
#'
#' y <- ts(rnorm(100,10,3),frequency=12)
#' # CES with and without holdout
#' auto.ces(y,h=20,holdout=TRUE)
#' auto.ces(y,h=20,holdout=FALSE)
#'
#' library("Mcomp")
#' \dontrun{y <- ts(c(M3$N0740$x,M3$N0740$xx),start=start(M3$N0740$x),frequency=frequency(M3$N0740$x))
#' # Selection between "none" and "full" seasonalities
#' auto.ces(y,h=8,holdout=TRUE,models=c("n","f"),intervals="p",level=0.8,ic="AIC")}
#'
#' y <- ts(c(M3$N1683$x,M3$N1683$xx),start=start(M3$N1683$x),frequency=frequency(M3$N1683$x))
#' ourModel <- auto.ces(y,h=18,holdout=TRUE,intervals="sp")
#'
#' summary(ourModel)
#' forecast(ourModel)
#' plot(forecast(ourModel))
#'
#' @export auto.ces
auto.ces <- function(data, models=c("none","simple","full"),
                initial=c("backcasting","optimal"), ic=c("AICc","AIC","BIC"),
                cfType=c("MSE","MAE","HAM","MLSTFE","MSTFE","MSEh"),
                h=10, holdout=FALSE,
                intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                intermittent=c("none","auto","fixed","croston","tsb","sba"),
                bounds=c("admissible","none"),
                silent=c("none","all","graph","legend","output"),
                xreg=NULL, xregDo=c("use","select"), initialX=NULL,
                updateX=FALSE, persistenceX=NULL, transitionX=NULL, ...){
# Function estimates several CES models in state-space form with sigma = error,
#  chooses the one with the lowest ic value and returns complex smoothing parameter
#  value, fitted values, residuals, point and interval forecasts, matrix of CES components
#  and values of information criteria

#    Copyright (C) 2015  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

##### Set environment for ssInput and make all the checks #####
    environment(ssAutoInput) <- environment();
    ssAutoInput(modelType="ces",ParentEnvironment=environment());

# If the pool of models is wrong, fall back to default
    if(length(models)!=1){
        modelsOk <- rep(FALSE,length(models));
        for(i in 1:length(models)){
            modelsOk[i] <- any(models[i]==c("n","s","p","f","none","simple","partial","full"));
        }
    }

    if(!all(modelsOk)){
        message("The pool of models includes a strange type of model! Reverting to default pool.");
        models <- c("n","s","p","f");
    }

    datafreq <- frequency(data);
    if(any(is.na(data))){
        if(silentText==FALSE){
            message("Data contains NAs. These observations will be excluded.")
        }
        datanew <- data[!is.na(data)];
        if(is.ts(data)){
            datanew <- ts(datanew,start=start(data),frequency=datafreq);
        }
        data <- datanew;
    }

    if(datafreq==1){
        if(silentText==FALSE){
        message("The data is not seasonal. Simple CES was the only solution here.");
        }

        CESModel <- ces(data, seasonality="n",
                        initial=initialType, ic=ic,
                        cfType=cfType,
                        h=h, holdout=holdout,
                        intervals=intervals, level=level,
                        intermittent=intermittent,
                        bounds=bounds, silent=silent,
                        xreg=xreg, xregDo=xregDo, initialX=initialX,
                        updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, FI=FI);
        return(CESModel);
    }

    if(cfType!="MSE"){
        warning(paste0("'",cfType,"' is used as cost function instead of 'MSE'. The results of model selection may be wrong."),call.=FALSE);
    }

# Check the number of observations and number of parameters.
    if(any(models=="F") & (obsInsample <= datafreq*2 + 2 + 4 + 1)){
        warning("Sorry, but you don't have enough observations for CES(F).",call.=FALSE);
        models <- models[models!="F"];
    }
    if(any(models=="P") & (obsInsample <= datafreq + 2 + 3 + 1)){
        warning("Sorry, but you don't have enough observations for CES(P).",call.=FALSE);
        models <- models[models!="P"];
    }
    if(any(models=="S") & (obsInsample <= datafreq*2 + 2 + 1)){
        warning("Sorry, but you don't have enough observations for CES(S).",call.=FALSE);
        models <- models[models!="S"];
    }

    CESModel <- as.list(models);
    IC.vector <- c(1:length(models));

    j <- 1;
    if(silentText==FALSE){
        cat("Estimating CES with seasonality: ")
    }
    for(i in models){
        if(silentText==FALSE){
            cat(paste0('"',i,'" '));
        }
        CESModel[[j]] <- ces(data, seasonality=i,
                             initial=initialType, ic=ic,
                             cfType=cfType,
                             h=h, holdout=holdout,
                             intervals=intervals, level=level,
                             intermittent=intermittent,
                             bounds=bounds, silent=TRUE,
                             xreg=xreg, xregDo=xregDo, initialX=initialX,
                             updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, FI=FI);
        IC.vector[j] <- CESModel[[j]]$ICs[ic];
        j <- j+1;
    }

    bestModel <- CESModel[[which(IC.vector==min(IC.vector))]];

    y.fit <- bestModel$fitted;
    y.for <- bestModel$forecast;
    y.high <- bestModel$upper;
    y.low <- bestModel$lower;
    modelname <- bestModel$model;

    if(silentText==FALSE){
        best.seasonality <- models[which(IC.vector==min(IC.vector))];
        cat(" \n");
        cat(paste0('The best model is with seasonality = "',best.seasonality,'"\n'));
    }

# Make plot
    if(silentGraph==FALSE){
        if(intervals==TRUE){
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
                       level=level,legend=!silentLegend,main=modelname);
        }
        else{
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                    level=level,legend=!silentLegend,main=modelname);
        }
    }

    bestModel$timeElapsed <- Sys.time()-startTime;

    return(bestModel);
}
