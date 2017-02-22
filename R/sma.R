#' Simple Moving Average
#'
#' Function constructs State-Space simple moving average of predefined order
#'
#' The function constructs ARIMA in the Single Source of Error State-space form
#' (first proposed in Snyder, 1985):
#'
#' \eqn{y_[t] = w' v_[t-l] + \epsilon_[t]}
#'
#' \eqn{v_[t] = F v_[t-1] + g \epsilon_[t]}
#'
#' Where \eqn{v_[t]} is a state vector (defined using \code{order}).
#'
#' @param data Data that needs to be forecasted.
#' @param order Order of simple moving average. If \code{NULL}, then it is
#' selected automatically using information criteria.
#' @param ic Information criterion to use in order selection.
#' @param h Length of forecasting horizon.
#' @param holdout If \code{TRUE}, holdout sample of size \code{h} is taken from
#' the end of the data.
#' @param intervals Type of intervals to construct. This can be:
#'
#' \itemize{ \item \code{none}, aka \code{n} - do not produce prediction
#' intervals.
#'
#' \item \code{parametric}, \code{p} - use state-space structure of ETS. In
#' case of mixed models this is done using simulations, which may take longer
#' time than for the pure additive and pure multiplicative models.
#'
#' \item \code{semiparametric}, \code{sp} - intervals based on covariance
#' matrix of 1 to h steps ahead errors and assumption of normal / log-normal
#' distribution (depending on error type).
#'
#' \item \code{nonparametric}, \code{np} - intervals based on values from a
#' quantile regression on error matrix (see Taylor and Bunn, 1999). The model
#' used in this process is e[j] = a j^b, where j=1,..,h.
#'
#' %\item Finally \code{asymmetric} are based on half moment of distribution.
#' }
#'
#' The parameter also accepts \code{TRUE} and \code{FALSE}. Former means that
#' parametric intervals are constructed, while latter is equivalent to
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
#' @param ...  Other non-documented parameters.  For example parameter
#' \code{model} can accept a previously estimated SMA model and use its
#' parameters.
#' @return Object of class "smooth" is returned. It contains the list of the
#' following values:
#'
#' \itemize{ \item \code{model} - the name of the estimated model.  \item
#' \code{timeElapsed} - time elapsed for the construction of the model.  \item
#' \code{states} - the matrix of the fuzzy components of ssarima, where
#' \code{rows} correspond to time and \code{cols} to states.  \item
#' \code{transition} - matrix F.  \item \code{persistence} - the persistence
#' vector. This is the place, where smoothing parameters live.  \item
#' \code{order} - order of moving average.  \item \code{initialType} - Typetof
#' initial values used.  \item \code{nParam} - number of estimated parameters.
#' \item \code{fitted} - the fitted values of ETS.  \item \code{forecast} - the
#' point forecast of ETS.  \item \code{lower} - the lower bound of prediction
#' interval. When \code{intervals=FALSE} then NA is returned.  \item
#' \code{upper} - the higher bound of prediction interval. When
#' \code{intervals=FALSE} then NA is returned.  \item \code{residuals} - the
#' residuals of the estimated model.  \item \code{errors} - The matrix of 1 to
#' h steps ahead errors.  \item \code{s2} - variance of the residuals (taking
#' degrees of freedom into account).  \item \code{intervals} - type of
#' intervals asked by user.  \item \code{level} - confidence level for
#' intervals.  \item \code{actuals} - the original data.  \item \code{holdout}
#' - the holdout part of the original data.  \item \code{ICs} - values of
#' information criteria of the model. Includes AIC, AICc and BIC.  \item
#' \code{logLik} - log-likelihood of the function.  \item \code{cf} - Cost
#' function value.  \item \code{cfType} - Type of cost function used in the
#' estimation.  \item \code{accuracy} - vector of accuracy measures for the
#' holdout sample. In case of non-intermittent data includes: MPE, MAPE, SMAPE,
#' MASE, sMAE, RelMAE, sMSE and Bias coefficient (based on complex numbers). In
#' case of intermittent data the set of errors will be: sMSE, sPIS, sCE (scaled
#' cumulative error) and Bias coefficient. This is available only when
#' \code{holdout=TRUE}.  }
#' @author Ivan Svetunkov
#' @seealso \code{\link[forecast]{ma}, \link[smooth]{es},
#' \link[smooth]{ssarima}}
#' @references \enumerate{ \item Snyder, R. D., 1985. Recursive Estimation of
#' Dynamic Linear Models. Journal of the Royal Statistical Society, Series B
#' (Methodological) 47 (2), 272-276.  \item Hyndman, R.J., Koehler, A.B., Ord,
#' J.K., and Snyder, R.D. (2008) Forecasting with exponential smoothing: the
#' state space approach, Springer-Verlag.
#' \url{http://www.exponentialsmoothing.net}.  }
#' @keywords SARIMA ARIMA
#' @examples
#'
#' # SMA of specific order
#' ourModel <- sma(rnorm(118,100,3),order=12,h=18,holdout=TRUE,intervals="p")
#'
#' # SMA of arbitrary order
#' ourModel <- sma(rnorm(118,100,3),h=18,holdout=TRUE,intervals="sp")
#'
#' summary(ourModel)
#' forecast(ourModel)
#' plot(forecast(ourModel))
#'
#' @export sma
sma <- function(data, order=NULL, ic=c("AICc","AIC","BIC"),
                h=10, holdout=FALSE,
                intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                silent=c("none","all","graph","legend","output"), ...){
# Function constructs simple moving average in state-space model

#    Copyright (C) 2016  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

# If a previous model provided as a model, write down the variables
    if(exists("model")){
        if(is.null(model$model)){
            stop("The provided model is not Simple Moving Average!",call.=FALSE);
        }
        else if(gregexpr("SMA",model$model)==-1){
            stop("The provided model is not Simple Moving Average!",call.=FALSE);
        }
        else{
            order <- model$order;
        }
    }

    initial <- "backcasting";
    intermittent <- "none";
    bounds <- "admissible";
    cfType <- "MSE";
    xreg <- NULL;
    nExovars <- 1;
    ivar <- 0;

##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- environment();
    ssInput(modelType="sma",ParentEnvironment=environment());

##### Preset y.fit, y.for, errors and basic parameters #####
    y.fit <- rep(NA,obsInsample);
    y.for <- rep(NA,h);
    errors <- rep(NA,obsInsample);
    maxlag <- 1;

# These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

    if(!is.null(order)){
        if(obsInsample < order){
            stop("Sorry, but we don't have enough observations for that order.",call.=FALSE);
        }
    }

# sd of residuals + a parameter... nComponents not included.
    nParam <- 1 + 1;

# Cost function for GES
CF <- function(C){
    fitting <- fitterwrap(matvt, matF, matw, y, vecg,
                          modellags, Etype, Ttype, Stype, initialType,
                          matxt, matat, matFX, vecgX, ot);

    cfRes <- mean(fitting$errors^2);

    return(cfRes);
}

CreatorSMA <- function(silentText=FALSE,...){
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();
    environment(CF) <- environment();

    nComponents <- order;
    #nParam <- nComponents + 1;
    if(order>1){
        matF <- rbind(cbind(rep(1/nComponents,nComponents-1),diag(nComponents-1)),c(1/nComponents,rep(0,nComponents-1)));
        matw <- matrix(c(1,rep(0,nComponents-1)),1,nComponents);
    }
    else{
        matF <- matrix(1,1,1);
        matw <- matrix(1,1,1);
    }
    vecg <- matrix(1/nComponents,nComponents);
    matvt <- matrix(NA,obsStates,nComponents);
    matvt[1:nComponents,1] <- rep(mean(y[1:nComponents]),nComponents);
    if(nComponents>1){
        for(i in 2:nComponents){
            matvt[1:(nComponents-i+1),i] <- matvt[1:(nComponents-i+1)+1,i-1] - matvt[1:(nComponents-i+1),1] * matF[i-1,1];
        }
    }

    modellags <- rep(1,nComponents);

##### Prepare exogenous variables #####
    xregdata <- ssXreg(data=data, xreg=NULL, updateX=FALSE,
                       persistenceX=NULL, transitionX=NULL, initialX=NULL,
                       obsInsample=obsInsample, obsAll=obsAll, obsStates=obsStates, maxlag=maxlag, h=h, silent=silentText);
    matxt <- xregdata$matxt;
    matat <- xregdata$matat;
    matFX <- xregdata$matFX;
    vecgX <- xregdata$vecgX;

    C <- NULL;
    cfObjective <- CF(C);

    ICValues <- ICFunction(nParam=nParam,C=C,Etype=Etype);
    ICs <- ICValues$ICs;
    logLik <- ICValues$llikelihood;
    bestIC <- ICs["AICc"];

    return(list(cfObjective=cfObjective,ICs=ICs,bestIC=bestIC,nParam=nParam,nComponents=nComponents,
                matF=matF,vecg=vecg,matvt=matvt,matw=matw,modellags=modellags,
                matxt=matxt,matat=matat,matFX=matFX,vecgX=vecgX,logLik=logLik));
}

#####Start the calculations#####
    environment(ssForecaster) <- environment();
    environment(ssFitter) <- environment();

    if(is.null(order)){
        maxOrder <- min(36,obsInsample/2);
        ICs <- rep(NA,maxOrder);
        smaValuesAll <- list(NA);
        for(i in 1:maxOrder){
            order <- i;
            smaValuesAll[[i]] <- CreatorSMA(silentText);
            ICs[i] <- smaValuesAll[[i]]$bestIC;
        }
        order <- which(ICs==min(ICs,na.rm=TRUE))[1];
        smaValues <- smaValuesAll[[order]];
    }
    else{
        smaValues <- CreatorSMA(silentText);
    }

    list2env(smaValues,environment());

##### Fit simple model and produce forecast #####
    ssFitter(ParentEnvironment=environment());
    ssForecaster(ParentEnvironment=environment());

##### Do final check and make some preparations for output #####

    if(holdout==T){
        y.holdout <- ts(data[(obsInsample+1):obsAll],start=start(y.for),frequency=frequency(data));
        errormeasures <- errorMeasurer(y.holdout,y.for,y);
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

    modelname <- paste0("SMA(",order,")");

##### Make a plot #####
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

##### Return values #####
    model <- list(model=modelname,timeElapsed=Sys.time()-startTime,
                  states=matvt,transition=matF,persistence=vecg,
                  order=order, initialType=initialType, nParam=nParam,
                  fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,
                  errors=errors.mat,s2=s2,intervals=intervalsType,level=level,
                  actuals=data,holdout=y.holdout,intermittent="none",
                  ICs=ICs,logLik=logLik,cf=cfObjective,cfType=cfType,accuracy=errormeasures);
    return(structure(model,class="smooth"));
}
