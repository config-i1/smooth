utils::globalVariables(c("silentText","silentGraph","silentLegend","initialType","ar.orders","i.orders","ma.orders"));



#' State-Space ARIMA
#'
#' Function selects the best State-Space ARIMA based on information criteria,
#' using fancy branch and bound mechanism. The resulting model can be not
#' optimal in IC meaning, but it is usually reasonable.
#'
#' The function constructs bunch of ARIMAs in Single Source of Error
#' State-space form (see \link[smooth]{ssarima} documentation) and selects the
#' best one based on information criterion.
#'
#' Due to the flexibility of the model, multiple seasonalities can be used. For
#' example, something crazy like this can be constructed:
#' SARIMA(1,1,1)(0,1,1)[24](2,0,1)[24*7](0,0,1)[24*30], but the estimation may
#' take a lot of time...
#'
#' @param data Data that needs to be forecasted.
#' @param orders List of maximum orders to check, containing vector variables
#' \code{ar}, \code{i} and \code{ma}. If a variable is not provided in the
#' list, then it is assumed to be equal to zero. At least one variable should
#' have the same length as \code{lags}.
#' @param lags Defines lags for the corresponding orders (see examples). The
#' length of \code{lags} must correspond to the length of either
#' \code{ar.orders} or \code{i.orders} or \code{ma.orders}. There is no
#' restrictions on the length of \code{lags} vector.
#' @param combine If \code{TRUE}, then resulting ARIMA is combined using AIC
#' weights.
#' @param workFast If \code{TRUE}, then some of the orders of ARIMA are
#' skipped. This is not advised for models with \code{lags} greater than 12.
#' @param initial Character value which defines how the model is initialised:
#' it can be \code{"optimal"}, meaning that the initial states are optimised,
#' or \code{"backcasting"}, meaning that the initials are produced using
#' backcasting procedure.
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
#' @param ...  Other non-documented parameters. For example \code{FI=TRUE} will
#' make the function also produce Fisher Information matrix, which then can be
#' used to calculated variances of parameters of the model.  Maximum orders to
#' check can also be specified separately, however \code{orders} variable must
#' be set to \code{NULL}: \code{ar.orders} - Maximum order of AR term. Can be
#' vector, defining max orders of AR, SAR etc.  \code{i.orders} - Maximum order
#' of I. Can be vector, defining max orders of I, SI etc.  \code{ma.orders} -
#' Maximum order of MA term. Can be vector, defining max orders of MA, SMA etc.
#' @return Object of class "smooth" is returned. See \link[smooth]{ssarima} for
#' details.
#' @author Ivan Svetunkov
#' @seealso \code{\link[forecast]{ets}, \link[smooth]{es}, \link[smooth]{ces},
#' \link[smooth]{sim.es}, \link[smooth]{ges}, \link[smooth]{ssarima}}
#' @references \enumerate{
#' \item Snyder, R. D., 1985. Recursive Estimation of Dynamic Linear Models.
#' Journal of the Royal Statistical Society, Series B (Methodological) 47 (2), 272-276.
#' \item Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008)
#' Forecasting with exponential smoothing: the state space approach,
#' Springer-Verlag. \url{http://www.exponentialsmoothing.net}.
#' \item Teunter R., Syntetos A., Babai Z. (2011). Intermittent demand:
#' Linking forecasting to inventory obsolescence. European Journal of
#' Operational Research, 214, 606-615.
#' \item Croston, J. (1972) Forecasting and stock control for intermittent
#' demands. Operational Research Quarterly, 23(3), 289-303.
#' \item Syntetos, A., Boylan J. (2005) The accuracy of intermittent demand
#' estimates. International Journal of Forecasting, 21(2), 303-314.
#' }
#' @keywords SARIMA ARIMA
#' @examples
#'
#' x <- rnorm(118,100,3)
#'
#' # The best ARIMA for the data
#' ourModel <- auto.ssarima(x,orders=list(ar=c(2,1),i=c(1,1),ma=c(2,1)),lags=c(1,12),
#'                      h=18,holdout=TRUE,intervals="np")
#'
#' # The other one using optimised states
#' \dontrun{auto.ssarima(x,orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2)),lags=c(1,12),
#'                      initial="o",h=18,holdout=TRUE)}
#'
#' # And now combined ARIMA
#' \dontrun{auto.ssarima(x,orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2)),lags=c(1,12),
#'                       combine=TRUE,h=18,holdout=TRUE)}
#'
#' summary(ourModel)
#' forecast(ourModel)
#' plot(forecast(ourModel))
#'
#'
#' @export auto.ssarima
auto.ssarima <- function(data, orders=list(ar=c(3,3),i=c(2,1),ma=c(3,3)), lags=c(1,frequency(data)),
                         combine=FALSE, workFast=TRUE,
                         initial=c("backcasting","optimal"), ic=c("AICc","AIC","BIC"),
                         cfType=c("MSE","MAE","HAM","MLSTFE","MSTFE","MSEh"),
                         h=10, holdout=FALSE,
                         intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                         intermittent=c("none","auto","fixed","croston","tsb","sba"),
                         bounds=c("admissible","none"),
                         silent=c("none","all","graph","legend","output"),
                         xreg=NULL, xregDo=c("use","select"), initialX=NULL,
                         updateX=FALSE, persistenceX=NULL, transitionX=NULL, ...){
# Function estimates several ssarima models and selects the best one using the selected information criterion.
#
#    Copyright (C) 2015 - 2016  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

    if(!is.null(orders)){
        ar.max <- orders$ar;
        i.max <- orders$i;
        ma.max <- orders$ma;
    }

# If orders are provided in ellipsis via ar.max, write them down.
    if(exists("ar.orders",inherits=FALSE)){
        if(is.null(ar.orders)){
            ar.max <- 0;
        }
        else{
            ar.max <- ar.orders;
        }
    }
    else{
        if(is.null(orders)){
            ar.max <- 0;
        }
    }
    if(exists("i.orders",inherits=FALSE)){
        if(is.null(i.orders)){
            i.max <- 0;
        }
        else{
            i.max <- i.orders;
        }
    }
    else{
        if(is.null(orders)){
            i.max <- 0;
        }
    }
    if(exists("ma.orders",inherits=FALSE)){
        if(is.null(ma.orders)){
            ma.max <- 0;
        }
        else{
            ma.max <- ma.orders
        }
    }
    else{
        if(is.null(orders)){
            ma.max <- 0;
        }
    }

##### Set environment for ssInput and make all the checks #####
    environment(ssAutoInput) <- environment();
    ssAutoInput(modelType="ssarima",ParentEnvironment=environment());

    if(any(is.complex(c(ar.max,i.max,ma.max,lags)))){
        stop("Come on! Be serious! This is ARIMA, not CES!",call.=FALSE);
    }

    if(any(c(ar.max,i.max,ma.max)<0)){
        stop("Funny guy! How am I gonna construct a model with negative order?",call.=FALSE);
    }

    if(any(c(lags)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
    }

    # If there are zero lags, drop them
    if(any(lags==0)){
        ar.max <- ar.max[lags!=0];
        i.max <- i.max[lags!=0];
        ma.max <- ma.max[lags!=0];
        lags <- lags[lags!=0];
    }

    # Define maxorder and make all the values look similar (for the polynomials)
    maxorder <- max(length(ar.max),length(i.max),length(ma.max));
    if(length(ar.max)!=maxorder){
        ar.max <- c(ar.max,rep(0,maxorder-length(ar.max)));
    }
    if(length(i.max)!=maxorder){
        i.max <- c(i.max,rep(0,maxorder-length(i.max)));
    }
    if(length(ma.max)!=maxorder){
        ma.max <- c(ma.max,rep(0,maxorder-length(ma.max)));
    }

    # If zeroes are defined as orders for some lags, drop them.
    if(any((ar.max + i.max + ma.max)==0)){
        orders2leave <- (ar.max + i.max + ma.max)!=0;
        if(all(orders2leave==FALSE)){
            orders2leave <- lags==min(lags);
        }
        ar.max <- ar.max[orders2leave];
        i.max <- i.max[orders2leave];
        ma.max <- ma.max[orders2leave];
        lags <- lags[orders2leave];
    }

    # Order things, so we would deal with the lowest level of seasonality first
    ar.max <- ar.max[order(lags,decreasing=FALSE)];
    i.max <- i.max[order(lags,decreasing=FALSE)];
    ma.max <- ma.max[order(lags,decreasing=FALSE)];
    lags <- sort(lags,decreasing=FALSE);

    # Get rid of duplicates in lags
    if(length(unique(lags))!=length(lags)){
        if(frequency(data)!=1){
            warning(paste0("'lags' variable contains duplicates: (",paste0(lags,collapse=","),"). Getting rid of some of them."),call.=FALSE);
        }
        lags.new <- unique(lags);
        ar.max.new <- i.max.new <- ma.max.new <- lags.new;
        for(i in 1:length(lags.new)){
            ar.max.new[i] <- max(ar.max[which(lags==lags.new[i])]);
            i.max.new[i] <- max(i.max[which(lags==lags.new[i])]);
            ma.max.new[i] <- max(ma.max[which(lags==lags.new[i])]);
        }
        ar.max <- ar.max.new;
        i.max <- i.max.new;
        ma.max <- ma.max.new;
        lags <- lags.new;
    }

# 1 stands for constant, the other one stands for variance
    nParamMax <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;

# Try to figure out if the number of parameters can be tuned in order to fit something smaller on small samples
# Don't try to fix anything if the number of seasonalities is greater than 2
    if(length(lags)<=2){
        if(obsInsample <= nParamMax){
            arma.length <- length(ar.max);
            while(obsInsample <= nParamMax){
                if(any(c(ar.max[arma.length],ma.max[arma.length])>0)){
                    ar.max[arma.length] <- max(0,ar.max[arma.length] - 1);
                    nParamMax <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                    if(obsInsample <= nParamMax){
                        ma.max[arma.length] <- max(0,ma.max[arma.length] - 1);
                        nParamMax <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                    }
                }
                else{
                    if(arma.length==2){
                        ar.max[1] <- ar.max[1] - 1;
                        nParamMax <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                        if(obsInsample <= nParamMax){
                            ma.max[1] <- ma.max[1] - 1;
                            nParamMax <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                        }
                    }
                    else{
                        break;
                    }
                }
                if(all(c(ar.max,ma.max)==0)){
                    if(i.max[arma.length]>0){
                        i.max[arma.length] <- max(0,i.max[arma.length] - 1);
                        nParamMax <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                    }
                    else if(i.max[1]>0){
                        if(obsInsample <= nParamMax){
                            i.max[1] <- max(0,i.max[1] - 1);
                            nParamMax <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
                        }
                    }
                    else{
                        break;
                    }
                }

            }
                nParamMax <- max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags) + sum(ar.max) + sum(ma.max) + 1 + 1;
        }
    }

    if(obsInsample <= nParamMax){
        message(paste0("Not enough observations for the reasonable fit. Number of possible parameters is ",
                        nParamMax," while the number of observations is ",obsInsample,"!"));
        stop("Redefine maximum orders and try again.",call.=FALSE)
    }

# 1 stands for constant/no constant, another one stands for ARIMA(0,0,0)
    nModels <- prod(i.max + 1) * (1 + sum(ma.max*(1 + sum(ar.max)))) + 1;
    testModel <- list(NA);
# Array with elements x maxorders x horizon x point/lower/upper
    if(combine){
        testForecasts <- list(NA);
        testFitted <- list(NA);
        testICs <- list(NA);
        testLevels <- list(NA);
        testStates <- list(NA);
        testTransition <- list(NA);
        testPersistence <- list(NA);
    }
    ICValue <- 1E+100;
    m <- 0;
    constant <- TRUE;

    test.lags <- ma.test <- ar.test <- i.test <- rep(0,length(lags));
    ar.best <- ma.best <- i.best <- rep(0,length(lags));
    ar.best.local <- ma.best.local <- i.best.local <- ar.best;

#### Function corrects IC taking number of parameters on previous step ####
    icCorrector <- function(icValue, nParam, obsInsample, nParamNew){
        if(ic=="AIC"){
            llikelihood <- (2*nParam - icValue)/2;
            correction <- 2*nParamNew - 2*llikelihood;
        }
        else if(ic=="AICc"){
            llikelihood <- (2*nParam*obsInsample/(obsInsample-nParam-1) - icValue)/2;
            correction <- 2*nParamNew*obsInsample/(obsInsample-nParamNew-1) - 2*llikelihood;
        }
        else if(ic=="BIC"){
            llikelihood <- (nParam*log(obsInsample) - icValue)/2;
            correction <- nParamNew*log(obsInsample) - 2*llikelihood;
        }

        return(correction);
    }

    if(silentText==FALSE){
        cat("Estimation progress:     ");
    }

### If for some reason we have model with zeroes for orders, return it.
    if(all(c(ar.max,i.max,ma.max)==0)){
        cat("\b\b\b\bDone!\n");
        bestModel <- ssarima(data,orders=list(ar=ar.best,i=(i.best),ma=(ma.best)),lags=(lags),
                             constant=TRUE,initial=initialType,cfType=cfType,
                             h=h,holdout=holdout,
                             intervals=intervals,level=level,
                             intermittent=intermittent,silent=TRUE,
                             xreg=xreg, xregDo=xregDo, initialX=initialX,
                             updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, FI=FI);
        return(bestModel);
    }

    if(cfType!="MSE"){
        warning(paste0("'",cfType,"' is used as cost function instead of 'MSE'. The results of model selection may be wrong."),call.=FALSE);
    }

    i.orders <- matrix(0,prod(i.max+1),ncol=length(i.max));

##### Loop for differences #####
    if(any(i.max!=0)){
        # Prepare table with differences
        i.orders[,1] <- rep(c(0:i.max[1]),times=prod(i.max[-1]+1));
        if(length(i.max)>1){
            for(seasLag in 2:length(i.max)){
                i.orders[,seasLag] <- rep(c(0:i.max[seasLag]),each=prod(i.max[1:(seasLag-1)]+1))
            }
        }

        # Start the loop with differences
        for(d in 1:nrow(i.orders)){
            m <- m + 1;
            if(silentText==FALSE){
                cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
                cat(paste0(round((m)/nModels,2)*100,"%"));
            }
            nParamOriginal <- 1;
            testModel <- ssarima(data,orders=list(ar=0,i=i.orders[d,],ma=0),lags=lags,
                                 constant=TRUE,initial=initialType,cfType=cfType,
                                 h=h,holdout=holdout,
                                 intervals=intervals,level=level,
                                 intermittent=intermittent,silent=TRUE,
                                 xreg=xreg, xregDo=xregDo, initialX=initialX,
                                 updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, FI=FI);
            ICValue <- testModel$ICs[ic];
            if(combine){
                testForecasts[[m]] <- matrix(NA,h,3);
                testForecasts[[m]][,1] <- testModel$forecast;
                testForecasts[[m]][,2] <- testModel$lower;
                testForecasts[[m]][,3] <- testModel$upper;
                testFitted[[m]] <- testModel$fitted;
                testICs[[m]] <- ICValue;
                testLevels[[m]] <- 1;
                testStates[[m]] <- testModel$states;
                testTransition[[m]] <- testModel$transition;
                testPersistence[[m]] <- testModel$persistence;
            }
            if(silent[1]=="d"){
                cat("I: ");cat(i.orders[d,]);cat(", ");
                cat(ICValue); cat("\n");
            }
            if(m==1){
                bestIC <- ICValue;
                dataI <- testModel$residuals;
                i.best <- i.orders[d,];
                bestICAR <- bestICI <- bestICMA <- bestIC;
            }
            else{
                if(ICValue < bestICI){
                    bestICI <- ICValue;
                    dataI <- testModel$residuals;
                    if(ICValue < bestIC){
                        i.best <- i.orders[d,];
                        bestIC <- ICValue;
                        ma.best <- ar.best <- rep(0,length(ar.test));
                    }
                }
                else{
                    if(workFast){
                        m <- m + sum(ma.max*(1 + sum(ar.max)));
                        next;
                    }
                }
            }

##### Loop for MA #####
            if(any(ma.max!=0)){
                bestICMA <- bestICI;
                ma.best.local <- ma.test <- rep(0,length(ma.test));
                for(seasSelectMA in 1:length(lags)){
                    if(ma.max[seasSelectMA]!=0){
                        for(maSelect in 1:ma.max[seasSelectMA]){
                            m <- m + 1;
                            if(silentText==FALSE){
                                cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
                                cat(paste0(round((m)/nModels,2)*100,"%"));
                            }
                            ma.test[seasSelectMA] <- ma.max[seasSelectMA] - maSelect + 1;
                            nParamMA <- sum(ma.test);
                            nParamNew <- nParamOriginal + nParamMA;

                            testModel <- ssarima(dataI,orders=list(ar=0,i=0,ma=ma.test),lags=lags,
                                                 constant=FALSE,initial=initialType,cfType=cfType,
                                                 h=h,holdout=FALSE,
                                                 intervals=intervals,level=level,
                                                 intermittent=intermittent,silent=TRUE,
                                                 xreg=NULL, xregDo="use", initialX=initialX,
                                                 updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, FI=FI);
                            ICValue <- icCorrector(testModel$ICs[ic], nParamMA, obsInsample, nParamNew);
                            if(combine){
                                testForecasts[[m]] <- matrix(NA,h,3);
                                testForecasts[[m]][,1] <- testModel$forecast;
                                testForecasts[[m]][,2] <- testModel$lower;
                                testForecasts[[m]][,3] <- testModel$upper;
                                testFitted[[m]] <- testModel$fitted;
                                testICs[[m]] <- ICValue;
                                testLevels[[m]] <- 2;
                                testStates[[m]] <- testModel$states;
                                testTransition[[m]] <- testModel$transition;
                                testPersistence[[m]] <- testModel$persistence;
                            }
                            if(silent[1]=="d"){
                                cat("MA: ");cat(ma.test);cat(", ");
                                cat(ICValue); cat("\n");
                            }
                            if(ICValue < bestICMA){
                                bestICMA <- ICValue;
                                ma.best.local <- ma.test;
                                if(ICValue < bestIC){
                                    bestIC <- bestICMA;
                                    i.best <- i.orders[d,];
                                    ma.best <- ma.test;
                                    ar.best <- rep(0,length(ar.test));
                                }
                                dataMA <- testModel$residuals;
                            }
                            else{
                                if(workFast){
                                    m <- m + ma.test[seasSelectMA] * (1 + sum(ar.max)) - 1;
                                    ma.test <- ma.best.local;
                                    break;
                                }
                                else{
                                    ma.test <- ma.best.local;
                                }
                            }

##### Loop for AR #####
                            if(any(ar.max!=0)){
                                bestICAR <- bestICMA;
                                ar.best.local <- ar.test <- rep(0,length(ar.test));
                                for(seasSelectAR in 1:length(lags)){
                                    test.lags[seasSelectAR] <- lags[seasSelectAR];
                                    if(ar.max[seasSelectAR]!=0){
                                        for(arSelect in 1:ar.max[seasSelectAR]){
                                            m <- m + 1;
                                            if(silentText==FALSE){
                                                cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
                                                cat(paste0(round((m)/nModels,2)*100,"%"));
                                            }
                                            ar.test[seasSelectAR] <- ar.max[seasSelectAR] - arSelect + 1;
                                            nParamAR <- sum(ar.test);
                                            nParamNew <- nParamOriginal + nParamMA + nParamAR;

                                            testModel <- ssarima(dataMA,orders=list(ar=ar.test,i=0,ma=0),lags=lags,
                                                                 constant=FALSE,initial=initialType,cfType=cfType,
                                                                 h=h,holdout=FALSE,
                                                                 intervals=intervals,level=level,
                                                                 intermittent=intermittent,silent=TRUE,
                                                                 xreg=NULL, xregDo="use", initialX=initialX,
                                                                 updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, FI=FI);
                                            ICValue <- icCorrector(testModel$ICs[ic], nParamAR, obsInsample, nParamNew);
                                            if(combine){
                                                testForecasts[[m]] <- matrix(NA,h,3);
                                                testForecasts[[m]][,1] <- testModel$forecast;
                                                testForecasts[[m]][,2] <- testModel$lower;
                                                testForecasts[[m]][,3] <- testModel$upper;
                                                testFitted[[m]] <- testModel$fitted;
                                                testICs[[m]] <- ICValue;
                                                testLevels[[m]] <- 3;
                                                testStates[[m]] <- testModel$states;
                                                testTransition[[m]] <- testModel$transition;
                                                testPersistence[[m]] <- testModel$persistence;
                                            }
                                            if(silent[1]=="d"){
                                                cat("AR: ");cat(ar.test);cat(", ");
                                                cat(ICValue); cat("\n");
                                            }
                                            if(ICValue < bestICAR){
                                                bestICAR <- ICValue;
                                                ar.best.local <- ar.test;
                                                if(ICValue < bestIC){
                                                    bestIC <- ICValue;
                                                    i.best <- i.orders[d,];
                                                    ar.best <- ar.test;
                                                    ma.best <- ma.test;
                                                }
                                            }
                                            else{
                                                if(workFast){
                                                    m <- m + ar.test[seasSelectAR] - 1;
                                                    ar.test <- ar.best.local;
                                                    break;
                                                }
                                                else{
                                                    ar.test <- ar.best.local;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    m <- m + 1;
    if(silentText==FALSE){
        cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
        cat(paste0(round((m)/nModels,2)*100,"%"));
    }

#### Test the constant ####
    if(any(c(ar.best,i.best,ma.best)!=0)){
        testModel <- ssarima(data,orders=list(ar=(ar.best),i=(i.best),ma=(ma.best)),lags=(lags),
                             constant=FALSE,initial=initialType,cfType=cfType,
                             h=h,holdout=holdout,
                             intervals=intervals,level=level,
                             intermittent=intermittent,silent=TRUE,
                             xreg=xreg, xregDo=xregDo, initialX=initialX,
                             updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, FI=FI);
        ICValue <- testModel$ICs[ic];
        if(combine){
            testForecasts[[m]] <- matrix(NA,h,3);
            testForecasts[[m]][,1] <- testModel$forecast;
            testForecasts[[m]][,2] <- testModel$lower;
            testForecasts[[m]][,3] <- testModel$upper;
            testFitted[[m]] <- testModel$fitted;
            testICs[[m]] <- ICValue;
            testLevels[[m]] <- 1;
            testStates[[m]] <- testModel$states;
            testTransition[[m]] <- testModel$transition;
            testPersistence[[m]] <- testModel$persistence;
        }
# cat("Constant: ");print(ICValue);
        if(ICValue < bestIC){
            bestModel <- testModel;
            constant <- FALSE;
        }
        else{
            constant <- TRUE;
        }
    }

    if(combine){
        testICs <- unlist(testICs);
        testLevels <- unlist(testLevels);
        testForecasts <- array(unlist(testForecasts),c(h,3,length(testICs)));
        testFitted <- matrix(unlist(testFitted),ncol=length(testICs));
        icWeights <- exp(-0.5*(testICs-min(testICs)))/sum(exp(-0.5*(testICs-min(testICs))));

        testForecastsNew <- testForecasts;
        testFittedNew <- testFitted;
        for(i in 1:length(testLevels)){
            if(testLevels[i]==1){
                j <- i;
            }
            else if(testLevels[i]==2){
                k <- i;
                testForecastsNew[,,i] <- testForecasts[,,j] + testForecasts[,,i];
                testFittedNew[,i] <- testFitted[,j] + testFitted[,i];
            }
            else if(testLevels[i]==3){
                testForecastsNew[,,i] <- testForecasts[,,j] + testForecasts[,,k] + testForecasts[,,i];
                testFittedNew[,i] <- testFitted[,j] + testFitted[,k] + testFitted[,i];
            }
        }
        y.for <- ts(testForecastsNew[,1,] %*% icWeights,start=start(testModel$forecast),frequency=frequency(testModel$forecast));
        y.low <- ts(testForecastsNew[,2,] %*% icWeights,start=start(testModel$lower),frequency=frequency(testModel$lower));
        y.high <- ts(testForecastsNew[,3,] %*% icWeights,start=start(testModel$upper),frequency=frequency(testModel$upper));
        y.fit <- ts(testFittedNew %*% icWeights,start=start(testModel$fitted),frequency=frequency(testModel$fitted));
        modelname <- "ARIMA combined";

        errors <- ts(y-c(y.fit),start=start(y.fit),frequency=frequency(y.fit));
        y.holdout <- ts(data[(obsInsample+1):obsAll],start=start(testModel$forecast),frequency=frequency(testModel$forecast));
        s2 <- mean(errors^2);
        errormeasures <- errorMeasurer(y.holdout,y.for,y);
        ICs <- c(t(testICs) %*% icWeights);
        names(ICs) <- ic;

        bestModel <- list(model=modelname,timeElapsed=Sys.time()-startTime,
                          initialType=initialType,
                          fitted=y.fit,forecast=y.for,
                          lower=y.low,upper=y.high,residuals=errors,s2=s2,intervals=intervals,level=level,
                          actuals=data,holdout=y.holdout,intermittent=intermittent,
                          xreg=xreg, xregDo=xregDo, initialX=initialX,
                          updateX=updateX, persistenceX=persistenceX, transitionX=transitionX,
                          ICs=ICs,ICw=icWeights,cf=NULL,cfType=cfType,accuracy=errormeasures);

        bestModel <- structure(bestModel,class="smooth");
    }
    else{
        if(constant){
            #### Reestimate the best model in order to get rid of bias ####
            bestModel <- ssarima(data,orders=list(ar=(ar.best),i=(i.best),ma=(ma.best)),lags=(lags),
                                 constant=TRUE,initial=initialType,cfType=cfType,
                                 h=h,holdout=holdout,
                                 intervals=intervals,level=level,
                                 intermittent=intermittent,silent=TRUE,
                                 xreg=xreg, xregDo=xregDo, initialX=initialX,
                                 updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, FI=FI);
        }

        y.fit <- bestModel$fitted;
        y.for <- bestModel$forecast;
        y.high <- bestModel$upper;
        y.low <- bestModel$lower;
        modelname <- bestModel$model;

        bestModel$timeElapsed <- Sys.time()-startTime;
    }

    if(silentText==FALSE){
        cat("... Done! \n");
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

    return(bestModel);
}
