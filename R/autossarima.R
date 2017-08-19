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
#' @template ssBasicParam
#' @template ssAdvancedParam
#' @template ssInitialParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssGeneralRef
#' @template ssIntermittentRef
#'
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
#' @param constant If \code{NULL}, then the function will check if constant is
#' needed. if \code{TRUE}, then constant is forced in the model. Otherwise
#' constant is not used.
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
#' @seealso \code{\link[forecast]{ets}, \link[smooth]{es}, \link[smooth]{ces},
#' \link[smooth]{sim.es}, \link[smooth]{ges}, \link[smooth]{ssarima}}
#'
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
                         combine=FALSE, workFast=TRUE, constant=NULL,
                         initial=c("backcasting","optimal"), ic=c("AICc","AIC","BIC"),
                         cfType=c("MSE","MAE","HAM","GMSTFE","MSTFE","MSEh","TFL"),
                         h=10, holdout=FALSE, cumulative=FALSE,
                         intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                         intermittent=c("none","auto","fixed","croston","tsb","sba"), imodel="MNN",
                         bounds=c("admissible","none"),
                         silent=c("all","graph","legend","output","none"),
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
    ssAutoInput("auto.ssarima",ParentEnvironment=environment());

    if(is.null(constant)){
        constantCheck <- TRUE;
        constantValue <- TRUE;
    }
    else{
        if(is.logical(constant)){
            constantCheck <- FALSE;
            constantValue <- constant;
        }
        else{
            constant <- NULL;
            constantCheck <- TRUE;
            constantValue <- TRUE;
            warning("Strange value of constant parameter. We changed it to default value.");
        }
    }

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

    # Get rid of duplicates in lags
    if(length(unique(lags))!=length(lags)){
        if(frequency(data)!=1){
            warning(paste0("'lags' variable contains duplicates: (",paste0(lags,collapse=","),"). Getting rid of some of them."),call.=FALSE);
        }
        lags.new <- unique(lags);
        ar.max.new <- i.max.new <- ma.max.new <- lags.new;
        for(i in 1:length(lags.new)){
            ar.max.new[i] <- max(ar.max[which(lags==lags.new[i])],na.rm=TRUE);
            i.max.new[i] <- max(i.max[which(lags==lags.new[i])],na.rm=TRUE);
            ma.max.new[i] <- max(ma.max[which(lags==lags.new[i])],na.rm=TRUE);
        }
        ar.max <- ar.max.new;
        i.max <- i.max.new;
        ma.max <- ma.max.new;
        lags <- lags.new;
    }

    # Order things, so we would deal with the lowest level of seasonality first
    ar.max <- ar.max[order(lags,decreasing=FALSE)];
    i.max <- i.max[order(lags,decreasing=FALSE)];
    ma.max <- ma.max[order(lags,decreasing=FALSE)];
    lags <- sort(lags,decreasing=FALSE);

# 1 stands for constant, the other one stands for variance
    nParamMax <- (1 + max(ar.max %*% lags + i.max %*% lags,ma.max %*% lags)
                  + sum(ar.max) + sum(ma.max) + constantCheck);

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
    if(all(ma.max==0)){
        nModels <- prod(i.max + 1) * (1 + sum(ar.max)) + constantCheck;
    }
    else{
        nModels <- prod(i.max + 1) * (1 + sum(ma.max*(1 + sum(ar.max)))) + constantCheck;
    }
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
    # constant <- TRUE;

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
        bestModel <- ssarima(data, orders=list(ar=ar.best,i=(i.best),ma=(ma.best)), lags=(lags),
                             constant=constantValue, initial=initialType, cfType=cfType,
                             h=h, holdout=holdout, cumulative=cumulative,
                             intervals=intervals, level=level,
                             intermittent=intermittent, imodel=imodel,
                             bounds=bounds, silent=TRUE,
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
    }

    # Start the loop with differences
    for(d in 1:nrow(i.orders)){
        m <- m + 1;
        if(silentText==FALSE){
            cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
            cat(paste0(round((m)/nModels,2)*100,"%"));
        }
        nParamOriginal <- 1;
        if(silent[1]=="d"){
            cat("I: ");cat(i.orders[d,]);cat(", ");
        }
        testModel <- ssarima(data, orders=list(ar=0,i=i.orders[d,],ma=0), lags=lags,
                             constant=constantValue, initial=initialType, cfType=cfType,
                             h=h, holdout=holdout, cumulative=cumulative,
                             intervals=intervals, level=level,
                             intermittent=intermittent, imodel=imodel,
                             bounds=bounds, silent=TRUE,
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
            cat(ICValue); cat("\n");
        }
        if(m==1){
            bestIC <- ICValue;
            dataMA <- dataI <- testModel$residuals;
            i.best <- i.orders[d,];
            bestICAR <- bestICI <- bestICMA <- bestIC;
        }
        else{
            if(ICValue < bestICI){
                bestICI <- ICValue;
                dataMA <- dataI <- testModel$residuals;
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

                        if(silent[1]=="d"){
                            cat("MA: ");cat(ma.test);cat(", ");
                        }
                        testModel <- ssarima(dataI, orders=list(ar=0,i=0,ma=ma.test), lags=lags,
                                             constant=FALSE, initial=initialType, cfType=cfType,
                                             h=h, holdout=FALSE,
                                             intervals=intervals, level=level,
                                             intermittent=intermittent, imodel=imodel,
                                             bounds=bounds, silent=TRUE,
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

                                        if(silent[1]=="d"){
                                            cat("AR: ");cat(ar.test);cat(", ");
                                        }
                                        testModel <- ssarima(dataMA, orders=list(ar=ar.test,i=0,ma=0), lags=lags,
                                                             constant=FALSE, initial=initialType, cfType=cfType,
                                                             h=h, holdout=FALSE,
                                                             intervals=intervals, level=level,
                                                             intermittent=intermittent, imodel=imodel,
                                                             bounds=bounds, silent=TRUE,
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
        else{
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
                            nParamNew <- nParamOriginal + nParamAR;

                            if(silent[1]=="d"){
                                cat("AR: ");cat(ar.test);cat(", ");
                            }
                            testModel <- ssarima(dataMA, orders=list(ar=ar.test,i=0,ma=0), lags=lags,
                                                 constant=FALSE, initial=initialType, cfType=cfType,
                                                 h=h, holdout=FALSE,
                                                 intervals=intervals, level=level,
                                                 intermittent=intermittent, imodel=imodel,
                                                 bounds=bounds, silent=TRUE,
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

#### Test the constant ####
    if(constantCheck){
        m <- m + 1;
        if(silentText==FALSE){
            cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
            cat(paste0(round((m)/nModels,2)*100,"%"));
        }

        if(any(c(ar.best,i.best,ma.best)!=0)){
            testModel <- ssarima(data, orders=list(ar=(ar.best),i=(i.best),ma=(ma.best)), lags=(lags),
                                 constant=FALSE, initial=initialType, cfType=cfType,
                                 h=h, holdout=holdout, cumulative=cumulative,
                                 intervals=intervals, level=level,
                                 intermittent=intermittent, imodel=imodel,
                                 bounds=bounds, silent=TRUE,
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
                constantValue <- FALSE;
            }
            else{
                constantValue <- TRUE;
            }
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
        y.for <- ts(testForecastsNew[,1,] %*% icWeights,start=start(testModel$forecast),frequency=datafreq);
        y.low <- ts(testForecastsNew[,2,] %*% icWeights,start=start(testModel$lower),frequency=datafreq);
        y.high <- ts(testForecastsNew[,3,] %*% icWeights,start=start(testModel$upper),frequency=datafreq);
        y.fit <- ts(testFittedNew %*% icWeights,start=start(testModel$fitted),frequency=datafreq);
        modelname <- "ARIMA combined";

        errors <- ts(y-c(y.fit),start=start(y.fit),frequency=frequency(y.fit));
        y.holdout <- ts(data[(obsInsample+1):obsAll],start=start(testModel$forecast),frequency=datafreq);
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
        #### Reestimate the best model in order to get rid of bias ####
        bestModel <- ssarima(data, orders=list(ar=(ar.best),i=(i.best),ma=(ma.best)), lags=(lags),
                             constant=constantValue, initial=initialType, cfType=cfType,
                             h=h, holdout=holdout, cumulative=cumulative,
                             intervals=intervals, level=level,
                             intermittent=intermittent, imodel=imodel,
                             bounds=bounds, silent=TRUE,
                             xreg=xreg, xregDo=xregDo, initialX=initialX,
                             updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, FI=FI);

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

##### Make a plot #####
    if(!silentGraph){
        y.for.new <- y.for;
        y.high.new <- y.high;
        y.low.new <- y.low;
        if(cumulative){
            y.for.new <- ts(rep(y.for/h,h),start=start(y.for),frequency=datafreq)
            if(intervals){
                y.high.new <- ts(rep(y.high/h,h),start=start(y.for),frequency=datafreq)
                y.low.new <- ts(rep(y.low/h,h),start=start(y.for),frequency=datafreq)
            }
        }

        if(intervals){
            graphmaker(actuals=data,forecast=y.for.new,fitted=y.fit, lower=y.low.new,upper=y.high.new,
                       level=level,legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
        else{
            graphmaker(actuals=data,forecast=y.for.new,fitted=y.fit,
                       legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
    }

    return(bestModel);
}
