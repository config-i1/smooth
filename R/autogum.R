utils::globalVariables(c("silentText","silentGraph","silentLegend","initialType"));

#' Automatic GUM
#'
#' Function selects the order of GUM model based on information criteria,
#' using fancy branch and bound mechanism.
#'
#' The function checks several GUM models (see \link[smooth]{gum} documentation)
#' and selects the best one based on the specified information criterion.
#'
#' The resulting model can be complicated and not straightforward, because GUM
#' allows capturing hidden orders that no ARIMA model can. It is advised to use
#' \code{initial="b"}, because optimising GUM of arbitrary order is not a simple
#' task.
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("gum","smooth")}
#'
#' @template ssBasicParam
#' @template ssAdvancedParam
#' @template ssIntervals
#' @template ssInitialParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssGeneralRef
#' @template ssIntermittentRef
#'
#' @param orders The value of the max order to check. This is the upper bound
#' of orders, but the real orders could be lower than this because of the
#' increasing number of parameters in the models with higher orders.
#' @param lags The value of the maximum lag to check. This should usually be
#' a maximum frequency of the data.
#' @param type Type of model. Can either be \code{"additive"} or
#' \code{"multiplicative"}. The latter means that the GUM is fitted on
#' log-transformed data. If \code{"select"}, then this is selected automatically,
#' which may slow down things twice.
#' @param ...  Other non-documented parameters. For example \code{FI=TRUE} will
#' make the function also produce Fisher Information matrix, which then can be
#' used to calculated variances of parameters of the model.
#' @return Object of class "smooth" is returned. See \link[smooth]{gum} for
#' details.
#' @seealso \code{\link[smooth]{gum}, \link[forecast]{ets}, \link[smooth]{es},
#' \link[smooth]{ces}, \link[smooth]{sim.es}, \link[smooth]{ssarima}}
#'
#' @examples
#'
#' x <- rnorm(50,100,3)
#'
#' # The best GUM model for the data
#' ourModel <- auto.gum(x,orders=2,lags=4,h=18,holdout=TRUE,interval="np")
#'
#' summary(ourModel)
#' forecast(ourModel)
#' plot(forecast(ourModel))
#'
#'
#' @export auto.gum
auto.gum <- function(y, orders=3, lags=frequency(y), type=c("additive","multiplicative","select"),
                     initial=c("backcasting","optimal"), ic=c("AICc","AIC","BIC","BICc"),
                     loss=c("MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                     h=10, holdout=FALSE, cumulative=FALSE,
                     interval=c("none","parametric","likelihood","semiparametric","nonparametric"), level=0.95,
                     bounds=c("restricted","admissible","none"),
                     silent=c("all","graph","legend","output","none"),
                     xreg=NULL, xregDo=c("use","select"), initialX=NULL, ...){
# Function estimates several GUM models and selects the best one using the selected information criterion.
#
#    Copyright (C) 2017 - Inf  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

    ### Depricate the old parameters
    ellipsis <- list(...)
    ellipsis <- depricator(ellipsis, "occurrence", "es");
    ellipsis <- depricator(ellipsis, "oesmodel", "es");
    ellipsis <- depricator(ellipsis, "updateX", "es");
    ellipsis <- depricator(ellipsis, "persistenceX", "es");
    ellipsis <- depricator(ellipsis, "transitionX", "es");
    updateX <- FALSE;
    persistenceX <- transitionX <- NULL;
    occurrence <- "none";
    oesmodel <- "MNN";

# Add all the variables in ellipsis to current environment
    list2env(ellipsis,environment());

    # If this is Mcomp data, then take the frequency from it
    if(any(class(y)=="Mdata") && lags==frequency(y)){
        lags <- frequency(y$x);
    }

##### Set environment for ssInput and make all the checks #####
    environment(ssAutoInput) <- environment();
    ssAutoInput("auto.gum",ParentEnvironment=environment());

    if(any(is.complex(c(orders,lags)))){
        stop("Complex numbers? Really? Be serious! This is GUM, not CES!",call.=FALSE);
    }

    if(any(c(orders)<0)){
        stop("Funny guy! How am I gonna construct a model with negative maximum order?",call.=FALSE);
    }

    if(any(c(lags)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
    }

    if(any(c(lags,orders)==0)){
        stop("Sorry, but we cannot construct GUM model with zero lags / orders.",call.=FALSE);
    }

    type <- substr(type[1],1,1);
    # Check if the multiplictive model is possible
    if(any(type==c("s","m"))){
        if(any(yInSample<=0)){
            warning("Multiplicative model can only be used on positive data. Switching to the additive one.",call.=FALSE);
            type <- "a";
        }
        if(type=="s"){
            type <- c("a","m");
        }
    }

    icsFinal <- rep(NA,length(type));
    lagsFinal <- list(NA);
    ordersFinal <- list(NA);

    if(!silentText){
        if(lags>12){
            message(paste0("You have large lags: ",lags,". So, the calculation may take some time."));
            if(lags<24){
                message(paste0("Go get some coffee, or tea, or whatever, while we do the work here.\n"));
            }
            else{
                message(paste0("Go for a lunch or something, while we do the work here.\n"));
            }
        }
        if(orders>3){
            message(paste0("Beware that you have specified large orders: ",orders,
                           ". This means that the calculations may take a lot of time.\n"));
        }
    }

    for(t in 1:length(type)){
        ics <- rep(NA,lags);
        lagsBest <- NULL

        if((!silentText) & length(type)!=1){
            cat(paste0("Checking model with a type=\"",type[t],"\".\n"));
        }

    #### Preliminary loop ####
        #Checking all the models with lag from 1 to lags
        if(!silentText){
            progressBar <- c("/","\u2014","\\","|");
            cat("Starting preliminary loop: ");
            cat(paste0(rep(" ",9+nchar(lags)),collapse=""));
        }
        for(i in 1:lags){
            gumModel <- gum(y,orders=c(1),lags=c(i),type=type[t],
                            silent=TRUE,h=h,holdout=holdout,
                            initial=initial,loss=loss,
                            cumulative=cumulative,
                            interval=intervalType, level=level,
                            bounds=bounds,
                            xreg=xreg, xregDo=xregDo, initialX=initialX, ...);
            ics[i] <- gumModel$ICs[ic];
            if(!silentText){
                cat(paste0(rep("\b",nchar(paste0(i-1," out of ",lags))),collapse=""));
                cat(paste0(i," out of ",lags));
            }
        }

        ##### Checking all the possible lags ####
        if(!silentText){
            cat(". Done.\n");
            cat("Searching for appropriate lags:  ");
        }
        lagsBest <- c(which(ics==min(ics)),lagsBest);
        icsBest <- 1E100;
        while(min(ics)<icsBest){
            for(i in 1:lags){
                if(!silentText){
                    cat("\b");
                    cat(progressBar[(i/4-floor(i/4))*4+1]);
                }
                if(any(i==lagsBest)){
                    next;
                }
                ordersTest <- rep(1,length(lagsBest)+1);
                lagsTest <- c(i,lagsBest);
                nComponents <- sum(ordersTest);
                nParamMax <- (1 + nComponents + nComponents + (nComponents^2)
                              + (ordersTest %*% lagsTest)*(initialType=="o"));
                if(obsNonzero<=nParamMax){
                    ics[i] <- 1E100;
                    next;
                }
                gumModel <- gum(y,orders=ordersTest,lags=lagsTest,type=type[t],
                                silent=TRUE,h=h,holdout=holdout,
                                initial=initial,loss=loss,
                                cumulative=cumulative,
                                interval=intervalType, level=level,
                                bounds=bounds,
                                xreg=xreg, xregDo=xregDo, initialX=initialX, ...);
                ics[i] <- gumModel$ICs[ic];
            }
            if(!any(which(ics==min(ics))==lagsBest)){
                lagsBest <- c(which(ics==min(ics)),lagsBest);
            }
            icsBest <- min(ics);
        }

        #### Checking all the possible orders ####
        if(!silentText){
            cat("\b");
            cat("We found them!\n");
            cat("Searching for appropriate orders:  ");
        }
        icsBest <- min(ics);
        ics <- array(c(1:(orders^length(lagsBest))),rep(orders,length(lagsBest)));
        ics[1] <- icsBest;
        for(i in 1:length(ics)){
            if(!silentText){
                cat("\b");
                cat(progressBar[(i/4-floor(i/4))*4+1]);
            }
            if(i==1){
                next;
            }
            ordersTest <- which(ics==ics[i],arr.ind=TRUE);
            nComponents <- sum(ordersTest);
            nParamMax <- (1 + nComponents + nComponents + (nComponents^2)
                          + (ordersTest %*% lagsBest)*(initialType=="o"));
            if(obsNonzero<=nParamMax){
                ics[i] <- NA;
                next;
            }
            gumModel <- gum(y,orders=ordersTest,lags=lagsBest,type=type[t],
                            silent=TRUE,h=h,holdout=holdout,
                            initial=initial,loss=loss,
                            cumulative=cumulative,
                            interval=intervalType, level=level,
                            bounds=bounds,
                            xreg=xreg, xregDo=xregDo, initialX=initialX, ...);
            ics[i] <- gumModel$ICs[ic];
        }
        ordersBest <- which(ics==min(ics,na.rm=TRUE),arr.ind=TRUE);
        if(!silentText){
            cat("\b");
            cat("Orders found.\n");
        }

        icsFinal[t] <- min(ics,na.rm=TRUE);
        lagsFinal[[t]] <- lagsBest;
        ordersFinal[[t]] <- ordersBest;
    }
    t <- which(icsFinal==min(icsFinal))[1];

    if(!silentText){
        cat("Reestimating the model. ");
    }

    bestModel <- gum(y,orders=ordersFinal[[t]],lags=lagsFinal[[t]],type=type[t],
                     silent=TRUE,h=h,holdout=holdout,
                     initial=initial,loss=loss,
                     cumulative=cumulative,
                     interval=intervalType, level=level,
                     bounds=bounds,
                     xreg=xreg, xregDo=xregDo, initialX=initialX, ...);

    yFitted <- bestModel$fitted;
    yForecast <- bestModel$forecast;
    yUpper <- bestModel$upper;
    yLower <- bestModel$lower;
    modelname <- bestModel$model;

    bestModel$timeElapsed <- Sys.time()-startTime;

    if(!silentText){
        cat("Done!\n");
    }

##### Make a plot #####
    if(!silentGraph){
        yForecastNew <- yForecast;
        yUpperNew <- yUpper;
        yLowerNew <- yLower;
        if(cumulative){
            yForecastNew <- ts(rep(yForecast/h,h),start=yForecastStart,frequency=dataFreq);
            if(interval){
                yUpperNew <- ts(rep(yUpper/h,h),start=yForecastStart,frequency=dataFreq);
                yLowerNew <- ts(rep(yLower/h,h),start=yForecastStart,frequency=dataFreq);
            }
        }

        if(interval){
            graphmaker(actuals=y,forecast=yForecastNew,fitted=yFitted, lower=yLowerNew,upper=yUpperNew,
                       level=level,legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
        else{
            graphmaker(actuals=y,forecast=yForecastNew,fitted=yFitted,
                       legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
    }

    return(bestModel);
}
