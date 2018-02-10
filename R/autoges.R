utils::globalVariables(c("silentText","silentGraph","silentLegend","initialType","ar.orders","i.orders","ma.orders"));

#' Automatic GES
#'
#' Function selects the order of GES model based on information criteria,
#' using fancy branch and bound mechanism.
#'
#' The function checks several GES models (see \link[smooth]{ges} documentation)
#' and selects the best one based on the specified information criterion.
#'
#' The resulting model can be complicated and not straightforward, because GES
#' allows capturing hidden orders that no ARIMA model can. It is advised to use
#' \code{initial="b"}, because optimising GES of arbitrary order is not a simple
#' task.
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
#' @param orderMax The value of the max order to check. This is the upper bound
#' of orders, but the real orders could be lower than this because of the
#' increasing number of parameters in the models with higher orders.
#' @param lagMax The value of the maximum lag to check. This should usually be
#' a maximum frequency of the data.
#' @param type Type of model. Can either be \code{"Additive"} or
#' \code{"Multiplicative"}. The latter means that the GES is fitted on
#' log-transformed data. If \code{"Z"}, then this is selected automatically,
#' which may slow down things twice.
#' @param ...  Other non-documented parameters. For example \code{FI=TRUE} will
#' make the function also produce Fisher Information matrix, which then can be
#' used to calculated variances of parameters of the model.
#' @return Object of class "smooth" is returned. See \link[smooth]{ges} for
#' details.
#' @seealso \code{\link[smooth]{ges}, \link[forecast]{ets}, \link[smooth]{es},
#' \link[smooth]{ces}, \link[smooth]{sim.es}, \link[smooth]{ssarima}}
#'
#' @examples
#'
#' x <- rnorm(50,100,3)
#'
#' # The best GES model for the data
#' ourModel <- auto.ges(x,orderMax=2,lagMax=4,h=18,holdout=TRUE,intervals="np")
#'
#' summary(ourModel)
#' forecast(ourModel)
#' plot(forecast(ourModel))
#'
#'
#' @export auto.ges
auto.ges <- function(data, orderMax=3, lagMax=frequency(data), type=c("A","M","Z"),
                     initial=c("backcasting","optimal"), ic=c("AICc","AIC","BIC"),
                     cfType=c("MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                     h=10, holdout=FALSE, cumulative=FALSE,
                     intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                     intermittent=c("none","auto","fixed","interval","probability","sba","logistic"),
                     imodel="MNN",
                     bounds=c("admissible","none"),
                     silent=c("all","graph","legend","output","none"),
                     xreg=NULL, xregDo=c("use","select"), initialX=NULL,
                     updateX=FALSE, persistenceX=NULL, transitionX=NULL, ...){
# Function estimates several GES models and selects the best one using the selected information criterion.
#
#    Copyright (C) 2017 - Inf  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

##### Set environment for ssInput and make all the checks #####
    environment(ssAutoInput) <- environment();
    ssAutoInput("auto.ges",ParentEnvironment=environment());

    if(any(is.complex(c(orderMax,lagMax)))){
        stop("Complex numbers? Really? Be serious! This is GES, not CES!",call.=FALSE);
    }

    if(any(c(orderMax)<0)){
        stop("Funny guy! How am I gonna construct a model with negative maximum order?",call.=FALSE);
    }

    if(any(c(lagMax)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
    }

    if(any(c(lagMax,orderMax)==0)){
        stop("Sorry, but we cannot construct GES model with zero lags / orders.",call.=FALSE);
    }

    type <- substr(type[1],1,1);
    if(type=="Z"){
        type <- c("A","M");
    }

    icsFinal <- rep(NA,length(type));
    lagsFinal <- list(NA);
    ordersFinal <- list(NA);

    if(!silentText){
        if(lagMax>12){
            message(paste0("You have large lagMax: ",lagMax,". So, the calculation may take some time."));
            if(lagMax<24){
                message(paste0("Go get some coffee, or tea, or whatever, while we do the work here.\n"));
            }
            else{
                message(paste0("Go for a lunch or something, while we do the work here.\n"));
            }
        }
        if(orderMax>3){
            message(paste0("Beware that you have specified large orderMax: ",orderMax,
                           ". This means that the calculations may take a lot of time.\n"));
        }
    }

    for(t in 1:length(type)){
        ics <- rep(NA,lagMax);
        lagsBest <- NULL

        if((!silentText) & length(type)!=1){
            cat(paste0("Checking model with a type=\"",type[t],"\".\n"));
        }

    #### Preliminary loop ####
        #Checking all the models with lag from 1 to lagMax
        if(!silentText){
            progressBar <- c("/","\u2014","\\","|");
            cat("Starting preliminary loop: ");
            cat(paste0(rep(" ",9+nchar(lagMax)),collapse=""));
        }
        for(i in 1:lagMax){
            gesModel <- ges(data,orders=c(1),lags=c(i),type=type[t],
                            silent=TRUE,h=h,holdout=holdout,
                            initial=initial,cfType=cfType,
                            cumulative=cumulative,
                            intervals=intervals, level=level,
                            intermittent=intermittent, imodel=imodel,
                            bounds=bounds,
                            xreg=xreg, xregDo=xregDo, initialX=initialX,
                            updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, ...);
            ics[i] <- gesModel$ICs[ic];
            if(!silentText){
                cat(paste0(rep("\b",nchar(paste0(i-1," out of ",lagMax))),collapse=""));
                cat(paste0(i," out of ",lagMax));
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
            for(i in 1:lagMax){
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
                gesModel <- ges(data,orders=ordersTest,lags=lagsTest,type=type[t],
                                silent=TRUE,h=h,holdout=holdout,
                                initial=initial,cfType=cfType,
                                cumulative=cumulative,
                                intervals=intervals, level=level,
                                intermittent=intermittent, imodel=imodel,
                                bounds=bounds,
                                xreg=xreg, xregDo=xregDo, initialX=initialX,
                                updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, ...);
                ics[i] <- gesModel$ICs[ic];
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
        ics <- array(c(1:(orderMax^length(lagsBest))),rep(orderMax,length(lagsBest)));
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
            gesModel <- ges(data,orders=ordersTest,lags=lagsBest,type=type[t],
                            silent=TRUE,h=h,holdout=holdout,
                            initial=initial,cfType=cfType,
                            cumulative=cumulative,
                            intervals=intervals, level=level,
                            intermittent=intermittent, imodel=imodel,
                            bounds=bounds,
                            xreg=xreg, xregDo=xregDo, initialX=initialX,
                            updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, ...);
            ics[i] <- gesModel$ICs[ic];
        }
        ordersBest <- which(ics==min(ics,na.rm=TRUE),arr.ind=TRUE)
        if(!silentText){
            cat("\b");
            cat("Orders found.\n");
        }

        icsFinal[t] <- min(ics,na.rm=TRUE);
        lagsFinal[[t]] <- lagsBest;
        ordersFinal[[t]] <- ordersBest;
    }
    t <- which(icsFinal==min(icsFinal));

    if(!silentText){
        cat("Reestimating the model. ");
    }

    bestModel <- ges(data,orders=ordersFinal[[t]],lags=lagsFinal[[t]],type=type[t],
                     silent=TRUE,h=h,holdout=holdout,
                     initial=initial,cfType=cfType,
                     cumulative=cumulative,
                     intervals=intervals, level=level,
                     intermittent=intermittent, imodel=imodel,
                     bounds=bounds,
                     xreg=xreg, xregDo=xregDo, initialX=initialX,
                     updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, ...);

    y.fit <- bestModel$fitted;
    y.for <- bestModel$forecast;
    y.high <- bestModel$upper;
    y.low <- bestModel$lower;
    modelname <- bestModel$model;

    bestModel$timeElapsed <- Sys.time()-startTime;

    if(silentText==FALSE){
        cat("Done!\n");
    }

##### Make a plot #####
    if(!silentGraph){
        y.for.new <- y.for;
        y.high.new <- y.high;
        y.low.new <- y.low;
        if(cumulative){
            y.for.new <- ts(rep(y.for/h,h),start=start(y.for),frequency=datafreq);
            if(intervals){
                y.high.new <- ts(rep(y.high/h,h),start=start(y.for),frequency=datafreq);
                y.low.new <- ts(rep(y.low/h,h),start=start(y.for),frequency=datafreq);
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
