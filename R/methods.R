# forecast <- function(object, ...) UseMethod("forecast")


#' Corrected Akaike's Information Criterion
#'
#' This function extracts AICc from "smooth" objects.
#'
#' AICc was proposed by Nariaki Sugiura in 1978 and is used on small samples.
#'
#' @aliases AICc AICc.default
#' @param object Time series model.
#' @param ...  Some stuff.
#' @return This function returns numeric value.
#' @author Ivan Svetunkov, \email{ivan@@svetunkov.ru}
#' @seealso \link[stats]{AIC}, \link[stats]{BIC}
#' @references Kenneth P. Burnham, David R. Anderson (1998). Model Selection
#' and Multimodel Inference. Springer Science & Business Media.
#' @keywords information criteria information criterion
#' @examples
#'
#' ourModel <- ces(rnorm(100,0,1),h=10)
#'
#' AICc(ourModel,h=10)
#'
#' @export AICc
AICc <- function(object, ...) UseMethod("AICc")


#' Functions that extract values from the fitted model
#'
#' These functions allow extracting orders and lags for \code{ssarima()}, \code{ges()} and \code{sma()}
#' and type of model from \code{es()} and \code{ces()}.
#'
#' \code{orders()} and \code{lags()} are usefull only for SSARIMA, GES and SMA. They return \code{NA} for other functions.
#' This can also be applied to \code{arima()}, \code{Arima()} and \code{auto.arima()} functions from stats and forecast packages.
#' \code{model.type()} is usefull only for ETS and CES. They return \code{NA} for other functions.
#' This can also be applied to \code{ets()} function from forecast package.
#'
#' @aliases orders orders.default
#' @param object Model estimated using one of the functions of smooth package.
#' @param ... Currently nothing is accepted via ellipsis.
#' @return     Either vector or list with values is returned.
#' \code{orders()} in case of ssarima returns list of values:
#' \itemize{
#' \item \code{ar} - AR orders.
#' \item \code{i} - I orders.
#' \item \code{ma} - MA orders.
#' }
#' @author Ivan Svetunkov, \email{ivan@@svetunkov.ru}
#' @seealso \link[forecast]{forecast}, \link[smooth]{ssarima}
#' @keywords forecasting
#' @examples
#'
#' x <- rnorm(100,0,1)
#'
#' # Just as example. orders and lags do not return anything for ces() and es(). But model.type does.
#' ourModel <- ces(x, h=10)
#' orders(ourModel)
#' lags(ourModel)
#' model.type(ourModel)
#' # And as another example it does the opposite for ges() and ssarima()
#' ourModel <- ges(x, h=10, orders=c(1,1), lags=c(1,4))
#' orders(ourModel)
#' lags(ourModel)
#' model.type(ourModel)
#'
#' # Finally these values can be used for simulate functions or original functions.
#' ourModel <- auto.ssarima(x)
#' ssarima(x, orders=orders(ourModel), lags=lags(ourModel), constant=ourModel$constant)
#' sim.ssarima(orders=orders(ourModel), lags=lags(ourModel), constant=ourModel$constant)
#'
#' @rdname orders
#' @export orders
orders <- function(object, ...) UseMethod("orders")

#' @aliases lags.default
#' @rdname orders
#' @export lags
lags <- function(object, ...) UseMethod("lags")

#' @aliases model.type.default
#' @rdname orders
#' @export model.type
model.type <-  function(object, ...) UseMethod("model.type")

##### Likelihood function and stuff #####
#' @importFrom stats logLik
#' @export
logLik.smooth <- function(object,...){
    obs <- nobs(object);
    structure(object$logLik,nobs=obs,df=object$nParam,class="logLik");
}
#' @export
logLik.smooth.sim <- function(object,...){
    obs <- nobs(object);
    structure(object$logLik,nobs=obs,df=0,class="logLik");
}
#' @export
logLik.iss <- function(object,...){
    obs <- nobs(object);
    structure(object$logLik,nobs=obs,df=object$nParam,class="logLik");
}

#' @importFrom stats nobs
#' @method nobs smooth
#' @export
nobs.smooth <- function(object, ...){
    return(length(object$fitted));
}
#' @method nobs smooth.sim
#' @export
nobs.smooth.sim <- function(object, ...){
    if(is.null(dim(object$data))){
        return(length(object$data));
    }
    else{
        return(nrow(object$data));
    }
}
#' @method nobs iss
#' @export
nobs.iss <- function(object, ...){
    return(length(object$fitted));
}

##### IC functions #####
#' @export
AICc.default <- function(object, ...){
        obs <- nobs(object);

    llikelihood <- logLik(object);
    nParam <- attributes(llikelihood)$df;
    llikelihood <- llikelihood[1:length(llikelihood)];

    IC <- 2*nParam - 2*llikelihood + 2 * nParam * (nParam + 1) / (obs - nParam - 1);

    return(IC);
}

#### Extraction of parameters of models ####
#' @export
coef.smooth <- function(object, ...)
{
    if(gregexpr("CES",object$model)!=-1){
        parameters <- c(object$A,object$B);
    }
    else if(gregexpr("ETS",object$model)!=-1){
        if(any(unlist(gregexpr("C",object$model))==-1)){
            # If this was normal ETS, return values
            parameters <- c(object$persistence,object$initial,object$initial.season,object$initialX);
        }
        else{
            # If we did combinations, we cannot return anything
            message("Combination of models was done, so there are no coefficients to return");
            parameters <- NULL;
        }
    }
    else if(gregexpr("GES",object$model)!=-1){
        parameters <- c(object$measurement,object$transition,object$persistence,object$initial);
        names(parameters) <- c(paste0("Measurement ",c(1:length(object$measurement))),
                               paste0("Transition ",c(1:length(object$transition))),
                               paste0("Persistence ",c(1:length(object$persistence))),
                               paste0("Initial ",c(1:length(object$initial))));
    }
    else if(gregexpr("ARIMA",object$model)!=-1){
        if(any(unlist(gregexpr("combine",object$model))==-1)){
            # If this was normal ARIMA, return values
            namesConstant <- NamesMA <- NamesAR <- parameters <- NULL;
            if(any(object$AR!=0)){
                parameters <- c(parameters,object$AR);
                NamesAR <- paste(rownames(object$AR),rep(colnames(object$AR),each=ncol(object$AR)),sep=", ");
            }
            if(any(object$MA!=0)){
                parameters <- c(parameters,object$MA);
                NamesMA <- paste(rownames(object$MA),rep(colnames(object$MA),each=ncol(object$MA)),sep=", ")
            }
            if(object$constant!=0){
                parameters <- c(parameters,object$constant);
                namesConstant <- "Constant";
            }
            names(parameters) <- c(NamesAR,NamesMA,namesConstant);
            parameters <- parameters[parameters!=0];
        }
        else{
            # If we did combinations, we cannot return anything
            message("Combination of models was done, so there are no coefficients to return");
            parameters <- NULL;
        }
    }
    else if(gregexpr("SMA",object$model)!=-1){
        parameters <- object$persistence;
    }

    return(parameters);
}

#' @importFrom forecast getResponse
#' @export
forecast::getResponse


#### Fitted, forecast and actual values ####
#' @export
fitted.smooth <- function(object, ...){
    return(object$fitted);
}

#' @importFrom forecast forecast
#' @export forecast
NULL

#' Forecasting time series using smooth functions
#'
#' This function is created in order for the package to be compatible with Rob
#' Hyndman's "forecast" package
#'
#' This is not a compulsary function. You can simply use \link[smooth]{es},
#' \link[smooth]{ces}, \link[smooth]{ges} or \link[smooth]{ssarima} without
#' \code{forecast.smooth}. But if you are really used to \code{forecast}
#' function, then go ahead!
#'
#' @aliases forecast forecast.smooth
#' @param object Time series model for which forecasts are required.
#' @param h Forecast horizon
#' @param intervals Type of intervals to construct. See \link[smooth]{es} for
#' details.
#' @param level Confidence level. Defines width of prediction interval.
#' @param ...  Other arguments accepted by either \link[smooth]{es},
#' \link[smooth]{ces}, \link[smooth]{ges} or \link[smooth]{ssarima}.
#' @return Returns object of class "smooth.forecast", which contains:
#'
#' \itemize{
#' \item \code{model} - the estimated model (ES / CES / GES / SSARIMA).
#' \item \code{method} - the name of the estimated model (ES / CES / GES / SSARIMA).
#' \item \code{fitted} - fitted values of the model.
#' \item \code{actuals} - actuals provided in the call of the model.
#' \item \code{forecast} aka \code{mean} - point forecasts of the model
#' (conditional mean).
#' \item \code{lower} - lower bound of prediction intervals.
#' \item \code{upper} - upper bound of prediction intervals.
#' \item \code{level} - confidence level.
#' \item \code{intervals} - binary variable (whether intervals were produced or not).
#' \item \code{residuals} - the residuals of the original model.
#' }
#' @author Ivan Svetunkov, \email{ivan@@svetunkov.ru}
#' @seealso \code{\link[forecast]{ets}, \link[forecast]{forecast}}
#' @references Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008)
#' Forecasting with exponential smoothing: the state space approach,
#' Springer-Verlag. \url{http://www.exponentialsmoothing.net}.
#' @keywords forecast
#' @examples
#'
#' ourModel <- ces(rnorm(100,0,1),h=10)
#'
#' forecast.smooth(ourModel,h=10)
#' forecast.smooth(ourModel,h=10,intervals=TRUE)
#' plot(forecast.smooth(ourModel,h=10,intervals=TRUE))
#'
#' @export forecast.smooth
#' @export
forecast.smooth <- function(object, h=10,
                            intervals=c("parametric","semiparametric","nonparametric","none"),
                            level=0.95, ...){
    intervals <- intervals[1];
    if(gregexpr("ETS",object$model)!=-1){
        newModel <- es(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
    }
    else if(gregexpr("CES",object$model)!=-1){
        newModel <- ces(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
    }
    else if(gregexpr("GES",object$model)!=-1){
        newModel <- ges(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
    }
    else if(gregexpr("ARIMA",object$model)!=-1){
        if(any(unlist(gregexpr("combine",object$model))==-1)){
            newModel <- ssarima(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
        }
        else{
            stop(paste0("Sorry, but in order to produce forecasts for this ARIMA we need to recombine it.\n",
                 "You will have to use auto.ssarima() function instead."),call.=FALSE);
        }
    }
    else if(gregexpr("SMA",object$model)!=-1){
        newModel <- sma(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
    }
    else{
        stop("Wrong object provided. This needs to be either 'ETS' or 'CES' or 'GES' or 'SSARIMA' model.",call.=FALSE);
    }
    output <- list(model=object,method=object$model,fitted=newModel$fitted,actuals=newModel$actuals,
                   forecast=newModel$forecast,lower=newModel$lower,upper=newModel$upper,level=newModel$level,
                   intervals=intervals,mean=newModel$forecast,x=object$actuals,residuals=object$residuals);

    return(structure(output,class=c("smooth.forecast","forecast")));
}

#' @export
getResponse.smooth <- function(object, ...){
    return(object$actuals);
}
#' @export
getResponse.smooth.forecast <- function(object, ...){
    return(object$model$actuals);
}

#### Function extracts lags of provided model ####
#' @export
lags.default <- function(object, ...){
    model <- object$model;
    if(!is.null(model)){
        if(gregexpr("GES",model)!=-1){
            lags <- as.numeric(substring(model,unlist(gregexpr("\\[",model))+1,unlist(gregexpr("\\]",model))-1));
        }
        else if(gregexpr("ARIMA",model)!=-1){
            if(any(unlist(gregexpr("combine",object$model))==-1)){
                if(any(unlist(gregexpr("\\[",model))!=-1)){
                    lags <- as.numeric(substring(model,unlist(gregexpr("\\[",model))+1,unlist(gregexpr("\\]",model))-1));
                }
                else{
                    lags <- 1;
                }
            }
            else{
                warning("ARIMA was combined and we cannot extract lags anymore. Sorry!",call.=FALSE);
            }
        }
        else if(gregexpr("SMA",model)!=-1){
            lags <- 1;
        }
        else{
            lags <- NA;
        }
    }
    else{
        lags <- NA;
    }

    return(lags);
}

#' @export
lags.Arima <- function(object, ...){
    model <- object$arma;

    lags <- c(1,model[5]);

    return(lags);
}

#### Function extracts type of model. For example "AAN" from ets ####
#' @export
model.type.default <- function(object, ...){
    model <- object$model;
    if(!is.null(model)){
        if(gregexpr("ETS",model)!=-1){
            modelType <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
        }
        else if(gregexpr("CES",model)!=-1){
            modelType <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
            if(modelType=="n"){
                modelType <- "none";
            }
            else if(modelType=="s"){
                modelType <- "simple";
            }
            else if(modelType=="p"){
                modelType <- "partial";
            }
            else{
                modelType <- "full";
            }
        }
        else{
            modelType <- NA;
        }
    }
    else{
        if(any(gregexpr("ets",object$call)!=-1)){
            model <- object$method;
            modelType <- gsub(",","",substring(model,5,nchar(model)-1));
        }
    }

    return(modelType);
}

#### Function extracts orders of provided model ####
#' @export
orders.default <- function(object, ...){
    model <- object$model;
    if(!is.null(model)){
        if(gregexpr("GES",model)!=-1){
            orders <- as.numeric(substring(model,unlist(gregexpr("\\[",model))-1,unlist(gregexpr("\\[",model))-1));
        }
        else if(gregexpr("ARIMA",model)!=-1){
            if(any(unlist(gregexpr("combine",object$model))==-1)){
                arima.orders <- paste0(c("",substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1),"")
                                       ,collapse=";");
                comas <- unlist(gregexpr("\\,",arima.orders));
                semicolons <- unlist(gregexpr("\\;",arima.orders));
                ar.orders <- as.numeric(substring(arima.orders,semicolons[-length(semicolons)]+1,comas[2*(1:(length(comas)/2))-1]-1));
                i.orders <- as.numeric(substring(arima.orders,comas[2*(1:(length(comas)/2))-1]+1,comas[2*(1:(length(comas)/2))-1]+1));
                ma.orders <- as.numeric(substring(arima.orders,comas[2*(1:(length(comas)/2))]+1,semicolons[-1]-1));

                orders <- list(ar=ar.orders,i=i.orders,ma=ma.orders);
            }
            else{
                warning("ARIMA was combined and we cannot extract orders anymore. Sorry!",call.=FALSE);
            }
        }
        else if(gregexpr("SMA",model)!=-1){
            orders <- as.numeric(substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1));
        }
        else{
            orders <- NA;
        }
    }
    else{
        orders <- NA;
    }

    return(orders);
}

#' @export
orders.Arima <- function(object, ...){
    model <- object$arma;

    ar.orders <- c(model[1],model[3]);
    i.orders <-  c(model[6],model[7]);
    ma.orders <- c(model[2],model[4]);

    orders <- list(ar=ar.orders,i=i.orders,ma=ma.orders);

    return(orders);
}

#### Plots of smooth objects ####
#' @importFrom graphics plot
#' @method plot smooth
#' @export
plot.smooth <- function(x, ...){
    ellipsis <- list(...);
    parDefault <- par(no.readonly = TRUE);
    if(gregexpr("ETS",x$model)!=-1){
        if(any(unlist(gregexpr("C",x$model))==-1)){
            if(ncol(x$states)>10){
                message("Too many states. Plotting them one by one on several graphs.");
                if(is.null(ellipsis$main)){
                    ellipsisMain <- NULL;
                }
                else{
                    ellipsisMain <- ellipsis$main;
                }
                nPlots <- ceiling(ncol(x$states)/10);
                for(i in 1:nPlots){
                    if(is.null(ellipsisMain)){
                        ellipsis$main <- paste0("States of ",x$model,", part ",i);
                    }
                    ellipsis$x <- x$states[,(1+(i-1)*10):min(i*10,ncol(x$states))];
                    do.call(plot, ellipsis);
                }
            }
            else{
                if(is.null(ellipsis$main)){
                    ellipsis$main <- paste0("States of ",x$model);
                }
                ellipsis$x <- x$states;
                do.call(plot, ellipsis);
            }
        }
        else{
            # If we did combinations, we cannot return anything
            message("Combination of models was done. Sorry, but there is nothing to plot.");
        }
    }
    else{
        if(any(unlist(gregexpr("combine",x$model))!=-1)){
            # If we did combinations, we cannot do anything
            message("Combination of models was done. Sorry, but there is nothing to plot.");
        }
        else{
            if(ncol(x$states)>10){
                message("Too many states. Plotting them one by one on several graphs.");
                if(is.null(ellipsis$main)){
                    ellipsisMain <- NULL;
                }
                else{
                    ellipsisMain <- ellipsis$main;
                }
                nPlots <- ceiling(ncol(x$states)/10);
                for(i in 1:nPlots){
                    if(is.null(ellipsisMain)){
                        ellipsis$main <- paste0("States of ",x$model,", part ",i);
                    }
                    ellipsis$x <- x$states[,(1+(i-1)*10):min(i*10,ncol(x$states))];
                    do.call(plot, ellipsis);
                }
            }
            else{
                if(is.null(ellipsis$main)){
                    ellipsis$main <- paste0("States of ",x$model);
                }
                ellipsis$x <- x$states;
                do.call(plot, ellipsis);
            }
        }
    }
    par(parDefault);
}

#' @method plot smooth.sim
#' @export
plot.smooth.sim <- function(x, ...){
    ellipsis <- list(...);
    if(is.null(ellipsis$main)){
        ellipsis$main <- x$model;
    }

    if(is.null(dim(x$data))){
        nsim <- 1
    }
    else{
        nsim <- dim(x$data)[2]
    }

    if(nsim==1){
        if(is.null(ellipsis$ylab)){
            ellipsis$ylab <- "Data";
        }
        ellipsis$x <- x$data;
        do.call(plot, ellipsis);
    }
    else{
        message(paste0("You have generated ",nsim," time series. Not sure which of them to plot.\n",
                       "Please use plot(ourSimulation$data[,k]) instead. Plotting a random series."));
        randomNumber <- ceiling(runif(1,1,nsim));
        if(is.null(ellipsis$ylab)){
            ellipsis$ylab <- paste0("Series N",randomNumber);
        }
        ellipsis$x <- x$data[,randomNumber];
        do.call(plot, ellipsis);
    }
}

#' @method plot smooth.forecast
#' @export
plot.smooth.forecast <- function(x, ...){
    if(any(x$intervals!=c("none","n"))){
        graphmaker(x$actuals,x$forecast,x$fitted,x$lower,x$upper,x$level,main=x$method);
    }
    else{
        graphmaker(x$actuals,x$forecast,x$fitted,main=x$method);
    }
}

#' @method plot iss
#' @export
plot.iss <- function(x, ...){
    intermittent <- x$intermittent
    if(intermittent=="c"){
        intermittent <- "Croston";
    }
    else if(intermittent=="t"){
        intermittent <- "TSB";
    }
    else if(intermittent=="f"){
        intermittent <- "Fixed probability";
    }
    else{
        intermittent <- "None";
    }
    graphmaker(x$actuals,x$forecast,x$fitted,main=paste0("iSS, ",intermittent));
}

#### Prints of smooth ####
#' @export
print.smooth <- function(x, ...){
    holdout <- any(!is.na(x$holdout));
    intervals <- any(!is.na(x$lower));
    cumulative <- x$cumulative;

    if(all(holdout,intervals)){
        if(!cumulative){
            insideintervals <- sum((x$holdout <= x$upper) & (x$holdout >= x$lower)) / length(x$forecast) * 100;
        }
        else{
            insideintervals <- NULL;
        }
    }
    else{
        insideintervals <- NULL;
    }

    intervalsType <- x$intervals;

    if(gregexpr("SMA",x$model)!=-1){
        x$iprob <- 1;
        x$initialType <- "b";
        x$intermittent <- "n";
    }
    else if(gregexpr("ETS",x$model)!=-1){
    # If cumulative forecast and Etype=="M", report that this was "parameteric" interval
        if(cumulative & substr(model.type(x),1,1)=="M"){
            intervalsType <- "p";
        }
    }

    ssOutput(x$timeElapsed, x$model, persistence=x$persistence, transition=x$transition, measurement=x$measurement,
             phi=x$phi, ARterms=x$AR, MAterms=x$MA, constant=x$constant, A=x$A, B=x$B,initialType=x$initialType,
             nParam=x$nParam, s2=x$s2, hadxreg=!is.null(x$xreg), wentwild=x$updateX,
             cfType=x$cfType, cfObjective=x$cf, intervals=intervals, cumulative=cumulative,
             intervalsType=intervalsType, level=x$level, ICs=x$ICs,
             holdout=holdout, insideintervals=insideintervals, errormeasures=x$accuracy,
             intermittent=x$intermittent, iprob=x$iprob[length(x$iprob)]);
}

#' @export
print.smooth.sim <- function(x, ...){
    if(is.null(dim(x$data))){
        nsim <- 1
    }
    else{
        nsim <- dim(x$data)[2]
    }

    cat(paste0("Data generated from: ",x$model,"\n"));
    cat(paste0("Number of generated series: ",nsim,"\n"));

    if(nsim==1){
        if(gregexpr("ETS",x$model)!=-1){
            cat(paste0("Persistence vector: \n"));
            xPersistence <- as.vector(x$persistence);
            names(xPersistence) <- rownames(x$persistence);
            print(round(xPersistence,3));
            if(x$phi!=1){
                cat(paste0("Phi: ",x$phi,"\n"));
            }
            cat(paste0("True likelihood: ",round(x$logLik,3),"\n"));
        }
        else if(gregexpr("ARIMA",x$model)!=-1){
            ar.orders <- orders(x)$ar;
            i.orders <- orders(x)$i;
            ma.orders <- orders(x)$ma;
            lags <- lags(x);
            # AR terms
            if(any(ar.orders!=0)){
                ARterms <- matrix(0,max(ar.orders),sum(ar.orders!=0),
                                  dimnames=list(paste0("AR(",c(1:max(ar.orders)),")"),
                                                paste0("Lag ",lags[ar.orders!=0])));
            }
            else{
                ARterms <- matrix(0,1,1);
            }
            # Differences
            if(any(i.orders!=0)){
                Iterms <- matrix(0,1,length(i.orders),
                                 dimnames=list("I(...)",paste0("Lag ",lags)));
                Iterms[,] <- i.orders;
            }
            else{
                Iterms <- 0;
            }
            # MA terms
            if(any(ma.orders!=0)){
                MAterms <- matrix(0,max(ma.orders),sum(ma.orders!=0),
                                  dimnames=list(paste0("MA(",c(1:max(ma.orders)),")"),
                                                paste0("Lag ",lags[ma.orders!=0])));
            }
            else{
                MAterms <- matrix(0,1,1);
            }

            n.coef <- ar.coef <- ma.coef <- 0;
            ar.i <- ma.i <- 1;
            for(i in 1:length(ar.orders)){
                if(ar.orders[i]!=0){
                    ARterms[1:ar.orders[i],ar.i] <- x$AR[ar.coef+(1:ar.orders[i])];
                    ar.coef <- ar.coef + ar.orders[i];
                    ar.i <- ar.i + 1;
                }
                if(ma.orders[i]!=0){
                    MAterms[1:ma.orders[i],ma.i] <- x$MA[ma.coef+(1:ma.orders[i])];
                    ma.coef <- ma.coef + ma.orders[i];
                    ma.i <- ma.i + 1;
                }
            }

            if(!is.null(x$AR)){
                cat(paste0("AR parameters: \n"));
                print(round(ARterms,3));
            }
            if(!is.null(x$MA)){
                cat(paste0("MA parameters: \n"));
                print(round(MAterms,3));
            }
            if(!is.na(x$constant)){
                cat(paste0("Constant value: ",round(x$constant,3),"\n"));
            }
            cat(paste0("True likelihood: ",round(x$logLik,3),"\n"));
        }
        else if(gregexpr("CES",x$model)!=-1){
            cat(paste0("Smoothing parameter A: ",round(x$A,3),"\n"));
            if(!is.null(x$B)){
                if(is.complex(x$B)){
                    cat(paste0("Smoothing parameter B: ",round(x$B,3),"\n"));
                }
                else{
                    cat(paste0("Smoothing parameter b: ",round(x$B,3),"\n"));
                }
            }
            cat(paste0("True likelihood: ",round(x$logLik,3),"\n"));
        }
    }
}

#' @export
print.smooth.forecast <- function(x, ...){
    if(any(x$intervals!=c("none","n"))){
        level <- x$level;
        if(level>1){
            level <- level/100;
        }
        output <- cbind(x$forecast,x$lower,x$upper);
        colnames(output) <- c("Point forecast",paste0("Lower bound (",(1-level)/2*100,"%)"),paste0("Upper bound (",(1+level)/2*100,"%)"));
    }
    else{
        output <- x$forecast;
    }
    print(output);
}

#' @export
print.iss <- function(x, ...){
    intermittent <- x$intermittent
    if(intermittent=="c"){
        intermittent <- "Croston";
    }
    else if(intermittent=="t"){
        intermittent <- "TSB";
    }
    else if(intermittent=="f"){
        intermittent <- "Fixed probability";
    }
    else{
        intermittent <- "None";
    }
    ICs <- round(c(AIC(x),AICc(x),BIC(x)),4);
    names(ICs) <- c("AIC","AICc","BIC");
    cat(paste0("Intermittent State-Space model estimated: ",intermittent,"\n"));
    cat(paste0("Smoothing parameter: ",round(x$C[1],3),"\n"));
    cat(paste0("Initial value: ",round(x$states[1],3),"\n"));
    cat(paste0("Probability forecast: ",round(x$forecast[1],3),"\n"));
    cat("Information criteria: \n");
    print(ICs);
}

#### Simulate data using provided object ####
#' @export
simulate.smooth <- function(object, nsim=1, seed=NULL, obs=NULL, ...){
    if(is.null(obs)){
        obs <- length(object$actuals);
    }
    if(!is.null(seed)){
        set.seed(seed);
    }

    if(gregexpr("ETS",object$model)!=-1){
        model <- object$model;
        model <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
        if(any(unlist(gregexpr("C",model))==-1)){
            if(substr(model,1,1)=="A"){
                randomizer <- "rnorm";
            }
            else{
                randomizer <- "rlnorm";
            }
            simulatedData <- sim.es(model=model, frequency=frequency(object$actuals), phi=object$phi,
                                    persistence=object$persistence, initial=object$initial, initialSeason=object$initialSeason,
                                    obs=obs,nsim=nsim,iprob=object$iprob[length(object$iprob)],
                                    randomizer=randomizer,mean=0,sd=sqrt(object$s2),...);
        }
        else{
            message("Sorry, but we cannot simulate data from combined model.");
            simulatedData <- NA;
        }
    }
    else if(gregexpr("ARIMA",object$model)!=-1){
        if(any(unlist(gregexpr("combine",object$model))==-1)){
            orders <- orders(object);
            lags <- lags(object);
            randomizer <- "rnorm";
            simulatedData <- sim.ssarima(orders=orders, lags=lags,
                                         frequency=frequency(object$actuals), AR=object$AR, MA=object$MA, constant=object$constant,
                                         initial=object$initial, obs=obs, nsim=nsim,
                                         iprob=object$iprob[length(object$iprob)], randomizer=randomizer, mean=0, sd=sqrt(object$s2),...);
        }
        else{
            message("Sorry, but we cannot simulate data from combined model.");
            simulatedData <- NA;
        }
    }
    else if(gregexpr("CES",object$model)!=-1){
        model <- substring(object$model,unlist(gregexpr("\\(",object$model))+1,unlist(gregexpr("\\)",object$model))-1);
        initial <- object$initial;
        randomizer <- "rnorm";
        simulatedData <- sim.ces(seasonality=model,
                                 frequency=frequency(object$actuals), A=object$A, B=object$B,
                                 initial=object$initial, obs=obs, nsim=nsim,
                                 iprob=object$iprob[length(object$iprob)], randomizer=randomizer, mean=0, sd=sqrt(object$s2),...);
    }
    else if(gregexpr("GES",object$model)!=-1){
        model <- object$model;
        orders <- as.numeric(substring(model,unlist(gregexpr("\\[",model))-1,unlist(gregexpr("\\[",model))-1));
        lags <- as.numeric(substring(model,unlist(gregexpr("\\[",model))+1,unlist(gregexpr("\\]",model))-1));
        initial <- object$initial;
        randomizer <- "rnorm";
        simulatedData <- sim.ges(orders=orders, lags=lags, frequency=frequency(object$actuals), measurement=object$measurement,
                                 transition=object$transition, persistence=object$persistence, initial=object$initial,
                                 obs=obs, nsim=nsim,
                                 iprob=object$iprob[length(object$iprob)], randomizer=randomizer, mean=0, sd=sqrt(object$s2),...);

    }
    else{
        model <- substring(object$model,1,unlist(gregexpr("\\(",object$model))[1]-1);
        message(paste0("Sorry, but simulate is not yet available for the model ",model,"."));
        simulatedData <- NA;
    }
    return(simulatedData);
}

#### Summary of objects ####
#' @export
summary.smooth <- function(object, ...){
    print(object);
}

#' @export
summary.smooth.forecast <- function(object, ...){
    print(object);
}

#' @export
summary.iss <- function(object, ...){
    print(object);
}
