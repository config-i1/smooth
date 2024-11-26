#' Multiple seasonal classical decomposition
#'
#' Function decomposes multiple seasonal time series into components using
#' the principles of classical decomposition.
#'
#' The function applies centred moving averages based on \link[stats]{filter}
#' function and order specified in \code{lags} variable in order to smooth the
#' original series and obtain level, trend and seasonal components of the series.
#'
#' @template smoothRef
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param y Vector or ts object, containing data needed to be smoothed.
#' @param lags Vector of lags, corresponding to the frequencies in the data.
#' @param type The type of decomposition. If \code{"multiplicative"} is selected,
#' then the logarithm of data is taken prior to the decomposition.
#' @param smoother The type of function used in the smoother of the data to
#' extract the trend and in seasonality smoothing. \code{smoother="ma"} relies
#' on the centred moving average and will result in the classical decomposition.
#' \code{smoother="lowess"} will use \link[stats]{lowess}, resulting in a
#' decomposition similar to the STL (\link[stats]{stl}). Finally,
#' \code{smoother="supsmu"} will use the Friedman's super smoother via
#' \link[stats]{supsmu}.
#' @param ... Other parameters passed to smoothers. Only works with
#' lowess/supsmu.
#'
#' @return The object of the class "msdecompose" is return, containing:
#' \itemize{
#' \item \code{y} - the original time series.
#' \item \code{initial} - the estimates of the initial level and trend.
#' \item \code{trend} - the long term trend in the data.
#' \item \code{seasonal} - the list of seasonal parameters.
#' \item \code{lags} - the provided lags.
#' \item \code{type} - the selected type of the decomposition.
#' \item \code{yName} - the name of the provided data.
#' }
#'
#' @seealso \code{\link[stats]{filter}}
#'
#' @examples
#'
#' # Decomposition of multiple frequency data
#' \dontrun{ourModel <- msdecompose(forecast::taylor, lags=c(48,336), type="m")}
#' ourModel <- msdecompose(AirPassengers, lags=c(12), type="m")
#'
#' plot(ourModel)
#' plot(forecast(ourModel, model="AAN", h=12))
#'
#' @importFrom stats filter poly .lm.fit supsmu
#' @export msdecompose
msdecompose <- function(y, lags=c(12), type=c("additive","multiplicative"),
                        smoother=c("ma","lowess","supsmu"), ...){
    # Function decomposes time series, assuming multiple frequencies provided in lags
    type <- match.arg(type);
    smoother <- match.arg(smoother);

    # paste0() is needed in order to avoid line breaks in the name
    yName <- paste0(deparse(substitute(y)),collapse="");

    # Remove the class
    y <- as.vector(y);

    seasonalLags <- any(lags>1);

    smoothingFunction <- function(y, order=NULL, smoother="ma"){
        if(smoother=="ma"){
            # If this is just the global average, don't bother with odd/even
            if((sum(!is.na(y))==order) || (order%%2 != 0)){
                smoothWeigths <- rep(1, order) / order;
            }
            else{
                smoothWeigths <- c(0.5, rep(1, order - 1), 0.5) / order;
            }
            return(filter(y[!is.na(y)], smoothWeigths))
        }
        else if(smoother=="lowess"){
            # The default value of the smoother
            if(is.null(order) || any(order==c(lags[lagsLength], obsInSample)) || order==1){
                order <- 3/2;
            }
            return(lowess(y, f=1/order, ...)$y)
        }
        else if(smoother=="supsmu"){
            # The default value of the smoother
            if(!is.null(order) && any(order==c(lags[lagsLength], obsInSample)) || order==1){
                span <- "cv";
            }
            else{
                span <- 1/order;
            }
            return(supsmu(1:length(y), y, span=span, ...)$y)
        }
    }

    obsInSample <- length(y);
    yNAValues <- is.na(y);

    # Transform the data if needed and split the sample
    if(type=="multiplicative"){
        shiftedData <- FALSE;
        # If there are non-positive values
        if(any(y[!yNAValues]<=0)){
            yNAValues[] <- yNAValues | y<=0;
        }
        yInsample <- suppressWarnings(log(y));
    }
    else{
        yInsample <- y;
    }

    # Treat the missing values
    if(any(yNAValues)){
        X <- cbind(1,poly(c(1:obsInSample),degree=min(max(trunc(obsInSample/10),1),5)),
                   sinpi(matrix(c(1:obsInSample)*rep(c(1:max(lags)),each=obsInSample)/max(lags), ncol=max(lags))));
        lmFit <- .lm.fit(X[!yNAValues,,drop=FALSE], matrix(yInsample[!yNAValues],ncol=1));
        yInsample[yNAValues] <- (X %*% coef(lmFit))[yNAValues];
        rm(X)
    }

    obsInSample <- length(y);
    lags <- sort(unique(lags));
    lagsLength <- length(lags);
    # List of smoothed values
    ySmooth <- vector("list",lagsLength+1);
    # Put actuals int he first element of the list
    ySmooth[[1]] <- yInsample;
    # List of cleared values
    yClear <- vector("list",lagsLength);

    # Smooth time series with different lags
    for(i in 1:lagsLength){
        ySmooth[[i+1]] <- smoothingFunction(yInsample,order=lags[i],smoother=smoother);
    }
    trend <- ySmooth[[lagsLength+1]];

    # Produce the cleared series
    # Do it only if there was a periodicity provided
    if(seasonalLags){
        for(i in 1:lagsLength){
            yClear[[i]] <- ySmooth[[i]] - ySmooth[[i+1]];
        }

        # The seasonal patterns
        patterns <- vector("list",lagsLength);
        for(i in 1:lagsLength){
            patterns[[i]] <- vector("numeric",obsInSample);
            for(j in 1:lags[i]){
                ySeasonal <- yClear[[i]][(1:ceiling(obsInSample/lags[i])-1)*lags[i]+j];
                ySeasonalSmooth <- smoothingFunction(ySeasonal[!is.na(ySeasonal)],
                                                     order=switch(smoother,
                                                                  "ma"=length(ySeasonal[!is.na(ySeasonal)]),
                                                                  obsInSample),
                                                     smoother=smoother);
                # In case of MA, only one value is returned
                if(smoother=="ma"){
                    patterns[[i]][(1:ceiling(obsInSample/lags[i])-1)*lags[i]+j] <- ySeasonalSmooth[!is.na(ySeasonalSmooth)];
                }
                else{
                    patterns[[i]][(1:length(ySeasonalSmooth)-1)*lags[i]+j] <- ySeasonalSmooth;
                }
            }
            patterns[[i]][] <- patterns[[i]] - mean(patterns[[i]], na.rm=TRUE);
        }
    }
    else{
        patterns <- NULL;
    }

    # Initial level and trend
    initial <- c(ySmooth[[lagsLength]][!is.na(ySmooth[[lagsLength]])][1],
                 mean(diff(ySmooth[[lagsLength]]),na.rm=T));
    # Fix the initial, to get to the begining of the sample
    initial[1] <- initial[1] - initial[2]*floor(max(lags)/2);
    names(initial) <- c("level","trend");

    # Return to the original scale
    if(type=="multiplicative"){
        initial[] <- exp(initial);
        trend <- exp(trend);
        if(seasonalLags){
            patterns[] <- lapply(patterns,exp);
        }
        if(shiftedData){
            initial[1] <- initial[1] - 1;
            trend[] <- trend -1;
        }
    }

    # Prepare the matrix of states
    yFitted <- trend;
    if(seasonalLags){
        states <- cbind(trend, c(NA,diff(trend)), matrix(unlist(patterns)[1:obsInSample], obsInSample, lagsLength));
        if(lagsLength>1){
            colnames(states) <- c("level","trend",paste0("seasonal",1:lagsLength));
        }
        else{
            colnames(states) <- c("level","trend","seasonal");
        }
        if(type=="additive"){
            for(i in 1:length(lags)){
                yFitted[] <- yFitted + rep(patterns[[i]],ceiling(obsInSample/lags[i]))[1:obsInSample];
            }
        }
        else{
            for(i in 1:length(lags)){
                yFitted[] <- yFitted * rep(patterns[[i]],ceiling(obsInSample/lags[i]))[1:obsInSample];
            }
        }
    }
    else{
        states <- cbind(trend, c(NA,diff(trend)));
        colnames(states) <- c("level","trend");
    }

    return(structure(list(y=y, states=states, initial=initial, seasonal=patterns, fitted=yFitted,
                          loss="MSE", lags=lags, type=type, yName=yName, smoother=smoother),
                     class=c("msdecompose","smooth")));
}

#' @export
actuals.msdecompose <- function(object, ...){
    return(object$y);
}

#' @export
errorType.msdecompose <- function(object, ...){
    if(object$type=="additive"){
        return("A");
    }
    else{
        return("M");
    }
}

#' @export
fitted.msdecompose <- function(object, ...){
    return(object$fitted);
}

#' @aliases forecast forecast.smooth
#' @param model The type of ETS model to fit on the decomposed trend. Only applicable to
#' "msdecompose" class. This is then returned in parameter "esmodel". If \code{NULL}, then
#' it will be selected automatically based on the type of the used decomposition (either
#' among pure additive or among pure multiplicative ETS models).
#' @rdname forecast.smooth
#' @export
forecast.msdecompose <- function(object, h=10,
                            interval=c("parametric","semiparametric","nonparametric","none"),
                            level=0.95, model=NULL, ...){
    interval <- match.arg(interval,c("parametric","semiparametric","nonparametric","none"));
    if(is.null(model)){
        model <- switch(errorType(object),
                        "A"="XXX",
                        "M"="YYY");
    }

    obsInSample <- nobs(object);
    yDeseasonalised <- actuals(object);
    yForecastStart <- time(yDeseasonalised)[length(time(yDeseasonalised))]+1/frequency(yDeseasonalised);
    if(errorType(object)=="A"){
        for(i in 1:length(object$lags)){
            yDeseasonalised <- yDeseasonalised - rep(object$seasonal[[i]],ceiling(obsInSample/object$lags[i]))[1:obsInSample];
        }
    }
    else{
        for(i in 1:length(object$lags)){
            yDeseasonalised <- yDeseasonalised / rep(object$seasonal[[i]],ceiling(obsInSample/object$lags[i]))[1:obsInSample];
        }
    }
    yesModel <- suppressWarnings(adam(yDeseasonalised,model=model,h=h,initial="b",...));
    yesModel <- forecast(yesModel,h=h,interval=interval,level=level);

    yValues <- ts(c(yDeseasonalised,yesModel$mean),start=start(yDeseasonalised),frequency=frequency(yDeseasonalised));
    if(interval!="none"){
        lower <- ts(c(yDeseasonalised,yesModel$lower),start=start(yDeseasonalised),frequency=frequency(yDeseasonalised));
        upper <- ts(c(yDeseasonalised,yesModel$upper),start=start(yDeseasonalised),frequency=frequency(yDeseasonalised));
    }
    else{
        lower <- upper <- NA;
    }

    # Add seasonality
    #### This is correct for MA only. If we used another smoother, we should forecast seasonal pattern ####
    if(errorType(object)=="A"){
        for(i in 1:length(object$lags)){
            yValues <- yValues + rep(object$seasonal[[i]],ceiling((obsInSample+h)/object$lags[i]))[1:(obsInSample+h)];
            if(interval!="none"){
                lower <- lower + rep(object$seasonal[[i]],ceiling((obsInSample+h)/object$lags[i]))[1:(obsInSample+h)];
                upper <- upper + rep(object$seasonal[[i]],ceiling((obsInSample+h)/object$lags[i]))[1:(obsInSample+h)];
            }
        }
    }
    else{
        for(i in 1:length(object$lags)){
            yValues <- yValues * rep(object$seasonal[[i]],ceiling((obsInSample+h)/object$lags[i]))[1:(obsInSample+h)];
            if(interval!="none"){
                lower <- lower * rep(object$seasonal[[i]],ceiling((obsInSample+h)/object$lags[i]))[1:(obsInSample+h)];
                upper <- upper * rep(object$seasonal[[i]],ceiling((obsInSample+h)/object$lags[i]))[1:(obsInSample+h)];
            }
        }
    }
    # Cut the forecasts
    yForecast <- window(yValues,yForecastStart);
    if(interval!="none"){
        lower <- window(lower,yForecastStart);
        upper <- window(upper,yForecastStart);
    }

    return(structure(list(model=object, esmodel=yesModel, method=paste0("ETS(",modelType(yesModel),") with decomposition"),
                          mean=yForecast, forecast=yForecast, lower=lower, upper=upper, side="both",
                          level=level, interval=interval),class=c("msdecompose.forecast","smooth.forecast","forecast")));
}

#' @rdname isFunctions
#' @export
is.msdecompose <- function(x){
    return(inherits(x,"msdecompose"))
}

#' @rdname isFunctions
#' @export
is.msdecompose.forecast <- function(x){
    return(inherits(x,"msdecompose.forecast"))
}

#' @export
lags.msdecompose <- function(object, ...){
    return(object$lags);
}

#' @export
modelType.msdecompose <- function(object, ...){
    return("Multiple Seasonal Decomposition");
}

#' @export
nobs.msdecompose <- function(object, ...){
    return(length(actuals(object)));
}

#' @export
nparam.msdecompose <- function(object, ...){
    return(length(object$lags)+1);
}

#' @rdname plot.smooth
#' @export
plot.msdecompose <- function(x, which=c(1,2,4,6), level=0.95, legend=FALSE,
                             ask=prod(par("mfcol")) < length(which) && dev.interactive(),
                             lowess=TRUE, ...){
    ellipsis <- list(...);
    obsInSample <- nobs(x);

    # Define, whether to wait for the hit of "Enter"
    if(ask){
        oask <- devAskNewPage(TRUE);
        on.exit(devAskNewPage(oask));
    }

    if(any(which %in% c(1:6))){
        plot.smooth(x, which=which[which %in% c(1:6)], level=level,
                    legend=legend, ask=FALSE, lowess=lowess, ...);
    }

    if(any(which==7)){
        ellipsis$x <- actuals(x);
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- x$yName;
        }
        yFitted <- fitted(x);

        do.call(plot,ellipsis);
        lines(yFitted, col="red");
    }

    if(any(which %in% c(8:11))){
        plot.smooth(x, which=which[which %in% c(8:11)], level=level,
                    legend=legend, ask=FALSE, lowess=lowess, ...);
    }

    if(any(which==12)){
        yDecomposed <- cbind(actuals(x), as.data.frame(x$states), residuals(x));
        # for(i in 1:length(x$seasonal)){
        #     yDecomposed <- cbind(yDecomposed,rep(x$seasonal[[i]],ceiling(obsInSample/x$lags[i]))[1:obsInSample]);
        # }
        # yDecomposed <- cbind(yDecomposed, residuals(x));
        colnames(yDecomposed)[c(1,ncol(yDecomposed))] <- c("actuals","residuals");

        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- paste0("Decomposition of ", x$yName);
        }
        if(ncol(x$states)<=5){
            ellipsis$nc <- 1;
        }
        ellipsis$x <- ts(yDecomposed);
        do.call(plot,ellipsis);
    }
}

#' @export
print.msdecompose <- function(x, ...){
    cat(paste0("Multiple seasonal decomposition of ",x$yName," with c(",paste0(x$lags,collapse=","),") lags"));
    cat("\nType of decomposition:",x$type);
    cat("\nSmoother type:",x$smoother);
}

#' @export
residuals.msdecompose <- function(object, ...){
    if(errorType(object)=="A"){
        return(actuals(object)-fitted(object));
    }
    else{
        return(log(actuals(object)/fitted(object)));
    }
}

#' @export
sigma.msdecompose <- function(object, ...){
    if(errorType(object)=="A"){
        return(sqrt(mean(residuals(object)^2,na.rm=TRUE)));
    }
    else{
        return(sqrt(mean(residuals(object)^2,na.rm=TRUE)));
    }
}
