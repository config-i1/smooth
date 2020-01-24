#' Multiple seasonal classical decomposition
#'
#' Function decomposes multiple seasonal time series into components using
#' the principles of classical decomposition.
#'
#' The function applies centred moving averages based on \link[forecast]{ma}
#' function and order specified in \code{lags} variable in order to smooth the
#' original series and obtain level, trend and seasonal components of the series.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param y Vector or ts object, containing data needed to be smoothed.
#' @param lags Vector of lags, corresponding to the frequencies in the data.
#' @param type The type of decomposition. If \code{"multiplicative"} is selected,
#' then the logarithm of data is taken prior to the decomposition.
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
#' @seealso \code{\link[forecast]{ma}}
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
#' @importFrom forecast ma
#' @export msdecompose
msdecompose <- function(y, lags=c(12), type=c("additive","multiplicative")){
    # Function decomposes time series, assuming multiple frequencies provided in lags
    type <- match.arg(type,c("additive","multiplicative"));
    if(type=="multiplicative"){
        yInsample <- log(y);
    }
    else{
        yInsample <- y;
    }
    yName <- deparse(substitute(y));

    obs <- length(y);
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
        ySmooth[[i+1]] <- ma(yInsample,lags[i],centre=TRUE);
    }
    trend <- ySmooth[[lagsLength+1]];

    # Initial level and trend
    initial <- c(mean(ySmooth[[lagsLength]],na.rm=T),
                 mean(diff(ySmooth[[lagsLength]]),na.rm=T));
    names(initial) <- c("level","trend");

    # Produce the cleared series
    for(i in 1:lagsLength){
        yClear[[i]] <- ySmooth[[i]] - ySmooth[[i+1]];
    }

    # The seasonal patterns
    patterns <- vector("list",lagsLength);
    for(i in 1:lagsLength){
        patterns[[i]] <- vector("numeric",lags[i]);
        for(j in 1:lags[i]){
            patterns[[i]][j] <- mean(yClear[[i]][(1:(obs/lags[i])-1)*lags[i]+j],na.rm=TRUE);
        }
    }

    # Return to the original scale
    if(type=="multiplicative"){
        initial[] <- exp(initial);
        trend <- exp(trend);
        patterns[] <- lapply(patterns,exp);
    }

    return(structure(list(y=y, initial=initial, trend=trend, seasonal=patterns,
                          lags=lags, type=type, yName=yName), class="msdecompose"));
}

#' @export
fitted.msdecompose <- function(object, ...){
    yFitted <- object$trend;
    if(object$type=="additive"){
        for(i in 1:length(object$lags)){
            yFitted <- yFitted + object$seasonal[[i]];
        }
    }
    else{
        for(i in 1:length(object$lags)){
            yFitted <- yFitted * object$seasonal[[i]];
        }
    }
    return(yFitted);
}

#' @export
plot.msdecompose <- function(x, ...){
    ellipsis <- list(...);
    ellipsis$x <- actuals(x);
    if(!any(names(ellipsis)=="ylab")){
        ellipsis$ylab <- x$yName;
    }
    yFitted <- fitted(x);

    do.call(plot,ellipsis);
    lines(yFitted, col="red");
}

#' @export
print.msdecompose <- function(x, ...){
    cat(paste0("Multiple seasonal decomposition of ",x$yName," using c(",paste0(x$lags,collapse=","),") lags.\n"));
    cat("Type of decomposition:",x$type);
}

#' @export
residuals.msdecompose <- function(object, ...){
    if(object$type=="additive"){
        return(actuals(object)-fitted(object));
    }
    else{
        return(actuals(object)/fitted(object));
    }
}

#' @export
forecast.msdecompose <- function(object, h=10, model="ZZN", ...){

    yTrend <- object$trend;
    obs <- length(actuals(object));
    NAsNumber <- sum(is.na(yTrend))/2;
    yForecastStart <- time(yTrend)[length(time(yTrend))]+1/frequency(yTrend);

    # Apply ETS model to the trend data
    yesModel <- es(yTrend[!is.na(yTrend)],model=model,h=h+NAsNumber);

    # Form a big vector of trend + forecast
    yValues <- ts(c(yTrend,yesModel$forecast[-c(1:NAsNumber)]),start=start(yTrend),frequency=frequency(yTrend));
    # Add seasonality
    if(object$type=="additive"){
        for(i in 1:length(object$lags)){
            yValues <- yValues + object$seasonal[[i]];
        }
    }
    else{
        for(i in 1:length(object$lags)){
            yValues <- yValues * object$seasonal[[i]];
        }
    }
    # Cut the forecasts
    yForecast <- window(yValues,yForecastStart);

    return(structure(list(mean=yForecast,model=object),class="msdecompose.forecast"));
}

#' @export
print.msdecompose.forecast <- function(x, ...){
    cat(x$mean);
}

#' @export
plot.msdecompose.forecast <- function(x, ...){
    graphmaker(actuals(x$model),x$mean,fitted(x$model), ...);
}
