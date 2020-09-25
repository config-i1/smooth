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
    type <- match.arg(type);
    if(type=="multiplicative"){
        shiftedData <- FALSE;
        # If there are negative values, stop
        if(any(y<0)){
            stop("Multiplicative decomposition is not available for the data with negative values.",
                 call.=FALSE);
        }
        # If there are zeroes, shift the variable up.
        # In the perfect world, we would need to interpolate and repeate seasonal patterns.
        else if(any(y==0)){
            shiftedData[] <- TRUE;
            y[] <- y + 1;
        }
        yInsample <- log(y);
    }
    else{
        yInsample <- y;
    }
    # paste0() is needed in order to avoid line breaks in the name
    yName <- paste0(deparse(substitute(y)),collapse="");

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
        patterns[[i]][] <- patterns[[i]] - mean(patterns[[i]]);
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
        patterns[] <- lapply(patterns,exp);
        if(shiftedData){
            initial[1] <- initial[1] - 1;
            trend[] <- trend -1;
        }
    }

    return(structure(list(y=y, initial=initial, trend=trend, seasonal=patterns, loss="MSE",
                          lags=lags, type=type, yName=yName), class=c("msdecompose","smooth")));
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
    yFitted <- object$trend;
    obs <- nobs(object);
    if(object$type=="additive"){
        for(i in 1:length(object$lags)){
            yFitted <- yFitted + rep(object$seasonal[[i]],ceiling(obs/object$lags[i]))[1:obs];
        }
    }
    else{
        for(i in 1:length(object$lags)){
            yFitted <- yFitted * rep(object$seasonal[[i]],ceiling(obs/object$lags[i]))[1:obs];
        }
    }
    return(yFitted);
}

#' @aliases forecast forecast.smooth
#' @param model The type of ETS model to fit on the decomposed trend. Only applicable to
#' "msdecompose" class. This is then returned in parameter "esmodel". If \code{NULL}, then
#' it will be selected automatically based on the type of the used decomposition (either
#' among pure additive or among pure additive ETS models).
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

    obs <- nobs(object);
    yDeseasonalised <- actuals(object);
    yForecastStart <- time(yDeseasonalised)[length(time(yDeseasonalised))]+1/frequency(yDeseasonalised);
    if(errorType(object)=="A"){
        for(i in 1:length(object$lags)){
            yDeseasonalised <- yDeseasonalised - rep(object$seasonal[[i]],ceiling(obs/object$lags[i]))[1:obs];
        }
    }
    else{
        for(i in 1:length(object$lags)){
            yDeseasonalised <- yDeseasonalised / rep(object$seasonal[[i]],ceiling(obs/object$lags[i]))[1:obs];
        }
    }
    yesModel <- suppressWarnings(es(yDeseasonalised,model=model,h=h,interval=interval,level=level,initial="b",...));

    yValues <- ts(c(yDeseasonalised,yesModel$forecast),start=start(yDeseasonalised),frequency=frequency(yDeseasonalised));
    if(interval!="none"){
        lower <- ts(c(yDeseasonalised,yesModel$lower),start=start(yDeseasonalised),frequency=frequency(yDeseasonalised));
        upper <- ts(c(yDeseasonalised,yesModel$upper),start=start(yDeseasonalised),frequency=frequency(yDeseasonalised));
    }
    else{
        lower <- upper <- NA;
    }
    # Add seasonality
    if(errorType(object)=="A"){
        for(i in 1:length(object$lags)){
            yValues <- yValues + rep(object$seasonal[[i]],ceiling((obs+h)/object$lags[i]))[1:(obs+h)];
            if(interval!="none"){
                lower <- lower + rep(object$seasonal[[i]],ceiling((obs+h)/object$lags[i]))[1:(obs+h)];
                upper <- upper + rep(object$seasonal[[i]],ceiling((obs+h)/object$lags[i]))[1:(obs+h)];
            }
        }
    }
    else{
        for(i in 1:length(object$lags)){
            yValues <- yValues * rep(object$seasonal[[i]],ceiling((obs+h)/object$lags[i]))[1:(obs+h)];
            if(interval!="none"){
                lower <- lower * rep(object$seasonal[[i]],ceiling((obs+h)/object$lags[i]))[1:(obs+h)];
                upper <- upper * rep(object$seasonal[[i]],ceiling((obs+h)/object$lags[i]))[1:(obs+h)];
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
                          mean=yForecast, forecast=yForecast, lower=lower, upper=upper,
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
    obs <- nobs(x);

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
        yDecomposed <- cbind(actuals(x),x$trend);
        for(i in 1:length(x$seasonal)){
            yDecomposed <- cbind(yDecomposed,rep(x$seasonal[[i]],ceiling(obs/x$lags[i]))[1:obs]);
        }
        yDecomposed <- cbind(yDecomposed, residuals(x));
        colnames(yDecomposed) <- c("Actuals","Trend",paste0("Seasonal ",c(1:length(x$seasonal))),"Residuals");

        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- paste0("Decomposition of ", x$yName);
        }
        ellipsis$x <- yDecomposed;
        do.call(plot,ellipsis);
    }
}

#' @export
print.msdecompose <- function(x, ...){
    cat(paste0("Multiple seasonal decomposition of ",x$yName," using c(",paste0(x$lags,collapse=","),") lags.\n"));
    cat("Type of decomposition:",x$type);
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

sigma.msdecompose <- function(object, ...){
    if(errorType(object)=="A"){
        return(sqrt(mean(residuals(object)^2,na.rm=TRUE)));
    }
    else{
        return(sqrt(mean(residuals(object)^2,na.rm=TRUE)));
    }
}
