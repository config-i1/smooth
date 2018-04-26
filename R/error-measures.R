#' Error measures
#'
#' Functions allow to calculate different types of errors: \enumerate{ \item MPE
#' - Mean Percentage Error, \item MAPE - Mean Absolute Percentage Error,
#' \item SMAPE - Symmetric Mean Absolute Percentage Error, \item MASE - Mean
#' Absolute Scaled Error, \item RelMAE - Average Relative Mean Absolute Error,
#' \item sMSE - Scaled Mean Squared Error, \item sPIS- Scaled Periods-In-Stock,
#' \item sCE - Scaled Cumulative Error.  }
#'
#' In case of \code{sMSE}, \code{scale} needs to be a squared value. Typical
#' one -- squared mean value of in-sample actuals.
#'
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @aliases Errors
#' @param actual The vector or matrix of actual values.
#' @param forecast The vector or matrix of forecasts values.
#' @param scale The value that should be used in the denominator of MASE. Can
#' be anything but advised values are: mean absolute deviation of in-sample one
#' step ahead Naive error or mean absolute value of the in-sample actuals.
#' @param benchmark The vector or matrix of the forecasts of the benchmark
#' model.
#' @param digits Number of digits of the output.
#' @return All the functions return the scalar value.
#' @references \itemize{
#' \item Fildes, R. (1992). The evaluation of
#' extrapolative forecasting methods. International Journal of Forecasting, 8,
#' pp.81-98.
#' \item Hyndman R.J., Koehler A.B. (2006). Another look at measures of
#' forecast accuracy. International Journal of Forecasting, 22, pp.679-688.
#' \item Makridakis, S. (1993). Accuracy measures: Theoretical and practical
#' concerns. International Journal of Forecasting, 9, pp.527-529.
#' \item Petropoulos F., Kourentzes N. (2015). Forecast combinations for
#' intermittent demand. Journal of the Operational Research Society, 66,
#' pp.914-924.
#' \item Wallstrom P., Segerstedt A. (2010). Evaluation of forecasting error
#' measurements and techniques for intermittent demand. International Journal
#' of Production Economics, 128, pp.625-636.
#' }
#' @examples
#'
#'
#' y <- rnorm(100,10,2)
#' esmodel <- es(y[1:90],model="ANN",h=10)
#'
#' MPE(y[91:100],esmodel$forecast,digits=5)
#' MAPE(y[91:100],esmodel$forecast,digits=5)
#' SMAPE(y[91:100],esmodel$forecast,digits=5)
#' MASE(y[91:100],esmodel$forecast,mean(abs(y[1:90])),digits=5)
#' MASE(y[91:100],esmodel$forecast,mean(abs(diff(y[1:90]))),digits=5)
#'
#' esmodel2 <- es(y[1:90],model="AAN",h=10)
#' RelMAE(y[91:100],esmodel2$forecast,esmodel$forecast,digits=5)
#'
#' MASE(y[91:100],esmodel$forecast,mean(abs(y[1:90]))^2,digits=5)
#'
#' sMSE(y[91:100],esmodel$forecast,mean(abs(y[1:90])),digits=5)
#' sPIS(y[91:100],esmodel$forecast,mean(abs(y[1:90])),digits=5)
#' sCE(y[91:100],esmodel$forecast,mean(abs(y[1:90])),digits=5)
#'
#' @rdname error-measures


#' @rdname error-measures
#' @export MPE
#' @aliases MPE
MPE <- function(actual,forecast,digits=3){
# This function calculates Mean / Median Percentage Error
# actual - actual values,
# forecast - forecasted or fitted values.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(round(mean((actual-forecast)/actual,na.rm=TRUE),digits=digits));
    }
}

#' @rdname error-measures
#' @export MAPE
#' @aliases MAPE
MAPE <- function(actual,forecast,digits=3){
# This function calculates Mean Absolute Percentage Error
# actual - actual values,
# forecast - forecasted values.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(round(mean(abs((actual-forecast)/actual),na.rm=TRUE),digits=digits));
    }
}

#' @rdname error-measures
#' @export SMAPE
#' @aliases SMAPE
SMAPE <- function(actual,forecast,digits=3){
# This function calculates Symmetric Mean / Median Absolute Percentage Error with
# sum of absolute values in the denominator
# actual - actual values,
# forecast - forecasted or fitted values.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(round(mean(2*abs(actual-forecast)/(abs(actual)+abs(forecast)),na.rm=TRUE),digits=digits));
    }
}

#' @rdname error-measures
#' @export MASE
#' @aliases MASE
MASE <- function(actual,forecast,scale,digits=3){
# This function calculates Mean Absolute Scaled Error as in Hyndman & Koehler, 2006
# actual - actual values,
# forecast - forecasted values.
# scale - the measure to scale errors with. Usually - MAE of in-sample.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(round(mean(abs(actual-forecast),na.rm=TRUE)/scale,digits=digits));
    }
}

#' @rdname error-measures
#' @export RelMAE
#' @aliases RelMAE
RelMAE <-function(actual,forecast,benchmark,digits=3){
# This function calculates Average Relative MAE
# actual - actual values,
# forecast - forecasted or fitted values.
# benchmark - forecasted or fitted values of etalon method.
    if((length(actual) != length(forecast)) | (length(actual) != length(benchmark)) | (length(benchmark) != length(forecast))){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        message(paste0("Length of benchmark: ",length(benchmark)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(round(mean(abs(actual-forecast),na.rm=TRUE)/mean(abs(actual-benchmark),na.rm=TRUE),digits=digits));
    }
}

#' @rdname error-measures
#' @export sMSE
#' @aliases sMSE
sMSE <- function(actual,forecast,scale,digits=3){
# This function calculates scaled Mean Squared Error.
# Attention! Scale factor should be provided as squares of something!
# actual - actual values,
# forecast - forecasted values.
# scale - the measure to scale errors with. Usually - MAE of in-sample.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(round(mean((actual-forecast)^2,na.rm=TRUE)/scale,digits=digits));
    }
}

#' @rdname error-measures
#' @export sPIS
#' @aliases sPIS
sPIS <- function(actual,forecast,scale,digits=3){
# This function calculates scaled Periods-In-Stock.
# actual - actual values,
# forecast - forecasted values.
# scale - the measure to scale errors with.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(round(sum(cumsum(forecast-actual))/scale,digits=digits));
    }
}

#' @rdname error-measures
#' @export sCE
#' @aliases sCE
sCE <- function(actual,forecast,scale,digits=3){
# This function calculates scaled Cumulative Error.
# actual - actual values,
# forecast - forecasted values.
# scale - the measure to scale errors with.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(round(sum(forecast-actual)/scale,digits=digits));
    }
}

#' Accuracy of forecasts
#'
#' Function calculates several error measures using the provided
#' data for the holdout sample.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @aliases Accuracy
#' @param holdout The vector of the holdout values.
#' @param forecast The vector of forecasts produced by a model.
#' @param actual The vector of actual in-sample values.
#' @param digits Number of digits of the output.
#' @return The functions returns the named vector of errors:
#' \itemize{
#' \item MPE,
#' \item cbias,
#' \item MAPE,
#' \item SMAPE,
#' \item MASE,
#' \item sMAE,
#' \item RelMAE,
#' \item sMSE,
#' \item sPIS,
#' \item sCE.
#' }
#' For the details on these errors, see \link[smooth]{Errors}.
#' @references \itemize{
#' \item Fildes, R. (1992). The evaluation of
#' extrapolative forecasting methods. International Journal of Forecasting, 8,
#' pp.81-98.
#' \item Hyndman R.J., Koehler A.B. (2006). Another look at measures of
#' forecast accuracy. International Journal of Forecasting, 22, pp.679-688.
#' \item Makridakis, S. (1993). Accuracy measures: Theoretical and practical
#' concerns. International Journal of Forecasting, 9, pp.527-529.
#' \item Petropoulos F., Kourentzes N. (2015). Forecast combinations for
#' intermittent demand. Journal of the Operational Research Society, 66,
#' pp.914-924.
#' \item Wallstrom P., Segerstedt A. (2010). Evaluation of forecasting error
#' measurements and techniques for intermittent demand. International Journal
#' of Production Economics, 128, pp.625-636.
#' }
#' @examples
#'
#'
#' y <- rnorm(100,10,2)
#' esmodel <- es(y[1:90],model="ANN",h=10)
#'
#' Accuracy(y[91:100],esmodel$forecast,y[1:90],digits=5)
#'
#' @export Accuracy
Accuracy <- function(holdout, forecast, actual, digits=3){
    holdout <- as.vector(holdout);
    forecast <- as.vector(forecast);
    actual <- as.vector(actual);
    errormeasures <- c(MPE(holdout,forecast,digits=digits),
                       cbias(holdout-forecast,0,digits=digits),
                       MAPE(holdout,forecast,digits=digits),
                       SMAPE(holdout,forecast,digits=digits),
                       MASE(holdout,forecast,mean(abs(diff(actual))),digits=digits),
                       MASE(holdout,forecast,mean(abs(actual)),digits=digits),
                       RelMAE(holdout,forecast,rep(actual[length(actual)],length(holdout)),digits=digits),
                       sMSE(holdout,forecast,mean(abs(actual[actual!=0]))^2,digits=digits),
                       sPIS(holdout,forecast,mean(abs(actual[actual!=0])),digits=digits),
                       sCE(holdout,forecast,mean(abs(actual[actual!=0])),digits=digits));
    names(errormeasures) <- c("MPE","cbias","MAPE","SMAPE","MASE","sMAE","RelMAE","sMSE","sPIS","sCE");

    return(errormeasures);
}


#' Half moment of a distribution and its derivatives.
#'
#' \code{hm} function estimates half moment from some predefined constant
#' \code{C}.  \code{cbias} function calculates bias based on \code{hm}.
#'
#' \code{NA} values of \code{x} are excluded on the first step of calculation.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @aliases hm
#' @param x A variable based on which HM is estimated.
#' @param C Centering parameter.
#' @param digits Number of digits for rounding.
#' @param ...  Other parameters passed to mean function.
#' @return A complex variable is returned for \code{hm} function and real value
#' is returned for \code{cbias}.
#' @examples
#'
#' x <- rnorm(100,0,1)
#' hm(x)
#' cbias(x)
#'
#' @export hm
#' @rdname hm
hm <- function(x,C=mean(x),digits=5,...){
    # This function calculates half moment
    x <- x[!is.na(x)];
    result <- round(mean(sqrt(as.complex(x-C)),...),digits=digits);
    return(result);
}

#' @rdname hm
#' @export cbias
#' @aliases cbias
cbias <- function(x,C=mean(x),digits=5,...){
    # This function calculates half moment
    result <- hm(x,C,digits);
    result <- round(1 - Arg(result)/(pi/4),digits);
    return(result);
}
