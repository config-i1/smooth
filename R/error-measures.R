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
#' data.
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


#' Prediction Likelihood Score
#'
#' Function estimates Prediction Likelihood Score of the holdout actuals based on the model.
#'
#' Prediction likelihood score (PLS) is based on either normal or log-normal
#' distribution of errors with the provided parameters. It returns the log of probability
#' that the data was "produced" by the estimated model. %In case of trace forecasts PLS is
#' %based on trace forecast likelihood but returns value devided by squared horizon (in order
#' %to keep scale consistent with non-trace cases).
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param actuals Actual values from the holdout.
#' @param forecasts Point forecasts for the holdout (conditional mean).
#' @param Etype Type of the error. If \code{Etype="A"}, then normal distribution
#' is used, if \code{Etype="M"}, then log-normal distribution is used.
#' @param sigma Value of variance of the errors. In case of \code{trace=TRUE}, this
#' needs to be a covariance matrix of trace errors.
#' @param trace If \code{TRUE}, then it is assumed that we are provided with trace
#' forecasts (multiple steps ahead), Trace Forecast Likelihood is used in this case.
#' @param iprob Vector of probabilities of occurrences for the holdout (only needed
#' for intermittent models).
#' @param digits Number of digits for rounding.
#' @param varVec Vector of 1 to h steps ahead analytical variance. Needed mainly for Etype=="M".
#' @param rounded Defines if the rounded up value is used for demand sizes.
#' @param ...  Other parameters passed to mean function.
#'
#' @return A value of the log-likelihood.
#' @references \itemize{
#' \item Snyder, R. D., Ord, J. K., Beaumont, A., 2012. Forecasting the intermittent
#' demand for slow-moving inventories: A modelling approach. International
#' Journal of Forecasting 28 (2), 485-496.
#' \item Kolassa, S., 2016. Evaluating predictive count data distributions in retail
#' sales forecasting. International Journal of Forecasting 32 (3), 788-803.
#' }
#' @examples
#'
#' # pls() function now works correctly only when varVec is provided
#' # And varVec is not provided by any function, but is generated inside them.
#'
#' # Generate data, apply es() with the holdout parameter and calculate PLS
#' x <- rnorm(100,0,1)
#' ourModel <- es(x, h=10, holdout=TRUE, intervals=TRUE)
#' sigma <- t(ourModel$errors) %*% (ourModel$errors) / length(ourModel$residuals)
#' Etype <- substr(modelType(ourModel),1,1)
#' pls(actuals=ourModel$holdout, forecasts=ourModel$forecast, Etype=Etype,
#'     sigma=sigma, trace=TRUE)
#'
#' # Do the same with intermittent data. Trace is not available yet for
#' # intermittent state-space models
#' x <- rpois(100,0.4)
#' ourModel <- es(x, h=10, holdout=TRUE, intermittent='a', intervals=TRUE)
#' Etype <- substr(modelType(ourModel),1,1)
#' iprob <- ourModel$imodel$fitted
#' pls(actuals=ourModel$holdout, forecasts=ourModel$forecast, Etype=Etype,
#'     sigma=ourModel$s2, trace=FALSE, iprob=iprob)
#'
#' @importFrom stats dlnorm
#' @importFrom stats dnorm
#' @export pls
pls <- function(actuals, forecasts, Etype=c("A","M"), sigma, trace=TRUE,
                iprob=1, digits=5, varVec=NULL, rounded=FALSE, ...){
    # This function calculates half moment
    if(length(actuals)!=length(forecasts)){
        warning("Length of actuals and forecasts differs. Using the shortest of the two.", call.=FALSE);
        lengthMin <- min(length(actuals),length(forecasts));
        actuals <- actuals[1:lengthMin];
        forecasts <- forecasts[1:lengthMin];
    }
    obsHoldout <- length(actuals);
    if(obsHoldout==1){
        trace=FALSE;
    }

    if(any(c(actuals,forecasts)<0) & Etype=="M"){
        warning("Error type cannot be multiplicative, with negative actuals and/or forecasts.",call.=FALSE)
        return(-Inf);
    }

    Etype <- Etype[1];
    if(!any(Etype==c("A","M"))){
        warning(paste0("Unknown type of error term: ",Etype,
                       "Switching to 'A'."), call.=FALSE);
    }

    if(trace){
        if(!is.matrix(sigma) | (length(sigma) != obsHoldout^2)){
            warning(paste0("sigma is not a covariance matrix, but it is supposed to be. ",
                           "Forcing trace=FALSE."), call.=FALSE);
            trace <- FALSE;
        }
    }
    if(all(sigma==0)){
        return(NA);
    }

    if(is.null(varVec)){
        varVec <- sigma;
    }

    if(all(iprob==1) & length(iprob)>1){
        warning("Probability for the holdout is equal to 1. Using non-intermittent model.", call.=FALSE);
        iprob <- 1;
    }

    if(!all(iprob==1)){
        if(length(iprob)!=obsHoldout){
            if(length(iprob) < obsHoldout){
                # Repeat last iprob as many times as needed
                iprob <- c(iprob,rep(iprob[length(iprob)],obsHoldout))[1:obsHoldout];
            }
            else{
                iprob <- iprob[1:obsHoldout];
            }
        }
        ot <- (actuals!=0);
        forecasts <- c(forecasts) / c(iprob);
        if(trace){
            warning("We cannot yet do trace and intermittent models.",call.=FALSE);
            trace <- FALSE;
        }
    }
    else{
        ot <- rep(1,obsHoldout);
    }

    if(Etype=="A"){
        errors <- as.matrix(c(actuals[ot] - forecasts[ot]));
    }
    else{
        errors <- as.matrix(c(log(actuals[ot]) - log(forecasts[ot])));
    }

    obsNonZero <- sum(ot);

    ##### Now do the calculations #####
    if(!rounded){
        if(all(iprob==1)){
            if(trace){
                if(Etype=="A"){
                    pls <- -(obsNonZero/2 * obsNonZero * log(2*pi*det(sigma)) + sum(t(errors) %*% solve(sigma) %*% errors) / 2);
                }
                else{
                    pls <- -(obsNonZero/2 * obsNonZero * log(2*pi*det(sigma)) + sum(t(errors) %*% solve(sigma) %*% errors) / 2 + obsNonZero * sum(log(actuals)));
                }
                # pls <- pls / obsNonZero^2;
            }
            else{
                if(Etype=="A"){
                    pls <- sum(log(dnorm(actuals[ot],forecasts[ot],sqrt(varVec))));
                    # pls <- -(obsNonZero/2 * log(2*pi*sigma) + sum(errors^2) / (2*sigma));
                }
                else{
                    pls <- sum(log(dlnorm(actuals[ot],log(forecasts[ot]),sqrt(varVec))));
                    # pls <- -(obsNonZero/2 * log(2*pi*sigma) + sum(errors^2) / (2*sigma) + sum(log(actuals)));
                }
            }
        }
        else{
            if(any(!ot)){
                if(trace){
                    if(Etype=="A"){
                        pls <- -(obsNonZero/2 * obsNonZero * log(2*pi*det(sigma)) + sum(t(errors) %*% solve(sigma) %*% errors) / 2) + (sum(log(iprob[ot])) + sum(log(1-iprob[!ot])));
                    }
                    else{
                        pls <- -(obsNonZero/2 * obsNonZero * log(2*pi*det(sigma)) + sum(t(errors) %*% solve(sigma) %*% errors) / 2 + obsNonZero * sum(log(actuals[ot]))) + (sum(log(iprob[ot])) + sum(log(1-iprob[!ot])));
                    }
                }
                else{
                    if(Etype=="A"){
                        pls <- (sum(log(dnorm(actuals[ot],forecasts[ot],sqrt(varVec[ot])))) +
                                    sum(log(iprob[ot])) + sum(log(1-iprob[!ot])));
                        # pls <- -(obsNonZero/2 * log(2*pi*sigma) + sum(errors^2) / (2*sigma)) + sum(log(iprob[ot])) + sum(log(1-iprob[!ot]));
                    }
                    else{
                        pls <- (sum(log(dlnorm(actuals[ot],log(forecasts[ot]),sqrt(varVec[ot])))) +
                                    sum(log(iprob[ot])) + sum(log(1-iprob[!ot])));
                        # pls <- -(obsNonZero/2 * log(2*pi*sigma) + sum(errors^2) / (2*sigma) + sum(log(actuals[ot]))) + sum(log(iprob[ot])) + sum(log(1-iprob[!ot]));
                    }
                }
            }
            else{
                pls <- sum(log(1-iprob[!ot]));
            }
        }
    }
    else{
        if(all(iprob==1)){
            if(Etype=="A"){
                pls <- sum(log(pnorm(ceiling(actuals),forecasts,sqrt(sigma))-pnorm(ceiling(actuals)-1,forecasts,sqrt(sigma))));
            }
            else{
                pls <- sum(log(plnorm(ceiling(actuals),log(forecasts),sqrt(sigma)) - pnorm(ceiling(actuals)-1,log(forecasts),sqrt(sigma))));
            }
        }
        else{
            if(any(!ot)){
                if(Etype=="A"){
                    pls <- (sum(log(pnorm(ceiling(actuals[ot]),forecasts[ot],sqrt(varVec[ot])) -
                                   pnorm(ceiling(actuals[ot])-1,forecasts[ot],sqrt(varVec[ot])))) +
                            sum(log(iprob[ot])) + sum(log(1-iprob[!ot])));
                }
                else{
                    pls <- (sum(log(plnorm(ceiling(actuals[ot]),log(forecasts[ot]),sqrt(varVec[ot])) -
                                    plnorm(ceiling(actuals[ot])-1,log(forecasts[ot]),sqrt(varVec[ot])))) +
                            sum(log(iprob[ot])) + sum(log(1-iprob[!ot])));
                }
            }
            else{
                pls <- sum(log(1-iprob[!ot]));
            }
        }
    }

    return(round(pls, digits=digits));
}
