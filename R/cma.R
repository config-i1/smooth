#' Centered Moving Average
#'
#' Function constructs centered moving average based on state space SMA
#'
#' If the order is odd, then the function constructs SMA(order) and
#' shifts it back in time. Otherwise an AR(order+1) model is constructed
#' with the preset parameters:
#'
#' phi_i = {0.5,1,1,...,0.5} / order
#'
#' This then corresponds to the centered MA with 0.5 weight for the
#' first observation and 0.5 weight for an additional one. e.g. if this is
#' monthly data and we use order=12, then half of the first january and
#' half of the new one is taken.
#'
#' This is not a forecasting tool. This is supposed to smooth the time
#' series in order to find trend. So don't expect any forecasts from this
#' function!
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template smoothRef
#'
#' @param data Vector or ts object, containing data needed to be smoothed.
#' @param order Order of centered moving average. If \code{NULL}, then the
#' function will try to select order of SMA based on information criteria.
#' See \link[smooth]{sma} for details.
#' @param silent If \code{TRUE}, then plot is not produced. Otherwise, there
#' is a plot...
#' @return Object of class "smooth" is returned. It contains the list of the
#' following values:
#'
#' \itemize{
#' \item \code{model} - the name of the estimated model.
#' \item \code{timeElapsed} - time elapsed for the construction of the model.
#' \item \code{order} - order of the moving average.
#' \item \code{nParam} - table with the number of estimated / provided parameters.
#' If a previous model was reused, then its initials are reused and the number of
#' provided parameters will take this into account.
#' \item \code{fitted} - the fitted values, shifted in time.
#' \item \code{forecast} - NAs, because this function does not produce forecasts.
#' \item \code{residuals} - the residuals of the SMA / AR model.
#' \item \code{s2} - variance of the residuals (taking degrees of freedom into
#' account) of the SMA / AR model.
#' \item \code{actuals} - the original data.
#' \item \code{ICs} - values of information criteria from the respective SMA or
#' AR model. Includes AIC, AICc, BIC and BICc.
#' \item \code{logLik} - log-likelihood of the SMA / AR model.
#' \item \code{cf} - Cost function value (for the SMA / AR model).
#' \item \code{cfType} - Type of cost function used in the estimation.
#' }
#'
#' @seealso \code{\link[forecast]{ma}, \link[smooth]{es},
#' \link[smooth]{ssarima}}
#'
#' @keywords SARIMA ARIMA
#' @examples
#'
#' # SMA of specific order
#' ourModel <- sma(rnorm(118,100,3),order=12,h=18,holdout=TRUE,intervals="p")
#'
#' # SMA of arbitrary order
#' ourModel <- sma(rnorm(118,100,3),h=18,holdout=TRUE,intervals="sp")
#'
#' summary(ourModel)
#' forecast(ourModel)
#' plot(forecast(ourModel))
#'
#' @export cma
cma <- function(data, order=NULL, silent=TRUE){

# Start measuring the time of calculations
    startTime <- Sys.time();

    holdout <- FALSE;
    h <- 0;

    # If a previous model provided as a model, write down the variables
    if(exists("model")){
        if(is.null(model$model)){
            stop("The provided model is not a Centered Moving Average!",call.=FALSE);
        }
        else if(smoothType(model)!="CMA"){
            stop("The provided model is not a Centered Moving Average!",call.=FALSE);
        }
        else{
            order <- model$order;
        }
    }

    ##### data #####
    if(any(is.smooth.sim(data))){
        data <- data$data;
    }
    else if(class(data)=="Mdata"){
        data <- ts(c(data$x,data$xx),start=start(data$x),frequency=frequency(data$x));
    }

    if(!is.numeric(data)){
        stop("The provided data is not a vector or ts object! Can't construct any model!", call.=FALSE);
    }
    if(!is.null(ncol(data))){
        if(ncol(data)>1){
            stop("The provided data is not a vector! Can't construct any model!", call.=FALSE);
        }
    }
    # Check the data for NAs
    if(any(is.na(data))){
        if(!silentText){
            warning("Data contains NAs. These observations will be substituted by zeroes.",call.=FALSE);
        }
        data[is.na(data)] <- 0;
    }

    # Define obs, the number of observations of in-sample
    obsInsample <- length(data) - holdout*h;

    # Define obsAll, the overal number of observations (in-sample + holdout)
    obsAll <- length(data) + (1 - holdout)*h;

    # If obsInsample is negative, this means that we can't do anything...
    if(obsInsample<=0){
        stop("Not enough observations in sample.",call.=FALSE);
    }
    # Define the actual values
    datafreq <- frequency(data);
    dataStart <- start(data);
    y <- ts(data[1:obsInsample], start=dataStart, frequency=datafreq);

    # Order of the model
    if(!is.null(order)){
        if(obsInsample < order){
            stop("Sorry, but we don't have enough observations for that order.",call.=FALSE);
        }

        if(!is.numeric(order)){
            stop("The provided order is not numeric.",call.=FALSE);
        }
        else{
            if(length(order)!=1){
                warning("The order should be a scalar. Using the first provided value.",call.=FALSE);
                order <- order[1];
            }

            if(order<1){
                stop("The order of the model must be a positive number.",call.=FALSE);
            }
        }
        orderSelect <- FALSE;
    }
    else{
        orderSelect <- TRUE;
    }

    if(orderSelect){
        order <- orders(sma(y));
    }

    if((order %% 2)!=0){
        smaModel <- sma(y, order=order, h=order, holdout=FALSE, cumulative=FALSE, silent=TRUE);
        yFitted <- c(smaModel$fitted[-c(1:((order+1)/2))],smaModel$forecast);
        logLik <- smaModel$logLik;
        errors <- residuals(smaModel);
    }
    else{
        ssarimaModel <- ssarima(y, orders=c(order+1,0,0), AR=c(0.5,rep(1,order-1),0.5)/order,
                         h=order, holdout=FALSE, silent=TRUE);
        yFitted <- c(ssarimaModel$fitted[-c(1:(order/2))],ssarimaModel$forecast);
        smaModel <- sma(y, order=1, h=order, holdout=FALSE, cumulative=FALSE, silent=TRUE);
        logLik <- ssarimaModel$logLik;
        errors <- residuals(ssarimaModel);
    }
    yForecast <- ts(NA, start=start(smaModel$forecast), frequency=datafreq);
    yFitted <- ts(yFitted[1:obsInsample], start=dataStart, frequency=datafreq);
    modelname <- paste0("CMA(",order,")");
    nParam <- smaModel$nParam;
    s2 <- sum(errors^2)/(obsInsample - 2);
    cfObjective <- mean(errors^2);

    model <- structure(list(model=modelname,timeElapsed=Sys.time()-startTime,
                            order=order, nParam=nParam,
                            fitted=yFitted,forecast=yForecast,residuals=errors,s2=s2,
                            actuals=data,
                            ICs=NULL,logLik=logLik,cf=cfObjective,cfType="MSE"),
                       class="smooth");

    ICs <- c(AIC(model),AICc(model),BIC(model),BICc(model));
    names(ICs) <- c("AIC","AICc","BIC","BICc");
    model$ICs <- ICs;

    if(!silent){
        graphmaker(data, yForecast, yFitted, legend=FALSE, vline=FALSE,
                   main=model$model);
    }

    return(model);
}
