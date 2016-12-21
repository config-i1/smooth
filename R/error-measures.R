MPE <- function(actual,forecast,digits=3)
{
# This function calculates Mean / Median Percentage Error
# actual - actual values,
# forecast - forecasted or fitted values.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        message("Can't procede further on.");
    }
    else{
        return(round(mean((actual-forecast)/actual,na.rm=TRUE),digits=digits));
    }
}

MAPE <- function(actual,forecast,digits=3){
# This function calculates Mean Absolute Percentage Error
# actual - actual values,
# forecast - forecasted values.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        message("Can't procede further on.");
    }
    else{
        return(round(mean(abs((actual-forecast)/actual),na.rm=TRUE),digits=digits));
    }
}

SMAPE <- function(actual,forecast,digits=3)
{
# This function calculates Symmetric Mean / Median Absolute Percentage Error with
# sum of absolute values in the denominator
# actual - actual values,
# forecast - forecasted or fitted values.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        message("Can't procede further on.");
    }
    else{
        return(round(mean(2*abs(actual-forecast)/(abs(actual)+abs(forecast)),na.rm=TRUE),digits=digits));
    }
}

MASE <- function(actual,forecast,scale,digits=3){
# This function calculates Mean Absolute Scaled Error as in Hyndman & Koehler, 2006
# actual - actual values,
# forecast - forecasted values.
# scale - the measure to scale errors with. Usually - MAE of in-sample.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        message("Can't procede further on.");
    }
    else{
        return(round(mean(abs(actual-forecast),na.rm=TRUE)/scale,digits=digits));
    }
}

RelMAE <-function(actual,forecast,benchmark,digits=3){
# This function calculates Average Rellative MAE
# actual - actual values,
# forecast - forecasted or fitted values.
# benchmark - forecasted or fitted values of etalon method.
    if((length(actual) != length(forecast)) | (length(actual) != length(benchmark)) | (length(benchmark) != length(forecast))){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        message(paste0("Length of benchmark: ",length(benchmark)));
        message("Can't procede further on.");
    }
    else{
        return(round(mean(abs(actual-forecast),na.rm=TRUE)/mean(abs(actual-benchmark),na.rm=TRUE),digits=digits));
    }
}

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
        message("Can't procede further on.");
    }
    else{
        return(round(mean((actual-forecast)^2,na.rm=TRUE)/scale,digits=digits));
    }
}

hm <- function(x,C=mean(x),digits=5,...)
{
# This function calculates half moment
    x <- x[!is.na(x)];
    result <- round(mean(sqrt(as.complex(x-C)),...),digits=digits);
    return(result);
}

cbias <- function(x,C=mean(x),digits=5,...)
{
# This function calculates half moment
    result <- hm(x,C,digits);
    result <- round(1 - Arg(result)/(pi/4),digits);
    return(result);
}

sPIS <- function(actual,forecast,scale,digits=3){
# This function calculates scaled Periods-In-Stock.
# actual - actual values,
# forecast - forecasted values.
# scale - the measure to scale errors with.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        message("Can't procede further on.");
    }
    else{
        return(round(sum(cumsum(forecast-actual))/scale,digits=digits));
    }
}

sCE <- function(actual,forecast,scale,digits=3){
# This function calculates scaled Cumulative Error.
# actual - actual values,
# forecast - forecasted values.
# scale - the measure to scale errors with.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        message("Can't procede further on.");
    }
    else{
        return(round(sum(forecast-actual)/scale,digits=digits));
    }
}

errorMeasurer <- function(holdout, forecast, actuals, digits=3,...){
    holdout <- as.vector(holdout);
    forecast <- as.vector(forecast);
    actuals <- as.vector(actuals);
    errormeasures <- c(MPE(holdout,forecast,digits=digits),
                       cbias(holdout-forecast,0,digits=digits),
                       MAPE(holdout,forecast,digits=digits),
                       SMAPE(holdout,forecast,digits=digits),
                       MASE(holdout,forecast,mean(abs(diff(actuals))),digits=digits),
                       MASE(holdout,forecast,mean(abs(actuals)),digits=digits),
                       RelMAE(holdout,forecast,rep(actuals[length(actuals)],length(holdout)),digits=digits),
                       sMSE(holdout,forecast,mean(abs(actuals[actuals!=0]))^2,digits=digits),
                       sPIS(holdout,forecast,mean(abs(actuals[actuals!=0])),digits=digits),
                       sCE(holdout,forecast,mean(abs(actuals[actuals!=0])),digits=digits));
    names(errormeasures) <- c("MPE","cbias","MAPE","SMAPE","MASE","sMAE","RelMAE","sMSE","sPIS","sCE");

    return(errormeasures);
}
