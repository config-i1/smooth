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

GMRAE <-function(actual,forecast,benchmark,digits=3){
# This function calculates Geometric Mean Relative Absolute Error
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
        return(round(exp(mean(log(abs(actual-forecast)/abs(actual-benchmark)),na.rm=TRUE)),digits=digits));
    }
}

TRAE <-function(actual,forecast,benchmark,digits=3){
# This function calculates Trigonometric Relative Absolute Error. It returns the vector or matrix of values.
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
        return(round(1-atan(abs(actual-forecast)/abs(actual-benchmark))/(pi/4),digits=digits));
    }
}

hm <- function(x,C=mean(x),digits=5,...)
{
# This function calculates half moment

    x <- x[!is.na(x)];
    result <- round(mean(sqrt(as.complex(x-C)),...),digits=digits);
    return(result);
}
