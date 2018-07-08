cma <- function(data, order=NULL, ic=c("AICc","AIC","BIC","BICc"),
                h=10, holdout=FALSE, cumulative=FALSE,
                intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                silent=c("all","graph","legend","output","none"),
                ...){

    # If a previous model provided as a model, write down the variables
    if(exists("model")){
        if(is.null(model$model)){
            stop("The provided model is not Simple Moving Average!",call.=FALSE);
        }
        else if(gregexpr("SMA",model$model)==-1){
            stop("The provided model is not Simple Moving Average!",call.=FALSE);
        }
        else{
            order <- model$order;
        }
    }

    ##### data #####
    if(any(class(data)=="smooth.sim")){
        data <- data$data;
    }
    else if(class(data)=="Mdata"){
        h <- data$h;
        holdout <- TRUE;
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
    y <- ts(matrix(data[1:obsInsample],obsInsample,1), start=dataStart, frequency=datafreq);

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

    if((order %% 2)!=0){
        model <- sma(y, order=order, ic=ic, h=max(h*2,order), holdout=FALSE, cumulative=FALSE,
                        intervals=intervals, level=level, silent=TRUE, ...);
        yFitted <- c(model$fitted[-c(1:((order+1)/2))],model$forecast);
        if(h!=0){
            yForecast <- yFitted[-(1:obsInsample)];
            yForecast <- ts(yForecast[1:h], start=start(model$forecast), frequency=datafreq);
            model$forecast <- yForecast;
            if(any(!is.na(model$upper))){
                model$upper <- ts(model$upper[-(1:((order+1)/2))][1:h], start=start(model$forecast),
                                  frequency=datafreq);
                model$lower <- ts(model$lower[-(1:((order+1)/2))][1:h], start=start(model$forecast),
                                  frequency=datafreq);
            }
        }
        else{
            model$forecast <- ts(NA, start=start(model$forecast), frequency=datafreq);
        }
        yFitted <- ts(yFitted[1:obsInsample], start=dataStart, frequency=datafreq);
        model$model <- paste0("CMA(",order,")");
        model$fitted <- yFitted;
        model$residuals <- ts(y - yFitted, start=dataStart, frequency=datafreq);
        model$s2 <- sum(model$residuals^2)/(obsInsample - 2);
        model$cf <- mean(model$residuals^2);
        model$logLik <- -obsInsample/2 *(log(2*pi*exp(1)) + log(model$cf));
        model$ICs <- c(AIC(model),AICc(model),BIC(model),BICc(model));
        names(model$ICs) <- c("AIC","AICc","BIC","BICc");
    }
    else{
    }

    if(!silent){
        graphmaker(data, model$forecast, model$fitted, model$upper, model$lower, level=level);
    }

    return(model);
}
