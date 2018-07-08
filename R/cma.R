cma <- function(data, order=NULL, ic=c("AICc","AIC","BIC","BICc"),
                h=10, holdout=FALSE, cumulative=FALSE,
                intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                silent=c("all","graph","legend","output","none"),
                ...){

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
    y <- matrix(data[1:obsInsample],obsInsample,1);
    datafreq <- frequency(data);
    dataStart <- start(data);

    smaModel <- sma(y, order=order, ic=ic, h=h*2, holdout=FALSE, cumulative=cumulative,
                    intervals=intervals, level=level, silent=silent, ...);


}
