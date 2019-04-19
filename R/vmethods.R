#' @export
logLik.vsmooth <- function(object,...){
    obs <- nobs(object);
    nParamPerSeries <- nparam(object);
    structure(object$logLik,nobs=obs,df=nParamPerSeries,class="logLik");
}

#' @export
logLik.viss <- function(object,...){
    obs <- nobs(object);
    nParamPerSeries <- nparam(object);
    structure(object$logLik,nobs=obs,df=nParamPerSeries,class="logLik");
}

#' @export
AICc.vsmooth <- function(object, ...){
    llikelihood <- logLik(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    nSeries <- ncol(object$actuals);
    # Remove covariances in the number of parameters
    nParamAll <- nparam(object) / nSeries - switch(object$cfType,
                                                   "likelihood" = nSeries*(nSeries+1)/2,
                                                   "trace" = ,
                                                   "diagonal" = 1);

    obs <- nobs(object);
    IC <- -2*llikelihood + ((2*obs*(nParamAll*nSeries + nSeries*(nSeries+1)/2)) /
                                (obs - (nParamAll + nSeries + 1)));

    return(IC);
}

#' @export
BICc.vsmooth <- function(object, ...){
    llikelihood <- logLik(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    nSeries <- ncol(object$actuals);
    # Remove covariances in the number of parameters
    nParamAll <- nparam(object) / nSeries - switch(object$cfType,
                                                   "likelihood" = nSeries*(nSeries+1)/2,
                                                   "trace" = ,
                                                   "diagonal" = 1);

    obs <- nobs(object);
    IC <- -2*llikelihood + (((nParamAll + nSeries*(nSeries+1)/2) *
                                 log(obs * nSeries) * obs * nSeries) /
                                (obs * nSeries - nParamAll - nSeries*(nSeries+1)/2));

    return(IC);
}

#' @export
nobs.vsmooth <- function(object, ...){
    return(nrow(object$fitted));
}
#' @export
nobs.viss <- function(object, ...){
    return(nrow(object$fitted));
}

#' @export
nparam.viss <- function(object, ...){
    nParamReturn <- object$nParam[1,4];
    return(nParamReturn);
}

#' @export
sigma.vsmooth <- function(object, ...){
    return(object$Sigma);
}

#### Extraction of parameters of models ####
#' @export
coef.vsmooth <- function(object, ...){

    parameters <- object$coefficients;

    return(parameters);
}

#' @export
modelType.vsmooth <- function(object, ...){
    model <- object$model;
    modelType <- NA;
    if(!is.null(model)){
        if(gregexpr("VES",model)!=-1){
            modelType <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
        }
    }

    return(modelType);
}

#### Plotting things ####
#' @export
plot.viss <- function(x, ...){
    ellipsis <- list(...);
    intermittent <- x$intermittent
    if(intermittent=="f"){
        intermittent <- "Fixed probability";
    }
    else if(intermittent=="l"){
        intermittent <- "Logistic probability";
    }
    else{
        intermittent <- "None";
    }

    actuals <- x$actuals;
    yForecast <- x$forecast;
    yFitted <- x$fitted;
    dataDeltat <- deltat(actuals);
    forecastStart <- start(yForecast);
    h <- nrow(yForecast);
    nSeries <- ncol(actuals);
    modelname <- paste0("iVES(",x$model,")")

    pages <- ceiling(nSeries / 5);
    parDefault <- par(no.readonly=TRUE);
    for(j in 1:pages){
        par(mfcol=c(min(5,floor(nSeries/j)),1));
        for(i in 1:nSeries){
            plotRange <- range(min(actuals[,i],yForecast[,i],yFitted[,i]),
                               max(actuals[,i],yForecast[,i],yFitted[,i]));
            plot(actuals[,i],main=paste0(modelname,", series ", i),ylab="Y",
                 ylim=plotRange, xlim=range(time(actuals[,i])[1],time(yForecast)[max(h,1)]),
                 type="l");
            lines(yFitted[,i],col="purple",lwd=2,lty=2);
            if(h>1){
                lines(yForecast[,i],col="blue",lwd=2);
            }
            else{
                points(yForecast[,i],col="blue",lwd=2,pch=4);
            }
            abline(v=dataDeltat*(forecastStart[2]-2)+forecastStart[1],col="red",lwd=2);
        }
        par(parDefault);
    }
}

#' @export
plot.vsmooth.sim <- function(x, ...){
    ellipsis <- list(...);
    if(is.null(ellipsis$main)){
        ellipsis$main <- x$model;
    }

    if(length(dim(x$data))==2){
        nsim <- 1;
    }
    else{
        nsim <- dim(x$data)[3];
    }

    nSeries <- dim(x$data)[2];
    if(nSeries>10){
        warning("You have generated more than ten time series. We will plot only first ten of them.",
                call.=FALSE);
        x$data <- x$data[,1:10,];
    }

    if(nsim==1){
        if(is.null(ellipsis$ylab)){
            ellipsis$ylab <- "Data";
        }
        ellipsis$x <- x$data;
        do.call(plot, ellipsis);
    }
    else{
        randomNumber <- ceiling(runif(1,1,nsim));
        message(paste0("You have generated ",nsim," time series. Not sure which of them to plot.\n",
                       "Please use plot(ourSimulation$data[,k]) instead. Plotting randomly selected series N",randomNumber,"."));
        if(is.null(ellipsis$ylab)){
            ellipsis$ylab <- paste0("Series N",randomNumber);
        }
        ellipsis$x <- ts(x$data[,,randomNumber]);
        do.call(plot, ellipsis);
    }
}

#### Prints of vector functions ####
#' @export
print.viss <- function(x, ...){

    if(x$probability=="i"){
        intermittent <- "Independent ";
    }
    else if(x$probability=="d"){
        intermittent <- "Dependent ";
    }

    if(x$intermittent=="l"){
        intermittent <- paste0(intermittent,"logistic probability");
    }
    else if(x$intermittent=="f"){
        intermittent <- paste0(intermittent,"fixed probability");
    }
    else{
        intermittent <- "None";
    }
    ICs <- round(c(AIC(x),AICc(x),BIC(x),BICc(x)),4);
    names(ICs) <- c("AIC","AICc","BIC","BICc");
    cat(paste0("Intermittent state space model estimated: ",intermittent,"\n"));
    if(!is.null(x$model)){
        cat(paste0("Underlying ETS model: ",x$model,"\n"));
    }
    cat("Information criteria: \n");
    print(ICs);
}

#' @export
print.vsmooth <- function(x, ...){
    holdout <- any(!is.na(x$holdout));
    intervals <- any(!is.na(x$PI));

    # if(all(holdout,intervals)){
    #     insideintervals <- sum((x$holdout <= x$upper) & (x$holdout >= x$lower)) / length(x$forecast) * 100;
    # }
    # else{
    #     insideintervals <- NULL;
    # }

    intervalsType <- x$intervals;

    cat(paste0("Time elapsed: ",round(as.numeric(x$timeElapsed,units="secs"),2)," seconds\n"));
    cat(paste0("Model estimated: ",x$model,"\n"));
    if(!is.null(x$imodel)){
        if(x$imodel$probability=="i"){
            intermittent <- "Independent ";
        }
        else if(x$imodel$probability=="d"){
            intermittent <- "Dependent ";
        }

        if(x$imodel$intermittent=="l"){
            intermittent <- paste0(intermittent,"logistic probability");
        }
        else if(x$imodel$intermittent=="f"){
            intermittent <- paste0(intermittent,"fixed probability");
        }
        else{
            intermittent <- "None";
        }

        cat(paste0("Intermittent model estimated: ",intermittent,"\n"));
        if(!is.null(x$imodel$model)){
            cat(paste0("Occurrence ETS model: ",x$model,"\n"));
        }
    }
    if(!is.null(x$nParam)){
        if(x$nParam[1,4]==1){
            cat(paste0(x$nParam[1,4]," parameter was estimated for ", ncol(x$actuals) ," time series in the process\n"));
        }
        else{
            cat(paste0(x$nParam[1,4]," parameters were estimated for ", ncol(x$actuals) ," time series in the process\n"));
        }

        if(x$nParam[2,4]>1){
            cat(paste0(x$nParam[2,4]," parameters were provided\n"));
        }
        else if(x$nParam[2,4]>0){
            cat(paste0(x$nParam[2,4]," parameter was provided\n"));
        }
    }

    cat(paste0("Cost function type: ",x$cfType))
    if(!is.null(x$cf)){
        cat(paste0("; Cost function value: ",round(x$cf,3),"\n"));
    }
    else{
        cat("\n");
    }

    cat("\nInformation criteria:\n");
    print(x$ICs);

    if(intervals){
        if(x$intervals=="c"){
            intervalsType <- "conditional";
        }
        else if(x$intervals=="u"){
            intervalsType <- "unconditional";
        }
        else if(x$intervals=="i"){
            intervalsType <- "independent";
        }
        cat(paste0(x$level*100,"% ",intervalsType," prediction intervals were constructed\n"));
    }

}

#### Simulate data using provided vector object ####
#' @export
simulate.vsmooth <- function(object, nsim=1, seed=NULL, obs=NULL, ...){
    ellipsis <- list(...);
    if(is.null(obs)){
        obs <- nobs(object);
    }
    if(!is.null(seed)){
        set.seed(seed);
    }

    # Start a list of arguments
    args <- vector("list",0);

    args$nSeries <- ncol(object$actuals);

    if(!is.null(ellipsis$randomizer)){
        randomizer <- ellipsis$randomizer;
    }
    else{
        randomizer <- "rnorm";
    }

    if(randomizer=="rnorm"){
        if(!is.null(ellipsis$mean)){
            args$mean <- ellipsis$mean;
        }
        else{
            args$mean <- 0;
        }

        if(!is.null(ellipsis$sd)){
            args$sd <- ellipsis$sd;
        }
        else{
            args$sd <- sqrt(mean(residuals(object)^2));
        }
    }
    else if(randomizer=="rlaplace"){
        if(!is.null(ellipsis$mu)){
            args$mu <- ellipsis$mu;
        }
        else{
            args$mu <- 0;
        }

        if(!is.null(ellipsis$b)){
            args$b <- ellipsis$b;
        }
        else{
            args$b <- mean(abs(residuals(object)));
        }
    }
    else if(randomizer=="rs"){
        if(!is.null(ellipsis$mu)){
            args$mu <- ellipsis$mu;
        }
        else{
            args$mu <- 0;
        }

        if(!is.null(ellipsis$b)){
            args$b <- ellipsis$b;
        }
        else{
            args$b <- mean(sqrt(abs(residuals(object))));
        }
    }
    else if(randomizer=="mvrnorm"){
        if(!is.null(ellipsis$mu)){
            args$mu <- ellipsis$mu;
        }
        else{
            args$mu <- 0;
        }

        if(!is.null(ellipsis$Sigma)){
            args$Sigma <- ellipsis$Sigma;
        }
        else{
            args$Sigma <- sigma(object);
        }
    }

    args$randomizer <- randomizer;
    args$frequency <- frequency(object$actuals);
    args$obs <- obs;
    args$nsim <- nsim;
    args$initial <- object$initial;

    if(gregexpr("VES",object$model)!=-1){
        if(all(object$phi==1)){
            phi <- 1;
        }
        else{
            phi <- object$phi;
        }
        model <- modelType(object);
        args <- c(args,list(model=model, phi=phi, persistence=object$persistence,
                            transition=object$transition,
                            initialSeason=object$initialSeason));

        simulatedData <- do.call("sim.ves",args);
    }
    else{
        model <- substring(object$model,1,unlist(gregexpr("\\(",object$model))[1]-1);
        message(paste0("Sorry, but simulate is not yet available for the model ",model,"."));
        simulatedData <- NA;
    }
    return(simulatedData);
}

#### Summary of objects ####
#' @export
summary.vsmooth <- function(object, ...){
    print(object);
}
#' @export
summary.viss <- function(object, ...){
    print(object);
}
