#' @export
logLik.viss <- function(object,...){
    obs <- nobs(object);
    structure(object$logLik,nobs=obs,df=nParam(object),class="logLik");
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
nParam.viss <- function(object, ...){
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
    nSeries <- ncol(yForecast);
    modelname <- paste0("iVES(",x$model,")")

    pages <- ceiling(nSeries / 5);
    parDefault <- par(no.readonly=TRUE);
    for(j in 1:pages){
        par(mfcol=c(min(5,floor(nSeries/j)),1));
        for(i in 1:nSeries){
            plotRange <- range(min(actuals[,i],yForecast[,i],yFitted[,i]),
                               max(actuals[,i],yForecast[,i],yFitted[,i]));
            plot(actuals[,i],main=paste0(modelname,", series ", i),ylab="Y",
                 ylim=plotRange,
                 xlim=range(time(actuals[,i])[1],time(yForecast)[max(h,1)]));
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
    ICs <- round(c(AIC(x),AICc(x),BIC(x)),4);
    names(ICs) <- c("AIC","AICc","BIC");
    cat(paste0("Intermittent State-Space model estimated: ",intermittent,"\n"));
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
            cat(paste0(x$nParam[1,4]," parameter was estimated in the process\n"));
        }
        else{
            cat(paste0(x$nParam[1,4]," parameters were estimated in the process\n"));
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

#### Summary of objects ####
#' @export
summary.vsmooth <- function(object, ...){
    print(object);
}
#' @export
summary.viss <- function(object, ...){
    print(object);
}
