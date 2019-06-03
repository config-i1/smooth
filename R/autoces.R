utils::globalVariables(c("silentText","silentGraph","silentLegend","initialType"));

#' Complex Exponential Smoothing Auto
#'
#' Function estimates CES in state space form with information potential equal
#' to errors with different seasonality types and chooses the one with the
#' lowest IC value.
#'
#' The function estimates several Complex Exponential Smoothing in the
#' state space 2 described in Svetunkov, Kourentzes (2015) with the information
#' potential equal to the approximation error using different types of
#' seasonality and chooses the one with the lowest value of information
#' criterion.
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("ces","smooth")}
#'
#'
#' @template ssBasicParam
#' @template ssAdvancedParam
#' @template ssInitialParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssCESRef
#'
#' @param models The vector containing several types of seasonality that should
#' be used in CES selection. See \link[smooth]{ces} for more details about the
#' possible types of seasonal models.
#' @param ...  Other non-documented parameters.  For example \code{FI=TRUE}
#' will make the function produce Fisher Information matrix, which then can be
#' used to calculated variances of parameters of the model.
#' @return Object of class "smooth" is returned. See \link[smooth]{ces} for
#' details.
#' @seealso \code{\link[smooth]{ces}, \link[forecast]{ets},
#' \link[forecast]{forecast}, \link[stats]{ts}}
#'
#' @examples
#'
#' y <- ts(rnorm(100,10,3),frequency=12)
#' # CES with and without holdout
#' auto.ces(y,h=20,holdout=TRUE)
#' auto.ces(y,h=20,holdout=FALSE)
#'
#' library("Mcomp")
#' \dontrun{y <- ts(c(M3$N0740$x,M3$N0740$xx),start=start(M3$N0740$x),frequency=frequency(M3$N0740$x))
#' # Selection between "none" and "full" seasonalities
#' auto.ces(y,h=8,holdout=TRUE,models=c("n","f"),interval="p",level=0.8,ic="AIC")}
#'
#' ourModel <- auto.ces(M3[[1683]],interval="sp")
#'
#' summary(ourModel)
#' forecast(ourModel)
#' plot(forecast(ourModel))
#'
#' @export auto.ces
auto.ces <- function(y, models=c("none","simple","full"),
                initial=c("optimal","backcasting"), ic=c("AICc","AIC","BIC","BICc"),
                loss=c("MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                h=10, holdout=FALSE, cumulative=FALSE,
                interval=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                occurrence=c("none","auto","fixed","general","odds-ratio","inverse-odds-ratio","direct"),
                oesmodel="MNN",
                bounds=c("admissible","none"),
                silent=c("all","graph","legend","output","none"),
                xreg=NULL, xregDo=c("use","select"), initialX=NULL,
                updateX=FALSE, persistenceX=NULL, transitionX=NULL, ...){
# Function estimates several CES models in state space form with sigma = error,
#  chooses the one with the lowest ic value and returns complex smoothing parameter
#  value, fitted values, residuals, point and interval forecasts, matrix of CES components
#  and values of information criteria

#    Copyright (C) 2015  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

    ##### Check if data was used instead of y. Remove by 2.6.0 #####
    y <- depricator(y, list(...), "data");
    loss <- depricator(loss, list(...), "cfType");
    interval <- depricator(interval, list(...), "intervals");

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

##### Set environment for ssInput and make all the checks #####
    environment(ssAutoInput) <- environment();
    ssAutoInput("auto.ces",ParentEnvironment=environment());

# If the pool of models is wrong, fall back to default
    if(length(models)!=1){
        modelsOk <- rep(FALSE,length(models));
        for(i in 1:length(models)){
            modelsOk[i] <- any(models[i]==c("n","s","p","f","none","simple","partial","full"));
        }
    }

    if(!all(modelsOk)){
        message("The pool of models includes a strange type of model! Reverting to default pool.");
        models <- c("n","s","p","f");
    }
    models <- substr(models,1,1);

    dataFreq <- frequency(y);

    # Define maximum needed number of parameters
    if(any(models=="n")){
    # 1 is for variance, 2 is for complex smoothing parameter
        nParamMax <- 3;
        if(initialType=="o"){
            nParamMax <- nParamMax + 2;
        }
    }
    if(any(models=="p")){
        nParamMax <- 4;
        if(initialType=="o"){
            nParamMax <- nParamMax + 2 + dataFreq;
        }
        if(obsNonzero <= nParamMax){
            warning("The sample is too small. We cannot use partial seasonal model.",call.=FALSE);
            models <- models[models!="p"];
        }
    }
    if(any(models=="s")){
        nParamMax <- 3;
        if(initialType=="o"){
            nParamMax <- nParamMax + 2*dataFreq;
        }
        if(obsNonzero <= nParamMax){
            warning("The sample is too small. We cannot use simple seasonal model.",call.=FALSE);
            models <- models[models!="s"];
        }
    }
    if(any(models=="f")){
        nParamMax <- 5;
        if(initialType=="o"){
            nParamMax <- nParamMax + 2 + 2*dataFreq;
        }
        if(obsNonzero <= nParamMax){
            warning("The sample is too small. We cannot use full seasonal model.",call.=FALSE);
            models <- models[models!="f"];
        }
    }

    if(dataFreq==1){
        if(!silentText){
            message("The data is not seasonal. Simple CES was the only solution here.");
        }

        CESModel <- ces(y, seasonality="n",
                        initial=initialType, ic=ic,
                        loss=loss,
                        h=h, holdout=holdout,cumulative=cumulative,
                        interval=intervalType, level=level,
                        occurrence=occurrence, oesmodel=oesmodel,
                        bounds=bounds, silent=silent,
                        xreg=xreg, xregDo=xregDo, initialX=initialX,
                        updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, FI=FI);
        return(CESModel);
    }

# Check the number of observations and number of parameters.
    if(any(models=="f") & (obsNonzero <= dataFreq*2 + 2 + 4 + 1)){
        warning("Sorry, but you don't have enough observations for CES(f).",call.=FALSE);
        models <- models[models!="f"];
    }
    if(any(models=="p") & (obsNonzero <= dataFreq + 2 + 3 + 1)){
        warning("Sorry, but you don't have enough observations for CES(p).",call.=FALSE);
        models <- models[models!="p"];
    }
    if(any(models=="s") & (obsNonzero <= dataFreq*2 + 2 + 1)){
        warning("Sorry, but you don't have enough observations for CES(s).",call.=FALSE);
        models <- models[models!="s"];
    }

    CESModel <- as.list(models);
    IC.vector <- c(1:length(models));

    j <- 1;
    if(!silentText){
        cat("Estimating CES with seasonality: ")
    }
    for(i in models){
        if(!silentText){
            cat(paste0('"',i,'" '));
        }
        CESModel[[j]] <- ces(y, seasonality=i,
                             initial=initialType, ic=ic,
                             loss=loss,
                             h=h, holdout=holdout,cumulative=cumulative,
                             interval=intervalType, level=level,
                             occurrence=occurrence, oesmodel=oesmodel,
                             bounds=bounds, silent=TRUE,
                             xreg=xreg, xregDo=xregDo, initialX=initialX,
                             updateX=updateX, persistenceX=persistenceX, transitionX=transitionX, FI=FI);
        IC.vector[j] <- CESModel[[j]]$ICs[ic];
        j <- j+1;
    }

    bestModel <- CESModel[[which(IC.vector==min(IC.vector))]];

    yFitted <- bestModel$fitted;
    yForecast <- bestModel$forecast;
    yUpper <- bestModel$upper;
    yLower <- bestModel$lower;
    modelname <- bestModel$model;

    if(!silentText){
        best.seasonality <- models[which(IC.vector==min(IC.vector))];
        cat(" \n");
        cat(paste0('The best model is with seasonality = "',best.seasonality,'"\n'));
    }

##### Make a plot #####
    if(!silentGraph){
        yForecastNew <- yForecast;
        yUpperNew <- yUpper;
        yLowerNew <- yLower;
        if(cumulative){
            yForecastNew <- ts(rep(yForecast/h,h),start=yForecastStart,frequency=dataFreq)
            if(interval){
                yUpperNew <- ts(rep(yUpper/h,h),start=yForecastStart,frequency=dataFreq)
                yLowerNew <- ts(rep(yLower/h,h),start=yForecastStart,frequency=dataFreq)
            }
        }

        if(interval){
            graphmaker(actuals=y,forecast=yForecastNew,fitted=yFitted, lower=yLowerNew,upper=yUpperNew,
                       level=level,legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
        else{
            graphmaker(actuals=y,forecast=yForecastNew,fitted=yFitted,
                       legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
    }

    bestModel$timeElapsed <- Sys.time()-startTime;

    return(bestModel);
}
