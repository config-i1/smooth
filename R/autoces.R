utils::globalVariables(c("silentText","silentGraph","silentLegend","initialType"));

auto.ces <- function(data, C=c(1.1, 1), models=c("none","simple","partial","full"),
                initial=c("backcasting","optimal"), ic=c("AICc","AIC","BIC"),
                cfType=c("MSE","MAE","HAM","MLSTFE","MSTFE","MSEh"),
                h=10, holdout=FALSE, intervals=FALSE, level=0.95,
                intervalsType=c("parametric","semiparametric","nonparametric"),
                intermittent=c("none","auto","fixed","croston","tsb"),
                bounds=c("admissible","none"),
                silent=c("none","all","graph","legend","output"),
                xreg=NULL, updateX=FALSE, ...){
# Function estimates several CES models in state-space form with sigma = error,
#  chooses the one with the lowest ic value and returns complex smoothing parameter
#  value, fitted values, residuals, point and interval forecasts, matrix of CES components
#  and values of information criteria

#    Copyright (C) 2015  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

##### Set environment for ssInput and make all the checks #####
    environment(ssAutoInput) <- environment();
    ssAutoInput(modelType="ces",ParentEnvironment=environment());

# If the pool of models is wrong, fall back to default
    if(all(models!=c("n","s","p","f","none","simple","partial","full"))){
        message("The pool of models includes a strange type of model! Reverting to default pool.");
        models <- c("n","s","p","f");
    }

    datafreq <- frequency(data);
    if(any(is.na(data))){
        if(silentText==FALSE){
            message("Data contains NAs. These observations will be excluded.")
        }
        datanew <- data[!is.na(data)];
        if(is.ts(data)){
            datanew <- ts(datanew,start=start(data),frequency=datafreq);
        }
        data <- datanew;
    }

    if(datafreq==1){
        if(silentText==FALSE){
        message("The data is not seasonal. Simple CES was the only solution here.");
        }

        CESModel <- ces(data, C=C, seasonality="n",
                         initial=initialType,
                         cfType=cfType,
                         h=h, holdout=holdout, intervals=intervals, level=level,
                         intervalsType=intervalsType,
                         intermittent=intermittent,
                         bounds=bounds, silent=silent,
                         xreg=xreg, updateX=updateX, FI=FI);
        return(CESModel);
    }

    if(cfType!="MSE"){
        warning(paste0("'",cfType,"' is used as cost function instead of 'MSE'. The results of model selection may be wrong."),call.=FALSE);
    }

# Check the number of observations and number of parameters.
    if(any(models=="F") & (obsInsample <= datafreq*2 + 2 + 4 + 1)){
        warning("Sorry, but you don't have enough observations for CES(F).",call.=FALSE);
        models <- models[models!="F"];
    }
    if(any(models=="P") & (obsInsample <= datafreq + 2 + 3 + 1)){
        warning("Sorry, but you don't have enough observations for CES(P).",call.=FALSE);
        models <- models[models!="P"];
    }
    if(any(models=="S") & (obsInsample <= datafreq*2 + 2 + 1)){
        warning("Sorry, but you don't have enough observations for CES(S).",call.=FALSE);
        models <- models[models!="S"];
    }

    CESModel <- as.list(models);
    IC.vector <- c(1:length(models));

    j <- 1;
    if(silentText==FALSE){
        cat("Estimating CES with seasonality: ")
    }
    for(i in models){
        if(silentText==FALSE){
            cat(paste0('"',i,'" '));
        }
        CESModel[[j]] <- ces(data, C=C, seasonality=i,
                              initial=initialType,
                              cfType=cfType,
                              h=h, holdout=holdout, intervals=intervals, level=level,
                              intervalsType=intervalsType,
                              intermittent=intermittent,
                              bounds=bounds, silent=TRUE,
                              xreg=xreg, updateX=updateX, FI=FI);
        IC.vector[j] <- CESModel[[j]]$ICs[ic];
        j <- j+1;
    }

    bestModel <- CESModel[[which(IC.vector==min(IC.vector))]];

    y.fit <- bestModel$fitted;
    y.for <- bestModel$forecast;
    y.high <- bestModel$upper;
    y.low <- bestModel$lower;
    modelname <- bestModel$model;

    if(silentText==FALSE){
        best.seasonality <- models[which(IC.vector==min(IC.vector))];
        cat(" \n");
        cat(paste0('The best model is with seasonality = "',best.seasonality,'"\n'));
    }

# Make plot
    if(silentGraph==FALSE){
        if(intervals==TRUE){
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
                       level=level,legend=!silentLegend,main=modelname);
        }
        else{
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                    level=level,legend=!silentLegend,main=modelname);
        }
    }

    bestModel$timeElapsed <- Sys.time()-startTime;

    return(bestModel);
}
