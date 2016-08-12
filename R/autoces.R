auto.ces <- function(data, C=c(1.1, 1), models=c("none","simple","partial","full"),
                initial=c("backcasting","optimal"), IC=c("AICc","AIC","BIC"),
                CF.type=c("MSE","MAE","HAM","trace","GV","TV","MSEh"),
                h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
                int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                intermittent=c("auto","none","fixed","croston","tsb"),
                bounds=c("admissible","none"), silent=c("none","all","graph","legend","output"),
                xreg=NULL, go.wild=FALSE, ...){
# Function estimates several CES models in state-space form with sigma = error,
#  chooses the one with the lowest IC value and returns complex smoothing parameter
#  value, fitted values, residuals, point and interval forecasts, matrix of CES components
#  and values of information criteria

#    Copyright (C) 2015  Ivan Svetunkov

# Start measuring the time of calculations
    start.time <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

##### Set environment for ssInput and make all the checks #####
    environment(ssAutoInput) <- environment();
    ssAutoInput(modelType="ssarima",ParentEnvironment=environment());

# If the pool of models is wrong, fall back to default
    if(all(models!=c("n","s","p","f","none","simple","partial","full"))){
        message("The pool of models includes a strange type of model! Reverting to default pool.");
        models <- c("n","s","p","f");
    }

    datafreq <- frequency(data);
    if(any(is.na(data))){
        if(silent.text==FALSE){
            message("Data contains NAs. These observations will be excluded.")
        }
        datanew <- data[!is.na(data)];
        if(is.ts(data)){
            datanew <- ts(datanew,start=start(data),frequency=datafreq);
        }
        data <- datanew;
    }

    if(datafreq==1){
        if(silent.text==FALSE){
        message("The data is not seasonal. Simple CES was the only solution here.");
        }

        ces.model <- ces(data, C=C, seasonality="n",
                         initial=fittertype,
                         CF.type=CF.type,
                         h=h, holdout=holdout, intervals=intervals, int.w=int.w,
                         int.type=int.type,
                         intermittent=intermittent,
                         bounds=bounds, silent=silent,
                         xreg=xreg, go.wild=go.wild, FI=FI);
        return(ces.model);
    }

# Check the number of observations and number of parameters.
    if(any(models=="F") & (obs <= datafreq*2 + 2 + 4 + 1)){
        warning("Sorry, but you don't have enough observations for CES(F).",call.=FALSE);
        models <- models[models!="F"];
    }
    if(any(models=="P") & (obs <= datafreq + 2 + 3 + 1)){
        warning("Sorry, but you don't have enough observations for CES(P).",call.=FALSE);
        models <- models[models!="P"];
    }
    if(any(models=="S") & (obs <= datafreq*2 + 2 + 1)){
        warning("Sorry, but you don't have enough observations for CES(S).",call.=FALSE);
        models <- models[models!="S"];
    }

    ces.model <- as.list(models);
    IC.vector <- c(1:length(models));

    j <- 1;
    if(silent.text==FALSE){
        cat("Estimating CES with seasonality: ")
    }
    for(i in models){
        if(silent.text==FALSE){
            cat(paste0('"',i,'" '));
        }
        ces.model[[j]] <- ces(data, C=C, seasonality=i,
                              initial=fittertype,
                              CF.type=CF.type,
                              h=h, holdout=holdout, intervals=intervals, int.w=int.w,
                              int.type=int.type,
                              intermittent=intermittent,
                              bounds=bounds, silent=TRUE,
                              xreg=xreg, go.wild=go.wild, FI=FI);
        IC.vector[j] <- ces.model[[j]]$ICs[IC];
        j <- j+1;
    }

    best.model <- ces.model[[which(IC.vector==min(IC.vector))]];

    y.fit <- best.model$fitted;
    y.for <- best.model$forecast;
    y.high <- best.model$upper;
    y.low <- best.model$lower;
    modelname <- best.model$model;

    if(silent.text==FALSE){
        best.seasonality <- models[which(IC.vector==min(IC.vector))];
        cat(" \n");
        cat(paste0('The best model is with seasonality = "',best.seasonality,'"\n'));

# Define obs.all, the overal number of observations (in-sample + holdout)
        obs.all <- length(data) + (1 - holdout)*h;
# Define obs, the number of observations of in-sample
        obs <- length(data) - holdout*h;

        errormeasures <- best.model$accuracy;

        n.components <- 2;
        if(best.seasonality=="S"){
            n.components <- frequency(data)*2;
        }
        else if(best.seasonality=="P"){
            n.components <- n.components + frequency(data);
        }
        else if(best.seasonality=="F"){
            n.components <- n.components + frequency(data) * 2;
        }

# Not the same as in the function, but should be fine...
        s2 <- as.vector(sum(best.model$residuals^2)/obs);

# Calculate the number os observations in the interval
        if(all(holdout==TRUE,intervals==TRUE)){
            insideintervals <- sum(as.vector(data)[(obs+1):obs.all]<=y.high &
                                   as.vector(data)[(obs+1):obs.all]>=y.low)/h*100;
        }
        else{
            insideintervals <- NULL;
        }
# Print output
        ssOutput(Sys.time() - start.time, best.model$model, persistence=NULL, transition=NULL, measurement=NULL,
                 phi=NULL, ARterms=NULL, MAterms=NULL, const=NULL, A=best.model$A, B=best.model$B,
                 n.components=n.components, s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
                 CF.type=CF.type, CF.objective=best.model$CF, intervals=intervals,
                 int.type=int.type, int.w=int.w, ICs=best.model$ICs,
                 holdout=holdout, insideintervals=insideintervals, errormeasures=best.model$accuracy);
    }

# Make plot
    if(silent.graph==FALSE){
        if(intervals==TRUE){
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
                       int.w=int.w,legend=legend,main=modelname);
        }
        else{
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                    int.w=int.w,legend=legend,main=modelname);
        }
    }

    return(best.model);
}
