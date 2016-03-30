ces.auto <- function(data, C=c(1.1, 1), models=c("N","S","P","F"),
                IC=c("CIC","AIC","AICc","BIC"),
                CF.type=c("MSE","MAE","HAM","trace","GV","TV","MSEh"),
                use.test=FALSE, intervals=FALSE, int.w=0.95,
                int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                bounds=FALSE, holdout=FALSE, h=1, silent=FALSE, legend=TRUE,
                xreg=NULL, go.wild=FALSE, intermittent=FALSE){
# Function estimates several CES models in state-space form with sigma = error,
#  chooses the one with the lowest IC value and returns complex smoothing parameter
#  value, fitted values, residuals, point and interval forecasts, matrix of CES components
#  and values of information criteria

#    Copyright (C) 2015  Ivan Svetunkov

    go.wild <- FALSE;

# Start measuring the time of calculations
    start.time <- Sys.time();

    CF.type <- CF.type[1];
# Check if the appropriate CF.type is defined
    if(any(CF.type==c("trace","TV","GV","MSEh"))){
        multisteps <- TRUE;
    }
    else if(any(CF.type==c("MSE","MAE","HAM"))){
        multisteps <- FALSE;
    }
    else{
        message(paste0("Strange cost function specified: ",CF.type,". Switching to 'MSE'."));
        CF.type <- "MSE";
        multisteps <- FALSE;
    }

    int.type <- substring(int.type[1],1,1);
# Check the provided type of interval
    if(all(int.type!=c("a","p","s","n"))){
        message(paste0("The wrong type of interval chosen: '",int.type, "'. Switching to 'parametric'."));
        int.type <- "p";
    }

# If the pool of models is wrong, fall back to default
    if(any(models!="N" & models!="S" & models!="P" & models!="F")){
        message("The pool of models includes a strange type of model! Reverting to default pool.");
        models <- c("N","S","P","F");
    }

    if(any(is.na(data))){
        if(silent==FALSE){
        message("Data contains NAs. These observations will be excluded.")
        }
        datanew <- data[!is.na(data)]
        if(is.ts(data)){
        datanew <- ts(datanew,start=start(data),frequency=frequency(data))
        }
        data <- datanew
    }

    if(frequency(data)==1){
        if(silent==FALSE){
        message("The data is not seasonal. Simple CES was the only solution here.");
        }

        ces.model <- ces(data, C=C, seasonality="N",
                         CF.type=CF.type,
                         use.test=use.test, intervals=intervals, int.w=int.w,
                         int.type=int.type,
                         bounds=bounds, holdout=holdout, h=h, silent=silent, legend=legend,
                         xreg=xreg, go.wild=go.wild, intermittent=intermittent);
        return(ces.model);
    }

    IC <- IC[1]

    ces.model <- as.list(models);
    IC.vector <- c(1:length(models));

    j <- 1;
    if(silent==FALSE){
        cat("Estimating CES with seasonality: ")
    }
    for(i in models){
        if(silent==FALSE){
            cat(paste0('"',i,'" '));
        }
        ces.model[[j]] <- ces(data, C=C, seasonality=i,
                              CF.type=CF.type,
                              use.test=use.test, intervals=intervals, int.w=int.w,
                              int.type=int.type,
                              bounds=bounds, holdout=holdout, h=h, silent=TRUE, legend=legend,
                              xreg=xreg, go.wild=go.wild, intermittent=intermittent);
        IC.vector[j] <- ces.model[[j]]$ICs[IC];
        j <- j+1;
    }

    best.model <- ces.model[[which(IC.vector==min(IC.vector))]];

    if(silent==FALSE){
        cat(" \n");
        cat(paste0('The best model is with seasonality = "',models[which(IC.vector==min(IC.vector))],'"\n'));

# Define obs.all, the overal number of observations (in-sample + holdout)
        obs.all <- length(data) + (1 - holdout)*h;
# Define obs, the number of observations of in-sample
        obs <- length(data) - holdout*h;

        y.fit <- best.model$fitted;
        y.for <- best.model$forecast;
        y.high <- best.model$upper;
        y.low <- best.model$lower;
        errormeasures <- best.model$accuracy;
        n.components <- 2;
        if(!is.null(best.model$B)){
            n.components <- n.components + 1 + 1*is.complex(best.model$B);
        }
# Not the same as in the function, but should be fine...
        s2 <- as.vector(sum(best.model$residuals^2)/obs);

# Make plot
        if(intervals==TRUE){
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                       lower=y.low,upper=y.high,int.w=int.w,legend=legend);
        }
        else{
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit,legend=legend);
        }

# Calculate the number os observations in the interval
        if(all(holdout==TRUE,intervals==TRUE)){
            insideintervals <- sum(as.vector(data)[(obs+1):obs.all]<=y.high &
                                   as.vector(data)[(obs+1):obs.all]>=y.low)/h*100;
        }
        else{
            insideintervals <- NULL;
        }
# Print output
        ssoutput(Sys.time() - start.time, best.model$model, persistence=NULL, transition=NULL, measurement=NULL,
                 phi=NULL, ARterms=NULL, MAterms=NULL, const=NULL, A=best.model$A, B=best.model$B,
                 n.components=n.components, s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
                 CF.type=CF.type, CF.objective=best.model$CF, intervals=intervals,
                 int.type=int.type, int.w=int.w, ICs=best.model$ICs,
                 holdout=holdout, insideintervals=insideintervals, errormeasures=best.model$accuracy);
    }

    return(best.model);
}
