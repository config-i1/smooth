forecast <- function(object, ...) UseMethod("forecast")
AICc <- function(object, ...) UseMethod("AICc")
orders <- function(object, ...) UseMethod("orders")
lags <- function(object, ...) UseMethod("lags")
modelType <-  function(object, ...) UseMethod("modelType")

##### Likelihood function
#logLik.smooth <- function(object,...)
#{
#  structure(object$logLik,df=length(coef(object)),class="logLik");
#}

##### IC functions #####
AIC.smooth <- function(object, ...){
    if(gregexpr("ETS",object$model)!=-1){
        if(any(unlist(gregexpr("C",object$model))==-1)){
            IC <- object$ICs["AIC"];
        }
        else{
            if(substring(names(object$ICs),10,nchar(names(object$ICs)))=="AIC"){
                IC <- object$ICs;
            }
            else{
                message("ICs were combined during the model construction. Nothing to return.");
                IC <- NA;
            }
        }
    }
    else{
        IC <- object$ICs["AIC"];
    }

    return(IC);
}

AICc.default <- function(object, ...){
    if(!is.null(object$model)){
        if(gregexpr("ETS",object$model)!=-1){
            if(any(unlist(gregexpr("C",object$model))==-1)){
                IC <- object$ICs["AICc"];
            }
            else{
                if(substring(names(object$ICs),10,nchar(names(object$ICs)))=="AICc"){
                    IC <- object$ICs;
                }
                else{
                    message("ICs were combined during the model construction. Nothing to return.");
                    IC <- NA;
                }
            }
        }
        else{
            IC <- object$ICs["AICc"];
        }
    }
    else{
        if(any(gregexpr("ets",object$call)!=-1)){
            IC <- object$aicc;
        }
        else{
            message("AICc is not available for the provided class.");
            IC <- NA;
        }
    }

    return(IC);
}

BIC.smooth <- function(object, ...){
    if(gregexpr("ETS",object$model)!=-1){
        if(any(unlist(gregexpr("C",object$model))==-1)){
            IC <- object$ICs["BIC"];
        }
        else{
            if(substring(names(object$ICs),10,nchar(names(object$ICs)))=="BIC"){
                IC <- object$ICs;
            }
            else{
                message("ICs were combined during the model construction. Nothing to return.");
                IC <- NULL;
            }
        }
    }
    else{
        IC <- object$ICs["BIC"];
    }

    return(IC);
}

#### Extraction of parameters of models ####
coef.smooth <- function(object, ...)
{
    if(gregexpr("CES",object$model)!=-1){
        parameters <- c(object$A,object$B);
    }
    else if(gregexpr("ETS",object$model)!=-1){
        if(any(unlist(gregexpr("C",object$model))==-1)){
            # If this was normal ETS, return values
            parameters <- c(object$persistence,object$initial,object$initial.season);
        }
        else{
            # If we did combinations, we cannot return anything
            message("Combination of models was done, so there are no coefficients to return");
            parameters <- NULL;
        }
    }
    else if(gregexpr("GES",object$model)!=-1){
        parameters <- c(object$measurement,object$transition,object$persistence,object$initial);
        names(parameters) <- c(paste0("Measurement ",c(1:length(object$measurement))),
                               paste0("Transition ",c(1:length(object$transition))),
                               paste0("Persistence ",c(1:length(object$persistence))),
                               paste0("Initial ",c(1:length(object$initial))));
    }
    else if(gregexpr("ARIMA",object$model)!=-1){
        namesConstant <- NamesMA <- NamesAR <- parameters <- NULL;
        if(any(object$AR!=0)){
            parameters <- c(parameters,object$AR);
            NamesAR <- paste(rownames(object$AR),rep(colnames(object$AR),each=ncol(object$AR)),sep=", ");
        }
        if(any(object$MA!=0)){
            parameters <- c(parameters,object$MA);
            NamesMA <- paste(rownames(object$MA),rep(colnames(object$MA),each=ncol(object$MA)),sep=", ")
        }
        if(object$constant!=0){
            parameters <- c(parameters,object$constant);
            namesConstant <- "Constant";
        }
        names(parameters) <- c(NamesAR,NamesMA,namesConstant);
        parameters <- parameters[parameters!=0];
    }
    else if(gregexpr("SMA",object$model)!=-1){
        parameters <- object$persistence;
    }

    return(parameters);
}

fitted.smooth <- function(object, ...){
    return(object$fitted);
}

forecast.smooth <- function(object, h=10,
                            intervals=c("none","parametric","semiparametric","nonparametric"),
                            level=0.95, ...){
    intervals <- intervals[1];
    if(gregexpr("ETS",object$model)!=-1){
        newModel <- es(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
    }
    else if(gregexpr("CES",object$model)!=-1){
        newModel <- ces(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
    }
    else if(gregexpr("GES",object$model)!=-1){
        newModel <- ges(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
    }
    else if(gregexpr("ARIMA",object$model)!=-1){
        newModel <- ssarima(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
    }
    else if(gregexpr("SMA",object$model)!=-1){
        newModel <- sma(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
    }
    else{
        stop("Wrong object provided. This needs to be either 'ETS' or 'CES' or 'GES' or 'SSARIMA' model.",call.=FALSE);
    }
    output <- list(model=newModel$model,fitted=newModel$fitted,actuals=newModel$actuals,
                   forecast=newModel$forecast,lower=newModel$lower,upper=newModel$upper,level=newModel$level,
                   intervals=intervals,mean=newModel$forecast);

    return(structure(output,class="forecastSmooth"));
}

#### Function extracts lags of provided model ####
lags.default <- function(object, ...){
    model <- object$model;
    if(!is.null(model)){
        if(gregexpr("GES",model)!=-1){
            lags <- as.numeric(substring(model,unlist(gregexpr("\\[",model))+1,unlist(gregexpr("\\]",model))-1));
        }
        else if(gregexpr("ARIMA",model)!=-1){
            if(any(unlist(gregexpr("\\[",model))!=-1)){
                lags <- as.numeric(substring(model,unlist(gregexpr("\\[",model))+1,unlist(gregexpr("\\]",model))-1));
            }
            else{
                lags <- 1;
            }
        }
        else if(gregexpr("SMA",model)!=-1){
            lags <- 1;
        }
        else{
            lags <- NA;
        }
    }
    else{
        lags <- NA;
    }

    return(lags);
}

#### Function extracts type of model. For example "AAN" from ets ####
modelType.default <- function(object, ...){
    model <- object$model;
    if(!is.null(model)){
        if(gregexpr("ETS",model)!=-1){
            modelType <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
        }
        else if(gregexpr("CES",model)!=-1){
            modelType <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
        }
        else{
            modelType <- NA;
        }
    }
    else{
        if(any(gregexpr("ets",object$call)!=-1)){
            model <- object$method;
            modelType <- gsub(",","",substring(model,5,nchar(model)-1));
        }
    }

    return(modelType);
}

#### Function extracts orders of provided model ####
orders.default <- function(object, ...){
    model <- object$model;
    if(!is.null(model)){
        if(gregexpr("GES",model)!=-1){
            orders <- as.numeric(substring(model,unlist(gregexpr("\\[",model))-1,unlist(gregexpr("\\[",model))-1));
        }
        else if(gregexpr("ARIMA",model)!=-1){
            arima.orders <- paste0(c("",substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1),"")
                                   ,collapse=";");
            comas <- unlist(gregexpr("\\,",arima.orders));
            semicolons <- unlist(gregexpr("\\;",arima.orders));
            ar.orders <- as.numeric(substring(arima.orders,semicolons[-length(semicolons)]+1,comas[2*(1:(length(comas)/2))-1]-1));
            i.orders <- as.numeric(substring(arima.orders,comas[2*(1:(length(comas)/2))-1]+1,comas[2*(1:(length(comas)/2))-1]+1));
            ma.orders <- as.numeric(substring(arima.orders,comas[2*(1:(length(comas)/2))]+1,semicolons[-1]-1));

            orders <- list(ar.orders=ar.orders,i.orders=i.orders,ma.orders=ma.orders);
        }
        else if(gregexpr("SMA",model)!=-1){
            orders <- as.numeric(substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1));
        }
        else{
            orders <- NA;
        }
    }
    else{
        orders <- NA;
    }

    return(orders);
}

#### Plots of smooth objects ####
plot.smooth <- function(x, ...){
    parDefault <- par(no.readonly = TRUE);
    if(gregexpr("ETS",x$model)!=-1){
        if(any(unlist(gregexpr("C",x$model))==-1)){
            if(ncol(x$states)>10){
                message("Too many states. Plotting them one by one on several graphs.");
                nPlots <- ceiling(ncol(x$states)/10);
                for(i in 1:nPlots){
                    plot(x$states[,(1+(i-1)*10):min(i*10,ncol(x$states))],main=paste0("States of ",x$model,", part ",i));
                }
            }
            else{
                plot(x$states,main=paste0("States of ",x$model));
            }
        }
        else{
            # If we did combinations, we cannot return anything
            message("Combination of models was done. Sorry, but there is nothing to plot.");
        }
    }
    else{
        if(ncol(x$states)>10){
            message("Too many states. Plotting them one by one on several graphs.");
            nPlots <- ceiling(ncol(x$states)/10);
            for(i in 1:nPlots){
                plot(x$states[,(1+(i-1)*10):min(i*10,ncol(x$states))],main=paste0("States of ",x$model,", part ",i));
            }
        }
        else{
            plot(x$states,main=paste0("States of ",x$model));
        }
    }
    par(parDefault);
}

plot.forecastSmooth <- function(x, ...){
    if(any(x$intervals!=c("none","n"))){
        graphmaker(x$actuals,x$forecast,x$fitted,x$lower,x$upper,x$level,main=x$model);
    }
    else{
        graphmaker(x$actuals,x$forecast,x$fitted,main=x$model);
    }
}

plot.iss <- function(x, ...){
    intermittent <- x$intermittent
    if(intermittent=="c"){
        intermittent <- "Croston";
    }
    else if(intermittent=="t"){
        intermittent <- "TSB";
    }
    else if(intermittent=="f"){
        intermittent <- "Fixed probability";
    }
    else{
        intermittent <- "None";
    }
    graphmaker(x$actuals,x$forecast,x$fitted,main=paste0("iSS, ",intermittent));
}

#### Prints of smooth ####
print.smooth <- function(x, ...){
    holdout <- any(!is.na(x$holdout));
    intervals <- any(!is.na(x$lower));
    if(all(holdout,intervals)){
        insideintervals <- sum((x$holdout <= x$upper) & (x$holdout >= x$lower)) / length(x$forecast) * 100;
    }
    else{
        insideintervals <- NULL;
    }

    if(gregexpr("SMA",x$model)!=-1){
        x$iprob <- 1;
        x$initialType <- "b";
        x$intermittent <- "n";
    }

    ssOutput(x$timeElapsed, x$model, persistence=x$persistence, transition=x$transition, measurement=x$measurement,
             phi=x$phi, ARterms=x$AR, MAterms=x$MA, constant=x$constant, A=x$A, B=x$B,initialType=x$initialType,
             nParam=x$nParam, s2=x$s2, hadxreg=!is.null(x$xreg), wentwild=x$updateX,
             cfType=x$cfType, cfObjective=x$cf, intervals=intervals,
             intervalsType=x$intervals, level=x$level, ICs=x$ICs,
             holdout=holdout, insideintervals=insideintervals, errormeasures=x$accuracy,
             intermittent=x$intermittent, iprob=x$iprob[length(x$iprob)]);
}

print.forecastSmooth <- function(x, ...){
    if(any(x$intervals!=c("none","n"))){
        level <- x$level;
        if(level>1){
            level <- level/100;
        }
        output <- cbind(x$forecast,x$lower,x$upper);
        colnames(output) <- c("Point forecast",paste0("Lower bound (",(1-level)/2*100,"%)"),paste0("Upper bound (",(1+level)/2*100,"%)"));
    }
    else{
        output <- x$forecast;
    }
    print(output);
}

print.iss <- function(x, ...){
    intermittent <- x$intermittent
    if(intermittent=="c"){
        intermittent <- "Croston";
    }
    else if(intermittent=="t"){
        intermittent <- "TSB";
    }
    else if(intermittent=="f"){
        intermittent <- "Fixed probability";
    }
    else{
        intermittent <- "None";
    }
    cat(paste0("Intermittent State-Space model estimated: ",intermittent,"\n"));
    cat(paste0("Smoothing parameter: ",round(x$C[1],3),"\n"));
    cat(paste0("Initial value: ",round(x$states[1],3),"\n"));
    cat(paste0("Probability forecast: ",round(x$forecast[1],3),"\n"));
}

#### Simulate data using provided object ####
simulate.smooth <- function(object, nsim=1, seed=NULL, obs=NULL, ...){
    if(is.null(obs)){
        obs <- length(object$actuals);
    }
    if(!is.null(seed)){
        set.seed(seed);
    }

    if(gregexpr("ETS",object$model)!=-1){
        model <- object$model;
        model <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
        if(any(unlist(gregexpr("C",model))==-1)){
            if(substr(model,1,1)=="A"){
                randomizer <- "rnorm";
            }
            else{
                randomizer <- "rlnorm";
            }
            simulatedData <- sim.es(model=model, frequency=frequency(object$actuals), phi=object$phi,
                                    persistence=object$persistence, initial=object$initial, initialSeason=object$initialSeason,
                                    obs=obs,nsim=nsim,silent=TRUE,iprob=object$iprob[length(object$iprob)],
                                    randomizer=randomizer,mean=0,sd=sqrt(object$s2),...);
        }
        else{
            message("Sorry, but we cannot simulate data out of combined model.");
            simulatedData <- NA;
        }
    }
    else{
        model <- substring(object$model,1,unlist(gregexpr("\\(",object$model))[1]-1);
        message(paste0("Sorry, but simulate is not yet available for the model ",model,"."));
        simulatedData <- NA;
    }
    return(simulatedData);
}

#### Summary of objects ####
summary.smooth <- function(object, ...){
    print(object);
}

summary.forecastSmooth <- function(object, ...){
    print(object);
}

summary.iss <- function(object, ...){
    print(object);
}
