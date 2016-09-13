forecast <- function(object,...) UseMethod("forecast")
AICc <- function(object,...) UseMethod("AICc")

##### Functions of "es" #####
#logLik.smooth <- function(object,...)
#{
#  structure(object$logLik,df=length(coef(object)),class="logLik");
#}

AIC.smooth <- function(object, ...){
    if(gregexpr("ETS",object$model)!=-1){
        if(any(unlist(gregexpr("C",object$model))==-1)){
            return(object$ICs["AIC"]);
        }
        else{
            if(substring(names(object$ICs),10,nchar(names(object$ICs)))=="AIC"){
                return(object$ICs);
            }
            else{
                message("ICs were combined during the model construction. Nothing to return.");
                return(NULL);
            }
        }
    }
    else{
        return(object$ICs["AIC"]);
    }
}

AICc.smooth <- function(object, ...){
    if(gregexpr("ETS",object$model)!=-1){
        if(any(unlist(gregexpr("C",object$model))==-1)){
            return(object$ICs["AICc"]);
        }
        else{
            if(substring(names(object$ICs),10,nchar(names(object$ICs)))=="AICc"){
                return(object$ICs);
            }
            else{
                message("ICs were combined during the model construction. Nothing to return.");
                return(NULL);
            }
        }
    }
    else{
        return(object$ICs["AICc"]);
    }
}

BIC.smooth <- function(object, ...){
    if(gregexpr("ETS",object$model)!=-1){
        if(any(unlist(gregexpr("C",object$model))==-1)){
            return(object$ICs["BIC"]);
        }
        else{
            if(substring(names(object$ICs),10,nchar(names(object$ICs)))=="BIC"){
                return(object$ICs);
            }
            else{
                message("ICs were combined during the model construction. Nothing to return.");
                return(NULL);
            }
        }
    }
    else{
        return(object$ICs["BIC"]);
    }
}

coef.smooth <- function(object, ...)
{
    if(gregexpr("CES",object$model)!=-1){
        return(c(object$A,object$B));
    }
    else if(gregexpr("ETS",object$model)!=-1){
        if(any(unlist(gregexpr("C",object$model))==-1)){
            # If this was normal ETS, return values
            return(c(object$persistence,object$initial,object$initial.season));
        }
        else{
            # If we did combinations, we cannot return anything
            message("Combination of models was done, so there are no coefficients to return");
            return(NULL);
        }
    }
    else if(gregexpr("GES",object$model)!=-1){
        parameters <- c(object$measurement,object$transition,object$persistence,object$initial);
        names(parameters) <- c(paste0("Measurement ",c(1:length(object$measurement))),
                               paste0("Transition ",c(1:length(object$transition))),
                               paste0("Persistence ",c(1:length(object$persistence))),
                               paste0("Initial ",c(1:length(object$initial))));
        return(parameters);
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

        return(parameters);
    }
}

fitted.smooth <- function(object, ...){
    return(object$fitted);
}

forecast.smooth <- function(object, h=10, intervals=TRUE,
                        intervalsType=c("parametric","semiparametric","nonparametric","asymmetric"),
                        level=0.95, ...){
    if(gregexpr("ETS",object$model)!=-1){
        newModel <- es(object$actuals,model=object,h=h,intervals=intervals,intervalsType=intervalsType,level=level,silent="all",...);
    }
    else if(gregexpr("CES",object$model)!=-1){
        newModel <- ces(object$actuals,model=object,h=h,intervals=intervals,intervalsType=intervalsType,level=level,silent="all",...);
    }
    else if(gregexpr("GES",object$model)!=-1){
        newModel <- ges(object$actuals,model=object,h=h,intervals=intervals,intervalsType=intervalsType,level=level,silent="all",...);
    }
    else if(gregexpr("ARIMA",object$model)!=-1){
        newModel <- ssarima(object$actuals,model=object,h=h,intervals=intervals,intervalsType=intervalsType,level=level,silent="all",...);
    }
    else if(gregexpr("SMA",object$model)!=-1){
        newModel <- sma(object$actuals,model=object,h=h,intervals=intervals,intervalsType=intervalsType,level=level,silent="all",...);
    }
    else{
        stop("Wrong object provided. This needs to be either 'ETS' or 'CES' or 'GES' or 'SSARIMA' model.",call.=FALSE);
    }

    output <- list(model=newModel$model,fitted=newModel$fitted,actuals=newModel$actuals,
                   forecast=newModel$forecast,lower=newModel$lower,upper=newModel$upper,level=newModel$level,
                   intervals=intervals,mean=newModel$forecast);
    return(structure(output,class="forecastSmooth"));
}

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
    if(x$intervals){
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

print.smooth <- function(x, ...){
    holdout <- any(!is.na(x$holdout));
    intervals <- any(!is.na(x$lower));
    if(all(holdout==TRUE,intervals==TRUE)){
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
             intervalsType=x$intervalsType, level=x$level, ICs=x$ICs,
             holdout=holdout, insideintervals=insideintervals, errormeasures=x$accuracy,
             intermittent=x$intermittent, iprob=x$iprob[length(x$iprob)]);
}

print.forecastSmooth <- function(x, ...){
    if(x$intervals){
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

summary.smooth <- function(object, ...){
    print(object);
}

summary.forecastSmooth <- function(object, ...){
    print(object);
}

summary.iss <- function(object, ...){
    print(object);
}
