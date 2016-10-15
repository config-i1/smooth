sma <- function(data, order=NULL, ic=c("AICc","AIC","BIC"),
                h=10, holdout=FALSE,
                intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                silent=c("none","all","graph","legend","output"), ...){
# Function constructs simple moving average in state-space model

#    Copyright (C) 2016  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

# If a previous model provided as a model, write down the variables
    if(exists("model")){
        if(is.null(model$model)){
        }
        else if(gregexpr("SMA",model$model)==-1){
            stop("The provided model is not Simple Moving Average!",call.=FALSE);
        }
        order <- model$order;
    }

    initial <- "backcasting";
    intermittent <- "none";
    bounds <- "admissible";
    cfType <- "MSE";
    xreg <- NULL;
    n.exovars <- 1;
    ivar <- 0;

##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- environment();
    ssInput(modelType="sma",ParentEnvironment=environment());

##### Preset y.fit, y.for, errors and basic parameters #####
    y.fit <- rep(NA,obsInsample);
    y.for <- rep(NA,h);
    errors <- rep(NA,obsInsample);
    maxlag <- 1;

# These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

    if(!is.null(order)){
        if(obsInsample < order){
            stop("Sorry, but we don't have enough observations for that order.",call.=FALSE);
        }
    }

# sd of residuals + a parameter... n.components not included.
    n.param <- 1 + 1;

# Cost function for GES
CF <- function(C){
    fitting <- fitterwrap(matvt, matF, matw, y, vecg,
                          modellags, Etype, Ttype, Stype, initialType,
                          matxt, matat, matFX, vecgX, ot);

    cfRes <- mean(fitting$errors^2);

    return(cfRes);
}

CreatorSMA <- function(silentText=FALSE,...){
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();
    environment(CF) <- environment();

    n.components <- order;
    #n.param <- n.components + 1;
    if(order>1){
        matF <- rbind(cbind(rep(1/n.components,n.components-1),diag(n.components-1)),c(1/n.components,rep(0,n.components-1)));
        matw <- matrix(c(1,rep(0,n.components-1)),1,n.components);
    }
    else{
        matF <- matrix(1,1,1);
        matw <- matrix(1,1,1);
    }
    vecg <- matrix(1/n.components,n.components);
    matvt <- matrix(NA,obsStates,n.components);
    matvt[1,] <- rep(mean(y[1:order]),n.components);
    modellags <- rep(1,n.components);

##### Prepare exogenous variables #####
    xregdata <- ssXreg(data=data, xreg=NULL, updateX=FALSE,
                       persistenceX=NULL, transitionX=NULL, initialX=NULL,
                       obsInsample=obsInsample, obsAll=obsAll, obsStates=obsStates, maxlag=maxlag, h=h, silent=silentText);
    matxt <- xregdata$matxt;
    matat <- xregdata$matat;
    matFX <- xregdata$matFX;
    vecgX <- xregdata$vecgX;

    C <- NULL;
    cfObjective <- CF(C);

    IC.values <- ICFunction(n.param=n.param,C=C,Etype=Etype);
    ICs <- IC.values$ICs;
    bestIC <- ICs["AICc"];

    return(list(cfObjective=cfObjective,ICs=ICs,bestIC=bestIC,n.param=n.param,n.components=n.components,
                matF=matF,vecg=vecg,matvt=matvt,matw=matw,modellags=modellags,
                matxt=matxt,matat=matat,matFX=matFX,vecgX=vecgX));
}

#####Start the calculations#####
    environment(ssForecaster) <- environment();
    environment(ssFitter) <- environment();

    if(is.null(order)){
        maxOrder <- min(36,obsInsample/2);
        ICs <- rep(NA,maxOrder);
        smaValuesAll <- list(NA);
        for(i in 1:maxOrder){
            order <- i;
            smaValuesAll[[i]] <- CreatorSMA(silentText);
            ICs[i] <- smaValuesAll[[i]]$bestIC;
        }
        order <- which(ICs==min(ICs,na.rm=TRUE))[1];
        smaValues <- smaValuesAll[[order]];
    }
    else{
        smaValues <- CreatorSMA(silentText);
    }

    list2env(smaValues,environment());

##### Fit simple model and produce forecast #####
    ssFitter(ParentEnvironment=environment());
    ssForecaster(ParentEnvironment=environment());

##### Do final check and make some preparations for output #####
    if(any(is.na(y.fit),is.na(y.for))){
        warning("Something went wrong during the optimisation and NAs were produced!",call.=FALSE,immediate.=TRUE);
        warning("Please check the input and report this error to the maintainer if it persists.",call.=FALSE,immediate.=TRUE);
    }

    if(holdout==T){
        y.holdout <- ts(data[(obsInsample+1):obsAll],start=start(y.for),frequency=frequency(data));
        errormeasures <- errorMeasurer(y.holdout,y.for,y);
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

    modelname <- paste0("SMA(",order,")");

##### Make a plot #####
    if(!silentGraph){
        if(intervals){
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
                       level=level,legend=!silentLegend,main=modelname);
        }
        else{
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                    level=level,legend=!silentLegend,main=modelname);
        }
    }

##### Return values #####
    model <- list(model=modelname,timeElapsed=Sys.time()-startTime,
                  states=matvt,transition=matF,persistence=vecg,
                  order=order, initialType=initialType, nParam=n.param,
                  fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,
                  errors=errors.mat,s2=s2,intervals=intervalsType,level=level,
                  actuals=data,holdout=y.holdout,
                  ICs=ICs,cf=cfObjective,cfType=cfType,accuracy=errormeasures);
    return(structure(model,class="smooth"));
}
