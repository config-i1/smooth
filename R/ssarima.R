utils::globalVariables(c("normalizer","constantValue","constantRequired","constantEstimate","C",
                         "ARValue","ARRequired","AREstimate","MAValue","MARequired","MAEstimate"));

ssarima <- function(data, ar.orders=c(0), i.orders=c(1), ma.orders=c(1), lags=c(1),
                    constant=FALSE, AR=NULL, MA=NULL,
                    initial=c("backcasting","optimal"),
                    cfType=c("MSE","MAE","HAM","MLSTFE","MSTFE","MSEh"),
                    h=10, holdout=FALSE, intervals=FALSE, level=0.95,
                    intervalsType=c("parametric","semiparametric","nonparametric"),
                    intermittent=c("none","auto","fixed","croston","tsb"),
                    bounds=c("admissible","none"),
                    silent=c("none","all","graph","legend","output"),
                    xreg=NULL, initialX=NULL, updateX=FALSE, persistenceX=NULL, transitionX=NULL, ...){
##### Function constructs SARIMA model (possible triple seasonality) using state-space approach
# ar.orders contains vector of seasonal ARs. ar.orders=c(2,1,3) will mean AR(2)*SAR(1)*SAR(3) - model with double seasonality.
#
#    Copyright (C) 2016  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

    # If a previous model provided as a model, write down the variables
    if(exists("model")){
        if(is.null(model$model)){
            stop("The provided model is not ARIMA.",call.=FALSE);
        }
        else if(gregexpr("ARIMA",model$model)==-1){
            stop("The provided model is not ARIMA.",call.=FALSE);
        }
        intermittent <- model$intermittent;
        if(any(intermittent==c("p","provided"))){
            warning("The provided model had predefined values of occurences for the holdout. We don't have them.",call.=FALSE);
            warning("Switching to intermittent='auto'.",call.=FALSE);
            intermittent <- "a";
        }
        if(!is.null(model$initial)){
            initial <- model$initial;
        }
        xreg <- model$xreg;
        initialX <- model$initialX;
        persistenceX <- model$persistenceX;
        transitionX <- model$transitionX;
        if(any(c(persistenceX)!=0) | any((transitionX!=0)&(transitionX!=1))){
            updateX <- TRUE;
        }
        AR <- model$AR;
        MA <- model$MA;
        constant <- model$constant;
        model <- model$model;
        arima.orders <- paste0(c("",substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1),"")
                               ,collapse=";");
        comas <- unlist(gregexpr("\\,",arima.orders));
        semicolons <- unlist(gregexpr("\\;",arima.orders));
        ar.orders <- as.numeric(substring(arima.orders,semicolons[-length(semicolons)]+1,comas[2*(1:(length(comas)/2))-1]-1));
        i.orders <- as.numeric(substring(arima.orders,comas[2*(1:(length(comas)/2))-1]+1,comas[2*(1:(length(comas)/2))-1]+1));
        ma.orders <- as.numeric(substring(arima.orders,comas[2*(1:(length(comas)/2))]+1,semicolons[-1]-1));
        if(any(unlist(gregexpr("\\[",model))!=-1)){
            lags <- as.numeric(substring(model,unlist(gregexpr("\\[",model))+1,unlist(gregexpr("\\]",model))-1));
        }
        else{
            lags <- 1;
        }
    }

##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- environment();
    ssInput(modelType="ssarima",ParentEnvironment=environment());

# Prepare lists for the polynomials
    P <- list(NA);
    D <- list(NA);
    Q <- list(NA);

    if(n.components > 0){
# Transition matrix, measurement vector and persistence vector + state vector
        matF <- rbind(cbind(rep(0,n.components-1),diag(n.components-1)),rep(0,n.components));
        matw <- matrix(c(1,rep(0,n.components-1)),1,n.components);
        vecg <- matrix(0.1,n.components,1);
        matvt <- matrix(NA,obsStates,n.components);
        if(constantRequired){
            matF <- cbind(rbind(matF,rep(0,n.components)),c(1,rep(0,n.components-1),1));
            matw <- cbind(matw,0);
            vecg <- rbind(vecg,0);
            matvt <- cbind(matvt,rep(1,obsStates));
        }
    }
    else{
        matw <- matF <- matrix(1,1,1);
        vecg <- matrix(0,1,1);
        matvt <- matrix(1,obsStates,1);
        modellags <- matrix(1,1,1);
    }

##### Preset y.fit, y.for, errors and basic parameters #####
    y.fit <- rep(NA,obsInsample);
    y.for <- rep(NA,h);
    errors <- rep(NA,obsInsample);

##### Prepare exogenous variables #####
    xregdata <- ssXreg(data=data, xreg=xreg, updateX=updateX,
                       persistenceX=persistenceX, transitionX=transitionX, initialX=initialX,
                       obsInsample=obsInsample, obsAll=obsAll, obsStates=obsStates, maxlag=maxlag, h=h, silent=silentText);
    n.exovars <- xregdata$n.exovars;
    matxt <- xregdata$matxt;
    matat <- xregdata$matat;
    matFX <- xregdata$matFX;
    vecgX <- xregdata$vecgX;
    xreg <- xregdata$xreg;
    xregEstimate <- xregdata$xregEstimate;
    FXEstimate <- xregdata$FXEstimate;
    gXEstimate <- xregdata$gXEstimate;
    initialXEstimate <- xregdata$initialXEstimate;
    xregNames <- colnames(matat);

# These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

    # Check number of parameters vs data
    n.param.max <- n.param.max + FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) + initialXEstimate*ncol(matat);

##### Check number of observations vs number of max parameters #####
    if(obsNonzero <= n.param.max){
        stop(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                        n.param.max," while the number of observations is ",obsNonzero,"!"),call.=FALSE);
    }

##### Preset values of matvt ######
    slope <- cov(yot[1:min(12,obsNonzero),],c(1:min(12,obsNonzero)))/var(c(1:min(12,obsNonzero)));
    intercept <- sum(yot[1:min(12,obsNonzero),])/min(12,obsNonzero) - slope * (sum(c(1:min(12,obsNonzero)))/min(12,obsNonzero) - 1);
    initialStuff <- c(intercept,-intercept,rep(slope,n.components));
    matvt[1,1:n.components] <- initialStuff[1:n.components];

polyroots <- function(C){
    polysos.ar <- 0;
    polysos.ma <- 0;
    n.coef <- 0;
    matF[,1] <- 0;
    if(n.components > 0){
        ar.inner.coef <- ma.inner.coef <- 0;
        for(i in 1:length(lags)){
            if((ar.orders*lags)[i]!=0){
                armat <- matrix(0,lags[i],ar.orders[i]);
                if(AREstimate){
                    armat[lags[i],] <- -C[n.coef+(1:ar.orders[i])];
                    n.coef <- n.coef + ar.orders[i];
                }
                else{
                    armat[lags[i],] <- -ARValue[ar.inner.coef+(1:ar.orders[i])];
                    ar.inner.coef <- ar.inner.coef + ar.orders[i];
                }
                P[[i]] <- c(1,c(armat));

            }
            else{
                P[[i]] <- 1;
            }

            if((i.orders*lags)[i]!=0){
                D[[i]] <- c(1,rep(0,max(lags[i]-1,0)),-1);
            }
            else{
                D[[i]] <- 1;
            }

            if((ma.orders*lags)[i]!=0){
                armat <- matrix(0,lags[i],ma.orders[i]);
                if(MAEstimate){
                    armat[lags[i],] <- C[n.coef+(1:ma.orders[i])];
                    n.coef <- n.coef + ma.orders[i];
                }
                else{
                    armat[lags[i],] <- MAValue[ma.inner.coef+(1:ma.orders[i])];
                    ma.inner.coef <- ma.inner.coef + ma.orders[i];
                }
                Q[[i]] <- c(1,c(armat));
            }
            else{
                Q[[i]] <- 1;
            }
        }

##### Polynom multiplication is the slowest part now #####
        polysos.i <- as.polynomial(1);
        for(i in 1:length(lags)){
            polysos.i <- polysos.i * polynomial(D[[i]])^i.orders[i];
        }

        polysos.ar <- 1;
        polysos.ma <- 1;
        for(i in 1:length(P)){
            polysos.ar <- polysos.ar * polynomial(P[[i]]);
        }
        polysos.ari <- polysos.ar * polysos.i;

        for(i in 1:length(Q)){
            polysos.ma <- polysos.ma * polynomial(Q[[i]]);
        }

        if(length((polysos.ari))!=1){
            matF[1:(length(polysos.ari)-1),1] <- -(polysos.ari)[2:length(polysos.ari)];
        }

### The MA parameters are in the style "1 + b1 * B".
        vecg[1:n.components,] <- (-polysos.ari + polysos.ma)[2:(n.components+1)];
        vecg[is.na(vecg),] <- 0;

        if(initialType=="o"){
            vt <- C[(n.coef + 1):(n.coef + n.components)];
            n.coef <- n.coef + n.components;
        }
        else if(initialType=="b"){
            vt <- matvt[1,];
            vt[-1] <- vt[1] * matF[-1,1];
        }
        else{
            vt <- initialValue;
        }

        if(constantRequired){
            if(constantEstimate){
                vt[n.components+constantRequired] <- C[(n.coef + 1)];
                n.coef <- n.coef + 1;
            }
            else{
                vt[n.components+constantRequired] <- constantValue;
            }
        }
    }
    else{
        matF[1,1] <- 1;
        if(constantEstimate){
            vt <- C[n.coef+1];
            n.coef <- n.coef + 1;
        }
        else{
            vt <- constantValue;
        }
    }

# If exogenous are included
    if(!is.null(xreg)){
        at <- matrix(NA,maxlag,n.exovars);
        if(initialXEstimate){
            at[,] <- rep(C[n.coef+(1:n.exovars)],each=maxlag);
            n.coef <- n.coef + n.exovars;
        }
        else{
            at <- matat[1:maxlag,];
        }
        if(FXEstimate){
            matFX <- matrix(C[n.coef+(1:(n.exovars^2))],n.exovars,n.exovars);
            n.coef <- n.coef + n.exovars^2;
        }

        if(gXEstimate){
            vecgX <- matrix(C[n.coef+(1:n.exovars)],n.exovars,1);
            n.coef <- n.coef + n.exovars;
        }
    }
    else{
        at <- matrix(0,maxlag,n.exovars);
    }

    return(list(matF=matF,vecg=vecg,vt=vt,at=at,matFX=matFX,vecgX=vecgX,polysos.ar=polysos.ar,polysos.ma=polysos.ma));
}

# Cost function for SSARIMA
CF <- function(C){
    elements <- polyroots(C);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;
    polysos.ar <- elements$polysos.ar;
    polysos.ma <- elements$polysos.ma;

    if(bounds=="a" & (n.components > 0)){
        arroots <- abs(polyroot(polysos.ar));
        if(any(arroots<1)){
            return(max(arroots)*1E+100);
        }
        maroots <- abs(polyroot(polysos.ma));
        if(any(maroots<1)){
            return(max(maroots)*1E+100);
        }
    }

    cfRes <- optimizerwrap(matvt, matF, matw, y, vecg,
                           h, modellags, Etype, Ttype, Stype,
                           multisteps, cfType, normalizer, initialType,
                           matxt, matat, matFX, vecgX, ot);

    if(is.nan(cfRes) | is.na(cfRes) | is.infinite(cfRes)){
        cfRes <- 1e+100;
    }

    return(cfRes);
}

##### Estimate ssarima or just use the provided values #####
CreatorSSARIMA <- function(silentText=FALSE,...){
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();

    n.param <- 1 + n.components*(initialType!="b") + sum(ar.orders)*ARRequired + sum(ma.orders)*MARequired + constantRequired + FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) + initialXEstimate*ncol(matat);

    # If there is something to optimise, let's do it.
    if(any((initialType=="o"),(AREstimate),(MAEstimate),
           (xregEstimate),(FXEstimate),(gXEstimate),(constantEstimate))){

        C <- NULL;
        if(n.components > 0){
# ar terms, ma terms from season to season...
            if(AREstimate){
                C <- c(C,rep(0.1,sum(ar.orders)));
            }
            if(MAEstimate){
                C <- c(C,rep(0.1,sum(ma.orders)));
            }

# initial values of state vector and the constant term
            if(initialType=="o"){
                slope <- cov(yot[1:min(12,obsNonzero),],c(1:min(12,obsNonzero)))/var(c(1:min(12,obsNonzero)));
                intercept <- sum(yot[1:min(12,obsNonzero),])/min(12,obsNonzero) - slope * (sum(c(1:min(12,obsNonzero)))/min(12,obsNonzero) - 1);
                initialStuff <- c(rep(intercept,n.components));
                C <- c(C,initialStuff[1:n.components]);
            }
        }

        if(constantEstimate){
            if(all(i.orders==0)){
                C <- c(C,sum(yot)/obsInsample);
            }
            else{
                C <- c(C,sum(diff(yot))/obsInsample);
            }
        }

# initials, transition matrix and persistence vector
        if(xregEstimate){
            if(initialXEstimate){
                C <- c(C,matat[maxlag,]);
            }
            if(updateX){
                if(FXEstimate){
                    C <- c(C,c(diag(n.exovars)));
                }
                if(gXEstimate){
                    C <- c(C,rep(0,n.exovars));
                }
            }
        }

# Optimise model. First run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
        C <- res$solution;
        if(initialType=="o"){
# Optimise model. Second run
            res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-10, "maxeval"=1000));
            C <- res$solution;
        }
        cfObjective <- res$objective;
    }
    else{
        C <- NULL;

# initial values of state vector and the constant term
        if(n.components>0 & initialType=="p"){
            matvt[1,1:n.components] <- initialValue;
        }
        if(constantRequired){
            matvt[1,(n.components+1)] <- constantValue;
        }

        cfObjective <- CF(C);
    }

# Change cfType for model selection
    if(multisteps){
        #     if(substring(cfType,1,1)=="a"){
        cfType <- "aTFL";
        #     }
        #     else{
        #         cfType <- "TFL";
        #     }
    }
    else{
        cfType <- "MSE";
    }

    IC.values <- ICFunction(n.param=n.param+n.param.intermittent,C=C,Etype=Etype);
    ICs <- IC.values$ICs;
    bestIC <- ICs["AICc"];

# Revert to the provided cost function
    cfType <- cfTypeOriginal

    return(list(cfObjective=cfObjective,C=C,ICs=ICs,bestIC=bestIC,n.param=n.param));
}

#####Start the calculations#####
    environment(intermittentParametersSetter) <- environment();
    environment(intermittentMaker) <- environment();
    environment(ssForecaster) <- environment();
    environment(ssFitter) <- environment();

    # If auto intermittent, then estimate model with intermittent="n" first.
    if(any(intermittent==c("a","n"))){
        intermittentParametersSetter(intermittent="n",ParentEnvironment=environment());
    }
    else{
        intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
        intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    }

    ssarimaValues <- CreatorSSARIMA(silentText);

##### If intermittent=="a", run a loop and select the best one #####
    if(intermittent=="a"){
        if(cfType!="MSE"){
            warning(paste0("'",cfType,"' is used as cost function instead of 'MSE'. A wrong intermittent model may be selected"),call.=FALSE);
        }
        if(!silentText){
            cat("Selecting appropriate type of intermittency... ");
        }
# Prepare stuff for intermittency selection
        intermittentModelsPool <- c("n","f","c","t");
        intermittentICs <- rep(1e+10,length(intermittentModelsPool));
        intermittentModelsList <- list(NA);
        intermittentICs <- ssarimaValues$bestIC;

        for(i in 2:length(intermittentModelsPool)){
            intermittentParametersSetter(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentMaker(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentModelsList[[i]] <- CreatorSSARIMA(silentText=TRUE);
            intermittentICs[i] <- intermittentModelsList[[i]]$bestIC;
            if(intermittentICs[i]>intermittentICs[i-1]){
                break;
            }
        }
        intermittentICs[is.nan(intermittentICs)] <- 1e+100;
        intermittentICs[is.na(intermittentICs)] <- 1e+100;
        iBest <- which(intermittentICs==min(intermittentICs));

        if(!silentText){
            cat("Done!\n");
        }
        if(iBest!=1){
            intermittent <- intermittentModelsPool[iBest];
            intermittentModel <- intermittentModelsList[[iBest]];
            ssarimaValues <- intermittentModelsList[[iBest]];
        }
        else{
            intermittent <- "n"
        }

        intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
        intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    }

    list2env(ssarimaValues,environment());

# Prepare for fitting
    elements <- polyroots(C);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;
    polysos.ar <- elements$polysos.ar;
    polysos.ma <- elements$polysos.ma;
    arroots <- abs(polyroot(polysos.ar));
    maroots <- abs(polyroot(polysos.ma));
    n.components <- n.components + constantRequired;

    # Write down Fisher Information if needed
    if(FI){
        environment(likelihoodFunction) <- environment();
        FI <- numDeriv::hessian(likelihoodFunction,C);
    }

##### Fit simple model and produce forecast #####
    ssFitter(ParentEnvironment=environment());
    ssForecaster(ParentEnvironment=environment());

##### Do final check and make some preparations for output #####
    if(any(is.na(y.fit),is.na(y.for))){
        warning("Something went wrong during the optimisation and NAs were produced!",call.=FALSE,immediate.=TRUE);
        warning("Please check the input and report this error to the maintainer if it persists.",call.=FALSE,immediate.=TRUE);
    }

# Write down initials of states vector and exogenous
    if(initialType!="p"){
        if(constantRequired==TRUE){
            initialValue <- matvt[1,-ncol(matvt)];
        }
        else{
            initialValue <- matvt[1,];
        }
    }
    if(initialXEstimate){
        initialX <- matat[1,];
    }

    # Write down the probabilities from intermittent models
    pt <- ts(c(as.vector(pt),as.vector(pt.for)),start=start(data),frequency=datafreq);
    if(intermittent=="f"){
        intermittent <- "fixed";
    }
    else if(intermittent=="c"){
        intermittent <- "croston";
    }
    else if(intermittent=="t"){
        intermittent <- "tsb";
    }
    else if(intermittent=="n"){
        intermittent <- "none";
    }
    else if(intermittent=="p"){
        intermittent <- "provided";
    }

# Fill in the rest of matvt
    matvt <- ts(matvt,start=(time(data)[1] - deltat(data)*maxlag),frequency=frequency(data));
    if(!is.null(xreg)){
        matvt <- cbind(matvt,matat[1:nrow(matvt),]);
        colnames(matvt) <- c(paste0("Component ",c(1:max(1,n.components))),colnames(matat));
        if(updateX){
            rownames(vecgX) <- xregNames;
            dimnames(matFX) <- list(xregNames,xregNames);
        }
    }
    else{
        colnames(matvt) <- paste0("Component ",c(1:max(1,n.components)));
    }
    if(constantRequired){
        colnames(matvt)[n.components] <- "Constant";
    }

# AR terms
    if(any(ar.orders!=0)){
        ARterms <- matrix(0,max(ar.orders),sum(ar.orders!=0),
                          dimnames=list(paste0("AR(",c(1:max(ar.orders)),")"),
                                        paste0("Lag ",lags[ar.orders!=0])));
    }
    else{
        ARterms <- matrix(0,1,1);
    }
# Differences
    if(any(i.orders!=0)){
        Iterms <- matrix(0,1,length(i.orders),
                          dimnames=list("I(...)",paste0("Lag ",lags)));
        Iterms[,] <- i.orders;
    }
    else{
        Iterms <- 0;
    }
# MA terms
    if(any(ma.orders!=0)){
        MAterms <- matrix(0,max(ma.orders),sum(ma.orders!=0),
                          dimnames=list(paste0("MA(",c(1:max(ma.orders)),")"),
                                        paste0("Lag ",lags[ma.orders!=0])));
    }
    else{
        MAterms <- matrix(0,1,1);
    }

    n.coef <- ar.coef <- ma.coef <- 0;
    ar.i <- ma.i <- 1;
    for(i in 1:length(ar.orders)){
        if(ar.orders[i]!=0){
            if(AREstimate){
                ARterms[1:ar.orders[i],ar.i] <- C[n.coef+(1:ar.orders[i])];
                n.coef <- n.coef + ar.orders[i];
            }
            else{
                ARterms[1:ar.orders[i],ar.i] <- ARValue[ar.coef+(1:ar.orders[i])];
                ar.coef <- ar.coef + ar.orders[i];
            }
            ar.i <- ar.i + 1;
        }
        if(ma.orders[i]!=0){
            if(MAEstimate){
                MAterms[1:ma.orders[i],ma.i] <- C[n.coef+(1:ma.orders[i])];
                n.coef <- n.coef + ma.orders[i];
            }
            else{
                MAterms[1:ma.orders[i],ma.i] <- MAValue[ma.coef+(1:ma.orders[i])];
                ma.coef <- ma.coef + ma.orders[i];
            }
            ma.i <- ma.i + 1;
        }
    }

    if(holdout==T){
        y.holdout <- ts(data[(obsInsample+1):obsAll],start=start(y.for),frequency=frequency(data));
        errormeasures <- errorMeasurer(y.holdout,y.for,y);
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

# Give model the name
    if((length(ar.orders)==1) && all(lags==1)){
        modelname <- paste0("ARIMA(",ar.orders,",",i.orders,",",ma.orders,")");
    }
    else{
        modelname <- "";
        for(i in 1:length(ar.orders)){
            modelname <- paste0(modelname,"(",ar.orders[i],",");
            modelname <- paste0(modelname,i.orders[i],",");
            modelname <- paste0(modelname,ma.orders[i],")[",lags[i],"]");
        }
        modelname <- paste0("SARIMA",modelname);
    }
    if(all(intermittent!=c("n","none"))){
        modelname <- paste0("i",modelname);
    }

    if(constantRequired){
        if(constantEstimate){
            constantValue <- matvt[1,n.components];
        }
        const <- constantValue;

        if(all(i.orders==0)){
            modelname <- paste0(modelname," with constant");
        }
        else{
            modelname <- paste0(modelname," with drift");
        }
    }
    else{
        const <- FALSE;
        constantValue <- NULL;
    }

##### Print warnings #####
    if(any(maroots<1)){
        if(bounds!="a"){
            warning("Unstable model was estimated! Use bounds='admissible' to address this issue!",call.=FALSE);
        }
        else{
            warning("Something went wrong in optimiser - unstable model was estimated! Please report this error to the maintainer."
                    ,call.=FALSE);
        }
    }
    if(any(arroots<1)){
        if(bounds!="a"){
            warning("Non-stationary model was estimated! Beware of explosions! Use bounds='admissible' to address this issue!"
                    ,call.=FALSE);
        }
        else{
            warning("Something went wrong in optimiser - non-stationary model was estimated! Please report this error to the maintainer."
                    ,call.=FALSE);
        }
    }

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
                  AR=ARterms,I=Iterms,MA=MAterms,constant=const,
                  initialType=initialType,initial=initialValue,
                  nParam=n.param,
                  fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,
                  errors=errors.mat,s2=s2,intervalsType=intervalsType,level=level,
                  actuals=data,holdout=y.holdout,iprob=pt,intermittent=intermittent,
                  xreg=xreg,updateX=updateX,initialX=initialX,persistenceX=vecgX,transitionX=matFX,
                  ICs=ICs,cf=cfObjective,cfType=cfType,FI=FI,accuracy=errormeasures);
    return(structure(model,class="smooth"));
}
