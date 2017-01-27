utils::globalVariables(c("normalizer","constantValue","constantRequired","constantEstimate","C",
                         "ARValue","ARRequired","AREstimate","MAValue","MARequired","MAEstimate"));

ssarima <- function(data, orders=list(ar=0,i=c(1),ma=c(1)), lags=c(1),
                    constant=FALSE, AR=NULL, MA=NULL,
                    initial=c("backcasting","optimal"), ic=c("AICc","AIC","BIC"),
                    cfType=c("MSE","MAE","HAM","MLSTFE","MSTFE","MSEh"),
                    h=10, holdout=FALSE,
                    intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                    intermittent=c("none","auto","fixed","croston","tsb","sba"),
                    bounds=c("admissible","none"),
                    silent=c("none","all","graph","legend","output"),
                    xreg=NULL, xregDo=c("use","select"), initialX=NULL,
                    updateX=FALSE, persistenceX=NULL, transitionX=NULL, ...){
##### Function constructs SARIMA model (possible triple seasonality) using state-space approach
# ar.orders contains vector of seasonal ARs. ar.orders=c(2,1,3) will mean AR(2)*SAR(1)*SAR(3) - model with double seasonality.
#
#    Copyright (C) 2016  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

    # If a previous model provided as a model, write down the variables
    if(exists("model",inherits=FALSE)){
        if(is.null(model$model)){
            stop("The provided model is not ARIMA.",call.=FALSE);
        }
        else if(gregexpr("ARIMA",model$model)==-1){
            stop("The provided model is not ARIMA.",call.=FALSE);
        }

# If this is a normal ARIMA, do things
        if(any(unlist(gregexpr("combine",model$model))==-1)){
            intermittent <- model$intermittent;
            if(any(intermittent==c("p","provided"))){
                warning("The provided model had predefined values of occurences for the holdout. We don't have them.",call.=FALSE);
                warning("Switching to intermittent='auto'.",call.=FALSE);
                intermittent <- "a";
            }
            if(!is.null(model$initial)){
                initial <- model$initial;
            }
            if(is.null(xreg)){
                xreg <- model$xreg;
            }
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
        else{
            stop("The provided model is a combination of ARIMAs. We cannot fit that.",call.=FALSE);
        }
    }
    else if(!is.null(orders)){
        ar.orders <- orders$ar;
        i.orders <- orders$i;
        ma.orders <- orders$ma;
    }

# If orders are provided in ellipsis via ar.orders, write them down.
    if(exists("ar.orders",inherits=FALSE)){
        if(is.null(ar.orders)){
            ar.orders <- 0;
        }
    }
    else{
        ar.orders <- 0;
    }
    if(exists("i.orders",inherits=FALSE)){
        if(is.null(i.orders)){
            i.orders <- 0;
        }
    }
    else{
        i.orders <- 0;
    }
    if(exists("ma.orders",inherits=FALSE)){
        if(is.null(ma.orders)){
            ma.orders <- 0;
        }
    }
    else{
        ma.orders <- 0;
    }

##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- environment();
    ssInput(modelType="ssarima",ParentEnvironment=environment());

# Cost function for SSARIMA
CF <- function(C){

    # cfRes <- costfuncARIMA(ar.orders, ma.orders, i.orders, lags, nComponents,
    #                        ARValue, MAValue, constantValue, C,
    #                        matvt, matF, matw, y, vecg,
    #                        h, modellags, Etype, Ttype, Stype,
    #                        multisteps, cfType, normalizer, initialType,
    #                        nExovars, matxt, matat, matFX, vecgX, ot,
    #                        AREstimate, MAEstimate, constantRequired, constantEstimate,
    #                        xregEstimate, updateX, FXEstimate, gXEstimate, initialXEstimate,
    #                        bounds);

    elements <- polysoswrap(ar.orders, ma.orders, i.orders, lags, nComponents,
                            ARValue, MAValue, constantValue, C,
                            matvt, vecg, matF,
                            initialType, nExovars, matat, matFX, vecgX,
                            AREstimate, MAEstimate, constantRequired, constantEstimate,
                            xregEstimate, updateX, FXEstimate, gXEstimate, initialXEstimate);
    # matF <- elements$matF;
    # vecg <- elements$vecg;
    # matvt[,] <- elements$matvt;
    # matvt[1,] <- matrixPowerWrap(matF,nComponents+1) %*% matvt[1,];
    # matat[,] <- elements$matat;
    # matFX <- elements$matFX;
    # vecgX <- elements$vecgX;
    polysos.ar <- elements$arPolynomial;
    polysos.ma <- elements$maPolynomial;

    if(bounds=="a" & (nComponents > 0)){
        arroots <- abs(polyroot(polysos.ar));
        if(any(arroots<1)){
            return(max(arroots)*1E+100);
        }
        maroots <- abs(polyroot(polysos.ma));
        if(any(maroots<1)){
            return(max(maroots)*1E+100);
        }
    }

    cfRes <- optimizerwrap(elements$matvt, elements$matF, matw, y, elements$vecg,
                           h, modellags, Etype, Ttype, Stype,
                           multisteps, cfType, normalizer, initialType,
                           matxt, elements$matat, elements$matFX, elements$vecgX, ot);

    if(is.nan(cfRes) | is.na(cfRes) | is.infinite(cfRes)){
        cfRes <- 1e+100;
    }

    return(cfRes);
}

##### Estimate ssarima or just use the provided values #####
CreatorSSARIMA <- function(silentText=FALSE,...){
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();

    nParam <- 1 + nComponents*(initialType!="b") + sum(ar.orders)*ARRequired + sum(ma.orders)*MARequired + constantRequired + (!is.null(xreg)) * nExovars + (updateX)*(nExovars^2 + nExovars);

    # If there is something to optimise, let's do it.
    if(any((initialType=="o"),(AREstimate),(MAEstimate),
           (xregEstimate),(FXEstimate),(gXEstimate),(constantEstimate))){

        C <- NULL;
        if(nComponents > 0){
# ar terms, ma terms from season to season...
            if(AREstimate){
                C <- c(C,rep(0.1,sum(ar.orders)));
            }
            if(MAEstimate){
                C <- c(C,rep(0.1,sum(ma.orders)));
            }

# initial values of state vector and the constant term
            if(initialType=="o"){
                # slope <- cov(yot[1:min(12,obsNonzero),],c(1:min(12,obsNonzero)))/var(c(1:min(12,obsNonzero)));
                # intercept <- sum(yot[1:min(12,obsNonzero),])/min(12,obsNonzero) - slope * (sum(c(1:min(12,obsNonzero)))/min(12,obsNonzero) - 1);
                # initialStuff <- c(rep(intercept,nComponents));
                # C <- c(C,initialStuff[1:nComponents]);
                C <- c(C,matvt[1:nComponents,1]);
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
                    C <- c(C,c(diag(nExovars)));
                }
                if(gXEstimate){
                    C <- c(C,rep(0,nExovars));
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
        if(nComponents>0 & initialType=="p"){
            matvt[1,1:nComponents] <- initialValue;
        }
        if(constantRequired){
            matvt[1,(nComponents+1)] <- constantValue;
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

    ICValues <- ICFunction(nParam=nParam+nParamIntermittent,C=C,Etype=Etype);
    ICs <- ICValues$ICs;
    bestIC <- ICs["AICc"];
    logLik <- ICValues$llikelihood;

# Revert to the provided cost function
    cfType <- cfTypeOriginal

    return(list(cfObjective=cfObjective,C=C,ICs=ICs,bestIC=bestIC,nParam=nParam,logLik=logLik));
}

    # Prepare lists for the polynomials
    P <- list(NA);
    D <- list(NA);
    Q <- list(NA);

##### Preset values of matvt and other matrices ######
    if(nComponents > 0){
        # Transition matrix, measurement vector and persistence vector + state vector
        matF <- rbind(cbind(rep(0,nComponents-1),diag(nComponents-1)),rep(0,nComponents));
        matw <- matrix(c(1,rep(0,nComponents-1)),1,nComponents);
        vecg <- matrix(0.1,nComponents,1);
        matvt <- matrix(NA,obsStates,nComponents);
        if(constantRequired){
            matF <- cbind(rbind(matF,rep(0,nComponents)),c(1,rep(0,nComponents-1),1));
            matw <- cbind(matw,0);
            vecg <- rbind(vecg,0);
            matvt <- cbind(matvt,rep(1,obsStates));
        }
        if(initialType=="p"){
            matvt[1,1:nComponents] <- initialValue;
        }
        else{
            if(obsInsample<(nComponents+datafreq)){
                matvt[1:nComponents,] <- y[1:nComponents] + diff(y[1:(nComponents+1)]);
            }
            else{
                matvt[1:nComponents,] <- (y[1:nComponents]+y[1:nComponents+datafreq])/2;
            }
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

    if(xregDo=="u"){
        nExovars <- xregdata$nExovars;
        matxt <- xregdata$matxt;
        matat <- xregdata$matat;
        xregEstimate <- xregdata$xregEstimate;
        matFX <- xregdata$matFX;
        vecgX <- xregdata$vecgX;
        xregNames <- colnames(matxt);
    }
    else{
        nExovars <- 1;
        nExovarsOriginal <- xregdata$nExovars;
        matxtOriginal <- xregdata$matxt;
        matatOriginal <- xregdata$matat;
        xregEstimateOriginal <- xregdata$xregEstimate;
        matFXOriginal <- xregdata$matFX;
        vecgXOriginal <- xregdata$vecgX;

        matxt <- matrix(1,nrow(matxtOriginal),1);
        matat <- matrix(0,nrow(matatOriginal),1);
        xregEstimate <- FALSE;
        matFX <- matrix(1,1,1);
        vecgX <- matrix(0,1,1);
        xregNames <- NULL;
    }
    xreg <- xregdata$xreg;
    FXEstimate <- xregdata$FXEstimate;
    gXEstimate <- xregdata$gXEstimate;
    initialXEstimate <- xregdata$initialXEstimate;

    # These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

    # Check number of parameters vs data
    nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) + initialXEstimate*ncol(matat);
    nParamMax <- nParamMax + nParamExo + (intermittent!="n");

##### Check number of observations vs number of max parameters #####
    if(obsNonzero <= nParamMax){
        stop(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                    nParamMax," while the number of observations is ",obsNonzero,"!"),call.=FALSE);
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
        intermittentModelsPool <- c("n","f","c","t","s");
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

    if(xregDo!="u"){
        # Prepare for fitting
        elements <- polysoswrap(ar.orders, ma.orders, i.orders, lags, nComponents,
                                ARValue, MAValue, constantValue, C,
                                matvt, vecg, matF,
                                initialType, nExovars, matat, matFX, vecgX,
                                AREstimate, MAEstimate, constantRequired, constantEstimate,
                                xregEstimate, updateX, FXEstimate, gXEstimate, initialXEstimate);
        matF <- elements$matF;
        vecg <- elements$vecg;
        matvt[,] <- elements$matvt;
        matat[,] <- elements$matat;
        matFX <- elements$matFX;
        vecgX <- elements$vecgX;
        polysos.ar <- elements$arPolynomial;
        polysos.ma <- elements$maPolynomial;
        arroots <- abs(polyroot(polysos.ar));
        maroots <- abs(polyroot(polysos.ma));

        ssFitter(ParentEnvironment=environment());

        xregNames <- colnames(matxtOriginal);
        xregNew <- cbind(errors,xreg[1:nrow(errors),]);
        colnames(xregNew)[1] <- "errors";
        colnames(xregNew)[-1] <- xregNames;
        xregNew <- as.data.frame(xregNew);
        xregResults <- stepwise(xregNew, ic=ic, silent=TRUE, df=nParam+nParamIntermittent-1);
        xregNames <- names(coef(xregResults))[-1];
        nExovars <- length(xregNames);
        if(nExovars>0){
            xregEstimate <- TRUE;
            matxt <- as.data.frame(matxtOriginal)[,xregNames];
            matat <- as.data.frame(matatOriginal)[,xregNames];
            matFX <- diag(nExovars);
            vecgX <- matrix(0,nExovars,1);

            if(nExovars==1){
                matxt <- matrix(matxt,ncol=1);
                matat <- matrix(matat,ncol=1);
                colnames(matxt) <- colnames(matat) <- xregNames;
            }
            else{
                matxt <- as.matrix(matxt);
                matat <- as.matrix(matat);
            }
        }
        else{
            nExovars <- 1;
            xreg <- NULL;
        }

        if(!is.null(xreg)){
            ssarimaValues <- CreatorSSARIMA(silentText);
            list2env(ssarimaValues,environment());
        }
    }

    if(!is.null(xreg)){
        if(ncol(matat)==1){
            colnames(matxt) <- colnames(matat) <- xregNames;
        }
        xreg <- matxt;
    }
# Prepare for fitting
    elements <- polysoswrap(ar.orders, ma.orders, i.orders, lags, nComponents,
                            ARValue, MAValue, constantValue, C,
                            matvt, vecg, matF,
                            initialType, nExovars, matat, matFX, vecgX,
                            AREstimate, MAEstimate, constantRequired, constantEstimate,
                            xregEstimate, updateX, FXEstimate, gXEstimate, initialXEstimate);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[,] <- elements$matvt;
    matat[,] <- elements$matat;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;
    polysos.ar <- elements$arPolynomial;
    polysos.ma <- elements$maPolynomial;
    arroots <- abs(polyroot(polysos.ar));
    maroots <- abs(polyroot(polysos.ma));

    nComponents <- nComponents + constantRequired;
    # Write down Fisher Information if needed
    if(FI){
        environment(likelihoodFunction) <- environment();
        FI <- numDeriv::hessian(likelihoodFunction,C);
    }

##### Fit simple model and produce forecast #####
    ssFitter(ParentEnvironment=environment());
    ssForecaster(ParentEnvironment=environment());

##### Do final check and make some preparations for output #####

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
        colnames(matvt) <- c(paste0("Component ",c(1:max(1,nComponents))),colnames(matat));
        if(updateX){
            rownames(vecgX) <- xregNames;
            dimnames(matFX) <- list(xregNames,xregNames);
        }
    }
    else{
        colnames(matvt) <- paste0("Component ",c(1:max(1,nComponents)));
    }
    if(constantRequired){
        colnames(matvt)[nComponents] <- "Constant";
    }

# AR terms
    if(any(ar.orders!=0)){
        ARterms <- matrix(0,max(ar.orders),sum(ar.orders!=0),
                          dimnames=list(paste0("AR(",c(1:max(ar.orders)),")"),
                                        paste0("Lag ",lags[ar.orders!=0])));
    }
    else{
        ARterms <- NULL;
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
        MAterms <- NULL;
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
        if(!is.null(xreg)){
            modelname <- "ARIMAX";
        }
        else{
            modelname <- "ARIMA";
        }
        modelname <- paste0(modelname,"(",ar.orders,",",i.orders,",",ma.orders,")");
    }
    else{
        modelname <- "";
        for(i in 1:length(ar.orders)){
            modelname <- paste0(modelname,"(",ar.orders[i],",");
            modelname <- paste0(modelname,i.orders[i],",");
            modelname <- paste0(modelname,ma.orders[i],")[",lags[i],"]");
        }
        if(!is.null(xreg)){
            modelname <- paste0("SARIMAX",modelname);
        }
        else{
            modelname <- paste0("SARIMA",modelname);
        }
    }
    if(all(intermittent!=c("n","none"))){
        modelname <- paste0("i",modelname);
    }

    if(constantRequired){
        if(constantEstimate){
            constantValue <- matvt[1,nComponents];
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
                  nParam=nParam,
                  fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,
                  errors=errors.mat,s2=s2,intervals=intervalsType,level=level,
                  actuals=data,holdout=y.holdout,iprob=pt,intermittent=intermittent,
                  xreg=xreg,updateX=updateX,initialX=initialX,persistenceX=vecgX,transitionX=matFX,
                  ICs=ICs,logLik=logLik,cf=cfObjective,cfType=cfType,FI=FI,accuracy=errormeasures);
    return(structure(model,class="smooth"));
}
