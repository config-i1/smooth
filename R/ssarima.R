utils::globalVariables(c("normalizer"));

ssarima <- function(data, ar.orders=c(0), i.orders=c(1), ma.orders=c(1), lags=c(1),
                    constant=FALSE, initial=c("backcasting","optimal"), AR=NULL, MA=NULL,
                    CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
                    h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
                    int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                    intermittent=c("auto","none","fixed","croston","tsb"),
                    bounds=c("admissible","none"), silent=c("none","all","graph","legend","output"),
                    xreg=NULL, initialX=NULL, go.wild=FALSE, persistenceX=NULL, transitionX=NULL, ...){
##### Function constructs SARIMA model (possible triple seasonality) using state-space approach
# ar.orders contains vector of seasonal ARs. ar.orders=c(2,1,3) will mean AR(2)*SAR(1)*SAR(3) - model with double seasonality.
#
#    Copyright (C) 2016  Ivan Svetunkov

# Start measuring the time of calculations
    start.time <- Sys.time();

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
        persistence <- model$persistence;
        transition <- model$transition;
        initialX <- model$initialX;
        persistenceX <- model$persistenceX;
        transitionX <- model$transitionX;
        if(any(c(persistenceX,transitionX)!=0)){
            go.wild <- TRUE;
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
        matvt <- matrix(NA,obs.vt,n.components);
        if(constant$required==TRUE){
            matF <- cbind(rbind(matF,rep(0,n.components)),c(1,rep(0,n.components-1),1));
            matw <- cbind(matw,0);
            vecg <- rbind(vecg,0);
            matvt <- cbind(matvt,rep(1,obs.vt));
        }
    }
    else{
        matw <- matF <- matrix(1,1,1);
        vecg <- matrix(0,1,1);
        matvt <- matrix(1,obs.vt,1);
        modellags <- matrix(1,1,1);
    }

##### Preset y.fit, y.for, errors and basic parameters #####
    y.fit <- rep(NA,obs);
    y.for <- rep(NA,h);
    errors <- rep(NA,obs);

##### Prepare exogenous variables #####
    xregdata <- ssXreg(data=data, xreg=xreg, go.wild=go.wild,
                       persistenceX=persistenceX, transitionX=transitionX, initialX=initialX,
                       obs=obs, obs.all=obs.all, obs.vt=obs.vt, maxlag=maxlag, h=h, silent=silent.text);
    n.exovars <- xregdata$n.exovars;
    matxt <- xregdata$matxt;
    matat <- xregdata$matat;
    matFX <- xregdata$matFX;
    vecgX <- xregdata$vecgX;
    estimate.xreg <- xregdata$estimate.xreg;
    estimate.FX <- xregdata$estimate.FX;
    estimate.gX <- xregdata$estimate.gX;
    estimate.initialX <- xregdata$estimate.initialX;

    # Check number of parameters vs data
    n.param.max <- n.param.max + estimate.FX*length(matFX) + estimate.gX*nrow(vecgX) + estimate.initialX*ncol(matat);

    if(obs.ot <= n.param.max){
        stop(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                        n.param.max," while the number of observations is ",obs.ot,"!"),call.=FALSE);
    }

# These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

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
                if(AR$estimate==TRUE){
                    armat[lags[i],] <- -C[n.coef+(1:ar.orders[i])];
                    n.coef <- n.coef + ar.orders[i];
                }
                else{
                    armat[lags[i],] <- -AR$value[ar.inner.coef+(1:ar.orders[i])];
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
                if(MA$estimate==TRUE){
                    armat[lags[i],] <- C[n.coef+(1:ma.orders[i])];
                    n.coef <- n.coef + ma.orders[i];
                }
                else{
                    armat[lags[i],] <- MA$value[ma.inner.coef+(1:ma.orders[i])];
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

        if(estimate.initial==TRUE){
            vt <- rep(NA,n.components+constant$required);
            if(fittertype=="o"){
                vt <- C[(n.coef + 1):(n.coef + n.components)];
                n.coef <- n.coef + n.components;
            }
            else{
                slope <- cov(yot[1:min(12,obs.ot),],c(1:min(12,obs.ot)))/var(c(1:min(12,obs.ot)));
                intercept <- sum(yot[1:min(12,obs.ot),])/min(12,obs.ot) - slope * (sum(c(1:min(12,obs.ot)))/min(12,obs.ot) - 1);
                initial.stuff <- c(intercept,-intercept,rep(slope,n.components));
                vt[1:n.components] <- initial.stuff[1:n.components];
                vt[-1] <- vt[1] * matF[-1,1];
            }
        }
        else{
            vt <- initial;
        }

        if(constant$required==TRUE){
            if(constant$estimate==TRUE){
                vt[n.components+constant$required] <- C[(n.coef + 1)];
                n.coef <- n.coef + 1;
            }
            else{
                vt[n.components+constant$required] <- constant$value;
            }
        }
    }
    else{
        matF[1,1] <- 1;
        if(constant$estimate==TRUE){
            vt <- C[n.coef+1];
            n.coef <- n.coef + 1;
        }
        else{
            vt <- constant$value;
        }
    }

# If exogenous are included
    if(estimate.xreg==TRUE){
        at <- matrix(NA,maxlag,n.exovars);
        if(estimate.initialX==TRUE){
            at[,] <- rep(C[n.coef+(1:n.exovars)],each=maxlag);
            n.coef <- n.coef + n.exovars;
        }
        else{
            at <- matat[1:maxlag,];
        }
        if(estimate.FX==TRUE){
            matFX <- matrix(C[n.coef+(1:(n.exovars^2))],n.exovars,n.exovars);
            n.coef <- n.coef + n.exovars^2;
        }

        if(estimate.gX==TRUE){
            vecgX <- matrix(C[n.coef+(1:n.exovars)],n.exovars,1);
            n.coef <- n.coef + n.exovars;
        }
    }
    else{
        at <- matrix(0,maxlag,n.exovars);
    }

    return(list(matF=matF,vecg=vecg,vt=vt,at=at,matFX=matFX,vecgX=vecgX,polysos.ar=polysos.ar,polysos.ma=polysos.ma));
}

# Cost function for GES
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

    CF.res <- optimizerwrap(matvt, matF, matw, y, vecg,
                            h, modellags, Etype, Ttype, Stype,
                            multisteps, CF.type, normalizer, fittertype,
                            matxt, matat, matFX, vecgX, ot);
    if(is.nan(CF.res) | is.na(CF.res) | is.infinite(CF.res)){
        CF.res <- 1e+100;
    }

    return(CF.res);
}

##### Estimate ssarima or just use the provided values #####
ssarimaCreator <- function(silent.text=FALSE,...){
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();

    n.param <- 1 + n.components*estimate.initial*(fittertype=="o") + sum(ar.orders)*AR$required +
        sum(ma.orders)*MA$required + constant$required + estimate.FX*length(matFX) +
        estimate.gX*nrow(vecgX) + estimate.initialX*ncol(matat);

    # If there is something to optimise, let's do it.
    if(((estimate.initial==TRUE) & fittertype=="o") | (AR$estimate==TRUE) | (MA$estimate==TRUE) |
       (estimate.xreg==TRUE) | (estimate.FX==TRUE) | (estimate.gX==TRUE) | (constant$estimate==TRUE) ){

        C <- NULL;
        if(n.components > 0){
# ar terms, ma terms from season to season...
            if(AR$estimate==TRUE){
                C <- c(C,rep(0.1,sum(ar.orders)));
            }
            if(MA$estimate==TRUE){
                C <- c(C,rep(0.1,sum(ma.orders)));
            }

# initial values of state vector and the constant term
            if(estimate.initial==TRUE){
                slope <- cov(yot[1:min(12,obs.ot),],c(1:min(12,obs.ot)))/var(c(1:min(12,obs.ot)));
                intercept <- sum(yot[1:min(12,obs.ot),])/min(12,obs.ot) - slope * (sum(c(1:min(12,obs.ot)))/min(12,obs.ot) - 1);
                if(fittertype=="o"){
                    initial.stuff <- c(rep(intercept,n.components));
                    C <- c(C,initial.stuff[1:n.components]);
                }
            }
        }

        if(constant$estimate==TRUE){
            if(all(i.orders==0)){
                C <- c(C,sum(yot)/obs);
            }
            else{
                C <- c(C,sum(diff(yot))/obs);
            }
        }

# initials, transition matrix and persistence vector
        if(estimate.xreg==TRUE){
            if(estimate.initialX==TRUE){
                C <- c(C,matat[maxlag,]);
            }
            if(go.wild==TRUE){
                if(estimate.FX==TRUE){
                    C <- c(C,c(diag(n.exovars)));
                }
                if(estimate.gX==TRUE){
                    C <- c(C,rep(0,n.exovars));
                }
            }
        }

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

# Optimise model. First run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
        C <- res$solution;
        if(fittertype=="o"){
# Optimise model. Second run
            res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-10, "maxeval"=1000));
            C <- res$solution;
        }
        CF.objective <- res$objective;
    }
    else{
        C <- NULL;
# initial values of state vector and the constant term
        slope <- cov(yot[1:min(12,obs.ot),],c(1:min(12,obs.ot)))/var(c(1:min(12,obs.ot)));
        intercept <- sum(yot[1:min(12,obs.ot),])/min(12,obs.ot) - slope * (sum(c(1:min(12,obs.ot)))/min(12,obs.ot) - 1);
        initial.stuff <- c(intercept,-intercept,rep(slope,n.components));
        matvt[1,1:n.components] <- initial.stuff[1:(n.components)];

        CF.objective <- CF(C);
    }

# Change CF.type for model selection
    if(multisteps==TRUE){
        if(substring(CF.type,1,1)=="a"){
            CF.type <- "aTFL";
        }
        else{
            CF.type <- "TFL";
        }
    }
    else{
        CF.type <- "MSE";
    }

    IC.values <- ICFunction(n.param=n.param+n.param.intermittent,C=C,Etype=Etype);
    ICs <- IC.values$ICs;
    bestIC <- ICs["AICc"];

# Revert to the provided cost function
    CF.type <- CF.type.original

    return(list(CF.objective=CF.objective,C=C,ICs=ICs,bestIC=bestIC,n.param=n.param));
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

    ssarimaValues <- ssarimaCreator();

##### If intermittent=="a", run a loop and select the best one #####
    if(intermittent=="a"){
        if(silent.text==FALSE){
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
            intermittentModelsList[[i]] <- ssarimaCreator(silent.text=TRUE);
            intermittentICs[i] <- intermittentModelsList[[i]]$bestIC;
            if(intermittentICs[i]>intermittentICs[i-1]){
                break;
            }
        }
        intermittentICs[is.nan(intermittentICs)] <- 1e+100;
        intermittentICs[is.na(intermittentICs)] <- 1e+100;
        iBest <- which(intermittentICs==min(intermittentICs));

        if(silent.text==FALSE){
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
    n.components <- n.components + constant$required;

    # Write down Fisher Information if needed
    if(FI==TRUE){
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
    if(estimate.initial==TRUE){
        if((n.components-constant$required)>0){
            initial <- matvt[1,1:(n.components-constant$required)];
        }
        else{
            initial <- NULL;
        }
    }
    if(estimate.initialX==TRUE){
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
    }
    else{
        colnames(matvt) <- paste0("Component ",c(1:max(1,n.components)));
    }
    if(constant$required==TRUE){
        colnames(matvt)[n.components] <- "Constant";
    }

# AR terms
    if(any(ar.orders!=0)){
        ARterms <- matrix(0,max(ar.orders),sum(ar.orders!=0),
                          dimnames=list(paste0("AR(",c(1:max(ar.orders)),")"),
                                        paste0("Lag ",lags[ar.orders!=0])));
    }
    else{
        ARterms <- 0;
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
        MAterms <- 0;
    }

    n.coef <- ar.coef <- ma.coef <- 0;
    ar.i <- ma.i <- 1;
    for(i in 1:length(ar.orders)){
        if(ar.orders[i]!=0){
            if(AR$estimate==TRUE){
                ARterms[1:ar.orders[i],ar.i] <- C[n.coef+(1:ar.orders[i])];
                n.coef <- n.coef + ar.orders[i];
            }
            else{
                ARterms[1:ar.orders[i],ar.i] <- AR$value[ar.coef+(1:ar.orders[i])];
                ar.coef <- ar.coef + ar.orders[i];
            }
            ar.i <- ar.i + 1;
        }
        if(ma.orders[i]!=0){
            if(MA$estimate==TRUE){
                MAterms[1:ma.orders[i],ma.i] <- C[n.coef+(1:ma.orders[i])];
                n.coef <- n.coef + ma.orders[i];
            }
            else{
                MAterms[1:ma.orders[i],ma.i] <- MA$value[ma.coef+(1:ma.orders[i])];
                ma.coef <- ma.coef + ma.orders[i];
            }
            ma.i <- ma.i + 1;
        }
    }

    if(holdout==T){
        y.holdout <- ts(data[(obs+1):obs.all],start=start(y.for),frequency=frequency(data));
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

    if(constant$required==TRUE){
        if(constant$estimate==TRUE){
            constant$value <- C[length(C)];
        }
        const <- constant$value;

        if(all(i.orders==0)){
            modelname <- paste0(modelname," with constant");
        }
        else{
            modelname <- paste0(modelname," with drift");
        }
    }
    else{
        const <- FALSE;
        constant$value <- NULL;
    }

##### Print output #####
    if(silent.text==FALSE){
        if(any(maroots<1)){
            if(bounds!="a"){
                message("Unstable model was estimated! Use bounds='admissible' to address this issue!");
            }
            else{
                message("Something went wrong in optimiser - unstable model was estimated! Please report this error to the maintainer.");
            }
        }
        if(any(arroots<1)){
            if(bounds!="a"){
                message("Non-stationary model was estimated! Beware of explosions! Use bounds='admissible' to address this issue!");
            }
            else{
                message("Something went wrong in optimiser - non-stationary model was estimated! Please report this error to the maintainer.");
            }
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
        ssOutput(Sys.time() - start.time, modelname, persistence=NULL, transition=NULL, measurement=NULL,
            phi=NULL, ARterms=ARterms, MAterms=MAterms, const=constant$value, A=NULL, B=NULL,
            n.components=(n.components-constant$required), s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
            CF.type=CF.type, CF.objective=CF.objective, intervals=intervals,
            int.type=int.type, int.w=int.w, ICs=ICs,
            holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures,intermittent=intermittent);
    }

##### Make a plot #####
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
##### Return values #####
return(list(model=modelname,states=matvt,initial=initial,transition=matF,persistence=vecg,
            AR=ARterms,I=Iterms,MA=MAterms,constant=const,
            fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,errors=errors.mat,
            actuals=data,holdout=y.holdout,iprob=pt,intermittent=intermittent,
            xreg=xreg,initialX=initialX,persistenceX=vecgX,transitionX=matFX,
            ICs=ICs,CF=CF.objective,CF.type=CF.type,FI=FI,accuracy=errormeasures));
}
