utils::globalVariables(c("silentText","silentGraph","silentLegend","initialType"));

ces <- function(data, seasonality=c("none","simple","partial","full"),
                initial=c("backcasting","optimal"), A=NULL, B=NULL,
                cfType=c("MSE","MAE","HAM","MLSTFE","MSTFE","MSEh"),
                h=10, holdout=FALSE,
                intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
                intermittent=c("none","auto","fixed","croston","tsb"),
                bounds=c("admissible","none"), silent=c("none","all","graph","legend","output"),
                xreg=NULL, initialX=NULL, updateX=FALSE, persistenceX=NULL, transitionX=NULL, ...){
# Function estimates CES in state-space form with sigma = error
#  and returns complex smoothing parameter value, fitted values,
#  residuals, point and interval forecasts, matrix of CES components and values of
#  information criteria.
#
#    Copyright (C) 2015 - 2016i  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

    # If a previous model provided as a model, write down the variables
    if(exists("model")){
        if(is.null(model$model)){
            stop("The provided model is not CES.",call.=FALSE);
        }
        else if(gregexpr("ES",model$model)==-1){
            stop("The provided model is not CES.",call.=FALSE);
        }
        intermittent <- model$intermittent;
        if(any(intermittent==c("p","provided"))){
            warning("The provided model had predefined values of occurences for the holdout. We don't have them.",call.=FALSE);
            warning("Switching to intermittent='auto'.",call.=FALSE);
            intermittent <- "a";
        }
        initial <- model$initial;
        A <- model$A;
        B <- model$B;
        if(is.null(xreg)){
            xreg <- model$xreg;
        }
        initialX <- model$initialX;
        persistenceX <- model$persistenceX;
        transitionX <- model$transitionX;
        if(any(c(persistenceX)!=0) | any((transitionX!=0)&(transitionX!=1))){
            updateX <- TRUE;
        }
        model <- model$model;
        seasonality <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
    }

##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- environment();
    ssInput(modelType="ces",ParentEnvironment=environment());

##### Preset y.fit, y.for, errors and basic parameters #####
    matvt <- matrix(NA,nrow=obsStates,ncol=n.components);
    y.fit <- rep(NA,obsInsample);
    y.for <- rep(NA,h);
    errors <- rep(NA,obsInsample);

##### Define parameters for different seasonality types #####
# Define "w" matrix, seasonal complex smoothing parameter, seasonality lag (if it is present).
#   matvt - the matrix with the components, lags is the lags used in pt matrix.
    if(seasonality=="n"){
# No seasonality
        matF <- matrix(1,2,2);
        vecg <- matrix(0,2);
        matw <- matrix(c(1,0),1,2);
        matvt <- matrix(NA,obsStates,2);
        colnames(matvt) <- c("level","potential");
        matvt[1,] <- c(mean(yot[1:min(10,obsNonzero)]),mean(yot[1:min(10,obsNonzero)])/1.1);
    }
    else if(seasonality=="s"){
# Simple seasonality, lagged CES
        matF <- matrix(1,2,2);
        vecg <- matrix(0,2);
        matw <- matrix(c(1,0),1,2);
        matvt <- matrix(NA,obsStates,2);
        colnames(matvt) <- c("level.s","potential.s");
        matvt[1:maxlag,1] <- y[1:maxlag];
        matvt[1:maxlag,2] <- matvt[1:maxlag,1]/1.1;
    }
    else if(seasonality=="p"){
# Partial seasonality with a real part only
        matF <- diag(3);
        matF[2,1] <- 1;
        vecg <- matrix(0,3);
        matw <- matrix(c(1,0,1),1,3);
        matvt <- matrix(NA,obsStates,3);
        colnames(matvt) <- c("level","potential","seasonal");
        matvt[1:maxlag,1] <- mean(y[1:maxlag]);
        matvt[1:maxlag,2] <- matvt[1:maxlag,1]/1.1;
        matvt[1:maxlag,3] <- decompose(ts(y,frequency=maxlag),type="additive")$figure;
    }
    else if(seasonality=="f"){
# Full seasonality with both real and imaginary parts
        matF <- diag(4);
        matF[2,1] <- 1;
        matF[4,3] <- 1;
        vecg <- matrix(0,4);
        matw <- matrix(c(1,0,1,0),1,4);
        matvt <- matrix(NA,obsStates,4);
        colnames(matvt) <- c("level","potential","seasonal 1", "seasonal 2");
        matvt[1:maxlag,1] <- mean(y[1:maxlag]);
        matvt[1:maxlag,2] <- matvt[1:maxlag,1]/1.1;
        matvt[1:maxlag,3] <- decompose(ts(y,frequency=maxlag),type="additive")$figure;
        matvt[1:maxlag,4] <- matvt[1:maxlag,3]/1.1;
    }

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
        if(!silentText){
            message(paste0("Number of non-zero observations is ",obsNonzero,
                           ", while the number of parameters to estimate is ", n.param.max,"."));
        }
        stop("Can't fit the model you ask.",call.=FALSE);
    }

##### Elements of CES #####
ElementsCES <- function(C){
    vt <- matrix(matvt[1:maxlag,],maxlag);
    n.coef <- 0;
    # No seasonality or Simple seasonality, lagged CES
    if(A$estimate){
        matF[1,2] <- C[2]-1;
        matF[2,2] <- 1-C[1];
        vecg[1:2,] <- c(C[1]-C[2],C[1]+C[2]);
        n.coef <- n.coef + 2;
    }
    else{
        matF[1,2] <- Im(A$value)-1;
        matF[2,2] <- 1-Re(A$value);
        vecg[1:2,] <- c(Re(A$value)-Im(A$value),Re(A$value)+Im(A$value));
    }

    if(seasonality=="p"){
    # Partial seasonality with a real part only
        if(B$estimate){
            vecg[3,] <- C[n.coef+1];
            n.coef <- n.coef + 1;
        }
        else{
            vecg[3,] <- B$value;
        }
    }
    else if(seasonality=="f"){
    # Full seasonality with both real and imaginary parts
        if(B$estimate){
            matF[3,4] <- C[n.coef+2]-1;
            matF[4,4] <- 1-C[n.coef+1];
            vecg[3:4,] <- c(C[n.coef+1]-C[n.coef+2],C[n.coef+1]+C[n.coef+2]);
            n.coef <- n.coef + 2;
        }
        else{
            matF[3,4] <- Im(B$value)-1;
            matF[4,4] <- 1-Re(B$value);
            vecg[3:4,] <- c(Re(B$value)-Im(B$value),Re(B$value)+Im(B$value));
        }
    }

    if(initialType=="o"){
        if(any(seasonality==c("n","s"))){
            vt[1:maxlag,] <- C[n.coef+(1:(2*maxlag))];
            n.coef <- n.coef + maxlag*2;
        }
        else if(seasonality=="p"){
            vt[,1:2] <- rep(C[n.coef+(1:2)],each=maxlag);
            n.coef <- n.coef + 2;
            vt[1:maxlag,3] <- C[n.coef+(1:maxlag)];
            n.coef <- n.coef + maxlag;
        }
        else if(seasonality=="f"){
            vt[,1:2] <- rep(C[n.coef+(1:2)],each=maxlag);
            n.coef <- n.coef + 2;
            vt[1:maxlag,3:4] <- C[n.coef+(1:(maxlag*2))];
            n.coef <- n.coef + maxlag*2;
        }
    }
    else if(initialType=="b"){
        vt[1:maxlag,] <- matvt[1:maxlag,];
    }
    else{
        vt[1:maxlag,] <- initialValue;
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
        if(updateX){
            if(FXEstimate){
                matFX <- matrix(C[n.coef+(1:(n.exovars^2))],n.exovars,n.exovars);
                n.coef <- n.coef + n.exovars^2;
            }

            if(gXEstimate){
                vecgX <- matrix(C[n.coef+(1:n.exovars)],n.exovars,1);
                n.coef <- n.coef + n.exovars;
            }
        }
    }
    else{
        at <- matrix(0,maxlag,n.exovars);
    }

    return(list(matF=matF,vecg=vecg,vt=vt,at=at,matFX=matFX,vecgX=vecgX));
}

##### Cost function for CES #####
CF <- function(C){
# Obtain the elements of CES
    elements <- ElementsCES(C);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

    cfRes <- costfunc(matvt, matF, matw, y, vecg,
                       h, modellags, Etype, Ttype, Stype,
                       multisteps, cfType, normalizer, initialType,
                       matxt, matat, matFX, vecgX, ot,
                       bounds);

    if(is.nan(cfRes) | is.na(cfRes)){
        cfRes <- 1e100;
    }
    return(cfRes);
}

##### Estimate ces or just use the provided values #####
CreatorCES <- function(silentText=FALSE,...){
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();

    n.param <- sum(modellags)*(initialType!="b") + A$number + B$number + (!is.null(xreg))*(ncol(matat) + updateX*(length(matFX) + nrow(vecgX))) + 1;

    if(any(initialType=="o",A$estimate,B$estimate,initialXEstimate,FXEstimate,gXEstimate)){
        C <- NULL;
        # If we don't need to estimate A
        if(A$estimate){
            C <- c(1.3,1);
        }

        if(any(seasonality==c("n","s"))){
            if(initialType=="o"){
                C <- c(C,c(matvt[1:maxlag,]));
            }
        }
        else if(seasonality=="p"){
            if(B$estimate){
                C <- c(C,0.1);
            }
            if(initialType=="o"){
                C <- c(C,c(matvt[1,1:2]));
                C <- c(C,c(matvt[1:maxlag,3]));
            }
        }
        else{
            if(B$estimate){
                C <- c(C,1.3,1);
            }
            if(initialType=="o"){
                C <- c(C,c(matvt[1,1:2]));
                C <- c(C,c(matvt[1:maxlag,3:4]));
            }
        }

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

        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
        C <- res$solution;

        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-8, "maxeval"=1000));
        C <- res$solution;
        cfObjective <- res$objective;
    }
    else{
        C <- c(A$value,B$value,initialValue,initialX,transitionX,persistenceX);
        cfObjective <- CF(C);
    }
    if(multisteps){
        cfType <- "aTFL";
    }
    else{
        cfType <- "MSE";
    }
    IC.values <- ICFunction(n.param=n.param+n.param.intermittent,C=C,Etype=Etype);
    ICs <- IC.values$ICs;
    bestIC <- ICs["AICc"];
    # Change back
    cfType <- cfTypeOriginal;

    return(list(cfObjective=cfObjective,C=C,ICs=ICs,bestIC=bestIC,n.param=n.param));
}

# Information criterion derived and used especially for CES
#   k here is equal to number of coefficients/2 (number of numbers) + number of complex initial states of CES.
#    CIC.coef <- 2 * (ceiling(length(C)/2) + maxlag) * h ^ multisteps - 2 * llikelihood;
#    ICs <- c(AIC.coef, AICc.coef, BIC.coef,CIC.coef);
#    names(ICs) <- c("AIC", "AICc", "BIC","CIC");

##### Start doing things #####
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

    cesValues <- CreatorCES(silentText=silentText);

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
        intermittentICs <- cesValues$bestIC;

        for(i in 2:length(intermittentModelsPool)){
            intermittentParametersSetter(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentMaker(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentModelsList[[i]] <- CreatorCES(silentText=TRUE);
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
            cesValues <- intermittentModelsList[[iBest]];
        }
        else{
            intermittent <- "n"
        }

        intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
        intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    }

    list2env(cesValues,environment());

# Prepare for fitting
    elements <- ElementsCES(C);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

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
        message("Something went wrong during the optimisation and NAs were produced!");
        message("Please check the input and report this error if it persists to the maintainer.");
    }

# Write down initials of states vector and exogenous
    if(initialType!="p"){
        initialValue <- matvt[1:maxlag,];
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

    if(!is.null(xreg)){
        statenames <- c(colnames(matvt),colnames(matat));
        matvt <- cbind(matvt,matat);
        colnames(matvt) <- statenames;
        if(updateX){
            rownames(vecgX) <- xregNames;
            dimnames(matFX) <- list(xregNames,xregNames);
        }
    }

# Right down the smoothing parameters
    n.coef <- 0;
    if(A$estimate){
        A$value <- complex(real=C[1],imaginary=C[2]);
        n.coef <- 2;
    }

    names(A$value) <- "a0+ia1";

    if(B$estimate){
        if(seasonality=="p"){
            B$value <- C[n.coef+1];
        }
        else if(seasonality=="f"){
            B$value <- complex(real=C[n.coef+1],imaginary=C[n.coef+2]);
        }
    }
    if(B$number!=0){
        if(is.complex(B$value)){
            names(B$value) <- "b0+ib1";
        }
        else{
            names(B$value) <- "b";
        }
    }

    modelname <- paste0("CES(",seasonality,")");

    if(all(intermittent!=c("n","none"))){
        modelname <- paste0("i",modelname);
    }

    if(holdout){
        y.holdout <- ts(data[(obsInsample+1):obsAll],start=start(y.for),frequency=datafreq);
        errormeasures <- errorMeasurer(y.holdout,y.for,y);
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

##### Print output #####
    if(!silentText){
        if(any(abs(eigen(matF - vecg %*% matw)$values)>(1 + 1E-10))){
            if(bounds!="a"){
                warning("Unstable model was estimated! Use bounds='admissible' to address this issue!",call.=FALSE);
            }
            else{
                warning("Something went wrong in optimiser - unstable model was estimated! Please report this error to the maintainer.",
                        call.=FALSE);
            }
        }
    }

# Make plot
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

    model <- list(model=modelname,timeElapsed=Sys.time()-startTime,
                  states=matvt,A=A$value,B=B$value,
                  initialType=initialType,initial=initialValue,
                  nParam=n.param,
                  fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,
                  errors=errors.mat,s2=s2,intervals=intervalsType,level=level,
                  actuals=data,holdout=y.holdout,iprob=pt,intermittent=intermittent,
                  xreg=xreg,updateX=updateX,initialX=initialX,persistenceX=vecgX,transitionX=matFX,
                  ICs=ICs,cf=cfObjective,cfType=cfType,FI=FI,accuracy=errormeasures);
    return(structure(model,class="smooth"));
}
