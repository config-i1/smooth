utils::globalVariables(c("estimate.initial","estimate.measurement","estimate.initial","estimate.transition",
                         "estimate.persistence","obs.vt","multisteps","ot","obs.ot","ICs","CF.objective",
                         "y.for","y.low","y.high","normalizer"));

ges <- function(data, orders=c(1,1), lags=c(1,frequency(data)), initial=c("optimal","backcasting"),
                persistence=NULL, transition=NULL, measurement=NULL,
                CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
                h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
                int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                intermittent=c("auto","none","fixed","croston","tsb"),
                bounds=c("admissible","none"), silent=c("none","all","graph","legend","output"),
                xreg=NULL, initialX=NULL, go.wild=FALSE, persistenceX=NULL, transitionX=NULL, ...){
# General Exponential Smoothing function. Crazy thing...
#
#    Copyright (C) 2016  Ivan Svetunkov

# Start measuring the time of calculations
    start.time <- Sys.time();

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

    # If a previous model provided as a model, write down the variables
    if(exists("model")){
        if(gregexpr("GES",model$model)!=1){
            stop("The provided model is not GES.",call.=FALSE);
        }
        intermittent <- model$intermittent;
        if(any(intermittent==c("p","provided"))){
            warning("The provided model had predefined values of occurences for the holdout. We don't have them.",call.=FALSE);
            warning("Switching to intermittent='auto'.",call.=FALSE);
            intermittent <- "a";
        }
        initial <- model$initial;
        persistence <- model$persistence;
        transition <- model$transition;
        measurement <- model$measurement;
        initialX <- model$initialX;
        persistenceX <- model$persistenceX;
        transitionX <- model$transitionX;
        if(any(persistenceX!=0)){
            go.wild <- TRUE;
        }
        model <- model$model;
        orders <- as.numeric(substring(model,unlist(gregexpr("\\[",model))-1,unlist(gregexpr("\\[",model))-1));
        lags <- as.numeric(substring(model,unlist(gregexpr("\\[",model))+1,unlist(gregexpr("\\]",model))-1));
    }

##### Set environment for ssinput and make all the checks #####
    environment(ssinput) <- environment();
    ssinput(modelType="ges",ParentEnvironment=environment());

##### Preset y.fit, y.for, errors and basic parameters #####
    matvt <- matrix(NA,nrow=obs.vt,ncol=n.components);
    y.fit <- rep(NA,obs);
    y.for <- rep(NA,h);
    errors <- rep(NA,obs);

##### Prepare exogenous variables #####
    xregdata <- ssxreg(data=data, xreg=xreg, go.wild=go.wild,
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

# These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

# Check number of parameters vs data
    n.param.max <- n.param.max + estimate.FX*length(matFX) + estimate.gX*nrow(vecgX) + estimate.initialX*ncol(matat);

##### Check number of observations vs number of max parameters #####
    if(obs.ot <= n.param.max){
        if(silent.text==FALSE){
            message(paste0("Number of non-zero observations is ",obs.ot,
                           ", while the number of parameters to estimate is ", n.param.max,"."));
        }
        stop("Can't fit the model you ask.",call.=FALSE);
    }

##### Initialise ges #####
elements.ges <- function(C){
    n.coef <- 0;
    if(estimate.measurement==TRUE){
        matw <- matrix(C[n.coef+(1:n.components)],1,n.components);
        n.coef <- n.coef + n.components;
    }
    else{
        matw <- matrix(measurement,1,n.components);
    }

    if(estimate.transition==TRUE){
        matF <- matrix(C[n.coef+(1:(n.components^2))],n.components,n.components);
        n.coef <- n.coef + n.components^2;
    }
    else{
        matF <- matrix(transition,n.components,n.components);
    }

    if(estimate.persistence==TRUE){
        vecg <- matrix(C[n.coef+(1:n.components)],n.components,1);
        n.coef <- n.coef + n.components;
    }
    else{
        vecg <- matrix(persistence,n.components,1);
    }

    if(fittertype=="o"){
        if(estimate.initial==TRUE){
            vtvalues <- C[n.coef+(1:(orders %*% lags))];
            n.coef <- n.coef + orders %*% lags;

        }
        else{
            vtvalues <- initial;
        }

        vt <- matrix(NA,maxlag,n.components);
        for(i in 1:n.components){
            vt[(maxlag - modellags + 1)[i]:maxlag,i] <- vtvalues[((cumsum(c(0,modellags))[i]+1):cumsum(c(0,modellags))[i+1])];
            vt[is.na(vt[1:maxlag,i]),i] <- rep(rev(vt[(maxlag - modellags + 1)[i]:maxlag,i]),
                                               ceiling((maxlag - modellags + 1) / modellags)[i])[is.na(vt[1:maxlag,i])];
        }
    }
    else{
        vt <- matvt[1:maxlag,n.components];
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

    return(list(matw=matw,matF=matF,vecg=vecg,vt=vt,at=at,matFX=matFX,vecgX=vecgX));
}

##### Cost Function for GES #####
CF <- function(C){
    elements <- elements.ges(C);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

    CF.res <- costfunc(matvt, matF, matw, y, vecg,
                       h, modellags, Etype, Ttype, Stype,
                       multisteps, CF.type, normalizer, fittertype,
                       matxt, matat, matFX, vecgX, ot,
                       bounds);

    return(CF.res);
}

##### Estimate ges or just use the provided values #####
gesCreator <- function(silent.text=FALSE,...){
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();

# 1 stands for the variance
    n.param <- n.components + n.components*(fittertype=="o") + n.components^2 + orders %*% lags +
        !is.null(xreg) * n.exovars + (go.wild==TRUE)*(n.exovars^2 + n.exovars) + 1;

# If there is something to optimise, let's do it.
    if((estimate.initial==TRUE) | (estimate.measurement==TRUE) | (estimate.transition==TRUE) | (estimate.persistence==TRUE) |
       (estimate.xreg==TRUE) | (estimate.FX==TRUE) | (estimate.gX==TRUE)){
# Initial values of matvt
        slope <- cov(yot[1:min(12,obs.ot),],c(1:min(12,obs.ot)))/var(c(1:min(12,obs.ot)));
        intercept <- sum(yot[1:min(12,obs.ot),])/min(12,obs.ot) - slope * (sum(c(1:min(12,obs.ot)))/min(12,obs.ot) - 1);

        C <- NULL;
# matw, matF, vecg, vt
        if(estimate.measurement==TRUE){
            C <- c(C,rep(1,n.components));
        }
        if(estimate.transition==TRUE){
            C <- c(C,rep(1,n.components^2));
        }
        if(estimate.persistence==TRUE){
            C <- c(C,rep(0.1,n.components));
        }
        if(estimate.initial==TRUE){
            if(fittertype=="o"){
                C <- c(C,intercept);
                if((orders %*% lags)>1){
                    C <- c(C,slope);
                }
                if((orders %*% lags)>2){
                    C <- c(C,yot[1:(orders %*% lags-2),]);
                }
            }
            else{
                vtvalues <- intercept;
                if((orders %*% lags)>1){
                    vtvalues <- c(vtvalues,slope);
                }
                if((orders %*% lags)>2){
                    vtvalues <- c(vtvalues,yot[1:(orders %*% lags-2),]);
                }

                vt <- matrix(NA,maxlag,n.components);
                for(i in 1:n.components){
                    vt[(maxlag - modellags + 1)[i]:maxlag,i] <- vtvalues[((cumsum(c(0,modellags))[i]+1):cumsum(c(0,modellags))[i+1])];
                    vt[is.na(vt[1:maxlag,i]),i] <- rep(rev(vt[(maxlag - modellags + 1)[i]:maxlag,i]),
                                                ceiling((maxlag - modellags + 1) / modellags)[i])[is.na(vt[1:maxlag,i])];
                }
                matvt[1:maxlag,] <- vt;
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

# Optimise model. First run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=5000));
        C <- res$solution;

# Optimise model. Second run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-10, "maxeval"=1000));
        C <- res$solution;
        CF.objective <- res$objective;
    }
    else{
# matw, matF, vecg, vt
        C <- c(measurement,
               c(transition),
               c(persistence),
               c(initial));

        C <- c(C,matat[maxlag,],
               c(transitionX),
               c(persistenceX));

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

##### Start the calculations #####
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

    gesValues <- gesCreator();

##### If intermittent=="a", run a loop and select the best one #####
    if(intermittent=="a"){
        if(silent.text==FALSE){
            cat("Selecting appropriate type of intermittency... ");
        }
# Prepare stuff for intermittency selection
        intermittentModelsPool <- c("n","f","c","t");
        intermittentICs <- rep(1e+10,length(intermittentModelsPool));
        intermittentModelsList <- list(NA);
        intermittentICs <- gesValues$bestIC;

        for(i in 2:length(intermittentModelsPool)){
            intermittentParametersSetter(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentMaker(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentModelsList[[i]] <- gesCreator(silent.text=TRUE);
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
            gesValues <- intermittentModelsList[[iBest]];
        }
        else{
            intermittent <- "n"
        }

        intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
        intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    }

    list2env(gesValues,environment());

# Prepare for fitting
    elements <- elements.ges(C);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

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
        initial <- C[2*n.components+n.components^2+(1:(orders %*% lags))];
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

# Make some preparations
    matvt <- ts(matvt,start=(time(data)[1] - deltat(data)*maxlag),frequency=frequency(data));
    if(!is.null(xreg)){
        matvt <- cbind(matvt,matat);
        colnames(matvt) <- c(paste0("Component ",c(1:n.components)),colnames(matat));
    }
    else{
        colnames(matvt) <- paste0("Component ",c(1:n.components));
    }

    if(holdout==T){
        y.holdout <- ts(data[(obs+1):obs.all],start=start(y.for),frequency=frequency(data));
        errormeasures <- errorMeasurer(y.holdout,y.for,y);
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

    modelname <- paste0("GES(",paste(orders,"[",lags,"]",collapse=",",sep=""),")");

##### Print output #####
    if(silent.text==FALSE){
        if(any(abs(eigen(matF - vecg %*% matw)$values)>(1 + 1E-10))){
            if(bounds!="a"){
                message("Unstable model was estimated! Use bounds='admissible' to address this issue!");
            }
            else{
                message("Something went wrong in optimiser - unstable model was estimated! Please report this error to the maintainer.");
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
        ssoutput(Sys.time() - start.time, modelname, persistence=vecg, transition=matF, measurement=matw,
                phi=NULL, ARterms=NULL, MAterms=NULL, const=NULL, A=NULL, B=NULL,
                n.components=orders %*% lags, s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
                CF.type=CF.type, CF.objective=CF.objective, intervals=intervals,
                int.type=int.type, int.w=int.w, ICs=ICs,
                holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures, intermittent=intermittent);
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
return(list(model=modelname,states=matvt,initial=initial,measurement=matw,transition=matF,persistence=vecg,
            fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,errors=errors.mat,
            actuals=data,holdout=y.holdout,iprob=pt,intermittent=intermittent,
            xreg=xreg,persistenceX=vecgX,transitionX=matFX,initialX=initialX,
            ICs=ICs,CF=CF.objective,CF.type=CF.type,FI=FI,accuracy=errormeasures));
}
