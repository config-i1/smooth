utils::globalVariables(c("vecg","n.components","modellags","fittertype","estimate.phi","y","datafreq",
                         "obs","obs.all","yot","maxlag","silent.text","allowMultiplicative","current.model",
                         "n.param.intermittent","CF.type.original","matF","matw","pt.for","errors.mat",
                         "iprob","results","s2","silent.graph","FI","intermittent","normalizer",
                         "estimate.persistence","estimate.initial","obs.vt","multisteps","ot","obs.ot"));

es <- function(data, model="ZZZ", persistence=NULL, phi=NULL,
               initial=c("optimal","backcasting"), initial.season=NULL, IC=c("AICc","AIC","BIC"),
               CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
               h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
               int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
               intermittent=c("auto","none","fixed","croston","tsb"),
               bounds=c("usual","admissible","none"),
               silent=c("none","all","graph","legend","output"),
               xreg=NULL, initialX=NULL, go.wild=FALSE, persistenceX=NULL, transitionX=NULL, ...){
# Copyright (C) 2015 - 2016  Ivan Svetunkov

# Start measuring the time of calculations
    start.time <- Sys.time();

# If a previous model provided as a model, write down the variables
    if(is.list(model)){
        if(gregexpr("ETS",model$model)!=1){
            stop("The provided model is not ETS.",call.=FALSE);
        }
        intermittent <- model$intermittent;
        if(any(intermittent==c("p","provided"))){
            warning("The provided model had predefined values of occurences for the holdout. We don't have them.",call.=FALSE);
            warning("Switching to intermittent='auto'.",call.=FALSE);
            intermittent <- "a";
        }
        persistence <- model$persistence;
        initial <- model$initial;
        initial.season <- model$initial.season;
        initialX <- model$initialX;
        persistenceX <- model$persistenceX;
        transitionX <- model$transitionX;
        if(any(persistenceX!=0)){
            go.wild <- TRUE;
        }
        model <- model$model;
        model <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
    }

# Add all the variables in ellipsis to current environment
    list2env(list(...),environment());

##### Set environment for ssinput and make all the checks #####
    environment(ssinput) <- environment();
    ssinput(modelType="es",ParentEnvironment=environment());

##### Cost Function for ES #####
CF <- function(C){
    init.ets <- etsmatrices(matvt, vecg, phi, matrix(C,nrow=1), n.components,
                            modellags, fittertype, Ttype, Stype, n.exovars, matat,
                            estimate.persistence, estimate.phi, estimate.initial, estimate.initial.season, estimate.xreg,
                            matFX, vecgX, go.wild, estimate.FX, estimate.gX, estimate.initialX);

    CF.res <- costfunc(init.ets$matvt, init.ets$matF, init.ets$matw, y, init.ets$vecg,
                       h, modellags, Etype, Ttype, Stype,
                       multisteps, CF.type, normalizer, fittertype,
                       matxt, init.ets$matat, init.ets$matFX, init.ets$vecgX, ot,
                       bounds);

    if(is.nan(CF.res) | is.na(CF.res) | is.infinite(CF.res)){
        CF.res <- 1e+100;
    }

    return(CF.res);
}

##### C values for estimation #####
# Function constructs default bounds where C values should lie
C.values <- function(bounds,Ttype,Stype,vecg,matvt,phi,maxlag,n.components,matat){
    C <- NA;
    C.lower <- NA;
    C.upper <- NA;

    if(bounds=="u"){
        if(estimate.persistence==TRUE){
            C <- c(C,vecg);
            C.lower <- c(C.lower,rep(0,length(vecg)));
            C.upper <- c(C.upper,rep(1,length(vecg)));
        }
        if(estimate.phi==TRUE){
            C <- c(C,phi);
            C.lower <- c(C.lower,0);
            C.upper <- c(C.upper,1);
        }
        if(fittertype=="o"){
            if(estimate.initial==TRUE){
                C <- c(C,matvt[maxlag,1:(n.components - (Stype!="N"))]);
                if(Ttype!="M"){
                    C.lower <- c(C.lower,rep(-Inf,(n.components - (Stype!="N"))));
                    C.upper <- c(C.upper,rep(Inf,(n.components - (Stype!="N"))));
                }
                else{
                    C.lower <- c(C.lower,0.1,0.01);
                    C.upper <- c(C.upper,Inf,5);
                }
            }
            if(Stype!="N"){
                if(estimate.initial.season==TRUE){
                    C <- c(C,matvt[1:maxlag,n.components]);
                    if(Stype=="A"){
                        C.lower <- c(C.lower,rep(-Inf,maxlag));
                        C.upper <- c(C.upper,rep(Inf,maxlag));
                    }
                    else{
                        C.lower <- c(C.lower,rep(1e-5,maxlag));
                        C.upper <- c(C.upper,rep(10,maxlag));
                    }
                }
            }
        }
    }
    else if(bounds=="a"){
        if(estimate.persistence==TRUE){
            C <- c(C,vecg);
            C.lower <- c(C.lower,rep(-5,length(vecg)));
            C.upper <- c(C.upper,rep(5,length(vecg)));
        }
        if(estimate.phi==TRUE){
            C <- c(C,phi);
            C.lower <- c(C.lower,0);
            C.upper <- c(C.upper,1);
        }
        if(fittertype=="o"){
            if(estimate.initial==TRUE){
                C <- c(C,matvt[maxlag,1:(n.components - (Stype!="N"))]);
                if(Ttype!="M"){
                    C.lower <- c(C.lower,rep(-Inf,(n.components - (Stype!="N"))));
                    C.upper <- c(C.upper,rep(Inf,(n.components - (Stype!="N"))));
                }
                else{
                    C.lower <- c(C.lower,0.1,0.01);
                    C.upper <- c(C.upper,Inf,3);
                }
            }
            if(Stype!="N"){
                if(estimate.initial.season==TRUE){
                    C <- c(C,matvt[1:maxlag,n.components]);
                    if(Stype=="A"){
                        C.lower <- c(C.lower,rep(-Inf,maxlag));
                        C.upper <- c(C.upper,rep(Inf,maxlag));
                    }
                    else{
                        C.lower <- c(C.lower,rep(-0.0001,maxlag));
                        C.upper <- c(C.upper,rep(20,maxlag));
                    }
                }
            }
        }
    }
    else{
        if(estimate.persistence==TRUE){
            C <- c(C,vecg);
            C.lower <- c(C.lower,rep(-Inf,length(vecg)));
            C.upper <- c(C.upper,rep(Inf,length(vecg)));
        }
        if(estimate.phi==TRUE){
            C <- c(C,phi);
            C.lower <- c(C.lower,-Inf);
            C.upper <- c(C.upper,Inf);
        }
        if(fittertype=="o"){
            if(estimate.initial==TRUE){
                C <- c(C,matvt[maxlag,1:(n.components - (Stype!="N"))]);
                if(Ttype!="M"){
                    C.lower <- c(C.lower,rep(-Inf,(n.components - (Stype!="N"))));
                    C.upper <- c(C.upper,rep(Inf,(n.components - (Stype!="N"))));
                }
                else{
                    C.lower <- c(C.lower,-Inf,-Inf);
                    C.upper <- c(C.upper,Inf,Inf);
                }
            }
            if(Stype!="N"){
                if(estimate.initial.season==TRUE){
                    C <- c(C,matvt[1:maxlag,n.components]);
                    if(Stype=="A"){
                        C.lower <- c(C.lower,rep(-Inf,maxlag));
                        C.upper <- c(C.upper,rep(Inf,maxlag));
                    }
                    else{
                        C.lower <- c(C.lower,rep(-Inf,maxlag));
                        C.upper <- c(C.upper,rep(Inf,maxlag));
                    }
                }
            }
        }
    }

    if(estimate.xreg==TRUE){
        if(estimate.initialX==TRUE){
            C <- c(C,matat[maxlag,]);
            C.lower <- c(C.lower,rep(-Inf,n.exovars));
            C.upper <- c(C.upper,rep(Inf,n.exovars));
        }
        if(go.wild==TRUE){
            if(estimate.FX==TRUE){
                C <- c(C,as.vector(matFX));
                C.lower <- c(C.lower,rep(-Inf,n.exovars^2));
                C.upper <- c(C.upper,rep(Inf,n.exovars^2));
            }
            if(estimate.gX==TRUE){
                C <- c(C,as.vector(vecgX));
                C.lower <- c(C.lower,rep(-Inf,n.exovars));
                C.upper <- c(C.upper,rep(Inf,n.exovars));
            }
        }
    }

    C <- C[!is.na(C)];
    C.lower <- C.lower[!is.na(C.lower)];
    C.upper <- C.upper[!is.na(C.upper)];

    return(list(C=C,C.lower=C.lower,C.upper=C.upper));
}

##### Basic parameter propagator #####
esBasicMaker <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    basicparams <- initparams(Ttype, Stype, datafreq, obs, obs.all, y,
                              damped, phi, smoothingparameters, initialstates, seasonalcoefs);
    list2env(basicparams,ParentEnvironment);
}

##### Basic parameter propagator #####
esBasicInitialiser <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    init.ets <- etsmatrices(matvt, vecg, phi, matrix(C,nrow=1), n.components,
                            modellags, fittertype, Ttype, Stype, n.exovars, matat,
                            estimate.persistence, estimate.phi, estimate.initial, estimate.initial.season, estimate.xreg,
                            matFX, vecgX, go.wild, estimate.FX, estimate.gX, estimate.initialX);

    list2env(init.ets,ParentEnvironment);
}

##### Set initialstates, initialsesons and persistence vector #####
# If initial values are provided, write them. If not, estimate them.
# First two columns are needed for additive seasonality, the 3rd and 4th - for the multiplicative
    if(Ttype!="N"){
        if(is.null(initial)){
            initialstates <- matrix(NA,1,4);
# "-1" is needed, so the level would correspond to the values before the in-sample
            initialstates[1,2] <- cov(yot[1:min(12,obs.ot)],c(1:min(12,obs.ot)))/var(c(1:min(12,obs.ot)));
            initialstates[1,1] <- mean(yot[1:min(12,obs.ot)]) - initialstates[1,2] * (mean(c(1:min(12,obs.ot))) - 1);
            initialstates[1,3] <- mean(yot[1:min(12,obs.ot)]);
            initialstates[1,4] <- 1;
        }
        else{
            initialstates <- matrix(rep(initial,2),nrow=1);
        }
    }
    else{
        if(is.null(initial)){
            initialstates <- matrix(rep(mean(yot[1:min(12,obs.ot)]),4),nrow=1);
        }
        else{
            initialstates <- matrix(rep(initial,4),nrow=1);
        }
    }

# Define matrix of seasonal coefficients. The first column consists of additive, the second - multiplicative elements
# If the seasonal model is chosen and initials are provided, fill in the first "maxlag" values of seasonal component.
    if(Stype!="N"){
        if(is.null(initial.season)){
            estimate.initial.season <- TRUE;
            seasonalcoefs <- decompose(ts(c(y),frequency=datafreq),type="additive")$seasonal[1:datafreq];
            seasonalcoefs <- cbind(seasonalcoefs,decompose(ts(c(y),frequency=datafreq),
                                                           type="multiplicative")$seasonal[1:datafreq]);
        }
        else{
            estimate.initial.season <- FALSE;
            seasonalcoefs <- cbind(initial.season,initial.season);
        }
    }
    else{
        estimate.initial.season <- FALSE;
        seasonalcoefs <- matrix(1,1,1);
    }

# If the persistence vector is provided, use it
    if(!is.null(persistence)){
        smoothingparameters <- cbind(persistence,persistence);
    }
    else{
        smoothingparameters <- cbind(c(0.2,0.1,0.05),rep(0.05,3));
    }

    if(!is.null(phi)){
        if(phi<0 | phi>2){
            message("Damping parameter should lie in (0, 2) region.");
            message("Changing to the estimation of phi.");
            phi <- NULL;
        }
    }

##### Preset y.fit, y.for, errors and basic parameters #####
    y.fit <- rep(NA,obs);
    y.for <- rep(NA,h);
    errors <- rep(NA,obs);

    esBasicMaker(ParentEnvironment=environment());

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
    xreg.names <- colnames(matat);

    n.param.max <- n.param.max + estimate.FX*length(matFX) + estimate.gX*nrow(vecgX) + estimate.initialX*ncol(matat);

##### Check number of observations vs number of max parameters #####
    if(obs.ot <= n.param.max){
        if(silent.text==FALSE){
            message(paste0("Number of non-zero observations is ",obs.ot,", while the maximum number of parameters to estimate is ", n.param.max,"."));
        }

        if(obs.ot > 3){
            models.pool <- c("ANN");
            if(allowMultiplicative==TRUE){
                models.pool <- c(models.pool,"MNN");
            }
            if(obs.ot > 5){
                models.pool <- c(models.pool,"AAN");
                if(allowMultiplicative==TRUE){
                    models.pool <- c(models.pool,"AMN","MAN","MMN");
                }
            }
            if(obs.ot > 6){
                models.pool <- c(models.pool,"AAdN");
                if(allowMultiplicative==TRUE){
                    models.pool <- c(models.pool,"AMdN","MAdN","MMdN");
                }
            }
            if((obs.ot > 2*datafreq) & datafreq!=1){
                models.pool <- c(models.pool,"ANA");
                if(allowMultiplicative==TRUE){
                    models.pool <- c(models.pool,"ANM","MNA","MNM");
                }
            }
            if((obs.ot > (6 + datafreq)) & (obs.ot > 2*datafreq) & datafreq!=1){
                models.pool <- c(models.pool,"AAA");
                if(allowMultiplicative==TRUE){
                    models.pool <- c(models.pool,"AAM","AMA","AMM","MAA","MAM","MMA","MMM");
                }
            }

            warning("Not enough observations for the fit of ETS(",model,")! Fitting what we can...",call.=FALSE,immediate.=TRUE);
            if(modelDo=="combine"){
                model <- "CNN";
                if(length(models.pool)>2){
                    model <- "CCN";
                }
                if(length(models.pool)>10){
                    model <- "CCC";
                }
            }
            else{
                model <- "ZZZ";
            }
        }
        else{
            stop("Not enough observations... Even for fitting of ETS('ANN')!",call.=FALSE);
        }
    }

##### Define modelDo #####
    if(any(estimate.persistence, estimate.initial*(fittertype=="o"), estimate.initial.season*(fittertype=="o"),
           estimate.FX, estimate.gX, estimate.initialX)){
        if(all(modelDo!=c("select","combine"))){
            modelDo <- "estimate";
        }
    }
    else{
        modelDo <- "nothing";
    }

##### Basic estimation function for es() #####
esEstimator <- function(...){
    environment(esBasicMaker) <- environment();
    environment(C.values) <- environment();
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();
    environment(CF) <- environment();
    esBasicMaker(ParentEnvironment=environment());

    Cs <- C.values(bounds,Ttype,Stype,vecg,matvt,phi,maxlag,n.components,matat);
    C <- Cs$C;
    C.upper <- Cs$C.upper;
    C.lower <- Cs$C.lower;

    # Parameters are chosen to speed up the optimisation process and have decent accuracy
    res <- nloptr(C, CF, lb=C.lower, ub=C.upper,
                  opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=500));
    C <- res$solution;

    # If the optimisation failed, then probably this is because of smoothing parameters in mixed models. Set them eqaul to zero.
    if(any(C==Cs$C)){
        if(C[1]==Cs$C[1]){
            C[1] <- 0;
        }
        if(Ttype!="N"){
            if(C[2]==Cs$C[2]){
                C[2] <- 0;
            }
            if(Stype!="N"){
                if(C[3]==Cs$C[3]){
                    C[3] <- 0;
                }
            }
        }
        else{
            if(Stype!="N"){
                if(C[2]==Cs$C[2]){
                    C[2] <- 0;
                }
            }
        }
        res <- nloptr(C, CF, lb=C.lower, ub=C.upper,
                      opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=500));
        C <- res$solution;
    }

    res <- nloptr(C, CF, lb=C.lower, ub=C.upper,
                  opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-6, "maxeval"=500));
    C <- res$solution;

    if(all(C==Cs$C)){
        warning(paste0("Failed to optimise the model ETS(",current.model,
                       "). Try different parameters maybe?\nAnd check all the messages and warnings...",
                       "If you did your best, but the optimiser still fails, report this to the maintainer, please."),
                call.=FALSE, immediate.=TRUE);
    }

    n.param <- n.components + damped + (n.components - (Stype!="N"))*(fittertype=="o") + maxlag*(fittertype=="o") +
        !is.null(xreg) * n.exovars + (go.wild==TRUE)*(n.exovars^2 + n.exovars) + 1;

#    n.param <- n.components*estimate.persistence + estimate.phi +
#        (n.components - (Stype!="N"))*estimate.initial*(fittertype=="o") +
#        maxlag*estimate.initial.season*(fittertype=="o") +
#        estimate.initialX * n.exovars + estimate.FX * n.exovars^2 + estimate.gX * n.exovars + 1;

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

    IC.values <- ICFunction(n.param=n.param+n.param.intermittent,C=res$solution,Etype=Etype);
    ICs <- IC.values$ICs;

    # Change back
    CF.type <- CF.type.original;
    return(list(ICs=ICs,objective=res$objective,C=C,n.param=n.param,FI=FI));
}

##### This function prepares pool of models to use #####
esPoolPreparer <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];
    environment(esEstimator) <- environment();

    if(!is.null(models.pool)){
        models.number <- length(models.pool);

# List for the estimated models in the pool
        results <- as.list(c(1:models.number));
        j <- 0;
    }
    else{
# Define the pool of models in case of "ZZZ" or "CCC" to select from
        if(allowMultiplicative==FALSE){
            if(silent.text==FALSE){
                message("Only additive models are allowed with non-positive data.");
            }
            errors.pool <- c("A");
            trends.pool <- c("N","A","Ad");
            season.pool <- c("N","A");
        }
        else{
            errors.pool <- c("A","M");
            trends.pool <- c("N","A","Ad","M","Md");
            season.pool <- c("N","A","M");
        }

        if(Etype!="Z"){
            errors.pool <- Etype;
        }

# List for the estimated models in the pool
        results <- list(NA);

### Use brains in order to define models to estimate ###
        if(modelDo=="select" & any(c(Ttype,Stype)=="Z")){
            if(silent.text==FALSE){
                cat("Forming the pool of models based on... ");
            }

# Some preparation variables
            if(Etype!="Z"){
                small.pool.error <- Etype;
                errors.pool <- Etype;
            }
            else{
                small.pool.error <- "A";
            }

            if(Ttype!="Z"){
                if(damped==TRUE){
                    small.pool.trend <- paste0(Ttype,"d");
                    trends.pool <- small.pool.trend;
                }
                else{
                    small.pool.trend <- Ttype;
                    trends.pool <- Ttype;
                }
                check.T <- FALSE;
            }
            else{
                small.pool.trend <- c("N","A");
                check.T <- TRUE;
            }

            if(Stype!="Z"){
                small.pool.season <- Stype;
                season.pool <- Stype;
                check.S <- FALSE;
            }
            else{
                small.pool.season <- c("N","A","M");
                check.S <- TRUE;
            }

# If ZZZ, then the vector is: "ANN" "ANA" "ANM" "AAN" "AAA" "AAM"
            small.pool <- paste0(rep(small.pool.error,length(small.pool.trend)*length(small.pool.season)),
                                 rep(small.pool.trend,each=length(small.pool.season)),
                                 rep(small.pool.season,length(small.pool.trend)));
            tested.model <- NULL;

# Counter + checks for the components
            j <- 1;
            i <- 0;
            check <- TRUE;
            besti <- bestj <- 1;

#### Branch and bound is here ####
            while(check==TRUE){
                i <- i + 1;
                current.model <- small.pool[j];
                if(silent.text==FALSE){
                    cat(paste0(current.model,", "));
                }
                Etype <- substring(current.model,1,1);
                Ttype <- substring(current.model,2,2);
                if(nchar(current.model)==4){
                    damped <- TRUE;
                    phi <- NULL;
                    Stype <- substring(current.model,4,4);
                }
                else{
                    damped <- FALSE;
                    phi <- 1;
                    Stype <- substring(current.model,3,3);
                }
                if(Stype!="N"){
                    estimate.initial.season <- TRUE;
                }
                else{
                    estimate.initial.season <- FALSE;
                }

                res <- esEstimator(ParentEnvironment=environment());
                results[[i]] <- c(res$ICs,Etype,Ttype,Stype,damped,res$objective,res$C,res$n.param);

                tested.model <- c(tested.model,current.model);

                if(j>1){
# If the first is better than the second, then choose first
                    if(as.numeric(results[[besti]][IC]) <= as.numeric(results[[i]][IC])){
# If Ttype is the same, then we checked seasonality
                        if(substring(current.model,2,2) == substring(small.pool[bestj],2,2)){
                            season.pool <- results[[besti]][6];
                            check.S <- FALSE;
                            j <- which(small.pool!=small.pool[bestj] &
                                           substring(small.pool,nchar(small.pool),nchar(small.pool))==season.pool);
                        }
# Otherwise we checked trend
                        else{
                            trends.pool <- results[[bestj]][5];
                            check.T <- FALSE;
                        }
                    }
                    else{
                        if(substring(current.model,2,2) == substring(small.pool[besti],2,2)){
                            season.pool <- season.pool[season.pool!=results[[besti]][6]];
                            if(length(season.pool)>1){
# Select another seasonal model, that is not from the previous iteration and not the current one
                                bestj <- j;
                                besti <- i;
                                j <- 3;
                            }
                            else{
                                bestj <- j;
                                besti <- i;
                                j <- which(substring(small.pool,nchar(small.pool),nchar(small.pool))==season.pool &
                                          substring(small.pool,2,2)!=substring(current.model,2,2));
                                check.S <- FALSE;
                            }
                        }
                        else{
                            trends.pool <- trends.pool[trends.pool!=results[[besti]][5]];
                            besti <- i;
                            bestj <- j;
                            check.T <- FALSE;
                        }
                    }

                    if(all(c(check.T,check.S)==FALSE)){
                        check <- FALSE;
                    }
                }
                else{
                    j <- 2;
                }
            }

            models.pool <- paste0(rep(errors.pool,each=length(trends.pool)*length(season.pool)),
                                  trends.pool,
                                  rep(season.pool,each=length(trends.pool)));

            models.pool <- unique(c(tested.model,models.pool));
            models.number <- length(models.pool);
            j <- length(tested.model);
        }
        else{
# Make the corrections in the pool for combinations
            if(Etype!="Z"){
                errors.pool <- Etype;
            }
            if(Ttype!="Z"){
                trends.pool <- Ttype;
            }
            if(Stype!="Z"){
                season.pool <- Stype;
            }

            models.number <- (length(errors.pool)*length(trends.pool)*length(season.pool));
            models.pool <- paste0(rep(errors.pool,each=length(trends.pool)*length(season.pool)),
                                  trends.pool,
                                  rep(season.pool,each=length(trends.pool)));
            j <- 0;
        }
    }
    assign("models.pool",models.pool,ParentEnvironment);
    assign("models.number",models.number,ParentEnvironment);
    assign("j",j,ParentEnvironment);
    assign("results",results,ParentEnvironment);
}

##### Function for estimation of pool of models #####
esPoolEstimator <- function(silent.text=FALSE,...){
    environment(esEstimator) <- environment();
    environment(esPoolPreparer) <- environment();
    esPoolValues <- esPoolPreparer(ParentEnvironment=environment());

    if(silent.text==FALSE){
        cat("Estimation progress:    ");
    }
# Start loop of models
    while(j < models.number){
        j <- j + 1;
        if(silent.text==FALSE){
            if(j==1){
                cat("\b");
            }
            cat(paste0(rep("\b",nchar(round((j-1)/models.number,2)*100)+1),collapse=""));
            cat(paste0(round(j/models.number,2)*100,"%"));
        }

        current.model <- models.pool[j];
        Etype <- substring(current.model,1,1);
        Ttype <- substring(current.model,2,2);
        if(nchar(current.model)==4){
            damped <- TRUE;
            phi <- NULL;
            Stype <- substring(current.model,4,4);
        }
        else{
            damped <- FALSE;
            phi <- 1;
            Stype <- substring(current.model,3,3);
        }
        if(Stype!="N"){
            estimate.initial.season <- TRUE;
        }
        else{
            estimate.initial.season <- FALSE;
        }

        res <- esEstimator(ParentEnvironment=environment());
        results[[j]] <- c(res$ICs,Etype,Ttype,Stype,damped,res$objective,res$C,res$n.param);
    }

    if(silent.text==FALSE){
        cat("... Done! \n");
    }
    IC.selection <- matrix(NA,models.number,3);
#    IC.selection <- rep(NA,models.number);
    for(i in 1:models.number){
#        IC.selection[i,] <- as.numeric(eval(parse(text=paste0("results[[",i,"]]['",IC,"']"))));
        IC.selection[i,] <- as.numeric(results[[i]][1:3]);
    }
    colnames(IC.selection) <- names(results[[1]])[1:3]

    IC.selection[is.nan(IC.selection)] <- 1E100;

    return(list(results=results,IC.selection=IC.selection));
}

##### Function selects the best es() based on IC #####
esCreator <- function(silent.text=FALSE,...){
    if(modelDo=="select"){
        environment(esPoolEstimator) <- environment();
        esPoolResults <- esPoolEstimator(silent.text=silent.text);
        results <- esPoolResults$results;
        IC.selection <- esPoolResults$IC.selection;

        bestIC <- min(IC.selection[,IC]);
        i <- which(IC.selection[,IC]==bestIC)[1];
        ICs <- IC.selection[i,];
        results <- results[[i]];

        Etype <- results[4];
        Ttype <- results[5];
        Stype <- results[6];
        damped <- as.logical(results[7]);
        if(damped==TRUE){
            phi <- NULL;
        }
        CF.objective <- as.numeric(results[8]);
        C <- as.numeric(results[-c(1:8)]);

        return(list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,phi=phi,
                    CF.objective=CF.objective,C=C,ICs=ICs,bestIC=bestIC,n.param=as.numeric(results[length(results)]),FI=FI));
    }
    else if(modelDo=="combine"){
        environment(esPoolEstimator) <- environment();
        esPoolResults <- esPoolEstimator(silent.text=silent.text);
        results <- esPoolResults$results;
        IC.selection <- esPoolResults$IC.selection;
        IC.selection <- IC.selection[,IC];
        bestIC <- min(IC.selection);
        IC.selection <- IC.selection/(h^multisteps);
        IC.weights <- exp(-0.5*(IC.selection-bestIC))/sum(exp(-0.5*(IC.selection-bestIC)));
        ICs <- sum(IC.selection * IC.weights);
        return(list(IC.weights=IC.weights,ICs=ICs,bestIC=bestIC,results=results));
    }
    else if(modelDo=="estimate"){
        environment(esEstimator) <- environment();
        res <- esEstimator(ParentEnvironment=environment());
        bestIC <- res$ICs[IC];

        return(list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,phi=phi,
                    CF.objective=res$objective,C=res$C,ICs=res$ICs,bestIC=bestIC,n.param=res$n.param,FI=FI));
    }
    else{
        environment(ICFunction) <- environment();
        environment(likelihoodFunction) <- environment();

        C <- c(vecg);
        if(damped==TRUE){
            C <- c(C,phi);
        }
        C <- c(C,initial,initial.season);
        if(estimate.xreg==TRUE){
            C <- c(C,initialX);
            if(go.wild==TRUE){
                C <- c(C,transitionX,persistenceX);
            }
        }

        CF.objective <- CF(C);

        # Number of parameters
        n.param <- n.components + damped + (n.components - (Stype!="N"))*(fittertype=="o") + maxlag*(fittertype=="o") +
            !is.null(xreg) * n.exovars + (go.wild==TRUE)*(n.exovars^2 + n.exovars) + 1;

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
        bestIC <- ICs[IC];
        # Change back
        CF.type <- CF.type.original;

        return(list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,phi=phi,
                    CF.objective=CF.objective,C=C,ICs=ICs,bestIC=bestIC,n.param=n.param,FI=FI));
    }
}

##### Now do estimation and model selection #####
    environment(intermittentParametersSetter) <- environment();
    environment(intermittentMaker) <- environment();
    environment(esBasicInitialiser) <- environment();
    environment(ssFitter) <- environment();
    environment(ssForecaster) <- environment();

# If auto intermittent, then estimate model with intermittent="n" first.
    if(any(intermittent==c("a","n"))){
        intermittentParametersSetter(intermittent="n",ParentEnvironment=environment());
    }
    else{
        intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
        intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    }
    esValues <- esCreator();

##### If intermittent=="a", run a loop and select the best one #####
    if(intermittent=="a"){
        if(silent.text==FALSE){
            cat("Selecting appropriate type of intermittency... ");
        }
# Prepare stuff for intermittency selection
        intermittentModelsPool <- c("n","f","c","t");
        intermittentICs <- rep(NA,length(intermittentModelsPool));
        intermittentModelsList <- list(NA);
        intermittentICs <- esValues$bestIC;

        for(i in 2:length(intermittentModelsPool)){
            intermittentParametersSetter(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentMaker(intermittent=intermittentModelsPool[i],ParentEnvironment=environment());
            intermittentModelsList[[i]] <- esCreator(silent.text=TRUE);
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
            esValues <- intermittentModelsList[[iBest]];
        }
        else{
            intermittent <- "n"
        }

        intermittentParametersSetter(intermittent=intermittent,ParentEnvironment=environment());
        intermittentMaker(intermittent=intermittent,ParentEnvironment=environment());
    }

##### Fit simple model and produce forecast #####
    if(modelDo!="combine"){
        list2env(esValues,environment());
        esBasicMaker(ParentEnvironment=environment());
        esBasicInitialiser(ParentEnvironment=environment());

        if(damped==TRUE){
            model <- paste0(Etype,Ttype,"d",Stype);
        }
        else{
            model <- paste0(Etype,Ttype,Stype);
        }

        # Write down Fisher Information if needed
        if(FI==TRUE){
            environment(likelihoodFunction) <- environment();
            FI <- numDeriv::hessian(likelihoodFunction,C);
        }

        ssFitter(ParentEnvironment=environment());
        ssForecaster(ParentEnvironment=environment());

        component.names <- "level";
        if(Ttype!="N"){
            component.names <- c(component.names,"trend");
        }
        if(Stype!="N"){
            component.names <- c(component.names,"seasonality");
        }

        if(!is.null(xreg)){
            matvt <- cbind(matvt,matat[1:nrow(matvt),]);
            colnames(matvt) <- c(component.names,xreg.names);
        }
        else{
            colnames(matvt) <- c(component.names);
        }

        # Write down the initials. Done especially for Nikos and issue #10
        if(estimate.initial==TRUE){
            initial <- matvt[maxlag,1:(n.components - (Stype!="N"))]
        }
        if(estimate.initialX==TRUE){
            initialX <- matat[1,];
        }

        if(estimate.initial.season==TRUE){
            if(Stype!="N"){
                initial.season <- matvt[1:maxlag,n.components]
            }
        }
    }
##### Produce fit and forecasts of combined model #####
    else{
        list2env(esValues,environment());

        # Produce the forecasts using AIC weights
        models.number <- length(IC.weights);
        model.current <- rep(NA,models.number);
        fitted.list <- matrix(NA,obs,models.number);
        errors.list <- matrix(NA,obs,models.number);
        forecasts.list <- matrix(NA,h,models.number);
        if(intervals==TRUE){
             lower.list <- matrix(NA,h,models.number);
             upper.list <- matrix(NA,h,models.number);
        }

        for(i in 1:length(IC.weights)){
            # Get all the parameters from the model
            Etype <- results[[i]][4];
            Ttype <- results[[i]][5];
            Stype <- results[[i]][6];
            damped <- as.logical(results[[i]][7]);
            CF.objective <- as.numeric(results[[i]][8]);
            C <- as.numeric(results[[i]][-c(1:8)]);
            n.param <- as.numeric(results[[i]][length(results[[i]])]);
            esBasicMaker(ParentEnvironment=environment());
            esBasicInitialiser(ParentEnvironment=environment());

            ssFitter(ParentEnvironment=environment());
            ssForecaster(ParentEnvironment=environment());

            fitted.list[,i] <- y.fit;
            forecasts.list[,i] <- y.for;
            if(intervals==TRUE){
                lower.list[,i] <- y.low;
                upper.list[,i] <- y.high;
            }
            phi <- NULL;
            if(damped==TRUE){
                model.current[i] <- paste0(Etype,Ttype,"d",Stype);
            }
            else{
                model.current[i] <- paste0(Etype,Ttype,Stype);
            }
        }
        y.fit <- ts(fitted.list %*% IC.weights,start=start(data),frequency=frequency(data));
        y.for <- ts(forecasts.list %*% IC.weights,start=time(data)[obs]+deltat(data),frequency=frequency(data));
        errors <- ts(c(y) - y.fit,start=start(data),frequency=frequency(data));
        names(IC.weights) <- model.current;
        if(intervals==TRUE){
            y.low <- ts(lower.list %*% IC.weights,start=start(y.for),frequency=frequency(data));
            y.high <- ts(upper.list %*% IC.weights,start=start(y.for),frequency=frequency(data));
        }
        else{
            y.low <- NA;
            y.high <- NA;
        }
        names(ICs) <- paste0("Combined ",IC);
    }

##### Do final check and make some preparations for output #####
    if(any(is.na(y.fit),is.na(y.for))){
        warning("Something went wrong during the optimisation and NAs were produced!",call.=FALSE,immediate.=TRUE);
        warning("Please check the input and report this error to the maintainer if it persists.",call.=FALSE,immediate.=TRUE);
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

##### Now let's deal with holdout #####
    if(holdout==TRUE){
        y.holdout <- ts(data[(obs+1):obs.all],start=start(y.for),frequency=frequency(data));
        errormeasures <- errorMeasurer(y.holdout,y.for,y);
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

    modelname <- paste0("ETS(",model,")");

##### Print output #####
    if(silent.text==FALSE){
        if(modelDo!="combine" & any(abs(eigen(matF - vecg %*% matw)$values)>(1 + 1E-10))){
            message(paste0("Model ETS(",model,") is unstable! Use a different value of 'bounds' parameter to address this issue!"));
        }
# Calculate the number of observations in the interval
        if(all(holdout==TRUE,intervals==TRUE)){
            insideintervals <- sum(as.vector(data)[(obs+1):obs.all]<=y.high &
                                   as.vector(data)[(obs+1):obs.all]>=y.low)/h*100;
        }
        else{
            insideintervals <- NULL;
        }

        if(modelDo!="combine"){
            if(damped==TRUE){
                phivalue <- phi;
            }
            else{
                phivalue <- NULL;
            }
            ssoutput(Sys.time() - start.time, modelname, persistence=vecg, transition=NULL, measurement=NULL,
                     phi=phivalue, ARterms=NULL, MAterms=NULL, const=NULL, A=NULL, B=NULL,
                    n.components=n.components, s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
                    CF.type=CF.type, CF.objective=CF.objective, intervals=intervals,
                    int.type=int.type, int.w=int.w, ICs=ICs,
                    holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures, intermittent=intermittent);
        }
        else{
            cat(paste0(IC," weights were used to produce the combination of forecasts\n"));
            ssoutput(Sys.time() - start.time, modelname, persistence=NULL, transition=NULL, measurement=NULL,
                    phi=NULL, ARterms=NULL, MAterms=NULL, const=NULL, A=NULL, B=NULL,
                    n.components=NULL, s2=NULL, hadxreg=!is.null(xreg), wentwild=go.wild,
                    CF.type=CF.type, CF.objective=NULL, intervals=intervals,
                    int.type=int.type, int.w=int.w, ICs=ICs,
                    holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures, intermittent=intermittent);
        }
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
    if(modelDo!="combine"){
        return(list(model=modelname,states=matvt,persistence=as.vector(vecg),phi=phi,
                    initial=initial,initial.season=initial.season,
                    fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,errors=errors.mat,
                    actuals=data,holdout=y.holdout,iprob=pt,intermittent=intermittent,
                    xreg=xreg,initialX=initialX,persistenceX=vecgX,transitionX=matFX,
                    ICs=ICs,CF=CF.objective,CF.type=CF.type,FI=FI,accuracy=errormeasures));
    }
    else{
        return(list(model=modelname,fitted=y.fit,forecast=y.for,
                    lower=y.low,upper=y.high,residuals=errors,
                    actuals=data,holdout=y.holdout,iprob=pt,intermittent=intermittent,
                    ICs=ICs,ICw=IC.weights,
                    CF.type=CF.type,xreg=xreg,accuracy=errormeasures));
    }
}
