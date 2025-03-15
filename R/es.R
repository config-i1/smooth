utils::globalVariables(c("vecg","nComponents","lagsModel","phiEstimate","yInSample","dataFreq","initialType",
                         "yot","lagsModelMax","silent","allowMultiplicative","modelCurrent",
                         "nParamOccurrence","matF","matw","pForecast","errors.mat",
                         "results","s2","FI","occurrence","normalizer",
                         "persistenceEstimate","initial","multisteps","ot",
                         "silentText","silentGraph","silentLegend","yForecastStart",
                         "icBest","icSelection","icWeights"));

#' @rdname es
#' @export
es_old <- function(y, model="ZZZ", persistence=NULL, phi=NULL,
               initial=c("optimal","backcasting"), initialSeason=NULL, ic=c("AICc","AIC","BIC","BICc"),
               loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
               h=10, holdout=FALSE, cumulative=FALSE,
               interval=c("none","parametric","likelihood","semiparametric","nonparametric"), level=0.95,
               bounds=c("usual","admissible","none"),
               silent=c("all","graph","legend","output","none"),
               xreg=NULL, regressors=c("use","select"), initialX=NULL, ...){
# Copyright (C) 2015 - Inf  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

    ### Depricate the old parameters
    ellipsis <- list(...)
    ellipsis <- depricator(ellipsis, "xregDo", "regressors");

    updateX <- FALSE;
    persistenceX <- transitionX <- NULL;
    occurrence <- "none";
    oesmodel <- "MNN";

    #This overrides the similar thing in ssfunctions.R but only for data generated from sim.es()
    if(is.smooth.sim(y)){
        if(smoothType(y)=="ETS"){
            model <- y;
            y <- y$data;
        }
    }
    else if(is.smooth(y)){
        model <- y;
        y <- y$y;
    }

# If a previous model provided as a model, write down the variables
    if(is.smooth(model) | is.smooth.sim(model)){
        if(smoothType(model)!="ETS"){
            stop("The provided model is not ETS.",call.=FALSE);
        }
        if(!is.null(model$occurrence)){
            occurrence <- model$occurrence;
        }
        # If this is the simulated data, extract the parameters
        if(is.smooth.sim(model) & !is.null(dim(model$data))){
            warning("The provided model has several submodels. Choosing a random one.",call.=FALSE);
            i <- round(runif(1,1:length(model$persistence)));
            persistence <- model$persistence[,i];
            initial <- model$initial[,i];
            initialSeason <- model$initialSeason[,i];
            if(any(model$probability!=1)){
                occurrence <- "a";
            }
        }
        else{
            persistence <- model$persistence;
            initial <- model$initial;
            initialSeason <- model$initialSeason;
            if(any(model$probability!=1)){
                occurrence <- "a";
            }
        }
        phi <- model$phi;
        if(is.null(xreg)){
            xreg <- model$xreg;
        }
        else{
            if(is.null(model$xreg)){
                xreg <- NULL;
            }
            else{
                if(ncol(xreg)!=ncol(model$xreg)){
                    xreg <- xreg[,colnames(model$xreg)];
                }
            }
        }

        initialX <- model$initialX;
        persistenceX <- model$persistenceX;
        transitionX <- model$transitionX;
        if(any(c(persistenceX)!=0) | any((transitionX!=0)&(transitionX!=1))){
            updateX <- TRUE;
        }
        model <- modelType(model);
        if(any(unlist(gregexpr("C",model))!=-1)){
            initial <- "o";
        }
    }
    else if(inherits(model,"ets")){
        # Extract smoothing parameters
        i <- 1;
        persistence <- coef(model)[i];
        if(model$components[2]!="N"){
            i <- i+1;
            persistence <- c(persistence,coef(model)[i]);
            if(model$components[3]!="N"){
                i <- i+1;
                persistence <- c(persistence,coef(model)[i]);
            }
        }
        else{
            if(model$components[3]!="N"){
                i <- i+1;
                persistence <- c(persistence,coef(model)[i]);
            }
        }

        # Damping parameter
        if(model$components[4]=="TRUE"){
            i <- i+1;
            phi <- coef(model)[i];
        }

        # Initials
        i <- i+1;
        initial <- coef(model)[i];
        if(model$components[2]!="N"){
            i <- i+1;
            initial <- c(initial,coef(model)[i]);
        }

        # Initials of seasonal component
        if(model$components[3]!="N"){
            if(model$components[2]!="N"){
                initialSeason <- rev(model$states[1,-c(1:2)]);
            }
            else{
                initialSeason <- rev(model$states[1,-c(1)]);
            }
        }
        model <- modelType(model);
    }
    else if(is.character(model)){
        # Everything is okay
    }
    else{
        warning("A model of an unknown class was provided. Switching to 'ZZZ'.",call.=FALSE);
        model <- "ZZZ";
    }

# Add all the variables in ellipsis to current environment
    list2env(ellipsis,environment());

##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- environment();
    ssInput("es",ParentEnvironment=environment());

##### Cost Function for ES #####
CF <- function(B){
    elements <- etsmatrices(matvt, vecg, phi, matrix(B,nrow=1), nComponents,
                            lagsModel, initialType, Ttype, Stype, nExovars, matat,
                            persistenceEstimate, phiEstimate, initialType=="o", initialSeasonEstimate, xregEstimate,
                            matFX, vecgX, updateX, FXEstimate, gXEstimate, initialXEstimate);

    cfRes <- costfunc(elements$matvt, elements$matF, elements$matw, yInSample, elements$vecg,
                      h, lagsModel, Etype, Ttype, Stype,
                      multisteps, loss, normalizer, initialType,
                      matxt, elements$matat, elements$matFX, elements$vecgX, ot,
                      bounds, elements$errorSD);

    if(is.nan(cfRes) | is.na(cfRes) | is.infinite(cfRes)){
        cfRes <- 1e+500;
    }

    return(cfRes);
}

##### B values for estimation #####
# Function constructs default bounds where B values should lie
BValues <- function(bounds,Etype,Ttype,Stype,vecg,matvt,phi,lagsModelMax,nComponents,matat){
    B <- NA;
    lb <- NA;
    ub <- NA;
    # This is the vector with the names of the elemenets of the vector of parameters
    BNames <- NA;

    if(bounds=="u"){
        if(persistenceEstimate){
            B <- c(B,vecg);
            lb <- c(lb,rep(0,length(vecg)));
            ub <- c(ub,rep(1,length(vecg)));
            if(Ttype!="N"){
                BNames <- c(BNames, c("alpha","beta","gamma")[1:nComponents]);
            }
            else{
                BNames <- c(BNames, c("alpha","gamma")[1:nComponents]);
            }
        }
        if(damped & phiEstimate){
            B <- c(B,phi);
            lb <- c(lb,0);
            ub <- c(ub,1);
            BNames <- c(BNames, "phi");
        }
        if(any(initialType==c("o","p"))){
            if(initialType=="o"){
                BNames <- c(BNames, "level");
                if(Etype=="A"){
                    B <- c(B,matvt[lagsModelMax,1:(nComponents - (Stype!="N"))]);
                    lb <- c(lb,-Inf);
                    ub <- c(ub,Inf);
                }
                else{
                    if(Ttype=="A"){
                        # This is something like ETS(M,A,N), so set level to mean, trend to zero for stability
                        B <- c(B,mean(yInSample[1:min(dataFreq,obsInSample)]),0);
                    }
                    else{
                        B <- c(B,abs(matvt[lagsModelMax,1:(nComponents - (Stype!="N"))]));
                    }
                    lb <- c(lb,1E-10);
                    ub <- c(ub,Inf);
                }
                if(Ttype=="A"){
                    lb <- c(lb,-Inf);
                    ub <- c(ub,Inf);
                    BNames <- c(BNames, "trend");
                }
                else if(Ttype=="M"){
                    lb <- c(lb,1E-20);
                    ub <- c(ub,3);
                    BNames <- c(BNames, "trend");
                }
            }
            if(Stype!="N"){
                if(initialSeasonEstimate){
                    B <- c(B,matvt[1:lagsModelMax,nComponents]);
                    if(Stype=="A"){
                        lb <- c(lb,rep(-Inf,lagsModelMax));
                        ub <- c(ub,rep(Inf,lagsModelMax));
                    }
                    else{
                        lb <- c(lb,matvt[1:lagsModelMax,nComponents]*seasonalRandomness[1]);
                        ub <- c(ub,matvt[1:lagsModelMax,nComponents]*seasonalRandomness[2]);
                    }
                    BNames <- c(BNames, paste0("seasonal",c(1:lagsModelMax)))
                }
            }
        }
    }
    else if(bounds=="a"){
        if(persistenceEstimate){
            B <- c(B,vecg);
            if(Etype=="A"){
                lb <- c(lb,rep(-5,length(vecg)));
            }
            else{
                lb <- c(lb,rep(0,length(vecg)));
            }
            ub <- c(ub,rep(5,length(vecg)));
            if(Ttype!="N"){
                BNames <- c(BNames, c("alpha","beta","gamma")[1:nComponents]);
            }
            else{
                BNames <- c(BNames, c("alpha","gamma")[1:nComponents]);
            }
        }
        if(damped & phiEstimate){
            B <- c(B,phi);
            lb <- c(lb,0);
            ub <- c(ub,1);
            BNames <- c(BNames, "phi");
        }
        if(any(initialType==c("o","p"))){
            if(initialType=="o"){
                BNames <- c(BNames, "level");
                if(Etype=="A"){
                    B <- c(B,matvt[lagsModelMax,1:(nComponents - (Stype!="N"))]);
                    lb <- c(lb,-Inf);
                    ub <- c(ub,Inf);
                }
                else{
                    if(Ttype=="A"){
                        # This is something like ETS(M,A,N), so set level to mean, trend to zero for stability
                        B <- c(B,mean(yInSample[1:dataFreq]),0);
                    }
                    else{
                        B <- c(B,abs(matvt[lagsModelMax,1:(nComponents - (Stype!="N"))]));
                    }
                    lb <- c(lb,0.1);
                    ub <- c(ub,Inf);
                }
                if(Ttype=="A"){
                    lb <- c(lb,-Inf);
                    ub <- c(ub,Inf);
                    BNames <- c(BNames, "trend");
                }
                else if(Ttype=="M"){
                    lb <- c(lb,0.01);
                    ub <- c(ub,3);
                    BNames <- c(BNames, "trend");
                }
            }
            if(Stype!="N"){
                if(initialSeasonEstimate){
                    B <- c(B,matvt[1:lagsModelMax,nComponents]);
                    if(Stype=="A"){
                        lb <- c(lb,rep(-Inf,lagsModelMax));
                        ub <- c(ub,rep(Inf,lagsModelMax));
                    }
                    else{
                        lb <- c(lb,matvt[1:lagsModelMax,nComponents]*seasonalRandomness[1]);
                        ub <- c(ub,matvt[1:lagsModelMax,nComponents]*seasonalRandomness[2]);
                    }
                    BNames <- c(BNames, paste0("seasonal",c(1:lagsModelMax)))
                }
            }
        }
    }
    else{
        if(persistenceEstimate){
            B <- c(B,vecg);
            lb <- c(lb,rep(-Inf,length(vecg)));
            ub <- c(ub,rep(Inf,length(vecg)));
            if(Ttype!="N"){
                BNames <- c(BNames, c("alpha","beta","gamma")[1:nComponents]);
            }
            else{
                BNames <- c(BNames, c("alpha","gamma")[1:nComponents]);
            }
        }
        if(damped & phiEstimate){
            B <- c(B,phi);
            lb <- c(lb,-Inf);
            ub <- c(ub,Inf);
            BNames <- c(BNames, "phi");
        }
        if(any(initialType==c("o","p"))){
            if(initialType=="o"){
                BNames <- c(BNames, "level");
                if(Etype=="A"){
                    B <- c(B,matvt[lagsModelMax,1:(nComponents - (Stype!="N"))]);
                    lb <- c(lb,-Inf);
                    ub <- c(ub,Inf);
                }
                else{
                    if(Ttype=="A"){
                        # This is something like ETS(M,A,N), so set level to mean, trend to zero for stability
                        B <- c(B,mean(yInSample[1:dataFreq]),0);
                    }
                    else{
                        B <- c(B,abs(matvt[lagsModelMax,1:(nComponents - (Stype!="N"))]));
                    }
                    lb <- c(lb,0.1);
                    ub <- c(ub,Inf);
                }
                if(Ttype=="A"){
                    lb <- c(lb,-Inf);
                    ub <- c(ub,Inf);
                    BNames <- c(BNames, "trend");
                }
                else if(Ttype=="M"){
                    lb <- c(lb,0.01);
                    ub <- c(ub,3);
                    BNames <- c(BNames, "trend");
                }
            }
            if(Stype!="N"){
                if(initialSeasonEstimate){
                    B <- c(B,matvt[1:lagsModelMax,nComponents]);
                    if(Stype=="A"){
                        lb <- c(lb,rep(-Inf,lagsModelMax));
                        ub <- c(ub,rep(Inf,lagsModelMax));
                    }
                    else{
                        lb <- c(lb,matvt[1:lagsModelMax,nComponents]*seasonalRandomness[1]);
                        ub <- c(ub,matvt[1:lagsModelMax,nComponents]*seasonalRandomness[2]);
                    }
                    BNames <- c(BNames, paste0("seasonal",c(1:lagsModelMax)));
                }
            }
        }
    }

    if(xregEstimate){
        if(initialXEstimate){
            if(Etype=="M"){
                B <- c(B,matatMultiplicative[1,xregNames]);
            }
            else{
                B <- c(B,matatOriginal[1,xregNames]);
            }
            lb <- c(lb,rep(-Inf,nExovars));
            ub <- c(ub,rep(Inf,nExovars));
            BNames <- c(BNames, xregNames);
        }
        if(updateX){
            if(FXEstimate){
                B <- c(B,as.vector(matFX));
                lb <- c(lb,rep(-Inf,nExovars^2));
                ub <- c(ub,rep(Inf,nExovars^2));
                BNames <- c(BNames, paste0("transitionX",c(1:nExovars^2)));
            }
            if(gXEstimate){
                B <- c(B,as.vector(vecgX));
                lb <- c(lb,rep(-Inf,nExovars));
                ub <- c(ub,rep(Inf,nExovars));
                BNames <- c(BNames, paste0("persistenceX",c(1:nExovars)));
            }
        }
    }

    names(B) <- BNames;
    B <- as.numeric(B[!is.na(B)]);
    lb <- as.numeric(lb[!is.na(lb)]);
    ub <- as.numeric(ub[!is.na(ub)]);

    return(list(B=B,lb=lb,ub=ub));
}

##### Basic parameter creator #####
# This function creates all the necessary matrices
BasicMakerES <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    basicparams <- initparams(Etype, Ttype, Stype, dataFreq, obsInSample, obsAll, yInSample,
                              damped, phi, smoothingParameters, initialstates, seasonalCoefs);
    list2env(basicparams,ParentEnvironment);
}

##### Basic parameter initialiser #####
# This function fills in all the necessary matrices
BasicInitialiserES <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    elements <- etsmatrices(matvt, vecg, phi, matrix(B,nrow=1), nComponents,
                            lagsModel, initialType, Ttype, Stype, nExovars, matat,
                            persistenceEstimate, phiEstimate, initialType=="o", initialSeasonEstimate, xregEstimate,
                            matFX, vecgX, updateX, FXEstimate, gXEstimate, initialXEstimate);

    list2env(elements,ParentEnvironment);
}

##### Basic estimation function for es() #####
EstimatorES <- function(...){
    environment(BasicMakerES) <- environment();
    environment(BValues) <- environment();
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();
    environment(CF) <- environment();
    BasicMakerES(ParentEnvironment=environment());

    BValuesList <- BValues(bounds,Etype,Ttype,Stype,vecg,matvt,phi,lagsModelMax,nComponents,matat);
    if(is.null(providedC)){
        B <- BValuesList$B;
    }
    else{
        # This part is needed for the regressors="select"
        B <- providedC;
        # If the generated B is larger, then probably there is updateX=T
        if(length(BValuesList$B)>length(B)){
            B <- c(B,BValuesList$B[-c(1:length(B))]);
        }
    }
    if(is.null(providedCLower)){
        lb <- BValuesList$lb;
    }
    if(is.null(providedCUpper)){
        ub <- BValuesList$ub;
    }

    if(rounded){
        loss <- "MSE";
    }

    if(any(is.infinite(B))){
        B[is.infinite(B)] <- 0.1;
    }

    # Change B if it is out of the bounds
    if(any((B>=ub),(B<=lb))){
        ub[B>=ub & B<0] <- B[B>=ub & B<0] * 0.999 + 0.001;
        ub[B>=ub & B>=0] <- B[B>=ub & B>=0] * 1.001 + 0.001;
        lb[B<=lb & B<0] <- B[B<=lb & B<0] * 1.001 - 0.001;
        lb[B<=lb & B>=0] <- B[B<=lb & B>=0] * 0.999 - 0.001;
    }

    # Parameters are chosen to speed up the optimisation process and have decent accuracy
    res <- nloptr(B, CF, lb=lb, ub=ub,
                  opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=xtol_rel, "maxeval"=maxeval, print_level=print_level));
    B[] <- res$solution;

    # If the optimisation failed, then probably this is because of mixed models...
    if(any(res$objective==c(1e+100,1e+300))){
        # Reset the smoothing parameters
        j <- 1;
        B[j] <- max(0,lb[j]);
        if(Ttype!="N"){
            j <- j+1;
            B[j] <- max(0,lb[j]);
            if(Stype!="N"){
                j <- j+1;
                B[j] <- max(0,lb[j]);
            }
        }
        else{
            if(Stype!="N"){
                j <- j+1;
                B[j] <- max(0,lb[j]);
            }
        }

        # If the optimiser fails, then it's probably due to the mixed models. So make all the initials non-negative
        if(any(c(Etype,Ttype,Stype)=="M")){
            B <- abs(B);
            if(Ttype=="A"){
                if(damped & phiEstimate){
                    j <- j+1;
                }
                j <- j+1;
                B[j] <- mean(yInSample[1:dataFreq]);
                j <- j+1;
                B[j] <- 0;
            }
        }

        res <- nloptr(B, CF, lb=lb, ub=ub,
                      opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=xtol_rel, "maxeval"=maxeval, print_level=print_level));
        B[] <- res$solution;
    }
    # Change B if it is out of the bounds
    if(any((B>ub),(B<lb))){
        ub[B>=ub & B<0] <- B[B>=ub & B<0] * 0.999 + 0.001;
        ub[B>=ub & B>=0] <- B[B>=ub & B>=0] * 1.001 + 0.001;
        lb[B<=lb & B<0] <- B[B<=lb & B<0] * 1.001 - 0.001;
        lb[B<=lb & B>=0] <- B[B<=lb & B>=0] * 0.999 - 0.001;
    }

    if(rounded){
        # Take the estimate of RMSE as an initial estimate of SD
        B <- c(B,sqrt(CF(B)));
        lb <- c(lb,0);
        ub <- c(ub,Inf);
        loss <- "Rounded";
        names(B)[length(B)] <- "SD";
    }

    res2 <- nloptr(B, CF, lb=lb, ub=ub,
                  opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=xtol_rel * 10^2, "maxeval"=maxeval, print_level=print_level));

    # This condition is needed in order to make sure that we did not make the solution worse
    if((res2$objective <= res$objective) | rounded){
        res <- res2;
    }
    B[] <- res$solution;

    if(!rounded && (initialType!="b") && all(B==BValuesList$B)){
        if(any(persistenceEstimate,gXEstimate,FXEstimate)){
            warning(paste0("Failed to optimise the model ETS(", modelCurrent,
                           "). Try different initialisation maybe?\nAnd check all the messages and warnings...",
                           "If you did your best, but the optimiser still fails, report this to the maintainer, please."),
                    call.=FALSE);
        }
    }
    # Parameters estimated + variance
    nParam <- length(B) + 1*(!rounded);

    # Write down Fisher Information if needed
    if(FI){
        boundOriginal <- bounds;
        bounds[] <- "n";
        environment(likelihoodFunction) <- environment();
        FI <- -numDeriv::hessian(likelihoodFunction,B);
        bounds <- boundOriginal;
    }

    # Check if smoothing parameters and phi reached the boundary conditions
    if(bounds=="u"){
        CNamesAvailable <- c("alpha","beta","gamma","phi")[c("alpha","beta","gamma","phi") %in% names(B)];
        # If the value is very close to zero, assume zero
        if(any(B[CNamesAvailable]<=1e-3)){
            CNamesAvailableChange <- CNamesAvailable[B[CNamesAvailable]<=1e-3]
            nParam[] <- nParam - length(CNamesAvailableChange);
            B[CNamesAvailableChange] <- 0;
            # if(!silentText){
            #     message("Some smoothing parameters reached the lower bound, we consider them as provided",);
            # }
        }
        # If the value is very close to one, assume one
        if(any(B[CNamesAvailable]>=1-1e-3)){
            CNamesAvailableChange <- CNamesAvailable[B[CNamesAvailable]>=1-1e-3]
            nParam[] <- nParam - length(CNamesAvailableChange);
            B[CNamesAvailableChange] <- 1;
            # if(!silentText){
            #     message("Some smoothing parameters reached the upper bound, we consider them as provided");
            # }
        }
    }

    yFittedSumLog <- 0;
    if(Etype=="M"){
        elements <- etsmatrices(matvt, vecg, phi, matrix(B,nrow=1), nComponents,
                                lagsModel, initialType, Ttype, Stype, nExovars, matat,
                                persistenceEstimate, phiEstimate, initialType=="o", initialSeasonEstimate, xregEstimate,
                                matFX, vecgX, updateX, FXEstimate, gXEstimate, initialXEstimate);
        yFittedSumLog[] <- sum(log(abs(fitterwrap(elements$matvt, elements$matF, elements$matw, yInSample, elements$vecg,
                                                  lagsModel, Etype, Ttype, Stype, initialType,
                                                  matxt, elements$matat, elements$matFX, elements$vecgX, ot)$yfit)));
    }
    ICValues <- ICFunction(nParam=nParam,nParamOccurrence=nParamOccurrence,
                           B=res$solution,Etype=Etype,yFittedSumLog=yFittedSumLog);
    ICs <- ICValues$ICs;
    logLik <- ICValues$llikelihood;

    return(list(ICs=ICs,objective=res$objective,B=B,nParam=nParam,FI=FI,logLik=logLik));
}

##### This function uses residuals in order to determine the needed xreg #####
XregSelector <- function(listToReturn){
# Prepare for fitting
    environment(BasicMakerES) <- environment();
    environment(BasicInitialiserES) <- environment();
    environment(EstimatorES) <- environment();
    environment(ssFitter) <- environment();
    list2env(listToReturn, environment());

    BasicMakerES(ParentEnvironment=environment());
    BasicInitialiserES(ParentEnvironment=environment());
    ssFitter(ParentEnvironment=environment());

    xregNames <- colnames(matxtOriginal);
    xregNew <- cbind(errors,xreg[1:obsInSample,]);
    colnames(xregNew)[1] <- "errors";
    colnames(xregNew)[-1] <- xregNames;
    xregNew <- as.data.frame(xregNew);
    xregResults <- stepwise(xregNew, ic=ic, silent=TRUE, df=nParam+nParamOccurrence-1);
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
        xregNames <- NULL;
        listToReturn$xregEstimate <- xregEstimate;
    }

    if(!is.null(xreg)){
        if(Etype=="M" & any(abs(coef(xregResults)[-1])>10)){
            providedC <- c(B,coef(xregResults)[-1]/max(abs(coef(xregResults)[-1])));
        }
        else{
            providedC <- c(B,coef(xregResults)[-1]);
        }
        phi <- NULL;

        res <- EstimatorES(ParentEnvironment=environment());
        icBest <- res$ICs[ic];
        logLik <- res$logLik;
        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,phi=phi,
                             cfObjective=res$objective,B=res$B,ICs=res$ICs,icBest=icBest,
                             nParam=res$nParam,FI=FI,logLik=logLik,xreg=xreg,xregEstimate=xregEstimate,
                             xregNames=xregNames,matFX=matFX,vecgX=vecgX,nExovars=nExovars);
    }

    return(listToReturn);
}

##### This function prepares pool of models to use #####
PoolPreparerES <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];
    environment(EstimatorES) <- environment();

    if(!is.null(modelsPool)){
        modelsNumber <- length(modelsPool);
# List for the estimated models in the pool
        results <- as.list(c(1:modelsNumber));
        j <- 0;
    }
    else{
# Define the pool of models in case of "ZZZ" or "CCC" to select from
        if(!allowMultiplicative){
            if(!silent){
                message("Only additive models are allowed with non-positive data.");
            }
            poolErrors <- c("A");
            poolTrends <- c("N","A","Ad");
            poolSeasonals <- c("N","A");
        }
        else{
            poolErrors <- c("A","M");
            poolTrends <- c("N","A","Ad","M","Md");
            poolSeasonals <- c("N","A","M");
        }

        if(all(Etype!=c("Z","C"))){
            poolErrors <- Etype;
        }

# List for the estimated models in the pool
        results <- list(NA);

### Use brains in order to define models to estimate ###
        if(modelDo=="select" &
           (any(c(Ttype,Stype)=="X") | any(c(Ttype,Stype)=="Y") | any(c(Ttype,Stype)=="Z"))){
            if(!silent){
                cat("Forming the pool of models based on... ");
            }

# Some preparation variables
            if(Etype!="Z"){
                small.pool.error <- Etype;
                poolErrors <- Etype;
            }
            else{
                small.pool.error <- c("A");
            }

            if(Ttype!="Z"){
                if(Ttype=="X"){
                    small.pool.trend <- c("N","A");
                    poolTrends <- c("N","A","Ad");
                    check.T <- TRUE;
                }
                else if(Ttype=="Y"){
                    small.pool.trend <- c("N","M");
                    poolTrends <- c("N","M","Md");
                    check.T <- TRUE;
                }
                else{
                    if(damped){
                        small.pool.trend <- paste0(Ttype,"d");
                        poolTrends <- small.pool.trend;
                    }
                    else{
                        small.pool.trend <- Ttype;
                        poolTrends <- Ttype;
                    }
                    check.T <- FALSE;
                }
            }
            else{
                small.pool.trend <- c("N","A");
                check.T <- TRUE;
            }

            if(Stype!="Z"){
                if(Stype=="X"){
                    small.pool.season <- c("N","A");
                    poolSeasonals <- c("N","A");
                    check.S <- TRUE;
                }
                else if(Stype=="Y"){
                    small.pool.season <- c("N","M");
                    poolSeasonals <- c("N","M");
                    check.S <- TRUE;
                }
                else{
                    small.pool.season <- Stype;
                    poolSeasonals <- Stype;
                    check.S <- FALSE;
                }
            }
            else{
                small.pool.season <- c("N","A","M");
                check.S <- TRUE;
            }

# If ZZZ, then the vector is: "ANN" "ANA" "ANM" "AAN" "AAA" "AAM"
            small.pool <- paste0(rep(small.pool.error,length(small.pool.trend)*length(small.pool.season)),
                                 rep(small.pool.trend,each=length(small.pool.season)),
                                 rep(small.pool.season,length(small.pool.trend)));
            # If the "M" is allowed, align errors with the seasonality
            if(allowMultiplicative){
                small.pool[substr(small.pool,3,3)=="M"] <- paste0("M",substr(small.pool[substr(small.pool,3,3)=="M"],2,3))
            }
            tested.model <- NULL;

# Counter + checks for the components
            j <- 1;
            i <- 0;
            check <- TRUE;
            besti <- bestj <- 1;

#### Branch and bound is here ####
            while(check){
                i <- i + 1;
                modelCurrent <- small.pool[j];
                if(!silent){
                    cat(paste0(modelCurrent,", "));
                }
                Etype <- substring(modelCurrent,1,1);
                Ttype <- substring(modelCurrent,2,2);
                if(nchar(modelCurrent)==4){
                    damped <- TRUE;
                    phi <- NULL;
                    Stype <- substring(modelCurrent,4,4);
                }
                else{
                    damped <- FALSE;
                    phi <- 1;
                    Stype <- substring(modelCurrent,3,3);
                }
                if(Stype!="N"){
                    initialSeasonEstimate <- TRUE;
                }
                else{
                    initialSeasonEstimate <- FALSE;
                }

                res <- EstimatorES(ParentEnvironment=environment());

                listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,phi=phi,
                                     cfObjective=res$objective,B=res$B,ICs=res$ICs,icBest=NULL,
                                     nParam=res$nParam,logLik=res$logLik,xreg=xreg,FI=res$FI,
                                     xregNames=xregNames,matFX=matFX,vecgX=vecgX,nExovars=nExovars);

                if(regressors!="u"){
                    listToReturn <- XregSelector(listToReturn=listToReturn);
                }
                results[[i]] <- listToReturn;

                tested.model <- c(tested.model,modelCurrent);

                if(j>1){
# If the first is better than the second, then choose first
                    if(results[[besti]]$ICs[ic] <= results[[i]]$ICs[ic]){
# If Ttype is the same, then we checked seasonality
                        if(substring(modelCurrent,2,2) == substring(small.pool[bestj],2,2)){
                            poolSeasonals <- results[[besti]]$Stype;
                            check.S <- FALSE;
                            j <- which(small.pool!=small.pool[bestj] &
                                           substring(small.pool,nchar(small.pool),nchar(small.pool))==poolSeasonals);
                        }
# Otherwise we checked trend
                        else{
                            poolTrends <- results[[bestj]]$Ttype;
                            check.T <- FALSE;
                        }
                    }
                    else{
                        if(substring(modelCurrent,2,2) == substring(small.pool[besti],2,2)){
                            poolSeasonals <- poolSeasonals[poolSeasonals!=results[[besti]]$Stype];
                            if(length(poolSeasonals)>1){
# Select another seasonal model, that is not from the previous iteration and not the current one
                                bestj <- j;
                                besti <- i;
                                j <- 3;
                            }
                            else{
                                bestj <- j;
                                besti <- i;
                                j <- which(substring(small.pool,nchar(small.pool),nchar(small.pool))==poolSeasonals &
                                          substring(small.pool,2,2)!=substring(modelCurrent,2,2));
                                check.S <- FALSE;
                            }
                        }
                        else{
                            poolTrends <- poolTrends[poolTrends!=results[[bestj]]$Ttype];
                            besti <- i;
                            bestj <- j;
                            check.T <- FALSE;
                        }
                    }

                    if(all(!c(check.T,check.S))){
                        check <- FALSE;
                    }
                }
                else{
                    j <- 2;
                }
            }

            modelsPool <- paste0(rep(poolErrors,each=length(poolTrends)*length(poolSeasonals)),
                                  poolTrends,
                                  rep(poolSeasonals,each=length(poolTrends)));

            modelsPool <- unique(c(tested.model,modelsPool));
            modelsNumber <- length(modelsPool);
            j <- length(tested.model);
        }
        else{
# Make the corrections in the pool for combinations
            if(all(Ttype!=c("Z","C"))){
                if(Ttype=="Y"){
                    poolTrends <- c("N","M","Md");
                }
                else if(Ttype=="X"){
                    poolTrends <- c("N","A","Ad");
                }
                else{
                    if(damped){
                        poolTrends <- paste0(Ttype,"d");
                    }
                    else{
                        poolTrends <- Ttype;
                    }
                }
            }
            if(all(Stype!=c("Z","C"))){
                if(Stype=="Y"){
                    poolTrends <- c("N","M");
                }
                else if(Stype=="X"){
                    poolTrends <- c("N","A");
                }
                else{
                    poolSeasonals <- Stype;
                }
            }

            modelsNumber <- (length(poolErrors)*length(poolTrends)*length(poolSeasonals));
            modelsPool <- paste0(rep(poolErrors,each=length(poolTrends)*length(poolSeasonals)),
                                  poolTrends,
                                  rep(poolSeasonals,each=length(poolTrends)));
            j <- 0;
        }
    }
    assign("modelsPool",modelsPool,ParentEnvironment);
    assign("modelsNumber",modelsNumber,ParentEnvironment);
    assign("j",j,ParentEnvironment);
    assign("results",results,ParentEnvironment);
}

##### Function for estimation of pool of models #####
PoolEstimatorES <- function(silent=FALSE,...){
    environment(EstimatorES) <- environment();
    environment(PoolPreparerES) <- environment();
    esPoolValues <- PoolPreparerES(ParentEnvironment=environment());

    if(!silent){
        cat("Estimation progress:    ");
    }
# Start loop of models
    while(j < modelsNumber){
        j <- j + 1;
        if(!silent){
            if(j==1){
                cat("\b");
            }
            cat(paste0(rep("\b",nchar(round((j-1)/modelsNumber,2)*100)+1),collapse=""));
            cat(paste0(round(j/modelsNumber,2)*100,"%"));
        }

        modelCurrent <- modelsPool[j];
        Etype <- substring(modelCurrent,1,1);
        Ttype <- substring(modelCurrent,2,2);
        if(nchar(modelCurrent)==4){
            damped <- TRUE;
            phi <- NULL;
            Stype <- substring(modelCurrent,4,4);
        }
        else{
            damped <- FALSE;
            phi <- 1;
            Stype <- substring(modelCurrent,3,3);
        }
        if(Stype!="N"){
            initialSeasonEstimate <- TRUE;
        }
        else{
            initialSeasonEstimate <- FALSE;
        }

        # Make sure that this thing is NULL This is needed for xregSelector
        providedC <- NULL;

        res <- EstimatorES(ParentEnvironment=environment());

        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,phi=phi,
                             cfObjective=res$objective,B=res$B,ICs=res$ICs,icBest=NULL,
                             nParam=res$nParam,logLik=res$logLik,xreg=xreg, FI=res$FI,
                             xregNames=xregNames,matFX=matFX,vecgX=vecgX,nExovars=nExovars);
        if(regressors!="u"){
            listToReturn <- XregSelector(listToReturn=listToReturn);
        }

        results[[j]] <- listToReturn;
    }

    if(!silent){
        cat("... Done! \n");
    }
    icSelection <- matrix(NA,modelsNumber,4);
    for(i in 1:modelsNumber){
        icSelection[i,] <- results[[i]]$ICs;
    }
    colnames(icSelection) <- names(results[[i]]$ICs);
    rownames(icSelection) <- modelsPool;

    icSelection[is.nan(icSelection)] <- 1E100;

    return(list(results=results,icSelection=icSelection));
}

##### Function selects the best es() based on IC #####
CreatorES <- function(silent=FALSE,...){
    if(modelDo=="select"){
        environment(PoolEstimatorES) <- environment();
        esPoolResults <- PoolEstimatorES(silent=silent);
        results <- esPoolResults$results;
        icSelection <- esPoolResults$icSelection;
        icBest <- apply(icSelection,2,min);
        i <- which(icSelection[,ic]==icBest[ic])[1];
        listToReturn <- results[[i]];
        listToReturn$icBest <- icBest;
        listToReturn$ICs <- icSelection;

        return(listToReturn);
    }
    else if(modelDo=="combine"){
        environment(PoolEstimatorES) <- environment();
        esPoolResults <- PoolEstimatorES(silent=silent);
        results <- esPoolResults$results;
        icSelection <- esPoolResults$icSelection;
        icSelection <- icSelection/(h^multisteps);
        icBest <- apply(icSelection,2,min);
        icBest <- matrix(icBest,nrow=nrow(icSelection),ncol=4,byrow=TRUE);
        icWeights <- (exp(-0.5*(icSelection-icBest)) /
                          matrix(colSums(exp(-0.5*(icSelection-icBest))),
                                 nrow=nrow(icSelection),ncol=4,byrow=TRUE));
        ICs <- colSums(icSelection * icWeights);
        return(list(icWeights=icWeights,ICs=ICs,icBest=icBest,results=results,cfObjective=NA,
                    icSelection=icSelection));
    }
    else if(modelDo=="estimate"){
        environment(EstimatorES) <- environment();
        res <- EstimatorES(ParentEnvironment=environment());
        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,phi=phi,
                             cfObjective=res$objective,B=res$B,ICs=res$ICs,icBest=res$ICs,
                             nParam=res$nParam,FI=res$FI,logLik=res$logLik,xreg=xreg,
                             xregNames=xregNames,matFX=matFX,vecgX=vecgX,nExovars=nExovars);

        if(regressors!="u"){
            listToReturn <- XregSelector(listToReturn=listToReturn);
        }

        return(listToReturn);
    }
    else{
        environment(CF) <- environment();
        environment(ICFunction) <- environment();
        environment(likelihoodFunction) <- environment();
        environment(BasicMakerES) <- environment();

        BasicMakerES(ParentEnvironment=environment());

        B <- c(vecg);
        if(damped){
            B <- c(B,phi);
        }
        B <- c(B,initialValue,initialSeason);
        if(xregEstimate){
            B <- c(B,initialX);
            if(updateX){
                B <- c(B,transitionX,persistenceX);
            }
        }

        cfObjective <- CF(B);

        # Only variance is estimated in this case
        nParam <- 1;

        ICValues <- ICFunction(nParam=nParam,nParamOccurrence=nParamOccurrence,
                               B=B,Etype=Etype);
        logLik <- ICValues$llikelihood;
        ICs <- ICValues$ICs;
        icBest <- ICs;

        listToReturn <- list(Etype=Etype,Ttype=Ttype,Stype=Stype,damped=damped,phi=phi,
                             cfObjective=cfObjective,B=B,ICs=ICs,icBest=icBest,
                             nParam=nParam,FI=FI,logLik=logLik,xreg=xreg,
                             xregNames=xregNames,matFX=matFX,vecgX=vecgX,nExovars=nExovars);
        return(listToReturn);
    }
}

##### Set initialstates, initialSesons and persistence vector #####
    # If initial values are provided, write them. If not, estimate them.
    # First two columns are needed for additive seasonality, the 3rd and 4th - for the multiplicative
    if(Ttype!="N"){
        if(initialType!="p"){
            initialstates <- matrix(NA,1,5);
            initialstates[1,2] <- (cov(yot[1:min(max(dataFreq,12),obsNonzero)],
                                       c(1:min(max(dataFreq,12),obsNonzero)))/
                                       var(c(1:min(max(dataFreq,12),obsNonzero))));
            initialstates[1,1] <- (mean(yot[1:min(max(dataFreq,12),obsNonzero)]) -
                                       initialstates[1,2] *
                                       mean(c(1:min(max(dataFreq,12), obsNonzero))));
            if(allowMultiplicative){
                initialstates[1,4] <- exp(cov(log(yot[1:min(max(dataFreq,12),obsNonzero)]),
                                              c(1:min(max(dataFreq,12),obsNonzero)))/
                                              var(c(1:min(max(dataFreq,12),obsNonzero))));
                initialstates[1,3] <- exp(mean(log(yot[1:min(max(dataFreq,12),obsNonzero)])) -
                                              log(initialstates[1,4]) *
                                              mean(c(1:min(max(dataFreq,12),obsNonzero))));
            }
            # Initials for non-trended model
            initialstates[1,5] <- mean(yot[1:min(max(dataFreq,12),obsNonzero)]);
        }
        else{
            initialstates <- matrix(rep(initialValue,3)[1:5],nrow=1);
        }
    }
    else{
        if(initialType!="p"){
            initialstates <- matrix(rep(mean(yot[1:min(max(dataFreq,12),obsNonzero)]),5),nrow=1);
        }
        else{
            initialstates <- matrix(rep(initialValue,5),nrow=1);
        }
    }

    # Define matrix of seasonal coefficients. The first column consists of additive, the second - multiplicative elements
    # If the seasonal model is chosen and initials are provided, fill in the first "lagsModelMax" values of seasonal component.
    if(Stype!="N"){
        if(is.null(initialSeason)){
            initialSeasonEstimate <- TRUE;
            seasonalCoefs <- decompose(ts(c(yInSample),frequency=dataFreq),type="additive")$figure;
            decompositionM <- decompose(ts(c(yInSample),frequency=dataFreq), type="multiplicative");
            seasonalCoefs <- cbind(seasonalCoefs,decompositionM$figure);
            seasonalRandomness <- c(min(decompositionM$random,na.rm=TRUE),
                                    max(decompositionM$random,na.rm=TRUE));
        }
        else{
            initialSeasonEstimate <- FALSE;
            seasonalCoefs <- cbind(initialSeason,initialSeason);
        }
    }
    else{
        initialSeasonEstimate <- FALSE;
        seasonalCoefs <- matrix(1,1,1);
    }

    # If the persistence vector is provided, use it
    if(!is.null(persistence)){
        smoothingParameters <- cbind(persistence,persistence);
    }
    else{
        smoothingParameters <- cbind(c(0.3,0.2,0.1),c(0.1,0.01,0.01));

        if(loss=="HAM"){
            smoothingParameters <- cbind(rep(0.01,3),rep(0.01,3));
        }
    }

##### Preset yFitted, yForecast, errors and basic parameters #####
    yFitted <- rep(NA,obsInSample);
    yForecast <- rep(NA,h);
    errors <- rep(NA,obsInSample);

    basicparams <- initparams(Etype, Ttype, Stype, dataFreq, obsInSample, obsAll, yInSample,
                              damped, phi, smoothingParameters, initialstates, seasonalCoefs);

##### Prepare exogenous variables #####
    xregdata <- ssXreg(y=y, Etype=Etype, xreg=xreg, updateX=updateX, ot=ot,
                       persistenceX=persistenceX, transitionX=transitionX, initialX=initialX,
                       obsInSample=obsInSample, obsAll=obsAll, obsStates=obsStates,
                       lagsModelMax=basicparams$lagsModelMax, h=h, regressors=regressors, silent=silentText,
                       allowMultiplicative=allowMultiplicative);

    if(regressors=="u"){
        nExovars <- xregdata$nExovars;
        matxtOriginal <- matxt <- xregdata$matxt;
        matatOriginal <- matat <- xregdata$matat;
        matatMultiplicative <- xregdata$matatMultiplicative;
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
        matatMultiplicative <- xregdata$matatMultiplicative;
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
    if(is.null(xreg)){
        regressors <- "u";
    }

    nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) + initialXEstimate*ncol(matat);
    # If this is the intermittent model, then at least one more observation is needed.
    nParamOccurrence <- all(occurrence!=c("n","p"))*1;
    nParamMax <- nParamMax + nParamExo + nParamOccurrence;

    if(regressors=="u"){
        parametersNumber[1,2] <- nParamExo;
        # If transition is provided and not identity, and other things are provided, write them as "provided"
        parametersNumber[2,2] <- (length(matFX)*(!is.null(transitionX) & !all(matFX==diag(ncol(matat)))) +
                                      nrow(vecgX)*(!is.null(persistenceX)) +
                                      ncol(matat)*(!is.null(initialX)));
    }

##### Check number of observations vs number of max parameters #####
    if(obsNonzero <= nParamMax){
        #### !!! This needs to be cleverer. If user asks for "ZNZ", then trends should be excluded.
        if(!silentText){
            message(paste0("Number of non-zero observations is ",obsNonzero,
                           ", while the maximum number of parameters to estimate is ", nParamMax,".\n",
                           "Updating pool of models."));
        }

        if(obsNonzero > (3 + nParamExo + nParamOccurrence) & is.null(modelsPool)){
            # We have enough observations for local level model
            modelsPool <- c("ANN");
            if(allowMultiplicative){
                modelsPool <- c(modelsPool,"MNN");
            }
            # We have enough observations for trend model
            if(obsNonzero > (5 + nParamExo + nParamOccurrence)){
                modelsPool <- c(modelsPool,"AAN");
                if(allowMultiplicative){
                    modelsPool <- c(modelsPool,"AMN","MAN","MMN");
                }
            }
            # We have enough observations for damped trend model
            if(obsNonzero > (6 + nParamExo + nParamOccurrence)){
                modelsPool <- c(modelsPool,"AAdN");
                if(allowMultiplicative){
                    modelsPool <- c(modelsPool,"AMdN","MAdN","MMdN");
                }
            }
            # We have enough observations for seasonal model
            if((obsNonzero > (2*dataFreq)) & dataFreq!=1){
                modelsPool <- c(modelsPool,"ANA");
                if(allowMultiplicative){
                    modelsPool <- c(modelsPool,"ANM","MNA","MNM");
                }
            }
            # We have enough observations for seasonal model with trend
            if((obsNonzero > (6 + dataFreq + nParamExo + nParamOccurrence)) & (obsNonzero > 2*dataFreq) & dataFreq!=1){
                modelsPool <- c(modelsPool,"AAA");
                if(allowMultiplicative){
                    modelsPool <- c(modelsPool,"AAM","AMA","AMM","MAA","MAM","MMA","MMM");
                }
            }

            warning("Not enought of non-zero observations for the fit of ETS(",
                    model,")! Fitting what we can...",call.=FALSE);
            if(modelDo=="combine"){
                model <- "CNN";
                if(length(modelsPool)>2){
                    model <- "CCN";
                }
                if(length(modelsPool)>10){
                    model <- "CCC";
                }
            }
            else{
                modelDo <- "select"
                model <- "ZZZ";
            }
        }
        else if(obsNonzero > (3 + nParamExo + nParamOccurrence) & !is.null(modelsPool)){
            # We don't have enough observations for seasonal models with damped trend
            if((obsNonzero <= (6 + dataFreq + 1 + nParamExo + nParamOccurrence))){
                modelsPool <- modelsPool[!(nchar(modelsPool)==4 &
                                               substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="A")];
                modelsPool <- modelsPool[!(nchar(modelsPool)==4 &
                                               substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="M")];
            }
            # We don't have enough observations for seasonal models with trend
            if((obsNonzero <= (5 + dataFreq + 1 + nParamExo + nParamOccurrence))){
                modelsPool <- modelsPool[!(substr(modelsPool,2,2)!="N" &
                                               substr(modelsPool,nchar(modelsPool),nchar(modelsPool))!="N")];
            }
            # We don't have enough observations for seasonal models
            if(obsNonzero <= 2*dataFreq){
                modelsPool <- modelsPool[substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="N"];
            }
            # We don't have enough observations for damped trend
            if(obsNonzero <= (6 + nParamExo + nParamOccurrence)){
                modelsPool <- modelsPool[nchar(modelsPool)!=4];
            }
            # We don't have enough observations for any trend
            if(obsNonzero <= (5 + nParamExo + nParamOccurrence)){
                modelsPool <- modelsPool[substr(modelsPool,2,2)=="N"];
            }

            modelsPool <- unique(modelsPool);
            warning("Not enough of non-zero observations for the fit of ETS(",model,")! Fitting what we can...",call.=FALSE);
            if(modelDo=="combine"){
                model <- "CNN";
                if(length(modelsPool)>2){
                    model <- "CCN";
                }
                if(length(modelsPool)>10){
                    model <- "CCC";
                }
            }
            else{
                modelDo <- "select"
                model <- "ZZZ";
            }
        }
        else if(obsNonzero==4){
            if(any(Etype==c("A","M"))){
                modelDo <- "estimate";
                Ttype <- "N";
                Stype <- "N";
            }
            else{
                modelsPool <- c("ANN");
                if(allowMultiplicative){
                    modelsPool <- c(modelsPool,"MNN");
                }
                modelDo <- "select";
                model <- "ZZZ";
                Etype <- "Z";
                Ttype <- "N";
                Stype <- "N";
                warning("You have a very small sample. The only available model is level model.",
                        call.=FALSE);
            }
            smoothingParameters <- matrix(0,3,2);
            damped <- FALSE;
            phiEstimate <- FALSE;
        }
        else if(obsNonzero==3){
            if(any(Etype==c("A","M"))){
                modelDo <- "estimate";
                Ttype <- "N";
                Stype <- "N";
            }
            else{
                modelsPool <- c("ANN");
                if(allowMultiplicative){
                    modelsPool <- c(modelsPool,"MNN");
                }
                modelDo <- "select";
                model <- "ZZZ";
                Etype <- "Z";
                Ttype <- "N";
                Stype <- "N";
            }
            persistence <- 0;
            persistenceEstimate <- FALSE;
            smoothingParameters <- matrix(0,3,2);
            warning("We did not have enough of non-zero observations, so persistence value was set to zero.",
                    call.=FALSE);
            damped <- FALSE;
            phiEstimate <- FALSE;
        }
        else if(obsNonzero==2){
            modelsPool <- NULL;
            persistence <- 0;
            persistenceEstimate <- FALSE;
            smoothingParameters <- matrix(0,3,2);
            initialValue <- mean(yInSample);
            initialType <- "p";
            initialstates <- matrix(rep(initialValue,3)[1:5],nrow=1);
            warning("We did not have enough of non-zero observations, so persistence value was set to zero and initial was preset.",
                    call.=FALSE);
            modelDo <- "nothing"
            model <- "ANN";
            Etype <- "A";
            Ttype <- "N";
            Stype <- "N";
            damped <- FALSE;
            phiEstimate <- FALSE;
            parametersNumber[1,1] <- 0;
            parametersNumber[2,1] <- 2;
        }
        else if(obsNonzero==1){
            modelsPool <- NULL;
            persistence <- 0;
            persistenceEstimate <- FALSE;
            smoothingParameters <- matrix(0,3,2);
            initialValue <- yInSample[yInSample!=0];
            initialType <- "p";
            initialstates <- matrix(rep(initialValue,3)[1:5],nrow=1);
            warning("We did not have enough of non-zero observations, so we used Naive.",call.=FALSE);
            modelDo <- "nothing"
            model <- "ANN";
            Etype <- "A";
            Ttype <- "N";
            Stype <- "N";
            damped <- FALSE;
            phiEstimate <- FALSE;
            parametersNumber[1,1] <- 0;
            parametersNumber[2,1] <- 2;
        }
        else{
            stop("Not enough observations... Even for fitting of ETS('ANN')!",call.=FALSE);
        }
    }

##### Define modelDo #####
    if(any(persistenceEstimate, (initialType=="o"), initialSeasonEstimate,
           phiEstimate, FXEstimate, gXEstimate, initialXEstimate)){
        if(all(modelDo!=c("select","combine"))){
            modelDo <- "estimate";
            modelCurrent <- model;
        }
        else{
            if(!any(loss==c("likelihood","MSE","MAE","HAM","MSEh","MAEh","HAMh","MSCE","MACE","CHAM",
                              "GPL","aGPL","Rounded","TSB","LogisticD","LogisticL"))){
                if(modelDo=="combine"){
                    warning(paste0("'",loss,"' is used as loss function instead of 'MSE'.",
                                   "The produced combination weights may be wrong."),call.=FALSE);
                }
                else{
                    warning(paste0("'",loss,"' is used as loss function instead of 'MSE'. ",
                                   "The results of the model selection may be wrong."),call.=FALSE);
                }
            }
        }
    }
    else{
        modelDo <- "nothing";
    }

    ellipsis <- list(...);
    if(any(names(ellipsis)=="B")){
        providedC <- ellipsis$B;
    }
    else{
        providedC <- NULL;
    }
    if(any(names(ellipsis)=="lb")){
        providedCLower <- ellipsis$lb;
    }
    else{
        providedCLower <- NULL;
    }
    if(any(names(ellipsis)=="ub")){
        providedCUpper <- ellipsis$ub;
    }
    else{
        providedCUpper <- NULL;
    }

    if(any(names(ellipsis)=="maxeval")){
        maxeval <- ellipsis$maxeval;
    }
    else{
        # If we have a lot of parameters, spend more time on the optimisation
        if(nParamMax + FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) +
           initialXEstimate*(ncol(matatOriginal) - ncol(matat)) > 10){
            maxeval <- 1000;
        }
        else{
            maxeval <- 500;
        }
    }
    if(any(names(ellipsis)=="print_level")){
        print_level <- ellipsis$print_level;
    }
    else{
        print_level <- 0;
    }
    if(any(names(ellipsis)=="xtol_rel")){
        xtol_rel <- ellipsis$xtol_rel;
    }
    else{
        xtol_rel <- 1e-8;
    }


##### Initials for optimiser #####
    if(!all(c(is.null(providedC),is.null(providedCLower),is.null(providedCUpper)))){
        if((modelDo==c("estimate")) & (regressors==c("u"))){
            environment(BasicMakerES) <- environment();
            BasicMakerES(ParentEnvironment=environment());

            # Variance is not needed here, because we do not optimise it
            # nComponents smoothing parameters, phi,
            # level and trend initials if we optimise them,
            # lagsModelMax seasonal initials if we do not backcast and they need to be estimated
            # intiials of xreg if they need to be estimated
            # updateX with transitionX and persistenceX
            nParamToEstimate <- (nComponents*persistenceEstimate + phiEstimate*damped +
                                     (nComponents - (Stype!="N")) * (initialType=="o") +
                                     lagsModelMax * (Stype!="N") * initialSeasonEstimate * (initialType!="b") +
                                     nExovars * initialXEstimate +
                                     (updateX)*((nExovars^2)*(FXEstimate) + nExovars*gXEstimate));

            if(!is.null(providedC)){
                if(nParamToEstimate!=length(providedC)){
                    warning(paste0("Number of parameters to optimise differes from the length of B: ",
                                   nParamToEstimate," vs ",length(providedC),".\n",
                                   "We will have to drop parameter B."),call.=FALSE);
                    providedC <- NULL;
                }
            }
            if(!is.null(providedCLower)){
                if(nParamToEstimate!=length(providedCLower)){
                    warning(paste0("Number of parameters to optimise differes from the length of lb: ",
                                   nParamToEstimate," vs ",length(providedCLower),".\n",
                                   "We will have to drop parameter lb."),call.=FALSE);
                    providedCLower <- NULL;
                }
            }
            if(!is.null(providedCUpper)){
                if(nParamToEstimate!=length(providedCUpper)){
                    warning(paste0("Number of parameters to optimise differes from the length of ub: ",
                                   nParamToEstimate," vs ",length(providedCUpper),".\n",
                                   "We will have to drop parameter ub."),call.=FALSE);
                    providedCUpper <- NULL;
                }
            }
            B <- providedC;
            lb <- providedCLower;
            ub <- providedCUpper;
        }
        else{
            if(modelDo==c("select")){
                warning("Predefined values of B cannot be used with model selection.",call.=FALSE);
            }
            else if(modelDo==c("combine")){
                warning("Predefined values of B cannot be used with combination of forecasts.",call.=FALSE);
            }
            else if(modelDo==c("nothing")){
                warning("Sorry, but there is nothing to optimise, so we have to drop parameter B.",call.=FALSE);
            }

            if(regressors==c("select")){
                warning("Predefined values of B cannot be used with xreg selection.",call.=FALSE);
            }
            B <- NULL;
            lb <- NULL;
            ub <- NULL;
        }

        # If we need to estimate phi, make it NULL, so the next maker works
        if(phiEstimate){
            phi <- NULL
        }
    }

##### Now do estimation and model selection #####
    environment(intermittentParametersSetter) <- environment();
    environment(intermittentMaker) <- environment();
    environment(BasicInitialiserES) <- environment();
    environment(ssFitter) <- environment();
    environment(ssForecaster) <- environment();

    EtypeOriginal <- Etype;
    TtypeOriginal <- Ttype;
    StypeOriginal <- Stype;

    #### The occurrence model ####
# If auto occurrence, then estimate model with occurrence="n" first.
    if(occurrence=="a"){
        if(!silentText){
            cat("Selecting the best occurrence model...\n");
        }
        # First produce the auto model
        intermittentParametersSetter(occurrence="a",ParentEnvironment=environment());
        intermittentMaker(occurrence="a",ParentEnvironment=environment());
        intermittentModel <- CreatorES(silent=silentText);
        occurrenceBest <- occurrence;
        occurrenceModelBest <- occurrenceModel;

        Etype[] <- switch(Etype,
                          "Z"=,
                          "Y"=,
                          "A"=,
                          "M"="A"
                          );
        Ttype[] <- switch(Ttype,
                          "Z"=,
                          "Y"="X",
                          "A"=,
                          "M"="A",
                          "N"="N"
                          );
        Stype[] <- switch(Stype,
                          "Z"=,
                          "Y"="X",
                          "A"=,
                          "M"="A",
                          "N"="N"
                          );

        if(!silentText){
            cat("Comparing it with the best non-intermittent model...\n");
        }
        # Then fit the model without the occurrence part
        occurrence[] <- "n";
        intermittentParametersSetter(occurrence=occurrence,ParentEnvironment=environment());
        intermittentMaker(occurrence=occurrence,ParentEnvironment=environment());
        nonIntermittentModel <- CreatorES(silent=silentText);

        # Compare the results and return the best
        if(nonIntermittentModel$icBest[ic] <= intermittentModel$icBest[ic]){
            esValues <- nonIntermittentModel;
        }
        # If this is the "auto", then use the selected occurrence to reset the parameters
        else{
            Etype[] <- EtypeOriginal;
            Ttype[] <- TtypeOriginal;
            Stype[] <- StypeOriginal;
            esValues <- intermittentModel;
            occurrenceModel <- occurrenceModelBest;
            occurrence[] <- occurrenceBest;
            intermittentParametersSetter(occurrence=occurrence,ParentEnvironment=environment());
            intermittentMaker(occurrence=occurrence,ParentEnvironment=environment());
        }
        rm(intermittentModel,nonIntermittentModel,occurrenceModelBest);
    }
    else{
        intermittentParametersSetter(occurrence=occurrence,ParentEnvironment=environment());
        intermittentMaker(occurrence=occurrence,ParentEnvironment=environment());

        esValues <- CreatorES(silent=silentText);
    }

##### Fit simple model and produce forecast #####
    if(modelDo!="combine"){
        list2env(esValues,environment());
        BasicMakerES(ParentEnvironment=environment());

        if(!is.null(xregNames)){
            matat <- as.matrix(matatOriginal[,xregNames]);
            matxt <- as.matrix(matxtOriginal[,xregNames]);
            if(ncol(matat)==1){
                colnames(matxt) <- xregNames;
            }
            xreg <- matxt;
        }
        else{
            xreg <- NULL;
        }
        BasicInitialiserES(ParentEnvironment=environment());
        if(!is.null(xregNames)){
            colnames(matat) <- xregNames;
        }

        if(damped){
            model <- paste0(Etype,Ttype,"d",Stype);
        }
        else{
            model <- paste0(Etype,Ttype,Stype);
        }

        ssFitter(ParentEnvironment=environment());
        # If this was rounded values, extract the variance
        if(rounded){
            s2 <- B[length(B)]^2;
            s2g <- log(1 + vecg %*% as.vector(errors*ot)) %*% t(log(1 + vecg %*% as.vector(errors*ot)))/obsInSample;
        }
        ssForecaster(ParentEnvironment=environment());

        componentNames <- "level";
        if(Ttype!="N"){
            componentNames <- c(componentNames,"trend");
        }
        if(Stype!="N"){
            componentNames <- c(componentNames,"seasonal");
        }

        if(!is.null(xregNames)){
            matvt <- cbind(matvt,matat[1:nrow(matvt),]);
            colnames(matvt) <- c(componentNames,xregNames);
            if(updateX){
                rownames(vecgX) <- xregNames;
                dimnames(matFX) <- list(xregNames,xregNames);
            }
        }
        else{
            colnames(matvt) <- c(componentNames);
        }

# Write down the initials. Done especially for Nikos and issue #10
        if(persistenceEstimate){
            persistence <- as.vector(vecg);
        }
        if(Ttype!="N"){
            names(persistence) <- c("alpha","beta","gamma")[1:nComponents];
        }
        else{
            names(persistence) <- c("alpha","gamma")[1:nComponents];
        }

        if(initialType!="p"){
            initialValue <- matvt[lagsModelMax,1:(nComponents - (Stype!="N"))];
        }

        if(initialXEstimate){
            initialX <- matat[1,];
            names(initialX) <- colnames(matat);
        }

        # Make initialX NULL if all xreg were dropped
        if(length(initialX)==1){
            if(initialX==0){
                initialX <- NULL;
            }
        }

        if(gXEstimate){
            persistenceX <- vecgX;
        }

        if(FXEstimate){
            transitionX <- matFX;
        }

        if(initialSeasonEstimate){
            if(Stype!="N"){
                initialSeason <- matvt[1:lagsModelMax,nComponents];
                names(initialSeason) <- paste0("s",1:lagsModelMax);
            }
        }

        # Number of estimated parameters + variance
        parametersNumber[1,1] <- length(B) + 1;

# Write down the formula of ETS
        esFormula <- "l[t-1]";
        if(Ttype=="A"){
            esFormula <- paste0(esFormula," + b[t-1]");
        }
        else if(Ttype=="M"){
            esFormula <- paste0(esFormula," * b[t-1]");
        }
        if(Stype=="A"){
            esFormula <- paste0(esFormula," + s[t-",lagsModelMax,"]");
        }
        else if(Stype=="M"){
            if(Ttype=="A"){
                esFormula <- paste0("(",esFormula,")");
            }
            esFormula <- paste0(esFormula," * s[t-",lagsModelMax,"]");
        }
        if(Etype=="A"){
            if(!is.null(xreg)){
                if(updateX){
                    esFormula <- paste0(esFormula," + ",paste0(paste0("a",c(1:nExovars),"[t-1] * "),paste0(xregNames,"[t]"),collapse=" + "));
                }
                else{
                    esFormula <- paste0(esFormula," + ",paste0(paste0("a",c(1:nExovars)," * "),paste0(xregNames,"[t]"),collapse=" + "));
                }
            }
            esFormula <- paste0(esFormula," + e[t]");
        }
        else{
            if(any(c(Ttype,Stype)=="A") & Stype!="M"){
                esFormula <- paste0("(",esFormula,")");
            }
            if(!is.null(xreg)){
                if(updateX){
                    esFormula <- paste0(esFormula," * exp(",
                                        paste0(paste0("a",c(1:nExovars),"[t-1] * "),
                                               paste0(xregNames,"[t]"),collapse=" + "),")");
                }
                else{
                    esFormula <- paste0(esFormula," * exp(",
                                        paste0(paste0("a",c(1:nExovars)," * "),
                                               paste0(xregNames,"[t]"),collapse=" + "),")");
                }
            }
            esFormula <- paste0(esFormula," * e[t]");
        }
        if(occurrence!="n"){
            esFormula <- paste0("o[t] * (",esFormula,")");
        }
        esFormula <- paste0("y[t] = ",esFormula);

        if(modelDo=="select"){
            ICs <- rbind(ICs,icBest);
            rownames(ICs)[nrow(ICs)] <- "Selected";
        }
        else{
            ICs <- t(as.matrix(ICs));
            rownames(ICs) <- model;
        }
    }
##### Produce fit and forecasts of combined model #####
    else{
        list2env(esValues,environment());

        if(!is.null(xreg) & (regressors=="u")){
            colnames(matat) <- xregNames;
            xreg <- matxt;
        }

        modelOriginal <- model;
        # Produce the forecasts using AIC weights
        modelsNumber <- nrow(icWeights);
        model.current <- rep(NA,modelsNumber);
        fittedList <- matrix(NA,obsInSample,modelsNumber);
        # errorsList <- matrix(NA,obsInSample,modelsNumber);
        forecastsList <- matrix(NA,h,modelsNumber);
        if(interval){
             lowerList <- matrix(NA,h,modelsNumber);
             upperList <- matrix(NA,h,modelsNumber);
        }
        for(i in 1:modelsNumber){
            # Get all the parameters from the model
            Etype <- results[[i]]$Etype;
            Ttype <- results[[i]]$Ttype;
            Stype <- results[[i]]$Stype;
            damped <- results[[i]]$damped;
            phi <- results[[i]]$phi;
            cfObjective <- results[[i]]$cfObjective;
            B <- results[[i]]$B;
            nParam <- results[[i]]$nParam;
            xregNames <- results[[i]]$xregNames
            if(regressors!="u"){
                if(!is.null(xregNames)){
                    matat <- as.matrix(matatOriginal[,xregNames]);
                    matxt <- as.matrix(matxtOriginal[,xregNames]);
                }
                else{
                    matxt <- matrix(1,nrow(matxtOriginal),1);
                    matat <- matrix(0,nrow(matatOriginal),1);
                }
                nExovars <- results[[i]]$nExovars;
                matFX <- results[[i]]$matFX;
                vecgX <- results[[i]]$vecgX;
                xregEstimate <- results[[i]]$xregEstimate;
            }

            BasicMakerES(ParentEnvironment=environment());
            BasicInitialiserES(ParentEnvironment=environment());
            if(damped){
                model.current[i] <- paste0(Etype,Ttype,"d",Stype);
            }
            else{
                model.current[i] <- paste0(Etype,Ttype,Stype);
            }
            model <- model.current[i];

            ssFitter(ParentEnvironment=environment());
            ssForecaster(ParentEnvironment=environment());

            fittedList[,i] <- yFitted;
            forecastsList[,i] <- yForecast;
            if(interval){
                lowerList[,i] <- yLower;
                upperList[,i] <- yUpper;
            }
            phi <- NULL;
        }
        badStuff <- apply(is.na(rbind(fittedList,forecastsList)),2,any);
        fittedList <- fittedList[,!badStuff];
        forecastsList <- forecastsList[,!badStuff];
        model.current <- model.current[!badStuff];
        yFitted <- ts(fittedList %*% icWeights[!badStuff,ic],start=dataStart,frequency=dataFreq);
        yForecast <- ts(forecastsList %*% icWeights[!badStuff,ic],start=yForecastStart,frequency=dataFreq);
        errors <- ts(c(yInSample) - yFitted,start=dataStart,frequency=dataFreq);
        s2 <- mean(errors^2);
        if(interval){
            lowerList <- lowerList[,!badStuff];
            upperList <- upperList[,!badStuff];
            yLower <- ts(lowerList %*% icWeights[!badStuff,ic],start=yForecastStart,frequency=dataFreq);
            yUpper <- ts(upperList %*% icWeights[!badStuff,ic],start=yForecastStart,frequency=dataFreq);
        }
        else{
            yLower <- NA;
            yUpper <- NA;
        }
        model <- modelOriginal;

# Write down the formula of ETS
        esFormula <- "y[t] = combination of ";
        if(occurrence!="n"){
            esFormula <- paste0(esFormula,"i");
        }
        esFormula <- paste0(esFormula,"ETS");
        if(!is.null(xreg)){
            esFormula <- paste0(esFormula,"X");
        }
        ICs <- rbind(icSelection,ICs);
        rownames(ICs)[nrow(ICs)] <- "Combined";
    }

##### Do final check and make some preparations for output #####
    # Write down the number of parameters of occurrence part of the model
    if(all(occurrence!=c("n","p")) & !occurrenceModelProvided){
        parametersNumber[1,3] <- nparam(occurrenceModel);
    }

    if(!is.null(xregNames)){
        nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) + initialXEstimate*ncol(matat);
        parametersNumber[1,2] <- nParamExo;
    }

    # Check if smoothing parameters and phi reached the boundary conditions.
    # Do that only in case of estimation / selection and usual bounds
    if(any(modelDo==c("estimate","select")) && bounds=="u"){
        CNamesAvailable <- c("alpha","beta","gamma","phi")[c("alpha","beta","gamma","phi") %in% names(B)];
        # If the value is very close to zero, assume zero
        if(any(B[CNamesAvailable]==0)){
            parametersNumber[1,1] <- parametersNumber[1,1] - sum(B[CNamesAvailable]==0);
            parametersNumber[2,1] <- parametersNumber[2,1] + sum(B[CNamesAvailable]==0);
        }
        # If the value is very close to one, assume one
        if(any(B[CNamesAvailable]==1)){
            parametersNumber[1,1] <- parametersNumber[1,1] - sum(B[CNamesAvailable]==1);
            parametersNumber[2,1] <- parametersNumber[2,1] + sum(B[CNamesAvailable]==1);
        }
        if("phi" %in% CNamesAvailable && B["phi"]==1){
            model <- paste0(substr(model,1,2),substr(model,nchar(model),nchar(model)));
            parametersNumber[2,1] <- parametersNumber[2,1] - 1;
            warning("The parameter phi is equal to one, so we reverted to the non-damped version of the model",
                    call.=FALSE);
        }
    }

    parametersNumber[1,4] <- sum(parametersNumber[1,1:3]);
    parametersNumber[2,4] <- sum(parametersNumber[2,1:3]);

##### Now let's deal with holdout #####
    if(holdout){
        yHoldout <- ts(y[(obsInSample+1):obsAll],start=yForecastStart,frequency=dataFreq);
        if(cumulative){
            errormeasures <- measures(sum(yHoldout),yForecast,h*yInSample);
        }
        else{
            errormeasures <- measures(yHoldout,yForecast,yInSample);
        }

        if(cumulative){
            yHoldout <- ts(sum(yHoldout),start=yForecastStart,frequency=dataFreq);
        }
    }
    else{
        yHoldout <- NA;
        errormeasures <- NA;
    }

    if(!is.null(xreg)){
        modelname <- "ETSX";
    }
    else{
        modelname <- "ETS";
    }
    modelnameForGraph <- modelname <- paste0(modelname,"(",model,")");
    if(all(occurrence!=c("n","none"))){
        modelname <- paste0("i",modelname);
        if(!silentGraph){
            modelnameForGraph <- paste0(modelname,"[",toupper(substr(occurrence,1,1)),"](",modelType(occurrenceModel),")");
        }
    }

##### Print output #####
    if(!silentText){
        if(modelDo!="combine" & any(abs(eigen(matF - vecg %*% matw, only.values=TRUE)$values)>(1 + 1E-10))){
            warning(paste0("Model ETS(",model,") is unstable! Use a different value of 'bounds' parameter to address this issue!"),
                    call.=FALSE);
        }
    }

##### Make a plot #####
    if(!silentGraph){
        yForecastNew <- yForecast;
        yUpperNew <- yUpper;
        yLowerNew <- yLower;
        if(cumulative){
            yForecastNew <- ts(rep(yForecast/h,h),start=yForecastStart,frequency=dataFreq)
            if(interval){
                yUpperNew <- ts(rep(yUpper/h,h),start=yForecastStart,frequency=dataFreq)
                yLowerNew <- ts(rep(yLower/h,h),start=yForecastStart,frequency=dataFreq)
            }
        }

        if(interval){
            graphmaker(actuals=y,forecast=yForecastNew,fitted=yFitted, lower=yLowerNew,upper=yUpperNew,
                       level=level,legend=!silentLegend, main=modelnameForGraph, cumulative=cumulative);
        }
        else{
            graphmaker(actuals=y,forecast=yForecastNew,fitted=yFitted,
                       legend=!silentLegend, main=modelnameForGraph, cumulative=cumulative);
        }
    }

##### Return values #####
    if(modelDo!="combine"){
        model <- list(model=modelname,formula=esFormula,timeElapsed=Sys.time()-startTime,
                      states=matvt,persistence=persistence,phi=phi,transition=matF,
                      measurement=matw,
                      initialType=initialType,initial=initialValue,initialSeason=initialSeason,
                      nParam=parametersNumber,
                      fitted=yFitted,forecast=yForecast,lower=yLower,upper=yUpper,residuals=errors,
                      errors=errors.mat,s2=s2,interval=intervalType,level=level,cumulative=cumulative,
                      y=y,holdout=yHoldout,
                      xreg=xreg,initialX=initialX,
                      ICs=ICs,logLik=logLik,lossValue=cfObjective,loss=loss,FI=FI,accuracy=errormeasures,
                      B=B);
        return(structure(model,class="smooth"));
    }
    else{
        model <- list(model=modelname,formula=esFormula,timeElapsed=Sys.time()-startTime,
                      initialType=initialType,
                      fitted=yFitted,forecast=yForecast,
                      lower=yLower,upper=yUpper,residuals=errors,s2=s2,interval=intervalType,level=level,
                      cumulative=cumulative,
                      y=y,holdout=yHoldout,occurrence=occurrenceModel,
                      xreg=xreg,
                      ICs=ICs,ICw=icWeights,lossValue=NULL,loss=loss,accuracy=errormeasures);
        return(structure(model,class="smooth"));
    }
}
