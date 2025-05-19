utils::globalVariables(c("silentText","silentGraph","silentLegend","initialType","yForecastStart"));

#'
#' @rdname ces
#' @export
ces_old <- function(y, seasonality=c("none","simple","partial","full"),
                    initial=c("backcasting","optimal"), a=NULL, b=NULL, ic=c("AICc","AIC","BIC","BICc"),
                    loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                    h=10, holdout=FALSE,
                    bounds=c("admissible","none"),
                    silent=c("all","graph","legend","output","none"),
                    xreg=NULL, regressors=c("use","select"), initialX=NULL,
                    ...){
# Function estimates CES in state space form with sigma = error
#  and returns complex smoothing parameter value, fitted values,
#  residuals, point and interval forecasts, matrix of CES components and values of
#  information criteria.
#
#    Copyright (C) 2015 - 2016i  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

    ### Depricate the old parameters
    ellipsis <- list(...);

    cumulative <- FALSE;
    interval <- ifelse(!is.null(ellipsis$interval),ellipsis$interval,"none");
    level <- ifelse(!is.null(ellipsis$level),ellipsis$level,0.95);
    xreg <- ellipsis$xreg;
    regressors <- ifelse(!is.null(ellipsis$regressors),ellipsis$regressors,"use");
    initialX <- ellipsis$initialX;

    updateX <- FALSE;
    persistenceX <- transitionX <- NULL;
    occurrence <- "none";
    oesmodel <- "MNN";

# Add all the variables in ellipsis to current environment
    list2env(ellipsis,environment());

    # If a previous model provided as a model, write down the variables
    if(exists("model",inherits=FALSE)){
        if(is.null(model$model)){
            stop("The provided model is not CES.",call.=FALSE);
        }
        else if(smoothType(model)!="CES"){
            stop("The provided model is not CES.",call.=FALSE);
        }
        if(!is.null(model$occurrence)){
            occurrence <- model$occurrence;
        }
        initial <- model$initial;
        a <- model$a;
        b <- model$b;
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
        model <- model$model;
        seasonality <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
    }

##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- environment();
    ssInput("ces",ParentEnvironment=environment());

##### Elements of CES #####
ElementsCES <- function(B){
    vt <- matrix(matvt[1:lagsModelMax,],lagsModelMax);
    nCoefficients <- 0;
    # No seasonality or Simple seasonality, lagged CES
    if(a$estimate){
        matF[1,2] <- B[2]-1;
        matF[2,2] <- 1-B[1];
        vecg[1:2,] <- c(B[1]-B[2],B[1]+B[2]);
        nCoefficients <- nCoefficients + 2;
    }
    else{
        matF[1,2] <- Im(a$value)-1;
        matF[2,2] <- 1-Re(a$value);
        vecg[1:2,] <- c(Re(a$value)-Im(a$value),Re(a$value)+Im(a$value));
    }

    if(seasonality=="p"){
    # Partial seasonality with a real part only
        if(b$estimate){
            vecg[3,] <- B[nCoefficients+1];
            nCoefficients <- nCoefficients + 1;
        }
        else{
            vecg[3,] <- b$value;
        }
    }
    else if(seasonality=="f"){
    # Full seasonality with both real and imaginary parts
        if(b$estimate){
            matF[3,4] <- B[nCoefficients+2]-1;
            matF[4,4] <- 1-B[nCoefficients+1];
            vecg[3:4,] <- c(B[nCoefficients+1]-B[nCoefficients+2],B[nCoefficients+1]+B[nCoefficients+2]);
            nCoefficients <- nCoefficients + 2;
        }
        else{
            matF[3,4] <- Im(b$value)-1;
            matF[4,4] <- 1-Re(b$value);
            vecg[3:4,] <- c(Re(b$value)-Im(b$value),Re(b$value)+Im(b$value));
        }
    }

    if(initialType=="o"){
        if(any(seasonality==c("n","s"))){
            vt[1:lagsModelMax,] <- B[nCoefficients+(1:(2*lagsModelMax))];
            nCoefficients <- nCoefficients + lagsModelMax*2;
        }
        else if(seasonality=="p"){
            vt[,1:2] <- rep(B[nCoefficients+(1:2)],each=lagsModelMax);
            nCoefficients <- nCoefficients + 2;
            vt[1:lagsModelMax,3] <- B[nCoefficients+(1:lagsModelMax)];
            nCoefficients <- nCoefficients + lagsModelMax;
        }
        else if(seasonality=="f"){
            vt[,1:2] <- rep(B[nCoefficients+(1:2)],each=lagsModelMax);
            nCoefficients <- nCoefficients + 2;
            vt[1:lagsModelMax,3:4] <- B[nCoefficients+(1:(lagsModelMax*2))];
            nCoefficients <- nCoefficients + lagsModelMax*2;
        }
    }
    else if(initialType=="b"){
        vt[1:lagsModelMax,] <- matvt[1:lagsModelMax,];
    }
    else{
        vt[1:lagsModelMax,] <- initialValue;
    }

# If exogenous are included
    if(xregEstimate){
        at <- matrix(NA,lagsModelMax,nExovars);
        if(initialXEstimate){
            at[,] <- rep(B[nCoefficients+(1:nExovars)],each=lagsModelMax);
            nCoefficients <- nCoefficients + nExovars;
        }
        else{
            at <- matat[1:lagsModelMax,];
        }
        if(updateX){
            if(FXEstimate){
                matFX <- matrix(B[nCoefficients+(1:(nExovars^2))],nExovars,nExovars);
                nCoefficients <- nCoefficients + nExovars^2;
            }

            if(gXEstimate){
                vecgX <- matrix(B[nCoefficients+(1:nExovars)],nExovars,1);
                nCoefficients <- nCoefficients + nExovars;
            }
        }
    }
    else{
        at <- matrix(matat[1:lagsModelMax,],lagsModelMax,nExovars);
    }

    return(list(matF=matF,vecg=vecg,vt=vt,at=at,matFX=matFX,vecgX=vecgX));
}

##### Cost function for CES #####
CF <- function(B){
# Obtain the elements of CES
    elements <- ElementsCES(B);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:lagsModelMax,] <- elements$vt;
    matat[1:lagsModelMax,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

    cfRes <- costfunc(matvt, matF, matw, yInSample, vecg,
                      h, lagsModel, Etype, Ttype, Stype,
                      multisteps, loss, normalizer, initialType,
                      matxt, matat, matFX, vecgX, ot,
                      bounds, 0);

    if(is.nan(cfRes) | is.na(cfRes)){
        cfRes <- 1e100;
    }
    return(cfRes);
}

##### Estimate ces or just use the provided values #####
CreatorCES <- function(silentText=FALSE,...){
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();

    if(any(initialType=="o",a$estimate,b$estimate,initialXEstimate,FXEstimate,gXEstimate)){
        B <- NULL;
        # If we don't need to estimate a
        if(a$estimate){
            B <- c(1.3,1);
        }

        if(any(seasonality==c("n","s"))){
            if(initialType=="o"){
                B <- c(B,c(matvt[1:lagsModelMax,]));
            }
        }
        else if(seasonality=="p"){
            if(b$estimate){
                B <- c(B,0.1);
            }
            if(initialType=="o"){
                B <- c(B,c(matvt[1,1:2]));
                B <- c(B,c(matvt[1:lagsModelMax,3]));
            }
        }
        else{
            if(b$estimate){
                B <- c(B,1.3,1);
            }
            if(initialType=="o"){
                B <- c(B,c(matvt[1,1:2]));
                B <- c(B,c(matvt[1:lagsModelMax,3:4]));
            }
        }

        if(xregEstimate){
            if(initialXEstimate){
                B <- c(B,matat[lagsModelMax,]);
            }
            if(updateX){
                if(FXEstimate){
                    B <- c(B,c(diag(nExovars)));
                }
                if(gXEstimate){
                    B <- c(B,rep(0,nExovars));
                }
            }
        }

        res <- nloptr(B, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=1000));
        B <- res$solution;

        #In cases of xreg the optimiser sometimes fails to find reasonable parameters
        if(!is.null(xreg)){
            res2 <- nloptr(B, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-8, "maxeval"=5000));
        }
        else{
            res2 <- nloptr(B, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-8, "maxeval"=1000));
        }
            # This condition is needed in order to make sure that we did not make the solution worse
        if(res2$objective <= res$objective){
            res <- res2;
        }

        B <- res$solution;
        cfObjective <- res$objective;

        # Parameters estimated + variance
        nParam <- length(B) + 1;
    }
    else{
        B <- c(a$value,b$value,initialValue,initialX,transitionX,persistenceX);
        cfObjective <- CF(B);

        # Only variance is estimated
        nParam <- 1;
    }

    ICValues <- ICFunction(nParam=nParam,nParamOccurrence=nParamOccurrence,
                           B=B,Etype=Etype);
    ICs <- ICValues$ICs;
    logLik <- ICValues$llikelihood;

    icBest <- ICs[ic];

    return(list(cfObjective=cfObjective,B=B,ICs=ICs,icBest=icBest,nParam=nParam,logLik=logLik));
}

##### Preset yFitted, yForecast, errors and basic parameters #####
    matvt <- matrix(NA,nrow=obsStates,ncol=nComponents);
    yFitted <- rep(NA,obsInSample);
    yForecast <- rep(NA,h);
    errors <- rep(NA,obsInSample);

##### Define parameters for different seasonality types #####
    # Define "w" matrix, seasonal complex smoothing parameter, seasonality lag (if it is present).
    #   matvt - the matrix with the components, lags is the lags used in pFitted matrix.
    if(seasonality=="n"){
        # No seasonality
        matF <- matrix(1,2,2);
        vecg <- matrix(0,2);
        matw <- matrix(c(1,0),1,2);
        matvt <- matrix(NA,obsStates,2);
        colnames(matvt) <- c("level","potential");
        matvt[1,] <- c(mean(yot[1:min(max(10,dataFreq),obsNonzero)]),mean(yot[1:min(max(10,dataFreq),obsNonzero)])/1.1);
    }
    else if(seasonality=="s"){
        # Simple seasonality, lagged CES
        matF <- matrix(1,2,2);
        vecg <- matrix(0,2);
        matw <- matrix(c(1,0),1,2);
        matvt <- matrix(NA,obsStates,2);
        colnames(matvt) <- c("level.s","potential.s");
        matvt[1:lagsModelMax,1] <- yInSample[1:lagsModelMax];
        matvt[1:lagsModelMax,2] <- matvt[1:lagsModelMax,1]/1.1;
    }
    else if(seasonality=="p"){
        # Partial seasonality with a real part only
        matF <- diag(3);
        matF[2,1] <- 1;
        vecg <- matrix(0,3);
        matw <- matrix(c(1,0,1),1,3);
        matvt <- matrix(NA,obsStates,3);
        colnames(matvt) <- c("level","potential","seasonal");
        matvt[1:lagsModelMax,1] <- mean(yInSample[1:lagsModelMax]);
        matvt[1:lagsModelMax,2] <- matvt[1:lagsModelMax,1]/1.1;
        matvt[1:lagsModelMax,3] <- decompose(ts(yInSample,frequency=lagsModelMax),type="additive")$figure;
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
        matvt[1:lagsModelMax,1] <- mean(yInSample[1:lagsModelMax]);
        matvt[1:lagsModelMax,2] <- matvt[1:lagsModelMax,1]/1.1;
        matvt[1:lagsModelMax,3] <- decompose(ts(yInSample,frequency=lagsModelMax),type="additive")$figure;
        matvt[1:lagsModelMax,4] <- matvt[1:lagsModelMax,3]/1.1;
    }

##### Prepare exogenous variables #####
    xregdata <- ssXreg(y=y, xreg=xreg, updateX=FALSE, ot=ot,
                       persistenceX=NULL, transitionX=NULL, initialX=initialX,
                       obsInSample=obsInSample, obsAll=obsAll, obsStates=obsStates,
                       lagsModelMax=lagsModelMax, h=h, regressors=regressors, silent=silentText);

    if(regressors=="u"){
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
    if(is.null(xreg)){
        regressors <- "u";
    }

    # These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

    # Check number of parameters vs data
    nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) + initialXEstimate*ncol(matat);
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
        if(regressors=="select"){
            if(obsNonzero <= (nParamMax - nParamExo)){
                warning(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                            nParamMax," while the number of observations is ",obsNonzero - nParamExo,"!"),call.=FALSE);
                tinySample <- TRUE;
            }
            else{
                warning(paste0("The potential number of exogenous variables is higher than the number of observations. ",
                               "This may cause problems in the estimation."),call.=FALSE);
            }
        }
        else{
            warning(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
                           nParamMax," while the number of observations is ",obsNonzero,"!"),call.=FALSE);
            tinySample <- TRUE;
        }
    }
    else{
        tinySample <- FALSE;
    }

# If this is tiny sample, use SES instead
    if(tinySample){
        warning("Not enough observations to fit CES. Switching to ETS(A,N,N).",call.=FALSE);
        return(es(y,"ANN",initial=initial,loss=loss,
                  h=h,holdout=holdout,cumulative=cumulative,
                  interval=interval,level=level,
                  occurrence=occurrence,
                  oesmodel=oesmodel,
                  bounds="u",
                  silent=silent,
                  xreg=xreg,regressors=regressors,initialX=initialX,
                  updateX=updateX,persistenceX=persistenceX,transitionX=transitionX));
    }

##### Start doing things #####
    environment(intermittentParametersSetter) <- environment();
    environment(intermittentMaker) <- environment();
    environment(ssForecaster) <- environment();
    environment(ssFitter) <- environment();

##### If occurrence=="a", run a loop and select the best one #####
    if(occurrence=="a"){
        if(!silentText){
            cat("Selecting the best occurrence model...\n");
        }
        # First produce the auto model
        intermittentParametersSetter(occurrence="a",ParentEnvironment=environment());
        intermittentMaker(occurrence="a",ParentEnvironment=environment());
        intermittentModel <- CreatorCES(silent=silentText);
        occurrenceBest <- occurrence;
        occurrenceModelBest <- occurrenceModel;

        if(!silentText){
            cat("Comparing it with the best non-intermittent model...\n");
        }
        # Then fit the model without the occurrence part
        occurrence[] <- "n";
        intermittentParametersSetter(occurrence=occurrence,ParentEnvironment=environment());
        intermittentMaker(occurrence=occurrence,ParentEnvironment=environment());
        nonIntermittentModel <- CreatorCES(silent=silentText);

        # Compare the results and return the best
        if(nonIntermittentModel$icBest[ic] <= intermittentModel$icBest[ic]){
            cesValues <- nonIntermittentModel;
        }
        # If this is the "auto", then use the selected occurrence to reset the parameters
        else{
            cesValues <- intermittentModel;
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

        cesValues <- CreatorCES(silent=silentText);
    }

    list2env(cesValues,environment());

    if(regressors!="u"){
        # Prepare for fitting
        elements <- ElementsCES(B);
        matF <- elements$matF;
        vecg <- elements$vecg;
        matvt[1:lagsModelMax,] <- elements$vt;
        matat[1:lagsModelMax,] <- elements$at;
        matFX <- elements$matFX;
        vecgX <- elements$vecgX;

        # cesValues <- CreatorCES(silentText=TRUE);
        ssFitter(ParentEnvironment=environment());

        xregNames <- colnames(matxtOriginal);
        xregNew <- cbind(errors,xreg[1:nrow(errors),]);
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
        }

        if(!is.null(xreg)){
            cesValues <- CreatorCES(silentText=TRUE);
            list2env(cesValues,environment());
        }
    }

    if(!is.null(xreg)){
        if(ncol(matat)==1){
            colnames(matxt) <- colnames(matat) <- xregNames;
        }
        xreg <- matxt;
        if(regressors=="s"){
            nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecgX) + initialXEstimate*ncol(matat);
            parametersNumber[1,2] <- nParamExo;
        }
    }

# Prepare for fitting
    elements <- ElementsCES(B);
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:lagsModelMax,] <- elements$vt;
    matat[1:lagsModelMax,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

##### Fit simple model and produce forecast #####
    ssFitter(ParentEnvironment=environment());
    ssForecaster(ParentEnvironment=environment());

##### Do final check and make some preparations for output #####

# Write down initials of states vector and exogenous
    if(initialType!="p"){
        initialValue <- matvt[1:lagsModelMax,];
        if(initialType!="b"){
            parametersNumber[1,1] <- (parametersNumber[1,1] + 2*(seasonality!="s") +
                                      lagsModelMax*(seasonality!="n") + lagsModelMax*any(seasonality==c("f","s")));
        }
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

    # Add variance estimation
    parametersNumber[1,1] <- parametersNumber[1,1] + 1;

    # Write down the number of parameters of occurrence
    if(all(occurrence!=c("n","p")) & !occurrenceModelProvided){
        parametersNumber[1,3] <- nparam(occurrenceModel);
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
    nCoefficients <- 0;
    if(a$estimate){
        a$value <- complex(real=B[1],imaginary=B[2]);
        nCoefficients <- 2;
        parametersNumber[1,1] <- parametersNumber[1,1] + 2;
    }

    names(a$value) <- "a0+ia1";

    if(b$estimate){
        if(seasonality=="p"){
            b$value <- B[nCoefficients+1];
            parametersNumber[1,1] <- parametersNumber[1,1] + 1;
        }
        else if(seasonality=="f"){
            b$value <- complex(real=B[nCoefficients+1],imaginary=B[nCoefficients+2]);
            parametersNumber[1,1] <- parametersNumber[1,1] + 2;
        }
    }
    if(b$number!=0){
        if(is.complex(b$value)){
            names(b$value) <- "b0+ib1";
        }
        else{
            names(b$value) <- "b";
        }
    }

    if(!is.null(xreg)){
        modelname <- "CESX";
    }
    else{
        modelname <- "CES";
    }
    modelname <- paste0(modelname,"(",seasonality,")");

    if(all(occurrence!=c("n","none"))){
        modelname <- paste0("i",modelname);
    }

    parametersNumber[1,4] <- sum(parametersNumber[1,1:3]);
    parametersNumber[2,4] <- sum(parametersNumber[2,1:3]);

    # Write down Fisher Information if needed
    if(FI & parametersNumber[1,4]>1){
        environment(likelihoodFunction) <- environment();
        FI <- -numDeriv::hessian(likelihoodFunction,B);
    }
    else{
        FI <- NA;
    }

    ##### Deal with the holdout sample #####
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

    ##### Print output #####
    if(!silentText){
        if(any(abs(eigen(matF - vecg %*% matw, only.values=TRUE)$values)>(1 + 1E-10))){
            if(bounds!="a"){
                warning("Unstable model was estimated! Use bounds='admissible' to address this issue!",call.=FALSE);
            }
            else{
                warning("Something went wrong in optimiser - unstable model was estimated! Please report this error to the maintainer.",
                        call.=FALSE);
            }
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
                       level=level,legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
        else{
            graphmaker(actuals=y,forecast=yForecastNew,fitted=yFitted,
                       legend=!silentLegend,main=modelname,cumulative=cumulative);
        }
    }

    ##### Return values #####
    model <- list(model=modelname,timeElapsed=Sys.time()-startTime,
                  states=matvt,a=a$value,b=b$value,
                  persistence=vecg,transition=matF,
                  measurement=matw,
                  initialType=initialType,initial=initialValue,
                  nParam=parametersNumber,
                  fitted=yFitted,forecast=yForecast,lower=yLower,upper=yUpper,residuals=errors,
                  errors=errors.mat,s2=s2,interval=intervalType,level=level,cumulative=cumulative,
                  y=y,holdout=yHoldout,
                  xreg=xreg,initialX=initialX,
                  ICs=ICs,logLik=logLik,lossValue=cfObjective,loss=loss,FI=FI,accuracy=errormeasures,
                  B=B);
    return(structure(model,class="smooth"));
}
