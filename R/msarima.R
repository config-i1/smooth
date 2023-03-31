utils::globalVariables(c("normalizer","constantValue","constantRequired","constantEstimate","B",
                         "ARValue","ARRequired","AREstimate","MAValue","MARequired","MAEstimate",
                         "yForecastStart","nonZeroARI","nonZeroMA"));

#' @template ssIntervals
#' @rdname msarima
#' @export
msarima_old <- function(y, orders=list(ar=c(0),i=c(1),ma=c(1)), lags=c(1),
                        constant=FALSE, AR=NULL, MA=NULL,
                        initial=c("backcasting","optimal"), ic=c("AICc","AIC","BIC","BICc"),
                        loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                        h=10, holdout=FALSE, cumulative=FALSE,
                        interval=c("none","parametric","likelihood","semiparametric","nonparametric"), level=0.95,
                        bounds=c("admissible","none"),
                        silent=c("all","graph","legend","output","none"),
                        xreg=NULL, regressors=c("use","select"), initialX=NULL, ...){
##### Function constructs SARIMA model (possible triple seasonality) using state space approach
# ar.orders contains vector of seasonal ARs. ar.orders=c(2,1,3) will mean AR(2)*SAR(1)*SAR(3) - model with double seasonality.
#
#    Copyright (C) 2016  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

    ### Depricate the old parameters
    ellipsis <- list(...)
    ellipsis <- depricator(ellipsis, "xregDo", "regressors");

    updateX <- FALSE;
    persistenceX <- transitionX <- NULL;
    occurrence <- "none";
    oesmodel <- "MNN";

# Add all the variables in ellipsis to current environment
    list2env(ellipsis,environment());

    # NLOPTR parameters
    if(is.null(ellipsis$print_level)){
        print_level <- 0;
    }
    if(is.null(ellipsis$maxeval)){
        maxeval <- 1000;
    }

    # If a previous model provided as a model, write down the variables
    if(exists("model",inherits=FALSE)){
        if(is.null(model$model)){
            stop("The provided model is not ARIMA.",call.=FALSE);
        }
        else if(smoothType(model)!="ARIMA"){
            stop("The provided model is not ARIMA.",call.=FALSE);
        }

# If this is a normal ARIMA, do things
        if(any(unlist(gregexpr("combine",model$model))==-1)){
            if(!is.null(model$occurrence)){
                occurrence <- model$occurrence;
            }
            if(!is.null(model$initial)){
                initial <- model$initial;
            }
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
            AR <- model$AR;
            MA <- model$MA;
            constant <- model$constant;
            model <- model$model;
            arimaOrders <- paste0(c("",substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1),"")
                                   ,collapse=";");
            comas <- unlist(gregexpr("\\,",arimaOrders));
            semicolons <- unlist(gregexpr("\\;",arimaOrders));
            ar.orders <- as.numeric(substring(arimaOrders,semicolons[-length(semicolons)]+1,comas[2*(1:(length(comas)/2))-1]-1));
            i.orders <- as.numeric(substring(arimaOrders,comas[2*(1:(length(comas)/2))-1]+1,comas[2*(1:(length(comas)/2))-1]+1));
            ma.orders <- as.numeric(substring(arimaOrders,comas[2*(1:(length(comas)/2))]+1,semicolons[-1]-1));
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
        if(is.list(orders)){
            ar.orders <- orders$ar;
            i.orders <- orders$i;
            ma.orders <- orders$ma;
        }
        else if(is.vector(orders)){
            ar.orders <- orders[1];
            i.orders <- orders[2];
            ma.orders <- orders[3];
            lags <- 1;
        }
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
    ssInput("msarima",ParentEnvironment=environment());

# Cost function for SSARIMA
CF <- function(B){
    cfRes <- costfuncARIMA(ar.orders, ma.orders, i.orders, lags, nComponents,
                           ARValue, MAValue, constantValue, B,
                           matvt, matF, matw, yInSample, vecg,
                           h, lagsModel, Etype, Ttype, Stype,
                           multisteps, loss, normalizer, initialType,
                           nExovars, matxt, matat, matFX, vecgX, ot,
                           AREstimate, MAEstimate, constantRequired, constantEstimate,
                           xregEstimate, updateX, FXEstimate, gXEstimate, initialXEstimate,
                           bounds,
                           # The last bit is "ssarimaOld"
                           FALSE, nonZeroARI, nonZeroMA, 0);

    if(is.nan(cfRes) | is.na(cfRes) | is.infinite(cfRes)){
        cfRes <- 1e+100;
    }

    return(cfRes);
}

##### Estimate ssarima or just use the provided values #####
CreatorSSARIMA <- function(silentText=FALSE,...){
    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();

    # If there is something to optimise, let's do it.
    if(any((initialType=="o"),(AREstimate),(MAEstimate),
           (initialXEstimate),(FXEstimate),(gXEstimate),(constantEstimate))){

        B <- NULL;
        # If there are AR / I / MA components
        if(nComponents > 0){
            # Initialise AR / MA parameters
            if(AREstimate){
                # B <- c(B,c(1:sum(ar.orders))/sum(sum(ar.orders):1));
                # B <- c(B,rep(1/sum(ar.orders),sum(ar.orders)));
                B <- c(B,rep(0.1,sum(ar.orders)));
            }
            if(MAEstimate){
                B <- c(B,rep(0.1,sum(ma.orders)));
                # B <- c(B,rep(1/sum(ma.orders),sum(ma.orders)));
            }

            # initial values of state vector
            if(initialType=="o"){
                B <- c(B,matvt[1:lagsModelMax,nComponents]);
            }
        }

        # The constant
        if(constantEstimate){
            if(all(i.orders==0)){
                B <- c(B,sum(yot)/obsInSample);
            }
            else{
                B <- c(B,sum(diff(yot))/obsInSample);
            }
        }

# initials, transition matrix and persistence vector
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

        # print(lagsModel)
        # print(head(matvt,15))
        # print(nonZeroARI)
        # print(nonZeroMA)
        # print(nComponents)
        # print(B)
        # stop()
# Optimise model. First run
        res <- nloptr(B, CF, opts=list(algorithm="NLOPT_LN_BOBYQA", xtol_rel=1e-8,
                                       maxeval=maxeval, print_level=print_level));
        B <- res$solution;

# Optimise model. Second run
        res2 <- nloptr(B, CF, opts=list(algorithm="NLOPT_LN_NELDERMEAD", xtol_rel=1e-6,
                                        maxeval=ceiling(maxeval/5), print_level=print_level));
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
        B <- NULL;

# initial values of state vector and the constant term
        if(nComponents>0 & initialType=="p"){
            matvt[1:lagsModelMax,1:nComponents] <- initialValue;
        }
        if(constantRequired){
            matvt[1:lagsModelMax,(nComponents+1)] <- constantValue;
        }

        cfObjective <- CF(B);

        # Only variance is estimated
        nParam <- 1;
    }

    ICValues <- ICFunction(nParam=nParam,nParamOccurrence=nParamOccurrence,
                           B=B,Etype=Etype);
    ICs <- ICValues$ICs;
    icBest <- ICs[ic];
    logLik <- ICValues$llikelihood;

    return(list(cfObjective=cfObjective,B=B,ICs=ICs,icBest=icBest,nParam=nParam,logLik=logLik));
}

    # Prepare lists for the polynomials
    P <- list(NA);
    D <- list(NA);
    Q <- list(NA);

##### Preset values of matvt and other matrices ######
    if(nComponents > 0){
        # Transition matrix, measurement vector and persistence vector + state vector
        matF <- matrix(0,nComponents,nComponents);
        matw <- matrix(1,1,nComponents);
        vecg <- matrix(0,nComponents,1);
        matvt <- matrix(NA,obsStates,nComponents);
        if(constantRequired){
            matF <- cbind(rbind(matF,rep(0,nComponents)),rep(1,nComponents+1));
            matw <- cbind(matw,1);
            vecg <- rbind(vecg,0);
            matvt <- cbind(matvt,rep(1,obsStates));
        }
        if(initialType=="p"){
            matvt[1:lagsModelMax,1:nComponents] <- initialValue;
        }
        else{
            for(i in 1:nComponents){
                nRepeats <- ceiling(lagsModelMax/lagsModel[i]);
                matvt[1:lagsModelMax,i] <- rep(yInSample[1:lagsModel[i]],nRepeats)[nRepeats*lagsModel[i]+(-lagsModelMax+1):0];
                # matvt[1:lagsModelMax,i] <- rep(yInSample[1:lagsModel[i]],nRepeats)[1:lagsModelMax];
            }
        }
    }
    else{
        matw <- matF <- matrix(1,1,1);
        vecg <- matrix(0,1,1);
        matvt <- matrix(1,obsStates,1);
        lagsModel <- matrix(1,1,1);
    }

##### Preset yFitted, yForecast, errors and basic parameters #####
    yFitted <- rep(NA,obsInSample);
    yForecast <- rep(NA,h);
    errors <- rep(NA,obsInSample);

##### Prepare exogenous variables #####
    xregdata <- ssXreg(y=y, xreg=xreg, updateX=updateX, ot=ot,
                       persistenceX=persistenceX, transitionX=transitionX, initialX=initialX,
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


# If this is tiny sample, use ARIMA with constant instead
    if(tinySample){
        warning("Not enough observations to fit ARIMA. Switching to ARIMA(0,0,0) with constant.",call.=FALSE);
        return(msarima(y,orders=list(ar=0,i=0,ma=0),lags=1,
                       constant=TRUE,
                       initial=initial,loss=loss,
                       h=h,holdout=holdout,cumulative=cumulative,
                       interval=interval,level=level,
                       occurrence=occurrence,
                       oesmodel=oesmodel,
                       bounds="u",
                       silent=silent,
                       xreg=xreg,regressors=regressors,initialX=initialX,
                       updateX=updateX,persistenceX=persistenceX,transitionX=transitionX));
    }

#####Start the calculations#####
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
        intermittentModel <- CreatorSSARIMA(silent=silentText);
        occurrenceBest <- occurrence;
        occurrenceModelBest <- occurrenceModel;

        if(!silentText){
            cat("Comparing it with the best non-intermittent model...\n");
        }
        # Then fit the model without the occurrence part
        occurrence[] <- "n";
        intermittentParametersSetter(occurrence=occurrence,ParentEnvironment=environment());
        intermittentMaker(occurrence=occurrence,ParentEnvironment=environment());
        nonIntermittentModel <- CreatorSSARIMA(silent=silentText);

        # Compare the results and return the best
        if(nonIntermittentModel$icBest[ic] <= intermittentModel$icBest[ic]){
            ssarimaValues <- nonIntermittentModel;
        }
        # If this is the "auto", then use the selected occurrence to reset the parameters
        else{
            ssarimaValues <- intermittentModel;
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

        # print("Start...");
        ssarimaValues <- CreatorSSARIMA(silent=silentText);
        # print("Optimised!");
    }

    list2env(ssarimaValues,environment());
    # print("list2env");

    if(regressors!="u"){
        # Prepare for fitting
        elements <- polysoswrap(ar.orders, ma.orders, i.orders, lags, nComponents,
                                ARValue, MAValue, constantValue, B,
                                matvt, vecg, matF,
                                initialType, nExovars, matat, matFX, vecgX,
                                AREstimate, MAEstimate, constantRequired, constantEstimate,
                                xregEstimate, updateX, FXEstimate, gXEstimate, initialXEstimate,
                                # The last bit is "ssarimaOld"
                                FALSE, lagsModel, nonZeroARI, nonZeroMA);
        matF <- elements$matF;
        vecg <- elements$vecg;
        matvt[,] <- elements$matvt;
        matat[,] <- elements$matat;
        matFX <- elements$matFX;
        vecgX <- elements$vecgX;
        polysos.ar <- elements$arPolynomial;
        polysos.ma <- elements$maPolynomial;

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
            ssarimaValues <- CreatorSSARIMA(silentText);
            list2env(ssarimaValues,environment());
        }
    }
    # print("regressors");

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
    # print("xreg");
# Prepare for fitting
    elements <- polysoswrap(ar.orders, ma.orders, i.orders, lags, nComponents,
                            ARValue, MAValue, constantValue, B,
                            matvt, vecg, matF,
                            initialType, nExovars, matat, matFX, vecgX,
                            AREstimate, MAEstimate, constantRequired, constantEstimate,
                            xregEstimate, updateX, FXEstimate, gXEstimate, initialXEstimate,
                            # The last bit is "ssarimaOld"
                            FALSE, lagsModel, nonZeroARI, nonZeroMA);
    # print("polysos");
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[,] <- elements$matvt;
    matat[,] <- elements$matat;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;
    polysos.ar <- elements$arPolynomial;
    polysos.ma <- elements$maPolynomial;

    nComponents <- nComponents + constantRequired;

##### Fit simple model and produce forecast #####
    # print("fit!");
    ssFitter(ParentEnvironment=environment());
    # print("fitted!");
    ssForecaster(ParentEnvironment=environment());
    # print("forecasted");

##### Do final check and make some preparations for output #####

# Write down initials of states vector and exogenous
    if(initialType!="p"){
        if(constantRequired){
            initialValue <- matvt[1:lagsModelMax,-ncol(matvt)];
        }
        else{
            initialValue <- matvt[1:lagsModelMax,];
        }
        if(initialType!="b"){
            parametersNumber[1,1] <- parametersNumber[1,1] + nComponents;
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

    # Write down the number of parameters of occurrence model
    if(all(occurrence!=c("n","p")) & !occurrenceModelProvided){
        parametersNumber[1,3] <- nparam(occurrenceModel);
    }

# Fill in the rest of matvt
    matvt <- ts(matvt,start=(time(y)[1] - deltat(y)*lagsModelMax),frequency=dataFreq);
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

    nCoef <- arCoef <- maCoef <- 0;
    arIndex <- maIndex <- 1;
    for(i in 1:length(ar.orders)){
        if(ar.orders[i]!=0){
            if(AREstimate){
                ARterms[1:ar.orders[i],arIndex] <- B[nCoef+(1:ar.orders[i])];
                names(B)[nCoef+(1:ar.orders[i])] <- paste0("AR(",1:ar.orders[i],"), ",colnames(ARterms)[arIndex]);
                nCoef <- nCoef + ar.orders[i];
                parametersNumber[1,1] <- parametersNumber[1,1] + ar.orders[i];
            }
            else{
                ARterms[1:ar.orders[i],arIndex] <- ARValue[arCoef+(1:ar.orders[i])];
                arCoef <- arCoef + ar.orders[i];
            }
            arIndex <- arIndex + 1;
        }
        if(ma.orders[i]!=0){
            if(MAEstimate){
                MAterms[1:ma.orders[i],maIndex] <- B[nCoef+(1:ma.orders[i])];
                names(B)[nCoef+(1:ma.orders[i])] <- paste0("MA(",1:ma.orders[i],"), ",colnames(MAterms)[maIndex]);
                nCoef <- nCoef + ma.orders[i];
                parametersNumber[1,1] <- parametersNumber[1,1] + ma.orders[i];
            }
            else{
                MAterms[1:ma.orders[i],maIndex] <- MAValue[maCoef+(1:ma.orders[i])];
                maCoef <- maCoef + ma.orders[i];
            }
            maIndex <- maIndex + 1;
        }
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
    if(all(occurrence!=c("n","none"))){
        modelname <- paste0("i",modelname);
    }

    if(constantRequired){
        if(constantEstimate){
            constantValue <- matvt[1,nComponents];
            parametersNumber[1,1] <- parametersNumber[1,1] + 1;
            if(!is.null(names(B))){
                names(B)[is.na(names(B))][1] <- "Constant";
            }
            else{
                names(B)[1] <- "Constant";
            }
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

    parametersNumber[1,4] <- sum(parametersNumber[1,1:3]);
    parametersNumber[2,4] <- sum(parametersNumber[2,1:3]);

    # Write down Fisher Information if needed
    if(FI & parametersNumber[1,4]>1){
        environment(likelihoodFunction) <- environment();
        FI <- -numDeriv::hessian(likelihoodFunction,B);
        rownames(FI) <- colnames(FI) <- names(B);
        if(initialType=="o"){
            # Leave only AR and MA parameters. Forget about the initials
            FI <- FI[!is.na(rownames(FI)),!is.na(colnames(FI))];
        }
    }
    else{
        FI <- NA;
    }
    # print("holdout");

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

    # print("Yes!");
##### Return values #####
    model <- list(model=modelname,timeElapsed=Sys.time()-startTime,
                  states=matvt,transition=matF,persistence=vecg,
                  measurement=matw,
                  AR=ARterms,I=Iterms,MA=MAterms,constant=const,
                  initialType=initialType,initial=initialValue,
                  nParam=parametersNumber, modelLags=lagsModel,
                  fitted=yFitted,forecast=yForecast,lower=yLower,upper=yUpper,residuals=errors,
                  errors=errors.mat,s2=s2,interval=intervalType,level=level,cumulative=cumulative,
                  y=y,holdout=yHoldout,
                  xreg=xreg,initialX=initialX,
                  ICs=ICs,logLik=logLik,lossValue=cfObjective,loss=loss,FI=FI,accuracy=errormeasures,
                  B=B);
    return(structure(model,class=c("smooth","msarima")));
}
