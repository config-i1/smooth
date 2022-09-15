utils::globalVariables(c("adamFitted","algorithm","arEstimate","arOrders","arRequired","arimaModel",
                         "arimaPolynomials","armaParameters","componentsNamesARIMA","componentsNamesETS",
                         "componentsNumberARIMA","componentsNumberETS","componentsNumberETSNonSeasonal",
                         "componentsNumberETSSeasonal","digits","etsModel","ftol_abs","ftol_rel",
                         "horizon","iOrders","iRequired","initialArima","initialArimaEstimate",
                         "initialArimaNumber","initialLevel","initialLevelEstimate","initialSeasonal",
                         "initialSeasonalEstimate","initialTrend","initialTrendEstimate","lagsModelARIMA",
                         "lagsModelAll","lagsModelSeasonal","profilesObservedTable","profilesRecentTable",
                         "other","otherParameterEstimate","lambda","lossFunction",
                         "maEstimate","maOrders","maRequired","matVt","matWt","maxtime","modelIsTrendy",
                         "nParamEstimated","persistenceLevel","persistenceLevelEstimate",
                         "persistenceSeasonal","persistenceSeasonalEstimate","persistenceTrend",
                         "persistenceTrendEstimate","vecG","xtol_abs","xtol_rel","stepSize","yClasses",
                         "yForecastIndex","yInSampleIndex","yIndexAll","yNAValues","yStart","responseName",
                         "xregParametersMissing","xregParametersIncluded","xregParametersEstimated",
                         "xregParametersPersistence","xregModelInitials","constantName","yDenominator",
                         "damped","dataStart","initialEstimate","initialSeasonEstimate","maxeval","icFunction",
                         "modelIsMultiplicative","modelIsSeasonal","nComponentsAll","nComponentsNonSeasonal"));

# @export
ces_new <- function(y, seasonality=c("none","simple","partial","full"), lags=c(frequency(data)),
                    initial=c("backcasting","optimal"), a=NULL, b=NULL, ic=c("AICc","AIC","BIC","BICc"),
                    loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                    h=10, holdout=FALSE,
                    bounds=c("admissible","none"),
                    silent=c("all","graph","legend","output","none"), ...){
# Function estimates CES in state space form with sigma = error
# and returns complex smoothing parameter value, fitted values,
# residuals, point and interval forecasts, matrix of CES components and values of
# information criteria.
#
#    Copyright (C) 2015 - 2022  Ivan Svetunkov
#
# Since 2022, the function does not support explanatory variables. But who cares anyway?!


# Start measuring the time of calculations
    startTime <- Sys.time();
    cl <- match.call();
    ellipsis <- list(...);

    # Check seasonality and loss
    seasonality <- match.arg(seasonality);
    loss <- match.arg(loss);

    # If a previous model provided as a model, write down the variables
    if(!is.null(ellipsis$model)){
        if(is.null(ellipsis$model$model)){
            stop("The provided model is not CES.",call.=FALSE);
        }
        else if(smoothType(ellipsis$model)!="CES"){
            stop("The provided model is not CES.",call.=FALSE);
        }
        initial <- ellipsis$model$initial;
        a <- ellipsis$model$a;
        b <- ellipsis$model$b;
        model <- ellipsis$model$model;
        seasonality <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
        modelDo <- "use";
    }
    else{
        modelDo <- "";
    }

    a <- list(value=a);
    b <- list(value=b);

    if(is.null(a$value)){
        a$estimate <- TRUE;
    }
    else{
        a$estimate <- FALSE;
    }
    if(all(is.null(b$value),any(seasonality==c("partial","full")))){
        b$estimate <- TRUE;
    }
    else{
        b$estimate <- FALSE;
    }

    # Make it look like ANN or ANA (to get correct lagsModelAll)
    model <- switch(seasonality,
                    "none"="ANN",
                    "ANA");

    if(any(!is.character(initial))){
        warning("Predefined initials are not supported in CES. Changing to \"backcasting\"",
                call.=FALSE);
        initial <- "backcasting";
    }

    ##### Set environment for ssInput and make all the checks #####
    checkerReturn <- parametersChecker(data=y, model, lags, formulaToUse=NULL, orders=NULL, constant=FALSE, arma=NULL,
                                       outliers="ignore", level=0.99,
                                       persistence=NULL, phi=NULL, initial,
                                       distribution="dnorm", loss, h, holdout, occurrence="none", ic, bounds=bounds[1],
                                       regressors="use", yName="y",
                                       silent, modelDo, ParentEnvironment=environment(), ellipsis, fast=FALSE);

    ##### Elements of CES #####
    filler <- function(B, matVt, matF, vecG, a, b){
        vt <- matVt[,1:lagsModelMax,drop=FALSE];
        nCoefficients <- 0;
        # No seasonality or Simple seasonality, lagged CES
        if(a$estimate){
            matF[1,2] <- B[2]-1;
            matF[2,2] <- 1-B[1];
            vecG[1:2,] <- c(B[1]-B[2],B[1]+B[2]);
            nCoefficients <- nCoefficients + 2;
        }
        else{
            matF[1,2] <- Im(a$value)-1;
            matF[2,2] <- 1-Re(a$value);
            vecG[1:2,] <- c(Re(a$value)-Im(a$value),Re(a$value)+Im(a$value));
        }

        if(seasonality=="p"){
            # Partial seasonality with a real part only
            if(b$estimate){
                vecG[3,] <- B[nCoefficients+1];
                nCoefficients <- nCoefficients + 1;
            }
            else{
                vecG[3,] <- b$value;
            }
        }
        else if(seasonality=="f"){
            # Full seasonality with both real and imaginary parts
            if(b$estimate){
                matF[3,4] <- B[nCoefficients+2]-1;
                matF[4,4] <- 1-B[nCoefficients+1];
                vecG[3:4,] <- c(B[nCoefficients+1]-B[nCoefficients+2],B[nCoefficients+1]+B[nCoefficients+2]);
                nCoefficients <- nCoefficients + 2;
            }
            else{
                matF[3,4] <- Im(b$value)-1;
                matF[4,4] <- 1-Re(b$value);
                vecG[3:4,] <- c(Re(b$value)-Im(b$value),Re(b$value)+Im(b$value));
            }
        }

        j <- 0;
        if(initialType=="o"){
            if(any(seasonality==c("n","s"))){
                vt[1:2,1:lagsModelMax] <- B[nCoefficients+(1:(2*lagsModelMax))];
                nCoefficients <- nCoefficients + lagsModelMax*2;
                j <- j+2;
            }
            else if(seasonality=="p"){
                vt[1:2,] <- B[nCoefficients+(1:2)];
                nCoefficients <- nCoefficients + 2;
                vt[3,1:lagsModelMax] <- B[nCoefficients+(1:lagsModelMax)];
                nCoefficients <- nCoefficients + lagsModelMax;
                j <- j+3;
            }
            else if(seasonality=="f"){
                vt[1:2,] <- B[nCoefficients+(1:2)];
                nCoefficients <- nCoefficients + 2;
                vt[3:4,1:lagsModelMax] <- B[nCoefficients+(1:(lagsModelMax*2))];
                nCoefficients <- nCoefficients + lagsModelMax*2;
                j <- j+4;
            }
        }
        else if(initialType=="p"){
            vt[,1:lagsModelMax] <- initialValue;
        }

        # If exogenous are included
        if(xregEstimate){
            if(initialXEstimate){
                vt[j+(1:xregNumber),] <- B[nCoefficients+(1:xregNumber)];
                nCoefficients <- nCoefficients + xregNumber;
            }
        }

        return(list(matF=matF,vecG=vecG,vt=vt));
    }

    ##### Cost function for CES #####
    CF <- function(B, matVt, matF, vecG, a, b){
        # Obtain the elements of CES
        elements <- filler(B, matVt, matF, vecG, a, b);
        matVt[,1:lagsModelMax] <- elements$vt;

        if(xregModel){
            # We drop the X parts from matrices
            indices <- c(1:componentsNumber)
            eigenValues <- abs(eigen(elements$matF[indices,indices,drop=FALSE] -
                                         elements$vecG[indices,,drop=FALSE] %*%
                                         matWt[obsInSample,indices,drop=FALSE],
                                     symmetric=FALSE, only.values=TRUE)$values);
        }
        else{
            eigenValues <- abs(eigen(elements$matF -
                                         elements$vecG %*% matWt[obsInSample,,drop=FALSE],
                                     symmetric=FALSE, only.values=TRUE)$values);
        }
        if(any(eigenValues>1+1E-50)){
            return(1E+100*max(eigenValues));
        }

        # Write down the initials in the recent profile
        profilesRecentTable[] <- elements$matVt[,1:lagsModelMax];

        adamFitted <- adamFitterWrap(matVt, matWt, elements$matF, elements$vecG,
                                     lagsModelAll, profilesObservedTable, profilesRecentTable,
                                     Etype, Ttype, Stype, componentsNumber, 0,
                                     0, xregNumber, FALSE,
                                     yInSample, ot, initialType=="backcasting");

        if(!multisteps){
            if(loss=="likelihood"){
                # Scale for different functions
                scale <- scaler(distribution="dnorm", Etype, adamFitted$errors[otLogical],
                                adamFitted$yFitted[otLogical], obsInSample, other);

                # Calculate the likelihood
                CFValue <- -sum(dnorm(x=yInSample[otLogical],
                                      mean=adamFitted$yFitted[otLogical],
                                      sd=scale, log=TRUE));
            }
            else if(loss=="MSE"){
                CFValue <- sum(adamFitted$errors^2)/obsInSample;
            }
            else if(loss=="MAE"){
                CFValue <- sum(abs(adamFitted$errors))/obsInSample;
            }
            else if(loss=="HAM"){
                CFValue <- sum(sqrt(abs(adamFitted$errors)))/obsInSample;
            }
            else if(loss=="custom"){
                CFValue <- lossFunction(actual=yInSample,fitted=adamFitted$yFitted,B=B);
            }
        }
        else{
            # Call for the Rcpp function to produce a matrix of multistep errors
            adamErrors <- adamErrorerWrap(adamFitted$matVt, adamElements$matWt, adamElements$matF,
                                          lagsModelAll, profilesObservedTable, profilesRecentTable,
                                          Etype, Ttype, Stype,
                                          componentsNumberETS, componentsNumberETSSeasonal,
                                          componentsNumberARIMA, xregNumber, constantRequired, h,
                                          yInSample, ot);

            # Not done yet: "aMSEh","aTMSE","aGTMSE","aMSCE","aGPL"
            CFValue <- switch(loss,
                              "MSEh"=sum(adamErrors[,h]^2)/(obsInSample-h),
                              "TMSE"=sum(colSums(adamErrors^2)/(obsInSample-h)),
                              "GTMSE"=sum(log(colSums(adamErrors^2)/(obsInSample-h))),
                              "MSCE"=sum(rowSums(adamErrors)^2)/(obsInSample-h),
                              "MAEh"=sum(abs(adamErrors[,h]))/(obsInSample-h),
                              "TMAE"=sum(colSums(abs(adamErrors))/(obsInSample-h)),
                              "GTMAE"=sum(log(colSums(abs(adamErrors))/(obsInSample-h))),
                              "MACE"=sum(abs(rowSums(adamErrors)))/(obsInSample-h),
                              "HAMh"=sum(sqrt(abs(adamErrors[,h])))/(obsInSample-h),
                              "THAM"=sum(colSums(sqrt(abs(adamErrors)))/(obsInSample-h)),
                              "GTHAM"=sum(log(colSums(sqrt(abs(adamErrors)))/(obsInSample-h))),
                              "CHAM"=sum(sqrt(abs(rowSums(adamErrors))))/(obsInSample-h),
                              "GPL"=log(det(t(adamErrors) %*% adamErrors/(obsInSample-h))),
                              0);
        }

        if(is.na(CFValue) || is.nan(CFValue)){
            CFValue[] <- 1e+300;
        }

        return(cfRes);
    }

    # Create all the necessary matrices and vectors
    componentsNumber <- switch(seasonality,
                               "none"=,
                               "simple"=2,
                               "partial"=3,
                               "full"=4);

    lagsModelAll <- switch(seasonality,
                               "none"=,
                               "simple"=c(1,1),
                               "partial"=c(1,1,lagsModelMax),
                               "full"=c(1,1,lagsModelMax,lagsModelMax));

    # Create ADAM profiles for correct treatment of seasonality
    adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                       lags=lags, yIndex=yIndexAll, yClasses=yClasses);
    profilesRecentTable <- adamProfiles$recent;
    profilesObservedTable <- adamProfiles$observed;

    matF <- diag(componentsNumber+xregNumber);
    vecG <- matrix(0,componentsNumber+xregNumber,1);
    matWt <- matrix(1, obsInSample, componentsNumber+xregNumber);
    matWt[,2] <- 0;
    matVt <- matrix(0, componentsNumber+xregNumber, obsStates);
    # Fix matrices for special cases
    if(seasonality=="full"){
        matF[2,1] <- 1;
        matF[4,3] <- 1;
        matWt[,4] <- 0;
        rownames(matVt) <- c("level","potential","seasonal 1", "seasonal 2");
        matVt[1,1:lagsModelMax] <- mean(yInSample[1:lagsModelMax]);
        matVt[2,1:lagsModelMax] <- matVt[1:lagsModelMax,1]/1.1;
        matVt[3,1:lagsModelMax] <- decompose(ts(yInSample,frequency=lagsModelMax),type="additive")$figure;
        matVt[4,1:lagsModelMax] <- matVt[1:lagsModelMax,3]/1.1;
    }
    else if(seasonality=="partial"){
        matF[2,1] <- 1;
        rownames(matVt) <- c("level","potential","seasonal");
        matVt[1,1:lagsModelMax] <- mean(yInSample[1:lagsModelMax]);
        matVt[2,1:lagsModelMax] <- matVt[1:lagsModelMax,1]/1.1;
        matVt[3,1:lagsModelMax] <- decompose(ts(yInSample,frequency=lagsModelMax),type="additive")$figure;
    }
    else if(seasonality=="simple"){
        rownames(matVt) <- c("level.s","potential.s");
        matVt[1,1:lagsModelMax] <- yInSample[1:lagsModelMax];
        matVt[2,1:lagsModelMax] <- matVt[1:lagsModelMax,1]/1.1;
    }
    else{
        rownames(matVt) <- c("level","potential");
        matVt[,1] <- c(mean(yInSample[1:min(max(10,lagsModelMax),obsInSample)]),
                       mean(yInSample[1:min(max(10,lagsModelMax),obsInSample)])/1.1);
    }

    ##### Pre-set yFitted, yForecast, errors and basic parameters #####
    yFitted <- rep(NA, obsInSample);
    yForecast <- rep(NA, h);
    errors <- rep(NA, obsInSample);

    # Values for xreg. No longer supported in ces()
    parametersNumber[1,2] <- 0;
    parametersNumber[2,2] <- 0;

    # Values for occurrence. No longer supported in ces()
    parametersNumber[1,3] <- 0;
    parametersNumber[2,3] <- 0;

    ##### Check number of observations vs number of max parameters #####
    # if(obsNonzero <= nParamMax){
    #     if(regressors=="select"){
    #         if(obsNonzero <= (nParamMax - nParamExo)){
    #             warning(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
    #                         nParamMax," while the number of observations is ",obsNonzero - nParamExo,"!"),call.=FALSE);
    #             tinySample <- TRUE;
    #         }
    #         else{
    #             warning(paste0("The potential number of exogenous variables is higher than the number of observations. ",
    #                            "This may cause problems in the estimation."),call.=FALSE);
    #         }
    #     }
    #     else{
    #         warning(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
    #                        nParamMax," while the number of observations is ",obsNonzero,"!"),call.=FALSE);
    #         tinySample <- TRUE;
    #     }
    # }
    # else{
    #     tinySample <- FALSE;
    # }

    # If this is tiny sample, use SES instead
    # if(tinySample){
    #     warning("Not enough observations to fit CES. Switching to ETS(A,N,N).",call.=FALSE);
    #     return(es(y,"ANN",initial=initial,loss=loss,
    #               h=h,holdout=holdout,cumulative=cumulative,
    #               interval=interval,level=level,
    #               occurrence=occurrence,
    #               oesmodel=oesmodel,
    #               bounds="u",
    #               silent=silent,
    #               xreg=xreg,regressors=regressors,initialX=initialX,
    #               updateX=updateX,persistenceX=persistenceX,transitionX=transitionX));
    # }

##### Estimate ces or just use the provided values #####
    # Initialisation before the optimiser
    if(any(initialType=="optimal",a$estimate,b$estimate)){
        B <- NULL;
        # If we don't need to estimate a
        if(a$estimate){
            B <- c(1.3,1);
        }

        # Index for states
        j <- 0
        if(any(seasonality==c("n","s"))){
            if(initialType=="optimal"){
                B <- c(B,c(matVt[1:2,1:lagsModelMax]));
                j <- 2;
            }
        }
        else if(seasonality=="p"){
            if(b$estimate){
                B <- c(B,0.1);
            }
            if(initialType=="optimal"){
                B <- c(B,c(matVt[1:2,1]));
                B <- c(B,c(matVt[3,1:lagsModelMax]));
                j <- 3;
            }
        }
        else{
            if(b$estimate){
                B <- c(B,1.3,1);
            }
            if(initialType=="optimal"){
                B <- c(B,c(matVt[1:2,1]));
                B <- c(B,c(matVt[3:4,1:lagsModelMax]));
                j <- 4;
            }
        }

        res <- suppressWarnings(nloptr(B, CF,
                                       opts=list(algorithm=algorithm, xtol_rel=xtol_rel, xtol_abs=xtol_abs,
                                                 ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                                 maxeval=maxevalUsed, maxtime=maxtime, print_level=print_level),
                                       matVt=matVt, matF=matF, vecG=vecG, a=a, b=b));

        B <- res$solution;
        cfObjective <- res$objective;

        # Parameters estimated + variance
        nParam <- length(B) + (loss=="likelihood")*1;
    }
    else{
        B <- c(a$value,b$value,initialValue);
        cfObjective <- CF(B, matVt, matF, vecG, a, b);

        # Only variance is estimated
        nParam <- 1;
    }

    ICValues <- ICFunction(nParam=nParam,nParamOccurrence=nParamOccurrence,
                           B=B,Etype=Etype);
    ICs <- ICValues$ICs;
    logLik <- ICValues$llikelihood;

    icBest <- ICs[ic];

#     return(list(cfObjective=cfObjective,B=B,ICs=ICs,icBest=icBest,nParam=nParam,logLik=logLik));
# }

    list2env(cesValues,environment());

#     if(regressors!="u"){
#         # Prepare for fitting
#         elements <- filler(B, matVt, matF, vecG, a, b);
#         matF <- elements$matF;
#         vecG <- elements$vecG;
#         matVt[1:lagsModelMax,] <- elements$vt;
#         matat[1:lagsModelMax,] <- elements$at;
#         matFX <- elements$matFX;
#         vecGX <- elements$vecGX;
#
#         # cesValues <- CreatorCES(silentText=TRUE);
#         ssFitter(ParentEnvironment=environment());
#
#         xregNames <- colnames(matxtOriginal);
#         xregNew <- cbind(errors,xreg[1:nrow(errors),]);
#         colnames(xregNew)[1] <- "errors";
#         colnames(xregNew)[-1] <- xregNames;
#         xregNew <- as.data.frame(xregNew);
#         xregResults <- stepwise(xregNew, ic=ic, silent=TRUE, df=nParam+nParamOccurrence-1);
#         xregNames <- names(coef(xregResults))[-1];
#         xregNumber <- length(xregNames);
#         if(xregNumber>0){
#             xregEstimate <- TRUE;
#             matxt <- as.data.frame(matxtOriginal)[,xregNames];
#             matat <- as.data.frame(matatOriginal)[,xregNames];
#             matFX <- diag(xregNumber);
#             vecGX <- matrix(0,xregNumber,1);
#
#             if(xregNumber==1){
#                 matxt <- matrix(matxt,ncol=1);
#                 matat <- matrix(matat,ncol=1);
#                 colnames(matxt) <- colnames(matat) <- xregNames;
#             }
#             else{
#                 matxt <- as.matrix(matxt);
#                 matat <- as.matrix(matat);
#             }
#         }
#         else{
#             xregNumber <- 1;
#             xreg <- NULL;
#         }
#
#         if(!is.null(xreg)){
#             cesValues <- CreatorCES(silentText=TRUE);
#             list2env(cesValues,environment());
#         }
#     }
#
#     if(!is.null(xreg)){
#         if(ncol(matat)==1){
#             colnames(matxt) <- colnames(matat) <- xregNames;
#         }
#         xreg <- matxt;
#         if(regressors=="s"){
#             nParamExo <- FXEstimate*length(matFX) + gXEstimate*nrow(vecGX) + initialXEstimate*ncol(matat);
#             parametersNumber[1,2] <- nParamExo;
#         }
#     }

# Prepare for fitting
    elements <- filler(B, matVt, matF, vecG, a, b);
    matF <- elements$matF;
    vecG <- elements$vecG;
    matVt[1:lagsModelMax,] <- elements$vt;
    matat[1:lagsModelMax,] <- elements$at;
    matFX <- elements$matFX;
    vecGX <- elements$vecGX;

##### Fit simple model and produce forecast #####
    ssFitter(ParentEnvironment=environment());
    ssForecaster(ParentEnvironment=environment());

##### Do final check and make some preparations for output #####

# Write down initials of states vector and exogenous
    if(initialType!="p"){
        initialValue <- matVt[1:lagsModelMax,];
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
        persistenceX <- vecGX;
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
        statenames <- c(colnames(matVt),colnames(matat));
        matVt <- cbind(matVt,matat);
        colnames(matVt) <- statenames;
        if(updateX){
            rownames(vecGX) <- xregNames;
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

    modelname <- "CES";
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
        if(any(abs(eigen(matF - vecG %*% matWt, only.values=TRUE)$values)>(1 + 1E-10))){
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
                  states=matVt,a=a$value,b=b$value,
                  persistence=vecG,transition=matF,
                  measurement=matWt,
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
