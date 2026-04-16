#### Internal helper functions shared across ADAM-family models ####
# These are extracted from parametersChecker() to allow reuse by om() and
# other future functions without duplicating logic.

#### Data preparation ####
#' @keywords internal
adam_checkData <- function(data, lags, h, holdout, yName, modelDo, formulaToUse) {
    responseName <- yName;

    # Extract from sim/Mdata objects
    if(is.adam.sim(data) || is.smooth.sim(data)){
        data <- data$data;
        lags <- frequency(data);
    }
    else if(inherits(data,"Mdata")){
        h <- data$h;
        holdout <- TRUE;
        if(modelDo!="use"){
            lags <- frequency(data$x);
        }
        data <- ts(c(data$x,data$xx),start=start(data$x),frequency=frequency(data$x));
    }

    # Extract index
    ### tsibble has its own index function, so shit happens because of it...
    if(inherits(data,"tbl_ts")){
        yIndex <- data[[1]];
        if(any(duplicated(yIndex))){
            warning(paste0("You have duplicated time stamps in the variable ",yName,
                           ". I will refactor this."),call.=FALSE);
            yIndex <- yIndex[1] + c(1:length(data[[1]])) * diff(tail(yIndex,2));
        }
    }
    else{
        yIndex <- try(time(data),silent=TRUE);
        if(inherits(yIndex,"try-error")){
            if(!is.data.frame(data) && !is.null(dim(data))){
                yIndex <- as.POSIXct(rownames(data));
            }
            else if(is.data.frame(data)){
                yIndex <- c(1:nrow(data));
            }
            else{
                yIndex <- c(1:length(data));
            }
        }
    }
    yClasses <- class(data);

    # Multi-column data: extract response and xreg
    if(!is.null(ncol(data)) && ncol(data)>1){
        xregData <- data;
        if(inherits(data,"tbl_df") || inherits(data,"tbl")){
            data <- as.data.frame(data);
        }
        if(!is.null(formulaToUse)){
            responseName <- all.vars(formulaToUse)[1];
            y <- data[,responseName];
        }
        else{
            responseName <- colnames(xregData)[1];
            if(inherits(data,"tbl_ts")){
                y <- data$value;
            }
            else if(inherits(data,"data.table") || inherits(data,"data.frame")){
                y <- data[[1]];
            }
            else if(inherits(data,"zoo")){
                if(ncol(data)>1){
                    xregData <- as.data.frame(data);
                }
                y <- zoo(data[,1],order.by=time(data));
            }
            else{
                y <- data[,1];
            }
        }
        yIndex <- try(time(y),silent=TRUE);
        if(inherits(yIndex,"try-error")){
            if(!is.null(dim(data))){
                yIndex <- try(as.POSIXct(rownames(data)),silent=TRUE);
                if(inherits(yIndex,"try-error")){
                    yIndex <- c(1:nrow(data));
                }
            }
            else{
                yIndex <- c(1:length(y));
            }
        }
        else{
            yClasses <- class(y);
        }
    }
    else{
        xregData <- NULL;
        if(!is.null(ncol(data)) && !is.null(colnames(data)[1])){
            responseName <- colnames(data)[1];
            y <- data[,1];
        }
        else{
            y <- data;
        }
    }

    responseName <- make.names(responseName);

    obsAll <- length(y) + (1 - holdout)*h;
    obsInSample <- length(y) - holdout*h;

    if(obsInSample<=0){
        stop("The number of in-sample observations is not positive. Cannot do anything.",
             call.=FALSE);
    }

    # Interpolate NAs using fourier + polynomials
    yNAValues <- is.na(y);
    if(any(yNAValues)){
        warning("Data contains NAs. The values will be ignored during the model construction.",
                call.=FALSE);
        X <- cbind(1,poly(c(1:obsAll),degree=min(max(trunc(obsAll/10),1),5)),
                   sinpi(matrix(c(1:obsAll)*rep(c(1:max(max(lags),10)),each=obsAll)/max(max(lags),10),
                                ncol=max(max(lags),10))));
        if(any(y[!yNAValues]<=0)){
            lmFit <- .lm.fit(X[!yNAValues,,drop=FALSE], matrix(y[!yNAValues],ncol=1));
            y[yNAValues] <- (X %*% coef(lmFit))[yNAValues];
        }
        else{
            lmFit <- .lm.fit(X[!yNAValues,,drop=FALSE], matrix(log(y[!yNAValues]),ncol=1));
            y[yNAValues] <- exp(X %*% coef(lmFit))[yNAValues];
        }
        if(!is.null(xregData)){
            xregData[yNAValues,responseName] <- y[yNAValues];
        }
        rm(X);
        if(obsInSample>10000){
            gc(verbose=FALSE);
        }
    }

    # Determine ts class
    if(all(yClasses=="integer") || all(yClasses=="numeric") ||
       all(yClasses=="data.frame") || all(yClasses=="matrix")){
        if(any(class(yIndex) %in% c("POSIXct","Date"))){
            yClasses <- "zoo";
        }
        else{
            yClasses <- "ts";
        }
    }
    yFrequency <- frequency(y);
    yStart <- yIndex[1];
    yInSample <- matrix(y[1:obsInSample],ncol=1);
    if(holdout){
        yForecastStart <- yIndex[obsInSample+1];
        yHoldout <- matrix(y[-c(1:obsInSample)],ncol=1);
        yForecastIndex <- yIndex[-c(1:obsInSample)];
        yInSampleIndex <- yIndex[c(1:obsInSample)];
        yIndexAll <- yIndex;
    }
    else{
        yInSampleIndex <- yIndex;
        if(any(yClasses=="ts")){
            yIndexDiff <- deltat(yIndex);
            yForecastIndex <- yIndex[obsInSample]+yIndexDiff*c(1:max(h,1));
        }
        else{
            yIndexDiff <- diff(tail(yIndex,2));
            yForecastIndex <- yIndex[obsInSample]+yIndexDiff*c(1:max(h,1));
        }
        yForecastStart <- yIndex[obsInSample]+yIndexDiff;
        yHoldout <- NULL;
        yIndexAll <- c(yIndex,yForecastIndex);
    }

    if(!is.numeric(yInSample)){
        stop("The provided data is not numeric! Can't construct any model!", call.=FALSE);
    }

    # Add trend variable to xreg if requested via formula but not present
    if(!is.null(formulaToUse) &&
       any(all.vars(formulaToUse)=="trend") && all(colnames(xregData)!="trend")){
        if(!is.null(xregData)){
            xregData <- cbind(xregData,trend=c(1:obsAll));
        }
        else{
            xregData <- cbind(y=y,trend=c(1:obsAll));
        }
    }

    parametersNumber <- matrix(0,2,5,
                               dimnames=list(c("Estimated","Provided"),
                                             c("nParamInternal","nParamXreg",
                                               "nParamOccurrence","nParamScale","nParamAll")));

    return(list(
        y = y,
        yHoldout = yHoldout,
        yInSample = yInSample,
        yNAValues = yNAValues,
        yIndex = yIndex,
        yClasses = yClasses,
        yFrequency = yFrequency,
        yStart = yStart,
        yForecastStart = yForecastStart,
        yInSampleIndex = yInSampleIndex,
        yForecastIndex = yForecastIndex,
        yIndexAll = yIndexAll,
        obsInSample = obsInSample,
        obsAll = obsAll,
        xregData = xregData,
        responseName = responseName,
        parametersNumber = parametersNumber,
        lags = lags,
        h = h,
        holdout = holdout
    ));
}

#### Optimiser / ellipsis parameter processing ####
#' @keywords internal
adam_checkOptimizer <- function(ellipsis, loss, distribution, initialType, lags, arimaModel) {
    if(is.null(ellipsis$maxeval)){
        maxeval <- NULL;
        if(any(lags>24) && arimaModel && any(initialType==c("optimal","two-stage"))){
            warning(paste0("The estimation of ARIMA model with initial='optimal' on high frequency ",
                           "data might take more time to converge to the optimum. Consider either ",
                           "setting maxeval parameter to a higher value (e.g. maxeval=10000, which ",
                           "will take ~25 times more than this) or using initial='backcasting'."),
                    call.=FALSE, immediate.=TRUE);
        }
    }
    else{
        maxeval <- ellipsis$maxeval;
    }
    maxtime <- if(is.null(ellipsis$maxtime)) -1 else ellipsis$maxtime;
    xtol_rel <- if(is.null(ellipsis$xtol_rel)) 1E-6 else ellipsis$xtol_rel;
    xtol_abs <- if(is.null(ellipsis$xtol_abs)) 1E-8 else ellipsis$xtol_abs;
    ftol_rel <- if(is.null(ellipsis$ftol_rel)) 1E-8 else ellipsis$ftol_rel;
    ftol_abs <- if(is.null(ellipsis$ftol_abs)) 0 else ellipsis$ftol_abs;
    algorithm <- if(is.null(ellipsis$algorithm)) "NLOPT_LN_NELDERMEAD" else ellipsis$algorithm;
    print_level <- if(is.null(ellipsis$print_level)) 0 else ellipsis$print_level;
    lb <- if(is.null(ellipsis$lb)) NULL else ellipsis$lb;
    ub <- if(is.null(ellipsis$ub)) NULL else ellipsis$ub;
    B  <- if(is.null(ellipsis$B))  NULL else ellipsis$B;

    lambda <- other <- NULL;
    otherParameterEstimate <- FALSE;

    if(any(loss==c("LASSO","RIDGE"))){
        if(is.null(ellipsis$lambda)){
            warning("You have not provided lambda parameter. I will set it to zero.", call.=FALSE);
            lambda <- 0;
        }
        else{
            lambda <- ellipsis$lambda;
        }
    }

    if(distribution=="dalaplace"){
        if(is.null(ellipsis$alpha)){
            other <- 0.5;
            otherParameterEstimate <- TRUE;
        }
        else{
            other <- ellipsis$alpha;
            otherParameterEstimate <- FALSE;
        }
        names(other) <- "alpha";
    }
    else if(any(distribution==c("dgnorm","dlgnorm"))){
        if(is.null(ellipsis$shape)){
            other <- 2;
            otherParameterEstimate <- TRUE;
        }
        else{
            other <- ellipsis$shape;
            otherParameterEstimate <- FALSE;
        }
        names(other) <- "shape";
    }
    else if(distribution=="dt"){
        if(is.null(ellipsis$nu)){
            other <- 2;
            otherParameterEstimate <- TRUE;
        }
        else{
            other <- ellipsis$nu;
            otherParameterEstimate <- FALSE;
        }
        names(other) <- "nu";
    }

    if(is.null(ellipsis$nIterations)){
        nIterations <- 1;
        if(any(initialType==c("complete","backcasting"))){
            nIterations[] <- 2;
        }
    }
    else{
        nIterations <- ellipsis$nIterations;
    }

    smoother <- if(is.null(ellipsis$smoother)) "global" else ellipsis$smoother;
    FI <- if(is.null(ellipsis$FI)) FALSE else ellipsis$FI;
    stepSize <- if(is.null(ellipsis$stepSize)) .Machine$double.eps^(1/4) else ellipsis$stepSize;
    dfForBack <- if(is.null(ellipsis$dfForBack)) FALSE else ellipsis$dfForBack;

    return(list(
        maxeval = maxeval,
        maxtime = maxtime,
        xtol_rel = xtol_rel,
        xtol_abs = xtol_abs,
        ftol_rel = ftol_rel,
        ftol_abs = ftol_abs,
        algorithm = algorithm,
        print_level = print_level,
        lb = lb,
        ub = ub,
        B = B,
        lambda = lambda,
        other = other,
        otherParameterEstimate = otherParameterEstimate,
        nIterations = nIterations,
        smoother = smoother,
        FI = FI,
        stepSize = stepSize,
        dfForBack = dfForBack
    ));
}

#### Model architecture and initial matrix creation ####
#' @keywords internal
adam_architector <- function(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                             xregNumber, obsInSample, initialType,
                             arimaModel, lagsModelARIMA, xregModel, constantRequired,
                             componentsNumberARIMA,
                             obsAll, yIndexAll, yClasses, adamETS,
                             profilesRecentTable=NULL, profilesRecentProvided=FALSE){
    if(etsModel){
        modelIsTrendy <- Ttype != "N";
        if(modelIsTrendy){
            lagsModel <- matrix(c(1, 1), ncol=1);
            componentsNamesETS <- c("level", "trend");
        }
        else{
            lagsModel <- matrix(c(1), ncol=1);
            componentsNamesETS <- c("level");
        }
        modelIsSeasonal <- Stype != "N";
        if(modelIsSeasonal){
            lagsModel <- matrix(c(lagsModel, lagsModelSeasonal), ncol=1);
            componentsNumberETSSeasonal <- length(lagsModelSeasonal);
            if(componentsNumberETSSeasonal > 1){
                componentsNamesETS <- c(componentsNamesETS,
                                        paste0("seasonal", c(1:componentsNumberETSSeasonal)));
            }
            else{
                componentsNamesETS <- c(componentsNamesETS, "seasonal");
            }
        }
        else{
            componentsNumberETSSeasonal <- 0;
        }
        lagsModelAll <- lagsModel;
        componentsNumberETS <- length(lagsModel);
    }
    else{
        modelIsTrendy <- modelIsSeasonal <- FALSE;
        componentsNumberETS <- componentsNumberETSSeasonal <- 0;
        componentsNamesETS <- NULL;
        lagsModelAll <- lagsModel <- NULL;
    }

    if(arimaModel){
        lagsModelAll <- matrix(c(lagsModel, lagsModelARIMA), ncol=1);
    }

    if(constantRequired){
        lagsModelAll <- matrix(c(lagsModelAll, 1), ncol=1);
    }

    if(xregModel){
        lagsModelAll <- matrix(c(lagsModelAll, rep(1, xregNumber)), ncol=1);
    }

    lagsModelMax <- max(lagsModelAll);
    obsStates <- obsInSample + lagsModelMax;

    adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                       lags=lags, yIndex=yIndexAll, yClasses=yClasses);
    if(profilesRecentProvided){
        profilesRecentTable <- profilesRecentTable[, 1:lagsModelMax, drop=FALSE];
    }
    else{
        profilesRecentTable <- adamProfiles$recent;
    }
    indexLookupTable <- adamProfiles$lookup;

    componentsNumberETSNonSeasonal <- componentsNumberETS - componentsNumberETSSeasonal;
    adamCpp <- new(adamCore,
                   lagsModelAll, Etype, Ttype, Stype,
                   componentsNumberETSNonSeasonal,
                   componentsNumberETSSeasonal,
                   componentsNumberETS, componentsNumberARIMA,
                   xregNumber, length(lagsModelAll),
                   constantRequired, adamETS);

    return(list(
        lagsModel = lagsModel,
        lagsModelAll = lagsModelAll,
        lagsModelMax = lagsModelMax,
        componentsNumberETS = componentsNumberETS,
        componentsNumberETSSeasonal = componentsNumberETSSeasonal,
        componentsNumberETSNonSeasonal = componentsNumberETSNonSeasonal,
        componentsNamesETS = componentsNamesETS,
        obsStates = obsStates,
        modelIsTrendy = modelIsTrendy,
        modelIsSeasonal = modelIsSeasonal,
        indexLookupTable = indexLookupTable,
        profilesRecentTable = profilesRecentTable,
        adamCpp = adamCpp
    ));
}

#' @keywords internal
adam_creator <- function(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                         lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                         profilesRecentTable=NULL, profilesRecentProvided=FALSE,
                         obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                         componentsNamesETS, otLogical, yInSample,
                         persistence, persistenceEstimate,
                         persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                         persistenceSeasonal, persistenceSeasonalEstimate,
                         persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                         phi,
                         initialType, initialEstimate,
                         initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                         initialSeasonal, initialSeasonalEstimate,
                         initialArima, initialArimaEstimate, initialArimaNumber,
                         initialXregEstimate, initialXregProvided,
                         arimaModel, arRequired, iRequired, maRequired, armaParameters,
                         arOrders, iOrders, maOrders,
                         componentsNumberARIMA, componentsNamesARIMA,
                         xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                         xregParametersPersistence,
                         constantRequired, constantEstimate, constantValue, constantName,
                         adamCpp,
                         arEstimate, maEstimate, smoother, nonZeroARI, nonZeroMA){

    matVt <- matrix(NA, componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired, obsStates,
                    dimnames=list(c(componentsNamesETS,componentsNamesARIMA,xregNames,constantName),NULL));

    matWt <- matrix(1, obsAll, componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired,
                    dimnames=list(NULL,c(componentsNamesETS,componentsNamesARIMA,xregNames,constantName)));

    if(xregModel){
        matWt[,componentsNumberETS+componentsNumberARIMA+1:xregNumber] <- xregData;
    }

    matF <- diag(componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired);

    vecG <- matrix(0, componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired, 1,
                   dimnames=list(c(componentsNamesETS,componentsNamesARIMA,xregNames,constantName),NULL));

    j <- 0;
    if(etsModel){
        j <- j+1;
        rownames(vecG)[j] <- "alpha";
        if(!persistenceLevelEstimate){
            vecG[j,] <- persistenceLevel;
        }
        if(modelIsTrendy){
            j <- j+1;
            rownames(vecG)[j] <- "beta";
            if(!persistenceTrendEstimate){
                vecG[j,] <- persistenceTrend;
            }
        }
        if(modelIsSeasonal){
            if(!all(persistenceSeasonalEstimate)){
                vecG[j+which(!persistenceSeasonalEstimate),] <- persistenceSeasonal;
            }
            if(componentsNumberETSSeasonal>1){
                rownames(vecG)[j+c(1:componentsNumberETSSeasonal)] <- paste0("gamma",c(1:componentsNumberETSSeasonal));
            }
            else{
                rownames(vecG)[j+1] <- "gamma";
            }
            j <- j+componentsNumberETSSeasonal;
        }
    }

    if(arimaModel){
        matF[j+1:componentsNumberARIMA,j+1:componentsNumberARIMA] <- 0;
        if(componentsNumberARIMA>1){
            rownames(vecG)[j+1:componentsNumberARIMA] <- paste0("psi",c(1:componentsNumberARIMA));
        }
        else{
            rownames(vecG)[j+1:componentsNumberARIMA] <- "psi";
        }
        j <- j+componentsNumberARIMA;
    }

    if(!arimaModel && constantRequired){
        matF[1,ncol(matF)] <- 1;
    }

    if(xregModel){
        if(persistenceXregProvided && !persistenceXregEstimate){
            vecG[j+1:xregNumber,] <- persistenceXreg;
        }
        rownames(vecG)[j+1:xregNumber] <- paste0("delta",xregParametersPersistence);
    }

    if(etsModel && modelIsTrendy){
        matF[1,2] <- phi;
        matF[2,2] <- phi;
        matWt[,2] <- phi;
    }

    if(arimaModel && (!arEstimate && !maEstimate)){
        arimaPolynomials <- lapply(adamCpp$polynomialise(0, arOrders, iOrders, maOrders,
                                                         arEstimate, maEstimate, armaParameters, lags), as.vector);
        if(nrow(nonZeroARI)>0){
            matF[componentsNumberETS+nonZeroARI[,2],componentsNumberETS+nonZeroARI[,2]] <-
                -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
        }
        if(nrow(nonZeroARI)>0){
            vecG[componentsNumberETS+nonZeroARI[,2]] <- -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
        }
        if(nrow(nonZeroMA)>0){
            vecG[componentsNumberETS+nonZeroMA[,2]] <- vecG[componentsNumberETS+nonZeroMA[,2]] +
                arimaPolynomials$maPolynomial[nonZeroMA[,1]];
        }
    }
    else{
        arimaPolynomials <- NULL;
    }

    if(!profilesRecentProvided){
        if(etsModel){
            if(initialEstimate){
                if(modelIsSeasonal){
                    yDecompositionAdditive <- msdecompose(yInSample, lags=lags[lags!=1],
                                                          type="additive", smoother=smoother);
                    if(any(c(Etype,Ttype,Stype)=="M")){
                        yDecompositionMultiplicative <- msdecompose(yInSample, lags=lags[lags!=1],
                                                                    type="multiplicative", smoother=smoother);
                    }
                    decompositionType <- c("additive","multiplicative")[any(c(Etype,Stype)=="M")+1];
                    yDecomposition <- switch(decompositionType,
                                             "additive"=yDecompositionAdditive,
                                             "multiplicative"=yDecompositionMultiplicative);
                    j <- 1;
                    if(initialLevelEstimate){
                        if(modelIsTrendy){
                            matVt[j,1:lagsModelMax] <- switch(Ttype,
                                                              "M"=yDecompositionMultiplicative$initial$nonseasonal[1],
                                                              yDecompositionAdditive$initial$nonseasonal[1]);
                        }
                        else{
                            matVt[j,1:lagsModelMax] <- mean(yInSample[otLogical]);
                        }
                        if(xregModel){
                            if(Etype=="A"){
                                matVt[j,1:lagsModelMax] <- matVt[j,1:lagsModelMax] -
                                    as.vector(xregModelInitials[[1]]$initialXreg %*% xregData[1,]);
                            }
                            else{
                                matVt[j,1:lagsModelMax] <- matVt[j,1:lagsModelMax] /
                                    as.vector(exp(xregModelInitials[[2]]$initialXreg %*% xregData[1,]));
                            }
                        }
                    }
                    else{
                        matVt[j,1:lagsModelMax] <- initialLevel;
                    }
                    j <- j+1;
                    if(modelIsTrendy){
                        if(initialTrendEstimate){
                            if(Ttype=="A"){
                                matVt[j,1:lagsModelMax] <- yDecompositionAdditive$initial$nonseasonal[2];
                                if(Stype=="M"){
                                    if(matVt[j,1]<0 && abs(matVt[j,1])>min(abs(yInSample[otLogical]))){
                                        matVt[j,1:lagsModelMax] <- 0;
                                    }
                                }
                            }
                            else if(Ttype=="M"){
                                matVt[j,1:lagsModelMax] <- yDecompositionMultiplicative$initial$nonseasonal[2];
                                if(any(matVt[1,1:lagsModelMax]<0)){
                                    matVt[1,1:lagsModelMax] <- yInSample[otLogical][1];
                                }
                            }
                        }
                        else{
                            matVt[j,1:lagsModelMax] <- initialTrend;
                        }
                        j <- j+1;
                    }
                    if(all(c(Etype,Stype)=="A") || all(c(Etype,Stype)=="M") ||
                       (Etype=="A" & Stype=="M")){
                        for(i in 1:componentsNumberETSSeasonal){
                            if(initialSeasonalEstimate[i]){
                                matVt[i+j-1,1:lagsModel[i+j-1]] <- yDecomposition$initial$seasonal[[i]];
                                if(Stype=="A"){
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                        matVt[i+j-1,1:lagsModel[i+j-1]] -
                                        mean(matVt[i+j-1,1:lagsModel[i+j-1]]);
                                }
                                else{
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                        matVt[i+j-1,1:lagsModel[i+j-1]] /
                                        exp(mean(log(matVt[i+j-1,1:lagsModel[i+j-1]])));
                                }
                            }
                            else{
                                matVt[i+j-1,1:lagsModel[i+j-1]] <- initialSeasonal[[i]];
                            }
                        }
                    }
                    else if(Etype=="M" && Stype=="A"){
                        for(i in 1:componentsNumberETSSeasonal){
                            if(initialSeasonalEstimate[i]){
                                matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                    log(yDecomposition$initial$seasonal[[i]])*min(yInSample[otLogical]);
                                if(Stype=="A"){
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <- matVt[i+j-1,1:lagsModel[i+j-1]] -
                                        mean(matVt[i+j-1,1:lagsModel[i+j-1]]);
                                }
                                else{
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <- matVt[i+j-1,1:lagsModel[i+j-1]] /
                                        exp(mean(log(matVt[i+j-1,1:lagsModel[i+j-1]])));
                                }
                            }
                            else{
                                matVt[i+j-1,1:lagsModel[i+j-1]] <- initialSeasonal[[i]];
                            }
                        }
                    }
                    if(Etype=="M" && matVt[1,1]<=0){
                        matVt[1,1:lagsModelMax] <- yInSample[1];
                    }
                }
                else{
                    yDecompositionAdditive <- msdecompose(yInSample, lags=1,
                                                          type="additive", smoother=smoother);
                    if(any(c(Etype,Ttype)=="M")){
                        yDecompositionMultiplicative <- msdecompose(yInSample, lags=1,
                                                                    type="multiplicative", smoother=smoother);
                    }
                    if(initialLevelEstimate){
                        if(modelIsTrendy){
                            matVt[1,1:lagsModelMax] <- switch(Ttype,
                                                              "M"=yDecompositionMultiplicative$initial$nonseasonal[1],
                                                              yDecompositionAdditive$initial$nonseasonal[1]);
                        }
                        else{
                            matVt[1,1:lagsModelMax] <- mean(yInSample[otLogical]);
                        }
                    }
                    else{
                        matVt[1,1:lagsModelMax] <- initialLevel;
                    }
                    if(modelIsTrendy){
                        if(initialTrendEstimate){
                            matVt[2,1:lagsModelMax] <- switch(Ttype,
                                                              "A"=yDecompositionAdditive$initial$nonseasonal[2],
                                                              "M"=yDecompositionMultiplicative$initial$nonseasonal[2]);
                        }
                        else{
                            matVt[2,1:lagsModelMax] <- initialTrend;
                        }
                    }
                    if(Etype=="M" && matVt[1,1]<=0){
                        matVt[1,1:lagsModelMax] <- yInSample[1];
                    }
                }
                if(initialLevelEstimate && Etype=="M" && matVt[1,lagsModelMax]==0){
                    matVt[1,1:lagsModelMax] <- mean(yInSample);
                }
            }
            else if(!initialEstimate && initialType=="provided"){
                j <- 1;
                matVt[j,1:lagsModelMax] <- initialLevel;
                if(modelIsTrendy){
                    j <- j+1;
                    matVt[j,1:lagsModelMax] <- initialTrend;
                }
                if(modelIsSeasonal){
                    for(i in 1:componentsNumberETSSeasonal){
                        matVt[j+i,1:lagsModel[j+i]] <- initialSeasonal[[i]];
                    }
                }
                j <- j+componentsNumberETSSeasonal;
            }
        }

        if(arimaModel){
            if(initialArimaEstimate){
                matVt[componentsNumberETS+1:componentsNumberARIMA, 1:initialArimaNumber] <-
                    switch(Etype, "A"=0, "M"=1);
                if(any(lags>1) && obsInSample > max(lags)*2){
                    yDecomposition <- tail(msdecompose(yInSample,
                                                       lags=lags[lags!=1],
                                                       type=switch(Etype,
                                                                   "A"="additive",
                                                                   "M"="multiplicative"),
                                                       smoother=smoother)$seasonal, 1)[[1]];
                }
                else if(any(lags>1) && obsInSample <= max(lags)*2){
                    yDecomposition <- yInSample[otLogical][1:obsInSample];
                }
                else{
                    yDecomposition <- switch(Etype,
                                             "A"=mean(diff(yInSample[otLogical])),
                                             "M"=exp(mean(diff(log(yInSample[otLogical])))));
                }
                matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber] <-
                    rep(yDecomposition, ceiling(initialArimaNumber/max(lags)))[1:initialArimaNumber];
            }
            else{
                matVt[componentsNumberETS+1:componentsNumberARIMA, 1:initialArimaNumber] <-
                    switch(Etype, "A"=0, "M"=1);
                matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber] <-
                    initialArima[1:initialArimaNumber];
            }
        }

        if(xregModel){
            if(Etype=="A" || initialXregProvided || is.null(xregModelInitials[[2]])){
                matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber, 1:lagsModelMax] <- 0;
                matVt[names(xregModelInitials[[1]]$initialXreg), 1:lagsModelMax] <-
                    xregModelInitials[[1]]$initialXreg;
            }
            else{
                matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber, 1:lagsModelMax] <- 0;
                matVt[names(xregModelInitials[[2]]$initialXreg), 1:lagsModelMax] <-
                    xregModelInitials[[2]]$initialXreg;
            }
        }

        if(constantRequired){
            if(constantEstimate){
                if(sum(iOrders)==0 && !etsModel){
                    matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <-
                        mean(yInSample[otLogical]);
                }
                else{
                    matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <-
                        switch(Etype,
                               "A"=mean(diff(yInSample[otLogical])),
                               "M"=exp(mean(diff(log(yInSample[otLogical])))));
                }
            }
            else{
                matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <- constantValue;
            }
            if(etsModel && initialLevelEstimate){
                if(Etype=="A"){
                    matVt[1,1:lagsModelMax] <- matVt[1,1:lagsModelMax] -
                        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1];
                }
                else{
                    matVt[1,1:lagsModelMax] <- matVt[1,1:lagsModelMax] /
                        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1];
                }
            }
            if(arimaModel && initialArimaEstimate){
                if(Etype=="A"){
                    matVt[componentsNumberETS+nonZeroARI[,2],1:initialArimaNumber] <-
                        matVt[componentsNumberETS+nonZeroARI[,2],1:initialArimaNumber] -
                        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1];
                }
                else{
                    matVt[componentsNumberETS+nonZeroARI[,2],1:initialArimaNumber] <-
                        matVt[componentsNumberETS+nonZeroARI[,2],1:initialArimaNumber] /
                        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1];
                }
            }
        }
    }
    else{
        matVt[,1:lagsModelMax] <- profilesRecentTable;
    }

    return(list(matVt=matVt, matWt=matWt, matF=matF, vecG=vecG, arimaPolynomials=arimaPolynomials));
}
